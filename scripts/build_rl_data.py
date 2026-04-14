"""
Pass@K 筛选脚本：从 SFT 模型生成 N 个响应，按通过率过滤 RL 训练集

筛选逻辑（proposal 第 3 节）：
  - 通过率 = 0/N  → 太难，丢弃（奖励信号全为 0，无法学习）
  - 通过率 = 1~(N-1)/N → 难度合适，保留
  - 通过率 = N/N  → 太简单，丢弃（没有提升空间）

目标：从 3509 条中筛出 ~1500-2000 条，保存为 rl_filtered.parquet

使用方法（在 cluster 上，有 SFT 权重后运行）：
    python scripts/build_rl_data.py \\
        --model_path    ./InternVL3-8B \\
        --adapter_path  out/geoqa/sft_geoqa_final \\
        --input_parquet dataset/geoqa/rl_base.parquet \\
        --output_parquet dataset/geoqa/rl_filtered.parquet \\
        --n_generations 6
"""

import os
import re
import sys
import io
import json
import argparse
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from peft import PeftModel


# ==================== 答案提取（与 eval_geoqa.py 保持一致）====================

def extract_answer(text: str) -> str | None:
    m = re.search(r'<answer>\s*([A-D])\s*</answer>', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'(?:故选|答案[是为]?|选|answer[:\s]*)\s*([A-D])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'\b([A-D])\b\s*$', text.strip())
    if m:
        return m.group(1).upper()
    return None


# ==================== 主逻辑 ====================

def main():
    parser = argparse.ArgumentParser(description='Pass@K RL data filtering')
    parser.add_argument('--model_path',      type=str, default='./InternVL3-8B')
    parser.add_argument('--adapter_path',    type=str, required=True,
                        help='SFT LoRA adapter 目录')
    parser.add_argument('--input_parquet',   type=str,
                        default='dataset/geoqa/rl_base.parquet')
    parser.add_argument('--output_parquet',  type=str,
                        default='dataset/geoqa/rl_filtered.parquet')
    parser.add_argument('--n_generations',   type=int, default=6,
                        help='每题生成响应数 N（Pass@N 的 N）')
    parser.add_argument('--max_new_tokens',  type=int, default=256)
    parser.add_argument('--temperature',     type=float, default=0.8,
                        help='采样温度（需要 > 0 才能生成多样响应）')
    parser.add_argument('--max_samples',     type=int, default=None,
                        help='限制处理条数（调试用）')
    parser.add_argument('--dtype',           type=str, default='bfloat16')
    parser.add_argument('--device',          type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    for attr in ('input_parquet', 'output_parquet'):
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.join(project_root, val))
    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    # ---------- 加载模型 ----------
    print('加载 SFT 模型（含 LoRA adapter）...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()
    model = model.to(args.device).eval()

    image_processor = CLIPImageProcessor.from_pretrained(
        args.model_path, size={'height': 448, 'width': 448}
    )
    image_token_str = f'<img>{"<IMG_CONTEXT>" * model.num_image_token}</img>'

    # ---------- 读取 rl_base.parquet ----------
    table = pq.read_table(args.input_parquet)
    n = len(table)
    if args.max_samples:
        n = min(n, args.max_samples)
    print(f'处理 {n} 条样本，每条生成 {args.n_generations} 个响应 ...')

    kept_rows = []
    stats = {'total': n, 'kept': 0, 'too_hard': 0, 'too_easy': 0}

    for idx in tqdm(range(n)):
        conversations = json.loads(table['conversations'][idx].as_py())
        image_bytes   = table['image_bytes'][idx].as_py()
        gt_answer     = table['answer'][idx].as_py()

        # 构建 prompt
        prompt = tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True
        )
        prompt = prompt.replace('<image>', image_token_str)
        input_ids = tokenizer(
            prompt, return_tensors='pt', add_special_tokens=False
        ).input_ids.to(args.device)

        # 图像预处理
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        pixel_values = image_processor(
            images=image, return_tensors='pt'
        )['pixel_values'].to(args.device)

        # 生成 N 个响应（采样）
        correct_count = 0
        with torch.no_grad():
            for _ in range(args.n_generations):
                output_ids = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                response = tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
                )
                pred = extract_answer(response)
                if pred == gt_answer:
                    correct_count += 1

        # 按通过率过滤
        if correct_count == 0:
            stats['too_hard'] += 1
        elif correct_count == args.n_generations:
            stats['too_easy'] += 1
        else:
            # 难度合适：保留
            kept_rows.append({
                'conversations': table['conversations'][idx].as_py(),
                'image_bytes':   image_bytes,
                'answer':        gt_answer,
                'pass_rate':     f'{correct_count}/{args.n_generations}',
            })
            stats['kept'] += 1

    # ---------- 保存结果 ----------
    if not kept_rows:
        print('警告：没有保留任何样本，检查模型和数据是否正确')
        return

    out_table = pa.table({
        'conversations': pa.array([r['conversations'] for r in kept_rows], type=pa.string()),
        'image_bytes':   pa.array([r['image_bytes']   for r in kept_rows], type=pa.binary()),
        'answer':        pa.array([r['answer']         for r in kept_rows], type=pa.string()),
        'pass_rate':     pa.array([r['pass_rate']      for r in kept_rows], type=pa.string()),
    })
    pq.write_table(out_table, args.output_parquet, compression='snappy')

    print('\n' + '=' * 50)
    print(f'总样本:    {stats["total"]}')
    print(f'太难(0/{args.n_generations}):  {stats["too_hard"]}')
    print(f'太简单({args.n_generations}/{args.n_generations}): {stats["too_easy"]}')
    print(f'保留:      {stats["kept"]}  → {args.output_parquet}')
    print('=' * 50)


if __name__ == '__main__':
    main()
