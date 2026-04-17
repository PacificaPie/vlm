"""
Pass@K 筛选脚本

对训练集每道题用 SFT 模型采样 K 次，
保留 0 < pass_rate < 1 的题（模型有时对有时错），
输出 rl_filtered.parquet 供 GRPO 训练使用。

Usage:
    python scripts/build_rl_filtered.py \
        --model_path ~/scratch/grpo_data/InternVL3-8B \
        --sft_path   ~/scratch/grpo_data/sft_geoqa_final \
        --train_data ~/scratch/grpo_data/geoqa/sft_train.parquet \
        --out        ~/scratch/grpo_data/geoqa/rl_filtered.parquet \
        --k          8
"""

import os
import re
import sys
import json
import glob
import argparse
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import io
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors

from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from peft import LoraConfig, get_peft_model, TaskType


def extract_answer(text: str):
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


def load_sft_model(model_path, sft_path, dtype, device):
    """SFT 权重加载（与 eval_geoqa.py / train_grpo_geoqa.py 保持一致）"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True,
                                      use_flash_attn=False)
    _lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
    )
    model.language_model = get_peft_model(model.language_model, _lora_cfg,
                                          autocast_adapter_dtype=False)
    shard_files = sorted(glob.glob(os.path.join(sft_path, '*.safetensors')))
    if not shard_files:
        raise FileNotFoundError(f'No .safetensors in {sft_path}')
    full_state = {}
    for sf in shard_files:
        full_state.update(load_safetensors(sf))
    model.load_state_dict(full_state, strict=False)
    model.language_model = model.language_model.merge_and_unload()
    model = model.to(device).eval()
    model.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--sft_path',   required=True,
                        help='sft_geoqa_final 目录')
    parser.add_argument('--train_data', required=True,
                        help='sft_train.parquet 路径')
    parser.add_argument('--out',        required=True,
                        help='输出 rl_filtered.parquet 路径')
    parser.add_argument('--k',          type=int, default=8,
                        help='每题采样次数')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--temperature',    type=float, default=0.7)
    parser.add_argument('--dtype',      type=str, default='bfloat16')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='调试用：只处理前 N 条')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype  = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    print(f'加载 SFT 模型: {args.sft_path}')
    model, tokenizer = load_sft_model(args.model_path, args.sft_path, dtype, device)

    image_processor = CLIPImageProcessor.from_pretrained(
        args.model_path, size={'height': 448, 'width': 448}
    )
    image_token_str = f'<img>{"<IMG_CONTEXT>" * model.num_image_token}</img>'

    table = pq.read_table(args.train_data)
    n = len(table) if args.max_samples is None else min(len(table), args.max_samples)
    print(f'训练集共 {len(table)} 条，处理前 {n} 条，K={args.k}')

    keep_indices = []
    pass_rate_log = []

    for idx in tqdm(range(n), desc='Pass@K filtering'):
        conversations = json.loads(table['conversations'][idx].as_py())
        image_bytes   = table['image_bytes'][idx].as_py()

        # ground truth
        assistant_content = conversations[1]['content'] if len(conversations) > 1 else ''
        m = re.search(r'<answer>\s*([A-D])\s*</answer>', assistant_content, re.IGNORECASE)
        gt = m.group(1).upper() if m else None
        if gt is None:
            continue

        # 构建 prompt（user turn only）
        prompt = tokenizer.apply_chat_template(
            [conversations[0]], tokenize=False, add_generation_prompt=True
        )
        prompt = prompt.replace('<image>', image_token_str)
        input_ids = tokenizer(
            prompt, return_tensors='pt', add_special_tokens=False
        ).input_ids  # [1, seq_len]

        # 图像预处理
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        pixel_values = image_processor(
            images=image, return_tensors='pt'
        )['pixel_values'].to(device=device, dtype=dtype)  # [1, C, H, W]

        # 复制 K 份，批量生成
        input_ids_k    = input_ids.expand(args.k, -1).to(device)    # [K, seq_len]
        pixel_values_k = pixel_values.expand(args.k, -1, -1, -1)    # [K, C, H, W]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids_k,
                pixel_values=pixel_values_k,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        correct = 0
        for i in range(args.k):
            resp = tokenizer.decode(output_ids[i], skip_special_tokens=True)
            if extract_answer(resp) == gt:
                correct += 1

        pass_rate = correct / args.k
        pass_rate_log.append(pass_rate)

        if 0 < pass_rate < 1:
            keep_indices.append(idx)

    # 统计
    total     = len(pass_rate_log)
    always_right = sum(1 for r in pass_rate_log if r == 1.0)
    always_wrong = sum(1 for r in pass_rate_log if r == 0.0)
    kept         = len(keep_indices)

    print(f'\n=== Pass@K 统计 ===')
    print(f'总题数:      {total}')
    print(f'全对 (1.0):  {always_right}  ({always_right/total:.1%})')
    print(f'全错 (0.0):  {always_wrong}  ({always_wrong/total:.1%})')
    print(f'保留 (0<p<1): {kept}         ({kept/total:.1%})')

    # 保存筛选后的 parquet
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    filtered = table.take(keep_indices)
    pq.write_table(filtered, args.out)
    print(f'\n已保存 → {args.out}  ({kept} 条)')


if __name__ == '__main__':
    main()
