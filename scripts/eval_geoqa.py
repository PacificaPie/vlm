"""
GeoQA 评测脚本

支持三种评测目标：
  1. baseline  —— InternVL3-8B 零样本（不加载任何 adapter）
  2. sft       —— 加载 SFT LoRA adapter
  3. grpo      —— 加载 GRPO LoRA adapter

答案提取规则（按优先级）：
  1. <answer>X</answer> 标签
  2. 末尾孤立的 A/B/C/D 字母
  3. 文中第一个大写字母 A/B/C/D

使用方法:
  # baseline
  python scripts/eval_geoqa.py --model_path ./InternVL3-8B --mode baseline

  # SFT 后
  python scripts/eval_geoqa.py --model_path ./InternVL3-8B \\
      --adapter_path out/geoqa/sft_geoqa_final --mode sft

  # GRPO 后
  python scripts/eval_geoqa.py --model_path ./InternVL3-8B \\
      --adapter_path out/geoqa/grpo_geoqa_final --mode grpo
"""

import os
import re
import sys
import json
import argparse
import warnings
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

import torch
import pyarrow.parquet as pq
from PIL import Image
import io
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from peft import PeftModel


# ==================== 答案提取 ====================

def extract_answer(text: str) -> str | None:
    """从模型输出中提取 A/B/C/D 答案字母"""
    # 1. 优先识别 <answer>X</answer> 标签
    m = re.search(r'<answer>\s*([A-D])\s*</answer>', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 2. 末尾孤立字母（"故选B" / "答案是C" / "选D"）
    m = re.search(r'(?:故选|答案[是为]?|选|answer[:\s]*)\s*([A-D])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3. 文末出现的独立字母
    m = re.search(r'\b([A-D])\b\s*$', text.strip())
    if m:
        return m.group(1).upper()

    return None


# ==================== 主评测逻辑 ====================

def evaluate(model, tokenizer, image_processor, image_token_str,
             test_parquet: str, args) -> dict:
    """在测试集上逐条推理，返回总体及分题型准确率"""

    table = pq.read_table(test_parquet)
    n = len(table)
    if args.max_samples:
        n = min(n, args.max_samples)

    correct = 0
    parse_fail = 0
    results = []

    for idx in tqdm(range(n), desc='Evaluating'):
        conversations = json.loads(table['conversations'][idx].as_py())
        image_bytes   = table['image_bytes'][idx].as_py()

        # 取 ground truth（assistant turn 的 <answer>X</answer>）
        assistant_content = conversations[1]['content']
        gt = extract_answer(assistant_content)
        if gt is None:
            parse_fail += 1
            continue

        # 构建 inference prompt（只含 user turn）
        user_conv = [conversations[0]]
        prompt = tokenizer.apply_chat_template(
            user_conv, tokenize=False, add_generation_prompt=True
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

        # 推理
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,        # greedy，保证可复现
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        pred = extract_answer(response)

        is_correct = (pred == gt) if pred else False
        correct += int(is_correct)
        results.append({'gt': gt, 'pred': pred, 'correct': is_correct})

        if args.verbose:
            print(f'[{idx+1}/{n}] GT={gt} Pred={pred} {"✓" if is_correct else "✗"}')
            print(f'  Response: {response[:100]}')

    accuracy = correct / max(n - parse_fail, 1)
    return {
        'accuracy':    accuracy,
        'correct':     correct,
        'total':       n - parse_fail,
        'parse_fail':  parse_fail,
        'results':     results,
    }


def main():
    parser = argparse.ArgumentParser(description='GeoQA Evaluation')
    parser.add_argument('--model_path',   type=str, default='./InternVL3-8B')
    parser.add_argument('--adapter_path', type=str, default=None,
                        help='LoRA adapter 目录（baseline 时不填）')
    parser.add_argument('--mode',         type=str, default='baseline',
                        choices=['baseline', 'sft', 'grpo'])
    parser.add_argument('--test_data',    type=str, default='dataset/geoqa/sft_test.parquet')
    parser.add_argument('--max_samples',  type=int, default=None,
                        help='限制评测样本数（调试用）')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--dtype',        type=str, default='bfloat16')
    parser.add_argument('--device',       type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--verbose',      action='store_true')
    parser.add_argument('--output_file',  type=str, default=None,
                        help='保存结果到 JSON 文件')
    args = parser.parse_args()

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if not os.path.isabs(args.test_data):
        args.test_data = os.path.join(project_root, args.test_data)

    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    # ---------- 加载模型 ----------
    print(f'[{args.mode}] 加载 {args.model_path} ...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True
    )

    if args.adapter_path:
        print(f'加载 LoRA adapter: {args.adapter_path}')
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()   # 合并权重，推理更快

    model = model.to(args.device).eval()

    image_processor = CLIPImageProcessor.from_pretrained(
        args.model_path, size={'height': 448, 'width': 448}
    )
    image_token_str = (
        f'<img>{"<IMG_CONTEXT>" * model.num_image_token}</img>'
    )

    # ---------- 评测 ----------
    print(f'评测数据: {args.test_data}')
    metrics = evaluate(model, tokenizer, image_processor,
                       image_token_str, args.test_data, args)

    # ---------- 打印结果 ----------
    print('\n' + '=' * 50)
    print(f'模式:     {args.mode}')
    print(f'准确率:   {metrics["accuracy"]:.1%}  '
          f'({metrics["correct"]}/{metrics["total"]})')
    print(f'解析失败: {metrics["parse_fail"]} 条')
    print('=' * 50)

    if args.output_file:
        out = {
            'mode':     args.mode,
            'accuracy': metrics['accuracy'],
            'correct':  metrics['correct'],
            'total':    metrics['total'],
        }
        with open(args.output_file, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'结果已保存 → {args.output_file}')


if __name__ == '__main__':
    main()
