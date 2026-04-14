"""
构建 GeoQA SFT 训练数据

数据来源: LoadingBFX/GeoQA-cot（中文，train=3509，test=755）
格式转换: HuggingFace dataset → Parquet（与现有 VLMDataset 兼容）

Parquet 列:
  - conversations: JSON 字符串，[{"role": "user", ...}, {"role": "assistant", ...}]
  - image_bytes:   bytes，PNG 格式图像

使用方法:
    python scripts/build_sft_data.py --output_dir dataset/geoqa
    python scripts/build_sft_data.py --output_dir dataset/geoqa --also_save_rl_base
"""

import os
import re
import io
import json
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


# ==================== 格式转换函数 ====================

def parse_answer_letter(answer_field: str) -> str:
    """从 answer 字段中提取选项字母 A/B/C/D"""
    m = re.search(r'<answer>([A-D])</answer>', answer_field)
    return m.group(1) if m else None


def build_user_content(problem: str) -> str:
    """
    清洗 problem 字段，构造 user 消息内容。

    原始格式：
        "<image>如图,在△ABC中...\nchoices{'A': '30°', ...}. 请用 A、B、C、D 作答."

    目标格式：
        "<image>\n如图,在△ABC中...\nchoices{'A': '30°', ...}\n请用 A、B、C、D 作答."
    """
    # <image> 占位符保留（InternVL processor 需要它定位图像位置）
    # 只把 <image> 后面紧跟的文字用换行隔开，让格式更清晰
    content = problem.strip()
    if not content.startswith('<image>'):
        content = '<image>\n' + content
    else:
        content = '<image>\n' + content[len('<image>'):].lstrip()
    return content


def image_to_bytes(pil_image: Image.Image) -> bytes:
    """将 PIL Image 转为 PNG bytes"""
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    return buf.getvalue()


def sample_to_row(sample: dict) -> dict | None:
    """
    将单条 GeoQA-cot 样本转换为 Parquet 行。

    返回 None 表示该样本应跳过（格式异常）。

    返回格式:
        {
            'conversations': JSON str,  # [user_turn, assistant_turn]
            'image_bytes':   bytes,     # PNG 图像
        }
    """
    images = sample['images']
    if isinstance(images, list):
        if len(images) == 0:
            return None
        pil_image = images[0]
    else:
        pil_image = images

    answer_field = sample['answer']
    letter = parse_answer_letter(answer_field)
    if letter is None:
        return None  # 解析失败，跳过

    user_content = build_user_content(sample['problem'])
    # assistant 内容保留完整的 <think>...</think><answer>X</answer> 格式
    assistant_content = answer_field.strip()

    conversations = [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    return {
        'conversations': json.dumps(conversations, ensure_ascii=False),
        'image_bytes':   image_to_bytes(pil_image),
    }


# ==================== 构建并保存 Parquet ====================

def build_parquet(samples, output_path: str):
    """将样本列表写入 Parquet 文件"""
    rows = []
    skipped = 0
    for sample in tqdm(samples, desc=f'Building {os.path.basename(output_path)}'):
        row = sample_to_row(sample)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    if not rows:
        print(f'  [warn] No valid rows, skipping {output_path}')
        return 0

    table = pa.table({
        'conversations': pa.array([r['conversations'] for r in rows], type=pa.string()),
        'image_bytes':   pa.array([r['image_bytes']   for r in rows], type=pa.binary()),
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pq.write_table(table, output_path, compression='snappy')
    print(f'  Saved {len(rows)} rows → {output_path}  (skipped {skipped})')
    return len(rows)


def build_rl_base_parquet(samples, output_path: str):
    """
    为 RL Pass@K 筛选准备基础数据：只保存 prompt + ground_truth answer。

    Parquet 列:
        - conversations: JSON str，只含 user turn（模型需要自行生成 assistant 部分）
        - image_bytes:   bytes
        - answer:        str，正确答案字母 A/B/C/D（供奖励函数使用）
    """
    rows = []
    skipped = 0
    for sample in tqdm(samples, desc=f'Building RL base {os.path.basename(output_path)}'):
        images = sample['images']
        pil_image = images[0] if isinstance(images, list) else images

        letter = parse_answer_letter(sample['answer'])
        if letter is None:
            skipped += 1
            continue

        user_content = build_user_content(sample['problem'])
        conversations = [{"role": "user", "content": user_content}]

        rows.append({
            'conversations': json.dumps(conversations, ensure_ascii=False),
            'image_bytes':   image_to_bytes(pil_image),
            'answer':        letter,
        })

    if not rows:
        print(f'  [warn] No valid rows, skipping {output_path}')
        return 0

    table = pa.table({
        'conversations': pa.array([r['conversations'] for r in rows], type=pa.string()),
        'image_bytes':   pa.array([r['image_bytes']   for r in rows], type=pa.binary()),
        'answer':        pa.array([r['answer']         for r in rows], type=pa.string()),
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pq.write_table(table, output_path, compression='snappy')
    print(f'  Saved {len(rows)} rows → {output_path}  (skipped {skipped})')
    return len(rows)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='Build GeoQA SFT/RL-base Parquet datasets')
    parser.add_argument('--output_dir', type=str, default='dataset/geoqa',
                        help='输出目录（相对于项目根目录）')
    parser.add_argument('--also_save_rl_base', action='store_true',
                        help='同时保存 RL Pass@K 筛选用的基础数据（只含 prompt + answer）')
    parser.add_argument('--hf_cache_dir', type=str, default='dataset/GeoQA_cache',
                        help='HuggingFace 数据集缓存目录')
    args = parser.parse_args()

    # 路径统一为项目根目录下的相对路径
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir  = os.path.join(project_root, args.output_dir)
    cache_dir   = os.path.join(project_root, args.hf_cache_dir)

    # ---------- 1. 加载数据集 ----------
    print('加载 LoadingBFX/GeoQA-cot ...')
    from datasets import load_dataset
    ds = load_dataset('LoadingBFX/GeoQA-cot', cache_dir=cache_dir)
    train_data = ds['train']   # 3509 条
    test_data  = ds['test']    # 755 条
    print(f'  train={len(train_data)}, test={len(test_data)}')

    # ---------- 2. 构建 SFT 训练/验证 Parquet ----------
    print('\n构建 SFT 数据...')
    build_parquet(train_data, os.path.join(output_dir, 'sft_train.parquet'))
    build_parquet(test_data,  os.path.join(output_dir, 'sft_test.parquet'))

    # ---------- 3. （可选）构建 RL 筛选基础数据 ----------
    if args.also_save_rl_base:
        print('\n构建 RL base 数据（用于 Pass@K 筛选）...')
        build_rl_base_parquet(train_data, os.path.join(output_dir, 'rl_base.parquet'))

    print('\n完成。')
    print(f'输出目录: {output_dir}')
    print('下一步: 用 sft_train.parquet 训练 SFT 模型，')
    print('        用 rl_base.parquet 跑 Pass@K 筛选（如需）。')


if __name__ == '__main__':
    main()
