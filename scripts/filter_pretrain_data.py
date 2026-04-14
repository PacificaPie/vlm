"""
过滤 pretrain_data.jsonl 和 sft_data.jsonl，只保留图像文件实际存在的数据

运行此脚本会自动处理两个数据文件：
1. dataset/pretrain_data.jsonl - 使用 dataset/pretrain_images 中的图像
2. dataset/sft_data.jsonl - 使用 dataset/sft_images 中的图像

使用方法：
    python scripts/filter_pretrain_data.py
"""
import os
import json
from tqdm import tqdm


def filter_pretrain_data(jsonl_path, images_path, output_path=None):
    """
    过滤 JSONL 文件，只保留图像文件存在的数据
    
    Args:
        jsonl_path: 输入的 JSONL 文件路径
        images_path: 图像文件目录路径
        output_path: 输出文件路径，如果为 None 则覆盖原文件
    """
    # 获取所有存在的图像文件名（使用集合以提高查找速度）
    print(f"正在扫描图像目录: {images_path}")
    existing_images = set()
    if os.path.exists(images_path):
        for filename in os.listdir(images_path):
            if os.path.isfile(os.path.join(images_path, filename)):
                existing_images.add(filename)
    print(f"找到 {len(existing_images)} 个图像文件")
    
    # 读取并过滤数据
    print(f"正在读取和过滤数据文件: {jsonl_path}")
    valid_data = []
    invalid_count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="处理数据"), 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                image_field = data.get('image', '')
                
                # 处理多个图像（用逗号分隔）
                image_names = [img.strip() for img in image_field.split(',')]
                
                # 检查所有图像文件是否存在
                all_exist = True
                for image_name in image_names:
                    if image_name and image_name not in existing_images:
                        all_exist = False
                        break
                
                if all_exist and image_names:  # 确保至少有一个图像
                    valid_data.append(line)
                else:
                    invalid_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"警告：第 {line_num} 行 JSON 解析错误: {e}")
                invalid_count += 1
                continue
    
    # 写入结果
    if output_path is None:
        output_path = jsonl_path + '.filtered'
        print(f"输出文件未指定，将保存到: {output_path}")
    
    print(f"\n统计信息:")
    print(f"  总数据量: {len(valid_data) + invalid_count}")
    print(f"  有效数据: {len(valid_data)}")
    print(f"  无效数据: {invalid_count}")
    print(f"  保留比例: {len(valid_data) / (len(valid_data) + invalid_count) * 100:.2f}%")
    
    print(f"\n正在写入过滤后的数据到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in tqdm(valid_data, desc="写入数据"):
            f.write(line + '\n')
    
    print(f"完成！过滤后的数据已保存到: {output_path}")
    
    # 如果输出路径不是原文件路径，询问是否覆盖原文件
    if output_path != jsonl_path:
        try:
            response = input(f"\n是否覆盖原文件 {jsonl_path}? (y/n，默认n): ")
            if response.lower() == 'y':
                import shutil
                shutil.move(output_path, jsonl_path)
                print(f"原文件已覆盖")
            else:
                print(f"原文件保持不变，新文件保存在: {output_path}")
        except (EOFError, KeyboardInterrupt):
            # 非交互模式（如通过管道或脚本调用），不覆盖原文件
            print(f"非交互模式：原文件保持不变，新文件保存在: {output_path}")


if __name__ == "__main__":
    import sys
    import os
    
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 定义要处理的数据文件配置
    data_configs = [
        {
            "name": "pretrain",
            "jsonl_path": os.path.join(project_root, "dataset", "pretrain_data.jsonl"),
            "images_path": os.path.join(project_root, "dataset", "pretrain_images"),
        },
        {
            "name": "sft",
            "jsonl_path": os.path.join(project_root, "dataset", "sft_data.jsonl"),
            "images_path": os.path.join(project_root, "dataset", "sft_images"),
        }
    ]
    
    print("=" * 80)
    print("开始过滤数据文件，确保图像文件存在")
    print("=" * 80)
    
    # 依次处理每个数据文件
    for i, config in enumerate(data_configs, 1):
        print(f"\n[{i}/{len(data_configs)}] 处理 {config['name']} 数据...")
        print("-" * 80)
        
        if not os.path.exists(config['jsonl_path']):
            print(f"警告：文件不存在，跳过: {config['jsonl_path']}")
            continue
        
        if not os.path.exists(config['images_path']):
            print(f"警告：图像目录不存在，跳过: {config['images_path']}")
            continue
        
        # 直接覆盖原文件
        filter_pretrain_data(
            config['jsonl_path'],
            config['images_path'],
            output_path=config['jsonl_path']  # 直接覆盖原文件
        )
    
    print("\n" + "=" * 80)
    print("所有数据文件过滤完成！")
    print("=" * 80)

