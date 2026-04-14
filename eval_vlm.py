"""
MiniMind-V 评估脚本

使用 AI4Math/MathVista 数据集进行评估测试
"""
import argparse
import os
import warnings
import torch
import re
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_vlm import MiniMindVLM, VLMConfig
from trainer.trainer_utils import setup_seed
warnings.filterwarnings('ignore')


def init_model(args):
    """初始化模型"""
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    tokenizer_path = os.path.join(script_dir, args.load_from)
    vision_model_path = os.path.join(script_dir, "model/vision_model/clip-vit-base-patch16")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = os.path.join(script_dir, args.save_dir, f'{args.weight}_{args.hidden_size}{moe_suffix}.pth')
        model = MiniMindVLM(
            VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)),
            vision_model_path=vision_model_path
        )
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(tokenizer_path, trust_remote_code=True)
        model.vision_encoder, model.processor = MiniMindVLM.get_vision_model(vision_model_path)
    
    print(f'VLM模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M(illion)')
    preprocess = model.processor
    return model.eval().to(args.device), tokenizer, preprocess


def load_mathvista_dataset(split='test', max_samples=100):
    """从 HuggingFace 加载 MathVista 数据集"""
    from datasets import load_dataset
    
    # 固定缓存目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "dataset/Mathvista")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"从 HuggingFace 加载 AI4Math/MathVista 数据集 (split={split})...")
    print(f"缓存目录: {cache_dir}")
    
    dataset = load_dataset("AI4Math/MathVista", split=split, cache_dir=cache_dir)
    samples = list(dataset)
    
    # 限制样本数量
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
    
    print(f"加载了 {len(samples)} 条测试数据")
    return samples


def extract_answer(text):
    """从模型输出中提取答案"""
    # 尝试提取 <answer>...</answer> 标签内的内容
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 如果没有标签，返回整个文本（去除推理部分）
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="MiniMind-V Evaluation with MathVista")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='sft_vlm', type=str, help="权重名称前缀")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--max_new_tokens', default=256, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.1, type=float, help="生成温度")
    parser.add_argument('--top_p', default=0.9, type=float, help="nucleus采样阈值")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--split', default='test', type=str, help="数据集分割 (testmini, test)")
    parser.add_argument('--max_samples', default=100, type=int, help="最大测试样本数")
    parser.add_argument('--verbose', action='store_true', help="显示详细输出")
    args = parser.parse_args()
    
    # 初始化模型
    model, tokenizer, preprocess = init_model(args)
    
    # 加载 MathVista 测试数据
    test_samples = load_mathvista_dataset(split=args.split, max_samples=args.max_samples)
    
    print("\n" + "=" * 60)
    print("开始评估 (AI4Math/MathVista)")
    print("=" * 60 + "\n")
    
    total = 0
    results = []
    
    for idx, sample in enumerate(test_samples):
        setup_seed(2026)
        
        # 获取问题
        question = sample.get('question', sample.get('query', ''))
        choices = sample.get('choices', None)
        image = sample.get('image', sample.get('decoded_image', None))
        
        # 处理图像
        if image is not None:
            if hasattr(image, 'convert'):
                img = image.convert('RGB') if image.mode != 'RGB' else image
            else:
                img = Image.new('RGB', (224, 224), color='white')
        else:
            img = Image.new('RGB', (224, 224), color='white')
        
        pixel_values = MiniMindVLM.image2tensor(img, preprocess).to(args.device).unsqueeze(0)
        
        # 构建 prompt（使用与 SFT 训练一致的格式）
        prompt = f"{model.params.image_special_token}\n{question}"
        
        # 添加选项（如果有）
        if choices and isinstance(choices, list) and len(choices) > 0:
            prompt += "\n\nChoices:\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
        
        # 添加推理引导 prompt（与 SFT 训练一致）
        prompt += "\n\nPlease first provide your reasoning on how to solve this question between <reasoning> and </reasoning>, then give your final answer between <answer> and </answer>."
        
        messages = [{"role": "user", "content": prompt}]
        inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                temperature=args.temperature,
                pixel_values=pixel_values
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_answer = extract_answer(response)
        total += 1
        
        results.append({
            'question': question,
            'pred_answer': pred_answer
        })
        
        # 打印进度
        if args.verbose or (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{len(test_samples)}]")
            print(f"  问题: {question[:50]}...")
            print(f"  预测答案: {pred_answer[:50]}...")
            print()
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总样本数: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
