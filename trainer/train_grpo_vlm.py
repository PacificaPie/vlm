
import os
import sys
import re
import time

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import warnings
import json
import io
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from PIL import Image
import pyarrow.parquet as pq
from model.model_vlm import MiniMindVLM, VLMConfig
from trainer.trainer_utils import (
    Logger,
    setup_seed,
    init_vlm_model,
    get_lr,
    GRPOCollator,
    GRPOTrainerBase,
)

warnings.filterwarnings('ignore')


# ==================== 奖励函数（图片描述任务）====================
def vocabulary_diversity_reward(response: str) -> float:
    """
    词汇丰富度奖励：综合评估词汇的多样性和丰富度
    
    评估维度:
    1. 词汇唯一率 (Type-Token Ratio)
    2. 实词比例（过滤停用词后的有效词汇）
    3. 词汇覆盖广度（是否使用了描述性词汇）
    
    输出范围 0-1
    """
    # 中文停用词（高频虚词）
    stopwords = {'的', '了', '是', '在', '有', '和', '与', '或', '等', '着', '被', '把', '给', '让', '向', '往', '从'}
    
    # 分词（中英文混合）
    words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', response.lower())
    
    if len(words) < 3:
        return 0.0
    
    # ==== 1. 基础词汇唯一率 ====
    unique_ratio = len(set(words)) / len(words)
    base_score = min(1.0, unique_ratio * 1.2)  # 稍微放大，因为自然语言有重复
    
    # ==== 2. 实词比例（过滤停用词）====
    content_words = [w for w in words if w not in stopwords and len(w) > 1]
    if len(words) > 0:
        content_ratio = len(content_words) / len(words)
        # 实词比例在 0.3-0.7 之间较合理
        if content_ratio < 0.2:
            base_score *= 0.7  # 虚词太多
        elif content_ratio > 0.8:
            base_score *= 0.9  # 可能缺少连接词，不够自然
    
    # ==== 3. 词汇覆盖广度 ====
    # 检查是否使用了描述性词汇（形容词、动词等的特征）
    descriptive_patterns = [
        r'[\u4e00-\u9fff]*[色彩形状大小高低长短]',  # 描述外观
        r'[\u4e00-\u9fff]*[美丽漂亮精致优雅]',      # 描述美感
        r'[\u4e00-\u9fff]*[位于坐落处于]',          # 描述位置
    ]
    has_descriptive = any(re.search(p, response) for p in descriptive_patterns)
    if has_descriptive:
        base_score = min(1.0, base_score + 0.1)
    
    return max(0.0, min(1.0, base_score))


def fluency_reward(response: str) -> float:
    """
    流畅性奖励：惩罚重复和不流畅的表达
    
    检测:
    1. 连续重复词（如"的的的"）
    2. 重复短语（3-gram 重复）
    
    输出范围 0-1
    """
    penalty = 0.0
    
    # 检查连续重复词
    words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', response)
    if len(words) > 1:
        repeat_count = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
        if repeat_count > 2:
            penalty += 0.4
    
    # 检查重复短语（3-gram）
    if len(words) >= 6:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        unique_trigrams = set(trigrams)
        if len(trigrams) > 0:
            repeat_ratio = 1 - len(unique_trigrams) / len(trigrams)
            if repeat_ratio > 0.3:
                penalty += 0.4
    
    # 基础分为 1.0，扣除惩罚后保底 0
    return max(0.0, 1.0 - penalty)


def format_and_length_reward(response: str, min_len: int = 20, ideal_len: int = 100, max_len: int = 300) -> float:
    """
    格式与长度奖励：综合评估输出的格式规范性和长度合理性
    
    评估维度:
    1. 长度合理性（占 0.4）
    2. 句子结构完整性（占 0.3）
    3. 无异常字符（占 0.3）
    
    输出范围 0-1
    """
    response = response.strip()
    
    if not response:
        return 0.0
    
    score = 0.0
    length = len(response)
    
    # ==== 1. 长度合理性 (0.4) ====
    if length < min_len:
        # 太短：线性增长 0 ~ 0.2
        score += 0.2 * (length / min_len)
    elif length <= ideal_len:
        # 理想长度：满分
        score += 0.4
    elif length <= max_len:
        # 稍长：轻微下降 0.4 ~ 0.2
        score += 0.4 - 0.2 * (length - ideal_len) / (max_len - ideal_len)
    else:
        # 太长：固定低分
        score += 0.1
    
    # ==== 2. 句子结构完整性 (0.3) ====
    # 以正确标点结尾
    if response[-1] in '。.！!？?':
        score += 0.15
    elif response[-1] in '；;':
        score += 0.1
    elif response[-1] in '，,、：:':
        score += 0.02  # 截断
    
    # 首字符合理
    first_char = response[0]
    if re.match(r'[\u4e00-\u9fff]', first_char) or first_char.isupper() or first_char.isdigit():
        score += 0.1
    elif first_char in '"「『【':
        score += 0.08
    
    # 有合理断句
    sentence_endings = len(re.findall(r'[。.！!？?]', response))
    if sentence_endings >= 1:
        score += 0.05
    
    # ==== 3. 无异常字符 (0.3) ====
    penalty = 0.0
    
    # 特殊 token 残留
    if '<|' in response or '|>' in response or '<s>' in response or '</s>' in response:
        penalty += 0.15
    
    # 重复标点
    if re.search(r'([。，！？,.!?])\1{2,}', response):
        penalty += 0.05
    
    # 乱码
    if re.search(r'[^\u4e00-\u9fff\u0000-\u007f\u3000-\u303f\uff00-\uffef]{3,}', response):
        penalty += 0.1
    
    score += max(0.0, 0.3 - penalty)
    
    return max(0.0, min(1.0, score))


def compute_reward(response: str) -> float:
    """
    计算图片描述任务的综合奖励
    
    奖励组成（每个奖励范围 0-1，总分范围 0-3）:
    - 格式与长度: 0 ~ 1
    - 词汇丰富度: 0 ~ 1
    - 流畅性: 0 ~ 1
    """
    reward = 0.0
    reward += format_and_length_reward(response)
    reward += vocabulary_diversity_reward(response)
    reward += fluency_reward(response)
    return reward


# ==================== 数据集（Parquet，与 lm_dataset 一致）====================
class GRPOVLMDataset(Dataset):
    """
    GRPO 训练数据集：从 Parquet 读取，与 dataset/lm_dataset.py 格式一致。
    - Parquet 列: conversations (JSON), image_bytes (bytes/list)
    - 只返回 prompt（用户问题 + 图像占位），模型生成响应后由奖励函数评分
    """
    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        preprocess,
        image_special_token: str = '@' * 196,
        max_length: int = 512,
    ):
        self.table = pq.read_table(parquet_path)
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.max_length = max_length
        Logger(f"加载了 {len(self.table)} 条数据（Parquet: {parquet_path}）")

    def __len__(self):
        return len(self.table)

    def _create_prompt(self, conversations: list) -> str:
        """从 conversations 取第一条 user 内容，替换 <image> 为 image_token，并套 chat 模板（仅 user，带 add_generation_prompt）"""
        user_content = ""
        for conv in conversations:
            if conv.get('role') == 'user' or (isinstance(conv, dict) and conv.get('content')):
                role = conv.get('role', 'user')
                if role == 'user' or user_content == "":
                    user_content = conv.get('content', '')
                    if role == 'user':
                        break
        user_content = user_content.replace('<image>', self.image_token)
        messages = [{"role": "user", "content": user_content}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def __getitem__(self, index: int):
        """
        返回: prompt_ids, pixel_values（与 GRPOCollator 期望一致）
        - prompt_ids: 用户问题的 token 序列
        - pixel_values: [num_images, 3, H, W]，从 image_bytes 解码
        """
        conversations = json.loads(self.table['conversations'][index].as_py())
        image_bytes = self.table['image_bytes'][index].as_py()
        if not isinstance(image_bytes, list):
            image_bytes = [image_bytes]

        prompt = self._create_prompt(conversations)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        prompt_ids = prompt_ids[:self.max_length]
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)

        image_tensors = [
            MiniMindVLM.image2tensor(Image.open(io.BytesIO(img)), self.preprocess).squeeze(0)
            for img in image_bytes
        ]
        if image_tensors:
            pixel_values = torch.stack(image_tensors, dim=0)  # [num_images, 3, H, W]
        else:
            pixel_values = torch.zeros(1, 3, 224, 224)

        return {
            'prompt_ids': prompt_tensor,
            'pixel_values': pixel_values,
        }


# ==================== GRPO Trainer ====================
class GRPOTrainer(GRPOTrainerBase):
    def train_step(self, batch):
        """
        执行一个 GRPO 训练步骤
        
        GRPO 算法流程:
        1. 对每个 prompt 生成 num_generations 个响应
        2. 计算每个响应的奖励
        3. 在组内计算相对优势 (advantage = (reward - mean) / std)
        4. 计算策略损失
        5. 计算 KL 惩罚项 (防止策略偏离参考模型太远)
        6. 反向传播更新模型
        """
        prompt_ids = batch['prompt_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        pixel_values = batch['pixel_values'].to(self.device)
        
        batch_size = prompt_ids.size(0)
        prompt_len = prompt_ids.size(1)
        
        # ==== 1. 生成多个响应 ====
        # 每个 prompt 生成 num_generations 个不同的响应（采样生成）
        generated_ids, generated_attention_mask, response_texts = self.generate_responses(
            prompt_ids, attention_mask, pixel_values
        )
        
        # ==== 2. 计算奖励 ====
        # 使用图片描述任务的奖励函数评估每个响应
        rewards = []
        for response in response_texts:
            reward = compute_reward(response)
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        # ==== 3. 计算组内相对优势 ====
        # 每组 num_generations 个响应，计算组内的相对优势
        rewards_grouped = rewards.view(batch_size, self.num_generations)
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
        std_rewards = rewards_grouped.std(dim=1, keepdim=True) + 1e-8  # 防止除零
        advantages = (rewards_grouped - mean_rewards) / std_rewards
        advantages = advantages.view(-1).detach()
        
        # ==== 4. 准备训练数据 ====
        # 扩展 pixel_values 以匹配生成的响应数量
        expanded_pixel_values = pixel_values.repeat_interleave(self.num_generations, dim=0)
        # 使用生成时产生的 attention_mask，正确标记 pad 位置
        expanded_attention_mask = generated_attention_mask
        
        # prompt_mask 用于区分 prompt 部分和生成部分，只对生成部分计算损失
        prompt_mask = torch.zeros_like(generated_ids)
        prompt_mask[:, :prompt_len] = 1
        
        # ==== 5. 计算 π_old 的 per-token log prob（生成时的策略，固定不更新）====
        with torch.no_grad():
            _, old_token_log_probs, response_mask = self.compute_log_probs(
                self.model, generated_ids, expanded_attention_mask,
                expanded_pixel_values, generated_ids, prompt_mask
            )
        
        # ==== 6. 计算 π_ref 的 per-token log prob（参考策略，用于 KL 惩罚）====
        with torch.no_grad():
            _, ref_token_log_probs, _ = self.compute_log_probs(
                self.ref_model, generated_ids, expanded_attention_mask,
                expanded_pixel_values, generated_ids, prompt_mask
            )
        
        # advantages 扩展到 token 维度 [batch] -> [batch, 1]
        token_advantages = advantages.unsqueeze(-1)
        
        # ==== 7. 在同一批数据上做多次梯度更新，让 ratio 逐渐偏离 1 ====
        num_inner_updates = 4
        for _ in range(num_inner_updates):
            # 计算 π_θ 的 per-token log prob
            _, token_log_probs, _ = self.compute_log_probs(
                self.model, generated_ids, expanded_attention_mask,
                expanded_pixel_values, generated_ids, prompt_mask
            )
            
            # ==== Per-token ratio（TRL 风格）====
            log_ratio = token_log_probs - old_token_log_probs
            ratio = torch.exp(log_ratio.clamp(-10, 10))
            
            # ==== GRPO Loss: -min(ratio * A, clip(ratio) * A) ====
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            per_token_loss1 = ratio * token_advantages
            per_token_loss2 = clipped_ratio * token_advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            
            # ==== Per-token KL 惩罚（相对于参考模型）====
            per_token_kl = torch.zeros_like(per_token_loss)
            if self.beta > 0:
                ref_log_ratio = token_log_probs - ref_token_log_probs
                ref_ratio = torch.exp(ref_log_ratio.clamp(-10, 10))
                per_token_kl = (ref_ratio - 1) - ref_log_ratio
            
            # 加入 KL 惩罚
            per_token_loss = per_token_loss + self.beta * per_token_kl
            
            # ==== 计算 masked loss（只对 response 部分）====
            # loss = sum(per_token_loss * mask) / sum(mask)
            masked_loss = (per_token_loss * response_mask).sum(dim=-1)
            seq_lengths = response_mask.sum(dim=-1).clamp(min=1)
            loss = (masked_loss / seq_lengths).mean()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        # 计算指标
        avg_ratio = ((ratio * response_mask).sum() / response_mask.sum().clamp(min=1)).item()
        avg_kl = ((per_token_kl * response_mask).sum() / response_mask.sum().clamp(min=1)).item()
        
        metrics = {
            'loss': loss.item(),
            'kl_div': avg_kl,
            'mean_reward': rewards.mean().item(),
            'max_reward': rewards.max().item(),
            'min_reward': rewards.min().item(),
            'ratio': avg_ratio,
        }
        
        return loss, metrics


# ==================== 主函数 ====================
def main():
    # 第 1 部分：命令行参数解析
    parser = argparse.ArgumentParser(description="MiniMind-V GRPO (图片描述)")
    
    # ---------- 数据相关参数 ----------
    parser.add_argument("--data_path", type=str, default="../dataset/sft_i2t.parquet",
                        help="训练数据路径（Parquet，需含 conversations、image_bytes 列）")
    parser.add_argument("--max_samples", type=int, default=500, 
                        help="最大训练样本数")
    
    # ---------- 模型结构参数 ----------
    parser.add_argument('--hidden_size', default=512, type=int,
                        help="LLM 隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int,
                        help="LLM Transformer 层数")
    parser.add_argument('--max_seq_len', default=1024, type=int,
                        help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1],
                        help="是否使用 MoE 架构")
    
    # ---------- GRPO 算法参数 ----------
    parser.add_argument("--num_generations", type=int, default=4,
                        help="每个 prompt 生成的响应数量（组大小）")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="生成的最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="采样温度，越高越随机")
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL 惩罚系数，越大越约束策略不偏离参考模型")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clip 范围，限制策略更新幅度")
    
    # ---------- 训练超参数 ----------
    parser.add_argument("--epochs", type=int, default=1,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="批次大小（GRPO实际计算量=batch_size*num_generations）")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                        help="学习率（RL 阶段建议很小）")
    
    # ---------- 保存和日志参数 ----------
    parser.add_argument("--save_dir", type=str, default="../out",
                        help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo_vlm', type=str,
                        help="保存权重的前缀名")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="模型保存间隔")
    
    # ---------- 其他参数 ----------
    parser.add_argument('--from_weight', default='sft_vlm', type=str,
                        help="基于哪个权重训练（通常是 SFT 模型）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="训练设备")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    parser.add_argument("--use_wandb", type=int, default=1, choices=[0, 1],
                        help="是否使用 wandb 记录日志 (默认开启)")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-GRPO",
                        help="wandb 项目名")
    
    args = parser.parse_args()

    # 第 2 部分：初始化环境
    setup_seed(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if args.data_path.startswith('../'):
        args.data_path = os.path.join(project_root, args.data_path.replace('../', ''))
    
    save_dir = os.path.join(project_root, args.save_dir.replace('../', ''))
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir
    
    # 第 3 部分：配置模型参数
    vlm_config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe)
    )

    # 第 4 部分：初始化模型
    # GRPO 需要两个模型：
    # - 策略模型 (model): 训练中更新参数
    # - 参考模型 (ref_model): 冻结参数，用于计算 KL 散度约束
    Logger("初始化策略模型...")
    model, tokenizer, preprocess = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,
        tokenizer_path=os.path.join(project_root, 'model'),
        vision_model_path=os.path.join(project_root, 'model/vision_model/clip-vit-base-patch16'),
        save_dir=os.path.join(project_root, 'out'),
        device=args.device,
        freeze_llm=False
    )
    
    Logger("初始化参考模型...")
    ref_model, _, _ = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,
        tokenizer_path=os.path.join(project_root, 'model'),
        vision_model_path=os.path.join(project_root, 'model/vision_model/clip-vit-base-patch16'),
        save_dir=os.path.join(project_root, 'out'),
        device=args.device,
        freeze_llm=False
    )
    # 冻结参考模型所有参数
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    # 第 5 部分：准备数据集
    Logger("=" * 50)
    Logger("加载图片描述数据集 (GRPO)")
    Logger("=" * 50)
    
    train_ds = GRPOVLMDataset(
        parquet_path=args.data_path,
        tokenizer=tokenizer,
        preprocess=preprocess,
        image_special_token=vlm_config.image_special_token,
        max_length=vlm_config.max_seq_len,
    )
    
    # 快速训练模式：随机采样数据
    max_samples = 100
    original_size = len(train_ds)
    if original_size > max_samples:
        seed = int(time.time()) % 10000
        random.seed(seed)
        np.random.seed(seed)
        sampled_indices = random.sample(range(original_size), max_samples)
        sampled_indices.sort()
        train_ds = Subset(train_ds, sampled_indices)
        Logger(f"⚠️  快速训练模式：随机采样{max_samples}条数据（总共{original_size}条，随机种子={seed}）")
    
    data_collator = GRPOCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 第 6 部分：配置 wandb 日志
    wandb = None
    if args.use_wandb:
        import swanlab as wandb
        wandb.init(project=args.wandb_project, name=f"GRPO-VLM-Epoch-{args.epochs}")

    # 第 7 部分：初始化优化器和训练器
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )
    
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=args.device,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        beta=args.beta,
        clip_range=args.clip_range,
    )

    # 第 8 部分：开始训练循环
    Logger("=" * 50)
    Logger("开始 GRPO 训练（图片描述任务）")
    Logger("=" * 50)
    Logger(f"训练样本数: {len(train_ds)}")
    Logger(f"批次大小: {args.batch_size}")
    Logger(f"每prompt生成数量: {args.num_generations}")
    Logger(f"学习率: {args.learning_rate}")
    
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_reward = 0.0
        
        for step, batch in enumerate(train_loader):
            loss, metrics = trainer.train_step(batch)
            
            epoch_loss += metrics['loss']
            epoch_reward += metrics['mean_reward']
            global_step += 1
            
            # 动态学习率调整
            lr = get_lr(global_step, args.epochs * len(train_loader), args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # 打印日志
            if step % args.log_interval == 0:
                Logger(f"Epoch [{epoch+1}/{args.epochs}] Step [{step}/{len(train_loader)}] "
                       f"Loss: {metrics['loss']:.4f} "
                       f"Reward: {metrics['mean_reward']:.4f} "
                       f"KL: {metrics['kl_div']:.4f} "
                       f"Ratio: {metrics['ratio']:.4f}")
                
                if wandb:
                    wandb.log({
                        "loss": metrics['loss'],
                        "mean_reward": metrics['mean_reward'],
                        "kl_div": metrics['kl_div'],
                        "ratio": metrics['ratio'],
                        "learning_rate": lr,
                    })
            
            # 保存检查点
            if step > 0 and step % args.save_interval == 0:
                save_checkpoint(model, vlm_config, args)
        
        avg_loss = epoch_loss / len(train_loader)
        avg_reward = epoch_reward / len(train_loader)
        Logger(f"Epoch [{epoch+1}/{args.epochs}] 完成 - Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

    # 第 9 部分：保存最终模型
    save_checkpoint(model, vlm_config, args)
    Logger("GRPO 训练完成！")


def save_checkpoint(model, vlm_config, args):
    """保存模型检查点"""
    model.eval()
    moe_suffix = '_moe' if vlm_config.use_moe else ''
    ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
    
    state_dict = model.state_dict()
    # 保存完整模型（包括 vision_encoder）
    clean_state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
    torch.save(clean_state_dict, ckp)
    Logger(f"模型已保存到: {ckp}")
    model.train()


if __name__ == "__main__":
    main()
