"""
InternVL3-8B LoRA GRPO on GeoQA
Proposal Stage 2: 强化学习阶段

奖励函数（proposal 第 3 节）:
  - 格式奖励  +1:  输出包含 <think> 标签（防止模型走捷径跳过推理）
  - 准确性奖励 +2:  <answer>X</answer> 与 ground truth 精确匹配

从 SFT 权重出发，使用 rl_filtered.parquet 中的 Pass@K 筛选数据训练。

使用方法:
    torchrun --nproc_per_node=2 trainer/train_grpo_geoqa.py \\
        --model_path   ./InternVL3-8B \\
        --adapter_path out/geoqa/sft_geoqa_final \\
        --train_data   dataset/geoqa/rl_filtered.parquet \\
        --save_dir     out/geoqa
"""

import os
import re
import sys
import json
import time
import argparse
import warnings

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

from dataset.lm_dataset import GeoQADataset
from trainer.trainer_utils import (
    Logger, get_lr, is_main_process, setup_seed, GRPOCollator, GRPOTrainerBase
)

warnings.filterwarnings('ignore')


# ==================== 奖励函数 ====================

def extract_answer(text: str) -> str | None:
    """提取 <answer>X</answer> 中的字母"""
    m = re.search(r'<answer>\s*([A-D])\s*</answer>', text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def compute_reward(response: str, ground_truth: str) -> float:
    """
    奖励 = 格式奖励 + 准确性奖励  （最大值 3.0）

    格式奖励 (+1.0):
        输出含 <think>...</think> 标签，防止模型跳过推理直接输出答案

    准确性奖励 (+2.0):
        <answer>X</answer> 与 ground_truth 精确匹配
    """
    # 格式奖励：必须包含 <think> 标签
    has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL | re.IGNORECASE))
    format_r = 1.0 if has_think else 0.0

    # 准确性奖励
    pred = extract_answer(response)
    accuracy_r = 2.0 if (pred and pred == ground_truth) else 0.0

    return format_r + accuracy_r


# ==================== GRPO Trainer ====================

class GeoQAGRPOTrainer(GRPOTrainerBase):
    """GeoQA 专用 GRPO Trainer，继承 GRPOTrainerBase 的生成和 log_prob 逻辑"""

    def train_step(self, batch, answers):
        prompt_ids       = batch['prompt_ids'].to(self.device)
        attention_mask   = batch['attention_mask'].to(self.device)
        pixel_values     = batch['pixel_values'].to(self.device)

        batch_size = prompt_ids.size(0)
        prompt_len = prompt_ids.size(1)

        # 1. 生成 num_generations 个响应
        generated_ids, gen_attn_mask, response_texts = self.generate_responses(
            prompt_ids, attention_mask, pixel_values
        )

        # 2. 计算每个响应的奖励
        rewards = torch.tensor([
            compute_reward(response_texts[i], answers[i // self.num_generations])
            for i in range(len(response_texts))
        ], dtype=torch.float32, device=self.device)

        # 3. 组内相对优势（GRPO 核心）
        rewards_grouped = rewards.view(batch_size, self.num_generations)
        mean_r = rewards_grouped.mean(dim=1, keepdim=True)
        std_r  = rewards_grouped.std(dim=1,  keepdim=True) + 1e-8
        advantages = ((rewards_grouped - mean_r) / std_r).view(-1).detach()

        expanded_pv    = pixel_values.repeat_interleave(self.num_generations, dim=0)
        prompt_mask    = torch.zeros_like(generated_ids)
        prompt_mask[:, :prompt_len] = 1

        # 4. 计算 π_old 和 π_ref 的 per-token log prob（不更新参数）
        with torch.no_grad():
            _, old_lp, response_mask = self.compute_log_probs(
                self.model, generated_ids, gen_attn_mask,
                expanded_pv, generated_ids, prompt_mask
            )
            _, ref_lp, _ = self.compute_log_probs(
                self.ref_model, generated_ids, gen_attn_mask,
                expanded_pv, generated_ids, prompt_mask
            )

        token_advantages = advantages.unsqueeze(-1)

        # 5. 多次内层更新（让 ratio 逐渐偏离 1）
        for _ in range(4):
            _, curr_lp, _ = self.compute_log_probs(
                self.model, generated_ids, gen_attn_mask,
                expanded_pv, generated_ids, prompt_mask
            )

            ratio         = torch.exp((curr_lp - old_lp).clamp(-10, 10))
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            pg_loss       = -torch.min(ratio * token_advantages,
                                       clipped_ratio * token_advantages)

            kl_penalty = torch.zeros_like(pg_loss)
            if self.beta > 0:
                ref_ratio  = torch.exp((curr_lp - ref_lp).clamp(-10, 10))
                kl_penalty = (ref_ratio - 1) - (curr_lp - ref_lp)

            per_token_loss = pg_loss + self.beta * kl_penalty
            masked_loss    = (per_token_loss * response_mask).sum(dim=-1)
            loss           = (masked_loss / response_mask.sum(dim=-1).clamp(min=1)).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
            )
            self.optimizer.step()

        return loss, {
            'loss':        loss.item(),
            'mean_reward': rewards.mean().item(),
            'max_reward':  rewards.max().item(),
            'kl_div':      ((kl_penalty * response_mask).sum() /
                            response_mask.sum().clamp(min=1)).item(),
        }


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='InternVL3-8B GRPO on GeoQA')

    parser.add_argument('--model_path',    type=str, default='./InternVL3-8B')
    parser.add_argument('--adapter_path',  type=str, required=True,
                        help='SFT LoRA adapter 目录（训练起点）')
    parser.add_argument('--train_data',    type=str,
                        default='dataset/geoqa/rl_filtered.parquet')
    parser.add_argument('--save_dir',      type=str, default='out/geoqa')

    parser.add_argument('--lora_rank',     type=int,   default=16)
    parser.add_argument('--max_length',    type=int,   default=512)
    parser.add_argument('--dtype',         type=str,   default='bfloat16')

    # GRPO 参数
    parser.add_argument('--num_generations', type=int,   default=4,
                        help='每个 prompt 生成的响应数（组大小）')
    parser.add_argument('--max_new_tokens',  type=int,   default=256)
    parser.add_argument('--temperature',     type=float, default=0.7)
    parser.add_argument('--beta',            type=float, default=0.04,
                        help='KL 惩罚系数')
    parser.add_argument('--clip_range',      type=float, default=0.2)

    # 训练参数
    parser.add_argument('--epochs',          type=int,   default=1)
    parser.add_argument('--batch_size',      type=int,   default=2)
    parser.add_argument('--learning_rate',   type=float, default=5e-7)

    parser.add_argument('--save_dir_out',    type=str,   default='out/geoqa')
    parser.add_argument('--log_interval',    type=int,   default=10)
    parser.add_argument('--save_interval',   type=int,   default=100)
    parser.add_argument('--num_workers',     type=int,   default=4)
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--use_wandb',       type=int,   default=1, choices=[0, 1])
    parser.add_argument('--wandb_project',   type=str,   default='GeoQA-GRPO')
    parser.add_argument('--debug',           action='store_true')

    args = parser.parse_args()
    setup_seed(args.seed)

    # 分布式初始化
    local_rank = 0
    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
    args.device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    for attr in ('train_data', 'save_dir'):
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.join(project_root, val))
    os.makedirs(args.save_dir, exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    # ---------- 加载模型 ----------
    Logger('加载策略模型（SFT 权重 + LoRA）...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        args.model_path, size={'height': 448, 'width': 448}
    )

    def load_model():
        m = AutoModel.from_pretrained(
            args.model_path, torch_dtype=dtype, trust_remote_code=True,
            use_flash_attn=False,
        )
        m = PeftModel.from_pretrained(m, args.adapter_path)
        return m.merge_and_unload().to(args.device)

    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    model     = load_model()
    ref_model = load_model()
    model.img_context_token_id     = img_context_token_id
    ref_model.img_context_token_id = img_context_token_id
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    # 给策略模型加新 LoRA（在 SFT 权重上继续用 LoRA 微调）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    image_token_str = f'<img>{"<IMG_CONTEXT>" * model.num_image_token}</img>'

    # ---------- 数据集 ----------
    Logger('加载 RL 训练数据 ...')
    train_ds = GeoQADataset(
        parquet_path=args.train_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mode='grpo',
        max_length=args.max_length,
        image_token_str=image_token_str,
    )
    if args.debug:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(20))

    sampler   = DistributedSampler(train_ds) if dist.is_initialized() else None
    collator  = GRPOCollator(pad_token_id=tokenizer.pad_token_id)
    loader    = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---------- wandb ----------
    wandb = None
    if args.use_wandb and not args.debug and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project, name='InternVL3-8B-GRPO')

    # ---------- 优化器 + Trainer ----------
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )
    trainer = GeoQAGRPOTrainer(
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

    # ---------- 训练循环 ----------
    Logger('=' * 60)
    Logger('开始 GRPO 训练')
    Logger(f'  数据量:      {len(train_ds)}')
    Logger(f'  batch_size:  {args.batch_size}  ×  num_gen={args.num_generations}')
    Logger(f'  epochs:      {args.epochs}')
    Logger('=' * 60)

    global_step = 0
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()

        for step, batch in enumerate(loader):
            answers = batch.pop('answers', None)  # GRPOCollator 传入的 ground truth 列表
            loss, metrics = trainer.train_step(batch, answers)

            lr = get_lr(global_step, args.epochs * len(loader), args.learning_rate)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            global_step += 1

            if step % args.log_interval == 0:
                Logger(
                    f'Epoch[{epoch+1}/{args.epochs}] Step[{step}/{len(loader)}] '
                    f'loss={metrics["loss"]:.4f} '
                    f'reward={metrics["mean_reward"]:.3f} '
                    f'kl={metrics["kl_div"]:.4f} '
                    f'lr={lr:.2e}'
                )
                if wandb and is_main_process():
                    wandb.log({
                        'grpo/loss':        metrics['loss'],
                        'grpo/mean_reward': metrics['mean_reward'],
                        'grpo/kl_div':      metrics['kl_div'],
                        'grpo/lr':          lr,
                    })

            if step % args.save_interval == 0 and is_main_process():
                _save_grpo(model, args, suffix=f'step{global_step}')

    if is_main_process():
        _save_grpo(model, args, suffix='final')
    Logger('GRPO 训练完成！')


def _save_grpo(model, args, suffix='final'):
    path = os.path.join(args.save_dir, f'grpo_geoqa_{suffix}')
    raw = model.module if hasattr(model, 'module') else model
    raw.save_pretrained(path)
    Logger(f'Saved → {path}')


if __name__ == '__main__':
    main()
