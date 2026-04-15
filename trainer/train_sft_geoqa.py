"""
InternVL3-8B LoRA SFT on GeoQA
Proposal Stage 1: 视觉热身（冻结 LLM）+ LoRA 全参微调

两阶段训练：
  Stage 1 - Visual Warmup (200 steps, lr=1e-4):
      冻结语言模型，只训练 ViT + MLP projector
      让模型先学会对齐几何图形的视觉-语言特征

  Stage 2 - LoRA SFT (3 epochs, lr=1e-5):
      解冻 LoRA adapters，全参微调（含视觉部分）
      用 <think>...</think><answer>X</answer> 格式的 CoT 数据训练

使用方法（单卡调试）:
    python trainer/train_sft_geoqa.py --model_path ./InternVL3-8B --debug

使用方法（多卡，由 SLURM 调用）:
    torchrun --nproc_per_node=2 trainer/train_sft_geoqa.py \\
        --model_path ./InternVL3-8B \\
        --train_data dataset/geoqa/sft_train.parquet \\
        --save_dir out/geoqa
"""

import os
import sys
import math
import time
import json
import argparse
import warnings

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from peft import LoraConfig, get_peft_model, TaskType

from dataset.lm_dataset import GeoQADataset
from trainer.trainer_utils import Logger, get_lr, is_main_process, setup_seed

warnings.filterwarnings('ignore')


# ==================== InternVL3 辅助函数 ====================

def build_image_token_str(model, tokenizer) -> str:
    """
    构造 InternVL3 的图像 token 串，用于替换 conversations 里的 <image>。

    InternVL3 格式: <img><IMG_CONTEXT>×num_image_token</img>
    训练时 model.forward() 会把 <IMG_CONTEXT> 位置的 embedding 替换为视觉特征。
    """
    num_image_token = model.num_image_token  # 通常 256
    img_context_token = '<IMG_CONTEXT>'
    # 确保 tokenizer 里有这些特殊 token
    assert img_context_token in tokenizer.get_vocab(), \
        f"{img_context_token} not in tokenizer vocab. Is this really InternVL3?"
    return f'<img>{img_context_token * num_image_token}</img>'


def freeze_llm(model):
    """Stage 1: 冻结语言模型，只训练 ViT + MLP projector"""
    for name, param in model.named_parameters():
        if 'language_model' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'[Warmup] 可训练参数: {trainable / 1e6:.1f}M (ViT + MLP projector only)')


def apply_lora_and_unfreeze(model, lora_rank: int = 16):
    """
    Stage 2: 对语言模型的 attention 层加 LoRA，并解冻视觉部分。
    LoRA 只加在语言模型上，视觉编码器继续全参训练。
    """
    # 先恢复所有参数可训练，再 apply LoRA（LoRA 会冻结非 adapter 的语言模型参数）
    for param in model.parameters():
        param.requires_grad = True

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[          # Qwen2 attention 层（InternVL3-8B 使用 Qwen2 LLM）
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ],
        # 只对 language_model 里的模块加 LoRA
        modules_to_save=None,
    )
    # peft 的 get_peft_model 会自动找 target_modules 并包装
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ==================== 数据整理器 ====================

class SFTCollator:
    """将变长序列 pad 到 batch 内最大长度"""

    def __init__(self, pad_token_id: int):
        self.pad_id = pad_token_id

    def __call__(self, batch):
        # batch: list of (X, Y, mask, pixel_values)
        max_len = max(x.size(0) for x, _, _, _ in batch)

        X_list, Y_list, mask_list, pv_list = [], [], [], []
        for X, Y, mask, pv in batch:
            pad = max_len - X.size(0)
            X_list.append(torch.cat([X, torch.full((pad,), self.pad_id)]))
            Y_list.append(torch.cat([Y, torch.full((pad,), -100)]))   # -100 → ignore
            mask_list.append(torch.cat([mask, torch.zeros(pad)]))
            pv_list.append(pv)  # [1, C, H, W]

        return (
            torch.stack(X_list),           # [B, L]
            torch.stack(Y_list),           # [B, L]
            torch.stack(mask_list),        # [B, L]
            torch.cat(pv_list, dim=0),     # [B, C, H, W]
        )


# ==================== 训练函数 ====================

def train_epoch(epoch, model, loader, optimizer, scaler, autocast_ctx,
                args, stage_name: str, wandb=None):
    model.train()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    iters = len(loader)

    for step, (X, Y, loss_mask, pixel_values) in enumerate(loader, start=1):
        X            = X.to(args.device)
        Y            = Y.to(args.device)
        loss_mask    = loss_mask.to(args.device)
        pixel_values = pixel_values.to(args.device)

        # 动态学习率（余弦退火）
        total_steps = args.warmup_steps if stage_name == 'warmup' else args.epochs * iters
        current_step = step if stage_name == 'warmup' else epoch * iters + step
        lr = get_lr(current_step, total_steps, args.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        with autocast_ctx:
            # image_flags: InternVL3 需要此参数标记哪些样本有图像
            # 单图模式下每个样本对应 1 个 tile，全部置 1
            image_flags = torch.ones(
                pixel_values.size(0), 1,
                dtype=torch.long, device=args.device
            )
            outputs = model(
                input_ids=X,
                pixel_values=pixel_values,
                image_flags=image_flags,
                labels=None,          # 自己算 loss，保留 loss_mask 的精确控制
            )
            logits = outputs.logits   # [B, L, V]

            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                Y.view(-1),
            ).view(Y.size())

            # 只对 assistant 回复部分计算平均损失
            loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                args.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            elapsed = time.time() - start_time
            real_loss = loss.item() * args.accumulation_steps
            eta = elapsed / step * iters // 60 - elapsed // 60
            Logger(
                f'[{stage_name}] Epoch[{epoch+1}/{args.epochs}] '
                f'Step[{step}/{iters}] loss={real_loss:.4f} '
                f'lr={lr:.2e} eta={eta:.0f}min'
            )
            if wandb and is_main_process():
                wandb.log({f'{stage_name}/loss': real_loss,
                           f'{stage_name}/lr':   lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            _save(model, args, suffix=f'epoch{epoch+1}')

        # 热身阶段只跑 warmup_steps 步
        if stage_name == 'warmup' and step >= args.warmup_steps:
            break


def _save(model, args, suffix='final'):
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f'sft_geoqa_{suffix}')
    # 用 peft 的 save_pretrained 只保存 LoRA adapter（体积小）
    # 若是 warmup 阶段（无 LoRA），用标准 save_pretrained
    raw = model.module if isinstance(model, DDP) else model
    if hasattr(raw, 'save_pretrained'):
        raw.save_pretrained(path)
        Logger(f'Saved → {path}')
    else:
        torch.save(raw.state_dict(), path + '.pth')
        Logger(f'Saved → {path}.pth')


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='InternVL3-8B LoRA SFT on GeoQA')

    # 路径
    parser.add_argument('--model_path',  type=str, default='./InternVL3-8B',
                        help='InternVL3-8B 权重目录（本地路径或 HF hub id）')
    parser.add_argument('--train_data',  type=str, default='dataset/geoqa/sft_train.parquet')
    parser.add_argument('--save_dir',    type=str, default='out/geoqa')

    # 模型
    parser.add_argument('--lora_rank',   type=int,   default=16)
    parser.add_argument('--max_length',  type=int,   default=1024)
    parser.add_argument('--dtype',       type=str,   default='bfloat16',
                        choices=['bfloat16', 'float16'])

    # Stage 1 热身
    parser.add_argument('--warmup_steps', type=int,   default=200)
    parser.add_argument('--warmup_lr',    type=float, default=1e-4)

    # Stage 2 SFT
    parser.add_argument('--epochs',           type=int,   default=3)
    parser.add_argument('--batch_size',       type=int,   default=4,
                        help='每卡 batch size（proposal 目标是 32，梯度累积补足）')
    parser.add_argument('--learning_rate',    type=float, default=1e-5)
    parser.add_argument('--accumulation_steps', type=int, default=8,
                        help='梯度累积步数，有效 batch = batch_size × accumulation_steps × n_gpu')
    parser.add_argument('--grad_clip',        type=float, default=1.0)

    # 杂项
    parser.add_argument('--log_interval',   type=int, default=10)
    parser.add_argument('--save_interval',  type=int, default=200)
    parser.add_argument('--num_workers',    type=int, default=4)
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--use_wandb',      type=int, default=1, choices=[0, 1])
    parser.add_argument('--wandb_project',  type=str, default='GeoQA-SFT')
    parser.add_argument('--debug',          action='store_true',
                        help='调试模式：只用 50 条数据，不启用 wandb')

    args = parser.parse_args()

    # ---------- 初始化 ----------
    setup_seed(args.seed)

    # 分布式初始化
    local_rank = 0
    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
    args.device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    # 统一路径到项目根目录
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if not os.path.isabs(args.train_data):
        args.train_data = os.path.join(project_root, args.train_data)
    if not os.path.isabs(args.save_dir):
        args.save_dir = os.path.join(project_root, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    device_type  = 'cuda' if 'cuda' in args.device else 'cpu'
    autocast_ctx = nullcontext() if device_type == 'cpu' \
                   else torch.cuda.amp.autocast(dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    # ---------- 加载模型和 tokenizer ----------
    Logger('加载 InternVL3-8B ...')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_flash_attn=False,
    ).to(args.device)

    # 图像处理器（448×448 单 tile）
    image_processor = CLIPImageProcessor.from_pretrained(
        args.model_path,
        size={'height': 448, 'width': 448},
    )

    # 构造 InternVL3 的图像 token 串
    image_token_str = build_image_token_str(model, tokenizer)
    Logger(f'image_token_str 长度: {len(image_token_str)} chars, '
           f'num_image_token={model.num_image_token}')

    # ---------- wandb ----------
    wandb = None
    if args.use_wandb and not args.debug and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project, name='InternVL3-8B-LoRA-SFT')

    # ---------- 数据集 ----------
    Logger('加载 GeoQA SFT 数据集 ...')
    train_ds = GeoQADataset(
        parquet_path=args.train_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mode='sft',
        max_length=args.max_length,
        image_token_str=image_token_str,
    )

    if args.debug:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(50))
        Logger('DEBUG 模式：只用前 50 条数据')

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)

    def make_loader(batch_size):
        return DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # ============================================================
    # Stage 1: Visual Warmup
    # 冻结语言模型，只训练 ViT + MLP projector
    # ============================================================
    Logger('=' * 60)
    Logger('Stage 1: Visual Warmup')
    Logger('=' * 60)
    freeze_llm(model)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.warmup_lr,
        weight_decay=0.01,
    )

    # 热身用较大 batch，快速扫数据
    warmup_loader = make_loader(batch_size=args.batch_size * 2)
    train_epoch(0, model, warmup_loader, optimizer, scaler, autocast_ctx,
                args, stage_name='warmup', wandb=wandb)

    if is_main_process():
        _save(model, args, suffix='warmup')

    # ============================================================
    # Stage 2: LoRA SFT
    # 解冻并加 LoRA adapter，用 CoT 数据做全参微调
    # ============================================================
    Logger('=' * 60)
    Logger('Stage 2: LoRA SFT')
    Logger('=' * 60)

    # apply LoRA（会修改 model，返回 PeftModel）
    model = apply_lora_and_unfreeze(model, lora_rank=args.lora_rank)
    model.to(args.device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    sft_loader = make_loader(batch_size=args.batch_size)
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        train_epoch(epoch, model, sft_loader, optimizer, scaler, autocast_ctx,
                    args, stage_name='sft', wandb=wandb)

    if is_main_process():
        _save(model, args, suffix='final')

    Logger('SFT 训练完成！')


if __name__ == '__main__':
    main()
