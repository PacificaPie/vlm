import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
import random
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig
from dataset.lm_dataset import VLMDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_vlm_model, vlm_checkpoint, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    start_time = time.time()
    for step, (input_ids, labels, pixel_values) in enumerate(loader, start=start_step + 1):
        """
        - input_ids: 完整 token 序列 [batch_size, seq_len]
        - labels: 与 input_ids 等长，仅对 assistant 回复部分为 token id，其余为 -100（不参与损失）
        - pixel_values: 图像张量
        """
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        pixel_values = pixel_values.to(args.device)
        
        # 动态学习率调整
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, pixel_values=pixel_values)
            # 因果 LM：logits[:, i] 预测下一 token，即 labels[:, i+1]；labels 中 -100 由 ignore_index 忽略
            loss = loss_fct(
                res.logits[:, :-1, :].reshape(-1, res.logits.size(-1)),
                labels[:, 1:].reshape(-1)
            )
            # 添加 MoE 辅助损失（用于负载均衡）
            loss += res.aux_loss
            # 梯度累积
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # 保存模型检查点（包含优化器状态，用于断点续训）
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir=args.save_dir, scaler=scaler)
            model.train()

        del input_ids, labels, pixel_values, res, loss
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # 第 1 部分：命令行参数解析
    parser = argparse.ArgumentParser(description="MiniMind-V SFT")
    
    # ---------- 保存相关参数 ----------
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='sft_vlm', type=str, help="保存权重的前缀名")
    
    # ---------- 训练超参数 ----------
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="每个设备的 batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="初始学习率（SFT 阶段较小）")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    
    # ---------- 设备和精度 ----------
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    
    # ---------- 日志和保存间隔 ----------
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    
    # ---------- 模型结构参数 ----------
    parser.add_argument('--hidden_size', default=512, type=int, help="LLM 隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="LLM Transformer 层数")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用 MoE 架构")
    
    # ---------- 数据路径 ----------
    parser.add_argument("--data_path", type=str, default="../dataset/sft_i2t.parquet", help="SFT 训练数据路径")
    
    # ---------- 模型加载和训练策略 ----------
    parser.add_argument('--from_weight', default='pretrain_vlm', type=str, help="基于哪个预训练权重初始化")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测并续训")
    parser.add_argument("--use_wandb", type=int, default=1, choices=[0, 1], help="是否使用 wandb/swanlab 记录日志 (默认开启)")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-SFT", help="wandb 项目名称")
    
    args = parser.parse_args()

    # 第 2 部分：初始化分布式环境和随机种子
    local_rank = init_distributed_mode()
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # 第 3 部分：配置目录、模型参数、检查点
    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = VLMConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        max_seq_len=args.max_seq_len, 
        use_moe=bool(args.use_moe)
    )
    ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir=args.save_dir) if args.from_resume==1 else None
    
    # 第 4 部分：设置混合精度训练
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # 第 5 部分：配置 wandb/swanlab 日志记录
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-V-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # 第 6 部分：初始化模型、数据集、优化器
    # SFT 阶段：基于 pretrain_vlm 权重，全参数微调
    model, tokenizer, preprocess = init_vlm_model(
        vlm_config, 
        from_weight=args.from_weight, 
        device=args.device
    )
    train_ds = VLMDataset(
        args.data_path, 
        tokenizer, 
        preprocess=preprocess,
        image_special_token=vlm_config.image_special_token,
        max_length=vlm_config.max_seq_len
    )
    
    # 快速训练模式：随机采样300条数据
    max_samples = 300
    original_size = len(train_ds)
    if original_size > max_samples:
        seed = int(time.time()) % 10000
        random.seed(seed)
        np.random.seed(seed)
        sampled_indices = random.sample(range(original_size), max_samples)
        sampled_indices.sort()
        train_ds = Subset(train_ds, sampled_indices)
        Logger(f"⚠️  快速训练模式：随机采样{max_samples}条数据（总共{original_size}条，随机种子={seed}）")
    
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 第 7 部分：从检查点恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # 第 8 部分：DDP 包装模型
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # 第 9 部分：开始训练循环
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        
        # 情况 1: 断点续训（从 epoch 的中间位置继续）
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        # 情况 2: 正常训练（从 epoch 开头开始）
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), 0, wandb)
