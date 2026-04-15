"""
训练工具函数集合
"""
import os
import gc
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM
    


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    # 标准余弦退火调度: 从lr开始，逐渐下降到lr/10
    min_lr = lr * 0.1
    return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * current_step / total_steps))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_vlm_model(vlm_config, from_weight='pretrain_vlm', tokenizer_path='../model', 
                   vision_model_path='../model/vision_model/clip-vit-base-patch16', 
                   save_dir='../out', device='cuda', freeze_llm=False):
    """
    freeze_llm=True 时：只训练 vision_proj + vision_encoder，冻结 llm
    freeze_llm=False 时：全参数训练（vision_encoder + vision_proj + llm）
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindVLM(vlm_config, vision_model_path=vision_model_path)
    
    if from_weight != 'none':
        moe_suffix = '_moe' if vlm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        # strict=False: 允许权重文件中缺少部分参数（如旧版本不含 vision_encoder）
        model.load_state_dict(weights, strict=False)
        Logger(f'已加载权重: {weight_path}')
    
    # Pretrain阶段：冻结 llm，只训练 vision_encoder + vision_proj
    if freeze_llm:
        for name, param in model.named_parameters():
            # 只训练 vision 相关参数
            if 'vision_encoder' in name or 'vision_proj' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    Logger(f'所加载VLM Model可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    preprocess = model.processor
    return model.to(device), tokenizer, preprocess


def vlm_checkpoint(vlm_config, weight='pretrain_vlm', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../out', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if vlm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}_resume.pth'
    
    if model is not None:  # 保存模式
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        
        # 保存模型权重 (half 精度，用于推理/部署)
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        # 获取 wandb/swanlab 运行 ID (用于续训时恢复日志)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
        
        # 保存完整检查点 (用于断点续训)
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        # 保存额外参数 (如 scaler)
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        gc.collect()
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            # GPU 数量变化时自动转换 step
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
    
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


# ==================== GRPO 公共组件 ====================

class GRPOCollator:
    """GRPO 数据整理器（公共基类）"""
    
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, features):
        max_len = max(f['prompt_ids'].size(0) for f in features)
        
        prompt_ids_list = []
        attention_mask_list = []
        
        for f in features:
            prompt_ids = f['prompt_ids']
            pad_len = max_len - prompt_ids.size(0)
            
            # 左填充
            if pad_len > 0:
                prompt_ids = F.pad(prompt_ids, (pad_len, 0), value=self.pad_token_id)
                attention_mask = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    torch.ones(f['prompt_ids'].size(0), dtype=torch.long)
                ])
            else:
                attention_mask = torch.ones(prompt_ids.size(0), dtype=torch.long)
            
            prompt_ids_list.append(prompt_ids)
            attention_mask_list.append(attention_mask)
        
        prompt_ids = torch.stack(prompt_ids_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)
        
        # 处理图像：每个样本固定 1 张图，直接 cat 成 [B, C, H, W]
        pixel_values_list = [f['pixel_values'] for f in features]
        # pv 形状为 [1, C, H, W]，cat 后得 [B, C, H, W]
        padded_pixel_values = torch.cat(pixel_values_list, dim=0)
        
        result = {
            'prompt_ids': prompt_ids,
            'attention_mask': attention_mask,
            'pixel_values': padded_pixel_values,
        }
        
        # 如果有 answer 字段，也返回
        if 'answer' in features[0]:
            result['answers'] = [f['answer'] for f in features]
        
        return result


class GRPOTrainerBase:
    """GRPO 训练器基类"""
    
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        optimizer,
        device,
        num_generations: int = 4,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        beta: float = 0.04,
        clip_range: float = 0.2,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.beta = beta
        self.clip_range = clip_range
    
    @torch.no_grad()
    def generate_responses(self, prompt_ids, attention_mask, pixel_values):
        """
        为每个 prompt 生成 num_generations 个响应
        
        Returns:
            all_responses: [batch * num_generations, max_seq_len] 的 token ids
            all_attention_masks: [batch * num_generations, max_seq_len] 的 attention mask
            all_response_texts: 解码后的文本列表
        """
        batch_size = prompt_ids.size(0)
        all_responses = []
        all_attention_masks = []
        all_response_texts = []
        
        self.model.eval()
        
        for _ in range(self.num_generations):
            generated, gen_attention_mask = self._generate_single(prompt_ids, attention_mask, pixel_values)
            all_responses.append(generated)
            all_attention_masks.append(gen_attention_mask)
            
            # 解码响应文本（跳过 prompt 部分）
            for i in range(batch_size):
                prompt_len = (attention_mask[i] == 1).sum().item()
                response_ids = generated[i, prompt_len:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                all_response_texts.append(response_text)
        
        self.model.train()
        
        # 右填充：对齐不同长度的生成序列，便于后续批量计算 log prob
        # 注意：解码时输入 prompt 是左填充的，这里是右填充，所以最终序列左右都可能有 pad
        max_len = max(r.size(1) for r in all_responses)
        padded_responses = []
        padded_attention_masks = []
        for r, m in zip(all_responses, all_attention_masks):
            if r.size(1) < max_len:
                pad_len = max_len - r.size(1)
                r = F.pad(r, (0, pad_len), value=self.tokenizer.pad_token_id)
                m = F.pad(m, (0, pad_len), value=0)  # pad 位置的 attention_mask 为 0
            padded_responses.append(r)
            padded_attention_masks.append(m)
        
        # 拼接：[num_generations 个 (batch, seq_len)] -> (batch * num_generations, seq_len)
        all_responses = torch.cat(padded_responses, dim=0)
        all_attention_masks = torch.cat(padded_attention_masks, dim=0)
        return all_responses, all_attention_masks, all_response_texts
    
    def _generate_single(self, prompt_ids, attention_mask, pixel_values):
        """自回归生成单个响应：每次预测下一个 token，直到 EOS 或达到最大长度"""
        batch_size, prompt_len = prompt_ids.shape
        
        generated = prompt_ids.clone()
        current_attention_mask = attention_mask.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(self.max_new_tokens):
            if finished.all():
                break
            
            with torch.no_grad():
                image_flags = torch.ones(
                    pixel_values.size(0), 1,
                    dtype=torch.long, device=self.device
                )
                outputs = self.model(
                    input_ids=generated,
                    attention_mask=current_attention_mask,
                    pixel_values=pixel_values,
                    image_flags=image_flags,
                )
            
            # 温度采样：温度越高越随机，越低越确定
            next_token_logits = outputs.logits[:, -1, :] / self.temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 拼接新 token 到序列末尾
            generated = torch.cat([generated, next_token], dim=-1)
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones(batch_size, 1, dtype=torch.long, device=self.device)
            ], dim=-1)
            
            # 标记已生成 EOS 的序列
            finished = finished | (next_token.squeeze(-1) == self.tokenizer.eos_token_id)
        
        return generated, current_attention_mask
    
    def compute_log_probs(self, model, input_ids, attention_mask, pixel_values, labels, prompt_mask):
        """
        计算序列的对数概率（只计算 response 部分）
        
        Args:
            input_ids: 完整序列 [batch, seq_len]
            attention_mask: 标记真实 token（1）和 pad（0）
            pixel_values: 图像特征
            labels: 预测目标（通常等于 input_ids）
            prompt_mask: 标记 prompt 部分（1）和 response+pad 部分（0）
        
        Returns:
            avg_log_probs: 每个序列的平均 log prob
            token_log_probs: 每个 token 的 log prob
            response_mask: 最终的 response mask
        """
        image_flags = torch.ones(
            pixel_values.size(0), 1,
            dtype=torch.long, device=pixel_values.device
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_flags=image_flags,
        )
        
        # 错位：用位置 i 的 logits 预测位置 i+1 的 token
        logits = outputs.logits[:, :-1, :]
        labels = labels[:, 1:]
        prompt_mask = prompt_mask[:, 1:]
        
        # 计算每个 token 的 log prob
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 构建 response_mask，只保留 response 的真实 token
        # 
        # 示例：序列 [pad, pad, A, B, C, R1, R2, pad]
        #   - 左边的 pad: 输入 prompt 时的左填充（让 prompt 末尾对齐，便于生成）
        #   - 右边的 pad: 生成后的右填充（让完整序列长度对齐，便于批量计算）
        #
        #   prompt_mask      = [1, 1, 1, 1, 1, 0, 0, 0]  ← prompt部分（含左pad）=1
        #   1 - prompt_mask  = [0, 0, 0, 0, 0, 1, 1, 1]  ← 非prompt部分
        #   attention_mask   = [0, 0, 1, 1, 1, 1, 1, 0]  ← 真实token=1, pad=0
        #   response_mask    = [0, 0, 0, 0, 0, 1, 1, 0]  ← 只有R1,R2保留
        #
        # 效果：prompt 被排除（prompt_mask=1），左右 pad 都被排除（attention_mask=0）
        response_mask = (1 - prompt_mask.float()) * attention_mask[:, 1:].float()
        
        # 计算 response 部分的平均 log prob
        seq_log_probs = (token_log_probs * response_mask).sum(dim=-1)
        seq_lengths = response_mask.sum(dim=-1).clamp(min=1)
        avg_log_probs = seq_log_probs / seq_lengths
        
        return avg_log_probs, token_log_probs, response_mask
