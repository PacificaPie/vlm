#!/bin/bash
#SBATCH --job-name=geoqa_sft
#SBATCH --account=cpsc4770
#SBATCH --partition=education_gpu      # 课程账号可用分区（RTX 5000 Ada, 32GB×4）
#SBATCH --gpus=rtx_5000_ada:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00               # 32GB 显存比 96GB 慢，留多一点时间
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@yale.edu

# ---------- 环境 ----------
export PATH="/home/cpsc4770_rz377/.conda/envs/geoqa/bin:$PATH"

# 项目根目录
cd /nfs/roberts/scratch/cpsc4770/cpsc4770_rz377/vlm

mkdir -p logs out/geoqa

# ---------- 下载模型（首次运行时取消注释）----------
# huggingface-cli download OpenGVLab/InternVL3-8B --local-dir ./InternVL3-8B

# ---------- 训练 ----------
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    trainer/train_sft_geoqa.py \
    --model_path    /nfs/roberts/scratch/cpsc4770/cpsc4770_rz377/InternVL3-8B \
    --train_data    dataset/geoqa/sft_train.parquet \
    --save_dir      out/geoqa \
    --lora_rank     16 \
    --warmup_steps  200 \
    --warmup_lr     1e-4 \
    --epochs        3 \
    --batch_size    4 \
    --accumulation_steps 8 \
    --learning_rate 1e-5 \
    --max_length    1024 \
    --dtype         bfloat16 \
    --use_wandb     0 \
    --log_interval  10 \
    --save_interval 200

# 有效 batch size = 8 (per GPU) × 2 (GPU) × 4 (accumulation) = 64 ≈ proposal 的 batch_size=32
