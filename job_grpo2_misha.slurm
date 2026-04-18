#!/bin/bash
#SBATCH --job-name=geoqa_grpo2
#SBATCH --account=lafferty
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/grpo2_%j.out
#SBATCH --error=logs/grpo2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhangruoyang511@gmail.com
#SBATCH --requeue

module load miniconda/24.3.0
conda activate geoqa

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/scratch/vlm
mkdir -p logs out/geoqa2

torchrun \
    --nproc_per_node=1 \
    --master_port=29502 \
    trainer/train_grpo_geoqa.py \
    --model_path      ~/scratch/grpo_data/InternVL3-8B \
    --adapter_path    ~/scratch/grpo_data/sft_geoqa_final \
    --train_data      ~/scratch/grpo_data/geoqa/rl_filtered.parquet \
    --save_dir        out/geoqa2 \
    --lora_rank       16 \
    --max_length      512 \
    --dtype           bfloat16 \
    --num_generations 4 \
    --max_new_tokens  128 \
    --temperature     0.7 \
    --beta            0.04 \
    --clip_range      0.2 \
    --epochs          2 \
    --batch_size      1 \
    --learning_rate   5e-7 \
    --log_interval    10 \
    --save_interval   200 \
    --use_wandb       0 \
    --seed            42
