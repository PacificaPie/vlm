#!/bin/bash
#SBATCH --job-name=rl_filter
#SBATCH --account=lafferty
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=logs/filter_%j.out
#SBATCH --error=logs/filter_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhangruoyang511@gmail.com

module load miniconda/24.3.0
conda activate geoqa

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/scratch/vlm
mkdir -p logs

python scripts/build_rl_filtered.py \
    --model_path ~/scratch/grpo_data/InternVL3-8B \
    --sft_path   ~/scratch/grpo_data/sft_geoqa_final \
    --train_data ~/scratch/grpo_data/geoqa/sft_train.parquet \
    --out        ~/scratch/grpo_data/geoqa/rl_filtered.parquet \
    --k          8 \
    --max_new_tokens 128 \
    --temperature    0.7 \
    --dtype          bfloat16
