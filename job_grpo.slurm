#!/bin/bash
#SBATCH --job-name=geoqa_grpo
#SBATCH --account=cpsc4770
#SBATCH --partition=education_gpu
#SBATCH --gpus=rtx_5000_ada:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@yale.edu

source ~/miniconda3/bin/activate
conda activate geoqa

cd $HOME/project_cpsc4770/vlm
mkdir -p logs out/geoqa

torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    trainer/train_grpo_geoqa.py \
    --model_path    ~/scratch_cpsc4770/InternVL3-8B \
    --adapter_path  out/geoqa/sft_geoqa_final \
    --train_data    dataset/geoqa/rl_filtered.parquet \
    --save_dir      out/geoqa \
    --lora_rank     16 \
    --epochs        1 \
    --batch_size    2 \
    --num_generations 4 \
    --max_new_tokens  256 \
    --temperature     0.7 \
    --learning_rate   5e-7 \
    --use_wandb       0 \
    --log_interval    10 \
    --save_interval   100
