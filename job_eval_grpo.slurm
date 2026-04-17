#!/bin/bash
#SBATCH --job-name=eval_grpo
#SBATCH --account=lafferty
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_grpo_%j.out
#SBATCH --error=logs/eval_grpo_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhangruoyang511@gmail.com

module load miniconda/24.3.0
conda activate geoqa

cd ~/scratch/vlm
mkdir -p logs

python scripts/eval_geoqa.py \
    --model_path  ~/scratch/grpo_data/InternVL3-8B \
    --adapter_path out/geoqa/grpo_geoqa_final \
    --mode        grpo \
    --dtype       bfloat16 \
    --output_file out/geoqa/eval_grpo.json
