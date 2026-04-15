#!/bin/bash
#SBATCH --job-name=geoqa_eval
#SBATCH --account=cpsc4770
#SBATCH --partition=education_gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/scratch/cpsc4770/cpsc4770_rz377/eval_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhangruoyang511@gmail.com

export PATH="/home/cpsc4770_rz377/.conda/envs/geoqa/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /nfs/roberts/scratch/cpsc4770/cpsc4770_rz377/vlm

echo "=== Baseline ==="
python scripts/eval_geoqa.py \
    --model_path /nfs/roberts/scratch/cpsc4770/cpsc4770_rz377/InternVL3-8B \
    --mode baseline \
    --output_file out/geoqa/eval_baseline.json

echo "=== SFT ==="
python scripts/eval_geoqa.py \
    --model_path /nfs/roberts/scratch/cpsc4770/cpsc4770_rz377/InternVL3-8B \
    --adapter_path out/geoqa/sft_geoqa_final \
    --mode sft \
    --output_file out/geoqa/eval_sft.json
