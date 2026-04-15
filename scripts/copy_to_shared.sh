#!/bin/bash
# 将 GRPO 所需文件从 Bouchet scratch 复制到共享 home (~scratch/grpo_data)
# 在 Bouchet 上运行：bash scripts/copy_to_shared.sh

BOUCHET_SCRATCH=/nfs/roberts/scratch/cpsc4770/cpsc4770_rz377
DEST=~/scratch/grpo_data

mkdir -p "$DEST"
echo "目标路径: $DEST"

echo "[1/3] 复制 InternVL3-8B 模型权重 (~16GB)..."
cp -r "$BOUCHET_SCRATCH/InternVL3-8B" "$DEST/" &
PID1=$!

echo "[2/3] 复制 SFT 权重 sft_geoqa_final (~16GB)..."
cp -r "$BOUCHET_SCRATCH/vlm/out/geoqa/sft_geoqa_final" "$DEST/" &
PID2=$!

echo "[3/3] 复制 GeoQA 数据集..."
cp -r "$BOUCHET_SCRATCH/vlm/dataset/geoqa" "$DEST/" &
PID3=$!

echo "三个复制任务已在后台启动（PID: $PID1, $PID2, $PID3）"
echo "可以退出登录，复制会继续运行。"
echo "完成后检查: ls ~/scratch/grpo_data/"
