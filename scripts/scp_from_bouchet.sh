#!/bin/bash
# 从 Bouchet 把数据拉到 Misha 的共享 home
# 在 Misha 上运行：bash scripts/scp_from_bouchet.sh

BOUCHET="cpsc4770_rz377@bouchet.ycrc.yale.edu"
BOUCHET_SCRATCH="/nfs/roberts/scratch/cpsc4770/cpsc4770_rz377"
DEST=~/scratch/grpo_data

mkdir -p "$DEST"
echo "目标路径: $DEST"

echo "[1/3] 拉取 InternVL3-8B (~16GB)..."
nohup scp -r "$BOUCHET:$BOUCHET_SCRATCH/InternVL3-8B" "$DEST/" \
    > ~/scratch/scp1.log 2>&1 &
echo "  PID: $!"

echo "[2/3] 拉取 sft_geoqa_final (~16GB)..."
nohup scp -r "$BOUCHET:$BOUCHET_SCRATCH/vlm/out/geoqa/sft_geoqa_final" "$DEST/" \
    > ~/scratch/scp2.log 2>&1 &
echo "  PID: $!"

echo "[3/3] 拉取 GeoQA 数据集..."
nohup scp -r "$BOUCHET:$BOUCHET_SCRATCH/vlm/dataset/geoqa" "$DEST/" \
    > ~/scratch/scp3.log 2>&1 &
echo "  PID: $!"

echo ""
echo "全部后台启动。可以退出登录，完成后检查："
echo "  ls ~/scratch/grpo_data/"
echo "  cat ~/scratch/scp1.log"
