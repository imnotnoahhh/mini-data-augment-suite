#!/usr/bin/env bash

# 统一生成 CIFAR-100 / STL-10 / Tiny-ImageNet-200 的 20% 训练 + 10% 验证划分。
# 默认使用项目目录下的 data/raw/ 作为数据缓存，可通过 DATA_ROOT/TINY_ROOT 环境变量覆盖。

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/raw}"
TINY_ROOT="${TINY_ROOT:-$DATA_ROOT/tiny-imagenet-200}"

mkdir -p "$DATA_ROOT"

train_ratio="${TRAIN_RATIO:-0.2}"
val_ratio="${VAL_RATIO:-0.1}"
seeds=(${SEEDS:-0})

run_split () {
  local dataset="$1"
  local seed="$2"
  echo "[build_all_splits] dataset=${dataset} seed=${seed}"
  if [[ "$dataset" == "tiny-imagenet-200" ]]; then
    python3 "$REPO_ROOT/scripts/build_subset_split.py" \
      --dataset "$dataset" \
      --seed "$seed" \
      --train-ratio "$train_ratio" \
      --val-ratio "$val_ratio" \
      --data-root "$DATA_ROOT" \
      --tiny-root "$TINY_ROOT"
  else
    python3 "$REPO_ROOT/scripts/build_subset_split.py" \
      --dataset "$dataset" \
      --seed "$seed" \
      --train-ratio "$train_ratio" \
      --val-ratio "$val_ratio" \
      --data-root "$DATA_ROOT" \
      --download
  fi
}

for seed in "${seeds[@]}"; do
  run_split "cifar100" "$seed"
  run_split "stl10" "$seed"
  run_split "tiny-imagenet-200" "$seed"
done

echo "[build_all_splits] done."
