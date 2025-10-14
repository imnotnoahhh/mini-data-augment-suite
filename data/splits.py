"""
数据集划分工具。

对应 implementation_plan.md 中的「数据集拆分脚本」与「公平性约束的落地」要求：
- 读取 `artifacts/splits/{dataset}/{dataset}_seed{seed}.json`；
- 若文件缺失/哈希不符需报错，不可隐式重采样；
- `build_subset_split.py` 将调用此模块中的 API。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

SPLIT_ROOT = Path("artifacts") / "splits"


@dataclass
class DatasetSplit:
    train_indices: List[int]
    val_indices: List[int]
    metadata: Dict[str, str]


def load_split(dataset: str, seed: int) -> DatasetSplit:
    """
    读取给定数据集 & seed 的划分文件并返回结构化数据。
    会校验：
      - 文件是否存在；
      - metadata 中记录的哈希值是否与实际匹配；
      - 索引是否已排序。
    """
    dataset = dataset.lower()
    root = _dataset_root(dataset)
    split_path = root / f"{dataset}_seed{seed}.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file {split_path} not found. Run scripts/build_subset_split.py --dataset {dataset} first."
        )

    payload = json.loads(split_path.read_text(encoding="utf-8"))

    metadata_path = root / "metadata.json"
    metadata: Dict[str, Dict] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    seeds_meta = metadata.get("seeds", {})
    seed_key = str(seed)
    if seed_key not in seeds_meta:
        raise ValueError(
            f"Metadata for dataset={dataset} seed={seed} missing. Re-run scripts/build_subset_split.py if needed."
        )

    expected_hash = seeds_meta[seed_key].get("hash")
    train_indices = list(payload["train_indices"])
    val_indices = list(payload["val_indices"])
    actual_hash = _compute_split_hash(train_indices, val_indices)
    if expected_hash != actual_hash:
        raise ValueError(
            f"Split hash mismatch for dataset={dataset} seed={seed}. "
            f"Expected {expected_hash}, got {actual_hash}. Refusing to proceed."
        )

    if train_indices != sorted(train_indices) or val_indices != sorted(val_indices):
        raise ValueError("Split indices must be sorted to ensure determinism.")

    return DatasetSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        metadata=metadata,
    )


def save_split(
    dataset: str,
    seed: int,
    train_indices: List[int],
    val_indices: List[int],
    metadata: Dict[str, Dict],
) -> None:
    """
    写出划分文件。此函数由 scripts/build_subset_split.py 调用，并负责：
    - 保证索引唯一且有序；
    - 计算 split 哈希并写入 metadata.json。
    """
    dataset = dataset.lower()
    root = _dataset_root(dataset)
    root.mkdir(parents=True, exist_ok=True)

    train_indices_sorted = sorted(dict.fromkeys(train_indices))
    val_indices_sorted = sorted(dict.fromkeys(val_indices))

    split_path = root / f"{dataset}_seed{seed}.json"
    split_payload = {
        "train_indices": train_indices_sorted,
        "val_indices": val_indices_sorted,
    }
    split_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata_path = root / "metadata.json"
    existing: Dict[str, Dict] = {"global": {}, "seeds": {}}
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing.setdefault("global", {})
        existing.setdefault("seeds", {})

    global_meta = metadata.get("global", {})
    if isinstance(global_meta, dict):
        existing["global"].update(global_meta)

    seed_meta = metadata.get("seed", {}).copy()
    seed_meta.update(
        {
            "dataset": dataset,
            "seed": seed,
            "hash": _compute_split_hash(train_indices_sorted, val_indices_sorted),
            "num_train": len(train_indices_sorted),
            "num_val": len(val_indices_sorted),
        }
    )
    existing["seeds"][str(seed)] = seed_meta

    metadata_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def _dataset_root(dataset: str) -> Path:
    return SPLIT_ROOT / dataset


def _compute_split_hash(train_indices: List[int], val_indices: List[int]) -> str:
    hasher = hashlib.sha256()
    train_bytes = ",".join(map(str, train_indices)).encode("utf-8")
    val_bytes = ",".join(map(str, val_indices)).encode("utf-8")
    hasher.update(train_bytes)
    hasher.update(b"|")
    hasher.update(val_bytes)
    return hasher.hexdigest()
