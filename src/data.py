"""Data utilities for small-sample image classification experiments.

This module covers:
- Loading dataset and augmentation configuration files.
- Generating deterministic train/val/test manifests and k-shot subsets.
- Constructing PyTorch DataLoader objects that follow the experimental contract.

Design choices follow the project specification and align with modern PyTorch
(>= 2.2) and torchvision (>= 0.17) best practices.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2 as T

from .augment import build_pipeline, load_augment_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CFG = ROOT / "configs" / "datasets.yaml"
DEFAULT_AUG_CFG = ROOT / "configs" / "augment" / "aug_search.yaml"

VAL_SPLIT_SEED = 42


@dataclass
class DataModuleConfig:
    name: str
    registry: str
    root: Path
    train_split: str
    test_split: str
    val_from_train: int
    upsample: bool
    manifest_dir: Path
    metadata: Mapping[str, object]


def load_yaml(path: Path) -> Mapping[str, object]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Mapping[str, object], path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def ensure_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_dataset_configs(cfg_path: Path = DEFAULT_DATA_CFG) -> Tuple[Mapping[str, object], Dict[str, DataModuleConfig]]:
    cfg = load_yaml(cfg_path)
    datasets_cfg: Dict[str, DataModuleConfig] = {}
    for name, entry in cfg["datasets"].items():
        datasets_cfg[name] = DataModuleConfig(
            name=name,
            registry=entry["registry"],
            root=Path(entry["root"]),
            train_split=entry["train_split"],
            test_split=entry["test_split"],
            val_from_train=int(entry.get("val_from_train", 0)),
            upsample=bool(entry.get("upsample", False)),
            manifest_dir=ROOT / entry["manifest_dir"],
            metadata=entry.get("metadata", {}),
        )
    return cfg, datasets_cfg


def _load_base_dataset(dm_cfg: DataModuleConfig, split: str, transform=None, download: bool = True) -> Dataset:
    registry = dm_cfg.registry.lower()
    root = dm_cfg.root
    if registry == "torchvision":
        if dm_cfg.name == "cifar100":
            is_train = split in {"train", "train_pool", "val"}
            return datasets.CIFAR100(root=str(root), train=is_train, download=download, transform=transform)
        raise ValueError(f"Unsupported torchvision dataset: {dm_cfg.name}")
    if registry == "tiny_imagenet":
        split_root = root / split
        return datasets.ImageFolder(str(split_root), transform=transform)
    if registry == "imagenet":
        split_root = root / split
        return datasets.ImageFolder(str(split_root), transform=transform)
    raise ValueError(f"Unknown registry type: {registry}")


def _extract_labels(dataset: Dataset) -> List[int]:
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if isinstance(targets, list):
            return [int(t) for t in targets]
        return [int(t) for t in targets.tolist()]  # type: ignore[call-arg]
    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]
    raise AttributeError("Unable to extract labels from dataset; expected 'targets' or 'samples'.")


def _extract_paths(dataset: Dataset, dataset_name: str) -> List[str]:
    if hasattr(dataset, "samples"):
        return [str(path) for path, _ in dataset.samples]
    if hasattr(dataset, "data"):
        # CIFAR-style in-memory arrays; fabricate stable identifiers.
        return [f"{dataset_name}/idx_{idx:05d}.png" for idx in range(len(dataset))]
    return [f"{dataset_name}/idx_{idx:05d}" for idx in range(len(dataset))]


def _gather_class_indices(labels: Sequence[int]) -> Dict[int, List[int]]:
    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        buckets[int(label)].append(idx)
    return buckets


def _split_val_from_train(indices_by_class: Mapping[int, List[int]], per_class: int, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    val_indices: List[int] = []
    train_pool_indices: List[int] = []
    for cls, indices in indices_by_class.items():
        indices = list(indices)
        rng.shuffle(indices)
        take = min(per_class, len(indices))
        val_indices.extend(indices[:take])
        train_pool_indices.extend(indices[take:])
    val_indices.sort()
    train_pool_indices.sort()
    return val_indices, train_pool_indices


def _write_kshot_csv(
    manifest_dir: Path,
    dataset_name: str,
    k: int,
    seed: int,
    subset_indices: Sequence[int],
    labels: Sequence[int],
    paths: Sequence[str],
) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx in subset_indices:
        records.append({
            "dataset": dataset_name,
            "index": int(idx),
            "class_idx": int(labels[idx]),
            "filepath": paths[idx],
        })
    df = pd.DataFrame(records)
    out_path = manifest_dir / f"kshot_{k}_seed{seed}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _write_index_jsonl(manifest_dir: Path, name: str, indices: Sequence[int], labels: Sequence[int], paths: Sequence[str]) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifest_dir / f"{name}_index.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for idx in indices:
            record = {
                "index": int(idx),
                "class_idx": int(labels[idx]) if labels is not None else None,
                "filepath": paths[idx],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def make_manifests(cfg_path: Path = DEFAULT_DATA_CFG, augment_cfg: Path = DEFAULT_AUG_CFG) -> None:
    cfg, datasets_cfg = _resolve_dataset_configs(cfg_path)
    base = cfg["base"]
    k_values = sorted(set(base["kshot"]))
    explore_seeds = list(base.get("explore_seeds", []))
    confirm_seeds = list(base.get("confirm_seeds", []))
    all_kshot_seeds = sorted(set(explore_seeds + confirm_seeds))

    for name, dm_cfg in datasets_cfg.items():
        manifest_dir = dm_cfg.manifest_dir
        manifest_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = _load_base_dataset(dm_cfg, dm_cfg.train_split, transform=None)
        test_dataset = _load_base_dataset(dm_cfg, dm_cfg.test_split, transform=None)

        train_labels = _extract_labels(train_dataset)
        test_labels = _extract_labels(test_dataset)
        train_paths = _extract_paths(train_dataset, name)
        test_paths = _extract_paths(test_dataset, name)

        class_indices = _gather_class_indices(train_labels)
        val_indices, train_pool_indices = _split_val_from_train(class_indices, dm_cfg.val_from_train, VAL_SPLIT_SEED)
        test_indices = list(range(len(test_dataset)))

        splits = {
            "splits": {
                "train_pool": train_pool_indices,
                "val": val_indices,
                "test": test_indices,
            },
            "metadata": {
                "num_classes": dm_cfg.metadata.get("num_classes"),
                "class_to_idx": getattr(train_dataset, "class_to_idx", None),
                "val_per_class": dm_cfg.val_from_train,
                "upsample": dm_cfg.upsample,
                "train_split": dm_cfg.train_split,
                "test_split": dm_cfg.test_split,
            },
        }
        save_yaml(splits, manifest_dir / "splits.yaml")

        _write_index_jsonl(manifest_dir, "train_pool", train_pool_indices, train_labels, train_paths)
        _write_index_jsonl(manifest_dir, "val", val_indices, train_labels, train_paths)
        _write_index_jsonl(manifest_dir, "test", test_indices, test_labels, test_paths)

        for k in k_values:
            for seed in all_kshot_seeds:
                rng = np.random.default_rng(seed)
                subset_indices = []
                for cls, indices in class_indices.items():
                    available = [idx for idx in indices if idx in train_pool_indices]
                    if len(available) < k:
                        raise RuntimeError(f"Dataset {name} class {cls} has < {k} samples after val split.")
                    selected = rng.choice(available, size=k, replace=False)
                    subset_indices.extend(int(i) for i in selected)
                subset_indices.sort()
                _write_kshot_csv(manifest_dir, name, k, seed, subset_indices, train_labels, train_paths)


def _load_split_indices(manifest_dir: Path, split: str) -> List[int]:
    data = load_yaml(manifest_dir / "splits.yaml")
    return list(map(int, data["splits"][split]))


def _load_kshot_indices(manifest_dir: Path, k: int, seed: int) -> List[int]:
    csv_path = manifest_dir / f"kshot_{k}_seed{seed}.csv"
    df = pd.read_csv(csv_path)
    return df["index"].astype(int).tolist()


def _upsample_if_needed(dm_cfg: DataModuleConfig, dataset: Dataset, base_cfg: Mapping[str, object]) -> Dataset:
    # Spatial upsampling is handled on-the-fly in torchvision transforms via RandomResizedCrop/Resize.
    return dataset


def build_dataloaders(
    dataset_name: str,
    phase: str,
    batch_size: int,
    kshot: Optional[int],
    seed: int,
    augment_config_path: Path = DEFAULT_AUG_CFG,
    data_config_path: Path = DEFAULT_DATA_CFG,
    shuffle_train: bool = True,
    num_workers: int = 10,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 4,
    combo: Optional[Sequence[str]] = None,
) -> Dict[str, DataLoader]:
    """Construct dataloaders for the requested dataset."""
    cfg, datasets_cfg = _resolve_dataset_configs(data_config_path)
    if dataset_name not in datasets_cfg:
        raise KeyError(f"Dataset {dataset_name} is not defined in {data_config_path}")
    dm_cfg = datasets_cfg[dataset_name]
    base_cfg = cfg["base"]
    manifest_dir = dm_cfg.manifest_dir

    augment_cfg = load_augment_config(augment_config_path)

    train_transform = build_pipeline(augment_cfg, stage="train", combo=combo)
    eval_transform = build_pipeline(augment_cfg, stage="eval")

    train_dataset = _load_base_dataset(dm_cfg, dm_cfg.train_split, transform=train_transform)
    train_dataset = _upsample_if_needed(dm_cfg, train_dataset, base_cfg)
    val_dataset = _load_base_dataset(dm_cfg, dm_cfg.train_split, transform=eval_transform)
    test_dataset = _load_base_dataset(dm_cfg, dm_cfg.test_split, transform=eval_transform)

    train_pool_indices = _load_split_indices(manifest_dir, "train_pool")
    val_indices = _load_split_indices(manifest_dir, "val")
    test_indices = _load_split_indices(manifest_dir, "test")

    if kshot is not None:
        subset_indices = _load_kshot_indices(manifest_dir, kshot, seed)
        train_subset = Subset(train_dataset, subset_indices)
    else:
        train_subset = Subset(train_dataset, train_pool_indices)

    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    eval_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        train_kwargs["persistent_workers"] = persistent_workers
        eval_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            train_kwargs["prefetch_factor"] = prefetch_factor
            eval_kwargs["prefetch_factor"] = prefetch_factor
    else:
        train_kwargs["persistent_workers"] = False
        eval_kwargs["persistent_workers"] = False

    train_loader = DataLoader(train_subset, **train_kwargs)
    val_loader = DataLoader(val_subset, **eval_kwargs)
    test_loader = DataLoader(test_subset, **eval_kwargs)

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset utilities CLI")
    parser.add_argument("--make-manifests", action="store_true", help="Generate manifests and k-shot splits")
    parser.add_argument("--datasets", type=Path, default=DEFAULT_DATA_CFG, help="Path to datasets.yaml")
    parser.add_argument("--augment", type=Path, default=DEFAULT_AUG_CFG, help="Path to augmentation config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.make_manifests:
        make_manifests(args.datasets, args.augment)
        print("Manifests generated successfully.")
    else:
        raise SystemExit("No action specified. Use --make-manifests.")


if __name__ == "__main__":
    main()
