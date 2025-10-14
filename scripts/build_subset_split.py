#!/usr/bin/env python3
"""
生成数据子集划分（CIFAR-100/STL-10/Tiny-ImageNet-200）。

实现要点：
- 支持统一的 20% 训练子集 + 10% 验证集分层划分；
- 输出存放于 `artifacts/splits/{dataset}/`，并写入哈希校验信息；
- 供主实验（CIFAR-100 × ResNet-18）与泛化验证（其余数据集/模型）共用。
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from data.splits import save_split

try:
    from torchvision.datasets import CIFAR100, STL10, ImageFolder  # type: ignore
except ImportError as exc:  # pragma: no cover - 运行时需安装 torchvision
    CIFAR100 = None
    STL10 = None
    ImageFolder = None
    _TORCHVISION_IMPORT_ERROR = exc
else:
    _TORCHVISION_IMPORT_ERROR = None

SUPPORTED_DATASETS = ("cifar100", "stl10", "tiny-imagenet-200")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stratified subset splits for multiple datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        default="cifar100",
        help="目标数据集（默认: cifar100）",
    )
    parser.add_argument("--train-ratio", type=float, default=0.2, help="训练子集占比 (默认 0.2)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集占训练子集比例 (默认 0.1)")
    parser.add_argument("--seed", type=int, required=True, help="随机种子（决定抽样）")
    default_data_root = (REPO_ROOT / "data" / "raw")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help=f"数据缓存根目录（CIFAR-100 / STL-10 使用，默认: {default_data_root}）",
    )
    parser.add_argument(
        "--tiny-root",
        type=Path,
        default=None,
        help="Tiny-ImageNet-200 解压路径（默认 data-root/tiny-imagenet-200）",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="若本地不存在数据集则尝试下载（仅对 CIFAR-100 / STL-10 生效）",
    )
    return parser.parse_args()


def gather_metadata() -> Dict[str, str]:
    metadata = {
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    try:
        torchvision_version = subprocess.check_output(
            ["python", "-c", "import torchvision; print(torchvision.__version__)"],
            text=True,
        ).strip()
        metadata["torchvision_version"] = torchvision_version
    except Exception:
        metadata["torchvision_version"] = "unknown"

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        metadata["git_hash"] = git_hash
    except Exception:
        metadata["git_hash"] = "unknown"

    return metadata


def load_labels(
    dataset: str,
    data_root: Path,
    tiny_root: Path | None,
    download: bool,
) -> Tuple[Sequence[int], Dict[str, int]]:
    dataset = dataset.lower()

    if dataset in {"cifar100", "stl10"} and _TORCHVISION_IMPORT_ERROR is not None:
        raise RuntimeError(
            "未检测到 torchvision，无法加载 CIFAR-100/STL-10。请在 conda 环境中安装 torch/torchvision 后再运行。\n"
            f"原始错误：{_TORCHVISION_IMPORT_ERROR}"
        )

    if dataset == "cifar100":
        ds = CIFAR100(root=str(data_root), train=True, download=download)  # type: ignore[arg-type]
        if hasattr(ds, "targets"):
            labels = ds.targets  # type: ignore[assignment]
        elif hasattr(ds, "train_labels"):
            labels = ds.train_labels  # type: ignore[assignment]
        else:
            raise RuntimeError("无法从 CIFAR-100 数据集中读取标签。请检查 torchvision 版本。")
        return labels, {"num_samples": len(labels), "num_classes": 100}

    if dataset == "stl10":
        ds = STL10(root=str(data_root), split="train", download=download)  # type: ignore[arg-type]
        labels = ds.labels  # type: ignore[assignment]
        return labels, {"num_samples": len(labels), "num_classes": 10}

    if dataset == "tiny-imagenet-200":
        if ImageFolder is None:
            raise RuntimeError(
                "未检测到 torchvision.datasets.ImageFolder，无法加载 Tiny-ImageNet-200。"
                f" 原始错误：{_TORCHVISION_IMPORT_ERROR}"
            )
        root = tiny_root or (data_root / "tiny-imagenet-200")
        train_dir = root / "train"
        if not train_dir.exists():
            raise FileNotFoundError(
                f"Tiny-ImageNet-200 训练目录 {train_dir} 不存在。请先从官方链接下载并解压，再运行脚本。"
            )
        ds = ImageFolder(str(train_dir))
        labels = [label for _, label in ds.samples]
        return labels, {"num_samples": len(labels), "num_classes": len(ds.classes)}

    raise ValueError(f"Unsupported dataset: {dataset}")


def main() -> None:
    args = parse_args()
    dataset = args.dataset.lower()

    if not 0 < args.train_ratio <= 1:
        raise ValueError("--train-ratio 必须在 (0, 1] 范围内")
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio 必须在 (0, 1) 范围内")

    data_root = args.data_root
    data_root.mkdir(parents=True, exist_ok=True)

    tiny_root = args.tiny_root
    if tiny_root is not None:
        tiny_root = tiny_root.expanduser()

    labels, stats = load_labels(
        dataset=dataset,
        data_root=data_root,
        tiny_root=tiny_root,
        download=args.download,
    )

    train_indices, val_indices = stratified_split(
        labels=labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    metadata = {
        "global": gather_metadata() | {"dataset": dataset},
        "seed": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "num_train_subset": len(train_indices) + len(val_indices),
            "num_train_full": stats.get("num_samples"),
            "num_classes": stats.get("num_classes"),
        },
    }

    save_split(dataset, args.seed, train_indices, val_indices, metadata)
    print(
        f"[dataset={dataset} seed={args.seed}] "
        f"subset={len(train_indices) + len(val_indices)} "
        f"train={len(train_indices)} val={len(val_indices)} "
        f"metadata写入完成 -> artifacts/splits/{dataset}/"
    )


def stratified_split(
    *,
    labels: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    per_class: Dict[int, List[int]] = collections.defaultdict(list)
    for idx, label in enumerate(labels):
        per_class[int(label)].append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for label, indices in per_class.items():
        if not indices:
            continue
        indices_copy = indices[:]
        rng.shuffle(indices_copy)

        subset_size = max(1, int(round(len(indices_copy) * train_ratio)))
        subset_size = min(subset_size, len(indices_copy))
        subset = indices_copy[:subset_size]

        val_size = max(1, int(round(len(subset) * val_ratio)))
        if val_size >= len(subset):
            val_size = max(1, len(subset) - 1)

        val_indices.extend(sorted(subset[:val_size]))
        train_indices.extend(sorted(subset[val_size:]))

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


if __name__ == "__main__":
    main()
