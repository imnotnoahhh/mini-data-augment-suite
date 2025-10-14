#!/usr/bin/env python3
"""Quickly exercise transform factories to catch runtime errors locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision import datasets

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data import transforms as aug_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test data/transforms factories")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "stl10", "tiny-imagenet-200"],
        help="Dataset to draw sample image from.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing raw datasets (default: data/raw).",
    )
    return parser.parse_args()


def load_sample(dataset: str, data_root: Path) -> torch.Tensor:
    dataset = dataset.lower()
    if dataset == "cifar100":
        ds = datasets.CIFAR100(root=str(data_root), train=True, download=False)
        sample, _ = ds[0]
        return sample
    if dataset == "stl10":
        ds = datasets.STL10(root=str(data_root), split="train", download=False)
        sample, _ = ds[0]
        return sample
    if dataset == "tiny-imagenet-200":
        ds = datasets.ImageFolder(str(data_root / "tiny-imagenet-200" / "train"))
        sample, _ = ds[0]
        return sample
    raise ValueError(f"Unsupported dataset: {dataset}")


def main() -> None:
    args = parse_args()
    sample = load_sample(args.dataset, args.data_root)
    print(f"Loaded sample from {args.dataset}: mode={sample.mode if hasattr(sample, 'mode') else 'tensor'}")

    # Single-factor transform (brightness example)
    single_tf = aug_transforms.make_single_factor_transform("brightness", strength=0.2)
    single_out = single_tf(sample)
    print(f"make_single_factor_transform ok -> tensor shape {tuple(single_out.shape)}")

    # Sobol transform with mid-range parameters
    sobol_row = {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.05,
        "crop": 4,
        "flip": 0.5,
        "rotation": 10,
        "scale_delta": 0.1,
        "erase": 0.12,
    }
    sobol_tf = aug_transforms.make_sobol_transform(sobol_row)
    sobol_out = sobol_tf(sample)
    print(f"make_sobol_transform ok -> tensor shape {tuple(sobol_out.shape)}")

    # RSM transform shares implementation with Sobol but test with different fields
    rsm_row = {
        "brightness": 0.15,
        "contrast": 0.25,
        "saturation": 0.1,
        "hue": 0.03,
        "crop": 2,
        "flip": 0.3,
        "rotation": 5,
        "scale_low": 0.9,
        "scale_high": 1.1,
        "erase": 0.08,
    }
    rsm_tf = aug_transforms.make_rsm_transform(rsm_row)
    rsm_out = rsm_tf(sample)
    print(f"make_rsm_transform ok -> tensor shape {tuple(rsm_out.shape)}")

    print("[transform_smoke_test] SUCCESS")


if __name__ == "__main__":
    main()
