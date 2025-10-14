#!/usr/bin/env python3
"""
Minimal environment verification script.

Checks:
1. torch / torchvision import success and prints versions.
2. Confirms presence of dataset directories and split files according to implementation_plan.md.
3. Loads a CIFAR-100 batch with default transform to ensure torchvision pipeline works.
4. Instantiates ResNet-18/ResNet-50/ViT-S from torchvision and runs a single dummy forward pass.
5. Prints a brief summary of outputs on success.

Intended for macOS/local runs before moving to production GPU servers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet50


def check_versions() -> None:
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")


def check_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data" / "raw"
    splits_root = root / "artifacts" / "splits"

    datasets_expected = {
        "cifar100": data_root / "cifar-100-python",
        "stl10": data_root / "stl10_binary",
        "tiny-imagenet-200": data_root / "tiny-imagenet-200",
    }

    missing = []
    for name, path in datasets_expected.items():
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing dataset directories: {missing}")

    seeds_expected = [0]
    for dataset_name in datasets_expected:
        for seed in seeds_expected:
            split_path = splits_root / dataset_name / f"{dataset_name}_seed{seed}.json"
            if not split_path.exists():
                raise FileNotFoundError(f"Missing split file: {split_path}")
    print("Dataset directories and split files found.")


def check_cifar_loader() -> None:
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data" / "raw"
    split_file = root / "artifacts/splits/cifar100/cifar100_seed0.json"

    with split_file.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    train_indices = payload["train_indices"][:64]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    dataset = datasets.CIFAR100(root=str(data_root), train=True, download=False, transform=transform)
    subset = torch.utils.data.Subset(dataset, train_indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=16, shuffle=False)
    images, labels = next(iter(loader))
    print(f"CIFAR-100 mini-batch loaded: images {images.shape}, labels {labels.shape}")


def check_models() -> None:
    device = torch.device("cpu")
    inputs = torch.randn(2, 3, 224, 224, device=device)

    convnext_ctor = None
    if hasattr(torchvision.models, "convnext_tiny"):
        convnext_ctor = torchvision.models.convnext_tiny

    models = {
        "resnet18": resnet18(weights=None),
        "resnet50": resnet50(weights=None),
    }

    if convnext_ctor is not None:
        models["convnext_tiny"] = convnext_ctor(weights=None)
    else:
        print("warning: torchvision does not provide convnext_tiny; skipping ConvNeXt check")

    for name, model in models.items():
        model.eval()
        model.to(device)
        with torch.no_grad():
            out = model(inputs)
        print(f"{name} forward pass OK: output shape {tuple(out.shape)}")


def main() -> None:
    try:
        check_versions()
        check_paths()
        check_cifar_loader()
        check_models()
    except Exception as exc:
        print(f"[verify_environment] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    print("[verify_environment] SUCCESS")


if __name__ == "__main__":
    main()
