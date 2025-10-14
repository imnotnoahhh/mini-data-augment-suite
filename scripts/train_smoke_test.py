#!/usr/bin/env python3
"""Minimal training smoke test to ensure gradients/optimizer execute."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data.splits import load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny training loop to validate pipeline")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "stl10"], help="Dataset name")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", "convnext_tiny"], help="Model backbone")
    parser.add_argument("--seed", type=int, default=0, help="Split seed to load")
    parser.add_argument("--steps", type=int, default=2, help="Training steps to run")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"), help="Where raw datasets reside")
    return parser.parse_args()


def build_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        return torchvision.models.resnet18(weights=None, num_classes=num_classes)
    if name == "resnet50":
        return torchvision.models.resnet50(weights=None, num_classes=num_classes)
    if name == "convnext_tiny":
        return torchvision.models.convnext_tiny(weights=None, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {name}")


def build_dataset(dataset: str, seed: int, data_root: Path) -> Subset:
    dataset = dataset.lower()
    if dataset == "cifar100":
        base = datasets.CIFAR100(root=str(data_root), train=True, download=False, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]))
        num_classes = 100
    elif dataset == "stl10":
        base = datasets.STL10(root=str(data_root), split="train", download=False, transform=transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ]))
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    split = load_split(dataset, seed)
    subset_indices = split.train_indices[: 4 * 16]  # take small subset
    ds = Subset(base, subset_indices)
    return ds, num_classes


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, num_classes = build_dataset(args.dataset, args.seed, args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = build_model(args.model, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    model.train()
    step_iter = itertools.islice(loader, args.steps)
    for step, (images, targets) in enumerate(step_iter, start=1):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"step {step}/{args.steps} loss={loss.item():.4f}")

    print("[train_smoke_test] SUCCESS")


if __name__ == "__main__":
    main()
