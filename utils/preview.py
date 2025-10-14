"""Utility functions for generating augmentation preview grids."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from data.splits import load_split

DATASET_STATS: Dict[str, Dict[str, Sequence[float]]] = {
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "split": "train",
    },
    "stl10": {
        "mean": (0.4467, 0.4398, 0.4066),
        "std": (0.2603, 0.2566, 0.2713),
        "split": "train",
    },
    "tiny-imagenet-200": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "split": "train",
    },
}


def _load_base_dataset(dataset: str, data_root: Path) -> datasets.VisionDataset:
    dataset = dataset.lower()
    if dataset == "cifar100":
        return datasets.CIFAR100(root=str(data_root), train=True, download=False, transform=None)
    if dataset == "stl10":
        return datasets.STL10(root=str(data_root), split="train", download=False, transform=None)
    if dataset == "tiny-imagenet-200":
        train_dir = data_root / "tiny-imagenet-200" / "train"
        return datasets.ImageFolder(str(train_dir))
    raise ValueError(f"Unsupported dataset: {dataset}")


def _denormalize(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    device = tensor.device
    mean_t = torch.tensor(mean, device=device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=device).view(-1, 1, 1)
    return torch.clamp(tensor * std_t + mean_t, 0.0, 1.0)


def _format_label(label: str) -> str:
    label = label.strip().lower().replace(" ", "_")
    label = re.sub(r"[^a-z0-9_\-]", "", label)
    return label


def save_transform_preview(
    *,
    dataset: str,
    stage: str,
    config_label: str,
    transform: Callable,
    num_images: int,
    data_root: Path = Path("data/raw"),
    split_seed: int = 0,
    output_root: Path = Path("artifacts/previews"),
    sample_seed: int = 0,
    target_size: int = 96,
    save_individual: bool = False,
) -> Path:
    """Generate before/after preview grid for a given transform.

    The resulting image has two stacked rows: the top row shows the original
    samples, the bottom row shows the transform outputs (after denormalisation).
    """

    dataset_key = dataset.lower()
    if dataset_key not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    stats = DATASET_STATS[dataset_key]

    base_dataset = _load_base_dataset(dataset_key, data_root)
    split = load_split(dataset_key, split_seed)
    indices = torch.tensor(split.train_indices)
    if indices.numel() < num_images:
        raise ValueError(f"Not enough samples ({indices.numel()}) to draw {num_images} images")

    generator = torch.Generator()
    generator.manual_seed(sample_seed)
    perm = torch.randperm(indices.numel(), generator=generator)
    selected_indices = indices[perm[:num_images]].tolist()

    to_tensor = transforms.ToTensor()
    orig_tensors: List[torch.Tensor] = []
    aug_tensors: List[torch.Tensor] = []

    for idx in selected_indices:
        pil_img, _ = base_dataset[idx]
        orig_tensors.append(torch.clamp(to_tensor(pil_img), 0.0, 1.0))
        aug_tensor = transform(pil_img)
        if not isinstance(aug_tensor, torch.Tensor):
            raise TypeError("Transform must return torch.Tensor")
        aug_tensor = aug_tensor.detach().cpu()
        if aug_tensor.min() < 0.0 or aug_tensor.max() > 1.0:
            aug_tensor = _denormalize(aug_tensor, stats["mean"], stats["std"])
        else:
            aug_tensor = torch.clamp(aug_tensor, 0.0, 1.0)
        aug_tensors.append(aug_tensor)

    orig_stack = torch.stack(orig_tensors)
    aug_stack = torch.stack(aug_tensors)

    # resize to target resolution (C x target_size x target_size)
    orig_stack = F.interpolate(orig_stack, size=(target_size, target_size), mode="nearest")
    aug_stack = F.interpolate(aug_stack, size=(target_size, target_size), mode="nearest")

    stage_safe = _format_label(stage)
    label_safe = _format_label(config_label)
    base_dir = output_root / dataset_key / stage_safe
    output_dir = base_dir / label_safe if save_individual else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_individual:
        for idx, (orig_img, aug_img) in enumerate(zip(orig_stack, aug_stack), start=1):
            suffix = f"_sample{idx:02d}"
            base_path = output_dir / f"{label_safe}{suffix}_base.png"
            aug_path = output_dir / f"{label_safe}{suffix}_aug.png"
            save_image(orig_img, str(base_path))
            save_image(aug_img, str(aug_path))
        return output_dir

    orig_grid = make_grid(orig_stack, nrow=num_images, padding=2)
    aug_grid = make_grid(aug_stack, nrow=num_images, padding=2)
    combined = torch.cat([orig_grid, aug_grid], dim=1)
    output_path = output_dir / f"{label_safe}.png"
    save_image(combined, str(output_path))
    return output_path
