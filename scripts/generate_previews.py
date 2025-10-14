#!/usr/bin/env python3
"""Generate before/after augmentation preview images from a config file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from torchvision import transforms as tv_transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data import transforms as aug_transforms
from utils.preview import save_transform_preview


def build_transform(cfg: Dict[str, Any]):
    cfg_type = cfg.get("type", "single_factor").lower()
    if cfg_type == "single_factor":
        return aug_transforms.make_single_factor_transform(cfg["operation"], cfg["strength"])
    if cfg_type == "sobol":
        return aug_transforms.make_sobol_transform(cfg["params"])
    if cfg_type == "rsm":
        return aug_transforms.make_rsm_transform(cfg["params"])
    if cfg_type == "identity":
        return tv_transforms.Compose([tv_transforms.ToTensor()])
    raise ValueError(f"Unsupported config type: {cfg_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate augmentation preview grids")
    parser.add_argument("--dataset", required=True, help="Dataset name (cifar100/stl10/tiny-imagenet-200)")
    parser.add_argument("--stage", required=True, help="Stage name (single_factor/sobol/rsm/final etc)")
    parser.add_argument("--config-file", required=True, type=Path, help="JSON file describing transform configs")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/previews"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=None, help="Override number of samples per config")
    parser.add_argument("--target-size", type=int, default=96, help="Resize previews so each tile is target_size x target_size (default: 96)")
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save each sample as separate baseline/augmented images instead of a combined grid",
    )
    args = parser.parse_args()

    dataset_key = args.dataset.lower()
    default_num = 8 if dataset_key == "cifar100" else 2
    num_images = args.num_images or default_num

    configs: List[Dict[str, Any]] = json.loads(args.config_file.read_text())
    for cfg in configs:
        label = cfg.get("label") or cfg.get("name") or cfg["type"]
        transform = build_transform(cfg)
        path = save_transform_preview(
            dataset=dataset_key,
            stage=args.stage,
            config_label=label,
            transform=transform,
            num_images=num_images,
            data_root=args.data_root,
            split_seed=args.split_seed,
            output_root=args.output_root,
            sample_seed=args.sample_seed,
            target_size=args.target_size,
            save_individual=args.save_individual,
        )
        print(f"Generated preview: {path}")


if __name__ == "__main__":
    main()
