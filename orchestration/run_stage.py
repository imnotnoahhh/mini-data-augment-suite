"""Stage orchestration entrypoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from data.transforms import make_single_factor_transform
from data.splits import load_split
from engine.trainer import Trainer, TrainerConfig
from utils.preview import DATASET_STATS


DATASET_NUM_CLASSES: Dict[str, int] = {
    "cifar100": 100,
    "stl10": 10,
    "tiny-imagenet-200": 200,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run augmentation stage experiment.")
    parser.add_argument("--stage", required=True, help="Stage name: single_factor, sobol, rsm.")
    parser.add_argument("--config-root", default="configs", help="Path to config directory.")
    parser.add_argument("--config", help="Optional explicit config path; overrides --stage lookup.")
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_")
    return cleaned.lower() or "config"


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        return models.resnet18(weights=None, num_classes=num_classes)
    if name == "resnet50":
        return models.resnet50(weights=None, num_classes=num_classes)
    if name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            last = model.classifier[-1]
            if isinstance(last, nn.Linear):
                model.classifier[-1] = nn.Linear(last.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model name: {name}")


def build_single_factor_loaders(
    dataset: str,
    data_root: Path,
    split_seed: int,
    train_transform,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    dataset = dataset.lower()
    if dataset not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    stats = DATASET_STATS[dataset]
    mean, std = stats["mean"], stats["std"]

    if dataset == "cifar100":
        base_train = datasets.CIFAR100(root=str(data_root), train=True, download=False, transform=train_transform)
        base_val = datasets.CIFAR100(
            root=str(data_root),
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
        )
    else:
        raise ValueError(f"Single factor stage currently only supports CIFAR-100. Got {dataset}.")

    split = load_split(dataset, split_seed)
    train_subset = Subset(base_train, split.train_indices)
    val_subset = Subset(base_val, split.val_indices)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def run_single_factor(config: Dict[str, Any], device: torch.device, config_path: Path) -> None:
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    model_cfg = config["model"]
    logging_cfg = config["logging"]
    single_cfg = config["single_factor"]

    dataset_name = dataset_cfg["name"].lower()
    data_root = Path(dataset_cfg.get("data_root", "data/raw"))
    split_seed = dataset_cfg.get("split_seed", 0)

    epochs = training_cfg["epochs"]
    batch_size = training_cfg["batch_size"]
    num_workers = training_cfg.get("num_workers", 8)
    seeds = training_cfg.get("seeds", [0])
    grad_clip = training_cfg.get("grad_clip")
    amp_enabled = training_cfg.get("amp", True)
    ema_cfg = training_cfg.get("ema", {"enabled": True, "decay": 0.999})

    output_root = Path(logging_cfg.get("output_root", "outputs/single_factor"))
    config_file = Path(single_cfg["config_file"])
    entries = load_json(config_file)
    single_entries = [e for e in entries if e.get("type") == "single_factor"]

    if not single_entries:
        raise ValueError(f"No single_factor entries found in {config_file}")

    num_classes = DATASET_NUM_CLASSES.get(dataset_name)
    if num_classes is None:
        raise ValueError(f"Unknown num_classes for dataset {dataset_name}")

    for entry in single_entries:
        operation = entry.get("operation")
        strength = entry.get("strength")
        label = entry.get("label") or f"{operation}_{strength}"
        label_safe = sanitize_label(label)

        for seed in seeds:
            run_id = f"{label_safe}_seed{seed}"
            combo_id = label

            train_transform = make_single_factor_transform(operation, strength)
            train_loader, val_loader = build_single_factor_loaders(
                dataset_name,
                data_root,
                split_seed,
                train_transform,
                batch_size,
                num_workers,
                device,
            )

            model = build_model(model_cfg["name"], num_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config["optimizer"]["lr"],
                momentum=config["optimizer"].get("momentum", 0.9),
                weight_decay=config["optimizer"].get("weight_decay", 0.0),
            )
            scheduler = None
            if config.get("lr_scheduler", {}).get("name") == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            trainer_cfg = TrainerConfig(
                output_root=output_root,
                stage=config["stage"],
                dataset=dataset_name,
                model_name=model_cfg["name"],
                seed=seed,
                enable_amp=amp_enabled,
                enable_ema=ema_cfg.get("enabled", True),
                ema_decay=ema_cfg.get("decay", 0.999),
                clip_grad_norm=grad_clip,
            )

            trainer = Trainer(trainer_cfg)
            summary = trainer.fit(
                run_id=run_id,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                epochs=epochs,
                num_classes=num_classes,
                device=device,
                combo_id=combo_id,
                config_snapshot={
                    "stage_config_path": str(config_path),
                    "operation": operation,
                    "strength": strength,
                    "label": label,
                    "seed": seed,
                },
            )

            run_dir = output_root / run_id
            (run_dir / "transform.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
            dataset_dump = dict(dataset_cfg)
            dataset_dump["data_root"] = str(data_root)
            logging_dump = dict(logging_cfg)
            logging_dump["output_root"] = str(output_root)
            config_dump = {
                "stage": config["stage"],
                "dataset": dataset_dump,
                "model": model_cfg,
                "training": training_cfg,
                "optimizer": config["optimizer"],
                "lr_scheduler": config.get("lr_scheduler", {}),
                "logging": logging_dump,
                "entry": entry,
                "seed": seed,
                "summary": summary,
            }
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config_dump, sort_keys=False, allow_unicode=True))


def main() -> None:
    args = parse_args()
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(args.config_root) / f"{args.stage}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found")

    config = load_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage = config.get("stage", args.stage)
    if stage != "single_factor":
        raise NotImplementedError("Only single_factor stage is currently implemented.")

    run_single_factor(config, device, config_path)


if __name__ == "__main__":
    main()
