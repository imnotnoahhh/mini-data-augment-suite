"""Training entry point for the small-data augmentation experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .augment import load_augment_config
from .data import build_dataloaders, load_yaml as load_data_yaml
from .models import ModelBundle, create_model, load_model_config
from .utils.logging import RunLogger, create_run_logger, hash_config, timestamp

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainRunConfig:
    dataset: str
    architecture: str
    phase: str
    kshot: int
    seed: int
    combo_tokens: Optional[Sequence[str]]
    epochs: Optional[int]
    batch_size: Optional[int]
    device: str
    output_suffix: Optional[str]


def load_yaml(path: Path) -> Mapping[str, object]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return None


def configure_hardware(cfg: Mapping[str, object]) -> None:
    backend = cfg.get("device", {}).get("backend", "cuda")
    torch.set_float32_matmul_precision(cfg.get("device", {}).get("matmul_precision", "medium"))
    torch.backends.cudnn.benchmark = bool(cfg.get("device", {}).get("benchmark", True))
    torch.backends.cudnn.deterministic = bool(cfg.get("device", {}).get("deterministic", False))
    if backend == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("device", {}).get("allow_tf32", True))


def build_run_id(config: TrainRunConfig, combo_label: str) -> str:
    parts = [
        config.phase,
        config.dataset,
        config.architecture,
        f"k{config.kshot}",
        f"seed{config.seed}",
        combo_label,
        timestamp().replace(":", "").replace("-", ""),
    ]
    if config.output_suffix:
        parts.append(config.output_suffix)
    return "-".join(parts)


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1)
        pred = pred.t()
        correct = pred.eq(target.unsqueeze(0))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k)
        return res


def train_one_epoch(
    bundle: ModelBundle,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    grad_clip: float,
    accumulation_steps: int,
    scaler: torch.cuda.amp.GradScaler,
    logger: RunLogger,
    epoch: int,
) -> Mapping[str, float]:
    model.train()
    optimizer = bundle.optimizer

    running_loss = 0.0
    correct1 = 0.0
    correct5 = 0.0
    total = 0

    optimizer.zero_grad(set_to_none=True)
    num_steps = len(train_loader)

    for step, (images, targets) in enumerate(train_loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=bundle.amp_dtype, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
            loss = loss / accumulation_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        do_update = (step % accumulation_steps == 0) or (step == num_steps)
        if do_update:
            if grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running_loss += loss.item() * accumulation_steps
        top1, top5 = accuracy(logits.detach(), targets, topk=(1, 5))
        correct1 += top1.item()
        correct5 += top5.item()
        total += targets.size(0)

    epoch_loss = running_loss / len(train_loader)
    metrics = {
        "loss": epoch_loss,
        "top1": correct1 / total * 100.0,
        "top5": correct5 / total * 100.0,
    }
    logger.log_metrics(epoch=epoch, split="train", metrics=metrics)
    return metrics


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
    logger: Optional[RunLogger] = None,
    epoch: Optional[int] = None,
    split: str = "val",
) -> Mapping[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct1 = 0.0
    correct5 = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            top1, top5 = accuracy(logits, targets, topk=(1, 5))
            correct1 += top1.item()
            correct5 += top5.item()
            total += targets.size(0)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = {
        "loss": total_loss / total,
        "top1": correct1 / total * 100.0,
        "top5": correct5 / total * 100.0,
    }
    if len(set(y_true)) > 1:
        metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro") * 100.0
    if logger is not None:
        logger.log_metrics(epoch=epoch or 0, split=split, metrics=metrics)
    return metrics


def save_checkpoint(path: Path, state: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def append_result_csv(path: Path, record: Mapping[str, object]) -> None:
    df_new = pd.DataFrame([record])
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_training(
    run_config: TrainRunConfig,
    train_cfg: Mapping[str, object],
    model_cfg_path: Path,
    dataset_cfg_path: Path,
    augment_cfg_path: Path,
    hardware_cfg_path: Path,
) -> Mapping[str, object]:
    set_global_seed(run_config.seed)
    hardware_cfg = load_yaml(hardware_cfg_path)
    configure_hardware(hardware_cfg)

    device = torch.device(run_config.device if torch.cuda.is_available() or run_config.device == "cpu" else "cpu")

    augment_cfg = load_augment_config(augment_cfg_path)
    combo_label = "baseline"
    combo_tokens = None
    if run_config.combo_tokens:
        combo_tokens = list(run_config.combo_tokens)
        combo_label = "+".join(token.replace(":", "=") for token in combo_tokens)
    elif "combos" in augment_cfg and run_config.output_suffix:
        combo_entry = augment_cfg["combos"].get(run_config.output_suffix)
        if combo_entry:
            combo_tokens = combo_entry
            combo_label = run_config.output_suffix

    data_cfg = load_data_yaml(dataset_cfg_path)
    dataset_entry = data_cfg['datasets'][run_config.dataset]
    num_classes = int(dataset_entry['metadata']['num_classes'])

    model_cfg = load_model_config(model_cfg_path)
    arch_cfg = model_cfg["architectures"][run_config.architecture]
    batch_rec = arch_cfg.get("batch_size", {})
    target_batch = run_config.batch_size or batch_rec.get(run_config.phase)
    if target_batch is None:
        raise ValueError("Batch size must be provided either via config or CLI")

    optimizer_cfg = train_cfg["optimizer"]

    hw_train_cfg = train_cfg.get("hardware", {})
    compile_cfg = hw_train_cfg.get("compile", {})
    compile_enabled = compile_cfg.get("enabled", True)
    compile_mode = compile_cfg.get("mode", hw_train_cfg.get("compile_mode", "reduce-overhead"))
    channels_last_flag = hw_train_cfg.get("channels_last", True)

    bundle = create_model(
        architecture=run_config.architecture,
        num_classes=num_classes,
        phase=run_config.phase,
        global_batch_size=target_batch,
        optimizer_cfg=optimizer_cfg,
        model_config_path=model_cfg_path,
        channels_last=channels_last_flag,
        compile_model=compile_enabled,
        compile_mode=compile_mode,
        device=device,
    )

    label_smoothing = model_cfg["common"].get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    outputs_cfg = train_cfg["experiment"]["outputs"]
    checkpoints_root = ROOT / outputs_cfg["checkpoints"]
    logs_root = ROOT / outputs_cfg["logs"]

    run_id = build_run_id(run_config, combo_label)

    metadata = {
        "dataset": run_config.dataset,
        "architecture": run_config.architecture,
        "phase": run_config.phase,
        "kshot": run_config.kshot,
        "seed": run_config.seed,
        "combo": combo_label,
        "batch_size": target_batch,
        "device": str(device),
        "config_hash": hash_config({
            "train": train_cfg,
            "model": arch_cfg,
            "dataset": run_config.dataset,
            "combo": combo_tokens,
        }),
        "num_parameters": sum(p.numel() for p in bundle.model.parameters()),
        "git_commit": get_git_commit(),
    }
    run_logger = create_run_logger(logs_root, run_id, metadata)

    dl_cfg = hw_train_cfg.get("dataloader", {})
    num_workers = dl_cfg.get("num_workers", 10)
    persistent_workers = dl_cfg.get("persistent_workers", True) and num_workers > 0
    prefetch_factor = dl_cfg.get("prefetch_factor", 4) if num_workers > 0 else None

    dataloaders = build_dataloaders(
        dataset_name=run_config.dataset,
        phase=run_config.phase,
        batch_size=target_batch,
        kshot=run_config.kshot,
        seed=run_config.seed,
        augment_config_path=augment_cfg_path,
        data_config_path=dataset_cfg_path,
        shuffle_train=True,
        num_workers=num_workers,
        pin_memory=dl_cfg.get("pin_memory", True),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        combo=combo_tokens,
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    total_epochs_cfg = train_cfg["scheduler_defaults"]["max_epochs"][run_config.phase]
    total_epochs = run_config.epochs or min(total_epochs_cfg, bundle.max_epochs)

    scheduler = bundle.scheduler_factory.factory(
        bundle.optimizer,
        epochs=total_epochs,
        steps_per_epoch=len(train_loader),
    )

    grad_clip = model_cfg["common"].get("grad_clip_norm", 1.0)
    accumulation_steps = int(train_cfg["scheduler_defaults"].get("gradient_accumulation", 1))

    best_val_top1 = -float("inf")
    best_state = None
    best_epoch = 0

    scaler = torch.cuda.amp.GradScaler(enabled=bundle.amp_dtype == torch.float16 and device.type == "cuda")

    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()
        model_metrics = train_one_epoch(
            bundle,
            bundle.model,
            train_loader,
            criterion,
            device,
            scheduler,
            grad_clip,
            accumulation_steps,
            scaler,
            run_logger,
            epoch,
        )
        val_metrics = evaluate(bundle.model, val_loader, criterion, device, bundle.amp_dtype, run_logger, epoch=epoch, split="val")

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            best_state = deepcopy(bundle.model.state_dict())
            best_epoch = epoch
            save_checkpoint(
                checkpoints_root / run_id / "best_val.pt",
                {
                    "model": best_state,
                    "optimizer": bundle.optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                },
            )

        save_checkpoint(
            checkpoints_root / run_id / "last.pt",
            {
                "model": bundle.model.state_dict(),
                "optimizer": bundle.optimizer.state_dict(),
                "epoch": epoch,
                "metrics": val_metrics,
            },
        )

        run_logger.write({
            "event": "epoch_summary",
            "epoch": epoch,
            "duration_sec": time.time() - epoch_start,
            "train": model_metrics,
            "val": val_metrics,
        })

    if best_state is not None:
        bundle.model.load_state_dict(best_state)

    test_metrics = evaluate(bundle.model, test_loader, criterion, device, bundle.amp_dtype, run_logger, epoch=total_epochs, split="test")

    elapsed = time.time() - start_time
    summary = {
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "test_macro_f1": test_metrics.get("macro_f1"),
        "wall_time_sec": elapsed,
    }
    run_logger.finalize(summary)

    # Append to confirm table if confirmation phase
    if run_config.phase == "confirm":
        table_path = ROOT / "reports" / "tables" / "confirm_10x.csv"
        record = {
            "dataset": run_config.dataset,
            "kshot": run_config.kshot,
            "model": run_config.architecture,
            "combo_id": combo_label,
            "top1_mean": test_metrics["top1"],
            "top1_std": 0.0,
            "top5_mean": test_metrics["top5"],
            "macro_f1_mean": test_metrics.get("macro_f1", 0.0),
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
        append_result_csv(table_path, record)

    summary.update({"run_id": run_id})
    summary.update(metadata)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models for small-data augmentation study")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--phase", choices=["explore", "confirm"], default="explore")
    parser.add_argument("--kshot", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--combo", nargs="*", help="Augmentation combo tokens, e.g. hflip:p=0.5")
    parser.add_argument("--combo-id", help="Combo key defined in configs/augment/aug_search.yaml")
    parser.add_argument("--epochs", type=int, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--suffix", help="Optional suffix for run id")
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs" / "train.yaml")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--dataset-config", type=Path, default=ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--augment-config", type=Path, default=ROOT / "configs" / "augment" / "aug_search.yaml")
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "configs" / "hardware.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml(args.train_config)

    combo_tokens = None
    if args.combo:
        combo_tokens = args.combo
    elif args.combo_id:
        aug_cfg = load_augment_config(args.augment_config)
        combo_tokens = aug_cfg.get("combos", {}).get(args.combo_id)
        if combo_tokens is None:
            raise ValueError(f"Combo id '{args.combo_id}' not found in augmentation config")

    run_config = TrainRunConfig(
        dataset=args.dataset,
        architecture=args.architecture,
        phase=args.phase,
        kshot=args.kshot,
        seed=args.seed,
        combo_tokens=combo_tokens,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_suffix=args.suffix or args.combo_id,
    )

    summary = run_training(
        run_config=run_config,
        train_cfg=train_cfg,
        model_cfg_path=args.model_config,
        dataset_cfg_path=args.dataset_config,
        augment_cfg_path=args.augment_config,
        hardware_cfg_path=args.hardware_config,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
