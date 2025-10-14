"""General trainer implementation with AMP / EMA / logging support."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainerConfig:
    output_root: Path
    stage: str
    dataset: str
    model_name: str
    seed: int
    enable_amp: bool = True
    enable_ema: bool = True
    ema_decay: float = 0.999
    clip_grad_norm: Optional[float] = None


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def store(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()

    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def _macro_f1(confusion: torch.Tensor) -> float:
    tp = torch.diag(confusion).float()
    fp = confusion.sum(dim=1).float() - tp
    fn = confusion.sum(dim=0).float() - tp
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(tp))
    return f1.mean().item()


def _topk_correct(output: torch.Tensor, target: torch.Tensor, k: int) -> int:
    k = min(k, output.size(1))
    _, pred = output.topk(k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return correct[:k].reshape(-1).float().sum().item()


def _vectorized_confusion(confusion: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor) -> None:
    num_classes = confusion.size(0)
    indices = preds.to(torch.int64) * num_classes + targets.to(torch.int64)
    values = torch.ones_like(indices, dtype=torch.int64)
    confusion.view(-1).index_add_(0, indices.cpu(), values.cpu())


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _create_grad_scaler(amp_enabled: bool, device: torch.device):
    if not amp_enabled or device.type != "cuda":
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type="cuda")
    from torch.cuda.amp import GradScaler  # type: ignore

    return GradScaler()


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    if device.type == "cuda":
        from torch.cuda.amp import autocast  # type: ignore

        return autocast(enabled=True)
    return nullcontext()


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.output_root = config.output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        run_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        criterion: nn.Module,
        epochs: int,
        num_classes: int,
        device: torch.device,
        combo_id: str = "na",
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        model.to(device)
        amp_enabled = self.config.enable_amp and device.type == "cuda"
        scaler = _create_grad_scaler(amp_enabled, device)
        ema = EMA(model, self.config.ema_decay) if self.config.enable_ema else None

        best_top1 = float("-inf")
        best_epoch = 0
        best_metrics: Dict[str, Any] = {}
        metrics_records: List[Dict[str, Any]] = []
        lr_history: List[float] = []

        _set_seed(self.config.seed)
        torch.backends.cudnn.benchmark = True

        for epoch in range(1, epochs + 1):
            train_stats = self._run_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                num_classes,
                scaler,
                ema,
                amp_enabled,
                training=True,
            )

            if scheduler is not None:
                scheduler.step()

            eval_stats = self._run_epoch(
                model,
                val_loader,
                optimizer,
                criterion,
                device,
                num_classes,
                scaler=None,
                ema=ema,
                amp_enabled=amp_enabled,
                training=False,
            )

            lr = optimizer.param_groups[0]["lr"]
            lr_history.append(lr)

            timestamp = datetime.utcnow().isoformat() + "Z"
            metrics_records.extend(
                [
                    self._compose_record(run_id, epoch, "train", train_stats, lr, timestamp, combo_id),
                    self._compose_record(run_id, epoch, "val", eval_stats, lr, timestamp, combo_id),
                ]
            )

            if eval_stats["top1"] > best_top1:
                best_top1 = eval_stats["top1"]
                best_epoch = epoch
                best_metrics = eval_stats.copy()
                torch.save(model.state_dict(), run_dir / "best.ckpt")

        self._write_metrics(run_dir, metrics_records)
        summary = self._compose_summary(
            run_id,
            combo_id,
            best_epoch,
            best_metrics,
            lr_history,
            config_snapshot,
        )
        self._write_summary(run_dir, summary)
        self._append_event(run_dir, "completed")
        return summary

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_classes: int,
        scaler: Optional[Any],
        ema: Optional[EMA],
        amp_enabled: bool,
        training: bool,
    ) -> Dict[str, float]:
        if training:
            model.train()
        else:
            model.eval()
            if ema is not None:
                ema.store(model)
                ema.copy_to(model)

        total_loss = 0.0
        correct_top1 = 0.0
        correct_top5 = 0.0
        total_samples = 0
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = inputs.size(0)

            if training:
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, amp_enabled):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if self.config.clip_grad_norm is not None:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if ema is not None:
                    ema.update(model)
            else:
                with torch.no_grad(), _autocast_context(device, amp_enabled):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            correct_top1 += _topk_correct(outputs, targets, 1)
            correct_top5 += _topk_correct(outputs, targets, 5)

            preds = outputs.argmax(dim=1)
            _vectorized_confusion(confusion, preds.cpu(), targets.cpu())

        if not training and ema is not None:
            ema.restore(model)

        avg_loss = total_loss / max(1, total_samples)
        top1 = (correct_top1 / max(1, total_samples)) * 100.0
        top5 = (correct_top5 / max(1, total_samples)) * 100.0
        macro_f1 = _macro_f1(confusion)

        return {
            "loss": avg_loss,
            "top1": top1,
            "top5": top5,
            "macro_f1": macro_f1,
            "num_samples": total_samples,
        }

    def _compose_record(
        self,
        run_id: str,
        epoch: int,
        split: str,
        stats: Dict[str, float],
        lr: float,
        timestamp: str,
        combo_id: str,
    ) -> Dict[str, Any]:
        return {
            "stage": self.config.stage,
            "dataset": self.config.dataset,
            "model": self.config.model_name,
            "seed": self.config.seed,
            "run_id": run_id,
            "combo_id": combo_id,
            "epoch": epoch,
            "split": split,
            "loss": stats["loss"],
            "top1": stats["top1"],
            "top5": stats["top5"],
            "macro_f1": stats["macro_f1"],
            "lr": lr,
            "num_samples": stats["num_samples"],
            "timestamp": timestamp,
        }

    def _write_metrics(self, run_dir: Path, records: List[Dict[str, Any]]) -> None:
        metrics_path = run_dir / "metrics.csv"
        fieldnames = list(records[0].keys()) if records else []
        with metrics_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record)

    def _compose_summary(
        self,
        run_id: str,
        combo_id: str,
        best_epoch: int,
        best_metrics: Dict[str, float],
        lr_history: List[float],
        config_snapshot: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        rng_state = torch.get_rng_state().cpu().numpy().tobytes()
        rng_hash = hashlib.sha256(rng_state).hexdigest()
        return {
            "stage": self.config.stage,
            "dataset": self.config.dataset,
            "model": self.config.model_name,
            "seed": self.config.seed,
            "run_id": run_id,
            "combo_id": combo_id,
            "best_epoch": best_epoch,
            "best_val_top1": best_metrics.get("top1", 0.0),
            "best_val_top5": best_metrics.get("top5", 0.0),
            "best_val_loss": best_metrics.get("loss", 0.0),
            "best_val_macro_f1": best_metrics.get("macro_f1", 0.0),
            "ema_enabled": self.config.enable_ema,
            "ema_decay": self.config.ema_decay if self.config.enable_ema else 0.0,
            "clip_grad_norm": self.config.clip_grad_norm,
            "lr_schedule_snapshot": lr_history,
            "rng_state_hash": rng_hash,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "config_snapshot": config_snapshot or {},
        }

    def _write_summary(self, run_dir: Path, summary: Dict[str, Any]) -> None:
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_event(self, run_dir: Path, status: str) -> None:
        events_path = run_dir / "events.log"
        event_line = f"[{datetime.utcnow().isoformat()}Z] stage={self.config.stage} status={status}\n"
        with events_path.open("a", encoding="utf-8") as fp:
            fp.write(event_line)
