"""
Single-factor stage reporting utilities.

This module centralises all heavy lifting so that both the CLI script and the
Jupyter Notebook can produce byte-identical tables and plots.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score, top_k_accuracy_score
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from data.splits import load_split
from utils.preview import DATASET_STATS

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover - torchvision is expected to be installed
    raise RuntimeError("torchvision is required to build evaluation datasets.") from exc

try:
    from orchestration.run_stage import build_model, DATASET_NUM_CLASSES
except ImportError as exc:  # pragma: no cover - defensive
    raise RuntimeError("Failed to import build_model from orchestration.run_stage") from exc

LOGGER = logging.getLogger(__name__)


@dataclass
class EventRecord:
    timestamp: datetime
    payload: Dict[str, str]


@dataclass
class RunRecord:
    run_path: Path
    run_id: str
    combo_id: str
    operation: str
    strength_label: str
    strength_value: float
    seed: int
    summary: Dict[str, object]
    metrics: pd.DataFrame
    events: List[EventRecord]
    config: Dict[str, object]
    best_checkpoint: Path
    dataset_name: str
    dataset_root: Path
    split_seed: int
    num_classes: int
    note: str = ""
    early_stop: bool = False
    eval_metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix_path: Optional[Path] = None
    training_curve_path: Optional[Path] = None
    lr_curve_path: Optional[Path] = None


@dataclass
class ReportArtifacts:
    run_dataframe: pd.DataFrame
    operation_stats: pd.DataFrame
    best_strengths: pd.DataFrame
    output_root: Path


def generate_report(
    stage_dir: Path,
    output_dir: Path,
    *,
    device: Optional[str] = None,
    batch_size: int = 256,
    num_workers: int = 0,
) -> ReportArtifacts:
    """
    High-level orchestration entry. Returns a set of pandas DataFrames for further use.
    """
    stage_dir = stage_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Scanning runs under %s", stage_dir)
    records = _collect_run_records(stage_dir)

    if not records:
        raise RuntimeError(f"No runs found under {stage_dir}")

    LOGGER.info("Creating run-level summary")
    run_df = _build_run_dataframe(records)
    run_summary_path = output_dir / "run_summary.csv"
    run_df.to_csv(run_summary_path, index=False)

    LOGGER.info("Aggregating operation × strength statistics")
    op_stats = _compute_operation_stats(run_df)
    op_stats_path = output_dir / "operation_strength_stats.csv"
    op_stats.to_csv(op_stats_path, index=False)
    _save_per_operation_tables(op_stats, output_dir / "tables")

    LOGGER.info("Evaluating checkpoints to produce confusion matrices")
    device_name = _resolve_device(device)
    _evaluate_confusion_matrices(
        records,
        device_name,
        batch_size=batch_size,
        num_workers=num_workers,
        output_root=output_dir / "plots" / "confusion_matrices",
    )

    LOGGER.info("Rebuilding run summary with evaluation metrics")
    run_df = _build_run_dataframe(records)
    run_df.to_csv(run_summary_path, index=False)

    LOGGER.info("Updating operation statistics with evaluation metrics and notes")
    op_stats = _compute_operation_stats(run_df)
    op_stats.to_csv(op_stats_path, index=False)
    _save_per_operation_tables(op_stats, output_dir / "tables", overwrite=True)

    LOGGER.info("Deriving best strength per operation")
    best_strengths = _extract_best_strengths(op_stats)
    best_strengths_path = output_dir / "best_strengths.csv"
    best_strengths.to_csv(best_strengths_path, index=False)

    LOGGER.info("Generating plots")
    _generate_strength_curves(op_stats, output_dir / "plots" / "strength_curves")
    _generate_best_strength_plot(best_strengths, output_dir / "plots" / "best_strength")
    _generate_training_curves(records, output_dir / "plots" / "training_curves")
    _generate_learning_rate_curves(records, output_dir / "plots" / "lr_curves")

    return ReportArtifacts(
        run_dataframe=run_df,
        operation_stats=op_stats,
        best_strengths=best_strengths,
        output_root=output_dir,
    )


# --------------------------------------------------------------------------- #
# Data ingestion
# --------------------------------------------------------------------------- #


def _collect_run_records(stage_dir: Path) -> List[RunRecord]:
    records: List[RunRecord] = []
    for run_path in sorted(p for p in stage_dir.iterdir() if p.is_dir()):
        run_id = run_path.name
        summary_path = run_path / "summary.json"
        metrics_path = run_path / "metrics.csv"
        transform_path = run_path / "transform.json"
        config_path = run_path / "config.yaml"
        best_ckpt = run_path / "best.ckpt"
        events_path = run_path / "events.log"

        if not summary_path.exists() or not metrics_path.exists():
            LOGGER.warning("Skipping run %s (missing summary or metrics)", run_id)
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = pd.read_csv(metrics_path)

        transform_payload = json.loads(transform_path.read_text(encoding="utf-8"))
        operation = str(transform_payload.get("operation", "unknown")).lower()
        strength_raw = transform_payload.get("strength", 0)
        strength_value = float(strength_raw)
        strength_label = str(strength_raw)

        seed = int(summary.get("seed", _infer_seed_from_run_id(run_id)))
        combo_id = str(summary.get("combo_id", run_id))

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dataset_cfg = config.get("dataset", {})
        dataset_name = str(dataset_cfg.get("name", summary.get("dataset", "unknown"))).lower()
        data_root = Path(dataset_cfg.get("data_root", "data/raw")).expanduser().resolve()
        split_seed = int(dataset_cfg.get("split_seed", summary.get("seed", 0)))
        num_classes = DATASET_NUM_CLASSES.get(dataset_name)
        if num_classes is None:
            raise ValueError(f"Unknown dataset {dataset_name} for run {run_id}")

        events = _parse_events(events_path)
        early_stop = any(evt.payload.get("status") == "early_stop" for evt in events)
        note = "; ".join(sorted({_event_to_note(evt) for evt in events})) or "status=unknown"

        record = RunRecord(
            run_path=run_path,
            run_id=run_id,
            combo_id=combo_id,
            operation=operation,
            strength_label=strength_label,
            strength_value=strength_value,
            seed=seed,
            summary=summary,
            metrics=metrics,
            events=events,
            config=config,
            best_checkpoint=best_ckpt,
            dataset_name=dataset_name,
            dataset_root=data_root,
            split_seed=split_seed,
            num_classes=num_classes,
            note=note,
            early_stop=early_stop,
        )
        records.append(record)
    return records


def _parse_events(events_path: Path) -> List[EventRecord]:
    if not events_path.exists():
        return []
    records: List[EventRecord] = []
    pattern = re.compile(r"^\[(?P<ts>.+?)\]\s+(?P<body>.+)$")
    for line in events_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        timestamp_str = match.group("ts")
        body = match.group("body")
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            timestamp = datetime.utcnow()
        payload: Dict[str, str] = {}
        for part in body.split():
            if "=" in part:
                key, value = part.split("=", 1)
                payload[key] = value
        records.append(EventRecord(timestamp=timestamp, payload=payload))
    return records


def _event_to_note(event: EventRecord) -> str:
    status = event.payload.get("status")
    if status:
        return f"status={status}"
    return "status=unknown"


def _infer_seed_from_run_id(run_id: str) -> int:
    match = re.search(r"seed(\d+)", run_id)
    if match:
        return int(match.group(1))
    return 0


# --------------------------------------------------------------------------- #
# Dataframe construction & aggregation
# --------------------------------------------------------------------------- #


def _build_run_dataframe(records: Sequence[RunRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        summary = rec.summary
        lr_snapshot = summary.get("lr_schedule_snapshot", [])
        best_epoch = summary.get("best_epoch", None)
        max_epoch = int(rec.metrics["epoch"].max())
        train_epochs = rec.metrics[rec.metrics["split"] == "train"]["epoch"]
        val_epochs = rec.metrics[rec.metrics["split"] == "val"]["epoch"]
        events_status = "; ".join(sorted({_event_to_note(evt) for evt in rec.events})) or "status=unknown"
        row = {
            "run_id": rec.run_id,
            "combo_id": rec.combo_id,
            "operation": rec.operation,
            "strength": rec.strength_value,
            "strength_label": rec.strength_label,
            "seed": rec.seed,
            "best_epoch": best_epoch,
            "max_epoch": max_epoch,
            "train_epochs": len(train_epochs.unique()),
            "val_epochs": len(val_epochs.unique()),
            "best_val_top1": summary.get("best_val_top1"),
            "best_val_top5": summary.get("best_val_top5"),
            "best_val_macro_f1": summary.get("best_val_macro_f1"),
            "best_val_loss": summary.get("best_val_loss"),
            "lr_points": len(lr_snapshot),
            "early_stop": rec.early_stop,
            "note": rec.note,
            "events": events_status,
            "confusion_matrix_path": str(rec.confusion_matrix_path) if rec.confusion_matrix_path else "",
            "training_curve_path": str(rec.training_curve_path) if rec.training_curve_path else "",
            "lr_curve_path": str(rec.lr_curve_path) if rec.lr_curve_path else "",
            "eval_top1": rec.eval_metrics.get("top1"),
            "eval_top5": rec.eval_metrics.get("top5"),
            "eval_macro_f1": rec.eval_metrics.get("macro_f1"),
            "eval_loss": rec.eval_metrics.get("loss"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.sort_values(["operation", "strength", "seed"], inplace=True)
    return df


def _compute_operation_stats(run_df: pd.DataFrame) -> pd.DataFrame:
    def _agg_ci(series: pd.Series) -> Tuple[float, float]:
        n = series.count()
        if n == 0:
            return (math.nan, math.nan)
        mean = series.mean()
        if n == 1:
            return (mean, mean)
        std = series.std(ddof=1)
        half = 1.96 * (std / math.sqrt(n))
        return (mean - half, mean + half)

    aggregations = []
    grouped = run_df.groupby(["operation", "strength", "strength_label"], dropna=False)
    for (operation, strength, strength_label), group in grouped:
        top1_ci = _agg_ci(group["best_val_top1"])
        top5_ci = _agg_ci(group["best_val_top5"])
        f1_ci = _agg_ci(group["best_val_macro_f1"])
        loss_ci = _agg_ci(group["best_val_loss"])
        eval_top1_ci = _agg_ci(group["eval_top1"])

        aggregations.append(
            {
                "operation": operation,
                "strength": strength,
                "strength_label": strength_label,
                "runs": len(group),
                "top1_mean": group["best_val_top1"].mean(),
                "top1_std": group["best_val_top1"].std(ddof=1),
                "top1_ci_low": top1_ci[0],
                "top1_ci_high": top1_ci[1],
                "top5_mean": group["best_val_top5"].mean(),
                "top5_std": group["best_val_top5"].std(ddof=1),
                "top5_ci_low": top5_ci[0],
                "top5_ci_high": top5_ci[1],
                "macro_f1_mean": group["best_val_macro_f1"].mean(),
                "macro_f1_std": group["best_val_macro_f1"].std(ddof=1),
                "macro_f1_ci_low": f1_ci[0],
                "macro_f1_ci_high": f1_ci[1],
                "loss_mean": group["best_val_loss"].mean(),
                "loss_std": group["best_val_loss"].std(ddof=1),
                "loss_ci_low": loss_ci[0],
                "loss_ci_high": loss_ci[1],
                "eval_top1_mean": group["eval_top1"].mean(),
                "eval_top1_ci_low": eval_top1_ci[0],
                "eval_top1_ci_high": eval_top1_ci[1],
                "any_early_stop": bool(group["early_stop"].any()),
                "notes": "; ".join(sorted(set(n for n in group["note"] if isinstance(n, str)))),
            }
        )
    df = pd.DataFrame(aggregations)
    df.sort_values(["operation", "strength"], inplace=True)
    return df


def _save_per_operation_tables(
    op_stats: pd.DataFrame,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for operation, group in op_stats.groupby("operation"):
        path = output_dir / f"{operation}_stats.csv"
        if path.exists() and not overwrite:
            continue
        group.to_csv(path, index=False)


def _extract_best_strengths(op_stats: pd.DataFrame) -> pd.DataFrame:
    idx = op_stats.groupby("operation")["top1_mean"].idxmax()
    best = op_stats.loc[idx].copy()
    best.sort_values("top1_mean", ascending=False, inplace=True)
    return best


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _evaluate_confusion_matrices(
    records: Sequence[RunRecord],
    device_name: str,
    *,
    batch_size: int,
    num_workers: int,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    for rec in tqdm(records, desc="Confusion matrices", leave=False):
        if not rec.best_checkpoint.exists():
            LOGGER.warning("Skipping evaluation for %s (missing checkpoint)", rec.run_id)
            continue

        matrix_path = output_root / f"{rec.run_id}_confusion_matrix.png"
        if matrix_path.exists():
            rec.confusion_matrix_path = matrix_path
            continue

        dataloader = _build_val_dataloader(
            dataset_name=rec.dataset_name,
            data_root=rec.dataset_root,
            split_seed=rec.split_seed,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        model = build_model(rec.summary.get("model", rec.config.get("model", {}).get("name", "resnet18")), rec.num_classes)
        state_dict = torch.load(rec.best_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        all_probs: List[np.ndarray] = []
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        running_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_probs.append(probs)
                all_preds.append(preds)
                all_targets.append(targets.cpu().numpy())

        probs = np.concatenate(all_probs, axis=0)
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        avg_loss = float(running_loss / len(targets))

        top1 = float((preds == targets).mean() * 100.0)
        top5 = float(top_k_accuracy_score(targets, probs, k=5) * 100.0)
        macro = float(f1_score(targets, preds, average="macro"))

        conf_mat = confusion_matrix(targets, preds, labels=list(range(rec.num_classes)))
        _plot_confusion_matrix(conf_mat, matrix_path, title=f"{rec.run_id} – Confusion Matrix")

        rec.eval_metrics = {
            "top1": top1,
            "top5": top5,
            "macro_f1": macro,
            "loss": avg_loss,
        }
        rec.confusion_matrix_path = matrix_path


def _build_val_dataloader(
    *,
    dataset_name: str,
    data_root: Path,
    split_seed: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_STATS:
        raise ValueError(f"Dataset {dataset_name} not registered in DATASET_STATS")
    stats = DATASET_STATS[dataset_name]
    mean, std = stats["mean"], stats["std"]

    if dataset_name == "cifar100":
        base_dataset = datasets.CIFAR100(
            root=str(data_root),
            train=True,
            download=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        )
    else:  # pragma: no cover - placeholder for future datasets
        raise NotImplementedError(f"Validation loader not implemented for dataset {dataset_name}")

    split = load_split(dataset_name, split_seed)
    val_subset = Subset(base_dataset, split.val_indices)
    return DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _plot_confusion_matrix(matrix: np.ndarray, output_path: Path, *, title: str) -> None:
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def _configure_matplotlib() -> None:
    sns.set_theme(context="paper", style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "figure.figsize": (6, 4),
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _generate_strength_curves(op_stats: pd.DataFrame, output_dir: Path) -> None:
    _configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    for operation, group in op_stats.groupby("operation"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
        strengths = group["strength"].values

        def _plot_metric(ax, mean_col, ci_low_col, ci_high_col, ylabel, color):
            means = group[mean_col].values
            lower = group[ci_low_col].values
            upper = group[ci_high_col].values
            ax.plot(strengths, means, marker="o", color=color)
            ax.fill_between(strengths, lower, upper, color=color, alpha=0.2)
            ax.set_title(f"{operation.title()} – {ylabel}")
            ax.set_xlabel("Strength")
            ax.set_ylabel(ylabel)

        _plot_metric(axes[0], "top1_mean", "top1_ci_low", "top1_ci_high", "Top-1 (%)", "#1f77b4")
        _plot_metric(
            axes[1],
            "macro_f1_mean",
            "macro_f1_ci_low",
            "macro_f1_ci_high",
            "Macro F1",
            "#2ca02c",
        )
        _plot_metric(axes[2], "loss_mean", "loss_ci_low", "loss_ci_high", "Val Loss", "#d62728")

        fig.suptitle(f"{operation.title()} – Strength Sweep", fontsize=14)
        plt.tight_layout()
        output_path = output_dir / f"{operation}_strength_curves.png"
        plt.savefig(output_path)
        plt.close(fig)


def _generate_best_strength_plot(best_strengths: pd.DataFrame, output_dir: Path) -> None:
    _configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        best_strengths["operation"].str.title(),
        best_strengths["top1_mean"],
        yerr=best_strengths["top1_mean"] - best_strengths["top1_ci_low"],
        capsize=4,
        color="#1f77b4",
    )
    ax.set_ylabel("Top-1 (%)")
    ax.set_xlabel("Operation")
    ax.set_title("Best Strength per Operation (Top-1 ± 95% CI)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    output_path = output_dir / "best_strengths.png"
    plt.savefig(output_path)
    plt.close(fig)


def _generate_training_curves(records: Sequence[RunRecord], output_dir: Path) -> None:
    _configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    for rec in tqdm(records, desc="Training curves", leave=False):
        df = rec.metrics.copy()
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(train_df["epoch"], train_df["loss"], label="Train", color="#1f77b4")
        axes[0].plot(val_df["epoch"], val_df["loss"], label="Val", color="#d62728")
        axes[0].set_title("Loss Curve")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(train_df["epoch"], train_df["top1"], label="Train Top-1", color="#2ca02c")
        axes[1].plot(val_df["epoch"], val_df["top1"], label="Val Top-1", color="#ff7f0e")
        axes[1].set_title("Top-1 Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()

        fig.suptitle(f"{rec.run_id} – Training Dynamics", fontsize=14)
        plt.tight_layout()
        output_path = output_dir / f"{rec.run_id}_training_curve.png"
        plt.savefig(output_path)
        plt.close(fig)
        rec.training_curve_path = output_path


def _generate_learning_rate_curves(records: Sequence[RunRecord], output_dir: Path) -> None:
    _configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    for rec in tqdm(records, desc="LR curves", leave=False):
        lr_snapshot = rec.summary.get("lr_schedule_snapshot", [])
        if not lr_snapshot:
            continue
        epochs = np.arange(1, len(lr_snapshot) + 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, lr_snapshot, color="#1f77b4")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title(f"{rec.run_id} – Learning Rate Schedule")
        plt.tight_layout()
        output_path = output_dir / f"{rec.run_id}_lr_curve.png"
        plt.savefig(output_path)
        plt.close(fig)
        rec.lr_curve_path = output_path


# --------------------------------------------------------------------------- #
# Public helpers (Notebook-friendly)
# --------------------------------------------------------------------------- #


def preview_tables(artifacts: ReportArtifacts, max_rows: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Convenience helper for notebooks: returns truncated tables for quick display.
    """
    return {
        "run_summary_head": artifacts.run_dataframe.head(max_rows),
        "operation_stats_head": artifacts.operation_stats.head(max_rows),
        "best_strengths": artifacts.best_strengths,
    }


__all__ = [
    "generate_report",
    "preview_tables",
    "ReportArtifacts",
]
