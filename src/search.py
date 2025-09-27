"""Augmentation search routines: single-operator sweeps and greedy stacking."""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .augment import load_augment_config
from .train import TrainRunConfig, run_training
from .train import load_yaml as load_train_yaml

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _file_lock(path: Path, poll_interval: float = 0.25):
    lock_path = path.with_suffix(path.suffix + ".lock")
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            time.sleep(poll_interval)
    try:
        yield
    finally:
        os.close(fd)
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def to_token(op: str, param_name: str, value) -> str:
    return f"{op}:{param_name}={json.dumps(value) if isinstance(value, (list, dict)) else value}"


def run_sweep(args: argparse.Namespace) -> None:
    train_cfg = load_train_yaml(args.train_config)
    augment_cfg = load_augment_config(args.augment_config)

    search_space = augment_cfg["search_space"][args.op]
    values = search_space["values"]
    param_name = search_space["parameter"]

    seeds = args.seeds or [args.seed]

    records: List[Mapping[str, object]] = []
    baseline_cache: Dict[int, Mapping[str, object]] = {}

    for seed in seeds:
        run_config = TrainRunConfig(
            dataset=args.dataset,
            architecture=args.architecture,
            phase=args.phase,
            kshot=args.kshot,
            seed=seed,
            combo_tokens=None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            output_suffix=args.suffix,
        )
        baseline_summary = run_training(
            run_config,
            train_cfg=train_cfg,
            model_cfg_path=args.model_config,
            dataset_cfg_path=args.dataset_config,
            augment_cfg_path=args.augment_config,
            hardware_cfg_path=args.hardware_config,
        )
        baseline_cache[seed] = baseline_summary
        records.append(
            {
                "dataset": args.dataset,
                "model": args.architecture,
                "phase": args.phase,
                "kshot": args.kshot,
                "seed": seed,
                "op": "baseline",
                "param_name": "-",
                "param_value": "{}",
                "top1": baseline_summary["test_top1"],
                "top5": baseline_summary["test_top5"],
                "delta_vs_baseline": 0.0,
                "run_id": baseline_summary["run_id"],
            }
        )

    for value in values:
        token = to_token(args.op, param_name, value)
        for seed in seeds:
            run_config = TrainRunConfig(
                dataset=args.dataset,
                architecture=args.architecture,
                phase=args.phase,
                kshot=args.kshot,
                seed=seed,
                combo_tokens=[token],
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                output_suffix=args.suffix or args.op,
            )
            summary = run_training(
                run_config,
                train_cfg=train_cfg,
                model_cfg_path=args.model_config,
                dataset_cfg_path=args.dataset_config,
                augment_cfg_path=args.augment_config,
                hardware_cfg_path=args.hardware_config,
            )
            baseline = baseline_cache[seed]
            delta = summary["test_top1"] - baseline["test_top1"]
            records.append(
                {
                    "dataset": args.dataset,
                    "model": args.architecture,
                    "phase": args.phase,
                    "kshot": args.kshot,
                    "seed": seed,
                    "op": args.op,
                    "param_name": param_name,
                    "param_value": json.dumps(value) if isinstance(value, (list, dict)) else value,
                    "top1": summary["test_top1"],
                    "top5": summary["test_top5"],
                    "delta_vs_baseline": delta,
                    "run_id": summary["run_id"],
                }
            )

    table_path = ROOT / "reports" / "tables" / "single_op_sweep.csv"
    df = pd.DataFrame(records)
    with _file_lock(table_path):
        if table_path.exists():
            existing = pd.read_csv(table_path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(
                subset=[
                    "dataset",
                    "model",
                    "phase",
                    "kshot",
                    "seed",
                    "op",
                    "param_name",
                    "param_value",
                    "run_id",
                ],
                keep="last",
            )
        table_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(table_path, index=False)


def parse_value(value_str: str):
    try:
        return ast.literal_eval(value_str)
    except Exception:
        return value_str


def load_candidate_tokens(args: argparse.Namespace) -> List[str]:
    table_path = ROOT / "reports" / "tables" / "single_op_sweep.csv"
    if not table_path.exists():
        raise FileNotFoundError("single_op_sweep.csv not found. Run sweep first.")
    df = pd.read_csv(table_path)
    df = df[(df["dataset"] == args.dataset) & (df["model"] == args.architecture) & (df["phase"] == args.phase) & (df["kshot"] == args.kshot)]
    if df.empty:
        raise RuntimeError("No sweep results matching the requested configuration.")
    df = df[df["op"] != "baseline"].sort_values("delta_vs_baseline", ascending=False)
    topk = df.head(args.topk)
    tokens = []
    for _, row in topk.iterrows():
        value = parse_value(row["param_value"])
        tokens.append(to_token(row["op"], row["param_name"], value))
    return tokens


def run_greedy(args: argparse.Namespace) -> None:
    train_cfg = load_train_yaml(args.train_config)
    augment_cfg = load_augment_config(args.augment_config)

    candidate_tokens = args.candidates or load_candidate_tokens(args)

    run_config = TrainRunConfig(
        dataset=args.dataset,
        architecture=args.architecture,
        phase=args.phase,
        kshot=args.kshot,
        seed=args.seed,
        combo_tokens=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_suffix=args.suffix,
    )

    baseline_summary = run_training(
        run_config,
        train_cfg=train_cfg,
        model_cfg_path=args.model_config,
        dataset_cfg_path=args.dataset_config,
        augment_cfg_path=args.augment_config,
        hardware_cfg_path=args.hardware_config,
    )

    records = [
        {
            "dataset": args.dataset,
            "model": args.architecture,
            "phase": args.phase,
            "kshot": args.kshot,
            "seed": args.seed,
            "step": 0,
            "added_op": "baseline",
            "params": "{}",
            "delta_step": 0.0,
            "delta_cum": 0.0,
            "run_id": baseline_summary["run_id"],
            "passed_epsilon": True,
        }
    ]

    selected: List[str] = []
    best_score = baseline_summary["test_top1"]

    for depth in range(1, args.max_depth + 1):
        best_candidate = None
        best_summary = None
        best_delta = 0.0

        for token in candidate_tokens:
            if token in selected:
                continue
            combo = selected + [token]
            combo_run = TrainRunConfig(
                dataset=args.dataset,
                architecture=args.architecture,
                phase=args.phase,
                kshot=args.kshot,
                seed=args.seed,
                combo_tokens=combo,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                output_suffix=args.suffix or f"greedy_{depth}",
            )
            summary = run_training(
                combo_run,
                train_cfg=train_cfg,
                model_cfg_path=args.model_config,
                dataset_cfg_path=args.dataset_config,
                augment_cfg_path=args.augment_config,
                hardware_cfg_path=args.hardware_config,
            )
            delta = summary["test_top1"] - best_score
            if delta > best_delta:
                best_delta = delta
                best_candidate = token
                best_summary = summary

        if best_candidate is None or best_delta < args.epsilon:
            break

        selected.append(best_candidate)
        best_score = best_summary["test_top1"]  # type: ignore[index]
        records.append(
            {
                "dataset": args.dataset,
                "model": args.architecture,
                "phase": args.phase,
                "kshot": args.kshot,
                "seed": args.seed,
                "step": len(selected),
                "added_op": best_candidate.split(":", 1)[0],
                "params": best_candidate,
                "delta_step": best_delta,
                "delta_cum": best_score - baseline_summary["test_top1"],
                "run_id": best_summary["run_id"],  # type: ignore[index]
                "passed_epsilon": True,
            }
        )

    table_path = ROOT / "reports" / "tables" / "greedy_path.csv"
    df = pd.DataFrame(records)
    if table_path.exists():
        existing = pd.read_csv(table_path)
        df = pd.concat([existing, df], ignore_index=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augmentation search helper")
    parser.add_argument("--mode", choices=["sweep", "greedy"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--phase", default="explore")
    parser.add_argument("--kshot", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", help="Optional list of seeds for sweep mode")
    parser.add_argument("--op", help="Operator to sweep")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--candidates", nargs="*", help="Candidate tokens for greedy search")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--suffix")
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs" / "train.yaml")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--dataset-config", type=Path, default=ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--augment-config", type=Path, default=ROOT / "configs" / "augment" / "aug_search.yaml")
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "configs" / "hardware.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "sweep":
        if not args.op:
            raise ValueError("--op is required for sweep mode")
        run_sweep(args)
    elif args.mode == "greedy":
        run_greedy(args)
    else:
        raise ValueError(f"Unsupported mode {args.mode}")


if __name__ == "__main__":
    main()
