"""Evaluation and statistical reporting utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def paired_summary(candidate: pd.Series, baseline: pd.Series) -> Mapping[str, float]:
    shared = candidate.index.intersection(baseline.index)
    cand = candidate.loc[shared]
    base = baseline.loc[shared]
    diffs = cand - base
    mean = cand.mean()
    std = cand.std(ddof=1) if len(cand) > 1 else 0.0
    delta = diffs.mean()
    if len(shared) > 1:
        t_stat, p_value = stats.ttest_rel(cand, base)
        denom = diffs.std(ddof=1)
        d = delta / denom if denom > 0 else 0.0
    else:
        p_value = np.nan
        d = np.nan
    return {
        "top1_mean": mean,
        "top1_std": std,
        "delta_vs_baseline": delta,
        "p_value": p_value,
        "cohens_d": d,
    }


def summarize_sweep(args: argparse.Namespace) -> None:
    table_path = ROOT / "reports" / "tables" / "single_op_sweep.csv"
    if not table_path.exists():
        raise FileNotFoundError("single_op_sweep.csv not found")
    df = pd.read_csv(table_path)
    df = df[(df["dataset"] == args.dataset) & (df["model"] == args.architecture) & (df["phase"] == args.phase) & (df["kshot"] == args.kshot)]
    if df.empty:
        raise RuntimeError("No sweep runs match the requested filters")

    baseline = df[df["op"] == "baseline"].set_index("seed")
    if baseline.empty:
        raise RuntimeError("Baseline rows missing from sweep results")

    rows: List[Mapping[str, object]] = []
    for (op, param_name, param_value), group in df[df["op"] != "baseline"].groupby(["op", "param_name", "param_value"]):
        candidate_scores = group.set_index("seed")["top1"]
        stats_dict = paired_summary(candidate_scores, baseline["top1"])
        row = {
            "dataset": args.dataset,
            "model": args.architecture,
            "phase": args.phase,
            "kshot": args.kshot,
            "op": op,
            "param_name": param_name,
            "param_value": param_value,
        }
        row.update(stats_dict)
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("delta_vs_baseline", ascending=False)
    out_path = ROOT / "reports" / "tables" / "single_op_sweep_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"Saved sweep summary to {out_path}")


def summarize_confirm(args: argparse.Namespace) -> None:
    table_path = ROOT / "reports" / "tables" / "confirm_10x.csv"
    if not table_path.exists():
        raise FileNotFoundError("confirm_10x.csv not found")
    df = pd.read_csv(table_path)
    df = df[(df["dataset"] == args.dataset) & (df["model"] == args.architecture) & (df["kshot"] == args.kshot)]
    if "seed" in df.columns:
        df = df[df["seed"] == args.seed]
    if df.empty:
        raise RuntimeError("No confirmation runs match the filters")

    groups = df.groupby("combo_id")
    rows = []
    for combo_id, group in groups:
        scores = group["top1_mean"].astype(float)
        mean = scores.mean()
        std = scores.std(ddof=1) if len(scores) > 1 else 0.0
        ci_low, ci_high = stats.t.interval(0.95, len(scores) - 1, loc=mean, scale=std / np.sqrt(len(scores))) if len(scores) > 1 else (np.nan, np.nan)
        rows.append(
            {
                "dataset": args.dataset,
                "model": args.architecture,
                "kshot": args.kshot,
                "combo_id": combo_id,
                "top1_mean": mean,
                "top1_std": std,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("top1_mean", ascending=False)
    out_path = ROOT / "reports" / "tables" / "confirm_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"Saved confirmation summary to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experimental metrics")
    parser.add_argument("--task", choices=["sweep", "confirm"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--phase", default="explore")
    parser.add_argument("--kshot", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "sweep":
        summarize_sweep(args)
    elif args.task == "confirm":
        summarize_confirm(args)
    else:
        raise ValueError(args.task)


if __name__ == "__main__":
    main()
