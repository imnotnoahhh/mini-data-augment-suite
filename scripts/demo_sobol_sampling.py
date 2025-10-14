#!/usr/bin/env python3
"""
生成少量 Sobol 组合示例，用于验证 CSV/JSON 字段与 implementation_plan.md 保持一致。

运行后将在 `artifacts/sobol/` 下写出：
- `sobol_results_demo.csv`
- `sobol_transforms_demo.json`
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

# 允许在脚本模式下导入仓库内模块
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

try:
    from data.transforms import snap_crop_padding, CROP_CHOICES  # type: ignore
except (ModuleNotFoundError, RuntimeError):
    # torchvision 可能尚未安装，直接复用实现以避免依赖。
    CROP_CHOICES = (0, 2, 4, 6, 8, 10, 12)

    def snap_crop_padding(value: float) -> int:
        idx = int(round(value))
        idx = max(0, min(idx, len(CROP_CHOICES) - 1))
        return CROP_CHOICES[idx]

try:
    from torch.quasirandom import SobolEngine
except ImportError:  # pragma: no cover - fallback for environments without torch
    SobolEngine = None

ARTIFACT_ROOT = Path("artifacts") / "sobol"
CSV_PATH = ARTIFACT_ROOT / "sobol_results_demo.csv"
TRANSFORM_JSON_PATH = ARTIFACT_ROOT / "sobol_transforms_demo.json"

PARAM_RANGES: Dict[str, tuple[float, float]] = {
    "brightness": (0.0, 0.4),
    "contrast": (0.0, 0.5),
    "saturation": (0.0, 0.5),
    "hue": (0.0, 0.15),
    "crop": (0, 6),
    "flip": (0.1, 0.9),
    "rotation": (0, 20),
    "scale_delta": (0.0, 0.3),
    "erase": (0.02, 0.24),
}

FIELD_ORDER: List[str] = list(PARAM_RANGES.keys())


def draw_unit_samples(num_samples: int, dimension: int) -> List[List[float]]:
    if SobolEngine is not None:
        engine = SobolEngine(dimension, scramble=True, seed=0)
        return engine.draw(num_samples).tolist()

    # Fallback：使用 numpy 随机数，但保持 seed 固定，说明已退化为随机采样。
    try:
        import numpy as np  # type: ignore
    except ImportError:  # pragma: no cover - final fallback
        import random

        random.seed(0)
        return [[random.random() for _ in range(dimension)] for _ in range(num_samples)]

    rng = np.random.default_rng(seed=0)
    return rng.random(size=(num_samples, dimension)).tolist()


def scale_sample(sample: List[float]) -> Dict[str, float]:
    scaled: Dict[str, float] = {}
    for key, value, bounds in zip(FIELD_ORDER, sample, PARAM_RANGES.values()):
        low, high = bounds
        scaled[key] = low + (high - low) * float(value)
    return scaled


def format_row(raw_row: Dict[str, float]) -> Dict[str, float]:
    formatted = raw_row.copy()
    # snap crop to discrete padding choices
    formatted["crop"] = float(snap_crop_padding(raw_row["crop"]))
    formatted["flip"] = round(raw_row["flip"], 4)
    formatted["rotation"] = round(raw_row["rotation"], 2)
    for key in ("brightness", "contrast", "saturation", "hue", "scale_delta", "erase"):
        formatted[key] = round(raw_row[key], 4)
    return formatted


def main(num_samples: int = 4) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    samples = draw_unit_samples(num_samples, dimension=len(PARAM_RANGES))

    csv_rows: List[Dict[str, float]] = []
    transform_rows: List[Dict[str, float]] = []

    for combo_id, sample in enumerate(samples):
        raw_row = scale_sample(sample)
        row = format_row(raw_row)
        csv_row = {"combo_id": combo_id, **row}
        csv_rows.append(csv_row)

        scale_delta = row["scale_delta"]
        transform_rows.append(
            {
                "combo_id": combo_id,
                "scale_delta": scale_delta,
                "scale_low": round(max(0.5, 1.0 - scale_delta), 4),
                "scale_high": round(1.0 + scale_delta, 4),
                "crop_padding": int(row["crop"]),
            }
        )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["combo_id", *FIELD_ORDER])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    TRANSFORM_JSON_PATH.write_text(json.dumps(transform_rows, indent=2), encoding="utf-8")

    if SobolEngine is None:
        print("torch 未安装，使用随机采样作为占位；正式运行请安装 PyTorch 以获得 Sobol 序列。")

    # 可选：尝试构造 torchvision transform 以确保参数合法
    try:
        from data.transforms import make_sobol_transform  # type: ignore

        _ = [make_sobol_transform({k: row[k] for k in FIELD_ORDER}) for row in csv_rows]
    except Exception as exc:  # pragma: no cover - 环境可能缺少 torchvision
        print(f"warning: 无法实例化 torchvision transform ({exc}); 可在安装 torch/torchvision 后重试。")

    print(f"Wrote {CSV_PATH} and {TRANSFORM_JSON_PATH}")


if __name__ == "__main__":
    main()
