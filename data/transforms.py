"""
增强构造工厂。

本文件落实 implementation_plan.md 中的增强约定：
- 单因子阶段：按 7 档强度生成；
- Sobol 阶段：读取 CSV 行构造组合，并处理 `scale_delta`、离散 `crop`；
- RSM 阶段：基于设计矩阵恢复真实参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

try:
    from torchvision import transforms as T
except ImportError as exc:  # pragma: no cover - 需要调用者安装 torchvision
    raise RuntimeError("torchvision 未安装，无法构造数据增强。请先安装 torchvision。") from exc

CROP_CHOICES: Tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12)
CIFAR100_MEAN: Tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
CIFAR100_STD: Tuple[float, float, float] = (0.2675, 0.2565, 0.2761)


@dataclass
class TransformConfig:
    """简单的数据结构，用于在日志中回溯构造参数。"""

    operation: str
    params: Dict[str, float]


def snap_crop_padding(value: float) -> int:
    """将连续的 padding 值映射到离散合法档位。"""
    idx = int(round(value))
    idx = max(0, min(idx, len(CROP_CHOICES) - 1))
    return CROP_CHOICES[idx]


@dataclass
class SingleFactorParams:
    crop_padding: int = 0
    flip_prob: float = 0.0
    rotation_deg: float = 0.0
    scale_delta: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    erasing_p: float = 0.0
    erasing_scale_max: float = 0.02


def _compose_pipeline(
    *,
    crop_padding: int,
    flip_prob: float,
    rotation_deg: float,
    scale_delta: float,
    color_kwargs: Dict[str, float],
    erasing_kwargs: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
) -> T.Compose:
    scale_low = max(0.5, 1.0 - scale_delta)
    scale_high = 1.0 + scale_delta

    ops: List[object] = [
        T.RandomCrop(32, padding=crop_padding),
        T.RandomHorizontalFlip(p=flip_prob),
        T.RandomRotation(degrees=rotation_deg),
        T.RandomAffine(degrees=0, scale=(scale_low, scale_high)),
        T.ColorJitter(
            brightness=color_kwargs.get("brightness", 0.0),
            contrast=color_kwargs.get("contrast", 0.0),
            saturation=color_kwargs.get("saturation", 0.0),
            hue=color_kwargs.get("hue", 0.0),
        ),
        T.ToTensor(),
    ]

    if erasing_kwargs and float(erasing_kwargs.get("p", 0.0)) > 0.0:
        ops.append(
            T.RandomErasing(
                p=float(erasing_kwargs["p"]),
                scale=erasing_kwargs.get("scale", (0.02, 0.02)),  # type: ignore[arg-type]
                ratio=erasing_kwargs.get("ratio", (0.3, 3.3)),  # type: ignore[arg-type]
            )
        )

    ops.append(T.Normalize(CIFAR100_MEAN, CIFAR100_STD))
    return T.Compose(ops)


def make_single_factor_transform(operation: str, strength: float) -> T.Compose:
    """构造仅调节单一增强调度的组合，其余操作保持最弱强度。"""
    params = SingleFactorParams()
    op = operation.lower()

    if op in {"random_crop", "crop"}:
        params.crop_padding = int(strength)
    elif op in {"horizontal_flip", "flip"}:
        params.flip_prob = float(strength)
    elif op in {"rotation", "random_rotation"}:
        params.rotation_deg = float(strength)
    elif op in {"scaling", "scale", "random_affine_scale"}:
        params.scale_delta = max(0.0, float(strength))
    elif op == "brightness":
        params.brightness = max(0.0, float(strength))
    elif op == "contrast":
        params.contrast = max(0.0, float(strength))
    elif op == "saturation":
        params.saturation = max(0.0, float(strength))
    elif op == "hue":
        params.hue = max(0.0, float(strength))
    elif op in {"random_erasing", "erasing"}:
        params.erasing_p = 0.5 if strength > 0 else 0.0
        params.erasing_scale_max = max(0.02, float(strength))
    else:
        raise NotImplementedError(f"Operation '{operation}' not implemented yet.")

    erasing_kwargs: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None
    if params.erasing_p > 0.0:
        erasing_kwargs = {
            "p": params.erasing_p,
            "scale": (0.02, params.erasing_scale_max),
            "ratio": (0.3, 3.3),
        }

    return _compose_pipeline(
        crop_padding=params.crop_padding,
        flip_prob=params.flip_prob,
        rotation_deg=params.rotation_deg,
        scale_delta=params.scale_delta,
        color_kwargs={
            "brightness": params.brightness,
            "contrast": params.contrast,
            "saturation": params.saturation,
            "hue": params.hue,
        },
        erasing_kwargs=erasing_kwargs,
    )


def make_sobol_transform(row: Dict[str, float]) -> T.Compose:
    """
    将 Sobol CSV 行转换为 torchvision 组合。

    参数说明：
    - row 中应包含 implementation_plan.md 中给出的字段；
    - `scale_delta` 用于生成 `(scale_low, scale_high)`；
    - 离散字段需 snap 到合法集合。
    """
    scale_delta = float(row["scale_delta"])
    scale_low = max(0.5, 1.0 - scale_delta)
    scale_high = 1.0 + scale_delta

    return _compose_pipeline(
        crop_padding=snap_crop_padding(row["crop"]),
        flip_prob=float(row["flip"]),
        rotation_deg=float(row["rotation"]),
        scale_delta=float(row["scale_delta"]),
        color_kwargs={
            "brightness": float(row["brightness"]),
            "contrast": float(row["contrast"]),
            "saturation": float(row["saturation"]),
            "hue": float(row["hue"]),
        },
        erasing_kwargs={
            "p": 0.5,
            "scale": (0.02, float(row["erase"])),
            "ratio": (0.3, 3.3),
        },
    )


def make_rsm_transform(row: Dict[str, float]) -> T.Compose:
    """
    根据 RSM 设计矩阵恢复增强组合。

    设计矩阵应提供原始 scale_delta 或直接给出 (scale_low, scale_high)。
    """
    processed = row.copy()
    if "scale_delta" not in processed and {"scale_low", "scale_high"} <= processed.keys():
        low = float(processed["scale_low"])
        high = float(processed["scale_high"])
        processed["scale_delta"] = max(0.0, (high - low) / 2.0)
    elif "scale_delta" not in processed:
        raise KeyError("设计矩阵缺少 scale_delta 或 (scale_low, scale_high) 字段。")

    return make_sobol_transform(processed)
