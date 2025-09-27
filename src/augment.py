"""Augmentation pipeline factory aligned with the experiment specification."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T

import yaml


def _is_zero_like(value: object) -> bool:
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    if isinstance(value, (list, tuple)):
        return all(_is_zero_like(v) for v in value)
    return False


def _normalize_affine_arg(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, tuple):
        return tuple(value)
    return value


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUG_CFG = ROOT / "configs" / "augment" / "aug_search.yaml"


def load_augment_config(path: Path = DEFAULT_AUG_CFG) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_interpolation(name: str) -> InterpolationMode:
    name = name.lower()
    if name == "bicubic":
        return InterpolationMode.BICUBIC
    if name == "bilinear":
        return InterpolationMode.BILINEAR
    if name == "nearest":
        return InterpolationMode.NEAREST
    raise ValueError(f"Unsupported interpolation mode: {name}")


def _parse_combo_tokens(combo: Sequence[str], search_space: Mapping[str, Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    overrides: Dict[str, Dict[str, object]] = {}
    for token in combo:
        if ":" not in token:
            raise ValueError(f"Combo token '{token}' must be in the form op:param=value")
        op_name, rest = token.split(":", 1)
        if "=" not in rest:
            raise ValueError(f"Combo token '{token}' missing '='")
        param_name, raw_value = rest.split("=", 1)
        space_entry = search_space.get(op_name)
        if space_entry is None:
            raise KeyError(f"Unknown operator '{op_name}' in combo")
        target = space_entry.get("target", op_name)
        try:
            value = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            value = raw_value
        overrides.setdefault(target, {})[param_name] = value
    return overrides


def _merge_overrides(base: MutableMapping[str, object], overrides: Mapping[str, Mapping[str, object]]) -> None:
    for stage, params in overrides.items():
        if stage not in base:
            base[stage] = {"params": {}}
        stage_entry = base[stage]
        if not isinstance(stage_entry, MutableMapping):
            raise TypeError(f"Stage entry for {stage} must be a mapping")
        stage_params = stage_entry.setdefault("params", {})
        if not isinstance(stage_params, MutableMapping):
            raise TypeError(f"Stage params for {stage} must be a mapping")
        stage_params.update(params)


def _instantiate_stage(stage_key: str, spec: Mapping[str, object], stage_type: str) -> Optional[T.Transform]:
    params = spec.get("params", {})
    antialias = bool(params.get("antialias", True))
    stage_key = stage_key.lower()
    stage_name = str(spec.get("name", stage_key)).lower()

    if stage_key == "resize" or stage_name in {"resize", "random_resized_crop"}:
        size = params.get("size", 224)
        interpolation = _resolve_interpolation(params.get("interpolation", "bicubic"))
        if stage_type == "train" and stage_name == "random_resized_crop":
            scale = tuple(params.get("scale", [0.8, 1.0]))
            ratio = tuple(params.get("ratio", [1.0, 1.0]))
            return T.RandomResizedCrop(
                size=size,
                scale=scale,
                ratio=ratio,
                interpolation=interpolation,
                antialias=antialias,
            )
        return T.Resize(size=(size, size), interpolation=interpolation, antialias=antialias)

    if stage_key == "hflip" or stage_name in {"hflip", "horizontal_flip", "random_horizontal_flip"}:
        p = float(params.get("p", 0.5))
        if p <= 0:
            return None
        return T.RandomHorizontalFlip(p=p)

    if stage_key == "affine" or stage_name in {"affine", "random_affine"}:
        degrees = params.get("degrees", 0)
        translate = _normalize_affine_arg(params.get("translate", [0.0, 0.0]))
        shear = _normalize_affine_arg(params.get("shear", [0.0, 0.0]))
        interpolation = _resolve_interpolation(params.get("interpolation", "bicubic"))
        if stage_type != "train" or (_is_zero_like(degrees) and _is_zero_like(translate) and _is_zero_like(shear)):
            if stage_type == "train" and (not _is_zero_like(degrees) or not _is_zero_like(translate) or not _is_zero_like(shear)):
                return T.RandomAffine(
                    degrees=_normalize_affine_arg(degrees),
                    translate=translate,
                    shear=shear,
                    interpolation=interpolation,
                )
            return None
        return T.RandomAffine(
            degrees=_normalize_affine_arg(degrees),
            translate=translate,
            shear=shear,
            interpolation=interpolation,
        )

    if stage_key == "color" or stage_name in {"color", "color_jitter"}:
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 0.0))
        saturation = float(params.get("saturation", 0.0))
        hue = float(params.get("hue", 0.0))
        if stage_type != "train" or (brightness == 0.0 and contrast == 0.0 and saturation == 0.0 and hue == 0.0):
            if stage_type == "train" and (brightness != 0 or contrast != 0 or saturation != 0 or hue != 0):
                return T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
            return None
        return T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    if stage_key == "blur" or stage_name in {"blur", "gaussian_blur"}:
        p = float(params.get("p", 0.0))
        if stage_type != "train" or p <= 0.0:
            return None
        kernel_size = int(params.get("kernel_size", 3))
        sigma = params.get("sigma", [0.1, 2.0])
        blur = T.GaussianBlur(kernel_size=kernel_size, sigma=tuple(sigma))
        return T.RandomApply([blur], p=p)

    if stage_key == "grayscale" or stage_name in {"grayscale", "random_grayscale"}:
        p = float(params.get("p", 0.0))
        if stage_type != "train" or p <= 0.0:
            return None
        return T.RandomGrayscale(p=p)

    if stage_key == "erasing" or stage_name in {"erasing", "random_erasing"}:
        p = float(params.get("p", 0.0))
        if stage_type != "train" or p <= 0.0:
            return None
        scale = tuple(params.get("scale", [0.02, 0.33]))
        ratio = tuple(params.get("ratio", [0.3, 3.3]))
        value = params.get("value", "random")
        return T.RandomErasing(p=p, scale=scale, ratio=ratio, value=value)

    if stage_key == "normalize" or stage_name == "normalize":
        mean = params.get("mean", [0.485, 0.456, 0.406])
        std = params.get("std", [0.229, 0.224, 0.225])
        return T.Normalize(mean=mean, std=std)

    raise ValueError(f"Unknown augmentation stage '{stage_name}' (key '{stage_key}')")


def build_pipeline(
    cfg: Mapping[str, object],
    stage: str = "train",
    combo: Optional[Sequence[str]] = None,
    extra_overrides: Optional[Mapping[str, Mapping[str, object]]] = None,
    dtype: torch.dtype = torch.float32,
) -> T.Compose:
    """Create a torchvision v2 Compose according to configuration.

    Args:
        cfg: augmentation configuration dictionary.
        stage: "train" or "eval".
        combo: optional list of combo tokens (e.g. ["hflip:p=0.5"]).
        extra_overrides: optional explicit overrides per stage.
        dtype: desired output dtype (float32 by default).
    """
    stage = stage.lower()
    if stage not in {"train", "eval", "test", "inference"}:
        raise ValueError(f"Unsupported stage '{stage}'")
    stage_type = "train" if stage == "train" else "eval"

    baseline = deepcopy(cfg["baseline"])

    overrides: Dict[str, Dict[str, object]] = {}
    if combo:
        overrides.update(_parse_combo_tokens(combo, cfg.get("search_space", {})))
    if extra_overrides:
        for key, value in extra_overrides.items():
            overrides.setdefault(key, {}).update(value)

    _merge_overrides(baseline, overrides)

    if stage_type != "train":
        # Disable stochastic components for evaluation/inference.
        if "hflip" in baseline:
            baseline["hflip"].setdefault("params", {})["p"] = 0.0
        if "affine" in baseline:
            params = baseline["affine"].setdefault("params", {})
            params.update({"degrees": 0, "translate": [0.0, 0.0], "shear": [0.0, 0.0]})
        if "color" in baseline:
            params = baseline["color"].setdefault("params", {})
            params.update({"brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0})
        if "blur" in baseline:
            baseline["blur"].setdefault("params", {})["p"] = 0.0
        if "grayscale" in baseline:
            baseline["grayscale"].setdefault("params", {})["p"] = 0.0
        if "erasing" in baseline:
            baseline["erasing"].setdefault("params", {})["p"] = 0.0
        # Force deterministic resize.
        if "resize" in baseline:
            baseline["resize"]["name"] = "resize"

    ordered_keys = list(baseline.keys())

    pre_tensor_ops = []
    post_tensor_ops = []

    for key in ordered_keys:
        spec = baseline[key]
        op = _instantiate_stage(key, spec, stage_type)
        if op is None:
            continue
        if isinstance(op, T.Normalize):
            post_tensor_ops.append(op)
        elif isinstance(op, T.RandomErasing):
            post_tensor_ops.append(op)
        else:
            pre_tensor_ops.append(op)

    # Ensure proper tensor conversion and dtype.
    to_image = T.ToImage()
    to_dtype = T.ToDtype(dtype, scale=True)

    pipeline = pre_tensor_ops + [to_image, to_dtype]

    # RandomErasing expects tensor input; insert before normalization if present.
    erasing_ops = [op for op in post_tensor_ops if isinstance(op, T.RandomErasing)]
    normalize_ops = [op for op in post_tensor_ops if isinstance(op, T.Normalize)]

    pipeline.extend(erasing_ops)
    pipeline.extend(normalize_ops)

    return T.Compose(pipeline)
