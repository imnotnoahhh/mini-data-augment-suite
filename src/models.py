"""Model factory and optimizer utilities for the augmentation experiments."""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    convnext_tiny,
    resnet18,
    resnet50,
    vit_t_16,
)
from torchvision.models.vision_transformer import ViT_T_16_Weights

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CFG = ROOT / "configs" / "models.yaml"


@dataclass
class SchedulerFactory:
    """Callable container that builds a scheduler given runtime metadata."""

    name: str
    phase: str
    factory: Callable[[optim.Optimizer, int, int], LRScheduler]


@dataclass
class ModelBundle:
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler_factory: SchedulerFactory
    amp_dtype: torch.dtype
    max_epochs: int
    config: Mapping[str, object]


def load_model_config(path: Path = DEFAULT_MODEL_CFG) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_resnet(arch: str, num_classes: int, dropout: float, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    constructor = resnet18
    if arch == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        constructor = resnet50
    model = constructor(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_convnext(num_classes: int, dropout: float, pretrained: bool) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.LayerNorm(in_features, eps=1e-6),
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_vit(num_classes: int, dropout: float, pretrained: bool) -> nn.Module:
    weights = ViT_T_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_t_16(weights=weights)
    embed_dim = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, num_classes),
    )
    if dropout > 0:
        model.heads.head.insert(1, nn.Dropout(dropout))  # type: ignore[arg-type]
    return model


def _set_memory_format(model: nn.Module, channels_last: bool) -> None:
    if not channels_last:
        return
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            module.to(memory_format=torch.channels_last)


def _group_parameters_for_llrd(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    decay_rate: float,
    min_lr_scale: float,
) -> List[Dict[str, object]]:
    parameter_groups: Dict[Tuple[int, bool], Dict[str, object]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        depth = name.count(".")
        scale = decay_rate ** depth
        scale = max(scale, min_lr_scale)
        lr = base_lr * scale
        no_decay = param.ndim <= 1
        key = (depth, no_decay)
        group = parameter_groups.setdefault(
            key,
            {
                "params": [],
                "lr": lr,
                "weight_decay": 0.0 if no_decay else weight_decay,
            },
        )
        group["params"].append(param)
    return list(parameter_groups.values())


def _make_scheduler_factory(arch_cfg: Mapping[str, object], phase: str) -> SchedulerFactory:
    sched_cfg = arch_cfg["scheduler"][phase]
    name = sched_cfg["name"].lower()

    def cosine_factory(optimizer: optim.Optimizer, epochs: int, steps_per_epoch: int) -> LRScheduler:
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))
        if warmup_epochs == 0:
            warmup_steps = 0
        else:
            warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = max(1, epochs * steps_per_epoch)
        cosine_steps = max(1, total_steps - warmup_steps)
        if warmup_steps > 0:
            warmup = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=sched_cfg.get("warmup_start_factor", 1e-3),
                total_iters=warmup_steps,
            )
            schedulers: List[LRScheduler] = [warmup]
            milestones = [warmup_steps]
        else:
            schedulers = []
            milestones = []
        eta_min = sched_cfg.get("eta_min")
        if eta_min is None:
            base_lr = optimizer.param_groups[0]["lr"]
            eta_min = base_lr * sched_cfg.get("eta_min_ratio", 1e-3)
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min)
        schedulers.append(cosine)
        if not milestones:
            return cosine
        return optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

    def onecycle_factory(optimizer: optim.Optimizer, epochs: int, steps_per_epoch: int) -> LRScheduler:
        max_lr = [group["lr"] for group in optimizer.param_groups]
        total_steps = epochs * steps_per_epoch
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=sched_cfg.get("pct_start", 0.1),
            div_factor=sched_cfg.get("div_factor", 25.0),
            final_div_factor=sched_cfg.get("final_div_factor", 25.0),
            anneal_strategy=sched_cfg.get("anneal_strategy", "cos"),
            three_phase=False,
        )

    if name == "cosine":
        return SchedulerFactory(name="cosine", phase=phase, factory=cosine_factory)
    if name == "onecycle":
        return SchedulerFactory(name="onecycle", phase=phase, factory=onecycle_factory)
    raise ValueError(f"Unsupported scheduler '{name}' for phase '{phase}'")


def create_model(
    architecture: str,
    num_classes: int,
    phase: str,
    global_batch_size: int,
    optimizer_cfg: Mapping[str, object],
    model_config_path: Path = DEFAULT_MODEL_CFG,
    channels_last: bool = True,
    compile_model: bool = True,
    compile_mode: str = "reduce-overhead",
    device: Optional[torch.device] = None,
) -> ModelBundle:
    """Instantiate model, optimizer, and scheduler factory for a given architecture."""
    cfg = load_model_config(model_config_path)
    common_cfg = cfg["common"]
    arch_cfg = deepcopy(cfg["architectures"][architecture])

    family = arch_cfg["family"]
    dropout = float(arch_cfg.get("dropout", 0.0))
    pretrained = bool(common_cfg.get("pretrained", True))

    if family == "convnet":
        model = _build_resnet(architecture, num_classes, dropout=dropout, pretrained=pretrained)
    elif family == "convnext":
        model = _build_convnext(num_classes, dropout=dropout, pretrained=pretrained)
    elif family == "vit":
        model = _build_vit(num_classes, dropout=dropout, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model family '{family}'")

    if channels_last and family in {"convnet", "convnext"}:
        _set_memory_format(model, channels_last=True)

    if device is not None:
        model = model.to(device)

    base_lr = float(optimizer_cfg.get("base_lr", 1e-3)) * (global_batch_size / 256.0)
    if family == "vit":
        base_lr *= 0.8

    llrd_cfg = common_cfg.get("llrd", {"decay": 1.0, "min_lr_scale": 1.0})
    decay_rate = float(llrd_cfg.get("decay", 1.0))
    min_lr_scale = float(llrd_cfg.get("min_lr_scale", 1.0))
    weight_decay = float(arch_cfg["optimizer"].get("weight_decay", optimizer_cfg.get("weight_decay", 0.05)))

    param_groups = _group_parameters_for_llrd(
        model,
        base_lr=base_lr,
        weight_decay=weight_decay,
        decay_rate=decay_rate,
        min_lr_scale=min_lr_scale,
    )

    betas = tuple(optimizer_cfg.get("betas", [0.9, 0.999]))
    eps = float(optimizer_cfg.get("eps", 1e-8))

    optimizer = optim.AdamW(param_groups, lr=base_lr, betas=betas, eps=eps)

    scheduler_factory = _make_scheduler_factory(arch_cfg, phase)

    amp_dtype_name = common_cfg.get("amp_dtype", "bfloat16")
    amp_dtype = torch.bfloat16 if amp_dtype_name.lower() == "bfloat16" else torch.float16

    if compile_model:
        try:
            model = torch.compile(model, mode=compile_mode)
        except Exception as exc:  # pragma: no cover - compilation fallback
            warnings.warn(f"torch.compile failed for {architecture}: {exc}")

    max_epochs = int(arch_cfg["scheduler"][phase].get("epochs", 1))

    return ModelBundle(
        model=model,
        optimizer=optimizer,
        scheduler_factory=scheduler_factory,
        amp_dtype=amp_dtype,
        max_epochs=max_epochs,
        config={
            "architecture": architecture,
            "family": family,
            "base_lr": base_lr,
            "weight_decay": weight_decay,
            "phase": phase,
            "batch_size_recommend": arch_cfg.get("batch_size", {}),
        },
    )
