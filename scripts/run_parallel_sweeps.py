#!/usr/bin/env python3
"""Launch augmentation sweeps with per-seed parallelism control."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]


def parse_seed_list(raw: Sequence[str]) -> List[int]:
    values: List[int] = []
    for token in raw:
        pieces = token.replace(",", " ").split()
        for piece in pieces:
            if not piece:
                continue
            values.append(int(piece))
    if not values:
        raise ValueError("At least one seed is required")
    return sorted(set(values))


def determine_parallel(seed_count: int, default_parallel: int) -> int:
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if seed_count == 3:
        return 3
    if seed_count == 10:
        return 2
    return min(seed_count, default_parallel)


def build_command(args: argparse.Namespace, op: str, seed: int) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "src.search",
        "--mode",
        "sweep",
        "--dataset",
        args.dataset,
        "--architecture",
        args.architecture,
        "--phase",
        args.phase,
        "--kshot",
        str(args.kshot),
        "--seed",
        str(seed),
        "--op",
        op,
    ]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.device:
        cmd += ["--device", args.device]
    if args.suffix:
        cmd += ["--suffix", args.suffix]
    if args.train_config:
        cmd += ["--train-config", str(args.train_config)]
    if args.model_config:
        cmd += ["--model-config", str(args.model_config)]
    if args.dataset_config:
        cmd += ["--dataset-config", str(args.dataset_config)]
    if args.augment_config:
        cmd += ["--augment-config", str(args.augment_config)]
    if args.hardware_config:
        cmd += ["--hardware-config", str(args.hardware_config)]
    if args.extra_args:
        cmd += list(args.extra_args)
    return cmd


def run_commands(commands: Iterable[List[str]], max_parallel: int, cwd: Path) -> None:
    queue: deque[List[str]] = deque(commands)
    running: List[tuple[subprocess.Popen[bytes], List[str]]] = []
    try:
        while queue or running:
            while queue and len(running) < max_parallel:
                cmd = queue.popleft()
                print(f"[launcher] start: {' '.join(shlex.quote(part) for part in cmd)}")
                proc = subprocess.Popen(cmd, cwd=cwd)
                running.append((proc, cmd))

            if not running:
                continue

            for idx, (proc, cmd) in enumerate(running):
                ret = proc.poll()
                if ret is None:
                    continue
                running.pop(idx)
                status = "ok" if ret == 0 else f"exit {ret}"
                print(f"[launcher] done: {' '.join(shlex.quote(part) for part in cmd)} -> {status}")
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, cmd)
                break
            else:
                time.sleep(1.0)
    except Exception:
        for proc, cmd in running:
            if proc.poll() is None:
                print(f"[launcher] terminating: {' '.join(shlex.quote(part) for part in cmd)}")
                proc.terminate()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run per-seed sweep jobs in parallel")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--architecture", default="resnet18")
    parser.add_argument("--phase", default="explore")
    parser.add_argument("--kshot", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2"], help="Seed list; accepts space or comma separated values")
    parser.add_argument(
        "--ops",
        nargs="+",
        default=[
            "hflip",
            "rotate",
            "translate",
            "shear",
            "rrc_scale",
            "brightness",
            "contrast",
            "saturation",
            "hue",
            "blur",
            "blur_sigma",
            "grayscale",
            "erasing",
            "erasing_scale",
        ],
        help="Operators to sweep",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--suffix")
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs" / "train.yaml")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--dataset-config", type=Path, default=ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--augment-config", type=Path, default=ROOT / "configs" / "augment" / "aug_search.yaml")
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "configs" / "hardware.yaml")
    parser.add_argument("--default-parallel", type=int, default=3, help="Fallback parallelism when no preset applies")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Additional flags passed to src.search")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    parallel = determine_parallel(len(seeds), args.default_parallel)

    for op in args.ops:
        print(f"[launcher] op={op} seeds={seeds} parallel={parallel}")
        commands = [build_command(args, op, seed) for seed in seeds]
        run_commands(commands, parallel, ROOT)


if __name__ == "__main__":
    main()

