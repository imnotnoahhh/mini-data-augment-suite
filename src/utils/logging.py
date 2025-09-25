"""Utility helpers for structured JSONL logging."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Mapping, MutableMapping, Optional


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def hash_config(config: Mapping[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class RunLogger:
    path: Path
    metadata: Mapping[str, object]
    file_handle: Optional[object] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = self.path.open("a", encoding="utf-8")
        self.write({"event": "start", **self.metadata})

    def write(self, record: Mapping[str, object]) -> None:
        if self.file_handle is None:
            raise RuntimeError("RunLogger not initialized")
        enriched = {"timestamp": timestamp(), **record}
        json.dump(enriched, self.file_handle, ensure_ascii=False)
        self.file_handle.write("\n")
        self.file_handle.flush()

    def log_metrics(self, epoch: int, split: str, metrics: Mapping[str, float]) -> None:
        self.write({
            "event": "metrics",
            "epoch": epoch,
            "split": split,
            "metrics": dict(metrics),
        })

    def finalize(self, summary: Mapping[str, object]) -> None:
        self.write({"event": "end", **summary})
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None


def create_run_logger(log_dir: Path, run_id: str, metadata: Mapping[str, object]) -> RunLogger:
    path = log_dir / f"{run_id}.jsonl"
    return RunLogger(path=path, metadata=metadata)
