"""
验证/测试评估入口。

需与 Trainer 共用日志规范，确保 metrics.csv 中的列一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class EvaluatorConfig:
    metrics: Dict[str, str]


class Evaluator:
    def __init__(self, config: EvaluatorConfig):
        self.config = config

    def evaluate(self, dataloader: Iterable) -> Dict[str, float]:
        """
        TODO:
        - 执行验证，返回 top-1/top-5/macro-F1/loss；
        - 支持 EMA 权重切换；
        - 与 Trainer 协同写日志。
        """
        raise NotImplementedError("Evaluator.evaluate is pending implementation.")
