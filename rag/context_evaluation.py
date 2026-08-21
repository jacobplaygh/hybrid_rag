"""Evaluation helpers for deterministic context selection."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ContextSelectionMetrics:
    """Summary metrics for one context selection report."""

    selected_count: int
    dropped_count: int
    duplicate_drop_rate: float
    source_diversity: int
    budget_utilization: float
    ordering: str


class ContextSelectionEvaluator:
    """Calculate quality and efficiency signals from a selection report."""

    @staticmethod
    def evaluate(report: Dict[str, Any]) -> ContextSelectionMetrics:
        selected: List[Dict[str, Any]] = report.get("selected", [])
        dropped: List[Dict[str, Any]] = report.get("dropped", [])
        available_tokens = max(int(report.get("available_tokens", 0)), 0)
        used_tokens = max(int(report.get("used_tokens", 0)), 0)
        duplicate_drops = sum(1 for item in dropped if item.get("reason") == "duplicate")
        selected_ids = {item.get("id") for item in selected}
        selected_sources = {item.get("source") for item in selected if item.get("source")}
        selected_sources = {
            item.get("source") for item in selected if item.get("source")
        }

        return ContextSelectionMetrics(
            selected_count=len(selected),
            dropped_count=len(dropped),
            duplicate_drop_rate=(duplicate_drops / len(dropped)) if dropped else 0.0,
            source_diversity=len(selected_sources),
            budget_utilization=(used_tokens / available_tokens) if available_tokens else 0.0,
            ordering=report.get("ordering", "retrieval_order"),
        )
