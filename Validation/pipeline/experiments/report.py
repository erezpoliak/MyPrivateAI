"""Experiment summary computation and reporting."""

from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean

from common.utils import get_logger

logger = get_logger(__name__)

METRIC_KEYS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


def safe_mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return mean(clean) if clean else None


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "—"


def compute_summary(results: list[dict]) -> dict:
    """Aggregate metrics + per-complexity breakdown for DB persistence."""
    summary: dict = {
        "num_questions": len(results),
        "avg_latency_s": safe_mean([r["latency_s"] for r in results]),
    }
    for key in METRIC_KEYS:
        summary[f"avg_{key}"] = safe_mean([r[key] for r in results])

    by_complexity: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_complexity[r["complexity"]].append(r)

    breakdown = {}
    for complexity, group in sorted(by_complexity.items()):
        breakdown[str(complexity)] = {
            "count": len(group),
            "avg_latency_s": safe_mean([r["latency_s"] for r in group]),
            **{f"avg_{k}": safe_mean([r[k] for r in group]) for k in METRIC_KEYS},
        }

    summary["breakdown_json"] = json.dumps(breakdown)
    return summary


def print_report(name: str, run_id: int, summary: dict) -> None:
    logger.info("=" * 60)
    logger.info("%s RESULTS  (run_id=%d)", name.upper(), run_id)
    logger.info("=" * 60)
    logger.info("Questions evaluated:    %d", summary["num_questions"])
    logger.info("Avg latency:            %s s", _fmt(summary["avg_latency_s"]))
    logger.info("Avg faithfulness:       %s", _fmt(summary["avg_faithfulness"]))
    logger.info("Avg answer_relevancy:   %s", _fmt(summary["avg_answer_relevancy"]))
    logger.info("Avg context_precision:  %s", _fmt(summary["avg_context_precision"]))
    logger.info("Avg context_recall:     %s", _fmt(summary["avg_context_recall"]))
    logger.info("Avg answer_correctness: %s", _fmt(summary["avg_answer_correctness"]))
    logger.info("-" * 60)

    breakdown = json.loads(summary["breakdown_json"])
    for complexity, stats in sorted(breakdown.items()):
        logger.info(
            "  Complexity %s  (n=%d):  correctness=%s  faithfulness=%s  latency=%s s",
            complexity,
            stats["count"],
            _fmt(stats["avg_answer_correctness"]),
            _fmt(stats["avg_faithfulness"]),
            _fmt(stats["avg_latency_s"]),
        )
    logger.info("=" * 60)
