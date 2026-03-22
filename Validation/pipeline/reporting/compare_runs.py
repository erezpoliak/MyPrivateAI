"""Compare all experiment runs side-by-side with per-complexity breakdown.

Reads the latest completed run for each experiment from the SQLite DB,
renders a Rich table to the terminal, and exports a CSV to results/.

Usage:
    python -m reporting.compare_runs
    python -m reporting.compare_runs --run-ids 1 3 5 7 9 11
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Ensure pipeline root is importable when running as a script
# ---------------------------------------------------------------------------
_PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from common.config import Config
from common.db import RunDB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical experiment order for display
EXPERIMENT_ORDER = [
    "closed_book",
    "baseline",
    "phase1",
    "phase2",
    "llama_gold_ref",
    "gpt4o_rag",
]

# Human-friendly labels
EXPERIMENT_LABELS = {
    "closed_book": "Closed Book",
    "baseline": "Baseline (Fixed / Vector)",
    "phase1": "Phase 1 (Semantic / Hybrid)",
    "phase2": "Phase 2 (Agentic RAG)",
    "llama_gold_ref": "Llama + Gold Ref",
    "gpt4o_rag": "GPT-4o RAG (Ceiling)",
}

METRIC_KEYS = ["faithfulness", "context_recall", "answer_correctness"]

# Metrics that are meaningless for closed-book runs
CONTEXT_METRICS = {"faithfulness", "context_recall"}

# Experiments that use the agentic workflow (trajectory data is meaningful)
AGENTIC_EXPERIMENTS = {"phase2", "gpt4o_rag"}

# Metrics only meaningful for agentic experiments
TRAJECTORY_METRICS = {"avg_trajectory_steps", "trajectory_success_rate"}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fmt(value: float | None, dash_metrics: set[str] | None = None,
         metric: str = "") -> str:
    """Format a metric value. Returns '—' for None or suppressed metrics."""
    if dash_metrics and metric in dash_metrics:
        return "—"
    if value is None:
        return "—"
    return f"{value:.4f}"


def _latest_run_per_experiment(db: RunDB) -> dict[str, dict]:
    """Return the most recent finished run for each experiment name."""
    runs = db.list_runs()  # ordered by run_id DESC
    latest: dict[str, dict] = {}
    for run in runs:
        name = run["experiment"]
        if name not in latest and run["finished_at"] is not None:
            latest[name] = run
    return latest


def _gather_data(
    db: RunDB,
    run_ids: list[int] | None = None,
) -> list[dict]:
    """Collect run + summary data for comparison.

    If *run_ids* is provided, use exactly those runs.
    Otherwise, pick the latest finished run per experiment.
    """
    if run_ids:
        runs_by_id = {r["run_id"]: r for r in db.list_runs()}
        runs = [runs_by_id[rid] for rid in run_ids if rid in runs_by_id]
    else:
        latest = _latest_run_per_experiment(db)
        runs = [latest[name] for name in EXPERIMENT_ORDER if name in latest]

    rows: list[dict] = []
    for run in runs:
        summary = db.get_summary(run["run_id"])
        if summary is None:
            continue
        breakdown = json.loads(summary["breakdown_json"]) if summary.get("breakdown_json") else {}
        is_closed = run["corpus_mode"] == "none"
        is_agentic = run["experiment"] in AGENTIC_EXPERIMENTS
        rows.append({
            "run_id": run["run_id"],
            "experiment": run["experiment"],
            "label": EXPERIMENT_LABELS.get(run["experiment"], run["experiment"]),
            "corpus_mode": run["corpus_mode"],
            "num_questions": summary["num_questions"],
            "avg_faithfulness": summary["avg_faithfulness"],
            "avg_context_recall": summary["avg_context_recall"],
            "avg_answer_correctness": summary["avg_answer_correctness"],
            "avg_latency_s": summary["avg_latency_s"],
            "avg_trajectory_steps": summary.get("avg_trajectory_steps"),
            "trajectory_success_rate": summary.get("trajectory_success_rate"),
            "breakdown": breakdown,
            "is_closed": is_closed,
            "is_agentic": is_agentic,
            "notes": run.get("notes", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# Rich table
# ---------------------------------------------------------------------------

def build_overall_table(rows: list[dict]) -> Table:
    """Build a Rich table comparing overall metrics across experiments."""
    table = Table(
        title="Experiment Comparison — Overall",
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("Experiment", style="bold")
    table.add_column("Run", justify="right")
    table.add_column("N", justify="right")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Ctx Recall", justify="right")
    table.add_column("Ans Correct", justify="right")
    table.add_column("Avg Steps", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Latency (s)", justify="right")

    for r in rows:
        dash = CONTEXT_METRICS if r["is_closed"] else None
        traj_dash = TRAJECTORY_METRICS if not r["is_agentic"] else None
        table.add_row(
            r["label"],
            str(r["run_id"]),
            str(r["num_questions"]),
            _fmt(r["avg_faithfulness"], dash, "faithfulness"),
            _fmt(r["avg_context_recall"], dash, "context_recall"),
            _fmt(r["avg_answer_correctness"]),
            _fmt(r["avg_trajectory_steps"], traj_dash, "avg_trajectory_steps"),
            _fmt(r["trajectory_success_rate"], traj_dash, "trajectory_success_rate"),
            _fmt(r["avg_latency_s"]),
        )
    return table


def build_breakdown_table(rows: list[dict]) -> Table:
    """Build a Rich table with per-complexity breakdown for each experiment."""
    table = Table(
        title="Per-Complexity Breakdown",
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("Experiment", style="bold")
    table.add_column("Complexity", justify="center")
    table.add_column("N", justify="right")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Ctx Recall", justify="right")
    table.add_column("Ans Correct", justify="right")
    table.add_column("Avg Steps", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Latency (s)", justify="right")

    for r in rows:
        dash = CONTEXT_METRICS if r["is_closed"] else None
        traj_dash = TRAJECTORY_METRICS if not r["is_agentic"] else None
        breakdown = r["breakdown"]
        for i, (complexity, stats) in enumerate(sorted(breakdown.items())):
            label = r["label"] if i == 0 else ""
            table.add_row(
                label,
                str(complexity),
                str(stats["count"]),
                _fmt(stats.get("avg_faithfulness"), dash, "faithfulness"),
                _fmt(stats.get("avg_context_recall"), dash, "context_recall"),
                _fmt(stats.get("avg_answer_correctness")),
                _fmt(stats.get("avg_trajectory_steps"), traj_dash, "avg_trajectory_steps"),
                _fmt(stats.get("trajectory_success_rate"), traj_dash, "trajectory_success_rate"),
                _fmt(stats.get("avg_latency_s")),
            )
    return table


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(rows: list[dict], output_path: Path) -> None:
    """Write comparison data to a CSV file (overall + per-complexity)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment", "complexity", "n", "faithfulness",
        "context_recall", "answer_correctness",
        "avg_trajectory_steps", "trajectory_success_rate", "latency_s",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            dash = CONTEXT_METRICS if r["is_closed"] else set()
            traj_dash = TRAJECTORY_METRICS if not r["is_agentic"] else set()

            # Overall row
            writer.writerow({
                "experiment": r["experiment"],
                "complexity": "all",
                "n": r["num_questions"],
                "faithfulness": _fmt(r["avg_faithfulness"], dash, "faithfulness"),
                "context_recall": _fmt(r["avg_context_recall"], dash, "context_recall"),
                "answer_correctness": _fmt(r["avg_answer_correctness"]),
                "avg_trajectory_steps": _fmt(r["avg_trajectory_steps"], traj_dash, "avg_trajectory_steps"),
                "trajectory_success_rate": _fmt(r["trajectory_success_rate"], traj_dash, "trajectory_success_rate"),
                "latency_s": _fmt(r["avg_latency_s"]),
            })

            # Per-complexity rows
            for complexity, stats in sorted(r["breakdown"].items()):
                writer.writerow({
                    "experiment": r["experiment"],
                    "complexity": complexity,
                    "n": stats["count"],
                    "faithfulness": _fmt(stats.get("avg_faithfulness"), dash, "faithfulness"),
                    "context_recall": _fmt(stats.get("avg_context_recall"), dash, "context_recall"),
                    "answer_correctness": _fmt(stats.get("avg_answer_correctness")),
                    "avg_trajectory_steps": _fmt(stats.get("avg_trajectory_steps"), traj_dash, "avg_trajectory_steps"),
                    "trajectory_success_rate": _fmt(stats.get("trajectory_success_rate"), traj_dash, "trajectory_success_rate"),
                    "latency_s": _fmt(stats.get("avg_latency_s")),
                })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare experiment runs side-by-side.",
    )
    parser.add_argument(
        "--run-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific run IDs to compare. Default: latest run per experiment.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSV output path. Default: results/comparison.csv",
    )
    parser.add_argument(
        "--no-breakdown",
        action="store_true",
        default=False,
        help="Skip per-complexity breakdown table.",
    )
    args = parser.parse_args(argv)

    config = Config()
    db = RunDB(config)
    rows = _gather_data(db, args.run_ids)

    if not rows:
        print("No completed runs found in the database.")
        return

    console = Console()
    console.print()
    console.print(build_overall_table(rows))

    if not args.no_breakdown:
        console.print()
        console.print(build_breakdown_table(rows))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    csv_path = Path(args.csv) if args.csv else config.results_dir / f"comparison_{stamp}.csv"
    export_csv(rows, csv_path)
    console.print(f"\n[green]CSV exported to:[/green] {csv_path}")


if __name__ == "__main__":
    main()
