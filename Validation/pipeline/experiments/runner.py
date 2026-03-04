"""Shared experiment runner — generic lifecycle parameterised by ExperimentSpec.

Each experiment module defines an ``ExperimentSpec`` (chunker factory,
retriever factory, prompt, config overrides) and delegates to
``run_experiment()`` for the common orchestration:

    load dataset → build corpus → build index → generate answers
    → RAGAS evaluation → persist to SQLite → print report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Optional, Protocol

from llama_index.core.schema import NodeWithScore, TextNode, VectorStoreIndex

# ---------------------------------------------------------------------------
# Ensure pipeline root is importable when running as a script
# ---------------------------------------------------------------------------
_PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from common.config import Config, CorpusMode
from common.data_loader import QAPair, load_dataset
from common.db import RunDB
from common.llm import get_llm
from common.metrics import EvalSample, RAGASEvaluator
from common.utils import get_logger
from ingestion.corpus_builder import Chunker, CorpusBuilder
from ingestion.document_store import DocumentStore

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocols & spec
# ---------------------------------------------------------------------------

class Retriever(Protocol):
    """Structural type satisfied by VectorRetriever and HybridRetriever."""

    def retrieve(self, query: str) -> list[NodeWithScore]: ...


GenerateResult = dict[str, Any]
"""Return type for a single-question generation step.

Required keys: qa, generated_answer, contexts, latency_s, error.
Phase 2 may add: trajectory_steps, trajectory_success.
"""

GenerateFn = Callable[
    [QAPair, "Retriever", Any, str, Config],
    GenerateResult,
]
"""Signature: (qa, retriever, llm, prompt_template, config) -> GenerateResult."""


@dataclass(frozen=True)
class ExperimentSpec:
    """Declares the experiment-specific pieces injected into the shared runner."""

    name: str
    """Experiment name stored in the DB (e.g. 'baseline', 'phase1')."""

    prompt_template: str
    """RAG prompt with ``{context}`` and ``{question}`` placeholders."""

    collection_name: str
    """ChromaDB collection name for the vector index."""

    create_chunker: Callable[[Config], Chunker]
    """Factory: config → Chunker instance."""

    create_retriever: Callable[[VectorStoreIndex, list[TextNode], Config], Retriever]
    """Factory: (index, nodes, config) → Retriever instance.

    Receives *nodes* so hybrid experiments can build BM25Retriever from them.
    """

    apply_config_overrides: Callable[[Config], None] | None = None
    """Optional: mutate Config with experiment-specific knobs."""

    generate_answer: Optional[GenerateFn] = None
    """Optional: custom generation function for Phase 2 agentic loop.

    When ``None``, the default retrieve → prompt → complete loop is used.
    """


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(
    description: str = "Run an experiment.",
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Shared CLI parser used by all experiment entry points."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Quick pipeline check: sets corpus-mode to gold_ref and subset to 5.",
    )
    parser.add_argument(
        "--corpus-mode",
        type=str,
        choices=["gold_ref", "fetched"],
        default="fetched",
        help="Corpus source (default: fetched). Overridden to gold_ref by --smoke-test.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Run on first N questions only. Overridden to 5 by --smoke-test.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-text notes stored with the run.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(question: str, contexts: list[str], template: str) -> str:
    block = "\n\n".join(contexts) if contexts else "(no context retrieved)"
    return template.format(context=block, question=question)


def _safe_mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return mean(clean) if clean else None


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "—"


_METRIC_KEYS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


def _compute_summary(results: list[dict]) -> dict:
    """Aggregate metrics + per-complexity breakdown for DB persistence."""
    summary: dict = {
        "num_questions": len(results),
        "avg_latency_s": _safe_mean([r["latency_s"] for r in results]),
    }
    for key in _METRIC_KEYS:
        summary[f"avg_{key}"] = _safe_mean([r[key] for r in results])

    by_complexity: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_complexity[r["complexity"]].append(r)

    breakdown = {}
    for complexity, group in sorted(by_complexity.items()):
        breakdown[str(complexity)] = {
            "count": len(group),
            "avg_latency_s": _safe_mean([r["latency_s"] for r in group]),
            **{f"avg_{k}": _safe_mean([r[k] for r in group]) for k in _METRIC_KEYS},
        }

    summary["breakdown_json"] = json.dumps(breakdown)
    return summary


# ---------------------------------------------------------------------------
# Default generation loop
# ---------------------------------------------------------------------------

def _default_generate(
    qa: QAPair,
    retriever: Retriever,
    llm: Any,
    prompt_template: str,
    config: Config,
) -> GenerateResult:
    """Standard retrieve → prompt → complete loop."""
    t0 = time.perf_counter()
    try:
        retrieved = retriever.retrieve(qa.question)
        contexts = [n.node.text for n in retrieved]
        prompt = _build_prompt(qa.question, contexts, prompt_template)
        response = llm.complete(prompt)
        answer = response.text.strip()
        return {
            "qa": qa,
            "generated_answer": answer,
            "contexts": contexts,
            "latency_s": time.perf_counter() - t0,
            "error": None,
        }
    except Exception as exc:
        return {
            "qa": qa,
            "generated_answer": "",
            "contexts": [],
            "latency_s": time.perf_counter() - t0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(name: str, run_id: int, summary: dict) -> None:
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_experiment(spec: ExperimentSpec, args: argparse.Namespace) -> int:
    """Execute an experiment defined by *spec* and return the run_id."""

    # ── Smoke-test override ─────────────────────────────────────────────
    if args.smoke_test:
        args.corpus_mode = "gold_ref"
        args.subset = args.subset or 5
        args.notes = args.notes or "smoke-test"

    # ── Config ───────────────────────────────────────────────────────────
    config = Config()
    config.corpus_mode = CorpusMode(args.corpus_mode)
    if spec.apply_config_overrides is not None:
        spec.apply_config_overrides(config)

    logger.info(
        "Experiment: %s | corpus_mode=%s | subset=%s",
        spec.name,
        config.corpus_mode.value,
        args.subset,
    )

    # ── Load dataset ─────────────────────────────────────────────────────
    qa_pairs = load_dataset(config)
    if args.subset:
        qa_pairs = qa_pairs[: args.subset]
    logger.info("Loaded %d questions", len(qa_pairs))

    # ── Build corpus ─────────────────────────────────────────────────────
    chunker = spec.create_chunker(config)
    builder = CorpusBuilder(chunker, config)
    nodes, manifest = builder.build(qa_pairs)
    logger.info(
        "Corpus: %d nodes from %d papers",
        manifest.total_nodes,
        len(manifest.succeeded),
    )

    # In fetched mode, drop questions whose papers have no corpus
    if config.corpus_mode == CorpusMode.FETCHED:
        before = len(qa_pairs)
        qa_pairs = [q for q in qa_pairs if manifest.has_corpus(q.source_idx)]
        logger.info(
            "Filtered %d → %d questions (fetched-mode coverage)",
            before,
            len(qa_pairs),
        )

    if not qa_pairs:
        logger.error("No questions to evaluate — aborting.")
        return -1

    # ── Build vector index ───────────────────────────────────────────────
    store = DocumentStore(config, collection_name=spec.collection_name)
    index = store.build_index(nodes)
    logger.info("Vector index ready (%d nodes)", len(nodes))

    # ── LLM + Retriever ──────────────────────────────────────────────────
    llm = get_llm(config)
    retriever = spec.create_retriever(index, nodes, config)

    # ── Generate answers ─────────────────────────────────────────────────
    generate_fn = spec.generate_answer or _default_generate
    generation_results: list[dict] = []

    for i, qa in enumerate(qa_pairs, 1):
        result = generate_fn(qa, retriever, llm, spec.prompt_template, config)
        if result.get("error"):
            logger.error("Question %s failed: %s", qa.id, result["error"])
        generation_results.append(result)

        if i % 10 == 0 or i == len(qa_pairs):
            logger.info("Generated %d/%d answers", i, len(qa_pairs))

    # ── RAGAS evaluation ─────────────────────────────────────────────────
    logger.info("Running RAGAS evaluation …")
    successful = [r for r in generation_results if r["error"] is None]
    samples = [
        EvalSample(
            question=r["qa"].question,
            generated_answer=r["generated_answer"],
            reference_answer=r["qa"].answer,
            contexts=r["contexts"],
        )
        for r in successful
    ]
    eval_results = (
        RAGASEvaluator(config).evaluate_batch(samples) if samples else []
    )

    # Map evaluation scores back onto generation results
    eval_iter = iter(eval_results)
    for r in generation_results:
        if r["error"] is None:
            ev = next(eval_iter)
            r["faithfulness"] = ev.faithfulness
            r["answer_relevancy"] = ev.answer_relevancy
            r["context_precision"] = ev.context_precision
            r["context_recall"] = ev.context_recall
            r["answer_correctness"] = ev.answer_correctness
        else:
            for key in _METRIC_KEYS:
                r[key] = None

    # ── Persist to DB ────────────────────────────────────────────────────
    db = RunDB(config)
    config_snapshot = json.loads(json.dumps(config.__dict__, default=str))
    run_id = db.start_run(
        spec.name,
        config.corpus_mode.value,
        config_snapshot,
        args.notes,
    )
    logger.info("Run ID: %d", run_id)

    summary_rows: list[dict] = []
    for r in generation_results:
        qa = r["qa"]
        db.insert_result(
            run_id,
            question_id=qa.id,
            source_idx=qa.source_idx,
            complexity=qa.complexity,
            answer_type=qa.answer_type,
            generated_answer=r["generated_answer"],
            contexts_json=r["contexts"],
            faithfulness=r["faithfulness"],
            answer_relevancy=r["answer_relevancy"],
            context_precision=r["context_precision"],
            context_recall=r["context_recall"],
            answer_correctness=r["answer_correctness"],
            latency_s=r["latency_s"],
            error=r["error"],
        )
        summary_rows.append({
            "complexity": qa.complexity,
            "latency_s": r["latency_s"],
            "faithfulness": r["faithfulness"],
            "answer_relevancy": r["answer_relevancy"],
            "context_precision": r["context_precision"],
            "context_recall": r["context_recall"],
            "answer_correctness": r["answer_correctness"],
        })

    # ── Summary ──────────────────────────────────────────────────────────
    summary = _compute_summary(summary_rows)
    db.save_summary(run_id, summary)
    db.finish_run(run_id)

    # ── Print report ─────────────────────────────────────────────────────
    _print_report(spec.name, run_id, summary)

    return run_id
