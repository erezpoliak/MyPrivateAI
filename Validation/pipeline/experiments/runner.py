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
from pathlib import Path
from typing import Any

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
from experiments.report import METRIC_KEYS, compute_summary, print_report
from experiments.spec import ExperimentSpec, GenerationResult, Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ingestion.corpus_builder import build_corpus
from ingestion.document_store import DocumentStore

logger = get_logger(__name__)


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


# ---------------------------------------------------------------------------
# Default generation loop
# ---------------------------------------------------------------------------

def _default_generate(
    qa: QAPair,
    retriever: Retriever,
    llm: Any,
    prompt_template: str,
    config: Config,
) -> GenerationResult:
    """Standard retrieve → prompt → complete loop."""
    t0 = time.perf_counter()
    try:
        retrieved = retriever.retrieve(qa.question)
        contexts = [n.node.text for n in retrieved]
        prompt = _build_prompt(qa.question, contexts, prompt_template)
        response = llm.complete(prompt)
        answer = response.text.strip()
        return GenerationResult(
            qa=qa,
            generated_answer=answer,
            contexts=contexts,
            latency_s=time.perf_counter() - t0,
            error=None,
        )
    except Exception as exc:
        return GenerationResult(
            qa=qa,
            generated_answer="",
            contexts=[],
            latency_s=time.perf_counter() - t0,
            error=str(exc),
        )


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

    # ── Shared embedding model ───────────────────────────────────────────
    embed_model = HuggingFaceEmbedding(
        model_name=config.embedding_model,
        device=config.device,
    )
    logger.info("Shared embed_model: %s on %s", config.embedding_model, config.device)

    # ── Build corpus ─────────────────────────────────────────────────────
    chunker = spec.create_chunker(config, embed_model)
    nodes, manifest = build_corpus(qa_pairs, chunker, config)
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
    store = DocumentStore(config, collection_name=spec.collection_name, embed_model=embed_model)
    index = store.build_index(nodes)
    logger.info("Vector index ready (%d nodes)", len(nodes))

    # ── LLM + Retriever ──────────────────────────────────────────────────
    llm = get_llm(config)
    retriever = spec.create_retriever(index, nodes, config)

    # ── Generate answers ─────────────────────────────────────────────────
    generate_fn = spec.generate_answer or _default_generate
    gen_results: list[GenerationResult] = []

    for i, qa in enumerate(qa_pairs, 1):
        result = generate_fn(qa, retriever, llm, spec.prompt_template, config)
        if result.error:
            logger.error("Question %s failed: %s", qa.id, result.error)
        gen_results.append(result)

        if i % 10 == 0 or i == len(qa_pairs):
            logger.info("Generated %d/%d answers", i, len(qa_pairs))

    # ── RAGAS evaluation ─────────────────────────────────────────────────
    logger.info("Running RAGAS evaluation …")
    successful = [r for r in gen_results if r.error is None]
    samples = [
        EvalSample(
            question=r.qa.question,
            generated_answer=r.generated_answer,
            reference_answer=r.qa.answer,
            contexts=r.contexts,
        )
        for r in successful
    ]
    eval_results = (
        RAGASEvaluator(config).evaluate_batch(samples) if samples else []
    )

    # Build scored rows: merge generation results with eval scores
    eval_iter = iter(eval_results)
    scored_rows: list[dict] = []
    for r in gen_results:
        if r.error is None:
            ev = next(eval_iter)
            scores = {
                "faithfulness": ev.faithfulness,
                "answer_relevancy": ev.answer_relevancy,
                "context_precision": ev.context_precision,
                "context_recall": ev.context_recall,
                "answer_correctness": ev.answer_correctness,
            }
        else:
            scores = {k: None for k in METRIC_KEYS}

        scored_rows.append({
            "complexity": r.qa.complexity,
            "latency_s": r.latency_s,
            **scores,
        })

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

    for r, row in zip(gen_results, scored_rows):
        db.insert_result(
            run_id,
            question_id=r.qa.id,
            source_idx=r.qa.source_idx,
            complexity=r.qa.complexity,
            answer_type=r.qa.answer_type,
            generated_answer=r.generated_answer,
            contexts_json=r.contexts,
            faithfulness=row["faithfulness"],
            answer_relevancy=row["answer_relevancy"],
            context_precision=row["context_precision"],
            context_recall=row["context_recall"],
            answer_correctness=row["answer_correctness"],
            latency_s=r.latency_s,
            error=r.error,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    summary = compute_summary(scored_rows)
    db.save_summary(run_id, summary)
    db.finish_run(run_id)

    # ── Print report ─────────────────────────────────────────────────────
    print_report(spec.name, run_id, summary)

    return run_id
