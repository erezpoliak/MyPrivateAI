"""Llama+Gold-REF experiment — Gold_REF as sole context, Llama 3.1 8B.

Injects each question's Gold_REF text as the only context and sends it to
Llama 3.1 8B via mlx-lm.  GPT-4.1-mini as RAGAS judge.  All RAGAS metrics
apply (context is present).  Corpus mode: ``gold_ref``.

CLI:
    python -m pipeline.experiments.llama_gold_ref                     # full run
    python -m pipeline.experiments.llama_gold_ref --smoke-test        # quick check (5 Qs)
    python -m pipeline.experiments.llama_gold_ref --subset 20 --notes "test run"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure pipeline root is importable when running as a script
# ---------------------------------------------------------------------------
_PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from common.config import CorpusMode
from common.llm import get_llm
from common.utils import get_logger
from experiments.runner import (
    evaluate_results,
    filter_to_corpus_coverage,
    load_qa_pairs,
    parse_args,
    persist_and_report,
    setup_config,
)
from experiments.spec import GenerationResult

logger = get_logger(__name__)

EXPERIMENT_NAME = "llama_gold_ref"

GOLD_REF_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Using the provided context, give the shortest complete answer possible. "
    "If the answer is truly absent from the context, say 'Not mentioned'.\n\n"
    "Question: {question}\n"
    "Answer: "
)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_llama_gold_ref(args) -> int:
    """Execute the Llama+Gold-REF experiment and return the run_id."""

    if args.smoke_test:
        args.subset = args.subset or 5
        args.notes = args.notes or "smoke-test"

    config = setup_config(EXPERIMENT_NAME, CorpusMode.GOLD_REF, args.subset)
    qa_pairs = load_qa_pairs(config, args.subset)
    qa_pairs = filter_to_corpus_coverage(qa_pairs, config)

    if not qa_pairs:
        logger.error("No questions to evaluate — aborting.")
        return -1

    llm = get_llm(config)

    # ── Generate answers (Gold_REF as sole context) ──────────────────────
    gen_results: list[GenerationResult] = []

    for i, qa in enumerate(qa_pairs, 1):
        t0 = time.perf_counter()
        error = None
        answer = ""
        try:
            prompt = GOLD_REF_PROMPT.format(
                context=qa.gold_ref,
                question=qa.question,
            )
            response = llm.complete(prompt)
            answer = response.text.strip()
        except Exception as exc:
            error = str(exc)
            logger.error("Question %s failed: %s", qa.id, error)

        gen_results.append(GenerationResult(
            qa=qa,
            generated_answer=answer,
            contexts=[qa.gold_ref],
            latency_s=time.perf_counter() - t0,
            error=error,
        ))

        if i % 10 == 0 or i == len(qa_pairs):
            logger.info("Generated %d/%d answers", i, len(qa_pairs))

    # ── Evaluate + persist ───────────────────────────────────────────────
    scored_rows = evaluate_results(gen_results, config)
    return persist_and_report(
        EXPERIMENT_NAME, config, gen_results, scored_rows, args.notes,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args(
        description="Llama+Gold-REF experiment: Gold_REF as sole context, Llama 3.1 8B.",
        corpus_choices=None,  # no --corpus-mode flag; always uses CorpusMode.GOLD_REF
    )
    run_llama_gold_ref(args)


if __name__ == "__main__":
    main()
