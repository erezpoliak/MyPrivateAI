"""Closed Book experiment — parametric knowledge only, no retrieval.

Sends each question directly to Llama 3.1 8B with no context.
GPT-4.1-mini as RAGAS judge. Only answer_correctness is computed;
context-dependent metrics are ``None``.

CLI:
    python -m pipeline.experiments.closed_book                     # full run
    python -m pipeline.experiments.closed_book --smoke-test        # quick check (5 Qs)
    python -m pipeline.experiments.closed_book --subset 20 --notes "test run"
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
    load_qa_pairs,
    parse_args,
    persist_and_report,
    setup_config,
)
from experiments.spec import GenerationResult

logger = get_logger(__name__)

EXPERIMENT_NAME = "closed_book"

CLOSED_BOOK_PROMPT = (
    "You are a knowledgeable research assistant.\n\n"
    "Question: {question}\n"
    "Answer: "
)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_closed_book(args) -> int:
    """Execute the closed-book experiment and return the run_id."""

    if args.smoke_test:
        args.subset = args.subset or 5
        args.notes = args.notes or "smoke-test"

    config = setup_config(EXPERIMENT_NAME, CorpusMode.NONE, args.subset)
    qa_pairs = load_qa_pairs(config, args.subset)

    if not qa_pairs:
        logger.error("No questions to evaluate — aborting.")
        return -1

    llm = get_llm(config, thinking=True)

    # ── Generate answers (no retrieval) ──────────────────────────────────
    gen_results: list[GenerationResult] = []

    for i, qa in enumerate(qa_pairs, 1):
        t0 = time.perf_counter()
        error = None
        answer = ""
        try:
            prompt = CLOSED_BOOK_PROMPT.format(question=qa.question)
            response = llm.complete(prompt)
            answer = response.text.strip()
        except Exception as exc:
            error = str(exc)
            logger.error("Question %s failed: %s", qa.id, error)

        gen_results.append(GenerationResult(
            qa=qa,
            generated_answer=answer,
            contexts=[],
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
        description="Closed Book experiment: parametric knowledge only, no retrieval.",
        corpus_choices=None,  # no --corpus-mode flag; always uses CorpusMode.NONE
    )
    run_closed_book(args)


if __name__ == "__main__":
    main()
