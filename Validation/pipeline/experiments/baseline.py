"""Baseline experiment — vector-only RAG with fixed 512-token chunks.

Pipeline: FixedSizeChunker (512 tok) → VectorRetriever (k=5) → Ollama LLM.

CLI:
    python -m pipeline.experiments.baseline                     # scored run (fetched PDFs)
    python -m pipeline.experiments.baseline --smoke-test        # quick pipeline check (gold_ref, 5 Qs)
    python -m pipeline.experiments.baseline --subset 20 --notes "test run"
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure pipeline root is importable when running as a script
# ---------------------------------------------------------------------------
_PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from common.config import Config
from experiments.runner import ExperimentSpec, parse_args, run_experiment
from ingestion.fixed_chunker import FixedSizeChunker
from retrieval.vector_retriever import VectorRetriever

# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------

RAG_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Using the provided context, answer the following question. "
    "If the context does not contain enough information, say so.\n\n"
    "Question: {question}\n"
    "Answer: "
)


def _apply_overrides(config: Config) -> None:
    config.vector_top_k = 5


spec = ExperimentSpec(
    name="baseline",
    prompt_template=RAG_PROMPT,
    collection_name="baseline",
    create_chunker=lambda config: FixedSizeChunker(config),
    create_retriever=lambda index, nodes, config: VectorRetriever(index, config),
    apply_config_overrides=_apply_overrides,
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args(
        description="Baseline experiment: vector-only RAG with fixed 512-token chunks.",
    )
    run_experiment(spec, args)


if __name__ == "__main__":
    main()
