"""Phase 1 experiment — semantic chunking + hybrid retrieval.

Pipeline: SemanticChunker → HybridRetriever (RRF fusion + FlashRank rerank) → Ollama LLM.

CLI:
    python -m pipeline.experiments.phase1                     # scored run (fetched PDFs)
    python -m pipeline.experiments.phase1 --smoke-test        # quick pipeline check (gold_ref, 5 Qs)
    python -m pipeline.experiments.phase1 --subset 20 --notes "test run"
"""

from __future__ import annotations

import sys
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from common.config import Config
from experiments.runner import parse_args, run_experiment
from experiments.spec import ExperimentSpec
from ingestion.semantic_chunker import SemanticChunker
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_retriever import VectorRetriever

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


spec = ExperimentSpec(
    name="phase1",
    prompt_template=RAG_PROMPT,
    collection_name="phase1",
    create_chunker=lambda config: SemanticChunker(config),
    create_retriever=lambda index, nodes, config: HybridRetriever(
        VectorRetriever(index, config),
        BM25Retriever(nodes, config),
        config,
    ),
)


def main() -> None:
    args = parse_args(
        description="Phase 1 experiment: semantic chunking + hybrid retrieval.",
    )
    run_experiment(spec, args)


if __name__ == "__main__":
    main()
