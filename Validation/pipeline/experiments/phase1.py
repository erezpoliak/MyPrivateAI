"""Phase 1 experiment — semantic chunking + hybrid retrieval.

Pipeline: CappedSemanticSplitter → HybridRetriever (RRF fusion + FlashRankRerank) → MLX LLM.

CLI:
    python -m pipeline.experiments.phase1                     # scored run (fetched PDFs)
    python -m pipeline.experiments.phase1 --smoke-test        # quick pipeline check (fetched, 5 Qs)
    python -m pipeline.experiments.phase1 --subset 20 --notes "test run"
"""

from __future__ import annotations

import sys
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from experiments.runner import parse_args, run_experiment
from experiments.spec import ExperimentSpec
from ingestion.semantic_chunker import CappedSemanticSplitter
from llama_index.retrievers.bm25 import BM25Retriever as LlamaBM25Retriever
from retrieval.hybrid_retriever import HybridRetriever

RAG_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Using the provided context, give the shortest complete answer possible. "
    "If the answer is truly absent from the context, say 'Not mentioned'.\n\n"
    "Question: {question}\n"
    "Answer: "
)


spec = ExperimentSpec(
    name="phase1",
    prompt_template=RAG_PROMPT,
    collection_name="phase1",
    create_chunker=lambda config, embed_model: CappedSemanticSplitter(embed_model, config),
    create_retriever=lambda index, nodes, config: HybridRetriever(
        index.as_retriever(similarity_top_k=config.vector_top_k),
        LlamaBM25Retriever.from_defaults(nodes=nodes, similarity_top_k=config.bm25_top_k),
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
