"""Phase 2 experiment — agentic RAG with critique-driven multi-hop retrieval.

Pipeline: CappedSemanticSplitter → HybridRetriever → AgentWorkflow
         (decompose → retrieve → synthesize → critique → [correct])

CLI:
    python -m pipeline.experiments.phase2                     # scored run (fetched PDFs)
    python -m pipeline.experiments.phase2 --smoke-test        # quick pipeline check (gold_ref, 5 Qs)
    python -m pipeline.experiments.phase2 --subset 20 --notes "test run"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from agent.workflow import run_agent_workflow
from common.config import Config
from common.data_loader import QAPair
from experiments.runner import parse_args, run_experiment
from experiments.spec import ExperimentSpec, GenerationResult, Retriever
from ingestion.semantic_chunker import CappedSemanticSplitter
from llama_index.retrievers.bm25 import BM25Retriever as LlamaBM25Retriever
from retrieval.hybrid_retriever import HybridRetriever


def _agent_generate(
    qa: QAPair,
    retriever: Retriever,
    llm: Any,
    prompt_template: str,
    config: Config,
) -> GenerationResult:
    """Run the agentic workflow and map trajectory data onto GenerationResult."""
    result = run_agent_workflow(qa, retriever, llm, config)
    return GenerationResult(
        qa=result.generation.qa,
        generated_answer=result.generation.generated_answer,
        contexts=result.generation.contexts,
        latency_s=result.generation.latency_s,
        error=result.generation.error,
        trajectory_steps=result.trajectory.num_steps,
        trajectory_success=int(result.trajectory.success),
    )


spec = ExperimentSpec(
    name="phase2",
    prompt_template="",  # unused — agent uses its own prompts internally
    collection_name="phase2",
    create_chunker=lambda config, embed_model: CappedSemanticSplitter(embed_model, config),
    create_retriever=lambda index, nodes, config: HybridRetriever(
        index.as_retriever(similarity_top_k=config.vector_top_k),
        LlamaBM25Retriever.from_defaults(nodes=nodes, similarity_top_k=config.bm25_top_k),
        config,
    ),
    generate_answer=_agent_generate,
)


def main() -> None:
    args = parse_args(
        description="Phase 2 experiment: agentic RAG with critique-driven multi-hop retrieval.",
    )
    run_experiment(spec, args)


if __name__ == "__main__":
    main()
