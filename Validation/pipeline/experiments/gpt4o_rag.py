"""GPT-4o RAG experiment — GPT-4o + same Phase 2 agentic pipeline.

Same retrieval pipeline as Phase 2 (fetched PDFs, semantic chunking,
hybrid BM25+vector RRF, FlashRank rerank, agent workflow).  The only
variable is the LLM: GPT-4o instead of Llama 3.1 8B.

Requires ``OPENAI_API_KEY`` environment variable.

CLI:
    python -m pipeline.experiments.gpt4o_rag                     # scored run (fetched PDFs)
    python -m pipeline.experiments.gpt4o_rag --smoke-test        # quick pipeline check (fetched, 5 Qs)
    python -m pipeline.experiments.gpt4o_rag --subset 20 --notes "test run"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever as LlamaBM25Retriever

from agent.workflow import run_agent_workflow
from common.config import Config
from common.data_loader import QAPair
from experiments.runner import parse_args, run_experiment
from experiments.spec import ExperimentSpec, GenerationResult, Retriever
from ingestion.semantic_chunker import CappedSemanticSplitter
from retrieval.hybrid_retriever import HybridRetriever


def _get_openai_llm(config: Config) -> OpenAI:
    """Return a configured GPT-4o LLM instance.

    Raises ``EnvironmentError`` if ``OPENAI_API_KEY`` is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is required for the GPT-4o experiment. "
            "Set it in your shell or in the .env file."
        )
    return OpenAI(
        model="gpt-4o",
        api_key=api_key,
        temperature=config.mlx_temperature,
    )


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
    name="gpt4o_rag",
    prompt_template="",  # unused — agent uses its own prompts internally
    collection_name="gpt4o_rag",
    create_chunker=lambda config, embed_model: CappedSemanticSplitter(embed_model, config),
    create_retriever=lambda index, nodes, config: HybridRetriever(
        index.as_retriever(similarity_top_k=config.vector_top_k),
        LlamaBM25Retriever.from_defaults(nodes=nodes, similarity_top_k=config.bm25_top_k),
        config,
    ),
    generate_answer=_agent_generate,
    create_llm=_get_openai_llm,
)


def main() -> None:
    args = parse_args(
        description="GPT-4o RAG experiment: GPT-4o + Phase 2 agentic pipeline.",
    )
    run_experiment(spec, args)


if __name__ == "__main__":
    main()
