"""Experiment contract types — ExperimentSpec, protocols, and result dataclass.

Every experiment module defines an ``ExperimentSpec`` and delegates to
``run_experiment()`` for the common orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode

from common.config import Config
from common.data_loader import QAPair
from ingestion.corpus_builder import Chunker


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class Retriever(Protocol):
    """Structural type satisfied by VectorRetriever and HybridRetriever."""

    def retrieve(self, query: str) -> list[NodeWithScore]: ...


# ---------------------------------------------------------------------------
# Generation result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationResult:
    """Output of a single-question generation step."""

    qa: QAPair
    generated_answer: str
    contexts: list[str]
    latency_s: float
    error: str | None


GenerateFn = Callable[
    [QAPair, Retriever, Any, str, Config],
    GenerationResult,
]
"""Signature: (qa, retriever, llm, prompt_template, config) -> GenerationResult."""


# ---------------------------------------------------------------------------
# Experiment spec
# ---------------------------------------------------------------------------

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
