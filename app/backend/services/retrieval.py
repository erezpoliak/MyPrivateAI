"""Lazy-rebuilding hybrid retriever service."""

from __future__ import annotations

from llama_index.retrievers.bm25 import BM25Retriever as LlamaBM25Retriever

from ..pipeline.common.config import Config
from ..pipeline.common.utils import get_logger
from ..pipeline.ingestion.document_store import DocumentStore
from ..pipeline.retrieval.hybrid_retriever import HybridRetriever

logger = get_logger(__name__)


class RetrieverService:
    """HybridRetriever with lazy rebuild on invalidation.

    Call ``invalidate()`` after any corpus change (document added or deleted).
    Call ``get()`` to obtain the current retriever — rebuilds from ChromaDB if stale.
    Returns ``None`` when the corpus is empty so callers can respond appropriately.
    """

    def __init__(self, store: DocumentStore, config: Config | None = None) -> None:
        self._store = store
        self._config = config or Config()
        self._retriever: HybridRetriever | None = None
        self._stale = True

    def invalidate(self) -> None:
        """Mark the cached retriever stale so the next ``get()`` rebuilds it."""
        self._stale = True
        self._retriever = None
        logger.debug("RetrieverService invalidated")

    def get(self) -> HybridRetriever | None:
        """Return the current retriever, rebuilding if stale.

        Returns ``None`` when the corpus has no documents.
        """
        if not self._stale and self._retriever is not None:
            return self._retriever
        self._retriever = self._build()
        self._stale = False
        return self._retriever

    def _build(self) -> HybridRetriever | None:
        """Reconstruct the HybridRetriever from the current ChromaDB state."""
        if self._store.index is None:
            logger.warning("DocumentStore index not loaded — cannot build retriever")
            return None

        nodes = self._store.get_all_nodes()
        if not nodes:
            logger.info("Corpus is empty — RetrieverService returning None")
            return None

        vector_retriever = self._store.index.as_retriever(
            similarity_top_k=self._config.vector_top_k
        )
        bm25_retriever = LlamaBM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=self._config.bm25_top_k,
        )

        logger.info("RetrieverService rebuilt HybridRetriever (%d nodes)", len(nodes))
        return HybridRetriever(vector_retriever, bm25_retriever, self._config)
