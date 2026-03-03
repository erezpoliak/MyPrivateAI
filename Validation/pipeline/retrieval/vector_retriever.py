"""Thin wrapper around LlamaIndex's VectorIndexRetriever.

Provides ``VectorRetriever`` which queries a ``VectorStoreIndex`` for the
top-k most similar nodes using the embedding model already attached to the
index.  The retrieval count defaults to ``Config.vector_top_k``.
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from ..common.config import Config
from ..common.utils import get_logger

logger = get_logger(__name__)


class VectorRetriever:
    """Retrieve nodes from a ``VectorStoreIndex`` by embedding similarity.

    Usage::

        from pipeline.ingestion.document_store import DocumentStore

        store = DocumentStore(config)
        index = store.load_index()

        retriever = VectorRetriever(index, config)
        results = retriever.retrieve("What is gradient descent?")
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        config: Config | None = None,
    ) -> None:
        self._config = config or Config()
        self._retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self._config.vector_top_k,
        )
        logger.info(
            "VectorRetriever initialised (top_k=%d)",
            self._config.vector_top_k,
        )

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """Return the top-k nodes most similar to *query*."""
        results = self._retriever.retrieve(QueryBundle(query_str=query))
        logger.debug(
            "Vector search for %r returned %d nodes",
            query,
            len(results),
        )
        return results
