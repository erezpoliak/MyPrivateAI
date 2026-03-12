"""Vector store backed by ChromaDB + LlamaIndex.

Provides ``DocumentStore`` which:
- Accepts a ``BaseEmbedding`` instance (initialized by the caller)
- ``build_index(nodes)`` — ingests TextNodes into a persistent ChromaDB
  collection and returns a ``VectorStoreIndex``
- ``load_index()`` — loads an existing ChromaDB collection into a
  ``VectorStoreIndex`` (no re-embedding required)
"""

from __future__ import annotations

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
from chromadb.errors import NotFoundError

from common.config import Config
from common.utils import ensure_dir, get_logger

logger = get_logger(__name__)

_DEFAULT_COLLECTION = "documents"


class DocumentStore:
    """Manage a LlamaIndex ``VectorStoreIndex`` backed by ChromaDB.

    Usage::

        store = DocumentStore(embed_model, config)

        # First run — build from corpus nodes
        index = store.build_index(nodes)

        # Subsequent runs — reload persisted collection
        index = store.load_index()

    The ``embed_model`` attribute is exposed so downstream retrieval code
    can reuse the same instance.
    """

    def __init__(
        self,
        embed_model: BaseEmbedding,
        config: Config | None = None,
        collection_name: str = _DEFAULT_COLLECTION,
    ) -> None:
        self._config = config or Config()
        self._collection_name = collection_name
        self.embed_model = embed_model
        self.index: VectorStoreIndex | None = None

    # -- index lifecycle ------------------------------------------------------

    def build_index(self, nodes: list[TextNode]) -> VectorStoreIndex:
        """Embed *nodes* and persist them in a ChromaDB collection.

        Any existing collection with the same name is dropped first so the
        index always reflects the current corpus.
        """
        ensure_dir(self._config.chroma_dir)

        client = chromadb.PersistentClient(path=str(self._config.chroma_dir))

        # Clear stale data — rebuild from scratch
        try:
            client.delete_collection(self._collection_name)
            logger.info("Deleted existing collection '%s'", self._collection_name)
        except (ValueError, NotFoundError):
            pass  # collection didn't exist yet

        collection = client.create_collection(self._collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

        logger.info("Building index from %d nodes …", len(nodes))
        self.index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_ctx,
            embed_model=self.embed_model,
            show_progress=True,
        )

        logger.info(
            "Index built and persisted (%d nodes in '%s')",
            len(nodes),
            self._collection_name,
        )
        return self.index

    def load_index(self) -> VectorStoreIndex:
        """Load a previously persisted ChromaDB collection into a VectorStoreIndex."""
        client = chromadb.PersistentClient(path=str(self._config.chroma_dir))
        collection = client.get_collection(self._collection_name)

        vector_store = ChromaVectorStore(chroma_collection=collection)

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model,
        )

        logger.info(
            "Index loaded from persisted collection '%s' (%d documents)",
            self._collection_name,
            collection.count(),
        )
        return self.index
