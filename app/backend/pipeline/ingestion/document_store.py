"""Vector store backed by ChromaDB + LlamaIndex."""

from __future__ import annotations

import json

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

from ..common.config import Config
from ..common.utils import ensure_dir, get_logger

logger = get_logger(__name__)

_DEFAULT_COLLECTION = "documents"


class DocumentStore:
    """Manage a LlamaIndex ``VectorStoreIndex`` backed by ChromaDB.

    App usage pattern::

        store = DocumentStore(embed_model, config)
        store.load_index()          # call once at startup (creates if absent)
        store.upsert_nodes(nodes)   # after each PDF ingestion
        store.delete_by_doc_id(id)  # on document delete
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
        self._chroma_collection: chromadb.Collection | None = None

    def load_index(self) -> VectorStoreIndex:
        """Load (or create) the ChromaDB collection and wrap it in a VectorStoreIndex.

        Safe to call on first startup when no documents have been ingested yet.
        """
        ensure_dir(self._config.chroma_dir)
        client = chromadb.PersistentClient(path=str(self._config.chroma_dir))
        self._chroma_collection = client.get_or_create_collection(self._collection_name)

        vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model,
        )

        logger.info(
            "Index loaded from '%s' (%d nodes)",
            self._collection_name,
            self._chroma_collection.count(),
        )
        return self.index

    def upsert_nodes(self, nodes: list[TextNode]) -> None:
        """Embed and insert *nodes* into the live index."""
        if self.index is None:
            raise RuntimeError("Call load_index() before upsert_nodes()")
        self.index.insert_nodes(nodes)
        logger.info("Upserted %d nodes into '%s'", len(nodes), self._collection_name)

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all nodes whose metadata ``doc_id`` matches *doc_id*."""
        if self._chroma_collection is None:
            raise RuntimeError("Call load_index() before delete_by_doc_id()")
        self._chroma_collection.delete(where={"doc_id": doc_id})
        logger.info("Deleted nodes for doc_id '%s' from '%s'", doc_id, self._collection_name)

    def get_all_nodes(self) -> list[TextNode]:
        """Return all TextNodes currently stored in the collection.

        Used by RetrieverService to rebuild the BM25 retriever after the corpus
        changes (document added or deleted).
        """
        if self._chroma_collection is None:
            raise RuntimeError("Call load_index() before get_all_nodes()")
        result = self._chroma_collection.get(include=["documents", "metadatas"])
        nodes: list[TextNode] = []
        for node_id, text, chroma_meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        ):
            # LlamaIndex serializes the full node into _node_content as JSON.
            # Parse it to recover the original metadata dict (doc_id, title, etc.).
            node_content = chroma_meta.get("_node_content") if chroma_meta else None
            if node_content:
                try:
                    meta = json.loads(node_content).get("metadata", {})
                except (json.JSONDecodeError, AttributeError):
                    meta = {}
            else:
                meta = {k: v for k, v in (chroma_meta or {}).items() if not k.startswith("_")}
            nodes.append(TextNode(id_=node_id, text=text or "", metadata=meta))
        return nodes
