"""Hybrid retriever combining vector and BM25 results via RRF + FlashRank.

Provides ``HybridRetriever`` which:

1. Queries both a vector retriever (semantic) and BM25 retriever (lexical).
2. Fuses results with weighted Reciprocal Rank Fusion (RRF).
3. Reranks the fused candidates with FlashRankRerank (LlamaIndex built-in).
4. Returns the top-N nodes after reranking.
"""

from __future__ import annotations

from typing import Protocol

from typing import Any

from flashrank import Ranker as _Ranker
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.postprocessor.flashrank_rerank import FlashRankRerank


class _PersistentFlashRankRerank(FlashRankRerank):
    """FlashRankRerank with a configurable cache_dir so models survive reboots."""

    cache_dir: str = Field(default="/tmp")

    def model_post_init(self, context: Any, /) -> None:
        self._reranker = _Ranker(
            model_name=self.model,
            max_length=self.max_length,
            cache_dir=self.cache_dir,
        )

from common.config import Config
from common.utils import get_logger

logger = get_logger(__name__)


class _Retriever(Protocol):
    """Structural type for any object with a .retrieve(query) method."""

    def retrieve(self, query: str) -> list[NodeWithScore]: ...


# ------------------------------------------------------------------
# Pure function — independently testable
# ------------------------------------------------------------------

def rrf_fuse(
    vector_results: list[NodeWithScore],
    bm25_results: list[NodeWithScore],
    config: Config,
) -> list[NodeWithScore]:
    """Merge two ranked lists using weighted Reciprocal Rank Fusion.

    For each result at rank *r* (1-based), the contribution is::

        weight * 1 / (k + r)

    Nodes appearing in both lists accumulate scores from each.
    """
    k = config.rrf_k
    scores: dict[str, tuple[float, NodeWithScore]] = {}

    for rank, nws in enumerate(vector_results, start=1):
        node_id = nws.node.node_id
        rrf_score = config.rrf_vector_weight / (k + rank)
        prev_score, _ = scores.get(node_id, (0.0, nws))
        scores[node_id] = (prev_score + rrf_score, nws)

    for rank, nws in enumerate(bm25_results, start=1):
        node_id = nws.node.node_id
        rrf_score = config.rrf_bm25_weight / (k + rank)
        prev_score, prev_nws = scores.get(node_id, (0.0, nws))
        scores[node_id] = (prev_score + rrf_score, prev_nws)

    ranked = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return [
        NodeWithScore(node=nws.node, score=score)
        for score, nws in ranked
    ]


# ------------------------------------------------------------------
# HybridRetriever class
# ------------------------------------------------------------------

class HybridRetriever:
    """Hybrid search combining vector similarity and BM25 term matching.

    Usage::

        from llama_index_retrievers_bm25 import BM25Retriever as LlamaBM25Retriever

        vector_retriever = index.as_retriever(similarity_top_k=config.vector_top_k)
        bm25_retriever = LlamaBM25Retriever.from_defaults(nodes=nodes, similarity_top_k=config.bm25_top_k)

        hybrid = HybridRetriever(vector_retriever, bm25_retriever, config)
        results = hybrid.retrieve("What is gradient descent?")
    """

    def __init__(
        self,
        vector_retriever: _Retriever,
        bm25_retriever: _Retriever,
        config: Config | None = None,
    ) -> None:
        self._config = config or Config()
        self._vector = vector_retriever
        self._bm25 = bm25_retriever

        self._config.reranker_cache_dir.mkdir(parents=True, exist_ok=True)
        self._reranker = _PersistentFlashRankRerank(
            model=self._config.reranker_model,
            top_n=self._config.hybrid_top_n,
            cache_dir=str(self._config.reranker_cache_dir),
        )

        logger.info(
            "HybridRetriever initialised (rrf_k=%d, bm25_w=%.2f, "
            "vector_w=%.2f, top_n=%d, reranker=%s)",
            self._config.rrf_k,
            self._config.rrf_bm25_weight,
            self._config.rrf_vector_weight,
            self._config.hybrid_top_n,
            self._config.reranker_model,
        )

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """Return top-N reranked nodes for *query*."""
        vector_results = self._vector.retrieve(query)
        bm25_results = self._bm25.retrieve(query)

        fused = rrf_fuse(vector_results, bm25_results, self._config)

        if not fused:
            logger.warning("RRF fusion produced zero candidates for %r", query)
            return []

        reranked = self._reranker.postprocess_nodes(
            fused, query_bundle=QueryBundle(query_str=query)
        )

        logger.debug(
            "Hybrid search for %r: %d vector + %d bm25 -> %d fused -> %d reranked",
            query,
            len(vector_results),
            len(bm25_results),
            len(fused),
            len(reranked),
        )
        return reranked
