"""Hybrid retriever combining vector and BM25 results via RRF + FlashRank.

Provides ``HybridRetriever`` which:

1. Queries both ``VectorRetriever`` (semantic) and ``BM25Retriever`` (lexical).
2. Fuses results with weighted Reciprocal Rank Fusion (RRF).
3. Reranks the fused candidates with FlashRank (cross-encoder).
4. Returns the top-N nodes after reranking.

"""

from __future__ import annotations

from flashrank import Ranker, RerankRequest

from llama_index.core.schema import NodeWithScore, TextNode

from ..common.config import Config
from ..common.utils import get_logger
from .bm25_retriever import BM25Retriever
from .vector_retriever import VectorRetriever

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Pure functions — independently testable
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
    # node_id → (accumulated_score, NodeWithScore)
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

    # Sort descending by fused score
    ranked = sorted(scores.values(), key=lambda x: x[0], reverse=True)

    return [
        NodeWithScore(node=nws.node, score=score)
        for score, nws in ranked
    ]


def rerank(
    query: str,
    candidates: list[NodeWithScore],
    ranker: Ranker,
) -> list[NodeWithScore]:
    """Rerank *candidates* using a FlashRank cross-encoder."""
    passages = [
        {"id": nws.node.node_id, "text": nws.node.get_content()}
        for nws in candidates
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    reranked_passages = ranker.rerank(rerank_request)

    # Build a lookup from node_id → NodeWithScore for reassembly
    node_map: dict[str, NodeWithScore] = {
        nws.node.node_id: nws for nws in candidates
    }

    return [
        NodeWithScore(
            node=node_map[p["id"]].node,
            score=float(p["score"]),
        )
        for p in reranked_passages
    ]


# ------------------------------------------------------------------
# HybridRetriever class
# ------------------------------------------------------------------

class HybridRetriever:
    """Hybrid search combining vector similarity and BM25 term matching.

    Usage::

        from pipeline.ingestion.document_store import DocumentStore
        from pipeline.retrieval.bm25_retriever import BM25Retriever
        from pipeline.retrieval.vector_retriever import VectorRetriever

        store = DocumentStore(config)
        index = store.load_index()

        vector_ret = VectorRetriever(index, config)
        bm25_ret = BM25Retriever(nodes, config)

        hybrid = HybridRetriever(vector_ret, bm25_ret, config)
        results = hybrid.retrieve("What is gradient descent?")
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        config: Config | None = None,
    ) -> None:
        self._config = config or Config()
        self._vector = vector_retriever
        self._bm25 = bm25_retriever

        # FlashRank cross-encoder reranker (runs on CPU — ONNX)
        self._ranker = Ranker(model_name=self._config.reranker_model)

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
        # 1. Retrieve from both sources
        vector_results = self._vector.retrieve(query)
        bm25_results = self._bm25.retrieve(query)

        # 2. Fuse via weighted RRF
        fused = rrf_fuse(vector_results, bm25_results, self._config)

        if not fused:
            logger.warning("RRF fusion produced zero candidates for %r", query)
            return []

        # 3. Rerank with FlashRank
        reranked = rerank(query, fused, self._ranker)

        # 4. Trim to top-N
        top_n = reranked[: self._config.hybrid_top_n]

        logger.debug(
            "Hybrid search for %r: %d vector + %d bm25 → %d fused → %d reranked",
            query,
            len(vector_results),
            len(bm25_results),
            len(fused),
            len(top_n),
        )
        return top_n
