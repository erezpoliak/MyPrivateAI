"""Unit tests for RRF fusion and HybridRetriever reranking logic."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import TextNode, NodeWithScore

from common.config import Config
from retrieval.hybrid_retriever import rrf_fuse, HybridRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _nws(node_id: str, score: float = 1.0) -> NodeWithScore:
    node = TextNode(text=f"text for {node_id}", id_=node_id)
    return NodeWithScore(node=node, score=score)


# ---------------------------------------------------------------------------
# rrf_fuse — pure function
# ---------------------------------------------------------------------------

class TestRrfFuse:

    def test_empty_both_lists_returns_empty(self):
        result = rrf_fuse([], [], _cfg())
        assert result == []

    def test_vector_only_returns_all_nodes(self):
        vector = [_nws("a"), _nws("b"), _nws("c")]
        result = rrf_fuse(vector, [], _cfg())
        ids = [n.node.node_id for n in result]
        assert set(ids) == {"a", "b", "c"}

    def test_bm25_only_returns_all_nodes(self):
        bm25 = [_nws("x"), _nws("y")]
        result = rrf_fuse([], bm25, _cfg())
        ids = [n.node.node_id for n in result]
        assert set(ids) == {"x", "y"}

    def test_overlap_node_accumulates_score(self):
        # node "a" appears rank-1 in both lists — should outscore nodes in only one list
        vector = [_nws("a"), _nws("b")]
        bm25 = [_nws("a"), _nws("c")]
        cfg = _cfg(rrf_k=60, rrf_vector_weight=0.6, rrf_bm25_weight=0.4)
        result = rrf_fuse(vector, bm25, cfg)
        top_id = result[0].node.node_id
        assert top_id == "a", "Overlapping node should have highest combined score"

    def test_result_sorted_descending_by_score(self):
        vector = [_nws("a"), _nws("b"), _nws("c")]
        bm25 = [_nws("c"), _nws("b"), _nws("a")]
        result = rrf_fuse(vector, bm25, _cfg())
        scores = [n.score for n in result]
        assert scores == sorted(scores, reverse=True)

    def test_output_scores_reflect_rrf_formula(self):
        cfg = _cfg(rrf_k=60, rrf_vector_weight=0.6, rrf_bm25_weight=0.4)
        vector = [_nws("a")]   # rank 1 vector
        bm25 = [_nws("b")]     # rank 1 bm25
        result = rrf_fuse(vector, bm25, cfg)
        node_scores = {n.node.node_id: n.score for n in result}

        expected_a = 0.6 / (60 + 1)
        expected_b = 0.4 / (60 + 1)
        assert abs(node_scores["a"] - expected_a) < 1e-9
        assert abs(node_scores["b"] - expected_b) < 1e-9

    def test_no_duplicate_nodes_in_output(self):
        # same node in both lists must appear exactly once
        vector = [_nws("dup"), _nws("vec_only")]
        bm25 = [_nws("dup"), _nws("bm25_only")]
        result = rrf_fuse(vector, bm25, _cfg())
        ids = [n.node.node_id for n in result]
        assert len(ids) == len(set(ids))

    def test_rank_position_affects_score(self):
        # rank-1 node gets higher score than rank-2 node in same list
        cfg = _cfg(rrf_k=60, rrf_vector_weight=0.6, rrf_bm25_weight=0.4)
        vector = [_nws("first"), _nws("second")]
        result = rrf_fuse(vector, [], cfg)
        node_scores = {n.node.node_id: n.score for n in result}
        assert node_scores["first"] > node_scores["second"]

    def test_custom_rrf_k_changes_scores(self):
        cfg_low_k = _cfg(rrf_k=1, rrf_vector_weight=1.0, rrf_bm25_weight=0.0)
        cfg_high_k = _cfg(rrf_k=100, rrf_vector_weight=1.0, rrf_bm25_weight=0.0)
        vector = [_nws("a")]
        score_low_k = rrf_fuse(vector, [], cfg_low_k)[0].score
        score_high_k = rrf_fuse(vector, [], cfg_high_k)[0].score
        # lower k → larger denominator offset → lower score
        assert score_low_k > score_high_k

    def test_disjoint_lists_union_all_nodes(self):
        vector = [_nws("v1"), _nws("v2")]
        bm25 = [_nws("b1"), _nws("b2")]
        result = rrf_fuse(vector, bm25, _cfg())
        assert len(result) == 4


# ---------------------------------------------------------------------------
# HybridRetriever — mocking retrievers and reranker
# ---------------------------------------------------------------------------

def _make_hybrid(
    vector_nodes: list[NodeWithScore],
    bm25_nodes: list[NodeWithScore],
    reranked_nodes: list[NodeWithScore] | None = None,
    cfg: Config | None = None,
) -> HybridRetriever:
    """Build HybridRetriever with mocked sub-retrievers and reranker."""
    config = cfg or _cfg()
    vector_retriever = MagicMock()
    vector_retriever.retrieve.return_value = vector_nodes
    bm25_retriever = MagicMock()
    bm25_retriever.retrieve.return_value = bm25_nodes

    with patch("retrieval.hybrid_retriever.FlashRankRerank") as mock_reranker_cls:
        mock_reranker = MagicMock()
        mock_reranker.postprocess_nodes.return_value = (
            reranked_nodes if reranked_nodes is not None else vector_nodes + bm25_nodes
        )
        mock_reranker_cls.return_value = mock_reranker
        hybrid = HybridRetriever(vector_retriever, bm25_retriever, config)
        hybrid._reranker = mock_reranker

    return hybrid


class TestHybridRetriever:

    def test_retrieve_calls_both_sub_retrievers(self):
        hybrid = _make_hybrid([_nws("a")], [_nws("b")])
        hybrid.retrieve("test query")
        hybrid._vector.retrieve.assert_called_once_with("test query")
        hybrid._bm25.retrieve.assert_called_once_with("test query")

    def test_retrieve_calls_reranker_with_fused_nodes(self):
        vector_nodes = [_nws("v1")]
        bm25_nodes = [_nws("b1")]
        hybrid = _make_hybrid(vector_nodes, bm25_nodes, reranked_nodes=[_nws("v1")])
        hybrid.retrieve("query")
        hybrid._reranker.postprocess_nodes.assert_called_once()

    def test_retrieve_returns_reranker_output(self):
        expected = [_nws("top1"), _nws("top2")]
        hybrid = _make_hybrid([_nws("a")], [_nws("b")], reranked_nodes=expected)
        result = hybrid.retrieve("query")
        assert result == expected

    def test_retrieve_empty_results_returns_empty(self):
        hybrid = _make_hybrid([], [], reranked_nodes=[])
        result = hybrid.retrieve("query")
        assert result == []

    def test_retrieve_skips_reranker_when_no_fused_candidates(self):
        hybrid = _make_hybrid([], [], reranked_nodes=[])
        hybrid.retrieve("query")
        hybrid._reranker.postprocess_nodes.assert_not_called()

    def test_uses_config_defaults_when_none_provided(self):
        # HybridRetriever(config=None) should use Config() defaults without error
        with patch("retrieval.hybrid_retriever.FlashRankRerank"):
            hr = HybridRetriever(MagicMock(), MagicMock(), config=None)
        assert hr._config.rrf_k == Config().rrf_k
