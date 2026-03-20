"""Unit tests for agent retrieval tools: retrieve_context, merge_results, format_context."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.tools import RetrievalResult, format_context, merge_results, retrieve_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(node_id: str, text: str, score: float) -> MagicMock:
    """Build a mock NodeWithScore with the minimal interface retrieve_context needs."""
    node = MagicMock()
    node.node.get_content.return_value = text
    node.node.node_id = node_id
    node.score = score
    return node


def _make_retriever(*nodes) -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = list(nodes)
    return retriever


def _result(query: str, ids: list[str], texts: list[str], scores: list[float]) -> RetrievalResult:
    return RetrievalResult(query=query, texts=texts, scores=scores, node_ids=ids)


# ---------------------------------------------------------------------------
# retrieve_context
# ---------------------------------------------------------------------------

class TestRetrieveContext:

    def test_calls_retriever_with_query(self):
        retriever = _make_retriever()
        retrieve_context("what is X?", retriever)
        retriever.retrieve.assert_called_once_with("what is X?")

    def test_returns_retrieval_result(self):
        node = _make_node("n1", "some text", 0.9)
        retriever = _make_retriever(node)
        result = retrieve_context("query", retriever)
        assert isinstance(result, RetrievalResult)

    def test_texts_extracted_from_nodes(self):
        nodes = [_make_node("n1", "text one", 0.9), _make_node("n2", "text two", 0.8)]
        result = retrieve_context("q", _make_retriever(*nodes))
        assert result.texts == ["text one", "text two"]

    def test_scores_extracted_from_nodes(self):
        nodes = [_make_node("n1", "t", 0.95), _make_node("n2", "t", 0.7)]
        result = retrieve_context("q", _make_retriever(*nodes))
        assert result.scores == pytest.approx([0.95, 0.7])

    def test_node_ids_extracted_from_nodes(self):
        nodes = [_make_node("abc", "t", 1.0), _make_node("def", "t", 0.5)]
        result = retrieve_context("q", _make_retriever(*nodes))
        assert result.node_ids == ["abc", "def"]

    def test_query_stored_in_result(self):
        result = retrieve_context("my question", _make_retriever())
        assert result.query == "my question"

    def test_empty_retriever_returns_empty_result(self):
        result = retrieve_context("q", _make_retriever())
        assert result.texts == []
        assert result.scores == []
        assert result.node_ids == []


# ---------------------------------------------------------------------------
# merge_results
# ---------------------------------------------------------------------------

class TestMergeResults:

    def test_empty_list_returns_empty_result(self):
        merged = merge_results([])
        assert merged.texts == []
        assert merged.scores == []
        assert merged.node_ids == []

    def test_single_result_content_preserved(self):
        r = _result("q1", ["n1", "n2"], ["t1", "t2"], [0.9, 0.8])
        merged = merge_results([r])
        assert merged.texts == ["t1", "t2"]
        assert merged.scores == pytest.approx([0.9, 0.8])
        assert merged.node_ids == ["n1", "n2"]

    def test_duplicate_node_ids_deduplicated(self):
        r1 = _result("q1", ["n1", "n2"], ["t1", "t2"], [0.9, 0.8])
        r2 = _result("q2", ["n2", "n3"], ["t2", "t3"], [0.7, 0.6])
        merged = merge_results([r1, r2])
        assert merged.node_ids.count("n2") == 1

    def test_first_occurrence_of_duplicate_is_kept(self):
        r1 = _result("q1", ["dup"], ["first text"], [0.9])
        r2 = _result("q2", ["dup"], ["second text"], [0.5])
        merged = merge_results([r1, r2])
        assert merged.texts == ["first text"]
        assert merged.scores == pytest.approx([0.9])

    def test_disjoint_results_all_included(self):
        r1 = _result("q1", ["a", "b"], ["ta", "tb"], [1.0, 0.9])
        r2 = _result("q2", ["c", "d"], ["tc", "td"], [0.8, 0.7])
        merged = merge_results([r1, r2])
        assert set(merged.node_ids) == {"a", "b", "c", "d"}

    def test_order_preserved_first_hop_first(self):
        r1 = _result("q1", ["a"], ["ta"], [1.0])
        r2 = _result("q2", ["b"], ["tb"], [0.5])
        merged = merge_results([r1, r2])
        assert merged.node_ids[0] == "a"
        assert merged.node_ids[1] == "b"

    def test_query_is_semicolon_joined(self):
        r1 = _result("first query", ["n1"], ["t"], [1.0])
        r2 = _result("second query", ["n2"], ["t"], [0.9])
        merged = merge_results([r1, r2])
        assert merged.query == "first query; second query"

    def test_single_result_query_unchanged(self):
        r = _result("only query", ["n1"], ["t"], [1.0])
        merged = merge_results([r])
        assert merged.query == "only query"

    def test_total_unique_count_correct_with_overlap(self):
        r1 = _result("q1", ["a", "b", "c"], ["", "", ""], [1.0, 0.9, 0.8])
        r2 = _result("q2", ["b", "c", "d"], ["", "", ""], [0.7, 0.6, 0.5])
        merged = merge_results([r1, r2])
        assert len(merged.node_ids) == 4  # a, b, c, d


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------

class TestFormatContext:

    def test_empty_texts_returns_no_context_message(self):
        r = _result("q", [], [], [])
        assert format_context(r) == "(no context retrieved)"

    def test_single_text_returned_as_is(self):
        r = _result("q", ["n1"], ["only chunk"], [1.0])
        assert format_context(r) == "only chunk"

    def test_multiple_texts_joined_by_double_newline(self):
        r = _result("q", ["n1", "n2"], ["chunk one", "chunk two"], [1.0, 0.9])
        assert format_context(r) == "chunk one\n\nchunk two"

    def test_three_texts_all_joined(self):
        r = _result("q", ["n1", "n2", "n3"], ["a", "b", "c"], [1.0, 0.9, 0.8])
        assert format_context(r) == "a\n\nb\n\nc"
