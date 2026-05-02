"""Unit tests for parse_citations."""

import pytest
from ..services.citations import parse_citations, Source


FAKE_CHUNKS = [
    {"text": "Alpha text about topic A.", "doc_id": "doc-1", "title": "Paper A", "page_start": 1, "page_end": 2},
    {"text": "Beta text about topic B.", "doc_id": "doc-2", "title": "Paper B", "page_start": 3, "page_end": 3},
    {"text": "Gamma text about topic C.", "doc_id": "doc-3", "title": "Paper C", "page_start": 5, "page_end": 7},
]


def test_returns_only_cited_chunks():
    answer = "According to [1], A is true. Also [3] confirms C."
    sources = parse_citations(answer, FAKE_CHUNKS)
    assert len(sources) == 2


def test_correct_doc_ids():
    answer = "According to [1], A is true. Also [3] confirms C."
    sources = parse_citations(answer, FAKE_CHUNKS)
    doc_ids = [s.doc_id for s in sources]
    assert "doc-1" in doc_ids
    assert "doc-3" in doc_ids


def test_absent_chunk_not_returned():
    answer = "According to [1], A is true. Also [3] confirms C."
    sources = parse_citations(answer, FAKE_CHUNKS)
    doc_ids = [s.doc_id for s in sources]
    assert "doc-2" not in doc_ids


def test_sources_ordered_by_marker():
    answer = "[3] says C. [1] says A."
    sources = parse_citations(answer, FAKE_CHUNKS)
    assert sources[0].chunk_index == 1
    assert sources[1].chunk_index == 3


def test_duplicate_markers_deduplicated():
    answer = "[1] and [1] again."
    sources = parse_citations(answer, FAKE_CHUNKS)
    assert len(sources) == 1


def test_out_of_range_marker_ignored():
    answer = "See [9] for details."
    sources = parse_citations(answer, FAKE_CHUNKS)
    assert sources == []


def test_no_markers_returns_empty():
    answer = "There are no citations here."
    sources = parse_citations(answer, FAKE_CHUNKS)
    assert sources == []


def test_snippet_truncated():
    long_chunk = {"text": "x" * 300, "doc_id": "doc-1", "title": "T", "page_start": 1, "page_end": 1}
    sources = parse_citations("[1] see here.", [long_chunk])
    assert len(sources[0].snippet) <= 201  # 200 chars + ellipsis
    assert sources[0].snippet.endswith("…")


def test_page_metadata_preserved():
    answer = "[2] confirms B."
    sources = parse_citations(answer, FAKE_CHUNKS)
    assert sources[0].page_start == 3
    assert sources[0].page_end == 3
