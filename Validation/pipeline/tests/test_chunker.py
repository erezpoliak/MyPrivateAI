"""Unit tests for fixed and semantic chunking strategies."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import Document, TextNode

from common.config import Config
from ingestion.fixed_chunker import FixedSizeChunker
from ingestion.semantic_chunker import CappedSemanticSplitter


def _doc(text: str) -> Document:
    return Document(text=text)


def _cfg(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------

class TestFixedSizeChunker:

    def test_long_document_splits_into_multiple_nodes(self):
        # ~300 words @ chunk_size=20 tokens → must produce multiple nodes
        chunker = FixedSizeChunker(_cfg(fixed_chunk_size=20, fixed_chunk_overlap=2))
        nodes = chunker.get_nodes_from_documents([_doc(" ".join(["word"] * 300))])
        assert len(nodes) > 1

    def test_short_document_produces_one_node(self):
        chunker = FixedSizeChunker(_cfg(fixed_chunk_size=512, fixed_chunk_overlap=50))
        nodes = chunker.get_nodes_from_documents([_doc("Hello world.")])
        assert len(nodes) == 1

    def test_nodes_have_non_empty_content(self):
        chunker = FixedSizeChunker(_cfg(fixed_chunk_size=10, fixed_chunk_overlap=0))
        nodes = chunker.get_nodes_from_documents([_doc(" ".join(["word"] * 100))])
        assert all(n.get_content().strip() for n in nodes)

    def test_multiple_documents_each_produce_a_node(self):
        chunker = FixedSizeChunker(_cfg(fixed_chunk_size=512, fixed_chunk_overlap=50))
        docs = [_doc("First document."), _doc("Second document.")]
        nodes = chunker.get_nodes_from_documents(docs)
        assert len(nodes) == 2


# ---------------------------------------------------------------------------
# CappedSemanticSplitter
# ---------------------------------------------------------------------------

def _make_capped_splitter(
    semantic_nodes: list[TextNode],
    max_tokens: int = 512,
) -> CappedSemanticSplitter:
    """Build a CappedSemanticSplitter with the semantic stage mocked out."""
    cfg = _cfg(
        semantic_max_tokens=max_tokens,
        fixed_chunk_overlap=0,
        semantic_threshold_pct=95,
    )
    with patch("ingestion.semantic_chunker.SemanticSplitterNodeParser"):
        splitter = CappedSemanticSplitter(embed_model=MagicMock(), config=cfg)
    splitter._semantic = MagicMock()
    splitter._semantic.get_nodes_from_documents.return_value = semantic_nodes
    return splitter


class TestCappedSemanticSplitter:

    def test_small_node_passes_through_unchanged(self):
        node = TextNode(text="Short text.")
        splitter = _make_capped_splitter([node], max_tokens=512)
        result = splitter.get_nodes_from_documents([_doc("Short text.")])
        assert len(result) == 1
        assert result[0].get_content() == "Short text."

    def test_oversized_node_gets_sub_split(self):
        # 200 words >> max_tokens=10 → capper must produce multiple nodes
        long_text = " ".join(["word"] * 200)
        splitter = _make_capped_splitter([TextNode(text=long_text)], max_tokens=10)
        result = splitter.get_nodes_from_documents([_doc(long_text)])
        assert len(result) > 1

    def test_all_result_nodes_have_non_empty_content(self):
        long_text = " ".join(["token"] * 100)
        splitter = _make_capped_splitter([TextNode(text=long_text)], max_tokens=10)
        result = splitter.get_nodes_from_documents([_doc(long_text)])
        assert all(n.get_content().strip() for n in result)

    def test_no_semantic_nodes_returns_empty_list(self):
        splitter = _make_capped_splitter([], max_tokens=512)
        result = splitter.get_nodes_from_documents([_doc("anything")])
        assert result == []

    def test_multiple_semantic_nodes_all_processed(self):
        nodes = [TextNode(text=f"Sentence {i}.") for i in range(5)]
        splitter = _make_capped_splitter(nodes, max_tokens=512)
        result = splitter.get_nodes_from_documents([_doc("ignored")])
        assert len(result) == 5
