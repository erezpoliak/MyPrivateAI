"""Semantic chunking via LlamaIndex SemanticSplitterNodeParser + token cap.

CappedSemanticSplitter composes SemanticSplitterNodeParser with a
SentenceSplitter to enforce a maximum token count per chunk
"""

from __future__ import annotations

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode, Document

from ..common.config import Config
from ..common.utils import get_logger

logger = get_logger(__name__)


class CappedSemanticSplitter:
    """Semantic split with a hard token cap on each resulting node.

    Steps:
    1. ``SemanticSplitterNodeParser`` splits on embedding-dissimilarity breakpoints.
    2. Any node exceeding ``config.semantic_max_tokens`` is sub-split by
       ``SentenceSplitter`` so nothing escapes the cap.
    """

    def __init__(self, embed_model: BaseEmbedding, config: Config | None = None) -> None:
        config = config or Config()
        self._semantic = SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=config.semantic_threshold_pct,
        )
        self._capper = SentenceSplitter(
            chunk_size=config.semantic_max_tokens,
            chunk_overlap=config.fixed_chunk_overlap,
        )

    def get_nodes_from_documents(
        self, documents: list[Document], **kwargs
    ) -> list[TextNode]:
        """Split *documents* semantically, then cap oversized nodes."""
        nodes = self._semantic.get_nodes_from_documents(documents, **kwargs)

        capped: list[TextNode] = []
        n_subsplit = 0
        for node in nodes:
            subs = self._capper.get_nodes_from_documents(
                [Document(text=node.get_content(), metadata=node.metadata)]
            )
            if len(subs) > 1:
                n_subsplit += 1
            capped.extend(subs)

        logger.debug(
            "CappedSemanticSplitter: %d -> %d nodes (%d sub-split)",
            len(nodes),
            len(capped),
            n_subsplit,
        )
        return capped
