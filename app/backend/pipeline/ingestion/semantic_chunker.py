"""Semantic chunking via LlamaIndex SemanticSplitterNodeParser + token cap.

CappedSemanticSplitter composes SemanticSplitterNodeParser with a
SentenceSplitter to enforce a maximum token count per chunk
"""

from __future__ import annotations

import bisect

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
            chunk_overlap=config.chunk_overlap,
        )

    def get_nodes_from_documents(
        self, documents: list[Document], **kwargs
    ) -> list[TextNode]:
        """Split *documents* semantically, then cap oversized nodes."""
        nodes = self._semantic.get_nodes_from_documents(documents, **kwargs)

        capped: list[TextNode] = []
        n_subsplit = 0
        for node in nodes:
            parent_start = node.start_char_idx or 0
            subs = self._capper.get_nodes_from_documents(
                [Document(text=node.get_content(), metadata=node.metadata)]
            )
            if len(subs) > 1:
                n_subsplit += 1
            # Sub-node char indices are relative to the node text; make them
            # absolute so page lookup works against the source document offsets.
            for sub in subs:
                if sub.start_char_idx is not None:
                    sub.start_char_idx += parent_start
                if sub.end_char_idx is not None:
                    sub.end_char_idx += parent_start
            capped.extend(subs)

        # Attach 1-based page_start / page_end to each node by binary-searching
        # page_offsets (inherited from Document.metadata via the splitter).
        for node in capped:
            offsets: list[int] | None = node.metadata.get("page_offsets")
            if not offsets:
                continue
            start = node.start_char_idx or 0
            end = (node.end_char_idx - 1) if node.end_char_idx else (start + len(node.get_content()) - 1)
            node.metadata["page_start"] = bisect.bisect_right(offsets, start)
            node.metadata["page_end"] = bisect.bisect_right(offsets, end)

        logger.debug(
            "CappedSemanticSplitter: %d -> %d nodes (%d sub-split)",
            len(nodes),
            len(capped),
            n_subsplit,
        )
        return capped
