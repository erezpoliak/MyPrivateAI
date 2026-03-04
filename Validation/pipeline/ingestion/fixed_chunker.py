"""Fixed-size chunking strategy.

FixedSizeChunker — 512-token sliding window via tiktoken.
"""

from __future__ import annotations

from typing import Any

import tiktoken
from llama_index.core.schema import TextNode

from ..common.config import Config
from ..common.utils import get_logger

logger = get_logger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")


class FixedSizeChunker:
    """Split text into fixed-size token windows with overlap.

    Uses tiktoken's ``cl100k_base`` encoding to count tokens so chunk
    boundaries align with sub-word units rather than raw characters.
    """

    def __init__(self, config: Config | None = None) -> None:
        config = config or Config()
        self._chunk_size = config.fixed_chunk_size
        self._overlap = config.fixed_chunk_overlap
        self._enc = _ENC

    def chunk(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[TextNode]:
        """Return a list of TextNodes, each up to *chunk_size* tokens."""
        tokens = self._enc.encode(text)
        if not tokens:
            return []

        metadata = metadata or {}
        nodes: list[TextNode] = []
        start = 0

        while start < len(tokens):
            end = min(start + self._chunk_size, len(tokens))
            chunk_text = self._enc.decode(tokens[start:end])
            nodes.append(
                TextNode(
                    text=chunk_text,
                    metadata={
                        **metadata,
                        "chunk_strategy": "fixed",
                        "chunk_index": len(nodes),
                        "token_count": end - start,
                    },
                )
            )
            if end == len(tokens):
                break
            start += self._chunk_size - self._overlap

        logger.debug(
            "FixedSizeChunker produced %d chunks from %d tokens",
            len(nodes),
            len(tokens),
        )
        return nodes
