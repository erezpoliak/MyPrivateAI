"""Fixed-size and semantic chunking strategies.

FixedSizeChunker  — 512-token sliding window via tiktoken.
SemanticChunker   — Splits on cosine-dissimilarity spikes (95th percentile)
                    using SentenceTransformer embeddings on GPU when available.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tiktoken
from llama_index.core.schema import TextNode
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from ..common.config import Config
from ..common.utils import get_logger

logger = get_logger(__name__)


# ── Fixed-size chunker ───────────────────────────────────────────────────────


class FixedSizeChunker:
    """Split text into fixed-size token windows with overlap.

    Uses tiktoken's ``cl100k_base`` encoding to count tokens so chunk
    boundaries align with sub-word units rather than raw characters.
    """

    def __init__(self, config: Config | None = None) -> None:
        config = config or Config()
        self._chunk_size = config.fixed_chunk_size
        self._overlap = config.fixed_chunk_overlap
        self._enc = tiktoken.get_encoding("cl100k_base")

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


# ── Semantic chunker ─────────────────────────────────────────────────────────


class SemanticChunker:
    """Split text at semantic boundaries detected by embedding dissimilarity.

    Steps:
    1. Sentence-tokenize with NLTK.
    2. Embed each sentence with SentenceTransformer (GPU when available).
    3. Compute cosine dissimilarity between consecutive embeddings.
    4. Split where dissimilarity exceeds the configured percentile threshold.
    """

    def __init__(self, config: Config | None = None) -> None:
        config = config or Config()
        self._threshold_pct = config.semantic_threshold_pct
        self._device = config.device

        logger.info(
            "Loading SentenceTransformer %s on %s",
            config.embedding_model,
            self._device,
        )
        self._model = SentenceTransformer(
            config.embedding_model, device=self._device
        )

    def chunk(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[TextNode]:
        """Return TextNodes split at semantic boundaries."""
        sentences = sent_tokenize(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [
                TextNode(
                    text=sentences[0],
                    metadata={
                        **(metadata or {}),
                        "chunk_strategy": "semantic",
                        "chunk_index": 0,
                    },
                )
            ]

        # Embed all sentences in one batch
        embeddings = self._model.encode(
            sentences, show_progress_bar=False, convert_to_numpy=True
        )

        # Cosine dissimilarity between consecutive sentence embeddings
        dissimilarities = self._cosine_dissimilarity(embeddings)

        # Threshold: split where dissimilarity exceeds the Nth percentile
        threshold = float(np.percentile(dissimilarities, self._threshold_pct))

        # Find split points (indices into the *gap* between sentences)
        split_indices = [
            i + 1
            for i, d in enumerate(dissimilarities)
            if d >= threshold
        ]

        # Build chunks from sentence groups
        metadata = metadata or {}
        groups = self._split_at(sentences, split_indices)
        nodes: list[TextNode] = []
        for idx, group in enumerate(groups):
            nodes.append(
                TextNode(
                    text=" ".join(group),
                    metadata={
                        **metadata,
                        "chunk_strategy": "semantic",
                        "chunk_index": idx,
                        "sentence_count": len(group),
                    },
                )
            )

        logger.debug(
            "SemanticChunker produced %d chunks from %d sentences "
            "(threshold=%.4f at p%d)",
            len(nodes),
            len(sentences),
            threshold,
            self._threshold_pct,
        )
        return nodes

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_dissimilarity(embeddings: np.ndarray) -> np.ndarray:
        """Return 1 − cos(e_i, e_{i+1}) for consecutive embedding pairs."""
        # Normalise rows to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        normed = embeddings / norms

        # Cosine similarity = dot product of adjacent normalised vectors
        cos_sim = np.sum(normed[:-1] * normed[1:], axis=1)
        return 1.0 - cos_sim

    @staticmethod
    def _split_at(items: list[str], indices: list[int]) -> list[list[str]]:
        """Split *items* into groups at the given indices."""
        groups: list[list[str]] = []
        prev = 0
        for idx in indices:
            groups.append(items[prev:idx])
            prev = idx
        groups.append(items[prev:])
        return groups
