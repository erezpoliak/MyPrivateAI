"""BM25 sparse retriever over NLTK-tokenized document nodes.

Provides ``BM25Retriever`` which builds a ``BM25Okapi`` index from a list of
LlamaIndex ``TextNode`` objects.  Each node's text is lowercased and tokenized
via NLTK ``word_tokenize``, with English stopwords removed.  At query time the
same preprocessing is applied before ranking.

The retrieval count defaults to ``Config.bm25_top_k``.
"""

from __future__ import annotations

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from ..common.config import Config
from ..common.utils import get_logger

logger = get_logger(__name__)

_STOPWORDS: set[str] = set(stopwords.words("english"))


def _tokenize(text: str) -> list[str]:
    """Lowercase, word-tokenize, and strip stopwords / non-alpha tokens."""
    return [
        tok
        for tok in word_tokenize(text.lower())
        if tok.isalpha() and tok not in _STOPWORDS
    ]


class BM25Retriever:
    """Retrieve nodes from a pre-built BM25 index by term overlap.

    Usage::

        from llama_index.core.schema import TextNode

        nodes = [TextNode(text="..."), ...]
        retriever = BM25Retriever(nodes, config)
        results = retriever.retrieve("What is gradient descent?")
    """

    def __init__(
        self,
        nodes: list[TextNode],
        config: Config | None = None,
    ) -> None:
        self._config = config or Config()
        self._nodes = nodes

        # Tokenize every node's text for BM25
        corpus = [_tokenize(node.get_content()) for node in nodes]
        self._bm25 = BM25Okapi(corpus)

        logger.info(
            "BM25Retriever initialised (%d nodes, top_k=%d)",
            len(nodes),
            self._config.bm25_top_k,
        )

    def retrieve(self, query: str) -> list[NodeWithScore]:
        """Return the top-k nodes most relevant to *query* by BM25 scoring."""
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Pair each node index with its BM25 score, sort descending
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        top_k = scored_indices[: self._config.bm25_top_k]

        results = [
            NodeWithScore(node=self._nodes[idx], score=float(score))
            for idx, score in top_k
            if score > 0.0
        ]

        logger.debug(
            "BM25 search for %r returned %d nodes",
            query,
            len(results),
        )
        return results
