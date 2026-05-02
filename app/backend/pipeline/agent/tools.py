"""Retrieval tools exposed to agent workflow steps.

Wraps the ``Retriever`` protocol (satisfied by ``HybridRetriever``,
``VectorIndexRetriever``, etc.) into structured helpers the agentic workflow
consumes at each hop.

Usage::

    from pipeline.retrieval.hybrid_retriever import HybridRetriever
    from pipeline.agent.tools import retrieve_context, format_context, merge_results

    retriever = HybridRetriever(vector_ret, bm25_ret, config)

    r1 = retrieve_context("What is X?", retriever)
    r2 = retrieve_context("How does X affect Y?", retriever)

    merged = merge_results([r1, r2])
    context_block = format_context(merged)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..common.utils import get_logger


class Retriever(Protocol):
    """Structural type for any object with a .retrieve(query) method."""

    def retrieve(self, query: str) -> list:
        ...

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalResult:
    """Structured output of a single retrieval call."""

    query: str
    texts: list[str]
    scores: list[float]
    node_ids: list[str]
    metadatas: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def retrieve_context(query: str, retriever: Retriever) -> RetrievalResult:
    """Run *retriever* on *query* and return a structured result.

    This is the single retrieval entry-point used by the workflow's
    retrieve step.  It converts ``NodeWithScore`` objects into plain
    strings / floats so downstream code never touches LlamaIndex types.
    """
    nodes = retriever.retrieve(query)

    result = RetrievalResult(
        query=query,
        texts=[n.node.get_content() for n in nodes],
        scores=[n.score for n in nodes],
        node_ids=[n.node.node_id for n in nodes],
        metadatas=[dict(n.node.metadata) for n in nodes],
    )
    logger.debug(
        "Retrieved %d chunks for %r (top score=%.4f)",
        len(result.texts),
        query,
        result.scores[0] if result.scores else 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Multi-hop helpers
# ---------------------------------------------------------------------------

def merge_results(results: list[RetrievalResult]) -> RetrievalResult:
    """Deduplicate and merge retrieval results across multiple hops.

    Chunks seen in earlier hops are kept; later duplicates (same
    ``node_id``) are dropped.  The merged query is a semicolon-joined
    concatenation of the individual queries (for logging only).
    """
    seen: set[str] = set()
    texts: list[str] = []
    scores: list[float] = []
    node_ids: list[str] = []
    metadatas: list[dict] = []

    for r in results:
        for nid, txt, score, meta in zip(r.node_ids, r.texts, r.scores, r.metadatas or [{}] * len(r.texts)):
            if nid not in seen:
                seen.add(nid)
                texts.append(txt)
                scores.append(score)
                node_ids.append(nid)
                metadatas.append(meta)

    merged_query = "; ".join(r.query for r in results)
    logger.debug(
        "Merged %d results → %d unique chunks",
        sum(len(r.texts) for r in results),
        len(texts),
    )
    return RetrievalResult(
        query=merged_query,
        texts=texts,
        scores=scores,
        node_ids=node_ids,
        metadatas=metadatas,
    )


def retrieve_multi(queries: list[str], retriever: Retriever) -> RetrievalResult:
    """Run retrieval for each query and return deduplicated merged results."""
    results = [retrieve_context(q, retriever) for q in queries]
    return merge_results(results)


def format_context(result: RetrievalResult) -> str:
    """Join retrieved texts into a numbered context block for prompt injection."""
    if not result.texts:
        return "(no context retrieved)"
    return "\n\n".join(f"[{i}] {text}" for i, text in enumerate(result.texts, 1))
