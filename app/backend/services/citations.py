"""Citation parsing: map [n] markers in an answer to their source chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass


_SNIPPET_LEN = 200


@dataclass(frozen=True)
class Source:
    chunk_index: int   # 1-based, matches [n] in answer
    doc_id: str
    title: str
    page_start: int | None
    page_end: int | None
    snippet: str


def parse_citations(answer: str, chunks: list[dict]) -> list[Source]:
    """Extract cited sources from *answer* by matching [n] markers to *chunks*.

    *chunks* is the list returned by ``AgentWorkflow.cited_chunks_for()``:
    index 0 corresponds to [1], index 1 to [2], etc.

    Only chunks whose [n] marker appears in the answer are returned, in
    ascending marker order, with duplicates removed.
    """
    indices = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)})

    sources: list[Source] = []
    for n in indices:
        i = n - 1
        if i < 0 or i >= len(chunks):
            continue
        chunk = chunks[i]
        text = chunk.get("text", "")
        snippet = text[:_SNIPPET_LEN].rstrip()
        if len(text) > _SNIPPET_LEN:
            snippet += "…"
        sources.append(Source(
            chunk_index=n,
            doc_id=chunk.get("doc_id", ""),
            title=chunk.get("title", ""),
            page_start=chunk.get("page_start"),
            page_end=chunk.get("page_end"),
            snippet=snippet,
        ))

    return sources
