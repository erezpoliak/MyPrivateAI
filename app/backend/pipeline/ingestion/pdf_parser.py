"""PDF text extraction via pymupdf4llm.

Extracts text from PDF files downloaded by :mod:`pdf_fetcher`, preserving
table structure as markdown. Handles multi-column academic paper layouts
better than plain text extraction.

Usage::

    result = parse_pdf(Path("data/papers/10.1038__s41467-024-44750-0.pdf"))
    if result.success:
        print(result.text[:500])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf
import pymupdf4llm

from ..common.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ParseResult:
    """Outcome of parsing a single PDF."""

    path: Path
    success: bool
    text: str = ""
    error: Optional[str] = None
    page_count: int = 0
    page_offsets: list[int] = field(default_factory=list)


def parse_pdf(path: Path) -> ParseResult:
    """Extract markdown text from a single PDF at *path*."""
    if not path.exists():
        return ParseResult(
            path=path, success=False, error=f"File not found: {path}"
        )

    try:
        doc = pymupdf.open(str(path))
        page_count = len(doc)
        doc.close()
        chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    except Exception as exc:
        logger.warning("Failed to parse PDF %s: %s", path.name, exc)
        return ParseResult(
            path=path, success=False, error=f"Parse error: {exc}"
        )

    # Build combined text page-by-page, recording the char offset where each
    # page begins so downstream chunkers can map spans back to page numbers.
    _SEP = "\n\n"
    page_texts: list[str] = []
    page_offsets: list[int] = []
    running = 0
    for chunk in chunks:
        pt = _clean_markdown(chunk["text"])
        page_offsets.append(running)
        page_texts.append(pt)
        running += len(pt) + len(_SEP)

    text = _SEP.join(page_texts)

    if not text.strip():
        logger.warning("No extractable text in %s", path.name)
        return ParseResult(
            path=path,
            success=False,
            page_count=page_count,
            error="No extractable text (likely a scanned/image-only PDF)",
        )

    logger.info(
        "Parsed %s: %d pages, %d chars",
        path.name,
        page_count,
        len(text),
    )
    return ParseResult(
        path=path,
        success=True,
        text=text,
        page_count=page_count,
        page_offsets=page_offsets,
    )


def parse_all_pdfs(papers_dir: Path) -> list[ParseResult]:
    """Parse every PDF in *papers_dir*, logging progress."""
    paths = sorted(papers_dir.glob("*.pdf"))
    if not paths:
        logger.warning("No PDFs found in %s", papers_dir)
        return []

    results: list[ParseResult] = []
    total = len(paths)
    for i, path in enumerate(paths, 1):
        logger.info("[%d/%d] Parsing %s", i, total, path.name)
        results.append(parse_pdf(path))

    succeeded = sum(1 for r in results if r.success)
    logger.info(
        "Batch parse complete: %d/%d succeeded (%.0f%%)",
        succeeded,
        total,
        100 * succeeded / total if total else 0,
    )
    return results


def _clean_markdown(text: str) -> str:
    """Normalise markdown output from pymupdf4llm."""
    if not text:
        return ""

    # Collapse runs of blank lines (keep max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
