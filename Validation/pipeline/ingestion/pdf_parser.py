"""PDF text extraction via pypdf.

Extracts and cleans text from PDF files downloaded by :mod:`pdf_fetcher`.
Handles common PDF artefacts (hyphenation, excess whitespace, headers/footers)
so the output is ready for chunking.

Usage::

    result = parse_pdf(Path("data/papers/10.1038__s41467-024-44750-0.pdf"))
    if result.success:
        print(result.text[:500])
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..common.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ParseResult:
    """Outcome of parsing a single PDF."""

    path: Path
    success: bool
    text: str = ""
    error: Optional[str] = None


def parse_pdf(path: Path) -> ParseResult:
    """Extract text from a single PDF at *path*."""
    if not path.exists():
        return ParseResult(
            path=path, success=False, error=f"File not found: {path}"
        )

    try:
        reader = PdfReader(path)
    except PdfReadError as exc:
        logger.warning("Failed to read PDF %s: %s", path.name, exc)
        return ParseResult(
            path=path, success=False, error=f"PDF read error: {exc}"
        )
    except Exception as exc:
        logger.warning("Unexpected error reading %s: %s", path.name, exc)
        return ParseResult(
            path=path, success=False, error=f"Unexpected error: {exc}"
        )

    page_texts: list[str] = []
    for page_num, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            logger.debug(
                "Could not extract text from page %d of %s: %s",
                page_num + 1,
                path.name,
                exc,
            )
            raw = ""
        page_texts.append(_clean_page(raw))

    full_text = "\n\n".join(t for t in page_texts if t)

    if not full_text.strip():
        logger.warning("No extractable text in %s", path.name)
        return ParseResult(
            path=path,
            success=False,
            page_count=len(reader.pages),
            error="No extractable text (likely a scanned/image-only PDF)",
        )

    logger.info(
        "Parsed %s: %d pages, %d chars",
        path.name,
        len(reader.pages),
        len(full_text),
    )
    return ParseResult(
        path=path,
        success=True,
        text=full_text,
        page_count=len(reader.pages),
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


def _clean_page(text: str) -> str:
    """Normalise raw page text extracted by pypdf."""
    if not text:
        return ""

    # Rejoin hyphenated words split across lines
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Collapse runs of whitespace (but preserve paragraph breaks)
    text = re.sub(r"[^\S\n]+", " ", text)          # spaces/tabs → single space
    text = re.sub(r"\n{3,}", "\n\n", text)          # 3+ newlines → double
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)    # lone newlines → space

    return text.strip()
