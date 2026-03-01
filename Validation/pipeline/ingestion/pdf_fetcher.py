"""Fetch open-access PDFs by DOI: Unpaywall API → Semantic Scholar fallback.

Rate-limited per source and cached to disk under ``config.papers_dir``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from ..common.config import Config
from ..common.utils import ensure_dir, get_logger, retry

logger = get_logger(__name__)


@dataclass(frozen=True)
class FetchResult:
    """Outcome of a single PDF fetch attempt."""

    doi: str
    success: bool
    path: Optional[Path] = None
    source: Optional[str] = None  # "unpaywall" | "semantic_scholar" | "cache"
    error: Optional[str] = None


class PDFFetcher:
    """Download open-access PDFs via Unpaywall and Semantic Scholar.

    Usage::

        fetcher = PDFFetcher(config)
        result = fetcher.fetch("10.1038/s41467-024-44750-0")
        if result.success:
            print(f"Saved to {result.path}")
    """

    _UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"
    _SEMANTIC_SCHOLAR_URL = (
        "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    )

    def __init__(self, config: Config | None = None) -> None:
        config = config or Config()
        self._papers_dir = ensure_dir(config.papers_dir)
        self._email = os.environ["UNPAYWALL_EMAIL"]
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": f"MyPrivateAI/1.0 (mailto:{self._email})",
        })
        # Per-source rate limiting
        self._last_request: dict[str, float] = {}
        # Unpaywall: ~1 req/sec; Semantic Scholar: 100/5min ≈ 1 req/3sec
        self._rate_limits: dict[str, float] = {
            "unpaywall": 1.0,
            "semantic_scholar": 3.0,
        }

    # -- Public API -----------------------------------------------------------

    def fetch(self, doi: str) -> FetchResult:
        """Fetch a PDF for *doi*. Returns a cached copy if already on disk."""
        cached = self._cached_path(doi)
        if cached.exists():
            logger.debug("Cache hit: %s", doi)
            return FetchResult(doi=doi, success=True, path=cached, source="cache")

        # Try Unpaywall first
        result = self._try_unpaywall(doi)
        if result.success:
            return result

        # Fallback to Semantic Scholar
        result = self._try_semantic_scholar(doi)
        if result.success:
            return result

        logger.warning("Failed to fetch PDF for DOI %s", doi)
        return FetchResult(
            doi=doi,
            success=False,
            error=result.error or "No open-access PDF found from any source",
        )

    def fetch_batch(self, dois: list[str]) -> list[FetchResult]:
        """Fetch PDFs for a list of DOIs, logging progress."""
        results: list[FetchResult] = []
        total = len(dois)
        for i, doi in enumerate(dois, 1):
            logger.info("[%d/%d] Fetching %s", i, total, doi)
            results.append(self.fetch(doi))

        succeeded = sum(1 for r in results if r.success)
        logger.info(
            "Batch complete: %d/%d succeeded (%.0f%%)",
            succeeded,
            total,
            100 * succeeded / total if total else 0,
        )
        return results

    # -- Source-specific fetchers ---------------------------------------------

    def _try_unpaywall(self, doi: str) -> FetchResult:
        """Query Unpaywall for an open-access PDF URL and download it."""
        try:
            self._wait("unpaywall")
            resp = self._session.get(
                self._UNPAYWALL_URL.format(doi=doi),
                params={"email": self._email},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            pdf_url = self._extract_unpaywall_pdf_url(data)
            if not pdf_url:
                return FetchResult(
                    doi=doi, success=False, error="Unpaywall: no OA PDF URL"
                )
            return self._download_pdf(doi, pdf_url, "unpaywall")
        except requests.RequestException as exc:
            return FetchResult(
                doi=doi, success=False, error=f"Unpaywall API error: {exc}"
            )

    def _try_semantic_scholar(self, doi: str) -> FetchResult:
        """Query Semantic Scholar for an open-access PDF URL and download it."""
        try:
            self._wait("semantic_scholar")
            resp = self._session.get(
                self._SEMANTIC_SCHOLAR_URL.format(doi=doi),
                params={"fields": "openAccessPdf"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            pdf_url = (data.get("openAccessPdf") or {}).get("url")
            if not pdf_url:
                return FetchResult(
                    doi=doi,
                    success=False,
                    error="Semantic Scholar: no OA PDF URL",
                )
            return self._download_pdf(doi, pdf_url, "semantic_scholar")
        except requests.RequestException as exc:
            return FetchResult(
                doi=doi,
                success=False,
                error=f"Semantic Scholar API error: {exc}",
            )

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _extract_unpaywall_pdf_url(data: dict) -> str | None:
        """Pull the best PDF URL from Unpaywall's JSON response."""
        best = data.get("best_oa_location")
        if best:
            url = best.get("url_for_pdf")
            if url:
                return url
        # Fallback: scan all OA locations
        for loc in data.get("oa_locations", []):
            url = loc.get("url_for_pdf")
            if url:
                return url
        return None

    @retry(max_attempts=3, delay=2.0, exceptions=(requests.RequestException,))
    def _download_pdf(self, doi: str, url: str, source: str) -> FetchResult:
        """Download a PDF from *url* and save it to the cache directory."""
        resp = self._session.get(url, timeout=60, stream=True)
        resp.raise_for_status()

        # Basic content-type validation
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and "octet-stream" not in content_type:
            return FetchResult(
                doi=doi,
                success=False,
                error=f"{source}: unexpected Content-Type {content_type!r}",
            )

        dest = self._cached_path(doi)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Downloaded %s via %s → %s", doi, source, dest.name)
        return FetchResult(doi=doi, success=True, path=dest, source=source)

    def _cached_path(self, doi: str) -> Path:
        """Return the local file path for a DOI's cached PDF."""
        return self._papers_dir / self._doi_to_filename(doi)

    @staticmethod
    def _doi_to_filename(doi: str) -> str:
        """Convert a DOI to a safe filename."""
        return doi.replace("/", "__").replace(":", "_") + ".pdf"

    def _wait(self, source: str) -> None:
        """Enforce per-source rate limits."""
        interval = self._rate_limits.get(source, 1.0)
        last = self._last_request.get(source, 0.0)
        elapsed = time.monotonic() - last
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request[source] = time.monotonic()
