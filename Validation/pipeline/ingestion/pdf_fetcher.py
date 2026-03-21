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

from common.config import Config
from common.utils import doi_to_path, ensure_dir, get_logger, retry

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
    _OPENALEX_URL = "https://api.openalex.org/works/doi:{doi}"
    _EUROPE_PMC_URL = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    )

    def __init__(self, config: Config | None = None) -> None:
        config = config or Config()
        self._papers_dir = ensure_dir(config.papers_dir)
        self._email = os.environ["UNPAYWALL_EMAIL"]
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,*/*;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        })
        # Per-source rate limiting
        self._last_request: dict[str, float] = {}
        # Unpaywall: ~1 req/sec; Semantic Scholar: 100/5min ≈ 1 req/3sec
        # OpenAlex: 10 req/sec (with email); Europe PMC: no hard limit, be polite
        self._rate_limits: dict[str, float] = {
            "unpaywall": 1.0,
            "semantic_scholar": 3.0,
            "openalex": 0.5,
            "europe_pmc": 1.0,
        }

    # -- Public API -----------------------------------------------------------

    def fetch(self, doi: str) -> FetchResult:
        """Fetch a PDF for *doi*. Returns a cached copy if already on disk."""
        cached = doi_to_path(self._papers_dir, doi)
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

        # Fallback to OpenAlex
        result = self._try_openalex(doi)
        if result.success:
            return result

        # Fallback to Europe PMC
        result = self._try_europe_pmc(doi)
        if result.success:
            return result

        logger.warning("Failed to fetch PDF for DOI %s", doi)
        return FetchResult(
            doi=doi,
            success=False,
            error=result.error or "No open-access PDF found from any source",
        )

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

    def _try_openalex(self, doi: str) -> FetchResult:
        """Query OpenAlex for an open-access PDF URL and download it."""
        try:
            self._wait("openalex")
            resp = self._session.get(
                self._OPENALEX_URL.format(doi=doi),
                params={
                    "select": "open_access,best_oa_location",
                    "mailto": self._email,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            # Prefer best_oa_location pdf_url (often a repository copy)
            best = data.get("best_oa_location") or {}
            pdf_url = best.get("pdf_url")
            if not pdf_url:
                pdf_url = (data.get("open_access") or {}).get("oa_url")
            if not pdf_url:
                return FetchResult(
                    doi=doi, success=False, error="OpenAlex: no OA PDF URL"
                )
            return self._download_pdf(doi, pdf_url, "openalex")
        except requests.RequestException as exc:
            return FetchResult(
                doi=doi, success=False, error=f"OpenAlex API error: {exc}"
            )

    def _try_europe_pmc(self, doi: str) -> FetchResult:
        """Query Europe PMC for a full-text PDF URL and download it."""
        try:
            self._wait("europe_pmc")
            resp = self._session.get(
                self._EUROPE_PMC_URL,
                params={
                    "query": f"DOI:{doi}",
                    "format": "json",
                    "resultType": "core",
                    "pageSize": 1,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            results = (data.get("resultList") or {}).get("result") or []
            if not results:
                return FetchResult(
                    doi=doi, success=False, error="Europe PMC: no results"
                )

            for url_entry in (results[0].get("fullTextUrlList") or {}).get(
                "fullTextUrl", []
            ):
                if url_entry.get("documentStyle") == "pdf":
                    return self._download_pdf(
                        doi, url_entry["url"], "europe_pmc"
                    )

            return FetchResult(
                doi=doi, success=False, error="Europe PMC: no PDF URL"
            )
        except requests.RequestException as exc:
            return FetchResult(
                doi=doi, success=False, error=f"Europe PMC API error: {exc}"
            )

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _extract_unpaywall_pdf_url(data: dict) -> str | None:
        """Pull the best PDF URL from Unpaywall's JSON response.

        Prefers repository URLs (PubMed Central, institutional repos, arXiv)
        over publisher URLs, since publishers block programmatic downloads even
        for open-access papers. Falls back to any available PDF URL.
        """
        locations = data.get("oa_locations", [])
        repo_urls = [
            loc["url_for_pdf"]
            for loc in locations
            if loc.get("host_type") == "repository" and loc.get("url_for_pdf")
        ]
        if repo_urls:
            return repo_urls[0]

        # Fall back to publisher URLs (may 403, but worth trying)
        for loc in locations:
            url = loc.get("url_for_pdf")
            if url:
                return url
        return None

    @retry(max_attempts=3, delay=2.0, exceptions=(requests.RequestException,))
    def _download_pdf(self, doi: str, url: str, source: str) -> FetchResult:
        """Download a PDF from *url* and save it to the cache directory."""
        resp = self._session.get(url, timeout=60, stream=True)

        # Don't retry client errors (4xx) — they won't recover
        if 400 <= resp.status_code < 500:
            return FetchResult(
                doi=doi,
                success=False,
                error=f"{source}: {resp.status_code} {resp.reason} for url: {url}",
            )

        resp.raise_for_status()

        # Basic content-type validation
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and "octet-stream" not in content_type:
            return FetchResult(
                doi=doi,
                success=False,
                error=f"{source}: unexpected Content-Type {content_type!r}",
            )

        dest = doi_to_path(self._papers_dir, doi)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Downloaded %s via %s → %s", doi, source, dest.name)
        return FetchResult(doi=doi, success=True, path=dest, source=source)

    def _wait(self, source: str) -> None:
        """Enforce per-source rate limits."""
        interval = self._rate_limits.get(source, 1.0)
        last = self._last_request.get(source, 0.0)
        elapsed = time.monotonic() - last
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request[source] = time.monotonic()
