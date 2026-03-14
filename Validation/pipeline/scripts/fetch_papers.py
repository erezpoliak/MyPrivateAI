"""One-time setup: download all papers in the SciRAG-QA dataset.

Fetches PDFs via Unpaywall → Semantic Scholar fallback for every unique DOI
in metadata.csv, writing them to ``data/papers/``.  After the run a JSON
manifest is written to ``data/fetch_manifest.json`` with per-paper outcome
and summary statistics.

Usage (from Validation/pipeline/)::

    python scripts/fetch_papers.py [--dry-run]

Environment variables required:
    UNPAYWALL_EMAIL — your e-mail address for the Unpaywall API
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow ``python scripts/fetch_papers.py`` to resolve sibling packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import Config
from common.data_loader import load_metadata
from common.utils import ensure_dir, get_logger
from ingestion.pdf_fetcher import PDFFetcher

logger = get_logger(__name__)

MANIFEST_PATH_RELATIVE = "data/fetch_manifest.json"


def _collect_dois(config: Config) -> dict[str, str]:
    """Return {source_idx: doi} for every unique paper in metadata.csv."""
    papers = load_metadata(config.metadata_csv)
    return {idx: meta.doi for idx, meta in sorted(papers.items()) if meta.doi}


def _write_manifest(config: Config, records: list[dict]) -> Path:
    """Persist fetch results as a JSON manifest; return the path."""
    total = len(records)
    succeeded = sum(1 for r in records if r["success"])

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_papers": total,
        "fetched": succeeded,
        "failed": total - succeeded,
        "coverage_pct": round(100 * succeeded / total, 1) if total else 0.0,
        "papers": records,
    }

    dest = config.pipeline_dir / MANIFEST_PATH_RELATIVE
    ensure_dir(dest.parent)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return dest


def run(dry_run: bool = False) -> None:
    config = Config()
    dois = _collect_dois(config)

    logger.info(
        "Found %d unique papers to fetch (papers_dir=%s)",
        len(dois),
        config.papers_dir,
    )

    if dry_run:
        for idx, doi in dois.items():
            logger.info("[DRY-RUN] Would fetch: source_idx=%s  doi=%s", idx, doi)
        return

    fetcher = PDFFetcher(config)
    records: list[dict] = []

    doi_list = list(dois.items())
    total = len(doi_list)
    for i, (source_idx, doi) in enumerate(doi_list, 1):
        logger.info("[%d/%d] source_idx=%s  doi=%s", i, total, source_idx, doi)
        result = fetcher.fetch(doi)
        records.append({
            "source_idx": source_idx,
            "doi": doi,
            "success": result.success,
            "source": result.source,
            "path": str(result.path) if result.path else None,
            "error": result.error,
        })

    # Summary
    succeeded = sum(1 for r in records if r["success"])
    coverage = 100 * succeeded / total if total else 0.0
    logger.info(
        "Fetch complete: %d/%d succeeded — coverage %.1f%%",
        succeeded,
        total,
        coverage,
    )

    manifest_path = _write_manifest(config, records)
    logger.info("Manifest written to %s", manifest_path)

    # Coverage gate (warn, don't hard-fail — user decides whether to continue)
    if coverage < 80.0:
        logger.warning(
            "Coverage %.1f%% is below the 80%% threshold. "
            "Review the manifest and find alternative PDF sources before "
            "running any fetched-mode experiment.",
            coverage,
        )
    else:
        logger.info("Coverage gate PASSED (%.1f%% >= 80%%).", coverage)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all SciRAG-QA papers to data/papers/."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without making any network requests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(dry_run=args.dry_run)
