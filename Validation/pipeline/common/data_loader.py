"""Load the SciRAG-QA dataset and paper metadata, joined on Source_IDX."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Config


@dataclass(frozen=True)
class PaperMeta:
    """Metadata for a single source paper."""
    source_idx: str
    area: str
    sub_area: str
    topic: str
    title: str
    doi: str
    authors: list[str]
    date: str
    venue: str
    publisher: str


@dataclass(frozen=True)
class QAPair:
    """One question–answer row, enriched with paper metadata."""
    id: str
    question: str
    answer: str
    answer_type: str
    complexity: int
    source_idx: str
    gold_ref: str
    paper: Optional[PaperMeta] = None


def _parse_authors(raw: str) -> list[str]:
    """Parse the stringified author list from the CSV."""
    try:
        return json.loads(raw.replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        return [raw] if raw else []


def load_metadata(path: Path) -> dict[str, PaperMeta]:
    """Return a dict mapping Source_IDX → PaperMeta."""
    papers: dict[str, PaperMeta] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # The first (unnamed) column holds Source_IDX
            idx = row.get("") or row.get("Source_IDX", "")
            papers[idx] = PaperMeta(
                source_idx=idx,
                area=row["Area"],
                sub_area=row["Sub-area"],
                topic=row["Topic"],
                title=row["Title"],
                doi=row["DOI"],
                authors=_parse_authors(row["Authors"]),
                date=row["Date"],
                venue=row["Venue"],
                publisher=row["Publisher"],
            )
    return papers


def load_dataset(config: Config | None = None) -> list[QAPair]:
    """Load dataset.json + metadata.csv, join on Source_IDX.

    Returns a list of QAPair sorted by (complexity, source_idx, id).
    """
    config = config or Config()

    # Load paper metadata keyed by Source_IDX
    papers = load_metadata(config.metadata_csv)

    # Load question/answer rows
    with open(config.dataset_json, encoding="utf-8") as f:
        raw_questions: list[dict] = json.load(f)

    qa_pairs: list[QAPair] = []
    for row in raw_questions:
        idx = row["Source_IDX"]
        qa_pairs.append(
            QAPair(
                id=row["ID"],
                question=row["Question"],
                answer=str(row["Answer"]) if row["Answer"] is not None else "",
                answer_type=row["Type"],
                complexity=row["Complexity"],
                source_idx=idx,
                gold_ref=row["Gold_REF"],
                paper=papers.get(idx),
            )
        )

    qa_pairs.sort(key=lambda q: (q.complexity, q.source_idx, q.id))
    return qa_pairs
