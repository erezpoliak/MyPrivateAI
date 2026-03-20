"""Unit tests for data_loader: _parse_authors, join logic, and sort order."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from common.config import Config
from common.data_loader import QAPair, _parse_authors, load_dataset, load_metadata


# ---------------------------------------------------------------------------
# _parse_authors
# ---------------------------------------------------------------------------

class TestParseAuthors:

    def test_single_quoted_list(self):
        assert _parse_authors("['Alice', 'Bob']") == ["Alice", "Bob"]

    def test_already_double_quoted_list(self):
        assert _parse_authors('["Alice", "Bob"]') == ["Alice", "Bob"]

    def test_single_author_string(self):
        # Not a list — falls back to wrapping raw string
        assert _parse_authors("Alice") == ["Alice"]

    def test_empty_string_returns_empty_list(self):
        assert _parse_authors("") == []

    def test_malformed_json_falls_back_to_raw(self):
        raw = "not valid json at all"
        assert _parse_authors(raw) == [raw]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_META_ROWS = [
    {
        "": "P1",
        "Area": "ML",
        "Sub-area": "NLP",
        "Topic": "RAG",
        "Title": "Paper One",
        "DOI": "10.1/one",
        "Authors": "['Auth A']",
        "Date": "2024",
        "Venue": "NeurIPS",
        "Publisher": "MIT",
    },
    {
        "": "P2",
        "Area": "CV",
        "Sub-area": "Detection",
        "Topic": "YOLO",
        "Title": "Paper Two",
        "DOI": "10.1/two",
        "Authors": "['Auth B', 'Auth C']",
        "Date": "2023",
        "Venue": "CVPR",
        "Publisher": "IEEE",
    },
]

_QA_ROWS = [
    {"ID": "q3", "Question": "Q3", "Answer": "A3", "Type": "bool",   "Complexity": 3, "Source_IDX": "P1", "Gold_REF": "ref3"},
    {"ID": "q1", "Question": "Q1", "Answer": "A1", "Type": "factoid","Complexity": 1, "Source_IDX": "P2", "Gold_REF": "ref1"},
    {"ID": "q2", "Question": "Q2", "Answer": "A2", "Type": "factoid","Complexity": 1, "Source_IDX": "P1", "Gold_REF": "ref2"},
    {"ID": "q4", "Question": "Q4", "Answer": "A4", "Type": "bool",   "Complexity": 2, "Source_IDX": "MISSING", "Gold_REF": "ref4"},
]


def _write_metadata(path: Path, rows: list[dict]) -> None:
    fieldnames = ["", "Area", "Sub-area", "Topic", "Title", "DOI", "Authors", "Date", "Venue", "Publisher"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_dataset(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)


def _cfg(tmp_path: Path) -> Config:
    meta = tmp_path / "metadata.csv"
    dataset = tmp_path / "dataset.json"
    _write_metadata(meta, _META_ROWS)
    _write_dataset(dataset, _QA_ROWS)
    cfg = Config()
    cfg.metadata_csv = meta
    cfg.dataset_json = dataset
    return cfg


# ---------------------------------------------------------------------------
# load_metadata
# ---------------------------------------------------------------------------

class TestLoadMetadata:

    def test_returns_dict_keyed_by_source_idx(self, tmp_path):
        meta_path = tmp_path / "metadata.csv"
        _write_metadata(meta_path, _META_ROWS)
        papers = load_metadata(meta_path)
        assert set(papers.keys()) == {"P1", "P2"}

    def test_authors_parsed_correctly(self, tmp_path):
        meta_path = tmp_path / "metadata.csv"
        _write_metadata(meta_path, _META_ROWS)
        papers = load_metadata(meta_path)
        assert papers["P2"].authors == ["Auth B", "Auth C"]


# ---------------------------------------------------------------------------
# load_dataset — join and sort
# ---------------------------------------------------------------------------

class TestLoadDataset:

    def test_returns_list_of_qa_pairs(self, tmp_path):
        pairs = load_dataset(_cfg(tmp_path))
        assert all(isinstance(p, QAPair) for p in pairs)
        assert len(pairs) == len(_QA_ROWS)

    def test_known_source_idx_joins_paper_metadata(self, tmp_path):
        pairs = load_dataset(_cfg(tmp_path))
        p1_pairs = [p for p in pairs if p.source_idx == "P1"]
        assert all(p.paper is not None for p in p1_pairs)
        assert p1_pairs[0].paper.title == "Paper One"

    def test_missing_source_idx_sets_paper_to_none(self, tmp_path):
        pairs = load_dataset(_cfg(tmp_path))
        missing = next(p for p in pairs if p.source_idx == "MISSING")
        assert missing.paper is None

    def test_sorted_by_complexity_then_source_idx_then_id(self, tmp_path):
        pairs = load_dataset(_cfg(tmp_path))
        keys = [(p.complexity, p.source_idx, p.id) for p in pairs]
        assert keys == sorted(keys)
