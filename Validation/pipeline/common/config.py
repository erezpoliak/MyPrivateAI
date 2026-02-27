"""Centralised configuration for the validation pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch


class CorpusMode(str, Enum):
    """Which corpus source to use for a run."""
    GOLD_REF = "gold_ref"
    FETCHED = "fetched"


def _detect_device() -> str:
    """Return best available device: cuda → mps → cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Resolve key directories relative to *this file* so the config works
# regardless of the caller's working directory.
# ---------------------------------------------------------------------------
_PIPELINE_DIR = Path(__file__).resolve().parent.parent          # Validation/pipeline
_VALIDATION_DIR = _PIPELINE_DIR.parent                          # Validation/
_DATASET_DIR = _VALIDATION_DIR / "SciRAG-QA"


@dataclass
class Config:
    """Single source of truth for every tuneable knob in the pipeline."""

    # -- Device --------------------------------------------------------------
    device: str = field(default_factory=_detect_device)

    # -- Paths ---------------------------------------------------------------
    pipeline_dir: Path = _PIPELINE_DIR
    dataset_dir: Path = _DATASET_DIR
    dataset_json: Path = _DATASET_DIR / "dataset.json"
    metadata_csv: Path = _DATASET_DIR / "metadata.csv"
    papers_dir: Path = _PIPELINE_DIR / "data" / "papers"
    chroma_dir: Path = _PIPELINE_DIR / "data" / "chroma"
    results_dir: Path = _PIPELINE_DIR / "results"
    db_path: Path = _PIPELINE_DIR / "results" / "runs.db"
    nltk_data_dir: Path = _PIPELINE_DIR / "data" / "nltk_data"

    # -- LLM -----------------------------------------------------------------
    ollama_model: str = "llama3.1:8b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: float = 120.0
    ollama_temperature: float = 0.1

    # -- Embeddings ----------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # -- Chunking ------------------------------------------------------------
    fixed_chunk_size: int = 512          # tokens
    fixed_chunk_overlap: int = 50        # tokens
    semantic_threshold_pct: int = 95     # percentile for cosine dissimilarity spike
    semantic_max_tokens: int = 512       # sub-split semantic chunks exceeding this limit, BGE-small max context is 512 tokens

    # -- Retrieval -----------------------------------------------------------
    vector_top_k: int = 10
    bm25_top_k: int = 10
    hybrid_top_n: int = 5               # after rerank
    rrf_k: int = 60
    rrf_bm25_weight: float = 0.4
    rrf_vector_weight: float = 0.6

    # -- Reranker ------------------------------------------------------------
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"

    # -- Corpus mode ---------------------------------------------------------
    corpus_mode: CorpusMode = CorpusMode.GOLD_REF

    # -- RAGAS ---------------------------------------------------------------
    ragas_judge_model: str = "gpt-4o-mini"
    ragas_ceiling_model: str = "gpt-4o"

    # -- Agent (Phase 2) -----------------------------------------------------
    max_agent_hops: int = 3

    def __post_init__(self) -> None:
        # Honour .env override for NLTK_DATA
        env_nltk = os.getenv("NLTK_DATA")
        if env_nltk:
            self.nltk_data_dir = Path(env_nltk)
