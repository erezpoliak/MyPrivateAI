"""Centralised configuration for the app pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch



def _detect_device() -> str:
    """Return best available device: cuda → mps → cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Resolve key directories relative to *this file* so the config works
# regardless of the caller's working directory — this is the source-tree
# default used in local dev.
#
# Both roots below are env-overridable so a packaged app can point them at
# an OS-appropriate location instead of the (read-only, once bundled) source
# tree:
#   MYPRIVATEAI_DATA_DIR      — writable: sqlite db, chroma index, uploaded PDFs
#   MYPRIVATEAI_RESOURCE_DIR  — read-only: reranker + NLTK model caches
# ---------------------------------------------------------------------------
_PIPELINE_DIR = Path(__file__).resolve().parent.parent          # app/backend/pipeline
_BACKEND_DIR = _PIPELINE_DIR.parent                              # app/backend
_DATA_DIR = Path(os.getenv("MYPRIVATEAI_DATA_DIR", str(_BACKEND_DIR / "data")))
_RESOURCE_DIR = Path(os.getenv("MYPRIVATEAI_RESOURCE_DIR", str(_BACKEND_DIR / "data")))
# Unchanged dev default (app/backend/storage/papers) so existing local uploads
# stay where they are when MYPRIVATEAI_DATA_DIR is unset.
_PAPERS_DIR = (
    _DATA_DIR / "storage" / "papers"
    if os.getenv("MYPRIVATEAI_DATA_DIR")
    else _BACKEND_DIR / "storage" / "papers"
)


@dataclass
class Config:
    """Single source of truth for every tuneable knob in the pipeline."""

    # -- Device --------------------------------------------------------------
    device: str = field(default_factory=_detect_device)

    # -- Paths ---------------------------------------------------------------
    # Writable at runtime.
    papers_dir: Path = _PAPERS_DIR
    chroma_dir: Path = _DATA_DIR / "chroma"
    db_path: Path = _DATA_DIR / "app.db"
    # Read-only model caches — never written to after packaging.
    reranker_cache_dir: Path = _RESOURCE_DIR / "reranker_models"
    nltk_data_dir: Path = _RESOURCE_DIR / "nltk_data"

    # -- LLM (MLX — Apple Silicon only) -------------------------------------
    mlx_model: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    # mlx_model: str = "mlx-community/Qwen3.5-9B-OptiQ-4bit"
    mlx_temperature: float = 0.1
    mlx_max_tokens: int = 512

    # -- Embeddings ----------------------------------------------------------
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # -- Chunking ------------------------------------------------------------
    semantic_threshold_pct: int = 95     # percentile for cosine dissimilarity spike
    semantic_max_tokens: int = 512       # sub-split semantic chunks exceeding this limit, Specter2 max context is 512 tokens
    chunk_overlap: int = 50              # token overlap between sub-split chunks

    # -- Retrieval -----------------------------------------------------------
    vector_top_k: int = 10
    bm25_top_k: int = 10
    hybrid_top_n: int = 5               # after rerank
    rrf_k: int = 60
    rrf_bm25_weight: float = 0.4
    rrf_vector_weight: float = 0.6

    # -- Reranker ------------------------------------------------------------
    reranker_model: str = "BAAI/bge-reranker-base"

    # -- Metadata enrichment -------------------------------------------------
    embed_paper_metadata: bool = True    # prepend title to chunk embeddings

    def __post_init__(self) -> None:
        # Honour .env override for NLTK_DATA (takes priority over
        # MYPRIVATEAI_RESOURCE_DIR — useful for pointing at a scratch dir
        # in dev without touching the resource dir env var).
        env_nltk = os.getenv("NLTK_DATA")
        if env_nltk:
            self.nltk_data_dir = Path(env_nltk)

        # Writable dirs must exist before anything tries to read/write them.
        # Read-only resource dirs (reranker_cache_dir, nltk_data_dir) are
        # never created here — in a packaged app they're pre-populated and
        # read-only; creating them would fail there and is unnecessary in dev.
        for path in (self.papers_dir, self.chroma_dir, self.db_path.parent):
            path.mkdir(parents=True, exist_ok=True)
