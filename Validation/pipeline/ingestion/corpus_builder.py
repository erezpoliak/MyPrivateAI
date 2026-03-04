"""Two-mode corpus orchestrator: gold_ref or fetched.

Calls the ingestion layer (pdf_parser, chunker) to produce a flat list of
LlamaIndex TextNodes, plus a CorpusManifest tracking per-paper status.

Usage::

    from pipeline.ingestion.fixed_chunker import FixedSizeChunker
    chunker = FixedSizeChunker(config)
    nodes, manifest = build_corpus(qa_pairs, chunker, config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol

from llama_index.core.schema import TextNode

from ..common.config import Config, CorpusMode
from ..common.data_loader import QAPair
from ..common.utils import doi_to_path, get_logger
from .pdf_parser import parse_pdf

logger = get_logger(__name__)


# ── Chunker protocol ────────────────────────────────────────────────────────

class Chunker(Protocol):
    """Structural type satisfied by both FixedSizeChunker and SemanticChunker."""

    def chunk(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[TextNode]: ...


# ── Per-paper status ─────────────────────────────────────────────────────────

class CorpusTier(str, Enum):
    """Outcome of corpus construction for a single paper."""

    GOLD_REF = "gold_ref"
    FETCHED = "fetched"
    FAILED = "failed"


@dataclass(frozen=True)
class PaperCorpusEntry:
    """Corpus build result for one paper."""

    source_idx: str
    tier: CorpusTier
    node_count: int = 0
    error: Optional[str] = None


# ── Manifest ─────────────────────────────────────────────────────────────────

@dataclass
class CorpusManifest:
    """Summary produced by a corpus build run."""

    mode: CorpusMode
    entries: dict[str, PaperCorpusEntry] = field(default_factory=dict)
    total_nodes: int = 0

    @property
    def succeeded(self) -> list[str]:
        """Source IDXs that produced at least one node."""
        return [idx for idx, e in self.entries.items()
                if e.tier != CorpusTier.FAILED]

    @property
    def failed(self) -> list[str]:
        """Source IDXs that could not be ingested."""
        return [idx for idx, e in self.entries.items()
                if e.tier == CorpusTier.FAILED]

    def has_corpus(self, source_idx: str) -> bool:
        """Return True if *source_idx* has usable corpus nodes."""
        entry = self.entries.get(source_idx)
        return entry is not None and entry.tier != CorpusTier.FAILED


# ── Corpus building functions ────────────────────────────────────────────────

def build_corpus(
    qa_pairs: list[QAPair],
    chunker: Chunker,
    config: Config,
) -> tuple[list[TextNode], CorpusManifest]:
    """Build corpus nodes based on ``config.corpus_mode``."""
    mode = config.corpus_mode
    if mode == CorpusMode.GOLD_REF:
        return _build_gold_ref(qa_pairs, chunker)
    if mode == CorpusMode.FETCHED:
        return _build_fetched(qa_pairs, chunker, config)
    raise ValueError(
        f"build_corpus does not handle mode {mode!r} "
        "(use CorpusMode.GOLD_REF or CorpusMode.FETCHED)"
    )


def _build_gold_ref(
    qa_pairs: list[QAPair],
    chunker: Chunker,
) -> tuple[list[TextNode], CorpusManifest]:
    """Chunk the Gold_REF passages grouped by paper."""
    # Collect unique gold_ref texts per paper
    paper_refs: dict[str, list[str]] = {}
    for qa in qa_pairs:
        if not qa.gold_ref:
            continue
        refs = paper_refs.setdefault(qa.source_idx, [])
        if qa.gold_ref not in refs:
            refs.append(qa.gold_ref)

    manifest = CorpusManifest(mode=CorpusMode.GOLD_REF)
    all_nodes: list[TextNode] = []

    for source_idx, refs in paper_refs.items():
        metadata = {
            "source_idx": source_idx,
            "corpus_mode": "gold_ref",
        }
        paper_nodes: list[TextNode] = []
        for ref_text in refs:
            paper_nodes.extend(chunker.chunk(ref_text, metadata))

        manifest.entries[source_idx] = PaperCorpusEntry(
            source_idx=source_idx,
            tier=CorpusTier.GOLD_REF,
            node_count=len(paper_nodes),
        )
        all_nodes.extend(paper_nodes)

    manifest.total_nodes = len(all_nodes)
    logger.info(
        "Gold-ref corpus: %d papers, %d nodes",
        len(manifest.entries),
        manifest.total_nodes,
    )
    return all_nodes, manifest


def _build_fetched(
    qa_pairs: list[QAPair],
    chunker: Chunker,
    config: Config,
) -> tuple[list[TextNode], CorpusManifest]:
    """Parse and chunk fetched PDFs. Papers without a PDF are marked FAILED."""
    # Deduplicate papers — keep first QAPair per source_idx for metadata
    papers: dict[str, QAPair] = {}
    for qa in qa_pairs:
        papers.setdefault(qa.source_idx, qa)

    manifest = CorpusManifest(mode=CorpusMode.FETCHED)
    all_nodes: list[TextNode] = []

    for source_idx, qa in papers.items():
        # Need a DOI to locate the cached PDF
        if not qa.paper or not qa.paper.doi:
            manifest.entries[source_idx] = PaperCorpusEntry(
                source_idx=source_idx,
                tier=CorpusTier.FAILED,
                error="No DOI in metadata",
            )
            continue

        pdf_path = doi_to_path(config.papers_dir, qa.paper.doi)
        if not pdf_path.exists():
            manifest.entries[source_idx] = PaperCorpusEntry(
                source_idx=source_idx,
                tier=CorpusTier.FAILED,
                error=f"PDF not found: {pdf_path.name}",
            )
            continue

        # Parse
        result = parse_pdf(pdf_path)
        if not result.success:
            manifest.entries[source_idx] = PaperCorpusEntry(
                source_idx=source_idx,
                tier=CorpusTier.FAILED,
                error=result.error or "Parse failed",
            )
            continue

        # Chunk
        metadata = {
            "source_idx": source_idx,
            "corpus_mode": "fetched",
            "doi": qa.paper.doi,
        }
        paper_nodes = chunker.chunk(result.text, metadata)

        manifest.entries[source_idx] = PaperCorpusEntry(
            source_idx=source_idx,
            tier=CorpusTier.FETCHED,
            node_count=len(paper_nodes),
        )
        all_nodes.extend(paper_nodes)

    manifest.total_nodes = len(all_nodes)
    succeeded = len(manifest.succeeded)
    total = len(papers)
    logger.info(
        "Fetched corpus: %d/%d papers succeeded, %d nodes",
        succeeded,
        total,
        manifest.total_nodes,
    )
    return all_nodes, manifest
