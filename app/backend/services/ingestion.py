"""Document ingestion service."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from llama_index.core.schema import Document

from ..db.app_db import AppDB
from ..pipeline.common.config import Config
from ..pipeline.common.utils import get_logger
from ..pipeline.ingestion.document_store import DocumentStore
from ..pipeline.ingestion.pdf_parser import parse_pdf
from ..pipeline.ingestion.semantic_chunker import CappedSemanticSplitter

logger = get_logger(__name__)


def ingest_document(
    doc_id: str,
    pdf_path: Path,
    filename: str,
    db: AppDB,
    store: DocumentStore,
    chunker: CappedSemanticSplitter,
    retriever_service: object,
    config: Config | None = None,
) -> None:
    """Parse, chunk, and index a PDF; update DB status when done."""
    cfg = config or Config()
    try:
        # Extract PyMuPDF metadata for title and page count
        fitz_doc = fitz.open(str(pdf_path))
        fitz_meta = fitz_doc.metadata or {}
        fitz_page_count = len(fitz_doc)
        fitz_doc.close()

        title = (fitz_meta.get("title") or "").strip() or Path(filename).stem

        # Parse text + page offsets
        result = parse_pdf(pdf_path)
        if not result.success:
            db.update_document_status(doc_id, "failed", error=result.error)
            return

        page_count = result.page_count or fitz_page_count

        # Mirror corpus_builder.py metadata policy, adapted for app documents.
        # page_offsets is needed by CappedSemanticSplitter for page attribution.
        metadata = {
            "doc_id": doc_id,
            "title": title,
            "page_offsets": result.page_offsets,
        }
        excluded_from_embed = ["doc_id", "page_offsets"]
        if not cfg.embed_paper_metadata:
            excluded_from_embed.append("title")

        doc = Document(
            text=result.text,
            metadata=metadata,
            excluded_embed_metadata_keys=excluded_from_embed,
            excluded_llm_metadata_keys=["doc_id", "page_offsets"],
        )

        nodes = chunker.get_nodes_from_documents([doc])
        store.upsert_nodes(nodes)

        db.update_document_status(
            doc_id, "ready", chunk_count=len(nodes), page_count=page_count
        )

        logger.info(
            "Ingested doc %s: %d pages, %d chunks", doc_id, page_count, len(nodes)
        )

    except Exception as exc:
        logger.exception("Ingestion failed for doc %s: %s", doc_id, exc)
        db.update_document_status(doc_id, "failed", error=str(exc))

    finally:
        retriever_service.invalidate()
