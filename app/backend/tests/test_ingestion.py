"""Integration test for ingest_document."""

import uuid
from pathlib import Path

import pytest
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ..db.app_db import AppDB
from ..pipeline.common.config import Config
from ..pipeline.ingestion.document_store import DocumentStore
from ..pipeline.ingestion.semantic_chunker import CappedSemanticSplitter
from ..services.ingestion import ingest_document
from ..services.retrieval import RetrieverService

# Smallest available paper — used as a read-only fixture (never modified)
_FIXTURE_PDF = (
    Path(__file__).resolve().parents[3]
    / "Validation/pipeline/data/papers/10.1016__j.erss.2023.103036.pdf"
)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    base = tmp_path_factory.mktemp("ingestion_test")
    config = Config(
        chroma_dir=base / "chroma",
        db_path=base / "app.db",
        papers_dir=base / "papers",
        reranker_cache_dir=base / "reranker",
        nltk_data_dir=base / "nltk",
    )
    return config


@pytest.fixture(scope="module")
def embed_model(cfg):
    return HuggingFaceEmbedding(model_name=cfg.embedding_model, device=cfg.device)


@pytest.fixture(scope="module")
def store(cfg, embed_model):
    s = DocumentStore(embed_model, cfg)
    s.load_index()
    return s


@pytest.fixture(scope="module")
def db(cfg):
    return AppDB(cfg)


@pytest.fixture(scope="module")
def chunker(embed_model, cfg):
    return CappedSemanticSplitter(embed_model, cfg)


@pytest.fixture(scope="module")
def retriever_service(store, cfg):
    return RetrieverService(store, cfg)


@pytest.fixture(scope="module")
def ingested_doc(cfg, db, store, chunker, retriever_service):
    """Run ingest_document once and return the doc_id."""
    pytest.importorskip("fitz")
    if not _FIXTURE_PDF.exists():
        pytest.skip("Fixture PDF not found")

    doc_id = str(uuid.uuid4())
    filename = _FIXTURE_PDF.name
    db.create_document(doc_id, filename, Path(filename).stem)

    ingest_document(
        doc_id=doc_id,
        pdf_path=_FIXTURE_PDF,
        filename=filename,
        db=db,
        store=store,
        chunker=chunker,
        retriever_service=retriever_service,
        config=cfg,
    )
    return doc_id


def test_db_status_is_ready(db, ingested_doc):
    row = db.get_document(ingested_doc)
    assert row is not None
    assert row["status"] == "ready"


def test_chunk_count_nonzero(db, ingested_doc):
    row = db.get_document(ingested_doc)
    assert row["chunk_count"] > 0


def test_page_count_nonzero(db, ingested_doc):
    row = db.get_document(ingested_doc)
    assert row["page_count"] > 0


def test_chroma_nodes_have_correct_doc_id(store, ingested_doc):
    nodes = store.get_all_nodes()
    doc_nodes = [n for n in nodes if n.metadata.get("doc_id") == ingested_doc]
    assert len(doc_nodes) > 0
