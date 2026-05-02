"""Documents blueprint: upload, list, delete."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from ..pipeline.ingestion.semantic_chunker import CappedSemanticSplitter
from ..services.ingestion import ingest_document

bp = Blueprint("documents", __name__)


@bp.post("/api/documents")
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted"}), 400

    db = current_app.config["DB"]
    store = current_app.config["STORE"]
    retriever_service = current_app.config["RETRIEVER_SERVICE"]
    cfg = current_app.config["PIPELINE_CONFIG"]

    doc_id = str(uuid.uuid4())
    filename = file.filename
    title = Path(filename).stem

    dest: Path = cfg.papers_dir / f"{doc_id}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    file.save(str(dest))

    row = db.create_document(doc_id, filename, title)

    embed_model = store.embed_model
    chunker = CappedSemanticSplitter(embed_model, cfg)

    thread = threading.Thread(
        target=ingest_document,
        kwargs=dict(
            doc_id=doc_id,
            pdf_path=dest,
            filename=filename,
            db=db,
            store=store,
            chunker=chunker,
            retriever_service=retriever_service,
            config=cfg,
        ),
        daemon=True,
    )
    thread.start()

    return jsonify(row), 201


@bp.get("/api/documents")
def list_documents():
    db = current_app.config["DB"]
    return jsonify(db.list_documents())


@bp.delete("/api/documents/<doc_id>")
def delete_document(doc_id: str):
    db = current_app.config["DB"]
    store = current_app.config["STORE"]
    retriever_service = current_app.config["RETRIEVER_SERVICE"]
    cfg = current_app.config["PIPELINE_CONFIG"]

    row = db.get_document(doc_id)
    if row is None:
        return jsonify({"error": "Document not found"}), 404

    store.delete_by_doc_id(doc_id)

    pdf_path: Path = cfg.papers_dir / f"{doc_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()

    db.delete_document(doc_id)
    retriever_service.invalidate()

    return "", 204
