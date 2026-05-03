"""Documents router: upload, list, delete."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile

from ..pipeline.ingestion.semantic_chunker import CappedSemanticSplitter
from ..services.ingestion import ingest_document

router = APIRouter()


@router.post("/api/documents", status_code=201)
async def upload_document(request: Request, file: UploadFile, background_tasks: BackgroundTasks):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    db = request.app.state.db
    store = request.app.state.store
    retriever_service = request.app.state.retriever_service
    cfg = request.app.state.config

    doc_id = str(uuid.uuid4())
    filename = file.filename
    title = Path(filename).stem

    dest: Path = cfg.papers_dir / f"{doc_id}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())

    row = db.create_document(doc_id, filename, title)

    chunker = CappedSemanticSplitter(store.embed_model, cfg)
    background_tasks.add_task(
        ingest_document,
        doc_id=doc_id,
        pdf_path=dest,
        filename=filename,
        db=db,
        store=store,
        chunker=chunker,
        retriever_service=retriever_service,
        config=cfg,
    )

    return row


@router.get("/api/documents")
async def list_documents(request: Request):
    return request.app.state.db.list_documents()


@router.delete("/api/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str, request: Request):
    db = request.app.state.db
    store = request.app.state.store
    retriever_service = request.app.state.retriever_service
    cfg = request.app.state.config

    if db.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    store.delete_by_doc_id(doc_id)

    pdf_path: Path = cfg.papers_dir / f"{doc_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()

    db.delete_document(doc_id)
    retriever_service.invalidate()
