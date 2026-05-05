"""Collections router: CRUD + document assignment."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class CollectionBody(BaseModel):
    name: str


@router.get("/api/collections")
async def list_collections(request: Request):
    return request.app.state.db.list_collections()


@router.post("/api/collections", status_code=201)
async def create_collection(body: CollectionBody, request: Request):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    return request.app.state.db.create_collection(str(uuid.uuid4()), name)


@router.put("/api/collections/{collection_id}")
async def rename_collection(collection_id: str, body: CollectionBody, request: Request):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    request.app.state.db.rename_collection(collection_id, name)
    return {"ok": True}


@router.delete("/api/collections/{collection_id}", status_code=204)
async def delete_collection(collection_id: str, request: Request):
    request.app.state.db.delete_collection(collection_id)


@router.post("/api/collections/{collection_id}/documents/{doc_id}", status_code=204)
async def add_document(collection_id: str, doc_id: str, request: Request):
    request.app.state.db.add_doc_to_collection(doc_id, collection_id)


@router.delete("/api/collections/{collection_id}/documents/{doc_id}", status_code=204)
async def remove_document(collection_id: str, doc_id: str, request: Request):
    request.app.state.db.remove_doc_from_collection(doc_id, collection_id)
