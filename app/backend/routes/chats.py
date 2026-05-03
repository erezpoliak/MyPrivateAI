"""Chats router: CRUD + streaming message endpoint."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..services.chat import stream_answer

router = APIRouter()


class MessageBody(BaseModel):
    content: str = ""


@router.get("/api/chats")
async def list_chats(request: Request):
    return request.app.state.db.list_chats()


@router.post("/api/chats", status_code=201)
async def create_chat(request: Request):
    db = request.app.state.db
    chat_id = str(uuid.uuid4())
    return db.create_chat(chat_id)


@router.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str, request: Request):
    db = request.app.state.db
    chat = db.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    messages = db.get_messages(chat_id)
    for msg in messages:
        msg["sources"] = db.get_message_sources(msg["id"])

    return {**chat, "messages": messages}


@router.delete("/api/chats/{chat_id}", status_code=204)
async def delete_chat(chat_id: str, request: Request):
    db = request.app.state.db
    if db.get_chat(chat_id) is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.delete_chat(chat_id)


@router.post("/api/chats/{chat_id}/messages")
async def post_message(chat_id: str, body: MessageBody, request: Request):
    db = request.app.state.db
    if db.get_chat(chat_id) is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")

    return StreamingResponse(
        stream_answer(
            chat_id=chat_id,
            content=content,
            db=db,
            llm=request.app.state.llm,
            retriever_service=request.app.state.retriever_service,
            config=request.app.state.config,
        ),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
