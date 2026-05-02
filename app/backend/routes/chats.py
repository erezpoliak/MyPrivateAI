"""Chats blueprint: CRUD + streaming message endpoint."""

from __future__ import annotations

import asyncio
import uuid
from typing import Generator

from flask import Blueprint, Response, current_app, jsonify, request

from ..services.chat import stream_answer

bp = Blueprint("chats", __name__)


def _sync_stream(async_gen) -> Generator[str, None, None]:
    """Drive an async generator synchronously for Flask streaming."""
    loop = asyncio.new_event_loop()
    try:
        ait = async_gen.__aiter__()
        while True:
            try:
                yield loop.run_until_complete(ait.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()


@bp.get("/api/chats")
def list_chats():
    db = current_app.config["DB"]
    return jsonify(db.list_chats())


@bp.post("/api/chats")
def create_chat():
    db = current_app.config["DB"]
    chat_id = str(uuid.uuid4())
    row = db.create_chat(chat_id)
    return jsonify(row), 201


@bp.get("/api/chats/<chat_id>")
def get_chat(chat_id: str):
    db = current_app.config["DB"]

    chat = db.get_chat(chat_id)
    if chat is None:
        return jsonify({"error": "Chat not found"}), 404

    messages = db.get_messages(chat_id)
    for msg in messages:
        msg["sources"] = db.get_message_sources(msg["id"])

    return jsonify({**chat, "messages": messages})


@bp.delete("/api/chats/<chat_id>")
def delete_chat(chat_id: str):
    db = current_app.config["DB"]

    if db.get_chat(chat_id) is None:
        return jsonify({"error": "Chat not found"}), 404

    db.delete_chat(chat_id)
    return "", 204


@bp.post("/api/chats/<chat_id>/messages")
def post_message(chat_id: str):
    db = current_app.config["DB"]

    if db.get_chat(chat_id) is None:
        return jsonify({"error": "Chat not found"}), 404

    body = request.get_json(silent=True) or {}
    content = (body.get("content") or "").strip()
    if not content:
        return jsonify({"error": "content is required"}), 400

    llm = current_app.config["LLM"]
    retriever_service = current_app.config["RETRIEVER_SERVICE"]
    cfg = current_app.config["PIPELINE_CONFIG"]

    gen = stream_answer(
        chat_id=chat_id,
        content=content,
        db=db,
        llm=llm,
        retriever_service=retriever_service,
        config=cfg,
    )

    response = Response(_sync_stream(gen), mimetype="text/event-stream")
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Cache-Control"] = "no-cache"
    return response
