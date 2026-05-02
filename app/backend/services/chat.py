"""Streaming chat service."""

from __future__ import annotations

import json
import uuid
from typing import AsyncGenerator

from ..db.app_db import AppDB
from ..pipeline.agent.events import TokenEvent, TraceEvent
from ..pipeline.agent.workflow import AgentWorkflow
from ..pipeline.common.config import Config
from .citations import parse_citations
from .retrieval import RetrieverService

_EMPTY_CORPUS_MSG = "Please upload a document first before asking questions."


def _sse(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


async def stream_answer(
    chat_id: str,
    content: str,
    db: AppDB,
    llm,
    retriever_service: RetrieverService,
    config: Config,
) -> AsyncGenerator[str, None]:
    """Async generator yielding SSE-formatted strings for a single chat turn.

    Yields ``trace``, ``token``, and finally ``done`` events.
    Persists user message, assistant message, and sources to DB.
    """
    user_msg_id = str(uuid.uuid4())
    db.create_message(user_msg_id, chat_id, "user", content)

    retriever = retriever_service.get()
    if retriever is None:
        assistant_msg_id = str(uuid.uuid4())
        db.create_message(assistant_msg_id, chat_id, "assistant", _EMPTY_CORPUS_MSG)
        yield _sse("done", {"message_id": assistant_msg_id, "sources": []})
        return

    workflow = AgentWorkflow(
        question=content,
        llm=llm,
        retriever=retriever,
        config=config,
    )
    handler = workflow.run()

    async for event in handler.stream_events():
        if isinstance(event, TraceEvent):
            yield _sse("trace", {
                "step": event.step,
                "status": event.status,
                "info": event.info,
            })
        elif isinstance(event, TokenEvent):
            yield _sse("token", {"text": event.text})

    answer = await handler

    chunks = workflow.cited_chunks_for(answer)
    sources = parse_citations(answer, chunks)

    assistant_msg_id = str(uuid.uuid4())
    db.create_message(assistant_msg_id, chat_id, "assistant", answer)

    if sources:
        db.create_message_sources([
            {
                "message_id": assistant_msg_id,
                "chunk_index": s.chunk_index,
                "doc_id": s.doc_id,
                "title": s.title,
                "page_start": s.page_start,
                "page_end": s.page_end,
                "snippet": s.snippet,
            }
            for s in sources
        ])

    yield _sse("done", {
        "message_id": assistant_msg_id,
        "sources": [
            {
                "chunk_index": s.chunk_index,
                "doc_id": s.doc_id,
                "title": s.title,
                "page_start": s.page_start,
                "page_end": s.page_end,
                "snippet": s.snippet,
            }
            for s in sources
        ],
    })
