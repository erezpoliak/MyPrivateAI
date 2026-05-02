"""Tests for stream_answer SSE output."""

import asyncio
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from ..pipeline.agent.events import TokenEvent, TraceEvent
from ..services.chat import stream_answer


# ---------------------------------------------------------------------------
# Stub AgentWorkflow
# ---------------------------------------------------------------------------

_STUB_ANSWER = "The answer is 42."

_STUB_EVENTS = [
    TraceEvent(step="retrieve", status="start", info="hop 1"),
    TraceEvent(step="retrieve", status="done", info="2 chunks"),
    TokenEvent(text="The "),
    TokenEvent(text="answer "),
    TokenEvent(text="is 42."),
]


class _StubHandler:
    async def stream_events(self):
        for ev in _STUB_EVENTS:
            yield ev

    def __await__(self):
        async def _result():
            return _STUB_ANSWER
        return _result().__await__()


class _StubWorkflow:
    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        return _StubHandler()

    def cited_chunks_for(self, answer):
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db():
    db = MagicMock()
    db.create_message.return_value = {"id": str(uuid.uuid4()), "role": "user", "content": "hi"}
    return db


def _make_retriever_service():
    svc = MagicMock()
    svc.get.return_value = MagicMock()  # non-None → corpus has documents
    return svc


async def _collect(gen) -> list[dict]:
    events = []
    async for chunk in gen:
        payload = chunk.removeprefix("data: ").strip()
        if payload:
            events.append(json.loads(payload))
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("app.backend.services.chat.AgentWorkflow", _StubWorkflow)
def test_yields_trace_events():
    db = _make_db()
    gen = stream_answer("chat-1", "question?", db, MagicMock(), _make_retriever_service(), MagicMock())
    events = asyncio.run(_collect(gen))
    trace_events = [e for e in events if e["type"] == "trace"]
    assert len(trace_events) >= 1


@patch("app.backend.services.chat.AgentWorkflow", _StubWorkflow)
def test_yields_token_events():
    db = _make_db()
    gen = stream_answer("chat-1", "question?", db, MagicMock(), _make_retriever_service(), MagicMock())
    events = asyncio.run(_collect(gen))
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) >= 1


@patch("app.backend.services.chat.AgentWorkflow", _StubWorkflow)
def test_yields_exactly_one_done_event():
    db = _make_db()
    gen = stream_answer("chat-1", "question?", db, MagicMock(), _make_retriever_service(), MagicMock())
    events = asyncio.run(_collect(gen))
    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1


@patch("app.backend.services.chat.AgentWorkflow", _StubWorkflow)
def test_done_event_has_message_id_and_sources():
    db = _make_db()
    gen = stream_answer("chat-1", "question?", db, MagicMock(), _make_retriever_service(), MagicMock())
    events = asyncio.run(_collect(gen))
    done = next(e for e in events if e["type"] == "done")
    assert "message_id" in done
    assert "sources" in done


def test_empty_corpus_yields_done_immediately():
    db = _make_db()
    svc = MagicMock()
    svc.get.return_value = None  # empty corpus

    gen = stream_answer("chat-1", "question?", db, MagicMock(), svc, MagicMock())
    events = asyncio.run(_collect(gen))

    assert len(events) == 1
    assert events[0]["type"] == "done"
    assert events[0]["sources"] == []
