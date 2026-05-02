from __future__ import annotations

from llama_index.core.workflow import Event


class TraceEvent(Event):
    step: str
    status: str
    info: str


class TokenEvent(Event):
    text: str
