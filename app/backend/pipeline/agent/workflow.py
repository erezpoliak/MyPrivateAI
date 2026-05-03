"""LlamaIndex Workflow: 3-hop agentic RAG.

Hop 1 — direct retrieval (original question) → synthesize → critique (PASS/FAIL)
Hop 2 — decompose question → multi-retrieve (sub-questions + original)
         → synthesize → critique (PASS/FAIL)
Hop 3 — inferential correction: reason over all accumulated context to derive
         an answer, including logical conclusions not explicitly stated.
         No new retrieval.

A PASS at hop 1 or 2 stops early. Hop 3 always returns its answer.
Context accumulates across hops 1 and 2.

Public API
----------
AgentWorkflow(question, llm, retriever, config) — instantiate and call .run() or stream via handler
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from ..common.config import Config
from ..common.utils import get_logger
from .events import TokenEvent, TraceEvent
from .prompts import (
    CORRECT_PROMPT,
    CRITIQUE_PROMPT,
    DECOMPOSE_PROMPT,
    FINAL_SYNTHESIS_PROMPT,
)
from .tools import RetrievalResult, format_context, merge_results, retrieve_context, retrieve_multi

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Trajectory logging
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajectoryEntry:
    """Single step in the agent trajectory."""

    step_name: str
    input_text: str
    output_text: str


@dataclass
class Trajectory:
    """Mutable log of every step the agent took."""

    entries: list[TrajectoryEntry] = field(default_factory=list)
    success: bool = False

    def log(self, step_name: str, input_text: str, output_text: str) -> None:
        self.entries.append(TrajectoryEntry(step_name, input_text, output_text))

    @property
    def num_steps(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# Workflow events
# ---------------------------------------------------------------------------

class RetrieveEvent(Event):
    """Triggers the next hop's retrieval."""
    pass


class SynthesizeEvent(Event):
    """Triggers synthesis from all accumulated context."""
    pass


class CritiqueEvent(Event):
    """Carries synthesised answer and context for quality evaluation."""

    answer: str
    all_context: str


class CorrectEvent(Event):
    """Triggers the final inferential correction step."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sub_questions(text: str) -> list[str]:
    """Extract one sub-question per non-empty line."""
    return [line.strip() for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# AgentWorkflow
# ---------------------------------------------------------------------------

class AgentWorkflow(Workflow):
    """3-hop agentic RAG: direct → decompose/multi-retrieve → infer.

    Each instance is single-use (one question).
    """

    def __init__(
        self,
        question: str,
        llm: Any,
        retriever: Any,
        config: Config,
        timeout: float | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(timeout=timeout, verbose=verbose)
        self._question = question
        self._llm = llm
        self._retriever = retriever
        self._config = config

        self.trajectory = Trajectory()
        self._hop = 0
        self._retrieval_results: list[RetrievalResult] = []
        self._context_texts: list[str] = []

    # ── Steps ──────────────────────────────────────────────────────────

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> RetrieveEvent:
        return RetrieveEvent()

    @step
    async def retrieve(self, ctx: Context, ev: RetrieveEvent) -> SynthesizeEvent:
        """Hop 1: direct retrieval. Hop 2: decompose then multi-retrieve."""
        self._hop += 1
        hop = self._hop

        ctx.write_event_to_stream(
            TraceEvent(step="retrieve", status="start", info=f"hop {hop}")
        )

        if hop == 1:
            logger.info("Hop 1: direct retrieval")
            result = retrieve_context(self._question, self._retriever)
            self.trajectory.log(
                "retrieve_direct",
                self._question,
                f"{len(result.texts)} chunks"
                + (f" (top score={result.scores[0]:.4f})" if result.scores else ""),
            )

        else:
            decompose_prompt = DECOMPOSE_PROMPT.format(question=self._question)
            raw = (await self._llm.acomplete(decompose_prompt)).text.strip()
            sub_questions = _parse_sub_questions(raw)
            self.trajectory.log("decompose", decompose_prompt, "\n".join(sub_questions))
            logger.info("Hop 2: %d sub-questions generated", len(sub_questions))
            for i, sq in enumerate(sub_questions, 1):
                logger.info("  Sub-question %d: %s", i, sq)
            ctx.write_event_to_stream(
                TraceEvent(step="decompose", status="done", info="\n".join(sub_questions))
            )

            queries = [self._question] + sub_questions
            result = retrieve_multi(queries, self._retriever)
            self.trajectory.log(
                "retrieve_multi",
                "; ".join(queries),
                f"{len(result.texts)} unique chunks",
            )

        self._retrieval_results.append(result)
        ctx.write_event_to_stream(
            TraceEvent(step="retrieve", status="done", info=f"{len(result.texts)} chunks")
        )
        return SynthesizeEvent()

    @step
    async def synthesize(
        self, ctx: Context, ev: SynthesizeEvent,
    ) -> CritiqueEvent:
        """Synthesize answer from all accumulated context."""
        merged = merge_results(self._retrieval_results)
        all_context = format_context(merged)
        self._context_texts = merged.texts

        final_prompt = FINAL_SYNTHESIS_PROMPT.format(
            question=self._question,
            context=all_context,
        )

        ctx.write_event_to_stream(
            TraceEvent(step="synthesize", status="start", info="")
        )

        answer_parts: list[str] = []
        async for chunk in await self._llm.astream_complete(final_prompt):
            if chunk.delta:
                ctx.write_event_to_stream(TokenEvent(text=chunk.delta))
                answer_parts.append(chunk.delta)
        answer = "".join(answer_parts).strip()

        self.trajectory.log("synthesize", final_prompt, answer)
        logger.info("Synthesis complete (%.120s)", answer)

        ctx.write_event_to_stream(
            TraceEvent(step="synthesize", status="done", info="")
        )
        return CritiqueEvent(answer=answer, all_context=all_context)

    @step
    async def critique(
        self, ctx: Context, ev: CritiqueEvent,
    ) -> StopEvent | RetrieveEvent | CorrectEvent:
        """PASS/FAIL evaluation after hops 1 and 2."""
        ctx.write_event_to_stream(
            TraceEvent(step="critique", status="start", info="")
        )

        prompt = CRITIQUE_PROMPT.format(
            question=self._question,
            answer=ev.answer,
            context=ev.all_context,
        )
        verdict = (await self._llm.acomplete(prompt)).text.strip()
        self.trajectory.log("critique", prompt, verdict)

        passed = verdict.upper().startswith("PASS")
        logger.info("Hop %d critique: %s", self._hop, "PASS" if passed else "FAIL")

        ctx.write_event_to_stream(
            TraceEvent(step="critique", status="done", info="PASS" if passed else "FAIL")
        )

        if passed:
            self.trajectory.success = True
            return StopEvent(result=ev.answer)

        if self._hop == 1:
            logger.info("Hop 1 failed — advancing to decompose/multi-retrieve")
            return RetrieveEvent()

        logger.info("Hop 2 failed — advancing to inferential correction")
        return CorrectEvent()

    @step
    async def correct(self, ctx: Context, ev: CorrectEvent) -> StopEvent:
        """Reason over all accumulated context to derive an answer. No new retrieval."""
        merged = merge_results(self._retrieval_results)
        all_context = format_context(merged)
        self._context_texts = merged.texts

        correct_prompt = CORRECT_PROMPT.format(
            question=self._question,
            context=all_context,
        )

        ctx.write_event_to_stream(
            TraceEvent(step="correct", status="start", info="")
        )

        answer_parts: list[str] = []
        async for chunk in await self._llm.astream_complete(correct_prompt):
            if chunk.delta:
                ctx.write_event_to_stream(TokenEvent(text=chunk.delta))
                answer_parts.append(chunk.delta)
        answer = "".join(answer_parts).strip()

        self.trajectory.log("correct", correct_prompt, answer)
        logger.info("Inferential correction complete (%.120s)", answer)

        ctx.write_event_to_stream(
            TraceEvent(step="correct", status="done", info="")
        )
        return StopEvent(result=answer)

    # ── Helpers ────────────────────────────────────────────────────────

    def cited_chunks_for(self, answer: str) -> list[dict]:
        """Return context chunks in the order they were rendered into the prompt.

        Index 0 → [1], index 1 → [2], etc. — matches the [n] markers the LLM
        was asked to emit. Pass this list to parse_citations together with the
        answer to resolve markers to source metadata.

        Each dict contains: text, doc_id, title, page_start, page_end.
        """
        merged = merge_results(self._retrieval_results)
        chunks = []
        for text, meta in zip(merged.texts, merged.metadatas):
            chunks.append({
                "text": text,
                "doc_id": meta.get("doc_id", ""),
                "title": meta.get("title", ""),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
            })
        return chunks
