"""LlamaIndex Workflow: 3-hop agentic RAG.

Hop 1 — direct retrieval (original question) → synthesize → critique (PASS/FAIL)
Hop 2 — decompose question → multi-retrieve (one query per sub-question + original)
         → synthesize → critique (PASS/FAIL + explanation text)
Hop 3 — correct: re-synthesize from all accumulated context + critique text
         (no new retrieval — reasoning correction only)

A PASS at hop 1 or 2 stops early. Hop 3 always returns its answer.
Context accumulates across hops 1 and 2.

Public API
----------
run_agent_workflow(qa, retriever, llm, config) → AgentResult
"""

from __future__ import annotations

import asyncio
import time
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

from common.config import Config
from common.data_loader import QAPair
from common.utils import get_logger
from experiments.spec import GenerationResult, Retriever
from .prompts import (
    CORRECT_PROMPT,
    CRITIQUE_DETAILED_PROMPT,
    CRITIQUE_SIMPLE_PROMPT,
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


@dataclass(frozen=True)
class AgentResult:
    """Workflow output: generation result + trajectory for analysis."""

    generation: GenerationResult
    trajectory: Trajectory


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
    """Triggers the final correction step with critique feedback."""

    critique_text: str


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
    """3-hop agentic RAG: direct → decompose/multi-retrieve → correct.

    Each instance is single-use (one question).
    """

    def __init__(
        self,
        question: str,
        llm: Any,
        retriever: Retriever,
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

            queries = [self._question] + sub_questions
            result = retrieve_multi(queries, self._retriever)
            self.trajectory.log(
                "retrieve_multi",
                "; ".join(queries),
                f"{len(result.texts)} unique chunks",
            )

        self._retrieval_results.append(result)
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
        answer = (await self._llm.acomplete(final_prompt)).text.strip()
        self.trajectory.log("synthesize", final_prompt, answer)
        logger.info("Synthesis complete (%.120s)", answer)

        return CritiqueEvent(answer=answer, all_context=all_context)

    @step
    async def critique(
        self, ctx: Context, ev: CritiqueEvent,
    ) -> StopEvent | RetrieveEvent | CorrectEvent:
        """Hop 1: simple PASS/FAIL. Hop 2: detailed PASS/FAIL + critique text."""
        if self._hop == 1:
            prompt = CRITIQUE_SIMPLE_PROMPT.format(
                question=self._question,
                answer=ev.answer,
                context=ev.all_context,
            )
        else:
            prompt = CRITIQUE_DETAILED_PROMPT.format(
                question=self._question,
                answer=ev.answer,
                context=ev.all_context,
            )

        verdict = (await self._llm.acomplete(prompt)).text.strip()
        self.trajectory.log("critique", prompt, verdict)

        passed = verdict.upper().startswith("PASS")
        logger.info("Hop %d critique: %s", self._hop, "PASS" if passed else "FAIL")

        if passed:
            self.trajectory.success = True
            return StopEvent(result=ev.answer)

        if self._hop == 1:
            logger.info("Hop 1 failed — advancing to decompose/multi-retrieve")
            return RetrieveEvent()

        # Hop 2 failed — extract critique text for correction step
        critique_text = verdict[4:].strip() if verdict.upper().startswith("FAIL") else verdict
        logger.info("Hop 2 failed — advancing to correction with critique: %.120s", critique_text)
        return CorrectEvent(critique_text=critique_text)

    @step
    async def correct(self, ctx: Context, ev: CorrectEvent) -> StopEvent:
        """Re-synthesize using all accumulated context + critique feedback. No new retrieval."""
        merged = merge_results(self._retrieval_results)
        all_context = format_context(merged)
        self._context_texts = merged.texts

        correct_prompt = CORRECT_PROMPT.format(
            question=self._question,
            critique=ev.critique_text,
            context=all_context,
        )
        answer = (await self._llm.acomplete(correct_prompt)).text.strip()
        self.trajectory.log("correct", correct_prompt, answer)
        logger.info("Correction complete (%.120s)", answer)

        return StopEvent(result=answer)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def _run_async(
    qa: QAPair,
    retriever: Retriever,
    llm: Any,
    config: Config,
) -> AgentResult:
    workflow = AgentWorkflow(
        question=qa.question,
        llm=llm,
        retriever=retriever,
        config=config,
    )

    t0 = time.perf_counter()
    try:
        answer = await workflow.run()
        return AgentResult(
            generation=GenerationResult(
                qa=qa,
                generated_answer=str(answer),
                contexts=workflow._context_texts,
                latency_s=time.perf_counter() - t0,
                error=None,
            ),
            trajectory=workflow.trajectory,
        )
    except Exception as exc:
        logger.error("Agent workflow failed for Q %s: %s", qa.id, exc)
        return AgentResult(
            generation=GenerationResult(
                qa=qa,
                generated_answer="",
                contexts=[],
                latency_s=time.perf_counter() - t0,
                error=str(exc),
            ),
            trajectory=workflow.trajectory,
        )


def run_agent_workflow(
    qa: QAPair,
    retriever: Retriever,
    llm: Any,
    config: Config,
) -> AgentResult:
    """Run the 3-hop agentic workflow synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(_run_async(qa, retriever, llm, config))
