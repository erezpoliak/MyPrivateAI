"""LlamaIndex Workflow: hop-specific agentic RAG.

Three fixed hops, each using a different retrieval strategy:
  Hop 1 — direct retrieval (original question) → synthesize → critique
  Hop 2 — HyDE (hypothetical answer as query)  → synthesize → critique
  Hop 3 — query rewrite (independent rephrase)  → synthesize → stop (no critique)

Critique is PASS/FAIL only. A PASS at any hop stops early. A FAIL advances
to the next hop. Hop 3 always returns its answer regardless of quality.
Context is accumulated across all hops before each synthesis.

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
    CRITIQUE_PROMPT,
    FINAL_SYNTHESIS_PROMPT,
    HYDE_PROMPT,
    REWRITE_PROMPT,
)
from .tools import RetrievalResult, format_context, merge_results, retrieve_context

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
    """Triggers the next hop's retrieval strategy."""
    pass


class SynthesizeEvent(Event):
    """Triggers synthesis from all accumulated context."""
    pass


class CritiqueEvent(Event):
    """Carries the synthesised answer for quality evaluation."""

    answer: str
    all_context: str


# ---------------------------------------------------------------------------
# AgentWorkflow
# ---------------------------------------------------------------------------

class AgentWorkflow(Workflow):
    """Hop-specific agentic RAG with three fixed retrieval strategies.

    Hop 1: direct retrieval → synthesize → critique
    Hop 2: HyDE            → synthesize → critique
    Hop 3: query rewrite   → synthesize → stop (final, no critique)

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
        """Select retrieval strategy for the current hop and fetch context."""
        self._hop += 1
        hop = self._hop
        max_hops = self._config.max_agent_hops

        if hop == 1:
            query = self._question
            logger.info("Hop 1/%d: direct retrieval", max_hops)

        elif hop == 2:
            hyde_prompt = HYDE_PROMPT.format(question=self._question)
            query = (await self._llm.acomplete(hyde_prompt)).text.strip()
            self.trajectory.log("hyde", hyde_prompt, query)
            logger.info("Hop 2/%d HyDE query: %.120s", max_hops, query)

        else:
            rewrite_prompt = REWRITE_PROMPT.format(question=self._question)
            query = (await self._llm.acomplete(rewrite_prompt)).text.strip()
            self.trajectory.log("rewrite", rewrite_prompt, query)
            logger.info("Hop 3/%d rewritten query: %.120s", max_hops, query)

        result = retrieve_context(query, self._retriever)
        self._retrieval_results.append(result)
        self.trajectory.log(
            "retrieve",
            query,
            f"{len(result.texts)} chunks"
            + (f" (top score={result.scores[0]:.4f})" if result.scores else ""),
        )

        return SynthesizeEvent()

    @step
    async def synthesize(
        self, ctx: Context, ev: SynthesizeEvent,
    ) -> CritiqueEvent | StopEvent:
        """Synthesize an answer from all accumulated context.

        On hop 3 (final hop) returns StopEvent directly, skipping critique.
        """
        merged = merge_results(self._retrieval_results)
        all_context = format_context(merged)
        self._context_texts = merged.texts

        final_prompt = FINAL_SYNTHESIS_PROMPT.format(
            question=self._question,
            context=all_context,
        )
        answer = (await self._llm.acomplete(final_prompt)).text.strip()
        self.trajectory.log("synthesize_final", final_prompt, answer)
        logger.info("Synthesis complete (%.120s)", answer)

        if self._hop >= self._config.max_agent_hops:
            logger.info("Hop 3 final answer — skipping critique")
            return StopEvent(result=answer)

        return CritiqueEvent(answer=answer, all_context=all_context)

    @step
    async def critique(
        self, ctx: Context, ev: CritiqueEvent,
    ) -> StopEvent | RetrieveEvent:
        """PASS/FAIL evaluation. PASS stops early; FAIL advances to next hop."""
        critique_prompt = CRITIQUE_PROMPT.format(
            question=self._question,
            answer=ev.answer,
            context=ev.all_context,
        )
        verdict = (await self._llm.acomplete(critique_prompt)).text.strip()
        self.trajectory.log("critique", critique_prompt, verdict)

        passed = verdict.upper().startswith("PASS")
        logger.info("Critique verdict: %s", "PASS" if passed else "FAIL")

        if passed:
            self.trajectory.success = True
            return StopEvent(result=ev.answer)

        logger.info("Critique failed, advancing to hop %d", self._hop + 1)
        return RetrieveEvent()


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
    """Run the hop-specific agent workflow synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(_run_async(qa, retriever, llm, config))
