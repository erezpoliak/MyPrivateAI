"""LlamaIndex Workflow: critique-driven multi-hop agentic RAG.

Hop 1: retrieve (original question) → full synthesize → critique
Hop 2+: decompose (informed by critique) → retrieve → full synthesize → critique
  - PASS at any hop → stop early
  - FAIL + hops remaining → decompose targets the gap identified by critique
  - FAIL + no hops left → one-shot correction as last resort → stop

Public API
----------
run_agent_workflow(qa, retriever, llm, config) → AgentResult
    Runs the critique-driven loop and returns the generation result
    together with a step-by-step trajectory log.
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
    CORRECTION_PROMPT,
    CRITIQUE_PROMPT,
    DECOMPOSE_PROMPT,
    FINAL_SYNTHESIS_PROMPT,
    format_prior_queries,
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

class DecomposeEvent(Event):
    """Triggers a retrieve (hop 1) or decompose → retrieve (hop 2+) cycle."""

    pass


class SynthesizeEvent(Event):
    """Triggers full synthesis of the original question from all context."""

    pass


class CritiqueEvent(Event):
    """Carries the synthesised answer for quality evaluation."""

    answer: str
    all_context: str


class CorrectEvent(Event):
    """Carries a failed answer + critique for correction."""

    answer: str
    all_context: str
    critique: str


# ---------------------------------------------------------------------------
# AgentWorkflow
# ---------------------------------------------------------------------------

class AgentWorkflow(Workflow):
    """Critique-driven multi-hop agentic RAG.

    Hop 1: retrieve (original question) → full synthesize → critique.
    Hop 2+: decompose (informed by critique) → retrieve → full synthesize → critique.
    Critique decides: stop (PASS), loop (FAIL + hops left), or correct (FAIL + done).

    Each instance is single-use (one question). State lives on instance
    attributes so callers can inspect ``trajectory`` after the run.

    Parameters
    ----------
    thinking_llm :
        LLM with thinking ON — used for decompose and critique steps.
    synthesis_llm :
        LLM with thinking OFF — used for synthesize and correct steps.
    """

    def __init__(
        self,
        question: str,
        thinking_llm: Any,
        synthesis_llm: Any,
        retriever: Retriever,
        config: Config,
        timeout: float | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(timeout=timeout, verbose=verbose)
        self._question = question
        self._thinking_llm = thinking_llm
        self._synthesis_llm = synthesis_llm
        self._retriever = retriever
        self._config = config

        # Per-run mutable state
        self.trajectory = Trajectory()
        self._hop = 0
        self._queries: list[str] = []
        self._retrieval_results: list[RetrievalResult] = []
        self._context_texts: list[str] = []
        self._last_critique: str = ""

    # ── Steps ──────────────────────────────────────────────────────────

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> DecomposeEvent:
        """Kick off the first hop."""
        return DecomposeEvent()

    @step
    async def decompose_and_retrieve(
        self, ctx: Context, ev: DecomposeEvent,
    ) -> SynthesizeEvent:
        """Retrieve context for the current hop.

        Hop 1 uses the original question directly. Hop 2+ decomposes a
        targeted sub-question informed by the critique feedback.
        The sub-question is only a retrieval query — it is not answered
        separately. All answering happens in the full synthesis step.
        """
        self._hop += 1
        hop = self._hop
        max_hops = self._config.max_agent_hops

        # ── Query selection ───────────────────────────────────────────
        if hop == 1:
            query = self._question
            logger.info("Hop 1/%d: using original question directly", max_hops)
        else:
            prior_queries = format_prior_queries(self._queries)
            critique_feedback = (
                f"The previous answer was rejected:\n{self._last_critique}\n\n"
                "Focus your next sub-question on the identified gaps.\n\n"
                if self._last_critique else ""
            )
            decompose_prompt = DECOMPOSE_PROMPT.format(
                question=self._question,
                prior_queries=prior_queries,
                critique_feedback=critique_feedback,
            )
            query = (await self._thinking_llm.acomplete(decompose_prompt)).text.strip()
            self.trajectory.log("decompose", decompose_prompt, query)
            logger.info("Hop %d/%d sub-question: %s", hop, max_hops, query)

        self._queries.append(query)

        # ── Retrieve ──────────────────────────────────────────────────
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
    ) -> CritiqueEvent:
        """Produce a final answer from all accumulated context."""
        merged = merge_results(self._retrieval_results)
        all_context = format_context(merged)
        self._context_texts = merged.texts

        final_prompt = FINAL_SYNTHESIS_PROMPT.format(
            question=self._question,
            context=all_context,
        )
        answer = (await self._synthesis_llm.acomplete(final_prompt)).text.strip()
        self.trajectory.log("synthesize_final", final_prompt, answer)
        logger.info("Final synthesis complete (%.120s)", answer)

        return CritiqueEvent(answer=answer, all_context=all_context)

    @step
    async def critique(
        self, ctx: Context, ev: CritiqueEvent,
    ) -> StopEvent | CorrectEvent | DecomposeEvent:
        """Evaluate the answer; stop early on PASS or loop back for more hops."""
        critique_prompt = CRITIQUE_PROMPT.format(
            question=self._question,
            answer=ev.answer,
            context=ev.all_context,
        )
        verdict = (await self._thinking_llm.acomplete(critique_prompt)).text.strip()
        self.trajectory.log("critique", critique_prompt, verdict)

        passed = verdict.upper().startswith("PASS")
        logger.info("Critique verdict: %s", "PASS" if passed else "FAIL")

        if passed:
            self.trajectory.success = True
            return StopEvent(result=ev.answer)

        # Hops remaining → feed critique into next decompose
        if self._hop < self._config.max_agent_hops:
            self._last_critique = verdict
            logger.info("Critique failed, triggering hop %d", self._hop + 1)
            return DecomposeEvent()

        # Out of hops → one-shot correction as last resort
        return CorrectEvent(
            answer=ev.answer,
            all_context=ev.all_context,
            critique=verdict,
        )

    @step
    async def correct(self, ctx: Context, ev: CorrectEvent) -> StopEvent:
        """Produce a corrected answer addressing critique deficiencies."""
        correction_prompt = CORRECTION_PROMPT.format(
            question=self._question,
            answer=ev.answer,
            context=ev.all_context,
            critique=ev.critique,
        )
        corrected = (await self._synthesis_llm.acomplete(correction_prompt)).text.strip()
        self.trajectory.log("correct", correction_prompt, corrected)
        logger.info("Correction applied (%.120s)", corrected)

        return StopEvent(result=corrected)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def _run_async(
    qa: QAPair,
    retriever: Retriever,
    thinking_llm: Any,
    synthesis_llm: Any,
    config: Config,
) -> AgentResult:
    """Async implementation of the agent workflow."""
    workflow = AgentWorkflow(
        question=qa.question,
        thinking_llm=thinking_llm,
        synthesis_llm=synthesis_llm,
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
    thinking_llm: Any,
    synthesis_llm: Any,
    config: Config,
) -> AgentResult:
    """Run the critique-driven agent workflow synchronously.

    Creates an ``AgentWorkflow``, executes the critique-driven loop
    (retrieve → synthesize → critique, with decompose on hop 2+),
    and returns the generation result together with a full trajectory log.

    Parameters
    ----------
    thinking_llm :
        LLM with thinking ON — used for decompose and critique.
    synthesis_llm :
        LLM with thinking OFF — used for synthesize and correct.
    """
    return asyncio.run(_run_async(qa, retriever, thinking_llm, synthesis_llm, config))
