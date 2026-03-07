"""RAGAS evaluation wrapper for the validation pipeline.

Provides ``RAGASEvaluator`` which runs GPT-4o-mini as LLM judge for **all**
experiments for consistent evaluation across the board. 

Closed-book mode (empty contexts) returns ``None`` for context-dependent
metrics (faithfulness, answer_relevancy, context_precision, context_recall)
and only computes answer_correctness + answer_similarity.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness

from .config import Config
from .utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalSample:
    """A single sample to be evaluated by RAGAS."""

    question: str
    generated_answer: str
    reference_answer: str
    contexts: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """RAGAS metric scores for a single sample.

    Context-dependent metrics are ``None`` when contexts are empty
    (Closed Book experiment).
    """

    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None
    answer_similarity: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nan_to_none(value: object) -> Optional[float]:
    """Convert NaN / non-finite floats to ``None``."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGASEvaluator:
    """Evaluates RAG pipeline outputs using RAGAS metrics with an LLM judge.

    Parameters
    ----------
    config : Config, optional
        Pipeline configuration (reads ``ragas_judge_model`` from it).
    """

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()
        self._judge_model = self._config.ragas_judge_model

        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY must be set for RAGAS evaluation"
            )

        self._llm = LangchainLLMWrapper(
            ChatOpenAI(model=self._judge_model)
        )
        self._embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings()
        )

    # -- public API ----------------------------------------------------------

    def evaluate_batch(
        self,
        samples: Sequence[EvalSample],
    ) -> list[EvalResult]:
        """Run RAGAS metrics on a batch of samples.

        For batches **with** retrieved contexts every metric is computed.
        For batches **without** contexts (Closed Book) only
        ``answer_correctness`` and ``answer_similarity`` are computed;
        context-dependent metrics are returned as ``None``.

        Returns one :class:`EvalResult` per input sample, in the same order.
        """
        if not samples:
            return []

        has_ctx = self._has_contexts(samples)
        metrics = self._select_metrics(has_ctx)
        dataset = self._build_dataset(samples)

        log.info(
            "Running RAGAS evaluation — %d samples, %d metrics, judge=%s",
            len(samples),
            len(metrics),
            self._judge_model,
        )

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._llm,
            embeddings=self._embeddings,
            raise_exceptions=False,
        )

        df = result.to_pandas()
        results: list[EvalResult] = []

        for _, row in df.iterrows():
            results.append(
                EvalResult(
                    faithfulness=_nan_to_none(row.get("faithfulness")) if has_ctx else None,
                    answer_relevancy=_nan_to_none(row.get("answer_relevancy")) if has_ctx else None,
                    context_precision=_nan_to_none(row.get("context_precision")) if has_ctx else None,
                    context_recall=_nan_to_none(row.get("context_recall")) if has_ctx else None,
                    answer_correctness=_nan_to_none(row.get("answer_correctness")),
                    answer_similarity=_nan_to_none(row.get("answer_similarity")),
                )
            )

        log.info("RAGAS evaluation complete — %d results", len(results))
        return results

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _has_contexts(samples: Sequence[EvalSample]) -> bool:
        """Check if any sample in the batch has non-empty contexts."""
        return any(s.contexts for s in samples)

    @staticmethod
    def _select_metrics(has_ctx: bool) -> list:
        """Return the appropriate RAGAS metric instances."""
        answer_metrics = [AnswerCorrectness(), AnswerSimilarity()]
        if not has_ctx:
            return answer_metrics
        context_metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ]
        return context_metrics + answer_metrics

    @staticmethod
    def _build_dataset(samples: Sequence[EvalSample]) -> EvaluationDataset:
        """Convert ``EvalSample`` objects into a RAGAS ``EvaluationDataset``."""
        ragas_samples = [
            SingleTurnSample(
                user_input=s.question,
                response=s.generated_answer or "",
                reference=s.reference_answer,
                retrieved_contexts=s.contexts if s.contexts else None,
            )
            for s in samples
        ]
        return EvaluationDataset(samples=ragas_samples)
