"""Unit tests for RAGAS batch evaluation and result parsing."""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from common.metrics import (
    EvalResult,
    EvalSample,
    RAGASEvaluator,
    _nan_to_none,
)


# ---------------------------------------------------------------------------
# _nan_to_none — pure helper
# ---------------------------------------------------------------------------

class TestNanToNone:

    def test_regular_float_returned_unchanged(self):
        assert _nan_to_none(0.75) == pytest.approx(0.75)

    def test_zero_returned_as_zero(self):
        assert _nan_to_none(0.0) == pytest.approx(0.0)

    def test_nan_returns_none(self):
        assert _nan_to_none(float("nan")) is None

    def test_inf_returns_none(self):
        assert _nan_to_none(float("inf")) is None

    def test_neg_inf_returns_none(self):
        assert _nan_to_none(float("-inf")) is None

    def test_none_input_returns_none(self):
        assert _nan_to_none(None) is None

    def test_string_numeric_is_converted(self):
        assert _nan_to_none("0.5") == pytest.approx(0.5)

    def test_non_numeric_string_returns_none(self):
        assert _nan_to_none("not-a-number") is None

    def test_integer_input_is_converted(self):
        assert _nan_to_none(1) == pytest.approx(1.0)

    def test_math_nan_returns_none(self):
        assert _nan_to_none(math.nan) is None


# ---------------------------------------------------------------------------
# EvalSample — dataclass behaviour
# ---------------------------------------------------------------------------

class TestEvalSample:

    def test_default_contexts_is_empty_list(self):
        sample = EvalSample(
            question="q",
            generated_answer="a",
            reference_answer="ref",
        )
        assert sample.contexts == []

    def test_frozen_raises_on_mutation(self):
        sample = EvalSample(question="q", generated_answer="a", reference_answer="ref")
        with pytest.raises((AttributeError, TypeError)):
            sample.question = "other"  # type: ignore[misc]

    def test_contexts_stored_correctly(self):
        sample = EvalSample(
            question="q",
            generated_answer="a",
            reference_answer="ref",
            contexts=["ctx1", "ctx2"],
        )
        assert sample.contexts == ["ctx1", "ctx2"]


# ---------------------------------------------------------------------------
# RAGASEvaluator static helpers — no API key needed
# ---------------------------------------------------------------------------

class TestHasContexts:

    def test_empty_sample_list_returns_false(self):
        assert RAGASEvaluator._has_contexts([]) is False

    def test_all_empty_contexts_returns_false(self):
        samples = [
            EvalSample("q1", "a1", "r1", contexts=[]),
            EvalSample("q2", "a2", "r2", contexts=[]),
        ]
        assert RAGASEvaluator._has_contexts(samples) is False

    def test_one_non_empty_context_returns_true(self):
        samples = [
            EvalSample("q1", "a1", "r1", contexts=[]),
            EvalSample("q2", "a2", "r2", contexts=["some context"]),
        ]
        assert RAGASEvaluator._has_contexts(samples) is True

    def test_all_non_empty_contexts_returns_true(self):
        samples = [
            EvalSample("q1", "a1", "r1", contexts=["ctx"]),
        ]
        assert RAGASEvaluator._has_contexts(samples) is True


class TestSelectMetrics:

    def test_no_context_returns_only_answer_correctness(self):
        metrics = RAGASEvaluator._select_metrics(has_ctx=False)
        names = [type(m).__name__ for m in metrics]
        assert names == ["AnswerCorrectness"]

    def test_with_context_includes_faithfulness_and_recall(self):
        metrics = RAGASEvaluator._select_metrics(has_ctx=True)
        names = [type(m).__name__ for m in metrics]
        assert "Faithfulness" in names
        assert "ContextRecall" in names
        assert "AnswerCorrectness" in names

    def test_with_context_returns_three_metrics(self):
        metrics = RAGASEvaluator._select_metrics(has_ctx=True)
        assert len(metrics) == 3

    def test_no_context_returns_one_metric(self):
        metrics = RAGASEvaluator._select_metrics(has_ctx=False)
        assert len(metrics) == 1


class TestBuildDataset:

    def test_returns_evaluation_dataset_with_correct_count(self):
        from ragas import EvaluationDataset

        samples = [
            EvalSample("q1", "a1", "r1", contexts=["c1"]),
            EvalSample("q2", "a2", "r2", contexts=["c2"]),
        ]
        dataset = RAGASEvaluator._build_dataset(samples)
        assert isinstance(dataset, EvaluationDataset)
        assert len(dataset.samples) == 2

    def test_empty_contexts_stored_as_none_in_dataset(self):
        samples = [EvalSample("q", "a", "ref", contexts=[])]
        dataset = RAGASEvaluator._build_dataset(samples)
        ragas_sample = dataset.samples[0]
        assert ragas_sample.retrieved_contexts is None

    def test_non_empty_contexts_preserved_in_dataset(self):
        samples = [EvalSample("q", "a", "ref", contexts=["ctx1", "ctx2"])]
        dataset = RAGASEvaluator._build_dataset(samples)
        ragas_sample = dataset.samples[0]
        assert ragas_sample.retrieved_contexts == ["ctx1", "ctx2"]

    def test_empty_answer_stored_as_empty_string(self):
        samples = [EvalSample("q", "", "ref")]
        dataset = RAGASEvaluator._build_dataset(samples)
        assert dataset.samples[0].response == ""

    def test_fields_mapped_correctly(self):
        sample = EvalSample("my question", "my answer", "my reference", contexts=["ctx"])
        dataset = RAGASEvaluator._build_dataset([sample])
        rs = dataset.samples[0]
        assert rs.user_input == "my question"
        assert rs.response == "my answer"
        assert rs.reference == "my reference"


# ---------------------------------------------------------------------------
# RAGASEvaluator.__init__ — environment validation
# ---------------------------------------------------------------------------

class TestRAGASEvaluatorInit:

    def test_raises_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            RAGASEvaluator()

    def test_init_succeeds_with_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        with (
            patch("common.metrics.ChatOpenAI"),
            patch("common.metrics.OpenAIEmbeddings"),
            patch("common.metrics.LangchainLLMWrapper"),
            patch("common.metrics.LangchainEmbeddingsWrapper"),
        ):
            evaluator = RAGASEvaluator()
        assert evaluator is not None

    def test_uses_ragas_judge_model_from_config(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from common.config import Config
        cfg = Config()
        cfg.ragas_judge_model = "gpt-4.1-mini"
        with (
            patch("common.metrics.ChatOpenAI") as mock_llm_cls,
            patch("common.metrics.OpenAIEmbeddings"),
            patch("common.metrics.LangchainLLMWrapper"),
            patch("common.metrics.LangchainEmbeddingsWrapper"),
        ):
            RAGASEvaluator(config=cfg)
            mock_llm_cls.assert_called_once_with(model="gpt-4.1-mini")


# ---------------------------------------------------------------------------
# RAGASEvaluator.evaluate_batch — mocked RAGAS evaluate()
# ---------------------------------------------------------------------------

def _make_evaluator(monkeypatch) -> RAGASEvaluator:
    """Build a RAGASEvaluator with all external dependencies mocked."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with (
        patch("common.metrics.ChatOpenAI"),
        patch("common.metrics.OpenAIEmbeddings"),
        patch("common.metrics.LangchainLLMWrapper"),
        patch("common.metrics.LangchainEmbeddingsWrapper"),
    ):
        return RAGASEvaluator()


def _mock_ragas_result(rows: list[dict]):
    """Build a mock RAGAS result whose to_pandas() returns a DataFrame."""
    import pandas as pd

    mock_result = MagicMock()
    mock_result.to_pandas.return_value = pd.DataFrame(rows)
    return mock_result


class TestEvaluateBatch:

    def test_empty_input_returns_empty_list(self, monkeypatch):
        ev = _make_evaluator(monkeypatch)
        assert ev.evaluate_batch([]) == []

    def test_returns_one_result_per_sample(self, monkeypatch):
        ev = _make_evaluator(monkeypatch)
        samples = [
            EvalSample("q1", "a1", "r1", contexts=["c1"]),
            EvalSample("q2", "a2", "r2", contexts=["c2"]),
        ]
        rows = [
            {"faithfulness": 0.9, "context_recall": 0.8, "answer_correctness": 0.7},
            {"faithfulness": 0.6, "context_recall": 0.5, "answer_correctness": 0.4},
        ]
        with patch("common.metrics.evaluate", return_value=_mock_ragas_result(rows)):
            results = ev.evaluate_batch(samples)
        assert len(results) == 2

    def test_with_context_populates_all_metrics(self, monkeypatch):
        ev = _make_evaluator(monkeypatch)
        samples = [EvalSample("q", "a", "ref", contexts=["ctx"])]
        rows = [{"faithfulness": 0.9, "context_recall": 0.8, "answer_correctness": 0.7}]
        with patch("common.metrics.evaluate", return_value=_mock_ragas_result(rows)):
            results = ev.evaluate_batch(samples)
        r = results[0]
        assert r.faithfulness == pytest.approx(0.9)
        assert r.context_recall == pytest.approx(0.8)
        assert r.answer_correctness == pytest.approx(0.7)

    def test_closed_book_context_metrics_are_none(self, monkeypatch):
        ev = _make_evaluator(monkeypatch)
        # No contexts → closed book mode
        samples = [EvalSample("q", "a", "ref", contexts=[])]
        rows = [{"answer_correctness": 0.65}]
        with patch("common.metrics.evaluate", return_value=_mock_ragas_result(rows)):
            results = ev.evaluate_batch(samples)
        r = results[0]
        assert r.faithfulness is None
        assert r.context_recall is None
        assert r.answer_correctness == pytest.approx(0.65)

    def test_nan_scores_converted_to_none(self, monkeypatch):
        ev = _make_evaluator(monkeypatch)
        samples = [EvalSample("q", "a", "ref", contexts=["ctx"])]
        rows = [{"faithfulness": float("nan"), "context_recall": 0.5, "answer_correctness": float("nan")}]
        with patch("common.metrics.evaluate", return_value=_mock_ragas_result(rows)):
            results = ev.evaluate_batch(samples)
        r = results[0]
        assert r.faithfulness is None
        assert r.answer_correctness is None
        assert r.context_recall == pytest.approx(0.5)

    def test_closed_book_uses_only_answer_correctness_metric(self, monkeypatch):
        ev = _make_evaluator(monkeypatch)
        samples = [EvalSample("q", "a", "ref", contexts=[])]
        rows = [{"answer_correctness": 0.5}]
        mock_result = _mock_ragas_result(rows)
        with patch("common.metrics.evaluate", return_value=mock_result) as mock_eval:
            ev.evaluate_batch(samples)
            called_metrics = mock_eval.call_args.kwargs["metrics"]
            names = [type(m).__name__ for m in called_metrics]
            assert names == ["AnswerCorrectness"]

    def test_with_context_calls_all_three_metrics(self, monkeypatch):
        ev = _make_evaluator(monkeypatch)
        samples = [EvalSample("q", "a", "ref", contexts=["ctx"])]
        rows = [{"faithfulness": 1.0, "context_recall": 1.0, "answer_correctness": 1.0}]
        with patch("common.metrics.evaluate", return_value=_mock_ragas_result(rows)) as mock_eval:
            ev.evaluate_batch(samples)
            called_metrics = mock_eval.call_args.kwargs["metrics"]
            assert len(called_metrics) == 3

    def test_result_order_matches_input_order(self, monkeypatch):
        ev = _make_evaluator(monkeypatch)
        samples = [
            EvalSample("q1", "a1", "r1", contexts=["c"]),
            EvalSample("q2", "a2", "r2", contexts=["c"]),
            EvalSample("q3", "a3", "r3", contexts=["c"]),
        ]
        rows = [
            {"faithfulness": 0.1, "context_recall": 0.1, "answer_correctness": 0.1},
            {"faithfulness": 0.2, "context_recall": 0.2, "answer_correctness": 0.2},
            {"faithfulness": 0.3, "context_recall": 0.3, "answer_correctness": 0.3},
        ]
        with patch("common.metrics.evaluate", return_value=_mock_ragas_result(rows)):
            results = ev.evaluate_batch(samples)
        scores = [r.answer_correctness for r in results]
        assert scores == pytest.approx([0.1, 0.2, 0.3])

    def test_evaluate_called_with_raise_exceptions_false(self, monkeypatch):
        ev = _make_evaluator(monkeypatch)
        samples = [EvalSample("q", "a", "ref", contexts=["ctx"])]
        rows = [{"faithfulness": 1.0, "context_recall": 1.0, "answer_correctness": 1.0}]
        with patch("common.metrics.evaluate", return_value=_mock_ragas_result(rows)) as mock_eval:
            ev.evaluate_batch(samples)
            assert mock_eval.call_args.kwargs.get("raise_exceptions") is False
