"""Unit tests for RunDB SQLite persistence."""
from __future__ import annotations

import json

import pytest

from common.config import Config
from common.db import RunDB


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path) -> RunDB:
    """Return a RunDB backed by a temp file (not the real results/runs.db)."""
    cfg = Config()
    cfg.db_path = tmp_path / "test_runs.db"
    return RunDB(config=cfg)


def _start(db: RunDB, experiment: str = "baseline", corpus_mode: str = "fetched") -> int:
    return db.start_run(experiment, corpus_mode, {"key": "val"}, notes="test")


# ---------------------------------------------------------------------------
# start_run / finish_run
# ---------------------------------------------------------------------------

class TestStartRun:

    def test_returns_integer_run_id(self, db):
        run_id = _start(db)
        assert isinstance(run_id, int)

    def test_run_ids_are_autoincremented(self, db):
        id1 = _start(db)
        id2 = _start(db)
        assert id2 == id1 + 1

    def test_run_appears_in_list_runs(self, db):
        run_id = _start(db, experiment="phase1")
        runs = db.list_runs()
        ids = [r["run_id"] for r in runs]
        assert run_id in ids

    def test_run_stores_experiment_name(self, db):
        run_id = _start(db, experiment="phase2")
        run = next(r for r in db.list_runs() if r["run_id"] == run_id)
        assert run["experiment"] == "phase2"

    def test_run_stores_corpus_mode(self, db):
        run_id = _start(db, corpus_mode="gold_ref")
        run = next(r for r in db.list_runs() if r["run_id"] == run_id)
        assert run["corpus_mode"] == "gold_ref"

    def test_run_stores_notes(self, db):
        run_id = db.start_run("baseline", "fetched", {}, notes="smoke test")
        run = next(r for r in db.list_runs() if r["run_id"] == run_id)
        assert run["notes"] == "smoke test"

    def test_config_json_is_serialised(self, db):
        run_id = db.start_run("x", "none", {"nested": [1, 2]})
        run = next(r for r in db.list_runs() if r["run_id"] == run_id)
        parsed = json.loads(run["config_json"])
        assert parsed["nested"] == [1, 2]

    def test_started_at_is_set(self, db):
        run_id = _start(db)
        run = next(r for r in db.list_runs() if r["run_id"] == run_id)
        assert run["started_at"] is not None

    def test_finished_at_is_none_before_finish(self, db):
        run_id = _start(db)
        run = next(r for r in db.list_runs() if r["run_id"] == run_id)
        assert run["finished_at"] is None


class TestFinishRun:

    def test_sets_finished_at(self, db):
        run_id = _start(db)
        db.finish_run(run_id)
        run = next(r for r in db.list_runs() if r["run_id"] == run_id)
        assert run["finished_at"] is not None

    def test_finished_at_after_started_at(self, db):
        run_id = _start(db)
        db.finish_run(run_id)
        run = next(r for r in db.list_runs() if r["run_id"] == run_id)
        assert run["finished_at"] >= run["started_at"]


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------

class TestListRuns:

    def test_empty_db_returns_empty_list(self, db):
        assert db.list_runs() == []

    def test_returns_all_runs(self, db):
        _start(db)
        _start(db)
        _start(db)
        assert len(db.list_runs()) == 3

    def test_ordered_descending_by_run_id(self, db):
        _start(db)
        _start(db)
        _start(db)
        runs = db.list_runs()
        ids = [r["run_id"] for r in runs]
        assert ids == sorted(ids, reverse=True)


# ---------------------------------------------------------------------------
# insert_result / get_results
# ---------------------------------------------------------------------------

class TestInsertResult:

    def test_returns_integer_result_id(self, db):
        run_id = _start(db)
        result_id = db.insert_result(
            run_id,
            question_id="q1",
            source_idx="S1",
            complexity=1,
            answer_type="factoid",
        )
        assert isinstance(result_id, int)

    def test_get_results_returns_inserted_row(self, db):
        run_id = _start(db)
        db.insert_result(
            run_id,
            question_id="q1",
            source_idx="S1",
            complexity=2,
            answer_type="factoid",
            generated_answer="answer text",
            answer_correctness=0.85,
        )
        results = db.get_results(run_id)
        assert len(results) == 1
        assert results[0]["question_id"] == "q1"
        assert results[0]["answer_correctness"] == pytest.approx(0.85)

    def test_get_results_filters_by_run_id(self, db):
        run1 = _start(db)
        run2 = _start(db)
        db.insert_result(run1, question_id="q1", source_idx="S1", complexity=1, answer_type="x")
        db.insert_result(run2, question_id="q2", source_idx="S2", complexity=1, answer_type="x")
        assert len(db.get_results(run1)) == 1
        assert db.get_results(run1)[0]["question_id"] == "q1"
        assert len(db.get_results(run2)) == 1
        assert db.get_results(run2)[0]["question_id"] == "q2"

    def test_empty_run_returns_empty_results(self, db):
        run_id = _start(db)
        assert db.get_results(run_id) == []

    def test_list_value_serialised_as_json_string(self, db):
        run_id = _start(db)
        db.insert_result(
            run_id,
            question_id="q1",
            source_idx="S1",
            complexity=1,
            answer_type="x",
            contexts_json=["chunk a", "chunk b"],
        )
        row = db.get_results(run_id)[0]
        # stored as JSON string in DB
        parsed = json.loads(row["contexts_json"])
        assert parsed == ["chunk a", "chunk b"]

    def test_multiple_results_all_returned(self, db):
        run_id = _start(db)
        for i in range(5):
            db.insert_result(
                run_id,
                question_id=f"q{i}",
                source_idx="S1",
                complexity=1,
                answer_type="x",
            )
        assert len(db.get_results(run_id)) == 5


# ---------------------------------------------------------------------------
# save_summary / get_summary
# ---------------------------------------------------------------------------

class TestSummary:

    def _summary(self) -> dict:
        return {
            "num_questions": 10,
            "avg_faithfulness": 0.8,
            "avg_context_recall": 0.7,
            "avg_answer_correctness": 0.75,
            "avg_latency_s": 1.2,
            "avg_trajectory_steps": 2.1,
            "trajectory_success_rate": 0.9,
            "breakdown_json": "{}",
        }

    def test_get_summary_returns_none_for_unknown_run(self, db):
        assert db.get_summary(999) is None

    def test_save_and_get_summary_round_trip(self, db):
        run_id = _start(db)
        db.save_summary(run_id, self._summary())
        result = db.get_summary(run_id)
        assert result is not None
        assert result["num_questions"] == 10
        assert result["avg_faithfulness"] == pytest.approx(0.8)

    def test_save_summary_stores_run_id(self, db):
        run_id = _start(db)
        db.save_summary(run_id, self._summary())
        result = db.get_summary(run_id)
        assert result["run_id"] == run_id

    def test_save_summary_replace_on_duplicate_run(self, db):
        run_id = _start(db)
        db.save_summary(run_id, {**self._summary(), "num_questions": 5})
        db.save_summary(run_id, {**self._summary(), "num_questions": 99})
        result = db.get_summary(run_id)
        assert result["num_questions"] == 99

    def test_summaries_isolated_between_runs(self, db):
        r1 = _start(db)
        r2 = _start(db)
        db.save_summary(r1, {**self._summary(), "num_questions": 1})
        db.save_summary(r2, {**self._summary(), "num_questions": 2})
        assert db.get_summary(r1)["num_questions"] == 1
        assert db.get_summary(r2)["num_questions"] == 2
