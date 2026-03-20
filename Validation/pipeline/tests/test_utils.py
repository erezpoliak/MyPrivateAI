"""Unit tests for retry decorator, ensure_dir, and doi_to_path."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import call, patch

import pytest

from common.utils import doi_to_path, ensure_dir, retry


# ---------------------------------------------------------------------------
# retry decorator
# ---------------------------------------------------------------------------

class TestRetry:

    def test_succeeds_on_first_attempt_no_sleep(self):
        calls = []

        @retry(max_attempts=3, delay=1.0)
        def func():
            calls.append(1)
            return "ok"

        with patch("time.sleep") as mock_sleep:
            result = func()

        assert result == "ok"
        assert len(calls) == 1
        mock_sleep.assert_not_called()

    def test_retries_and_succeeds_on_second_attempt(self):
        attempts = []

        @retry(max_attempts=3, delay=0.0, backoff=1.0)
        def func():
            attempts.append(1)
            if len(attempts) < 2:
                raise ValueError("fail")
            return "ok"

        with patch("time.sleep"):
            result = func()

        assert result == "ok"
        assert len(attempts) == 2

    def test_raises_after_max_attempts_exhausted(self):
        @retry(max_attempts=3, delay=0.0, backoff=1.0)
        def always_fails():
            raise RuntimeError("always")

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="always"):
                always_fails()

    def test_call_count_equals_max_attempts_on_all_failures(self):
        calls = []

        @retry(max_attempts=4, delay=0.0, backoff=1.0)
        def func():
            calls.append(1)
            raise ValueError("x")

        with patch("time.sleep"):
            with pytest.raises(ValueError):
                func()

        assert len(calls) == 4

    def test_does_not_retry_unlisted_exception(self):
        @retry(max_attempts=3, delay=0.0, exceptions=(ValueError,))
        def func():
            raise TypeError("not in list")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(TypeError):
                func()

        mock_sleep.assert_not_called()

    def test_retries_only_listed_exception(self):
        calls = []

        @retry(max_attempts=3, delay=0.0, backoff=1.0, exceptions=(ValueError,))
        def func():
            calls.append(1)
            raise ValueError("retry me")

        with patch("time.sleep"):
            with pytest.raises(ValueError):
                func()

        assert len(calls) == 3

    def test_exponential_backoff_delays(self):
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def always_fails():
            raise ValueError

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                always_fails()

        # Attempt 1 sleeps 1.0, attempt 2 sleeps 2.0, attempt 3 does not sleep
        assert mock_sleep.call_args_list == [call(1.0), call(2.0)]

    def test_preserves_function_name(self):
        @retry()
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_passes_args_and_kwargs(self):
        received = {}

        @retry()
        def func(a, b, key=None):
            received["a"] = a
            received["b"] = b
            received["key"] = key

        func(1, 2, key="val")
        assert received == {"a": 1, "b": 2, "key": "val"}

    def test_returns_value_from_successful_retry(self):
        attempt = [0]

        @retry(max_attempts=3, delay=0.0)
        def func():
            attempt[0] += 1
            if attempt[0] < 3:
                raise ValueError
            return 42

        with patch("time.sleep"):
            assert func() == 42

    def test_max_attempts_one_never_retries(self):
        calls = []

        @retry(max_attempts=1, delay=0.0)
        def func():
            calls.append(1)
            raise ValueError("immediate")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                func()

        assert len(calls) == 1
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# ensure_dir
# ---------------------------------------------------------------------------

class TestEnsureDir:

    def test_creates_directory_if_not_exists(self, tmp_path):
        new_dir = tmp_path / "new" / "nested"
        result = ensure_dir(new_dir)
        assert new_dir.is_dir()
        assert result == new_dir

    def test_idempotent_on_existing_directory(self, tmp_path):
        existing = tmp_path / "existing"
        existing.mkdir()
        result = ensure_dir(existing)  # should not raise
        assert result == existing

    def test_returns_the_path(self, tmp_path):
        p = tmp_path / "sub"
        returned = ensure_dir(p)
        assert returned is p

    def test_creates_nested_parents(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        ensure_dir(deep)
        assert deep.is_dir()


# ---------------------------------------------------------------------------
# doi_to_path
# ---------------------------------------------------------------------------

class TestDoiToPath:

    def test_slash_replaced_with_double_underscore(self, tmp_path):
        path = doi_to_path(tmp_path, "10.1000/xyz123")
        assert "__" in path.name
        assert "/" not in path.name

    def test_colon_replaced_with_underscore(self, tmp_path):
        path = doi_to_path(tmp_path, "10.1000:abc")
        assert ":" not in path.name
        assert "_" in path.name

    def test_pdf_extension_appended(self, tmp_path):
        path = doi_to_path(tmp_path, "10.1000/xyz")
        assert path.suffix == ".pdf"

    def test_correct_parent_directory(self, tmp_path):
        path = doi_to_path(tmp_path, "10.1000/xyz")
        assert path.parent == tmp_path

    def test_full_transformation(self, tmp_path):
        path = doi_to_path(tmp_path, "10.1038/s41586-020-2649-2")
        assert path == tmp_path / "10.1038__s41586-020-2649-2.pdf"

    def test_doi_without_special_chars_unchanged(self, tmp_path):
        path = doi_to_path(tmp_path, "simple")
        assert path == tmp_path / "simple.pdf"
