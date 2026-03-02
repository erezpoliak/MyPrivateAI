"""Shared utilities: logging, retry, hashing, directory helpers, NLTK bootstrap."""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import nltk

from .config import Config

F = TypeVar("F", bound=Callable[..., Any])

# ── Structured logging ──────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent format across the pipeline."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Retry decorator ─────────────────────────────────────────────────────────

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[F], F]:
    """Retry a function on failure with exponential backoff."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _delay = delay
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        get_logger(func.__module__).warning(
                            "%s attempt %d/%d failed: %s — retrying in %.1fs",
                            func.__name__, attempt, max_attempts, exc, _delay,
                        )
                        time.sleep(_delay)
                        _delay *= backoff
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator


# ── Hashing ─────────────────────────────────────────────────────────────────

def hash_string(text: str) -> str:
    """Return a stable SHA-256 hex digest for *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Directory helpers ───────────────────────────────────────────────────────

def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it doesn't exist; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── DOI helpers ────────────────────────────────────────────────────────────

def doi_to_path(papers_dir: Path, doi: str) -> Path:
    """Return the local PDF path for a DOI inside *papers_dir*."""
    return papers_dir / (doi.replace("/", "__").replace(":", "_") + ".pdf")


# ── NLTK bootstrap ──────────────────────────────────────────────────────────

def _bootstrap_nltk(config: Config | None = None) -> None:
    """Download punkt and stopwords into the project-local NLTK data dir."""
    config = config or Config()
    nltk_dir = str(config.nltk_data_dir)

    # Make sure NLTK looks at our project-local path
    os.environ.setdefault("NLTK_DATA", nltk_dir)
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)

    ensure_dir(config.nltk_data_dir)
    for pkg in ("punkt_tab", "stopwords"):
        nltk.download(pkg, download_dir=nltk_dir, quiet=True)


# Run once on import
_bootstrap_nltk()
