"""Factory for the Ollama LLM used across all experiments."""

from __future__ import annotations

from llama_index.llms.ollama import Ollama

from .config import Config
from .utils import get_logger

logger = get_logger(__name__)


def get_llm(config: Config | None = None) -> Ollama:
    """Return a configured Ollama LLM instance."""
    config = config or Config()
    logger.info(
        "Connecting to Ollama at %s (model: %s)",
        config.ollama_base_url,
        config.ollama_model,
    )
    return Ollama(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        request_timeout=config.ollama_timeout,
        temperature=config.ollama_temperature,
    )
