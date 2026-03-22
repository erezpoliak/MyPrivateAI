"""Factory for the Ollama LLM used across all experiments."""

from __future__ import annotations

from llama_index.llms.ollama import Ollama

from .config import Config


def get_llm(config: Config | None = None) -> Ollama:
    """Return a configured Ollama LLM instance."""
    config = config or Config()
    return Ollama(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        request_timeout=config.ollama_timeout,
        temperature=config.ollama_temperature,
    )
