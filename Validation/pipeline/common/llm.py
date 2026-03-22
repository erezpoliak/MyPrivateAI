"""Factory for the Ollama LLM used across all experiments."""

from __future__ import annotations

from llama_index.llms.ollama import Ollama

from .config import Config


def get_llm(config: Config | None = None, thinking: bool = True) -> Ollama:
    """Return a configured Ollama LLM instance.

    Parameters
    ----------
    thinking : bool
        Whether to enable Qwen3's thinking mode (default: True).
        Pass ``False`` for synthesis steps where chain-of-thought adds
        latency without quality benefit.
    """
    config = config or Config()
    return Ollama(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        request_timeout=config.ollama_timeout,
        temperature=config.ollama_temperature,
        thinking=thinking,
    )
