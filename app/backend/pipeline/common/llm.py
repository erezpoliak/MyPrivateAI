"""Factory for the LLM used across all experiments (MLX, Apple Silicon only)."""

from __future__ import annotations

from typing import Any

from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import PrivateAttr

from .config import Config


class MlxLM(CustomLLM):
    """LlamaIndex CustomLLM backed by mlx-lm (Apple Silicon only)."""

    model_name: str
    temperature: float
    max_tokens: int

    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)

    def __init__(self, model_name: str, temperature: float = 0.1, max_tokens: int = 512) -> None:
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        from mlx_lm import load
        self._model, self._tokenizer = load(model_name)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model_name, num_output=self.max_tokens)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        formatted = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        text = generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=self.temperature),
            verbose=False,
        )
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        formatted = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        accumulated = ""
        for token in stream_generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=self.temperature),
        ):
            accumulated += token
            yield CompletionResponse(text=accumulated, delta=token)

    @llm_completion_callback()
    async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseAsyncGen:
        async def _gen() -> CompletionResponseAsyncGen:
            for response in self.stream_complete(prompt, **kwargs):
                yield response
        return _gen()


def get_llm(config: Config | None = None) -> MlxLM:
    """Return a configured MlxLM instance."""
    config = config or Config()
    return MlxLM(
        model_name=config.mlx_model,
        temperature=config.mlx_temperature,
        max_tokens=config.mlx_max_tokens,
    )
