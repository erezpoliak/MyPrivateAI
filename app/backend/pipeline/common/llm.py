"""Factory for the LLM used across all experiments (MLX, Apple Silicon only)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue as thread_queue
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

# All MLX GPU ops must run on the thread that owns the Metal stream.
# A single-worker executor guarantees every call lands on the same thread.
_mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


class MlxLM(CustomLLM):
    """LlamaIndex CustomLLM backed by mlx-lm (Apple Silicon only)."""

    model_name: str
    temperature: float
    max_tokens: int

    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)

    def __init__(self, model_name: str, temperature: float = 0.1, max_tokens: int = 512) -> None:
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        def _load():
            from mlx_lm import load
            return load(model_name)
        self._model, self._tokenizer = _mlx_executor.submit(_load).result()

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model_name, num_output=self.max_tokens)

    def _format_prompt(self, prompt: str) -> str:
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        def _run():
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
            return generate(
                self._model,
                self._tokenizer,
                prompt=self._format_prompt(prompt),
                max_tokens=self.max_tokens,
                sampler=make_sampler(temp=self.temperature),
                verbose=False,
            )
        return CompletionResponse(text=_mlx_executor.submit(_run).result())

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        q: thread_queue.Queue = thread_queue.Queue()

        def _produce():
            try:
                from mlx_lm import stream_generate
                from mlx_lm.sample_utils import make_sampler
                accumulated = ""
                for token in stream_generate(
                    self._model,
                    self._tokenizer,
                    prompt=self._format_prompt(prompt),
                    max_tokens=self.max_tokens,
                    sampler=make_sampler(temp=self.temperature),
                ):
                    accumulated += token.text
                    q.put(CompletionResponse(text=accumulated, delta=token.text))
                q.put(None)
            except Exception as exc:
                q.put(exc)

        _mlx_executor.submit(_produce)
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseAsyncGen:
        async def _gen() -> CompletionResponseAsyncGen:
            q: thread_queue.Queue = thread_queue.Queue()
            loop = asyncio.get_running_loop()

            def _produce():
                try:
                    from mlx_lm import stream_generate
                    from mlx_lm.sample_utils import make_sampler
                    accumulated = ""
                    for token in stream_generate(
                        self._model,
                        self._tokenizer,
                        prompt=self._format_prompt(prompt),
                        max_tokens=self.max_tokens,
                        sampler=make_sampler(temp=self.temperature),
                    ):
                        accumulated += token.text
                        q.put(CompletionResponse(text=accumulated, delta=token.text))
                    q.put(None)
                except Exception as exc:
                    q.put(exc)

            _mlx_executor.submit(_produce)
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

        return _gen()


def get_llm(config: Config | None = None) -> MlxLM:
    """Return a configured MlxLM instance."""
    config = config or Config()
    return MlxLM(
        model_name=config.mlx_model,
        temperature=config.mlx_temperature,
        max_tokens=config.mlx_max_tokens,
    )
