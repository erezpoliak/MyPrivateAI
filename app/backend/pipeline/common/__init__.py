"""Shared infrastructure used across the pipeline.

Public API:
    config  — Config dataclass (single source of truth for all path/model settings)
    llm     — get_llm() factory for MLX LLM instances
    utils   — get_logger(), @retry decorator, ensure_dir()
"""
