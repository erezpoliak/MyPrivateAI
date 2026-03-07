"""Shared infrastructure used across the entire pipeline.

Public API:
    config          — Config dataclass (single source of truth for all settings)
    data_loader     — load_dataset() to load QAPairs with paper metadata
    db              — RunDB for persisting experiment runs and results to SQLite
    llm             — get_llm() factory for Ollama LLM instances
    metrics         — RAGASEvaluator for batch evaluation with GPT-4o-mini judge
    utils           — get_logger(), @retry decorator, ensure_dir()
"""
