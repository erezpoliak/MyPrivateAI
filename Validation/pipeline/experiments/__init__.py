"""Experiment definitions and shared runner.

Public API:
    spec        — ExperimentSpec (experiment config), Retriever protocol,
                  GenerationResult dataclass, GenerateFn type alias
    runner      — run_experiment() orchestrates the full pipeline lifecycle:
                  load dataset -> build corpus -> build index -> generate answers
                  -> RAGAS evaluation -> persist to SQLite -> print report
                  Also exports parse_args (shared CLI)
    report      — compute_summary(), print_report(), METRIC_KEYS

Experiments (each defines an ExperimentSpec and calls run_experiment):
    baseline        — Vector-only RAG with fixed 512-token chunks (k=5)
    phase1          — SemanticChunker + HybridRetriever
    phase2          — ReAct agent workflow (decompose -> retrieve -> synthesize -> self-correct)
    closed_book     — Llama 3.1 8B with no context (parametric knowledge only)
    llama_gold_ref  — Llama 3.1 8B with Gold_REF as sole context
    ceiling         — GPT-4o with Gold_REF as sole context
"""
