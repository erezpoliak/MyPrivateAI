"""Agentic RAG — multi-hop decompose → retrieve → synthesize → self-correct.

Public API (used by phase2 experiment):
    prompts     — DECOMPOSE_PROMPT, SYNTHESIS_PROMPT, CRITIQUE_PROMPT, CORRECTION_PROMPT
                  Also exports format_intermediate_context() helper
    tools       — Retrieval tools exposed to agent workflow steps
    workflow    — LlamaIndex Workflow: multi-hop agentic loop with trajectory logging
"""
