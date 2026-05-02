"""Agentic RAG — critique-driven multi-hop workflow.

Hop 1: retrieve (original question) → synthesize → critique
Hop 2+: decompose (informed by critique) → retrieve → synthesize → critique
  - PASS at any hop → stop early
  - FAIL + no hops left → one-shot correction → stop

Public API (used by phase2 experiment):
    prompts     — DECOMPOSE_PROMPT, FINAL_SYNTHESIS_PROMPT, CRITIQUE_PROMPT,
                  CORRECTION_PROMPT, format_prior_queries()
    tools       — retrieve_context(), merge_results(), format_context()
    workflow    — run_agent_workflow() entry point + AgentResult / Trajectory types
"""
