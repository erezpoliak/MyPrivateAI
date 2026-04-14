"""Prompt templates for the critique-driven agentic RAG workflow.

Three templates drive the hop-specific retrieval strategies:

  Hop 1 — direct retrieval using the original question
  Hop 2 — HyDE: generate a hypothetical answer, use it as retrieval query
  Hop 3 — query rewrite: rephrase independently, no critique (final hop)

Placeholders
------------
HYDE_PROMPT
    {question}  — original user question

REWRITE_PROMPT
    {question}  — original user question

FINAL_SYNTHESIS_PROMPT
    {question}  — original user question
    {context}   — all accumulated context across hops

CRITIQUE_PROMPT  (hops 1–2 only)
    {question}  — original user question
    {answer}    — synthesised answer to evaluate
    {context}   — all retrieved context used during synthesis
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# HyDE — generate a hypothetical answer to use as retrieval query (hop 2)
# ---------------------------------------------------------------------------
HYDE_PROMPT = (
    "Write 1-2 sentences from a scientific paper that directly contain the answer "
    "to the question below. Be specific, use precise technical language, no explanations.\n\n"
    "Question: {question}\n\n"
    "Sentences: "
)

# ---------------------------------------------------------------------------
# REWRITE — rephrase the question independently for broader retrieval (hop 3)
# ---------------------------------------------------------------------------
REWRITE_PROMPT = (
    "Rephrase the following question using different terminology or perspective "
    "to improve document retrieval. "
    "Output ONLY the rephrased question, nothing else.\n\n"
    "Question: {question}\n\n"
    "Rephrased question: "
)

# ---------------------------------------------------------------------------
# FINAL SYNTHESIS — answer the original question from all accumulated context
# ---------------------------------------------------------------------------
FINAL_SYNTHESIS_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Using ONLY the provided context, give the shortest complete answer possible. "
    "Be factual.\n\n"
    "Question: {question}\n"
    "Answer: "
)

# ---------------------------------------------------------------------------
# CRITIQUE — PASS/FAIL evaluation (hops 1–2 only)
# ---------------------------------------------------------------------------
CRITIQUE_PROMPT = (
    "You are an evaluator. Assess whether the answer below adequately answers "
    "the original question based on the provided context.\n\n"
    "Context:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Original question: {question}\n"
    "Answer to evaluate: {answer}\n\n"
    "Respond with PASS if the answer is correct, addresses what was asked, "
    "and is grounded in the context — minor gaps are acceptable.\n"
    "Respond with FAIL if a key piece of information the question asks for "
    "is clearly missing or the answer is substantially incomplete.\n\n"
    "Verdict: "
)
