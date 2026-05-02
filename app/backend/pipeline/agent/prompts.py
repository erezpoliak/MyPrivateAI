"""Prompt templates for the 3-hop agentic RAG workflow.

Hop 1 — direct retrieval, PASS/FAIL critique
Hop 2 — decompose question into sub-questions, multi-retrieve, PASS/FAIL critique
Hop 3 — inferential correction: reason over all accumulated context to derive answer

Placeholders
------------
DECOMPOSE_PROMPT
    {question}  — original user question

FINAL_SYNTHESIS_PROMPT
    {question}  — original user question
    {context}   — all accumulated context across hops

CRITIQUE_PROMPT  (hops 1 and 2)
    {question}  — original user question
    {answer}    — synthesised answer to evaluate
    {context}   — all accumulated context

CORRECT_PROMPT  (hop 3)
    {question}  — original user question
    {context}   — all accumulated context across hops 1 and 2
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# DECOMPOSE — break question into sub-questions for multi-retrieve (hop 2)
# ---------------------------------------------------------------------------
DECOMPOSE_PROMPT = (
    "Break the following question into focused sub-questions that together "
    "cover all the information needed to answer it. Each sub-question should "
    "target a specific fact or concept. Use at most 3 sub-questions — fewer "
    "if the question is simple.\n\n"
    "Output ONLY the sub-questions, one per line, no numbering or bullet points.\n\n"
    "Question: {question}\n\n"
    "Sub-questions:\n"
)

# ---------------------------------------------------------------------------
# FINAL SYNTHESIS — answer original question from all accumulated context
# ---------------------------------------------------------------------------
FINAL_SYNTHESIS_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Using ONLY the provided context, give the shortest complete answer possible. "
    "Be factual. "
    "After each claim, cite the source number in brackets, e.g. [1] or [2].\n\n"
    "Question: {question}\n"
    "Answer: "
)

# ---------------------------------------------------------------------------
# CRITIQUE — PASS/FAIL, used after hops 1 and 2
# ---------------------------------------------------------------------------
CRITIQUE_PROMPT = (
    "Does the answer below adequately answer the original question based on "
    "the provided context?\n\n"
    "Context:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Original question: {question}\n"
    "Answer: {answer}\n\n"
    "Respond with PASS if the answer addresses the core of the question — "
    "minor gaps, omissions, or paraphrasing are acceptable.\n"
    "Respond with FAIL only if the answer is fundamentally wrong or a critical "
    "piece of information is completely absent.\n\n"
    "Verdict: "
)

# ---------------------------------------------------------------------------
# CORRECT — inferential reasoning over all accumulated context (hop 3)
# ---------------------------------------------------------------------------
CORRECT_PROMPT = (
    "Previous attempts to answer this question did not find an explicit answer "
    "in the retrieved context.\n\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Using the provided context, reason over the information to derive the best "
    "possible answer. If the answer is not stated directly, draw logical conclusions "
    "from what the context does say. Give the shortest complete answer possible — "
    "state the conclusion directly, no explanations or caveats. "
    "After each claim, cite the source number in brackets, e.g. [1] or [2].\n\n"
    "Question: {question}\n"
    "Answer: "
)
