"""Prompt templates for the 3-hop agentic RAG workflow.

Hop 1 — direct retrieval, simple PASS/FAIL critique
Hop 2 — decompose question into sub-questions, multi-retrieve, detailed critique
Hop 3 — correction: re-synthesize from all accumulated context + critique text

Placeholders
------------
DECOMPOSE_PROMPT
    {question}  — original user question

FINAL_SYNTHESIS_PROMPT
    {question}  — original user question
    {context}   — all accumulated context across hops

CRITIQUE_SIMPLE_PROMPT  (hop 1)
    {question}  — original user question
    {answer}    — synthesised answer to evaluate
    {context}   — retrieved context used during synthesis

CRITIQUE_DETAILED_PROMPT  (hop 2)
    {question}  — original user question
    {answer}    — synthesised answer to evaluate
    {context}   — all accumulated context

CORRECT_PROMPT  (hop 3)
    {question}  — original user question
    {critique}  — critique text from hop 2 explaining what is wrong
    {context}   — all accumulated context across hops 1 and 2
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# DECOMPOSE — break question into sub-questions for multi-retrieve (hop 2)
# ---------------------------------------------------------------------------
DECOMPOSE_PROMPT = (
    "Break the following question into focused sub-questions that together "
    "cover all the information needed to answer it. Each sub-question should "
    "target a specific fact or concept. Decide how many sub-questions are needed "
    "— use fewer for simple questions, more for complex ones requiring information "
    "from multiple sources.\n\n"
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
    "Be factual.\n\n"
    "Question: {question}\n"
    "Answer: "
)

# ---------------------------------------------------------------------------
# CRITIQUE SIMPLE — PASS/FAIL only, used after hop 1
# ---------------------------------------------------------------------------
CRITIQUE_SIMPLE_PROMPT = (
    "Does the answer below adequately answer the original question based on "
    "the provided context?\n\n"
    "Context:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Original question: {question}\n"
    "Answer: {answer}\n\n"
    "Respond with PASS if the answer is correct and substantially complete — minor gaps or paraphrasing are acceptable.\n"
    "Respond with FAIL only if a key fact is clearly missing or the answer is factually wrong.\n\n"
    "Verdict: "
)

# ---------------------------------------------------------------------------
# CRITIQUE DETAILED — PASS/FAIL + explanation, used after hop 2
# ---------------------------------------------------------------------------
CRITIQUE_DETAILED_PROMPT = (
    "Assess whether the answer below adequately answers the original question "
    "based on the provided context.\n\n"
    "Context:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Original question: {question}\n"
    "Answer: {answer}\n\n"
    "Respond with PASS if the answer is correct and substantially complete — minor gaps or paraphrasing are acceptable.\n"
    "Respond with FAIL followed by one sentence describing the key missing fact or error.\n\n"
    "Verdict: "
)

# ---------------------------------------------------------------------------
# CORRECT — re-synthesize using critique feedback, no new retrieval (hop 3)
# ---------------------------------------------------------------------------
CORRECT_PROMPT = (
    "A previous attempt to answer this question was evaluated as insufficient.\n\n"
    "Critique of the previous answer:\n"
    "{critique}\n\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Using ONLY the provided context, give a corrected answer that addresses "
    "the critique. Be factual and concise.\n\n"
    "Question: {question}\n"
    "Corrected answer: "
)
