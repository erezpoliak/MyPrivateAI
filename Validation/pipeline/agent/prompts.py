"""Versioned prompt templates for the agentic RAG workflow.

Four templates drive the decompose → retrieve → synthesize → citique → self-correct
loop.  All are plain ``str.format`` templates with named placeholders.

Placeholders per template
-------------------------
DECOMPOSE_PROMPT
    {question}              — original user question
    {hop}                   — current hop number (1-based)
    {max_hops}              — maximum hops allowed
    {intermediate_context}  — "" on hop 1; prior sub-Q/answer pairs on hop 2+

SYNTHESIS_PROMPT
    {sub_question}  — the sub-question being answered
    {context}       — retrieved chunks joined by blank lines

CRITIQUE_PROMPT
    {question}  — original user question
    {answer}    — synthesised answer to evaluate
    {context}   — all retrieved context used during synthesis

CORRECTION_PROMPT
    {question}  — original user question
    {answer}    — answer that failed critique
    {context}   — all retrieved context
    {critique}  — critique text explaining the deficiencies
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# DECOMPOSE — break a complex question into a focused sub-question
# ---------------------------------------------------------------------------
DECOMPOSE_PROMPT = (
    "You are a research assistant that decomposes complex questions into "
    "simpler sub-questions for step-by-step retrieval.\n\n"
    "Original question: {question}\n\n"
    "Hop {hop} of {max_hops}.\n"
    "{intermediate_context}"
    "Based on what is still unknown, produce exactly ONE focused "
    "sub-question that will help answer the original question.\n"
    "Output ONLY the sub-question, nothing else.\n\n"
    "Sub-question: "
)

# ---------------------------------------------------------------------------
# SYNTHESIS — answer a sub-question given retrieved context
# ---------------------------------------------------------------------------
SYNTHESIS_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Using ONLY the provided context, answer the following sub-question. "
    "Be concise and factual. If the context does not contain enough "
    "information, state what is missing.\n\n"
    "Sub-question: {sub_question}\n"
    "Answer: "
)

# ---------------------------------------------------------------------------
# CRITIQUE — evaluate an answer for completeness and faithfulness
# ---------------------------------------------------------------------------
CRITIQUE_PROMPT = (
    "You are a strict evaluator. Assess whether the answer below fully and "
    "faithfully answers the original question based on the provided context.\n\n"
    "Context:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Original question: {question}\n"
    "Answer to evaluate: {answer}\n\n"
    "Check for:\n"
    "1. Completeness — does the answer address every part of the question?\n"
    "2. Faithfulness — is every claim supported by the context?\n"
    "3. Accuracy — are there any factual errors?\n\n"
    "If the answer is satisfactory, respond with exactly: PASS\n"
    "Otherwise, respond with: FAIL\n"
    "Then on the next line explain the specific deficiencies.\n\n"
    "Verdict: "
)

# ---------------------------------------------------------------------------
# CORRECTION — fix a flawed answer using the critique
# ---------------------------------------------------------------------------
CORRECTION_PROMPT = (
    "The previous answer was found to be deficient. Using the critique and "
    "the original context, produce a corrected answer.\n\n"
    "Context:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Original question: {question}\n"
    "Previous answer: {answer}\n"
    "Critique: {critique}\n\n"
    "Write a corrected answer that addresses every deficiency raised in the "
    "critique. Be concise and stay faithful to the context.\n\n"
    "Corrected answer: "
)


def format_intermediate_context(
    sub_questions: list[str],
    answers: list[str],
) -> str:
    """Build the ``{intermediate_context}`` block for DECOMPOSE on hop 2+.

    Returns an empty string when no prior hops exist (hop 1).
    """
    if not sub_questions:
        return ""
    lines = ["Information gathered so far:\n"]
    for i, (sq, ans) in enumerate(zip(sub_questions, answers), 1):
        lines.append(f"  Hop {i} sub-question: {sq}")
        lines.append(f"  Hop {i} answer: {ans}\n")
    lines.append("")
    return "\n".join(lines)
