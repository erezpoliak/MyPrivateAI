"""Prompt templates for the critique-driven agentic RAG workflow.

Four templates drive the retrieve → synthesize → critique loop, with
decompose joining on hop 2+.  All are plain ``str.format`` templates
with named placeholders.

Placeholders per template
-------------------------
DECOMPOSE_PROMPT (hop 2+ only)
    {question}           — original user question
    {prior_queries}      — retrieval queries already tried (so they aren't repeated)
    {critique_feedback}  — critique text explaining why the previous answer
                           was rejected

FINAL_SYNTHESIS_PROMPT
    {question}  — original user question
    {context}   — all accumulated context across hops

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
# DECOMPOSE — generate a targeted retrieval query to fill gaps (hop 2+ only)
# ---------------------------------------------------------------------------
DECOMPOSE_PROMPT = (
    "You are a research assistant. A previous attempt to answer the question "
    "below was not sufficient. Generate a focused retrieval query to find "
    "the missing information.\n\n"
    "Original question: {question}\n\n"
    "{prior_queries}"
    "{critique_feedback}"
    "Produce exactly ONE focused sub-question that targets the identified "
    "gaps. Do NOT repeat previous queries.\n"
    "Output ONLY the sub-question, nothing else.\n\n"
    "Sub-question: "
)

# ---------------------------------------------------------------------------
# FINAL SYNTHESIS — answer the original question from all accumulated context
# ---------------------------------------------------------------------------
FINAL_SYNTHESIS_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Using ONLY the provided context, answer the following question. "
    "Be comprehensive and factual.\n\n"
    "Question: {question}\n"
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


def format_prior_queries(queries: list[str]) -> str:
    """Build the ``{prior_queries}`` block for DECOMPOSE on hop 2+.

    Lists retrieval queries already tried so decompose avoids repeating them.
    Returns an empty string when no prior queries exist.
    """
    if not queries:
        return ""
    lines = ["Retrieval queries already tried:\n"]
    for i, q in enumerate(queries, 1):
        lines.append(f"  {i}. {q}")
    lines.append("\n")
    return "\n".join(lines)
