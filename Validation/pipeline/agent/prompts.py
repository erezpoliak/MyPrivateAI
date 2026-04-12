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
    "A previous attempt to answer the question below was rejected because "
    "specific information was missing.\n\n"
    "Original question: {question}\n\n"
    "{critique_feedback}"
    "{prior_queries}"
    "Write ONE retrieval query that will directly find the missing information "
    "identified above. Do NOT repeat previous queries. Do NOT elaborate beyond "
    "what is needed.\n"
    "Output ONLY the query, nothing else.\n\n"
    "Query: "
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
# CRITIQUE — evaluate an answer for completeness and faithfulness
# ---------------------------------------------------------------------------
CRITIQUE_PROMPT = (
    "You are a fair evaluator. Assess whether the answer below reasonably "
    "answers the original question based on the provided context.\n\n"
    "Context:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Original question: {question}\n"
    "Answer to evaluate: {answer}\n\n"
    "If the answer addresses the core of the question and is supported by "
    "the context, respond with exactly: PASS\n"
    "Only respond with FAIL if a specific, clearly identifiable piece of "
    "information is missing from the answer. In that case respond with: FAIL\n"
    "Then on the next line write ONE sentence naming exactly what specific "
    "information is missing (e.g. 'Missing: the exact percentage of X').\n\n"
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
    "critique. Give the shortest complete answer possible and stay faithful to the context.\n\n"
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
