# Agentic RAG Workflow

## Overview

Critique-driven multi-hop RAG pipeline. The critique evaluates answer quality
against the original question — not retrieval quality — because answer correctness
is the metric that matters (measured by RAGAS).

## Flow

```
Hop 1: retrieve(original Q) --> synthesize(original Q) --> critique(PASS/FAIL)
  PASS --> done
  FAIL -->

Hop 2: decompose(question) --> retrieve(each sub-Q + original Q) --> synthesize(original Q) --> critique(PASS/FAIL + text)
  PASS --> done
  FAIL -->

Hop 3: correct(all accumulated context + critique text) --> done
  [no new retrieval — reasoning correction only]
```

## Steps

### Retrieve (hops 1–2)
Fetches document chunks from the hybrid retriever (vector + BM25 + reranking).
- Hop 1: single retrieval using the original question.
- Hop 2: multi-retrieval — runs one retrieval per sub-question plus the original
  question, then deduplicates and merges all results.

### Synthesize (hops 1–2)
Answers the **original question** (not the sub-questions) using **all** accumulated
chunks across all hops. Context grows with each hop. Uses `FINAL_SYNTHESIS_PROMPT`.

### Critique (hops 1–2)
Evaluates the synthesized answer against the original question.
- Hop 1: simple PASS/FAIL only (`CRITIQUE_SIMPLE_PROMPT`).
- Hop 2: PASS/FAIL + one-sentence explanation of the key gap (`CRITIQUE_DETAILED_PROMPT`).

Minor gaps or paraphrasing are acceptable — FAIL only when a key fact is clearly
missing or factually wrong.

### Decompose (hop 2 only)
Breaks the original question into focused sub-questions to broaden retrieval coverage.
The model decides how many sub-questions are needed based on question complexity.
Sub-questions are **retrieval queries only** — they are never answered separately.

### Correct (hop 3 — no new retrieval)
Re-synthesizes the answer using all chunks accumulated in hops 1–2 plus the critique
text from hop 2 as explicit correction guidance. Targets reasoning failures where
the right context was already retrieved but the model reasoned incorrectly.
Uses `CORRECT_PROMPT`.
