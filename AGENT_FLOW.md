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

Hop 2: decompose(question) --> retrieve(each sub-Q + original Q) --> synthesize(original Q) --> critique(PASS/FAIL)
  PASS --> done
  FAIL -->

Hop 3: correct(all accumulated context) --> done
  [no new retrieval — inferential reasoning over full context]
```

## Steps

### Retrieve (hops 1–2)
Fetches document chunks from the hybrid retriever (vector + BM25 + reranking).
- Hop 1: single retrieval using the original question.
- Hop 2: multi-retrieval — runs one retrieval per sub-question (up to 3) plus the
  original question, then deduplicates and merges all results.

### Synthesize (hops 1–2)
Answers the **original question** (not the sub-questions) using **all** accumulated
chunks across all hops. Context grows with each hop. Uses `FINAL_SYNTHESIS_PROMPT`.

### Critique (hops 1–2)
PASS/FAIL evaluation. PASS if the answer addresses the core of the question —
minor gaps or paraphrasing are acceptable. When in doubt, PASS.
FAIL only if the answer is fundamentally wrong or a critical piece of information
is completely absent.

### Decompose (hop 2 only)
Breaks the original question into up to 3 focused sub-questions to broaden
retrieval coverage across multiple sources. Sub-questions are **retrieval queries
only** — they are never answered separately.

### Correct (hop 3 — no new retrieval)
Reasons over all chunks accumulated in hops 1–2 to derive the best possible answer.
If the answer is not explicitly stated in the context, draws logical conclusions
from what the context does say. Targets both reasoning failures and cases where
the answer requires inference rather than direct extraction.
Uses `CORRECT_PROMPT`.
