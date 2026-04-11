# MyPrivateAI

**A local-first desktop application that brings cloud-level AI to sensitive private documents — without sacrificing security, cost, or accessibility.**

While existing local AI tools require technical expertise, MyPrivateAI offers a simple installer for researchers, lawyers, and professionals handling confidential data.

**The core innovation:** Local LLMs typically underperform on complex reasoning compared to GPT-4. Our Optimized Agentic RAG system combines advanced retrieval with multi-turn reasoning agents, enabling complex multi-document analysis once reserved for cloud services. Through systematic validation on research benchmarks, we aim to demonstrate that frontier-level performance is achievable on consumer hardware.

MyPrivateAI will deliver a "cloud LLM experience" locally — completely free, private, and offline — making advanced AI accessible to anyone with sensitive documents.

---

## Validation Experiment Design

Before building the full application, we rigorously validate our core hypothesis:

> *An Optimized Agentic RAG pipeline can compensate for an 8B model's limitations and achieve performance close to GPT-4o on complex mutli-paper private document tasks.*

We use a 6-experiment design evaluated with [RAGAS](https://docs.ragas.io/) metrics on the **SciRAG-QA** benchmark. Each transition between experiments isolates a single variable, letting us decompose exactly where performance gains (and losses) come from.

| # | Experiment | What it tests |
|---|---|---|
| 1 | **Closed Book** | Local LLM parametric knowledge alone — no retrieval |
| 2 | **Baseline RAG** | Fixed-size chunking + basic vector search |
| 3 | **Phase 1 — Optimized Retrieval** | Semantic chunking + hybrid BM25/vector search + FlashRank reranking + metadata enrichment |
| 4 | **Phase 2 — Agentic RAG** | Critique-driven multi-hop agent over the optimized retrieval pipeline |
| 5 | **Llama + Gold References** | Perfect retrieval (gold contexts injected) — isolates Llama's comprehension ceiling |
| 6 | **GPT-4o RAG (Ceiling)** | GPT-4o running the identical Phase 2 pipeline — LLM is the only variable |

### Gap Analysis

```
1. Closed Book          ──┐
                          ├─ RAG value-add
2. Baseline             ──┘──┐
                              ├─ Better retrieval + semantic chunking + metadata enrichment
3. Phase 1              ─────┘──┐
                                 ├─ Agentic multi-hop reasoning
4. Phase 2              ────────┘──┬─── Retrieval quality gap (Llama only)
                                    │              └──> 5. Llama+Gold_REF
                                    │
                                    ├─ HYPOTHESIS TEST (target: ≥85%)
6. GPT-4o RAG (Ceiling) ──────────┘
```

**Success criteria:** Phase 2 achieves ≥ 85% of the GPT-4o RAG ceiling on answer correctness (LLM is the only variable), and matches or exceeds Llama+Gold on complexity 3-4 questions (demonstrating that the agent's reasoning compensates for imperfect retrieval on hard questions).

### Phase 2 — Agent Flow

```
Hop 1:  retrieve(original Q) ──> synthesize ──> critique
          PASS ──> done
          FAIL ──>

Hop 2:  decompose(critique) ──> retrieve(sub-Q) ──> synthesize ──> critique
          PASS ──> done
          FAIL ──>

Hop 3:  decompose(critique) ──> retrieve(sub-Q) ──> synthesize ──> critique
          PASS ──> done
          FAIL ──> correct ──> done
```

- **Hop 1** uses the original question directly — giving it a fair shot before decomposing
- **Hop 2+** decomposes a targeted retrieval query based on the critique's feedback
- **Synthesize** always answers the original question from all accumulated context
- **Correct** is a last-resort rewrite when all hops are exhausted

---

## Installation

> **Coming soon.** MyPrivateAI is currently in the validation phase. A guided installer will be provided in a future release.

---

## Experiment Results

> **Coming soon.** Results will be published here as each experiment is completed.

---
