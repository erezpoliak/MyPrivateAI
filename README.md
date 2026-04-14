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
| 5 | **Qwen+Gold_REF** | Perfect retrieval (gold contexts injected) — isolates the model's comprehension ceiling |
| 6 | **GPT-4o RAG (Ceiling)** | GPT-4o running the identical Phase 2 pipeline — LLM is the only variable |

### Gap Analysis

```
1. Closed Book          ──┐
                          ├─ RAG value-add
2. Baseline             ──┘──┐
                              ├─ Better retrieval + semantic chunking + metadata enrichment
3. Phase 1              ─────┘──┐
                                 ├─ Agentic multi-hop reasoning
4. Phase 2              ────────┘──┬─── Retrieval quality gap
                                    │              └──> 5. Qwen+Gold_REF
                                    │
                                    ├─ HYPOTHESIS TEST (target: ≥85%)
6. GPT-4o RAG (Ceiling) ──────────┘
```

**Success criteria:** Phase 2 achieves ≥ 85% of the GPT-4o RAG ceiling on answer correctness (LLM is the only variable), and matches or exceeds Qwen+Gold_REF on complexity 3-4 questions (demonstrating that the agent's reasoning compensates for imperfect retrieval on hard questions).

### Phase 2 — Agent Flow

```
Hop 1:  retrieve(original Q) ──> synthesize ──> critique(PASS/FAIL)
          PASS ──> done
          FAIL ──>

Hop 2:  decompose(question) ──> retrieve(each sub-Q + original Q) ──> synthesize ──> critique(PASS/FAIL + text)
          PASS ──> done
          FAIL ──>

Hop 3:  correct(all accumulated context + critique text) ──> done
        [no new retrieval — reasoning correction only]
```

- **Hop 1** retrieves using the original question directly — giving it a fair shot first
- **Hop 2** decomposes the question into sub-questions (model decides how many), retrieves for each, then synthesizes from the combined context — targets cross-paper retrieval failures
- **Hop 3** re-synthesizes using the critique text from hop 2 as explicit correction guidance — targets reasoning failures where the right context was retrieved but the model reasoned incorrectly
- **Synthesize** always answers the original question from all accumulated context

---

## Installation

> **Coming soon.** MyPrivateAI is currently in the validation phase. A guided installer will be provided in a future release.

---

## Experiment Results

> **Coming soon.** Results will be published here as each experiment is completed.

---
