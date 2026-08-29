# MyPrivateAI

**A local-first desktop application that brings cloud-level AI to sensitive private documents — without sacrificing security, cost, or accessibility.**

Many professionals can't use cloud AI tools because their documents contain confidential information, and existing local alternatives are either hard to use or weak at complex document reasoning. MyPrivateAI closes that gap with a simple installer for researchers, lawyers, and other professionals handling confidential data — no technical expertise required.

**The core innovation:** our Optimized Agentic RAG system combines advanced RAG techniques with multi-turn reasoning agents, enabling complex multi-document analysis once reserved for cloud services. Through systematic validation on the SciRAG-QA benchmark, we demonstrate that frontier-level performance is achievable on consumer hardware — **MyPrivateAI achieved 95% of GPT-5.1 + Gold Ref, a ceiling simulating a perfect cloud model with perfect retrieval.**

MyPrivateAI delivers a "cloud LLM experience" locally — completely free, private, and offline — making advanced AI accessible to anyone with sensitive documents.

---

## Validation Experiment Design

Before building the full application, we rigorously validate our core hypothesis:

> *An Optimized Agentic RAG pipeline can compensate for an 8B model's limitations and achieve performance close to GPT-5.1 on complex mutli-paper private document tasks.*

We use a 6-experiment design evaluated with [RAGAS](https://docs.ragas.io/) metrics on the **SciRAG-QA** benchmark. Each transition between experiments isolates a single variable, letting us decompose exactly where performance gains (and losses) come from.

| # | Experiment | What it tests |
|---|---|---|
| 1 | **Closed Book** | Local LLM parametric knowledge alone — no retrieval |
| 2 | **Baseline RAG** | Fixed-size chunking + basic vector search |
| 3 | **Phase 1 — Optimized Retrieval** | Semantic chunking + hybrid BM25/vector search + FlashRank reranking + metadata enrichment |
| 4 | **Phase 2 — Agentic RAG** | Critique-driven multi-hop agent over the optimized retrieval pipeline |
| 5 | **Llama+Gold_REF** | Perfect retrieval (gold contexts injected) — isolates the model's comprehension ceiling |
| 6 | **GPT-5.1 Gold_REF (Ceiling)** | GPT-5.1 with perfect context (Gold_REF injected) — absolute ceiling combining strongest model + perfect retrieval |

### Gap Analysis

```
1. Closed Book          ──┐
                          ├─ RAG value-add
2. Baseline             ──┘──┐
                              ├─ Better retrieval + semantic chunking + metadata enrichment
3. Phase 1              ─────┘──┐
                                 ├─ Agentic multi-hop reasoning
4. Phase 2              ────────┘──┬─── Retrieval quality gap
                                    │              └──> 5. Llama+Gold_REF
                                    │
                                    ├─ HYPOTHESIS TEST (target: ≥85%)
6. GPT-5.1 Gold_REF (Ceiling) ────┘
```

**Success criteria:** Phase 2 achieves ≥ 85% of the GPT-5.1 Gold_REF ceiling on answer correctness (the absolute ceiling combining strongest model + perfect context), and matches or exceeds Llama+Gold_REF on complexity 3-4 questions (demonstrating that the agent's reasoning compensates for imperfect retrieval on hard questions).

### Phase 2 — Agent Flow

```
Hop 1:  retrieve(original Q) ──> synthesize ──> critique(PASS/FAIL)
          PASS ──> done
          FAIL ──>

Hop 2:  decompose(question) ──> retrieve(each sub-Q + original Q) ──> synthesize ──> critique(PASS/FAIL)
          PASS ──> done
          FAIL ──>

Hop 3:  correct(all accumulated context) ──> done
        [no new retrieval — inferential reasoning over full context]
```

- **Hop 1** retrieves using the original question directly — giving it a fair shot first
- **Hop 2** decomposes the question into up to 3 sub-questions, retrieves for each, then synthesizes from the combined context — targets cross-paper retrieval failures
- **Hop 3** reasons over all accumulated context to derive the best possible answer, including logical conclusions not explicitly stated — targets reasoning failures and inference gaps
- **Synthesize** always answers the original question from all accumulated context

---

## Experiment Results

**Phase 2 (Agentic RAG) achieved 95% of GPT-5.1 + Gold Ref performance — a ceiling that simulates a perfect cloud model with perfect retrieval.**

All experiments evaluated with [RAGAS](https://docs.ragas.io/) metrics on the SciRAG-QA benchmark. `—` indicates the metric is not applicable (no retrieval).

| Experiment | Faithfulness | Context Recall | Context Precision | Answer Correctness |
|---|:---:|:---:|:---:|:---:|
| Closed Book | — | — | — | 0.2696 |
| Baseline (Fixed / Vector) | 0.7500 | 0.6897 | 0.6185 | 0.5543 |
| Phase 1 (Semantic / Hybrid) | 0.8534 | 0.7931 | 0.7667 | 0.7346 |
| Phase 2 (Agentic RAG) | 0.7615 | 0.8276 | 0.7363 | 0.7816 |
| Llama + Gold Ref | 0.6351 | 0.9310 | 0.9138 | 0.6607 |
| GPT-5.1 + Gold Ref (Ceiling) | 0.5737 | 0.8793 | 0.9138 | 0.8195 |

---

## Installation

**Requirements:** Apple Silicon Mac (M1 or later), macOS 13+, 16 GB RAM, ~8 GB free disk space.

1. Download [`MyPrivateAI-0.1.0-arm64.dmg`](https://transfer.it/t/cjmGmxwjHRBT) and open it.
2. Drag **MyPrivateAI** into **Applications**, then launch it.
3. The app isn't code-signed yet, so macOS will block the first launch with a
   "malware" warning — this is expected. Go to **System Settings → Privacy &
   Security**, click **Open Anyway** next to the MyPrivateAI notice, and confirm.
   You'll only need to do this once.
4. A splash screen appears while the app loads on first launch — everything
   (models included) ships inside the app, so no downloads or internet needed.

Your documents and chats stay local, in `~/Library/Application Support/MyPrivateAI`.

---
