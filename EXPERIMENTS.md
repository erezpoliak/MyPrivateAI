# Validation Pipeline — Experiment Design

## Experiment Table

| # | Experiment | Chunking | Retrieval | Reranking | LLM | RAGAS Judge | Key Targets |
|---|---|---|---|---|---|---|---|
| 1 | **Closed Book** | N/A | None | None | Qwen-3.5 9B | GPT-4o-mini | Reference floor (no targets) |
| 2 | **Baseline** | Fixed 512 | Vector k=5 | None | Qwen-3.5 9B | GPT-4o-mini | AC > Closed Book |
| 3 | **Phase 1** | Semantic | Hybrid BM25+Vector (RRF) | FlashRank top-3 | Qwen-3.5 9B | GPT-4o-mini | CR > 0.92, Faith > 0.98 |
| 4 | **Phase 2** | Semantic | Multi-hop agent using Hybrid BM25+Vector (RRF) | FlashRank top-3 | Qwen-3.5 9B | GPT-4o-mini | AC > 85% ceiling, Traj > 0.80, AC(cplx 3-4) ≥ Qwen+Gold_REF |
| 5 | **Qwen+Gold_REF** | N/A | Gold_REF injected | None | Qwen-3.5 9B | GPT-4o-mini | Qwen comprehension ceiling |
| 6 | **GPT-4o RAG (Ceiling)** | Semantic | Multi-hop agent using Hybrid BM25+Vector (RRF) | FlashRank top-3 | GPT-4o | GPT-4o-mini | Upper bound anchor — identical pipeline to Phase 2, only LLM differs |

---

## Gap Analysis

Each transition between experiments isolates a single variable:

```
1. Closed Book          ──┐
                          ├─ RAG value-add
2. Baseline             ──┘──┐
                              ├─ Better retrieval + semantic chunking
3. Phase 1              ─────┘──┐
                                 ├─ Agentic multi-hop reasoning
4. Phase 2              ────────┘──┬─── Retrieval quality gap (Qwen only)
                                    │              └──> 5. Qwen+Gold_REF
                                    │
                                    ├─ HYPOTHESIS TEST (target: ≥85%)
6. GPT-4o RAG (Ceiling) ──────────┘
```

| Gap | What it measures | Why it matters |
|---|---|---|
| Closed Book → Baseline | How much value RAG adds over parametric knowledge | Validates that retrieval is worth doing at all |
| Baseline → Phase 1 | Impact of semantic chunking + hybrid search + reranking | Quantifies retrieval optimization gains |
| Phase 1 → Phase 2 | Impact of agentic multi-hop reasoning | Quantifies value of the ReAct agent |
| Phase 2 → Qwen+Gold_REF | Performance lost to imperfect retrieval (Qwen) | Shows how much headroom better retrieval could unlock |
| Phase 2 → GPT-4o RAG | Model capability gap — direct hypothesis test | LLM is the only variable; ≤15% gap confirms the hypothesis |

---

## Metric Applicability

Not all RAGAS metrics are meaningful for every experiment:

| Metric | Closed Book | Baseline | Phase 1 | Phase 2 | Qwen+Gold_REF | Ceiling |
|---|---|---|---|---|---|---|
| answer_correctness | Yes | Yes | Yes | Yes | Yes | Yes |
| context_recall | N/A | Yes | Yes | Yes | Yes | Yes |
| faithfulness | N/A | Yes | Yes | Yes | Yes | Yes |
| trajectory_success | N/A | N/A | N/A | Yes | N/A | N/A |
| trajectory_steps | N/A | N/A | N/A | Yes | N/A | N/A |

**Closed Book** has no retrieved contexts, so all context-dependent metrics are `None`.
**Trajectory metrics** only apply to Phase 2 (the agentic experiment).

---

## Corpus Mode per Experiment

| Experiment | Corpus Mode | Notes |
|---|---|---|
| Closed Book | `none` | No corpus — parametric knowledge only |
| Baseline | `fetched` | Real PDFs for scored runs |
| Phase 1 | `fetched` | Real PDFs for scored runs |
| Phase 2 | `fetched` | Real PDFs for scored runs |
| Qwen+Gold_REF | `gold_ref` | Gold_REF text injected directly as context |
| GPT-4o RAG (Ceiling) | `fetched` | Same fetched PDFs and pipeline as Phase 2 — only LLM differs |

---

## Core Hypothesis Test

> An Optimized Agentic RAG pipeline can compensate for a 9B model's limitations and achieve performance close to GPT-4o on complex private document tasks.

The 6-experiment design lets us decompose this hypothesis precisely:

- **Phase 2 AC / GPT-4o RAG AC ≥ 85%** → Hypothesis supported — the pipeline closes the model gap (LLM is the only variable)
- **Phase 2 AC on complexity 3-4 ≥ Qwen+Gold_REF AC on complexity 3-4** → Agent's reasoning compensates for imperfect retrieval on hard questions
- **Phase 2 vs Qwen+Gold_REF** → Isolates how much of the remaining gap to GPT-4o RAG is retrieval quality vs model capability
