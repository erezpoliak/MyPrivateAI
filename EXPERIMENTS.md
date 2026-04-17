# Validation Pipeline — Experiment Design

## Experiment Table

| # | Experiment | Chunking | Retrieval | Reranking | Metadata Enrichment | LLM | RAGAS Judge | Key Targets |
|---|---|---|---|---|---|---|---|---|
| 1 | **Closed Book** | N/A | None | None | No | Llama 3.1 8B | GPT-4.1-mini | Reference floor (no targets) |
| 2 | **Baseline** | Fixed 512 | Vector k=5 | None | No | Llama 3.1 8B | GPT-4.1-mini | AC > Closed Book |
| 3 | **Phase 1** | Semantic | Hybrid BM25+Vector (RRF) | FlashRank top-3 | Yes | Llama 3.1 8B | GPT-4.1-mini | CR > 0.92, Faith > 0.98 |
| 4 | **Phase 2** | Semantic | Critique-driven agent: hop 1 direct, hop 2 decompose+multi-retrieve, hop 3 correct | FlashRank top-3 | Yes | Llama 3.1 8B | GPT-4.1-mini | AC > 85% ceiling, Traj > 0.80, AC(cplx 3-4) ≥ Llama+Gold_REF |
| 5 | **Llama+Gold_REF** | N/A | Gold_REF injected | None | No | Llama 3.1 8B | GPT-4.1-mini | Model comprehension ceiling |
| 6 | **GPT-5.1 Gold_REF (Ceiling)** | N/A | Gold_REF injected | None | No | GPT-5.1 | GPT-4.1-mini | Upper bound anchor — GPT-5.1 with perfect context |

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
4. Phase 2              ────────┘──┬─── Retrieval quality gap
                                    │              └──> 5. Llama+Gold_REF
                                    │
                                    ├─ HYPOTHESIS TEST (target: ≥85%)
6. GPT-5.1 Gold_REF (Ceiling) ────┘
```

| Gap | What it measures | Why it matters |
|---|---|---|
| Closed Book → Baseline | How much value RAG adds over parametric knowledge | Validates that retrieval is worth doing at all |
| Baseline → Phase 1 | Impact of semantic chunking + hybrid search + reranking + metadata enrichment | Quantifies retrieval optimization gains |
| Phase 1 → Phase 2 | Impact of agentic multi-hop reasoning | Quantifies value of the ReAct agent |
| Phase 2 → Llama+Gold_REF | Performance lost to imperfect retrieval | Shows how much headroom better retrieval could unlock |
| Phase 2 → GPT-5.1 Gold_REF | Combined model + retrieval ceiling gap — direct hypothesis test | GPT-5.1 Gold_REF represents the absolute ceiling (strongest model + perfect context); ≤15% gap confirms the pipeline compensates for 8B limitations |

---

## Metric Applicability

Not all RAGAS metrics are meaningful for every experiment:

| Metric | Closed Book | Baseline | Phase 1 | Phase 2 | Llama+Gold_REF | Ceiling |
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
| Llama+Gold_REF | `gold_ref` | Gold_REF text injected directly as context |
| GPT-5.1 Gold_REF (Ceiling) | `gold_ref` | Gold_REF text injected directly as context — GPT-5.1 with perfect retrieval |

---

## Core Hypothesis Test

> An Optimized Agentic RAG pipeline can compensate for an 8B model's limitations and achieve performance close to GPT-5.1 on complex private document tasks.

The 6-experiment design lets us decompose this hypothesis precisely:

- **Phase 2 AC / GPT-5.1 Gold_REF AC ≥ 85%** → Hypothesis supported — the pipeline closes the model gap
- **Phase 2 AC on complexity 3-4 ≥ Llama+Gold_REF AC on complexity 3-4** → Agent's reasoning compensates for imperfect retrieval on hard questions
- **Phase 2 vs Llama+Gold_REF** → Isolates how much of the remaining gap to GPT-5.1 Gold_REF is retrieval quality vs model capability
