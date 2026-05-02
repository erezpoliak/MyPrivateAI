"""Retrieval strategies — from query to ranked document nodes.

Public API (plugged into experiments via ExperimentSpec):
    hybrid_retriever    — HybridRetriever: RRF fusion + FlashRankRerank reranking
                          Also exports rrf_fuse() as a standalone pure function

Vector retrieval uses index.as_retriever() directly (LlamaIndex built-in).
BM25 retrieval uses llama-index-retrievers-bm25 (LlamaBM25Retriever).
"""
