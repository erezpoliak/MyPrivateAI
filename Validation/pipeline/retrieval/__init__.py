"""Retrieval strategies — from query to ranked document nodes.

Public API (plugged into experiments via ExperimentSpec):
    vector_retriever    — VectorRetriever: cosine similarity search over the vector index
    hybrid_retriever    — HybridRetriever: RRF fusion + FlashRank reranking
                          Also exports rrf_fuse() and rerank() as standalone pure functions

Internal (used by HybridRetriever):
    bm25_retriever      — BM25Retriever: sparse keyword retrieval via BM25Okapi
"""
