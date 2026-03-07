"""Corpus ingestion — from raw documents to chunked TextNodes.

Public API (used by experiments):
    corpus_builder      — build_corpus() orchestrates chunking for gold_ref or fetched mode
    document_store      — DocumentStore manages ChromaDB vector index (accepts BaseEmbedding)

Internal (used by corpus_builder or experiment specs):
    fixed_chunker       — FixedSizeChunker: wraps LlamaIndex TokenTextSplitter
    semantic_chunker    — CappedSemanticSplitter: LlamaIndex SemanticSplitterNodeParser + token cap
    pdf_parser          — parse_pdf() extracts and cleans text from PDFs
    pdf_fetcher         — PDFFetcher downloads open-access PDFs via Unpaywall / Semantic Scholar
"""
