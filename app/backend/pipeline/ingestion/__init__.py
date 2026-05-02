"""Ingestion — from uploaded PDFs to chunked TextNodes in ChromaDB.

Public API:
    document_store   — DocumentStore: manages ChromaDB vector index
    pdf_parser       — parse_pdf(): extracts and cleans text from a PDF
    semantic_chunker — CappedSemanticSplitter: semantic chunking with token cap
"""
