"""Fixed-size chunking via LlamaIndex TokenTextSplitter.
using cl100k_base tokenizer with the chunk size and overlap from Config.
"""

from __future__ import annotations

from typing import Optional

from llama_index.core.node_parser import TokenTextSplitter

from llama_index.core.schema import TextNode, Document

from common.config import Config



class FixedSizeChunker:
    """Wrapper around LlamaIndex TokenTextSplitter to satisfy the Chunker protocol."""

    def __init__(self, config: Optional[Config] = None) -> None:
        config = config or Config()
        self._splitter = TokenTextSplitter(
            chunk_size=config.fixed_chunk_size,
            chunk_overlap=config.fixed_chunk_overlap,
        )
    
    def get_nodes_from_documents(self, documents: list[Document], **kwargs) -> list[TextNode]:
        """Delegate to the underlying TokenTextSplitter."""
        return self._splitter.get_nodes_from_documents(documents, **kwargs)
