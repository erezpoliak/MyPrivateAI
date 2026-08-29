"""FastAPI app factory for MyPrivateAI backend.

Run with:
    python -m app.backend.server
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .db.app_db import AppDB
from .pipeline.common.config import Config
from .pipeline.common.llm import get_llm
from .pipeline.ingestion.document_store import DocumentStore
from .services.retrieval import RetrieverService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(config: Config | None = None) -> FastAPI:
    cfg = config or Config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading embedding model: %s", cfg.embedding_model)
        embed_model = HuggingFaceEmbedding(
            model_name=cfg.embedding_model,
            device=cfg.device,
        )

        logger.info("Loading LLM: %s", cfg.mlx_model)
        llm = get_llm(cfg)

        logger.info("Loading document store")
        store = DocumentStore(embed_model, cfg)
        store.load_index()

        app.state.db = AppDB(cfg)
        app.state.llm = llm
        app.state.store = store
        app.state.retriever_service = RetrieverService(store, cfg)
        app.state.config = cfg

        logger.info("Server ready")
        yield

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from .routes.documents import router as documents_router
    from .routes.chats import router as chats_router
    from .routes.collections import router as collections_router
    app.include_router(documents_router)
    app.include_router(chats_router)
    app.include_router(collections_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        # Uvicorn only accepts connections after `lifespan` finishes, so a
        # response here already implies every model is loaded. The Electron
        # shell polls this to know when to swap the splash screen for the UI.
        return {"status": "ok"}

    # In a packaged app, FastAPI also serves the built frontend so the whole
    # thing is same-origin — no CORS, no separate Vite process. Mounted last
    # (and only when configured) so it never shadows the API routes above.
    frontend_dir = os.getenv("MYPRIVATEAI_FRONTEND_DIR")
    if frontend_dir:
        app.mount("/", StaticFiles(directory=Path(frontend_dir), html=True), name="frontend")

    return app


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5001"))
    uvicorn.run(create_app(), host=host, port=port)
