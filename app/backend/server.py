"""Flask app factory for MyPrivateAI backend.

Run with:
    python -m app.backend.server
"""

from __future__ import annotations

import logging

from flask import Flask
from flask_cors import CORS
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .db.app_db import AppDB
from .pipeline.common.config import Config
from .pipeline.common.llm import get_llm
from .pipeline.ingestion.document_store import DocumentStore
from .services.retrieval import RetrieverService

logging.basicConfig(level=logging.INFO)


def create_app(config: Config | None = None) -> Flask:
    app = Flask(__name__)
    CORS(app, origins=["http://localhost:5173"])

    cfg = config or Config()

    # ── Persistent storage ────────────────────────────────────────────
    db = AppDB(cfg)

    # ── Heavy resources (loaded once; shared across all requests) ─────
    app.logger.info("Loading embedding model: %s", cfg.embedding_model)
    embed_model = HuggingFaceEmbedding(
        model_name=cfg.embedding_model,
        device=cfg.device,
    )

    app.logger.info("Loading LLM: %s", cfg.mlx_model)
    llm = get_llm(cfg)

    app.logger.info("Loading document store")
    store = DocumentStore(embed_model, cfg)
    store.load_index()

    retriever_service = RetrieverService(store, cfg)

    # ── Attach to app config for blueprint access ─────────────────────
    app.config["DB"] = db
    app.config["LLM"] = llm
    app.config["STORE"] = store
    app.config["RETRIEVER_SERVICE"] = retriever_service
    app.config["PIPELINE_CONFIG"] = cfg

    # ── Blueprints ────────────────────────────────────────────────────
    from .routes.documents import bp as documents_bp
    from .routes.chats import bp as chats_bp
    app.register_blueprint(documents_bp)
    app.register_blueprint(chats_bp)

    app.logger.info("Server ready")
    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5001, debug=False)
