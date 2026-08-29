#!/usr/bin/env bash
# Copies only the model caches the shipping app actually uses into
# build/models/, laid out exactly as Resources/models/ will be in the
# bundle: hf/ (HF_HOME), reranker_models/ (CrossEncoder cache_folder — must
# match Config.reranker_cache_dir's subdirectory name exactly), nltk/
# (NLTK_DATA). Everything else in ~/.cache/huggingface is Validation-only
# (specter2, bge-small, docling) and is intentionally left out.
#
# Requires the models to already be present locally, i.e. the app has been
# run at least once in dev so HuggingFace / the reranker have downloaded
# them (see README).
#
# Usage: scripts/stage-models.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
MODELS_DIR="$BUILD_DIR/models"

HF_HUB="$HOME/.cache/huggingface/hub"
LLAMA_MODEL_DIR="models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit"
BGE_BASE_MODEL_DIR="models--BAAI--bge-base-en-v1.5"

RERANKER_SRC="$REPO_ROOT/app/backend/data/reranker_models"
NLTK_SRC="$REPO_ROOT/app/backend/data/nltk_data"

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "error: expected model dir missing: $1" >&2
    echo "  Run the app in dev mode first (python -m app.backend.server) so it downloads." >&2
    exit 1
  fi
}

require_dir "$HF_HUB/$LLAMA_MODEL_DIR"
require_dir "$HF_HUB/$BGE_BASE_MODEL_DIR"
require_dir "$RERANKER_SRC"
require_dir "$NLTK_SRC"

echo "== Staging models into $MODELS_DIR =="
mkdir -p "$MODELS_DIR/hf/hub" "$MODELS_DIR/reranker_models" "$MODELS_DIR/nltk"

# --info=progress2 needs GNU rsync; macOS ships openrsync, so stick to the
# portable --progress flag (per-file, not aggregate).
rsync -a --progress "$HF_HUB/$LLAMA_MODEL_DIR" "$MODELS_DIR/hf/hub/"
rsync -a --progress "$HF_HUB/$BGE_BASE_MODEL_DIR" "$MODELS_DIR/hf/hub/"
rsync -a --progress "$RERANKER_SRC/" "$MODELS_DIR/reranker_models/"
rsync -a --progress "$NLTK_SRC/" "$MODELS_DIR/nltk/"

echo "== Done: $MODELS_DIR ($(du -sh "$MODELS_DIR" | cut -f1)) =="
echo "   hf:       $(du -sh "$MODELS_DIR/hf" | cut -f1)"
echo "   reranker: $(du -sh "$MODELS_DIR/reranker_models" | cut -f1)"
echo "   nltk:     $(du -sh "$MODELS_DIR/nltk" | cut -f1)"
