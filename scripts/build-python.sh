#!/usr/bin/env bash
# Downloads a standalone, relocatable CPython 3.13 (arm64 macOS) and installs
# the backend's dependencies into it. The resulting build/python/ tree is
# copied into the Electron bundle as-is (Resources/python) — CPython resolves
# sys.prefix from the executable's own location, so no path rewriting is
# needed for it to work from wherever electron-builder places it.
#
# We deliberately do NOT relocate app/.venv: its pyvenv.cfg and script
# shebangs embed this machine's absolute path, so it isn't portable.
#
# Usage: scripts/build-python.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
PYTHON_DIR="$BUILD_DIR/python"

# python-build-standalone release + asset to use. Pinned explicitly so builds
# are reproducible; bump PBS_TAG/PBS_PYVER together when updating.
PBS_TAG="20260825"
PBS_PYVER="3.13.15"
PBS_ASSET="cpython-${PBS_PYVER}+${PBS_TAG}-aarch64-apple-darwin-install_only.tar.gz"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${PBS_ASSET}"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "error: this build targets Apple Silicon (arm64) only — MLX has no Intel path." >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"

if [[ -x "$PYTHON_DIR/bin/python3.13" ]]; then
  echo "== build/python already exists, skipping download/extract (rm -rf build/python to force) =="
else
  echo "== Downloading standalone CPython ${PBS_PYVER} (${PBS_TAG}) =="
  TARBALL="$BUILD_DIR/${PBS_ASSET}"
  curl -fL --progress-bar -o "$TARBALL" "$PBS_URL"

  echo "== Extracting =="
  rm -rf "$PYTHON_DIR"
  # The install_only tarball's top-level dir is named "python/".
  tar -xzf "$TARBALL" -C "$BUILD_DIR"
  rm -f "$TARBALL"
fi

"$PYTHON_DIR/bin/python3.13" --version

echo "== Installing backend requirements into the bundled interpreter =="
"$PYTHON_DIR/bin/python3.13" -m pip install --upgrade pip --quiet
"$PYTHON_DIR/bin/python3.13" -m pip install -r "$REPO_ROOT/app/backend/requirements.txt"

echo "== Done: $PYTHON_DIR ($(du -sh "$PYTHON_DIR" | cut -f1)) =="
