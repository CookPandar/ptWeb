#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PT_VIEWER_PYTHON:-python3}"

exec "$PYTHON_BIN" -m app.main
