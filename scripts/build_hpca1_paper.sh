#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$REPO_ROOT/paper/hpca1_hail_mary/revised_paper"

export LC_ALL=C
export TZ=UTC
export SOURCE_DATE_EPOCH=1785456000

make -C "$PAPER_DIR" clean
make -C "$PAPER_DIR"
python3 "$REPO_ROOT/scripts/preflight_hpca1_paper.py" \
  --paper-dir "$PAPER_DIR" \
  --json-out "$PAPER_DIR/BUILD_PREFLIGHT.json"
sha256sum "$PAPER_DIR/main.tex" "$PAPER_DIR/main.pdf"
