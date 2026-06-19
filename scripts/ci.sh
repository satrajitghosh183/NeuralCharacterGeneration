#!/usr/bin/env bash
# One-shot configure -> build -> test for the H100. Run after each push.
#   LIBTORCH_ROOT=/path/to/libtorch scripts/ci.sh [preset]
set -euo pipefail

PRESET="${1:-h100-release}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${LIBTORCH_ROOT:-}" ]]; then
  echo "ERROR: set LIBTORCH_ROOT to your LibTorch (cxx11-ABI, CUDA 12.x) dir." >&2
  exit 1
fi

echo "==> configure ($PRESET)"
cmake --preset "$PRESET"

echo "==> build"
cmake --build --preset "$PRESET"

echo "==> test"
ctest --test-dir "build/$PRESET" --output-on-failure

echo "==> done"
