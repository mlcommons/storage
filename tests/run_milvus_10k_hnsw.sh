#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer git repo root when available.
ROOT_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"

# Fallback: walk upward until pyproject.toml is found.
if [[ -z "${ROOT_DIR}" ]]; then
  SEARCH_DIR="${SCRIPT_DIR}"
  while [[ "${SEARCH_DIR}" != "/" && ! -f "${SEARCH_DIR}/pyproject.toml" ]]; do
    SEARCH_DIR="$(dirname "${SEARCH_DIR}")"
  done
  ROOT_DIR="${SEARCH_DIR}"
fi

cd "${ROOT_DIR}"

if [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: pyproject.toml not found."
  echo "Current directory: $(pwd)"
  echo "Set ROOT_DIR manually or run this script from inside the repo."
  exit 1
fi

CONFIG="${CONFIG:-tests/configs/milvus_10k_hnsw.yaml}"
OUT_DIR="${OUT_DIR:-/tmp/pr316_milvus_hnsw_10k}"

MILVUS_HOST="${MILVUS_HOST:-127.0.0.1}"
MILVUS_PORT="${MILVUS_PORT:-19530}"

echo "Running Milvus modular VDB smoke test"
echo "Repo root: $(pwd)"
echo "Config: ${CONFIG}"
echo "Output: ${OUT_DIR}"
echo "Milvus: ${MILVUS_HOST}:${MILVUS_PORT}"

uv sync --extra vectordb-milvus
uv pip install -e ./vdb_benchmark

rm -rf "${OUT_DIR}"

MILVUS__HOST="${MILVUS_HOST}" \
MILVUS__PORT="${MILVUS_PORT}" \
uv run python -m vdbbench.benchmark \
  --config "${CONFIG}" \
  --backend milvus \
  --mode both \
  --force \
  --output-dir "${OUT_DIR}"

test -f "${OUT_DIR}/query_vectors.npy"
test -f "${OUT_DIR}/ground_truth.npz"
test -f "${OUT_DIR}/search_results.json"
test -f "${OUT_DIR}/benchmark_meta.json"

uv run python - <<PY
import json
from pathlib import Path

out_dir = Path("${OUT_DIR}")
with open(out_dir / "search_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

assert results["total_queries"] == 400, results
assert results["qps"] > 0, results
assert 0 <= results["recall_at_k"] <= 1, results

print("Milvus smoke test passed")
print(json.dumps({
    "total_queries": results["total_queries"],
    "qps": results["qps"],
    "recall_at_k": results["recall_at_k"],
    "latency_p50_ms": results.get("latency_p50_ms"),
    "latency_p99_ms": results.get("latency_p99_ms"),
}, indent=2))
PY
