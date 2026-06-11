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


CONFIG="${CONFIG:-tests/configs/elasticsearch_5k_hnsw.yaml}"
OUT_DIR="${OUT_DIR:-/tmp/pr316_elasticsearch_hnsw_5k}"

ELASTICSEARCH_HOST="${ELASTICSEARCH_HOST:-http://localhost:9200}"
ELASTICSEARCH_API_KEY="${ELASTICSEARCH_API_KEY:-}"
ELASTICSEARCH_CLOUD_ID="${ELASTICSEARCH_CLOUD_ID:-}"

echo "Running Elasticsearch modular VDB smoke test"
echo "Config: ${CONFIG}"
echo "Output: ${OUT_DIR}"
echo "Elasticsearch host: ${ELASTICSEARCH_HOST}"

uv sync --extra vectordb-elasticsearch
uv pip install -e ./vdb_benchmark

rm -rf "${OUT_DIR}"

if [[ -n "${ELASTICSEARCH_API_KEY}" || -n "${ELASTICSEARCH_CLOUD_ID}" ]]; then
  ELASTICSEARCH__HOST="${ELASTICSEARCH_HOST}" \
  ELASTICSEARCH__API_KEY="${ELASTICSEARCH_API_KEY}" \
  ELASTICSEARCH__CLOUD_ID="${ELASTICSEARCH_CLOUD_ID}" \
  uv run python -m vdbbench.benchmark \
    --config "${CONFIG}" \
    --backend elasticsearch \
    --mode both \
    --force \
    --output-dir "${OUT_DIR}"
else
  ELASTICSEARCH__HOST="${ELASTICSEARCH_HOST}" \
  uv run python -m vdbbench.benchmark \
    --config "${CONFIG}" \
    --backend elasticsearch \
    --mode both \
    --force \
    --output-dir "${OUT_DIR}"
fi

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

assert results["total_queries"] == 200, results
assert results["qps"] > 0, results
assert 0 <= results["recall_at_k"] <= 1, results

print("Elasticsearch smoke test passed")
print(json.dumps({
    "total_queries": results["total_queries"],
    "qps": results["qps"],
    "recall_at_k": results["recall_at_k"],
    "latency_p50_ms": results["latency_p50_ms"],
    "latency_p99_ms": results["latency_p99_ms"],
}, indent=2))
PY
