"""Unit smoke test for the modular VDB benchmark runner.

This test uses an in-memory exact backend so CI can exercise the modular
orchestrator without requiring Milvus, PostgreSQL/pgvector, or Elasticsearch.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
VDB_BENCHMARK_DIR = ROOT_DIR / "vdb_benchmark"

if str(VDB_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(VDB_BENCHMARK_DIR))

import json
from typing import Any, Dict, List, Optional

import numpy as np

from vdbbench.benchmark.backends.base import (
    CollectionInfo,
    IndexProgress,
    VectorDBBackend,
)
from vdbbench.benchmark.orchestrator import (
    BenchmarkConfig,
    BenchmarkOrchestrator,
)


class FakeExactBackend(VectorDBBackend):
    """Minimal in-memory backend implementing the modular VDB contract."""

    def __init__(self) -> None:
        self.connected = False
        self.collections: Dict[str, Dict[str, Any]] = {}

    def connect(self, **kwargs: Any) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric_type: str = "COSINE",
        index_type: str = "FLAT",
        index_params: Optional[Dict[str, Any]] = None,
        num_shards: int = 1,
        force: bool = False,
    ) -> CollectionInfo:
        if force or name not in self.collections:
            self.collections[name] = {
                "dimension": dimension,
                "metric_type": metric_type,
                "index_type": index_type,
                "index_params": index_params or {},
                "num_shards": num_shards,
                "ids": [],
                "vectors": [],
            }

        return CollectionInfo(
            name=name,
            dimension=dimension,
            metric_type=metric_type,
            index_type=index_type,
            row_count=self.row_count(name),
        )

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def drop_collection(self, name: str) -> None:
        self.collections.pop(name, None)

    def insert_batch(
        self,
        name: str,
        ids: np.ndarray,
        vectors: np.ndarray,
    ) -> int:
        collection = self.collections[name]
        ids = np.asarray(ids, dtype=np.int64)
        vectors = np.asarray(vectors, dtype=np.float32)

        collection["ids"].extend(ids.tolist())
        collection["vectors"].append(vectors)

        return int(len(ids))

    def flush(self, name: str) -> None:
        # In-memory backend has no pending writes.
        return None

    def compact(self, name: str) -> None:
        # In-memory backend has no compaction step.
        return None

    def search(
        self,
        name: str,
        query_vectors: np.ndarray,
        top_k: int,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[List[int]]:
        collection = self.collections[name]

        ids = np.asarray(collection["ids"], dtype=np.int64)
        vectors = np.vstack(collection["vectors"]).astype(np.float32)
        query_vectors = np.asarray(query_vectors, dtype=np.float32)

        # The generator normalizes vectors. The ground-truth path ranks by
        # inner product, which is equivalent for normalized COSINE/IP/L2 cases.
        scores = query_vectors @ vectors.T
        order = np.argsort(-scores, axis=1)[:, :top_k]

        return ids[order].tolist()

    def row_count(self, name: str) -> int:
        if name not in self.collections:
            return 0
        return len(self.collections[name]["ids"])

    def get_index_progress(self, name: str) -> IndexProgress:
        return IndexProgress(
            is_ready=True,
            total_rows=self.row_count(name),
            indexed_rows=self.row_count(name),
            pending_rows=0,
            status="ready",
        )

    def list_collections(self) -> List[str]:
        return sorted(self.collections.keys())

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        collection = self.collections[name]
        return {
            "name": name,
            "row_count": self.row_count(name),
            "dimension": collection["dimension"],
            "metric_type": collection["metric_type"],
            "index_type": collection["index_type"],
            "schema": [
                {"name": "id", "type": "BIGINT", "primary_key": True},
                {
                    "name": "vector",
                    "type": f"VECTOR({collection['dimension']})",
                    "primary_key": False,
                },
            ],
        }

    def list_indexes(self, name: str) -> List[Dict[str, Any]]:
        collection = self.collections[name]
        return [
            {
                "index_name": f"{name}_fake_exact_idx",
                "index_type": collection["index_type"],
                "params": collection["index_params"],
            }
        ]

    def drop_index(
        self,
        name: str,
        index_name: Optional[str] = None,
    ) -> None:
        # No-op for in-memory backend.
        return None


def test_modular_orchestrator_with_fake_exact_backend(tmp_path):
    cfg = BenchmarkConfig(
        mode="both",
        collection_name="ci_fake_exact",
        num_vectors=1000,
        dimension=32,
        distribution="uniform",
        seed=42,
        block_size=250,
        batch_size=100,
        num_query_vectors=50,
        query_seed=99,
        truth_k=10,
        search_k=10,
        num_search_rounds=1,
        search_batch_size=5,
        log_interval=25,
        force=True,
        index_type="FLAT",
        metric_type="COSINE",
    )

    backend = FakeExactBackend()
    backend.connect()

    orchestrator = BenchmarkOrchestrator(cfg, backend)
    summary = orchestrator.run()
    paths = orchestrator.save(str(tmp_path))

    assert summary["total_vectors_inserted"] == 1000
    assert summary["blocks_processed"] == 4
    assert summary["num_query_vectors"] == 50
    assert summary["truth_k"] == 10
    assert summary["truth_table_shape"] == [50, 10]

    assert summary["search_total_queries"] == 50
    assert summary["search_qps"] > 0
    assert summary["search_recall_at_k"] == 1.0
    assert summary["search_latency_mean_ms"] >= 0

    assert "query_vectors" in paths
    assert "ground_truth" in paths
    assert "search_results" in paths
    assert "meta" in paths

    query_vectors = np.load(paths["query_vectors"])
    assert query_vectors.shape == (50, 32)

    ground_truth = np.load(paths["ground_truth"])
    assert ground_truth["truth_table"].shape == (50, 10)
    assert ground_truth["query_vectors"].shape == (50, 32)

    with open(paths["search_results"], "r", encoding="utf-8") as f:
        search_results = json.load(f)

    assert search_results["total_queries"] == 50
    assert search_results["qps"] > 0
    assert search_results["recall_at_k"] == 1.0

    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["config"]["collection_name"] == "ci_fake_exact"
    assert meta["config"]["num_vectors"] == 1000
    assert meta["config"]["dimension"] == 32
