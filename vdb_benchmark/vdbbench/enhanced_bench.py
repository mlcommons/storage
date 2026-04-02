#!/usr/bin/env python3
"""
enhanced_bench.py  (merged: enhanced_bench + simple_bench)

Unified Milvus vector search benchmark that combines:

FROM enhanced_bench (advanced features):
- Index-aware search params (HNSW/DISKANN/AISAQ/IVF*/FLAT)
- GT disk cache (per dataset + query-set + metric + K)
- Single-thread and multi-process execution
- Parameter sweep to hit one or multiple recall targets and select "best" params
- Warm/cold/both cache regimes (cold uses host drop-caches command)
- Optional disk I/O deltas via /proc/diskstats (Linux)
- Optional container-aware RSS via `docker stats --no-stream`
- Host memory snapshots via /proc/meminfo (before/after each run)
- Budget mode: container RSS budget + host MemAvailable reserve budget
- Memory footprint estimator (rough planning) based on index type + vector count + dim
- YAML config support

FROM simple_bench (operational features):
- Automated FLAT ground truth collection creation from source collection
  (copy all vectors + PKs, build FLAT index — no manual GT prep required)
- Full per-query recall statistics (p95/p99 recall, not just mean)
- Runtime-based AND query-count-based benchmark execution modes
- Per-worker CSV output with staggered process startup
- Full latency statistics (P99.9, P99.99) via pandas
- Collection verification + tabulate display
- Graceful shutdown via SIGINT/SIGTERM
- search_ef / search_list CLI override for precise parameter control

Guardrails:
- Fail fast if vector field is BINARY_VECTOR (assumes FLOAT vectors).

YAML support:
- Use --config path.yaml to load defaults.  CLI flags override YAML.

Usage modes:
  Execution path A (timed / query-count, simple_bench style):
      python enhanced_bench.py --collection <name> --runtime 120 --batch-size 10 --processes 4
      python enhanced_bench.py --collection <name> --queries 10000 --batch-size 10

  Execution path B (sweep / budget, enhanced_bench style):
      python enhanced_bench.py --collection <name> --mode both --sweep --cache-state both
      python enhanced_bench.py --collection <name> --mode single --target-recall 0.95

  Estimator-only mode:
      python enhanced_bench.py --estimate-only --est-index-type HNSW --est-n 10000000 --est-dim 1536
"""

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import shlex
import signal
import subprocess
import sys
import time
import hashlib
import uuid
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from pymilvus import (Collection, CollectionSchema, FieldSchema,
                          connections, utility, DataType)
except ImportError:
    print("Error: pymilvus not found. Install with: pip install pymilvus numpy")
    sys.exit(1)

try:
    import yaml  # pip install pyyaml
except ImportError:
    yaml = None

try:
    from tabulate import tabulate as _tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# Optional vdbbench package imports (available when running from the repo)
try:
    from vdbbench.config_loader import load_config, merge_config_with_args
    from vdbbench.list_collections import get_collection_info
    _VDBBENCH_PKG = True
except ImportError:
    _VDBBENCH_PKG = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGGER_INTERVAL_SEC = 0.1

# Global flag for graceful shutdown (simple_bench execution path)
shutdown_flag = mp.Value('i', 0)

# CSV header fields for per-worker output files
csv_fields = [
    "process_id",
    "batch_id",
    "timestamp",
    "batch_size",
    "batch_time_seconds",
    "avg_query_time_seconds",
    "success",
]


# =============================================================================
# YAML helpers (CLI overrides YAML)
# =============================================================================

def load_yaml_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise SystemExit("pyyaml is required for --config. Install with: pip install pyyaml")
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"YAML config not found: {path}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"YAML root must be a mapping/dict. Got: {type(data)}")
    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override into base recursively.
    Dicts merge recursively; Lists/Scalars overwrite."""
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_yaml_to_args(args: argparse.Namespace, cfg: Dict[str, Any],
                       ap: argparse.ArgumentParser) -> argparse.Namespace:
    """YAML provides defaults, CLI wins."""
    dest_to_opts: Dict[str, List[str]] = {}
    for action in ap._actions:
        if not action.option_strings:
            continue
        dest_to_opts.setdefault(action.dest, []).extend(action.option_strings)

    argv = set(sys.argv[1:])

    def user_set(dest: str) -> bool:
        return any(opt in argv for opt in dest_to_opts.get(dest, []))

    for k, v in (cfg or {}).items():
        dest = k.replace("-", "_")
        if not hasattr(args, dest):
            continue
        if user_set(dest):
            continue
        setattr(args, dest, v)

    return args


# =============================================================================
# Diskstats (Linux)
# =============================================================================

def read_disk_stats() -> Dict[str, Dict[str, int]]:
    """
    Read disk I/O statistics from /proc/diskstats.

    Captures per-device:
      - bytes_read / bytes_written  (sectors × 512)
      - read_ios   / write_ios      (completed I/O operations — source of IOPS)
      - read_ms    / write_ms       (time spent in read/write I/Os in ms)

    /proc/diskstats field layout (1-indexed):
      [1] major  [2] minor  [3] device
      [4] reads_completed   [5] reads_merged    [6] sectors_read  [7] read_ms
      [8] writes_completed  [9] writes_merged  [10] sectors_written [11] write_ms
      [12] ios_in_progress [13] io_ms          [14] weighted_io_ms
    """
    stats: Dict[str, Dict[str, int]] = {}
    try:
        with open("/proc/diskstats", "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 14:
                    dev = parts[2]
                    read_ios   = int(parts[3])
                    sectors_read = int(parts[5])
                    read_ms    = int(parts[6])
                    write_ios  = int(parts[7])
                    sectors_written = int(parts[9])
                    write_ms   = int(parts[10])
                    stats[dev] = {
                        "bytes_read":    sectors_read    * 512,
                        "bytes_written": sectors_written * 512,
                        "read_ios":      read_ios,
                        "write_ios":     write_ios,
                        "read_ms":       read_ms,
                        "write_ms":      write_ms,
                    }
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return stats


def disk_stats_diff(a: Dict[str, Dict[str, int]],
                    b: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """Return field-by-field delta between two read_disk_stats() snapshots."""
    out: Dict[str, Dict[str, int]] = {}
    fields = ("bytes_read", "bytes_written", "read_ios", "write_ios",
              "read_ms", "write_ms")
    for dev in b:
        if dev in a:
            out[dev] = {f: b[dev].get(f, 0) - a[dev].get(f, 0) for f in fields}
    return out


# Alias used in the simple_bench execution path
calculate_disk_io_diff = disk_stats_diff


def filter_real_disk_devices(stats: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """Filter out virtual/loop devices, keeping only real disks."""
    excluded_prefixes = ['loop', 'ram', 'dm-', 'sr', 'md']
    return {dev: data for dev, data in stats.items()
            if not any(dev.startswith(prefix) for prefix in excluded_prefixes)}


def format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    v = float(n)
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    return f"{v:.2f} {units[i]}"


# =============================================================================
# Host memory snapshot (/proc/meminfo)
# =============================================================================

@dataclass
class HostMemSnapshot:
    ts: float
    mem_total_bytes: int
    mem_free_bytes: int
    mem_available_bytes: int
    buffers_bytes: int
    cached_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int

    @staticmethod
    def from_proc_meminfo() -> "HostMemSnapshot":
        kv: Dict[str, int] = {}
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        val = int(parts[1])
                        unit = parts[2] if len(parts) >= 3 else "kB"
                        kv[key] = val * 1024 if unit.lower() == "kb" else val
        except Exception:
            kv = {}

        def g(k: str) -> int:
            return int(kv.get(k, 0))

        return HostMemSnapshot(
            ts=time.time(),
            mem_total_bytes=g("MemTotal"),
            mem_free_bytes=g("MemFree"),
            mem_available_bytes=g("MemAvailable"),
            buffers_bytes=g("Buffers"),
            cached_bytes=g("Cached"),
            swap_total_bytes=g("SwapTotal"),
            swap_free_bytes=g("SwapFree"),
        )


def bytes_to_gb(x: int) -> float:
    return x / (1024 ** 3)


# =============================================================================
# Shell helpers + container RSS
# =============================================================================

def run_cmd(cmd: str) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, shell=True, check=False,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def parse_human_bytes(s: str) -> int:
    s = s.strip()
    if not s:
        return 0
    parts = s.replace("iB", "ib").replace("IB", "ib").split()
    if len(parts) == 1:
        token = parts[0]
        num = ""
        unit = ""
        for ch in token:
            if ch.isdigit() or ch in ".-":
                num += ch
            else:
                unit += ch
        try:
            val = float(num)
        except Exception:
            return 0
        unit = unit.strip().lower()
    else:
        try:
            val = float(parts[0])
        except Exception:
            return 0
        unit = parts[1].strip().lower()

    scale_map = {
        "b": 1, "": 1,
        "kib": 1024, "ki": 1024, "k": 1024,
        "mib": 1024 ** 2, "mi": 1024 ** 2, "m": 1024 ** 2,
        "gib": 1024 ** 3, "gi": 1024 ** 3, "g": 1024 ** 3,
        "tib": 1024 ** 4, "ti": 1024 ** 4, "t": 1024 ** 4,
        "kb": 1000, "mb": 1000 ** 2, "gb": 1000 ** 3, "tb": 1000 ** 4,
    }
    return int(val * scale_map.get(unit, 1))


def get_rss_bytes_for_containers(container_names: List[str]) -> Optional[int]:
    if not container_names:
        return None
    total = 0
    any_ok = False
    for name in container_names:
        cmd = f'docker stats --no-stream --format "{{{{.MemUsage}}}}" {shlex.quote(name)}'
        rc, out, _err = run_cmd(cmd)
        if rc != 0 or not out:
            continue
        any_ok = True
        mem_usage = out.split("/")[0].strip()
        total += parse_human_bytes(mem_usage)
    return total if any_ok else None


# =============================================================================
# Signal handling (graceful shutdown)
# =============================================================================

def signal_handler(sig, frame):
    """Handle SIGINT/SIGTERM to gracefully stop worker processes."""
    print("\nReceived interrupt signal. Shutting down workers gracefully...")
    with shutdown_flag.get_lock():
        shutdown_flag.value = 1


# =============================================================================
# Recall metric — full per-query statistics  (from simple_bench)
# =============================================================================

def calc_recall(
    ann_results: Dict[int, List[int]],
    ground_truth: Dict[int, List[int]],
    k: int,
) -> Dict[str, Any]:
    """
    Calculate recall@k by comparing ANN search results against ground truth.

    Follows the VectorDBBench approach:
      recall@k = |ANN_top_k ∩ GT_top_k| / k

    Ground truth should come from a FLAT (brute-force) index which guarantees
    exact nearest neighbor results, NOT from the ANN index itself.

    Args:
        ann_results: Dict mapping query_index -> list of IDs from ANN search.
        ground_truth: Dict mapping query_index -> list of true nearest neighbor
                      IDs from FLAT index search.
        k: Number of top results to evaluate.

    Returns:
        Dict with recall statistics (mean, min, max, percentiles).
    """
    per_query_recall = []

    for query_idx in sorted(ann_results.keys()):
        if query_idx not in ground_truth:
            continue
        ann_ids = set(ann_results[query_idx][:k])
        gt_ids = set(ground_truth[query_idx][:k])
        if len(gt_ids) == 0:
            continue
        intersection_size = len(ann_ids & gt_ids)
        per_query_recall.append(intersection_size / k)

    if not per_query_recall:
        return {
            "recall_at_k": 0.0,
            "num_queries_evaluated": 0,
            "k": k,
            "min_recall": 0.0,
            "max_recall": 0.0,
            "mean_recall": 0.0,
            "median_recall": 0.0,
            "p95_recall": 0.0,
            "p99_recall": 0.0,
        }

    recalls_arr = np.array(per_query_recall)
    return {
        "recall_at_k": float(np.mean(recalls_arr)),
        "num_queries_evaluated": len(per_query_recall),
        "k": k,
        "min_recall": float(np.min(recalls_arr)),
        "max_recall": float(np.max(recalls_arr)),
        "mean_recall": float(np.mean(recalls_arr)),
        "median_recall": float(np.median(recalls_arr)),
        "p95_recall": float(np.percentile(recalls_arr, 95)),
        "p99_recall": float(np.percentile(recalls_arr, 99)),
    }


# Simpler scalar recall used by enhanced_bench execution path
def recall_at_k(gt: List[List[Any]], pred: List[List[Any]], k: int) -> float:
    if not gt or not pred or len(gt) != len(pred):
        return 0.0
    hit_sum = 0
    for g, p in zip(gt, pred):
        hit_sum += len(set(g[:k]).intersection(set(p[:k])))
    return hit_sum / (len(gt) * k)


# =============================================================================
# Schema detection helpers  (from simple_bench)
# =============================================================================

def _detect_schema_fields(collection: Collection) -> Tuple[str, str, DataType]:
    """
    Detect primary key and vector field names from a collection's schema.

    Returns:
        (pk_field_name, vector_field_name, pk_dtype) tuple.

    Raises:
        ValueError if required fields cannot be detected.
    """
    pk_field = None
    pk_dtype = None
    vec_field = None
    for field in collection.schema.fields:
        if field.is_primary:
            pk_field = field.name
            pk_dtype = field.dtype
        if field.dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR,
                           getattr(DataType, "FLOAT16_VECTOR", None),
                           getattr(DataType, "BFLOAT16_VECTOR", None)):
            if field.dtype is not None:
                vec_field = field.name

    if pk_field is None:
        raise ValueError(f"Cannot detect primary key field in collection "
                         f"'{collection.name}'. Schema: {collection.schema}")
    if vec_field is None:
        raise ValueError(f"Cannot detect vector field in collection "
                         f"'{collection.name}'. Schema: {collection.schema}")

    return pk_field, vec_field, pk_dtype


# =============================================================================
# Milvus helpers  (from enhanced_bench)
# =============================================================================

def _dtype_to_str(dt) -> str:
    if hasattr(dt, "name"):
        return dt.name
    try:
        return DataType(dt).name
    except Exception:
        return str(dt)


def _is_vector_dtype(dt) -> bool:
    vec_types = {DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR}
    if hasattr(DataType, "FLOAT16_VECTOR"):
        vec_types.add(DataType.FLOAT16_VECTOR)
    if hasattr(DataType, "BFLOAT16_VECTOR"):
        vec_types.add(DataType.BFLOAT16_VECTOR)
    return dt in vec_types


def get_vector_field_info(collection: Collection) -> Tuple[Optional[str], Optional[int], Optional[Any], Optional[str]]:
    """Returns: (vector_field_name, dim, dtype_obj, dtype_name)"""
    for field in collection.schema.fields:
        dt = getattr(field, "dtype", None)
        if dt is not None and _is_vector_dtype(dt):
            dim = field.params.get("dim")
            return field.name, dim, dt, _dtype_to_str(dt)
    return None, None, None, None


def is_binary_vector_dtype(dtype_obj) -> bool:
    return dtype_obj == DataType.BINARY_VECTOR


def get_index_params(collection: Collection) -> Tuple[str, str, Dict[str, Any]]:
    """Returns (index_type, metric_type, build_params)"""
    if not collection.indexes:
        return "FLAT", "L2", {}
    idx = collection.indexes[0]
    idx_type = idx.params.get("index_type", "FLAT")
    metric_type = idx.params.get("metric_type", "L2")
    build_params = idx.params.get("params", {}) or {}
    return idx_type, metric_type, build_params


def minimal_search_params_for_index(index_type: str) -> Dict[str, Any]:
    """Minimal params for maximum throughput (at cost of lower recall)."""
    t = (index_type or "FLAT").lower()
    if t == "hnsw":
        return {"ef": 10}
    if t == "diskann":
        return {"search_list": 10}
    if t == "aisaq":
        return {"search_list": 10}
    if t.startswith("ivf"):
        return {"nprobe": 1}
    return {}


def default_search_params_for_index(index_type: str, build_params: Dict[str, Any]) -> Dict[str, Any]:
    t = (index_type or "FLAT").lower()
    if t == "hnsw":
        return {"ef": 128}
    if t == "diskann":
        return {"search_list": 200}
    if t == "aisaq":
        return {"search_list": int(build_params.get("search_list_size", 100))}
    if t.startswith("ivf"):
        nlist = int(build_params.get("nlist", 1024))
        return {"nprobe": max(1, min(16, nlist // 8))}
    return {}


def validate_search_params(index_type: str, params: Dict[str, Any]) -> None:
    """Validate search parameters for the given index type."""
    t = (index_type or "FLAT").lower()
    if t == "hnsw":
        ef = params.get("ef", 0)
        if ef <= 0:
            raise ValueError(f"Invalid HNSW ef={ef}, must be > 0")
    elif t == "diskann":
        sl = params.get("search_list", 0)
        if sl <= 0:
            raise ValueError(f"Invalid DiskANN search_list={sl}, must be > 0")
    elif t == "aisaq":
        sl = params.get("search_list", 0)
        if sl <= 0:
            raise ValueError(f"Invalid AISAQ search_list={sl}, must be > 0")
    elif t.startswith("ivf"):
        nprobe = params.get("nprobe", 0)
        if nprobe <= 0:
            raise ValueError(f"Invalid IVF nprobe={nprobe}, must be > 0")


def make_search_params_full(metric_type: str, algo_params: Dict[str, Any]) -> Dict[str, Any]:
    return {"metric_type": metric_type, "params": algo_params or {}}


def normalize_for_cosine(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def generate_queries(dim: int, count: int, seed: int, normalize: bool) -> np.ndarray:
    """Generate queries as NumPy array (enhanced_bench path)."""
    rng = np.random.default_rng(seed)
    q = rng.random((count, dim), dtype=np.float32)
    return normalize_for_cosine(q) if normalize else q


def generate_query_vectors(num_queries: int, dimension: int, seed: int = 42) -> List[List[float]]:
    """
    Pre-generate a fixed set of query vectors as Python lists.

    Pre-generating ensures:
      - Consistent queries between ANN and FLAT searches
      - Ground truth can be computed before the timed benchmark
      - No random generation overhead during the benchmark

    Args:
        num_queries: Number of query vectors to generate.
        dimension: Vector dimension.
        seed: Random seed for reproducibility.

    Returns:
        List of normalized query vectors.
    """
    rng = np.random.RandomState(seed)
    vectors = rng.random((num_queries, dimension)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    return vectors.tolist()


def generate_random_vector(dim: int) -> List[float]:
    """Generate a single random normalized vector."""
    vec = np.random.random(dim).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


# =============================================================================
# GT cache helpers  (from enhanced_bench)
# =============================================================================

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def gt_signature(
    gt_collection_name: str,
    gt_num_entities: int,
    gt_vector_field: str,
    dim: int,
    metric_type: str,
    k: int,
    query_seed: int,
    query_count: int,
    normalize_cosine: bool,
) -> Dict[str, Any]:
    return {
        "gt_collection": gt_collection_name,
        "gt_num_entities": int(gt_num_entities),
        "gt_vector_field": gt_vector_field,
        "dim": int(dim),
        "metric_type": str(metric_type).upper(),
        "k": int(k),
        "query_seed": int(query_seed),
        "query_count": int(query_count),
        "normalize_cosine": bool(normalize_cosine),
        "version": 2,
    }


def gt_cache_paths(cache_dir: Path, signature: Dict[str, Any]) -> Tuple[Path, Path]:
    key = sha256_hex(json.dumps(signature, sort_keys=True))
    npz_path = cache_dir / f"gt_{key}.npz"
    meta_path = cache_dir / f"gt_{key}.meta.json"
    return npz_path, meta_path


def save_gt_cache(npz_path: Path, meta_path: Path,
                  signature: Dict[str, Any], gt_ids: List[List[Any]]) -> None:
    arr = np.array(gt_ids, dtype=object)
    np.savez_compressed(npz_path, ids=arr)
    meta_path.write_text(json.dumps(signature, indent=2, sort_keys=True), encoding="utf-8")


def load_gt_cache(npz_path: Path) -> List[List[Any]]:
    data = np.load(npz_path, allow_pickle=True)
    arr = data["ids"]
    return arr.tolist()


# =============================================================================
# FLAT GT collection creation  (from simple_bench — major new addition)
# =============================================================================

def create_flat_collection(
    host: str,
    port: str,
    source_collection_name: str,
    flat_collection_name: str,
    vector_dim: int,
    metric_type: str = "COSINE",
) -> bool:
    """
    Create a duplicate collection with FLAT index for ground truth computation.

    FLAT index performs brute-force exact search which gives true nearest
    neighbors — unlike ANN indexes (DiskANN, HNSW, IVF) which approximate.

    CRITICAL: The FLAT collection preserves the source collection's primary
    key values (auto_id=False). This ensures that the IDs returned by FLAT
    search match the IDs returned by the ANN search on the source collection,
    so the recall set-intersection calculation is correct.

    Uses query_iterator() to avoid the Milvus maxQueryResultWindow offset
    limit (default 16384) that breaks offset-based pagination on large
    collections.

    Args:
        host: Milvus server host.
        port: Milvus server port.
        source_collection_name: Name of the original ANN-indexed collection.
        flat_collection_name: Name for the new FLAT-indexed collection.
        vector_dim: Vector dimension.
        metric_type: Distance metric (COSINE, L2, IP).

    Returns:
        True if the FLAT collection is ready, False on failure.
    """
    conn_alias = f"flat_setup_{uuid.uuid4().hex[:8]}"
    try:
        connections.connect(alias=conn_alias, host=host, port=port)
    except Exception as e:
        print(f"Failed to connect for FLAT collection setup: {e}")
        return False

    try:
        # Re-use existing FLAT collection if it's already fully populated
        if utility.has_collection(flat_collection_name, using=conn_alias):
            flat_coll = Collection(flat_collection_name, using=conn_alias)
            source_coll = Collection(source_collection_name, using=conn_alias)
            if flat_coll.num_entities > 0 and flat_coll.num_entities == source_coll.num_entities:
                print(f"FLAT collection '{flat_collection_name}' already exists "
                      f"with {flat_coll.num_entities} vectors, reusing it.")
                flat_coll.load()
                return True
            else:
                print(f"FLAT collection exists but has {flat_coll.num_entities} vs "
                      f"{source_coll.num_entities} vectors. Dropping and recreating...")
                utility.drop_collection(flat_collection_name, using=conn_alias)

        print(f"Creating FLAT collection '{flat_collection_name}' "
              f"from source '{source_collection_name}'...")

        source_coll = Collection(source_collection_name, using=conn_alias)
        source_coll.load()
        # Flush to ensure num_entities is up-to-date
        source_coll.flush()
        total_vectors = source_coll.num_entities
        if total_vectors == 0:
            print(f"ERROR: Source collection '{source_collection_name}' "
                  f"reports 0 vectors after flush. Cannot create ground truth.")
            return False

        src_pk_field, src_vec_field, src_pk_dtype = _detect_schema_fields(source_coll)
        print(f"Source schema: pk_field='{src_pk_field}' ({src_pk_dtype.name}), "
              f"vec_field='{src_vec_field}', vectors={total_vectors}")

        # CRITICAL: auto_id=False — copy source PK values so FLAT search IDs
        # match ANN search IDs in the recall set-intersection.
        pk_kwargs = {"max_length": 256} if src_pk_dtype == DataType.VARCHAR else {}
        fields = [
            FieldSchema(name="pk", dtype=src_pk_dtype,
                        is_primary=True, auto_id=False, **pk_kwargs),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        ]
        schema = CollectionSchema(fields, description="FLAT index ground truth collection")
        flat_coll = Collection(flat_collection_name, schema, using=conn_alias)

        copy_batch_size = 5000
        print(f"Copying {total_vectors} vectors to FLAT collection "
              f"(batch_size={copy_batch_size})...")

        copied = 0
        use_iterator = hasattr(source_coll, 'query_iterator')

        if use_iterator:
            # pymilvus >= 2.3: use built-in iterator
            try:
                iterator = source_coll.query_iterator(
                    batch_size=copy_batch_size,
                    output_fields=[src_pk_field, src_vec_field],
                )
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    pk_values = [row[src_pk_field] for row in batch]
                    vectors = [row[src_vec_field] for row in batch]
                    flat_coll.insert([pk_values, vectors])
                    copied += len(vectors)
                    if copied % (copy_batch_size * 20) < copy_batch_size:
                        print(f"  Copied {copied}/{total_vectors} vectors "
                              f"({100.0 * copied / total_vectors:.1f}%)")
                iterator.close()
            except Exception as iter_err:
                print(f"  query_iterator failed ({iter_err}), "
                      f"falling back to pk-cursor pagination...")
                use_iterator = False
                copied = 0
                utility.drop_collection(flat_collection_name, using=conn_alias)
                flat_coll = Collection(flat_collection_name, schema, using=conn_alias)

        if not use_iterator:
            # Fallback: pk-cursor pagination + search-based vector retrieval.
            # query() cannot return vector fields on many Milvus versions;
            # use search() with output_fields instead.
            is_int_pk = src_pk_dtype in (DataType.INT64, DataType.INT32,
                                         DataType.INT16, DataType.INT8)
            last_pk = -2 ** 63 if is_int_pk else ""
            page_limit = min(copy_batch_size, 16384)
            dummy_vec = np.random.random(vector_dim).astype(np.float32)
            dummy_vec = (dummy_vec / np.linalg.norm(dummy_vec)).tolist()

            while copied < total_vectors:
                expr = (f"{src_pk_field} > {last_pk}" if is_int_pk
                        else f'{src_pk_field} > "{last_pk}"')

                try:
                    pk_batch = source_coll.query(
                        expr=expr, output_fields=[src_pk_field], limit=page_limit)
                except Exception as qe:
                    print(f"  query() failed: {qe}")
                    break
                if not pk_batch:
                    break

                pk_batch.sort(key=lambda r: r[src_pk_field] if is_int_pk else str(r[src_pk_field]))
                last_pk = pk_batch[-1][src_pk_field]
                pk_values_batch = [row[src_pk_field] for row in pk_batch]

                if is_int_pk:
                    pk_filter = f"{src_pk_field} in {pk_values_batch}"
                else:
                    escaped = [str(v).replace('"', '\\"') for v in pk_values_batch]
                    pk_filter = (f'{src_pk_field} in [' +
                                 ','.join(f'"{v}"' for v in escaped) + ']')

                try:
                    search_results = source_coll.search(
                        data=[dummy_vec], anns_field=src_vec_field,
                        param={"metric_type": metric_type, "params": {}},
                        limit=len(pk_values_batch), expr=pk_filter,
                        output_fields=[src_vec_field],
                    )
                except Exception as se:
                    print(f"  search() for vector retrieval failed: {se}")
                    break

                pk_vec_map = {}
                if search_results:
                    for hit in search_results[0]:
                        hit_vec = hit.entity.get(src_vec_field)
                        if hit_vec is not None:
                            pk_vec_map[hit.id] = hit_vec

                insert_pks = []
                insert_vecs = []
                for pk_val in pk_values_batch:
                    if pk_val in pk_vec_map:
                        insert_pks.append(pk_val)
                        insert_vecs.append(pk_vec_map[pk_val])

                if insert_pks:
                    flat_coll.insert([insert_pks, insert_vecs])
                    copied += len(insert_pks)
                else:
                    # Last-resort: direct query with vector output (pymilvus >= 2.3)
                    try:
                        vec_batch = source_coll.query(
                            expr=pk_filter,
                            output_fields=[src_pk_field, src_vec_field],
                            limit=len(pk_values_batch),
                        )
                        if vec_batch:
                            pks = [row[src_pk_field] for row in vec_batch]
                            vecs = [row[src_vec_field] for row in vec_batch]
                            flat_coll.insert([pks, vecs])
                            copied += len(pks)
                    except Exception:
                        print(f"  WARNING: Could not retrieve vectors for "
                              f"{len(pk_values_batch)} PKs, skipping batch.")
                        continue

                if copied % (page_limit * 20) < page_limit:
                    pct = min(100.0, 100.0 * copied / total_vectors)
                    print(f"  Copied {copied}/{total_vectors} vectors ({pct:.1f}%)")

        print(f"  Copied {copied}/{total_vectors} vectors (100.0%)")
        flat_coll.flush()

        # Wait for entity count to stabilize after flush
        actual_count = 0
        for attempt in range(10):
            actual_count = flat_coll.num_entities
            if actual_count >= copied:
                break
            time.sleep(1)
            print(f"  Waiting for flush to complete ({actual_count}/{copied} visible)...")

        if actual_count < copied:
            print(f"  WARNING: Only {actual_count}/{copied} vectors visible "
                  f"after flush. Proceeding anyway.")

        # Build FLAT index (brute-force, exact results)
        print("Building FLAT index...")
        flat_coll.create_index(
            field_name="vector",
            index_params={"index_type": "FLAT", "metric_type": metric_type, "params": {}},
        )
        flat_coll.load()
        print(f"FLAT collection '{flat_collection_name}' ready with "
              f"{flat_coll.num_entities} vectors.")
        return True

    except Exception as e:
        print(f"Error creating FLAT collection: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            connections.disconnect(conn_alias)
        except Exception:
            pass


# =============================================================================
# Ground truth pre-computation  (from simple_bench)
# =============================================================================

def precompute_ground_truth(
    host: str,
    port: str,
    flat_collection_name: str,
    query_vectors: List[List[float]],
    top_k: int,
    metric_type: str = "COSINE",
) -> Dict[int, List[int]]:
    """
    Pre-compute ground truth by running queries against the FLAT collection.

    Runs OUTSIDE the timed benchmark — zero impact on performance measurements.

    Args:
        host: Milvus host.
        port: Milvus port.
        flat_collection_name: Name of the FLAT-indexed collection.
        query_vectors: List of query vectors.
        top_k: Number of nearest neighbors to retrieve.
        metric_type: Distance metric.

    Returns:
        Dict mapping query_index -> list of ground truth nearest neighbor IDs.
    """
    conn_alias = f"gt_compute_{uuid.uuid4().hex[:8]}"
    try:
        connections.connect(alias=conn_alias, host=host, port=port)
    except Exception as e:
        print(f"Failed to connect for ground truth computation: {e}")
        return {}

    try:
        flat_coll = Collection(flat_collection_name, using=conn_alias)
        flat_coll.load()

        entity_count = flat_coll.num_entities
        effective_top_k = min(top_k, entity_count) if entity_count > 0 else top_k
        if effective_top_k != top_k:
            print(f"  NOTE: top_k capped from {top_k} to {effective_top_k} "
                  f"(collection has {entity_count} vectors)")
        effective_top_k = min(effective_top_k, 16384)  # Milvus hard limit

        ground_truth: Dict[int, List[int]] = {}
        gt_batch_size = 100  # Process queries in batches for efficiency

        print(f"Pre-computing ground truth for {len(query_vectors)} queries "
              f"using FLAT index (top_k={effective_top_k})...")

        gt_start = time.time()

        for batch_start in range(0, len(query_vectors), gt_batch_size):
            batch_end_idx = min(batch_start + gt_batch_size, len(query_vectors))
            batch_vectors = query_vectors[batch_start:batch_end_idx]

            results = flat_coll.search(
                data=batch_vectors,
                anns_field="vector",
                param={"metric_type": metric_type, "params": {}},
                limit=effective_top_k,
            )

            for i, hits in enumerate(results):
                ground_truth[batch_start + i] = [hit.id for hit in hits]

        gt_elapsed = time.time() - gt_start
        print(f"Ground truth pre-computation complete: "
              f"{len(ground_truth)} queries in {gt_elapsed:.2f}s")
        return ground_truth

    except Exception as e:
        print(f"Error computing ground truth: {e}")
        import traceback
        traceback.print_exc()
        return {}
    finally:
        try:
            connections.disconnect(conn_alias)
        except Exception:
            pass


# =============================================================================
# Ground truth computation for enhanced path (cached, from enhanced_bench)
# =============================================================================

def ids_from_hits(hits) -> List[Any]:
    return [getattr(h, "id", None) for h in hits]


def compute_ground_truth(
    gt_collection: Collection,
    queries: np.ndarray,
    vector_field: str,
    metric_type: str,
    k: int,
    *,
    cache_dir: Optional[Path] = None,
    cache_disable: bool = False,
    cache_force_refresh: bool = False,
    query_seed: Optional[int] = None,
    normalize_cosine: bool = False,
) -> List[List[Any]]:
    if cache_dir is not None and (not cache_disable) and (query_seed is not None):
        ensure_dir(cache_dir)
        sig = gt_signature(
            gt_collection_name=gt_collection.name,
            gt_num_entities=gt_collection.num_entities,
            gt_vector_field=vector_field,
            dim=int(queries.shape[1]),
            metric_type=metric_type,
            k=k,
            query_seed=query_seed,
            query_count=int(queries.shape[0]),
            normalize_cosine=normalize_cosine,
        )
        npz_path, _meta_path = gt_cache_paths(cache_dir, sig)

        if npz_path.exists() and not cache_force_refresh:
            try:
                return load_gt_cache(npz_path)
            except Exception:
                pass

    params = make_search_params_full(metric_type, {})
    results = gt_collection.search(
        data=queries.tolist(), anns_field=vector_field, param=params, limit=k)
    gt_ids = [ids_from_hits(r) for r in results]

    if cache_dir is not None and (not cache_disable) and (query_seed is not None):
        try:
            sig = gt_signature(
                gt_collection_name=gt_collection.name,
                gt_num_entities=gt_collection.num_entities,
                gt_vector_field=vector_field,
                dim=int(queries.shape[1]),
                metric_type=metric_type,
                k=k,
                query_seed=query_seed,
                query_count=int(queries.shape[0]),
                normalize_cosine=normalize_cosine,
            )
            npz_path, meta_path = gt_cache_paths(cache_dir, sig)
            save_gt_cache(npz_path, meta_path, sig, gt_ids)
        except Exception:
            pass

    return gt_ids


# =============================================================================
# Stats utilities
# =============================================================================

def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    w = idx - lo
    return s[lo] * (1 - w) + s[hi] * w


# =============================================================================
# Full statistics aggregation from per-worker CSV files  (from simple_bench)
# =============================================================================

def calculate_statistics(
    results_dir: str,
    recall_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate statistics from benchmark results stored in per-process CSV files.

    Args:
        results_dir: Directory containing milvus_benchmark_p*.csv files.
        recall_stats: Recall metrics dict from calc_recall(); always included.

    Returns:
        Dict with latency, batch, throughput, and recall statistics.
    """
    if not _HAS_PANDAS:
        return {
            "error": "pandas not installed; install with 'pip install pandas' for full statistics",
            "recall": recall_stats,
        }

    file_paths = list(Path(results_dir).glob("milvus_benchmark_p*.csv"))
    if not file_paths:
        return {"error": "No benchmark result files found", "recall": recall_stats}

    dfs = []
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Error reading result file {fp}: {e}")

    if not dfs:
        return {"error": "No valid data found in benchmark result files", "recall": recall_stats}

    all_data = pd.concat(dfs, ignore_index=True)
    all_data.sort_values('timestamp', inplace=True)

    file_start_time = min(all_data['timestamp'])
    file_end_time = max(all_data['timestamp'] + all_data['batch_time_seconds'])
    total_time_seconds = file_end_time - file_start_time

    all_latencies = []
    for _, row in all_data.iterrows():
        query_time_ms = row['avg_query_time_seconds'] * 1000
        all_latencies.extend([query_time_ms] * int(row['batch_size']))

    batch_times_ms = all_data['batch_time_seconds'] * 1000
    latencies = np.array(all_latencies)
    batch_times = np.array(batch_times_ms)
    total_queries = len(latencies)

    return {
        "total_queries": total_queries,
        "total_time_seconds": total_time_seconds,
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "p999_latency_ms": float(np.percentile(latencies, 99.9)),
        "p9999_latency_ms": float(np.percentile(latencies, 99.99)),
        "throughput_qps": (float(total_queries / total_time_seconds)
                           if total_time_seconds > 0 else 0),
        "batch_count": int(len(batch_times)),
        "min_batch_time_ms": float(np.min(batch_times)) if len(batch_times) > 0 else 0,
        "max_batch_time_ms": float(np.max(batch_times)) if len(batch_times) > 0 else 0,
        "mean_batch_time_ms": float(np.mean(batch_times)) if len(batch_times) > 0 else 0,
        "median_batch_time_ms": float(np.median(batch_times)) if len(batch_times) > 0 else 0,
        "p95_batch_time_ms": float(np.percentile(batch_times, 95)) if len(batch_times) > 0 else 0,
        "p99_batch_time_ms": float(np.percentile(batch_times, 99)) if len(batch_times) > 0 else 0,
        "p999_batch_time_ms": (float(np.percentile(batch_times, 99.9))
                               if len(batch_times) > 0 else 0),
        "p9999_batch_time_ms": (float(np.percentile(batch_times, 99.99))
                                if len(batch_times) > 0 else 0),
        "recall": recall_stats,
    }


# =============================================================================
# Collection loading + display  (from simple_bench)
# =============================================================================

def connect_to_milvus(host: str, port: str):
    """Establish connection to Milvus server."""
    try:
        connections.connect(alias="default", host=host, port=port)
        return connections
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return False


def load_database(host: str, port: str, collection_name: str,
                  reload: bool = False) -> Optional[dict]:
    """
    Verify Milvus connection, load collection, and display collection info.

    Returns:
        collection_info dict (from get_collection_info) or None on failure.
    """
    print(f'Connecting to Milvus server at {host}:{port}...', flush=True)
    conn = connect_to_milvus(host, port)
    if not conn:
        print('Unable to connect to Milvus server', flush=True)
        return None

    try:
        collection = Collection(collection_name)
    except Exception as e:
        print(f"Unable to connect to Milvus collection {collection_name}: {e}", flush=True)
        return None

    try:
        from pymilvus import utility as _util
        state = _util.load_state(collection_name)
        if reload or state.name != "Loaded":
            label = "Reloading" if reload else "Loading"
            print(f'{label} the collection {collection_name}...')
            t0 = time.time()
            collection.load()
            print(f'Collection {collection_name} loaded in {time.time() - t0:.2f} seconds',
                  flush=True)
        else:
            print(f'Collection {collection_name} already loaded.')
    except Exception as e:
        print(f'Unable to load collection {collection_name}: {e}')
        return None

    # Display collection stats
    if _VDBBENCH_PKG:
        try:
            collection_info = get_collection_info(collection_name, release=False)
            index_types = ", ".join(
                [idx.get("index_type", "N/A") for idx in collection_info.get("index_info", [])])
            metric_types = ", ".join(
                [idx.get("metric_type", "N/A") for idx in collection_info.get("index_info", [])])
            table_data = [[
                collection_info["name"],
                collection_info.get("row_count", "N/A"),
                collection_info.get("dimension", "N/A"),
                index_types,
                metric_types,
                len(collection_info.get("partitions", [])),
            ]]
            headers = ["Collection Name", "Vector Count", "Dimension",
                       "Index Types", "Metric Types", "Partitions"]
            if _HAS_TABULATE:
                print(f'\n{_tabulate(table_data, headers=headers, tablefmt="grid")}', flush=True)
            else:
                print(f'\nCollection info: {dict(zip(headers, table_data[0]))}', flush=True)
            return collection_info
        except Exception as e:
            print(f"Could not retrieve collection info via vdbbench: {e}")

    # Fallback: build minimal collection_info without vdbbench package
    try:
        col = Collection(collection_name)
        idx_type, metric_type, _ = get_index_params(col)
        _, dim, _, _ = get_vector_field_info(col)
        collection_info = {
            "name": collection_name,
            "row_count": col.num_entities,
            "dimension": dim,
            "index_info": [{"index_type": idx_type, "metric_type": metric_type}],
            "partitions": col.partitions,
        }
        print(f"\nCollection: {collection_name}  vectors={col.num_entities}  "
              f"dim={dim}  index={idx_type}  metric={metric_type}", flush=True)
        return collection_info
    except Exception as e:
        print(f"Could not retrieve fallback collection info: {e}")
        return None


# =============================================================================
# Memory estimator  (from enhanced_bench)
# =============================================================================

def estimate_memory_bytes(index_type: str, n: int, dim: int,
                          *, hnsw_m: int = 16) -> Dict[str, Any]:
    t = (index_type or "FLAT").lower()
    vector_bytes = int(n) * int(dim) * 4
    notes = []
    index_bytes = 0

    if t == "flat":
        notes.append("FLAT: exact search; memory dominated by vectors + Milvus overhead.")
    elif t == "hnsw":
        per_node_graph = hnsw_m * 8
        base_graph = int(n) * per_node_graph
        index_bytes = int(base_graph * 2.0)
        notes.append(f"HNSW: assumes M={hnsw_m}, ~{per_node_graph}B/node, meta_factor=2.0.")
    elif t == "diskann":
        index_bytes = int(n * 64)
        notes.append("DiskANN: RSS can be low; performance depends on host page cache + SSD I/O.")
    elif t == "aisaq":
        index_bytes = int(n * 64)
        notes.append("AISAQ: similar caution to DiskANN; estimate is coarse.")
    else:
        index_bytes = int(n * 64)
        notes.append(f"Unknown index_type '{index_type}': using coarse index_bytes ~ n*64B.")

    total = vector_bytes + index_bytes
    return {
        "index_type": index_type,
        "n": int(n),
        "dim": int(dim),
        "vector_bytes_est": vector_bytes,
        "index_bytes_est": index_bytes,
        "total_bytes_est": total,
        "total_gb_est": bytes_to_gb(total),
        "notes": notes,
    }


# =============================================================================
# RunResult dataclass  (from enhanced_bench)
# =============================================================================

@dataclass
class RunResult:
    mode: str
    index_type: str
    metric_type: str
    algo_params: Dict[str, Any]
    k: int
    queries: int
    qps: float
    lat_ms_avg: float
    lat_ms_p50: float
    lat_ms_p95: float
    lat_ms_p99: float

    recall: Optional[float] = None          # mean recall@k (scalar, for CSV/backward compat)
    recall_stats: Optional[Dict[str, Any]] = field(default=None)  # full recall dict
    is_max_throughput: bool = False

    disk_read_bytes: Optional[int] = None
    disk_write_bytes: Optional[int] = None
    read_bytes_per_query: Optional[float] = None
    disk_read_iops: Optional[float] = None
    disk_write_iops: Optional[float] = None
    disk_read_mbps: Optional[float] = None
    disk_write_mbps: Optional[float] = None
    disk_duration_sec: Optional[float] = None

    rss_bytes: Optional[int] = None
    cache_state: Optional[str] = None

    host_mem_avail_before: Optional[int] = None
    host_mem_avail_after: Optional[int] = None
    host_mem_cached_before: Optional[int] = None
    host_mem_cached_after: Optional[int] = None

    budget_rss_ok: Optional[bool] = None
    budget_host_ok: Optional[bool] = None
    budget_reason: Optional[str] = None

    quality_score: Optional[float] = None
    cost_score: Optional[float] = None


# =============================================================================
# Shared helpers — disk totals, recall conversion, unified summary print
# Used by BOTH Path A and Path B to produce identical statistics output.
# =============================================================================

def _disk_totals(
    diff: Dict[str, Dict[str, int]],
    disk_devices: Optional[List[str]],
    elapsed_sec: float,
) -> Dict[str, Any]:
    """
    Aggregate disk diff into totals + derived rates (MB/s, IOPS).

    Args:
        diff: Output of disk_stats_diff() — {device: {bytes_read, bytes_written,
              read_ios, write_ios, ...}}.
        disk_devices: If set, only sum these device names; else sum all real devices.
        elapsed_sec: Wall-clock seconds over which the diff was measured.

    Returns:
        Dict with keys: bytes_read, bytes_written, read_ios, write_ios,
        read_mbps, write_mbps, read_iops, write_iops, read_bpq (requires ok_count),
        duration_sec, available (bool).
    """
    if not diff:
        return {"available": False, "bytes_read": 0, "bytes_written": 0,
                "read_ios": 0, "write_ios": 0,
                "read_mbps": 0.0, "write_mbps": 0.0,
                "read_iops": 0.0, "write_iops": 0.0, "duration_sec": elapsed_sec}

    if disk_devices:
        devs = {d: diff[d] for d in disk_devices if d in diff}
    else:
        devs = filter_real_disk_devices(diff)

    rd = wr = rio = wio = 0
    for s in devs.values():
        rd  += s.get("bytes_read",  0)
        wr  += s.get("bytes_written", 0)
        rio += s.get("read_ios",  0)
        wio += s.get("write_ios", 0)

    t = max(elapsed_sec, 1e-6)
    return {
        "available":      True,
        "bytes_read":     rd,
        "bytes_written":  wr,
        "read_ios":       rio,
        "write_ios":      wio,
        "read_mbps":      rd  / t / (1024 * 1024),
        "write_mbps":     wr  / t / (1024 * 1024),
        "read_iops":      rio / t,
        "write_iops":     wio / t,
        "duration_sec":   elapsed_sec,
    }


def _recall_from_lists(
    gt_list:   List[List[Any]],
    pred_list: List[List[Any]],
    k: int,
) -> Optional[Dict[str, Any]]:
    """
    Compute full recall stats (mean/median/p95/p99) from ordered lists.

    Converts list-indexed inputs to the dict format expected by calc_recall(),
    which is more robust than list-zip alignment (no silent truncation).
    Returns None if either input is empty.
    """
    if not gt_list or not pred_list:
        return None
    n = min(len(gt_list), len(pred_list))
    if n == 0:
        return None
    gt_dict   = {i: gt_list[i]   for i in range(n)}
    pred_dict = {i: pred_list[i] for i in range(n)}
    return calc_recall(pred_dict, gt_dict, k)


def print_bench_summary(
    r: RunResult,
    label: str = "",
    total_queries: Optional[int] = None,
    total_batches: Optional[int] = None,
) -> None:
    """
    Print a unified benchmark summary block identical in structure to Path A.

    Works for both Path B single-run results and Path A aggregate results when
    caller maps aggregate stats into a synthetic RunResult.  The format is:
        BENCHMARK SUMMARY
        QUERY STATISTICS   (latency + QPS)
        RECALL STATISTICS  (full dict if recall_stats populated, else scalar)
        DISK I/O           (MB/s + IOPS if available)
    """
    width = 60
    hdr = f"BENCHMARK SUMMARY{(' — ' + label) if label else ''}"
    print("\n" + "=" * width)
    print(hdr)
    print("=" * width)
    print(f"Index:         {r.index_type}  |  Metric: {r.metric_type}")
    print(f"Params:        {r.algo_params}")
    if r.cache_state:
        print(f"Cache:         {r.cache_state}")
    if total_queries is not None:
        print(f"Total Queries: {total_queries}")
    else:
        print(f"Total Queries: {r.queries}")
    if total_batches is not None:
        print(f"Total Batches: {total_batches}")

    print("\nQUERY STATISTICS")
    print("-" * width)
    print(f"Mean Latency:      {r.lat_ms_avg:.2f} ms")
    print(f"Median Latency:    {r.lat_ms_p50:.2f} ms")
    print(f"P95 Latency:       {r.lat_ms_p95:.2f} ms")
    print(f"P99 Latency:       {r.lat_ms_p99:.2f} ms")
    print(f"Throughput:        {r.qps:.2f} queries/second")

    # Recall — prefer full stats dict, fall back to scalar
    rs = r.recall_stats
    if rs:
        print(f"\nRECALL STATISTICS (recall@{rs.get('k', r.k)})")
        print("-" * width)
        print(f"Mean Recall:       {rs.get('mean_recall',   0):.4f}")
        print(f"Median Recall:     {rs.get('median_recall', 0):.4f}")
        print(f"Min Recall:        {rs.get('min_recall',    0):.4f}")
        print(f"Max Recall:        {rs.get('max_recall',    0):.4f}")
        print(f"P95 Recall:        {rs.get('p95_recall',    0):.4f}")
        print(f"P99 Recall:        {rs.get('p99_recall',    0):.4f}")
        print(f"Queries Evaluated: {rs.get('num_queries_evaluated', 0)}")
    elif r.recall is not None:
        print(f"\nRECALL STATISTICS (recall@{r.k})")
        print("-" * width)
        print(f"Mean Recall:       {r.recall:.4f}  (scalar; no per-query distribution)")

    # Disk I/O
    print("\nDISK I/O DURING BENCHMARK")
    print("-" * width)
    if r.disk_read_bytes is not None:
        rmb  = r.disk_read_mbps  if r.disk_read_mbps  is not None else 0.0
        wmb  = r.disk_write_mbps if r.disk_write_mbps is not None else 0.0
        riops = r.disk_read_iops  if r.disk_read_iops  is not None else 0.0
        wiops = r.disk_write_iops if r.disk_write_iops is not None else 0.0
        print(f"Total Read:        {format_bytes(r.disk_read_bytes)}"
              f"  ({rmb:.2f} MB/s,  {riops:.0f} IOPS)")
        print(f"Total Write:       {format_bytes(r.disk_write_bytes or 0)}"
              f"  ({wmb:.2f} MB/s,  {wiops:.0f} IOPS)")
        if r.read_bytes_per_query is not None:
            print(f"Read / Query:      {format_bytes(int(r.read_bytes_per_query))}")
    else:
        print("Disk I/O statistics not available")

    if r.rss_bytes is not None:
        print(f"\nRSS:               {format_bytes(r.rss_bytes)}")

    print("=" * width)


# =============================================================================
# bench_single  (from enhanced_bench)
# =============================================================================

def bench_single(
    collection: Collection,
    queries: np.ndarray,
    vector_field: str,
    metric_type: str,
    algo_params: Dict[str, Any],
    k: int,
    gt_ids: Optional[List[List[Any]]] = None,
    disk_devices: Optional[List[str]] = None,
    rss_bytes: Optional[int] = None,
    cache_state: Optional[str] = None,
    host_before: Optional[HostMemSnapshot] = None,
    host_after: Optional[HostMemSnapshot] = None,
) -> RunResult:
    params = make_search_params_full(metric_type, algo_params)
    lat_ms: List[float] = []
    pred_ids: List[List[Any]] = []

    disk_start = read_disk_stats()
    t0 = time.time()
    ok = 0
    failed = 0

    for qv in queries:
        qs = time.time()
        try:
            hits = collection.search([qv.tolist()], vector_field, params, limit=k)[0]
            pred_ids.append(ids_from_hits(hits))
            ok += 1
        except Exception:
            pred_ids.append([])
            failed += 1
        lat_ms.append((time.time() - qs) * 1000.0)

    if failed > 0:
        print(f"⚠️  {failed}/{len(queries)} queries failed in single-thread mode")

    total = time.time() - t0
    disk_end = read_disk_stats()

    qps = ok / total if total > 0 else 0.0

    # Full recall stats (mean/median/p95/p99) via shared helper
    recall_stats = _recall_from_lists(gt_ids, pred_ids, k) if gt_ids is not None else None
    mean_recall  = recall_stats["mean_recall"] if recall_stats else None

    # Disk totals + rates via shared helper
    diff = disk_stats_diff(disk_start, disk_end)
    dt   = _disk_totals(diff, disk_devices, total)
    rd, wr = dt["bytes_read"], dt["bytes_written"]
    read_bpq = (rd / max(1, ok)) if dt["available"] else None
    rss_gb = (rss_bytes / (1024 ** 3)) if rss_bytes else None

    return RunResult(
        mode="single",
        index_type=get_index_params(collection)[0],
        metric_type=metric_type,
        algo_params=algo_params,
        k=k,
        queries=len(queries),
        qps=qps,
        lat_ms_avg=float(np.mean(lat_ms)) if lat_ms else float("nan"),
        lat_ms_p50=percentile(lat_ms, 50),
        lat_ms_p95=percentile(lat_ms, 95),
        lat_ms_p99=percentile(lat_ms, 99),
        recall=mean_recall,
        recall_stats=recall_stats,
        disk_read_bytes=rd  if dt["available"] else None,
        disk_write_bytes=wr if dt["available"] else None,
        read_bytes_per_query=read_bpq,
        disk_read_iops=dt["read_iops"]   if dt["available"] else None,
        disk_write_iops=dt["write_iops"] if dt["available"] else None,
        disk_read_mbps=dt["read_mbps"]   if dt["available"] else None,
        disk_write_mbps=dt["write_mbps"] if dt["available"] else None,
        disk_duration_sec=total           if dt["available"] else None,
        rss_bytes=rss_bytes,
        cache_state=cache_state,
        host_mem_avail_before=(host_before.mem_available_bytes if host_before else None),
        host_mem_avail_after=(host_after.mem_available_bytes  if host_after  else None),
        host_mem_cached_before=(host_before.cached_bytes if host_before else None),
        host_mem_cached_after=(host_after.cached_bytes   if host_after  else None),
        quality_score=qps,
        cost_score=(qps / rss_gb) if (rss_gb and rss_gb > 0) else None,
    )


# =============================================================================
# Multi-process worker — enhanced_bench path (chunk-based, returns results)
# =============================================================================

def _worker_mp(
    worker_id: int,
    host: str,
    port: str,
    collection_name: str,
    vector_field: str,
    metric_type: str,
    algo_params: Dict[str, Any],
    k: int,
    q_chunk: np.ndarray,
    out_q: mp.Queue,
) -> None:
    try:
        connections.connect(alias=f"w{worker_id}", host=host, port=port)
        col = Collection(collection_name, using=f"w{worker_id}")
        col.load()
        params = make_search_params_full(metric_type, algo_params)

        lat_ms: List[float] = []
        pred_ids: List[List[Any]] = []
        ok = 0
        for qv in q_chunk:
            t0 = time.time()
            try:
                hits = col.search([qv.tolist()], vector_field, params, limit=k)[0]
                pred_ids.append(ids_from_hits(hits))
                ok += 1
            except Exception:
                pred_ids.append([])
            lat_ms.append((time.time() - t0) * 1000.0)

        out_q.put({"worker_id": worker_id, "ok": ok, "lat_ms": lat_ms, "pred_ids": pred_ids})
    except Exception as e:
        out_q.put({"worker_id": worker_id, "ok": 0, "lat_ms": [], "pred_ids": [], "error": str(e)})


def bench_multiprocess(
    host: str,
    port: str,
    collection_name: str,
    vector_field: str,
    metric_type: str,
    algo_params: Dict[str, Any],
    k: int,
    queries: np.ndarray,
    processes: int,
    disk_devices: Optional[List[str]] = None,
    gt_ids: Optional[List[List[Any]]] = None,
) -> Dict[str, Any]:
    """
    Run a multi-process benchmark chunk and return a unified result dict.

    Returns dict with keys: qps, all_lat, ok_total, rd, wr, read_bpq,
    recall (mean float), recall_stats (full dict), disk (from _disk_totals).
    """
    chunks = np.array_split(queries, processes)
    out_q: mp.Queue = mp.Queue()

    disk_start = read_disk_stats()
    t0 = time.time()

    procs = []
    for i, chunk in enumerate(chunks):
        p = mp.Process(
            target=_worker_mp,
            args=(i, host, port, collection_name, vector_field, metric_type,
                  algo_params, k, chunk, out_q),
        )
        p.start()
        procs.append(p)

    results = [out_q.get() for _ in range(processes)]
    for p in procs:
        p.join()

    total = time.time() - t0
    disk_end = read_disk_stats()

    results.sort(key=lambda r: r.get("worker_id", 0))

    all_lat: List[float] = []
    all_pred_ids: List[List[Any]] = []
    ok_total = 0
    failed_total = 0
    for res in results:
        ok_total += int(res.get("ok", 0))
        all_lat.extend(res.get("lat_ms", []))
        chunk_preds = res.get("pred_ids", [])
        all_pred_ids.extend(chunk_preds)
        failed_total += len(chunk_preds) - int(res.get("ok", 0))

    if failed_total > 0:
        print(f"⚠️  {failed_total}/{len(queries)} queries failed in multi-process mode")

    qps = ok_total / total if total > 0 else 0.0

    # Full recall stats via shared helper (handles length mismatches via dict keys)
    recall_stats = _recall_from_lists(gt_ids, all_pred_ids, k) if gt_ids is not None else None
    mean_recall  = recall_stats["mean_recall"] if recall_stats else None

    # Disk totals + rates via shared helper
    diff = disk_stats_diff(disk_start, disk_end)
    dt   = _disk_totals(diff, disk_devices, total)
    rd, wr = dt["bytes_read"], dt["bytes_written"]
    read_bpq = (rd / max(1, ok_total)) if dt["available"] else None

    return {
        "qps":          qps,
        "all_lat":      all_lat,
        "ok_total":     ok_total,
        "rd":           rd,
        "wr":           wr,
        "read_bpq":     read_bpq,
        "recall":       mean_recall,
        "recall_stats": recall_stats,
        "disk":         dt,
        "total_sec":    total,
    }


# =============================================================================
# execute_batch_queries  (from simple_bench)
# Timed / query-count controlled worker with per-process CSV output.
# Captures ANN result IDs into a shared dict for post-hoc recall.
# =============================================================================

def load_recall_hits(output_dir: str) -> Dict[int, List[int]]:
    """
    Merge per-worker recall-hits JSONL files into a single dict.

    Each file contains one JSON object per line: {"q": <query_idx>, "ids": [...]}
    Only the first record for each query_idx is kept (deduplication across workers).

    Returns:
        Dict mapping query_idx -> list of ANN result IDs.
    """
    ann_results: Dict[int, List[int]] = {}
    pattern = Path(output_dir) / "recall_hits_p*.jsonl"
    import glob
    for fpath in sorted(glob.glob(str(pattern))):
        try:
            with open(fpath, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        q_idx = int(rec["q"])
                        if q_idx not in ann_results:
                            ann_results[q_idx] = [int(x) for x in rec["ids"]]
                    except (KeyError, ValueError, json.JSONDecodeError):
                        continue
        except OSError:
            pass
    return ann_results


def execute_batch_queries(
    process_id: int,
    host: str,
    port: str,
    collection_name: str,
    vector_dim: int,
    batch_size: int,
    report_count: int,
    max_queries: Optional[int],
    runtime_seconds: Optional[int],
    output_dir: str,
    shutdown_flag: mp.Value,
    pre_generated_queries: List[List[float]] = None,
    ann_results_dict: dict = None,   # kept for API compat, no longer used
    search_limit: int = 10,
    search_ef: int = 200,
    anns_field: str = "vector",
    metric_type: str = "COSINE",
    index_type: str = "HNSW",
) -> None:
    """
    Execute batches of vector queries and log results to per-process CSV files.

    ANN result IDs are written to a per-worker ``recall_hits_p<N>.jsonl`` file
    (one JSON line per first-seen query index) instead of a shared Manager dict.
    This avoids the IPC race conditions that caused recall=0 with Manager dict
    under multiprocessing fork.

    CRITICAL TIMING NOTE:
      batch_end is recorded IMMEDIATELY after collection.search() returns.
      All recall capture (writing hit IDs) happens AFTER batch_end — zero
      impact on latency / throughput numbers.

    Args:
        process_id: Worker process ID.
        host / port: Milvus connection details.
        collection_name: Target collection.
        vector_dim: Vector dimension (unused when queries pre-generated).
        batch_size: Queries per batch.
        report_count: Batches between stdout progress reports.
        max_queries: Query count limit (None = no limit).
        runtime_seconds: Time limit in seconds (None = no limit).
        output_dir: Directory for per-process output files.
        shutdown_flag: Shared mp.Value for graceful shutdown.
        pre_generated_queries: Deterministic query vectors (list of lists).
        ann_results_dict: Deprecated — no longer used; kept for API compatibility.
        search_limit: Top-k results per query.
        search_ef: ef/search_list override for HNSW/DiskANN/AISAQ.
        anns_field: Vector field name in the collection.
        metric_type: Distance metric.
        index_type: Index type string (determines search param key name).
    """
    print(f'Process {process_id} initialized')

    # Build search params based on index type
    idx_t = (index_type or "HNSW").upper()
    if idx_t == "HNSW":
        search_params = {"metric_type": metric_type, "params": {"ef": search_ef}}
    elif idx_t in ("DISKANN", "AISAQ"):
        search_params = {"metric_type": metric_type, "params": {"search_list": search_ef}}
    elif idx_t.startswith("IVF"):
        search_params = {"metric_type": metric_type, "params": {"nprobe": search_ef}}
    else:
        search_params = {"metric_type": metric_type, "params": {}}

    conn = connect_to_milvus(host, port)
    if not conn:
        print(f'Process {process_id} - No Milvus connection')
        return

    try:
        collection = Collection(collection_name)
        print(f'Process {process_id} - Loading collection')
        collection.load()
    except Exception as e:
        print(f"Process {process_id}: Failed to load collection: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    csv_file  = Path(output_dir) / f"milvus_benchmark_p{process_id}.csv"
    hits_file = Path(output_dir) / f"recall_hits_p{process_id}.jsonl"
    sys.stdout.write(f"Process {process_id}: Writing results to {csv_file}\r\n")

    num_pre_generated = len(pre_generated_queries) if pre_generated_queries else 0
    if num_pre_generated == 0:
        print(f"Process {process_id}: ERROR — no pre-generated query vectors provided.")
        return

    start_time = time.time()
    query_count = 0
    batch_count = 0
    seen_query_indices: set = set()   # local dedup; no IPC needed

    sys.stdout.write(f"Process {process_id}: Starting benchmark ...\r\n")
    sys.stdout.flush()

    try:
        with open(csv_file, 'w') as f_csv, open(hits_file, 'w') as f_hits:
            writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
            writer.writeheader()

            while True:
                with shutdown_flag.get_lock():
                    if shutdown_flag.value == 1:
                        break

                current_time = time.time()
                elapsed_time = current_time - start_time

                if runtime_seconds is not None and elapsed_time >= runtime_seconds:
                    break
                if max_queries is not None and query_count >= max_queries:
                    break

                # Build batch from pre-generated queries (deterministic cycling)
                batch_vectors = []
                batch_query_indices = []
                for b in range(batch_size):
                    idx = (query_count + b) % num_pre_generated
                    batch_vectors.append(pre_generated_queries[idx])
                    batch_query_indices.append(idx)

                # ---- TIMED SECTION: Only the primary ANN search ----
                batch_start = time.time()
                try:
                    results = collection.search(
                        data=batch_vectors,
                        anns_field=anns_field,
                        param=search_params,
                        limit=search_limit,
                    )
                    # CRITICAL: batch_end recorded HERE, before any recall work.
                    batch_end = time.time()
                    batch_success = True
                except Exception as e:
                    print(f"Process {process_id}: Search error: {e}")
                    batch_end = time.time()
                    batch_success = False
                    results = None
                # ---- END TIMED SECTION ----

                # Capture ANN result IDs into per-worker JSONL (NOT timed).
                # Using a local file per worker avoids all Manager dict IPC issues.
                if results is not None:
                    for i, hits in enumerate(results):
                        q_idx = batch_query_indices[i]
                        if q_idx not in seen_query_indices:
                            seen_query_indices.add(q_idx)
                            result_ids = [hit.id for hit in hits]
                            f_hits.write(
                                json.dumps({"q": q_idx, "ids": result_ids}) + "\n"
                            )

                batch_time = batch_end - batch_start
                batch_count += 1
                query_count += batch_size

                writer.writerow({
                    "process_id": process_id,
                    "batch_id": batch_count,
                    "timestamp": current_time,
                    "batch_size": batch_size,
                    "batch_time_seconds": batch_time,
                    "avg_query_time_seconds": batch_time / batch_size,
                    "success": batch_success,
                })
                f_csv.flush()

                if batch_count % report_count == 0:
                    sys.stdout.write(
                        f"Process {process_id}: Completed {query_count} queries "
                        f"in {elapsed_time:.2f} seconds.\r\n")
                    sys.stdout.flush()

    except Exception as e:
        print(f"Process {process_id}: Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            connections.disconnect("default")
        except Exception:
            pass
        print(f"Process {process_id}: Finished. Executed {query_count} queries "
              f"in {time.time() - start_time:.2f} seconds", flush=True)


# =============================================================================
# Sweep logic  (from enhanced_bench)
# =============================================================================

def sweep_candidates(index_type: str, build_params: Optional[Dict[str, Any]] = None,
                     include_minimal: bool = True) -> List[Dict[str, Any]]:
    t = (index_type or "FLAT").lower()
    cands: List[Dict[str, Any]] = []
    build_params = build_params or {}

    if t == "hnsw":
        base_values = [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
        if include_minimal:
            base_values = [10] + base_values
        return [{"ef": ef} for ef in base_values]

    if t == "diskann":
        search_list_size = build_params.get("search_list_size", 5000)
        max_sl = min(4000, search_list_size)
        base_values = [10, 20, 50, 100, 200, 400, 800, 1200, 1600, 2000, 2500, 3000, 4000]
        if max_sl < 4000:
            print(f"⚠️  DiskANN build param search_list_size={search_list_size} limits sweep to {max_sl}")
        return [{"search_list": sl} for sl in base_values if sl <= max_sl]

    if t == "aisaq":
        search_list_size = build_params.get("search_list_size", 5000)
        max_sl = min(3000, search_list_size)
        base_values = [10, 20, 50, 100, 200, 400, 800, 1200, 1600, 2000, 2500, 3000]
        if max_sl < 3000:
            print(f"⚠️  AISAQ build param search_list_size={search_list_size} limits sweep to {max_sl}")
            print(f"   Rebuild index with higher search_list_size for better recall potential")
        return [{"search_list": sl} for sl in base_values if sl <= max_sl]

    if t.startswith("ivf"):
        return [{"nprobe": n} for n in [1, 2, 4, 8, 16, 32, 64, 128]]

    return [{}]


def pick_best_by_target_recall(
    collection: Collection,
    gt_collection: Collection,
    queries: np.ndarray,
    vector_field: str,
    metric_type: str,
    k: int,
    index_type: str,
    target_recall: float,
    optimize: str = "quality",
    rss_bytes: Optional[int] = None,
    cache_state: Optional[str] = None,
    build_params: Optional[Dict[str, Any]] = None,
    *,
    gt_cache_dir: Optional[Path] = None,
    gt_cache_disable: bool = False,
    gt_cache_force_refresh: bool = False,
    gt_query_seed: Optional[int] = None,
    normalize_cosine: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

    gt_ids = compute_ground_truth(
        gt_collection, queries, vector_field, metric_type, k,
        cache_dir=gt_cache_dir, cache_disable=gt_cache_disable,
        cache_force_refresh=gt_cache_force_refresh,
        query_seed=gt_query_seed, normalize_cosine=normalize_cosine,
    )

    best: Optional[RunResult] = None
    report: List[Dict[str, Any]] = []

    for algo in sweep_candidates(index_type, build_params):
        host_before = HostMemSnapshot.from_proc_meminfo()
        r = bench_single(
            collection=collection, queries=queries, vector_field=vector_field,
            metric_type=metric_type, algo_params=algo, k=k, gt_ids=gt_ids,
            rss_bytes=rss_bytes, cache_state=cache_state,
            host_before=host_before, host_after=HostMemSnapshot.from_proc_meminfo(),
        )

        rss_gb = (r.rss_bytes / (1024 ** 3)) if r.rss_bytes else None
        qps_per_gb = (r.qps / rss_gb) if (rss_gb and rss_gb > 0) else None

        report.append({
            "algo_params": algo, "recall": r.recall, "qps": r.qps,
            "lat_ms_p95": r.lat_ms_p95, "lat_ms_avg": r.lat_ms_avg,
            "rss_bytes": r.rss_bytes, "qps_per_gb": qps_per_gb,
            "read_bytes_per_query": r.read_bytes_per_query,
            "cache_state": cache_state,
            "host_mem_avail_before": r.host_mem_avail_before,
            "host_mem_avail_after": r.host_mem_avail_after,
        })

        if r.recall is None or r.recall < target_recall:
            continue

        if best is None:
            best = r
            continue

        if optimize == "quality":
            if r.qps > best.qps or (
                abs(r.qps - best.qps) / (best.qps + 1e-9) < 1e-6
                    and r.lat_ms_p95 < best.lat_ms_p95):
                best = r
        elif optimize == "latency":
            if r.lat_ms_p95 < best.lat_ms_p95 or (
                abs(r.lat_ms_p95 - best.lat_ms_p95) / (best.lat_ms_p95 + 1e-9) < 1e-6
                    and r.qps > best.qps):
                best = r
        elif optimize == "cost":
            def cost_score(rr: RunResult) -> float:
                if rr.rss_bytes and rr.rss_bytes > 0:
                    return rr.qps / (rr.rss_bytes / (1024 ** 3))
                return -1.0
            if cost_score(r) > cost_score(best):
                best = r
        else:
            if r.qps > best.qps:
                best = r

    if best is None:
        best_row = None
        for row in report:
            if best_row is None:
                best_row = row
                continue
            if row["recall"] is None:
                continue
            if (best_row["recall"] is None or row["recall"] > best_row["recall"] or
                    (row["recall"] == best_row["recall"] and row["qps"] > best_row["qps"])):
                best_row = row
        if best_row and best_row.get("recall") is not None:
            best_recall = best_row["recall"]
            if best_recall < target_recall:
                print(f"⚠️  WARNING: Could not achieve target recall {target_recall:.3f}. "
                      f"Best found: {best_recall:.4f} with params {best_row['algo_params']}")
                print(f"   Consider increasing sweep range or adjusting index build parameters.")
        return (best_row["algo_params"] if best_row else {}), report

    return best.algo_params, report


# =============================================================================
# Output writers  (from enhanced_bench)
# =============================================================================

def write_outputs(out_dir: Path, base: str, runs: List[RunResult],
                  sweep_report: Optional[List[Dict[str, Any]]] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {"runs": [asdict(r) for r in runs], "sweep": sweep_report}
    (out_dir / f"{base}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    csv_path = out_dir / f"{base}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "mode", "index_type", "metric_type", "algo_params",
            "k", "queries", "qps", "lat_ms_avg", "lat_ms_p50",
            "lat_ms_p95", "lat_ms_p99",
            "recall_mean", "recall_median", "recall_p95", "recall_p99",
            "recall_min", "recall_max", "recall_queries_evaluated",
            "disk_read_bytes", "disk_write_bytes", "read_bytes_per_query",
            "disk_read_mbps", "disk_write_mbps",
            "disk_read_iops", "disk_write_iops", "disk_duration_sec",
            "rss_bytes", "cache_state",
            "host_mem_avail_before", "host_mem_avail_after",
            "host_mem_cached_before", "host_mem_cached_after",
            "budget_rss_ok", "budget_host_ok", "budget_reason",
            "quality_score", "cost_score", "is_max_throughput",
        ])
        for r in runs:
            rs = r.recall_stats or {}
            w.writerow([
                r.mode, r.index_type, r.metric_type, json.dumps(r.algo_params),
                r.k, r.queries, r.qps, r.lat_ms_avg, r.lat_ms_p50,
                r.lat_ms_p95, r.lat_ms_p99,
                rs.get("mean_recall",   r.recall),
                rs.get("median_recall"),
                rs.get("p95_recall"),
                rs.get("p99_recall"),
                rs.get("min_recall"),
                rs.get("max_recall"),
                rs.get("num_queries_evaluated"),
                r.disk_read_bytes, r.disk_write_bytes, r.read_bytes_per_query,
                r.disk_read_mbps,  r.disk_write_mbps,
                r.disk_read_iops,  r.disk_write_iops,  r.disk_duration_sec,
                r.rss_bytes, r.cache_state,
                r.host_mem_avail_before, r.host_mem_avail_after,
                r.host_mem_cached_before, r.host_mem_cached_after,
                r.budget_rss_ok, r.budget_host_ok, r.budget_reason,
                r.quality_score, r.cost_score, r.is_max_throughput,
            ])

    if sweep_report is not None:
        swp = out_dir / f"{base}.sweep.csv"
        with swp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "index_type", "recall_target", "optimize", "algo_params",
                "recall", "qps", "lat_ms_p95", "lat_ms_avg", "rss_bytes",
                "qps_per_gb", "read_bytes_per_query", "cache_state",
                "host_mem_avail_before", "host_mem_avail_after",
            ])
            for row in sweep_report:
                w.writerow([
                    row.get("index_type"), row.get("recall_target"),
                    row.get("optimize"), json.dumps(row.get("algo_params")),
                    row.get("recall"), row.get("qps"), row.get("lat_ms_p95"),
                    row.get("lat_ms_avg"), row.get("rss_bytes"),
                    row.get("qps_per_gb"), row.get("read_bytes_per_query"),
                    row.get("cache_state"),
                    row.get("host_mem_avail_before"), row.get("host_mem_avail_after"),
                ])


# =============================================================================
# Budget enforcement  (from enhanced_bench)
# =============================================================================

def check_budgets(
    *,
    rss_bytes: Optional[int],
    host_before: HostMemSnapshot,
    mem_budget_gb: Optional[float],
    host_mem_reserve_gb: Optional[float],
) -> Tuple[bool, bool, str]:
    rss_ok = True
    host_ok = True
    reasons = []

    if mem_budget_gb is not None:
        if rss_bytes is None:
            rss_ok = False
            reasons.append("mem_budget_gb set but rss_bytes unavailable (provide --milvus-container).")
        elif bytes_to_gb(rss_bytes) > mem_budget_gb:
            rss_ok = False
            reasons.append(f"RSS {bytes_to_gb(rss_bytes):.2f}GB > budget {mem_budget_gb:.2f}GB")

    if host_mem_reserve_gb is not None:
        if bytes_to_gb(host_before.mem_available_bytes) < host_mem_reserve_gb:
            host_ok = False
            reasons.append(
                f"Host MemAvailable {bytes_to_gb(host_before.mem_available_bytes):.2f}GB "
                f"< reserve {host_mem_reserve_gb:.2f}GB")

    return rss_ok, host_ok, "; ".join(reasons) if reasons else ""


# =============================================================================
# Main entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Enhanced Milvus VDB Benchmark\n"
            "Supports two execution paths:\n"
            "  A) Runtime/query-count mode (--runtime or --queries + --batch-size)\n"
            "  B) Sweep/cache mode         (--mode + optionally --sweep)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # YAML
    ap.add_argument("--config", default=None,
                    help="YAML config file. CLI flags override YAML.")

    # Estimator-only mode
    ap.add_argument("--estimate-only", action="store_true",
                    help="Only estimate memory footprint and exit "
                         "(requires --est-index-type --est-n --est-dim).")
    ap.add_argument("--est-index-type", default=None,
                    help="Estimator: index type (HNSW/DISKANN/AISAQ/FLAT)")
    ap.add_argument("--est-n", type=int, default=None, help="Estimator: vector count")
    ap.add_argument("--est-dim", type=int, default=None, help="Estimator: dimension")
    ap.add_argument("--est-hnsw-m", type=int, default=16, help="Estimator: HNSW M (if known)")

    # Connectivity
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", default="19530")

    # Collections — support both naming conventions
    ap.add_argument("--collection", "--collection-name", dest="collection",
                    help="Collection under test (ANN-indexed)")
    ap.add_argument("--gt-collection", default=None,
                    help="Ground-truth collection name. If not given and --auto-create-flat is "
                         "set, it defaults to <collection>_flat_gt. "
                         "For enhanced_bench path: recommended FLAT index. "
                         "For simple_bench path: auto-created from source if needed.")

    # Ground truth / recall
    ap.add_argument("--auto-create-flat", action="store_true",
                    help="Auto-create FLAT GT collection from source collection "
                         "(simple_bench path). Copies all vectors + PKs, builds FLAT index.")
    ap.add_argument("--num-query-vectors", type=int, default=1000,
                    help="Number of pre-generated query vectors for recall (default: 1000).")
    ap.add_argument("--recall-k", type=int, default=None,
                    help="K for recall@k (default: same as --search-limit or --k).")
    ap.add_argument("--vector-dim", type=int, default=1536,
                    help="Vector dimension (default: 1536). "
                         "Auto-detected from collection schema when possible.")

    # Search parameters — explicit overrides
    ap.add_argument("--search-limit", type=int, default=10,
                    help="Top-k results per query (default: 10).")
    ap.add_argument("--search-ef", type=int, default=200,
                    help="HNSW ef / DiskANN search_list / AISAQ search_list / IVF nprobe override "
                         "(used in runtime/query-count mode). Default: 200.")

    # Runtime / query-count execution (simple_bench path)
    ap.add_argument("--runtime", type=int, default=None,
                    help="Benchmark runtime in seconds (activates simple_bench execution path).")
    ap.add_argument("--queries", type=int, default=1000,
                    help="Total queries to execute. Used by both paths "
                         "(query-count termination in simple_bench path, "
                         "query set size in enhanced_bench path). Default: 1000.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Queries per batch (required for runtime/query-count mode).")
    ap.add_argument("--report-count", type=int, default=10,
                    help="Batches between progress reports (default: 10).")
    ap.add_argument("--output-dir", default=None,
                    help="Directory for per-process CSV files and statistics "
                         "(simple_bench path). Default: vdbbench_results/<timestamp>.")
    ap.add_argument("--json-output", action="store_true",
                    help="Print benchmark summary as JSON (simple_bench path).")

    # Enhanced path query / execution settings
    ap.add_argument("--k", type=int, default=10,
                    help="Top-k for enhanced_bench path (default: 10).")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--normalize-cosine", action="store_true")
    ap.add_argument("--mode", choices=["single", "mp", "both"], default="both",
                    help="Enhanced_bench execution mode (default: both).")
    ap.add_argument("--processes", type=int, default=8,
                    help="Worker processes (both paths, default: 8).")

    # Output (enhanced path)
    ap.add_argument("--out-dir", default="results",
                    help="Output directory for enhanced_bench JSON/CSV (default: results).")
    ap.add_argument("--tag", default=None)

    # Sweep (enhanced path)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--recall-targets", type=float, nargs="*", default=None)
    ap.add_argument("--optimize", choices=["quality", "cost", "latency"], default="quality")
    ap.add_argument("--sweep-queries", type=int, default=300)

    # Cache regime (enhanced path)
    ap.add_argument("--cache-state", choices=["warm", "cold", "both"], default="both")
    ap.add_argument("--drop-caches-cmd",
                    default="sync; echo 3 | sudo tee /proc/sys/vm/drop_caches")
    ap.add_argument("--restart-milvus-cmd", default=None)

    # Container RSS
    ap.add_argument("--milvus-container", action="append", default=None)

    # Diskstats filter
    ap.add_argument("--disk-dev", action="append", default=None)

    # GT cache (enhanced path)
    ap.add_argument("--gt-cache-dir", default="gt_cache")
    ap.add_argument("--gt-cache-disable", action="store_true")
    ap.add_argument("--gt-cache-force-refresh", action="store_true")

    # Budget mode (enhanced path)
    ap.add_argument("--mem-budget-gb", type=float, default=None)
    ap.add_argument("--host-mem-reserve-gb", type=float, default=None)
    ap.add_argument("--budget-soft", action="store_true")
    ap.add_argument("--budget-label", default=None)

    args = ap.parse_args()

    # Apply YAML defaults (CLI wins)
    if args.config:
        cfg = load_yaml_config(args.config)
        args = apply_yaml_to_args(args, cfg, ap)
        # Also try vdbbench config_loader if available
        if _VDBBENCH_PKG:
            try:
                vdb_cfg = load_config(args.config)
                args = merge_config_with_args(vdb_cfg, args)
            except Exception:
                pass

    # -------- Estimator-only mode --------
    if args.estimate_only:
        if not (args.est_index_type and args.est_n and args.est_dim):
            raise SystemExit("--estimate-only requires --est-index-type --est-n --est-dim")
        est = estimate_memory_bytes(args.est_index_type, args.est_n, args.est_dim,
                                    hnsw_m=args.est_hnsw_m)
        print(json.dumps(est, indent=2))
        return

    if not args.collection:
        raise SystemExit("Missing --collection (or use --estimate-only).")

    # -------------------------------------------------------------------------
    # Determine execution path
    # simple_bench path: --runtime or (--queries + --batch-size) provided
    # enhanced_bench path: neither --runtime nor --batch-size provided
    # -------------------------------------------------------------------------
    use_simple_path = (args.runtime is not None) or (args.batch_size is not None)

    # =========================================================================
    # PATH A: simple_bench execution (runtime / query-count, per-worker CSV)
    # =========================================================================
    if use_simple_path:
        if args.batch_size is None:
            raise SystemExit("--batch-size is required when using --runtime or query-count mode.")
        if args.runtime is None and args.queries is None:
            raise SystemExit("At least one of --runtime or --queries must be specified.")

        # Register graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("\n" + "=" * 60)
        print("ENHANCED VDB BENCH — runtime/query-count mode")
        print("=" * 60)

        # Output directory
        if not args.output_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("vdbbench_results", ts)
        else:
            output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")

        # recall_k default
        recall_k = args.recall_k if args.recall_k else args.search_limit

        # ---- Database verification ----
        print("\n" + "=" * 60)
        print("Database Verification and Collection Loading")
        print("=" * 60)

        conn = connect_to_milvus(args.host, args.port)
        collection_info = load_database(args.host, args.port, args.collection)
        if not collection_info:
            print("Unable to load the specified collection")
            sys.exit(1)

        connections.disconnect("default")

        # Auto-detect vector dim and metric type from collection info
        vec_count = collection_info.get("row_count", 0)
        if isinstance(vec_count, str):
            try:
                vec_count = int(vec_count)
            except ValueError:
                vec_count = 0

        detected_dim = collection_info.get("dimension")
        if detected_dim and detected_dim != "N/A":
            try:
                args.vector_dim = int(detected_dim)
            except (ValueError, TypeError):
                pass

        metric_type = "COSINE"
        if collection_info.get("index_info"):
            mt = collection_info["index_info"][0].get("metric_type")
            if mt:
                metric_type = mt

        index_type = "HNSW"
        if collection_info.get("index_info"):
            it = collection_info["index_info"][0].get("index_type")
            if it:
                index_type = it

        # Cap recall_k
        if vec_count > 0 and recall_k > vec_count:
            print(f"NOTE: recall_k capped from {recall_k} to {vec_count}")
            recall_k = vec_count
        recall_k = min(recall_k, 16384)

        # Detect source vector field name
        source_vec_field = "vector"
        try:
            _tc = connect_to_milvus(args.host, args.port)
            if _tc:
                _src_coll = Collection(args.collection)
                _, source_vec_field, _ = _detect_schema_fields(_src_coll)
                connections.disconnect("default")
                print(f"Detected source vector field: '{source_vec_field}'")
        except Exception as e:
            print(f"Could not detect vector field, using default '{source_vec_field}': {e}")

        # Save config
        config = {
            "timestamp": datetime.now().isoformat(),
            "processes": args.processes,
            "batch_size": args.batch_size,
            "report_count": args.report_count,
            "vector_dim": args.vector_dim,
            "host": args.host,
            "port": args.port,
            "collection_name": args.collection,
            "runtime_seconds": args.runtime,
            "total_queries": args.queries,
            "search_limit": args.search_limit,
            "search_ef": args.search_ef,
            "gt_collection": args.gt_collection,
            "num_query_vectors": args.num_query_vectors,
            "recall_k": recall_k,
            "metric_type": metric_type,
            "index_type": index_type,
        }
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        # ---- Recall setup (outside benchmark timing) ----
        print("\n" + "=" * 60)
        print("RECALL SETUP (outside benchmark timing)")
        print("=" * 60)
        print("Ground truth is pre-computed using a FLAT (brute-force) index.")
        print(f"Using metric type: {metric_type}")

        # Generate deterministic query vectors
        print(f"\nGenerating {args.num_query_vectors} query vectors "
              f"(dim={args.vector_dim}, seed=42)...")
        pre_generated_queries = generate_query_vectors(
            args.num_query_vectors, args.vector_dim, seed=42)
        print(f"Generated {len(pre_generated_queries)} query vectors.")

        # Create / reuse FLAT GT collection
        gt_collection_name = args.gt_collection or f"{args.collection}_flat_gt"

        if args.auto_create_flat:
            print(f"\nSetting up FLAT collection: {gt_collection_name}")
            flat_ok = create_flat_collection(
                host=args.host, port=args.port,
                source_collection_name=args.collection,
                flat_collection_name=gt_collection_name,
                vector_dim=args.vector_dim,
                metric_type=metric_type,
            )
            if not flat_ok:
                print("ERROR: FLAT collection setup failed. Cannot compute recall.")
                sys.exit(1)
        else:
            # Check if GT collection exists; if not, suggest --auto-create-flat
            _tc2 = connect_to_milvus(args.host, args.port)
            if _tc2:
                if not utility.has_collection(gt_collection_name):
                    print(f"⚠️  GT collection '{gt_collection_name}' not found.")
                    print(f"   Run with --auto-create-flat to auto-create it from source.")
                    print(f"   Or specify an existing FLAT collection with --gt-collection.")
                    connections.disconnect("default")
                    sys.exit(1)
                connections.disconnect("default")

        # Pre-compute ground truth
        ground_truth = precompute_ground_truth(
            host=args.host, port=args.port,
            flat_collection_name=gt_collection_name,
            query_vectors=pre_generated_queries,
            top_k=recall_k,
            metric_type=metric_type,
        )

        if not ground_truth:
            print("ERROR: Ground truth computation failed. Cannot compute recall.")
            sys.exit(1)

        print(f"Ground truth ready: {len(ground_truth)} queries pre-computed.")

        # Initial disk stats
        print('\nCollecting initial disk statistics...')
        start_disk_stats = read_disk_stats()

        # ---- Benchmark execution ----
        max_queries_per_process = None
        remainder = 0
        if args.queries is not None and args.processes > 1:
            max_queries_per_process = args.queries // args.processes
            remainder = args.queries % args.processes

        print("\n" + "=" * 60)
        print("Benchmark Execution")
        print("=" * 60)
        if max_queries_per_process is not None:
            print(f"Starting benchmark: {args.processes} processes × "
                  f"{max_queries_per_process} queries/process")
        else:
            print(f"Starting benchmark: {args.processes} processes, "
                  f"runtime={args.runtime}s")
        print(f"Recall: {len(pre_generated_queries)} pre-generated queries, recall@{recall_k}")
        print(f"NOTE: batch_end timing is placed BEFORE recall capture — performance unaffected.")
        print(f"NOTE: recall hits written to per-worker recall_hits_p<N>.jsonl files.")

        processes_list = []
        stagger = 1.0 / max(1, args.processes)

        if args.processes > 1:
            print(f"Staggering process startup by {stagger:.3f}s")
            try:
                for i in range(args.processes):
                    if i > 0:
                        time.sleep(stagger)

                    process_max_queries = None
                    if max_queries_per_process is not None:
                        process_max_queries = max_queries_per_process + (remainder if i == 0 else 0)

                    p = mp.Process(
                        target=execute_batch_queries,
                        args=(
                            i, args.host, args.port, args.collection,
                            args.vector_dim, args.batch_size, args.report_count,
                            process_max_queries, args.runtime,
                            output_dir, shutdown_flag, pre_generated_queries,
                            None,   # ann_results_dict deprecated; workers write JSONL files
                            args.search_limit, args.search_ef,
                            source_vec_field, metric_type, index_type,
                        ),
                    )
                    print(f'Starting process {i}...')
                    p.start()
                    processes_list.append(p)

                for p in processes_list:
                    p.join()

            except Exception as e:
                print(f"Error during benchmark execution: {e}")
                with shutdown_flag.get_lock():
                    shutdown_flag.value = 1
                for p in processes_list:
                    if p.is_alive():
                        p.join(timeout=5)
                        if p.is_alive():
                            p.terminate()
        else:
            process_max_queries = args.queries if args.queries is not None else None
            execute_batch_queries(
                0, args.host, args.port, args.collection, args.vector_dim,
                args.batch_size, args.report_count,
                process_max_queries, args.runtime, output_dir, shutdown_flag,
                pre_generated_queries, None,  # ann_results_dict deprecated
                args.search_limit, args.search_ef, source_vec_field,
                metric_type, index_type,
            )

        # Final disk stats
        print('Reading final disk statistics...')
        end_disk_stats = read_disk_stats()
        disk_io_diff = calculate_disk_io_diff(start_disk_stats, end_disk_stats)

        # ---- Post-hoc recall calculation ----
        print("\nCalculating recall from per-worker JSONL files...")
        ann_results_by_query = load_recall_hits(output_dir)
        print(f"  Loaded ANN hits for {len(ann_results_by_query)} unique query indices "
              f"from {args.processes} worker(s).")

        recall_stats = calc_recall(ann_results_by_query, ground_truth, recall_k)

        recall_output_file = os.path.join(output_dir, "recall_stats.json")
        with open(recall_output_file, 'w') as f:
            json.dump(recall_stats, f, indent=2)

        # ---- Aggregate statistics ----
        print("Calculating benchmark statistics...")
        stats = calculate_statistics(output_dir, recall_stats=recall_stats)

        if disk_io_diff:
            total_bytes_read    = sum(d["bytes_read"]    for d in disk_io_diff.values())
            total_bytes_written = sum(d["bytes_written"] for d in disk_io_diff.values())
            total_read_ios      = sum(d.get("read_ios",  0) for d in disk_io_diff.values())
            total_write_ios     = sum(d.get("write_ios", 0) for d in disk_io_diff.values())
            total_time = max(stats.get("total_time_seconds", 1), 1e-6)

            read_mbps  = total_bytes_read    / total_time / (1024 * 1024)
            write_mbps = total_bytes_written / total_time / (1024 * 1024)
            read_iops  = total_read_ios  / total_time
            write_iops = total_write_ios / total_time

            dev_stats_out = {}
            for dev, s in disk_io_diff.items():
                if s["bytes_read"] > 0 or s["bytes_written"] > 0 or \
                   s.get("read_ios", 0) > 0 or s.get("write_ios", 0) > 0:
                    dev_read_mbps  = s["bytes_read"]    / total_time / (1024 * 1024)
                    dev_write_mbps = s["bytes_written"] / total_time / (1024 * 1024)
                    dev_read_iops  = s.get("read_ios",  0) / total_time
                    dev_write_iops = s.get("write_ios", 0) / total_time
                    dev_stats_out[dev] = {
                        "bytes_read":    s["bytes_read"],
                        "bytes_written": s["bytes_written"],
                        "read_ios":      s.get("read_ios",  0),
                        "write_ios":     s.get("write_ios", 0),
                        "read_formatted":  format_bytes(s["bytes_read"]),
                        "write_formatted": format_bytes(s["bytes_written"]),
                        "read_mbps":     round(dev_read_mbps,  2),
                        "write_mbps":    round(dev_write_mbps, 2),
                        "read_iops":     round(dev_read_iops,  1),
                        "write_iops":    round(dev_write_iops, 1),
                    }

            stats["disk_io"] = {
                "total_bytes_read":         total_bytes_read,
                "total_bytes_written":      total_bytes_written,
                "total_read_ios":           total_read_ios,
                "total_write_ios":          total_write_ios,
                "total_read_formatted":     format_bytes(total_bytes_read),
                "total_write_formatted":    format_bytes(total_bytes_written),
                "read_mbps":                round(read_mbps,  2),
                "write_mbps":               round(write_mbps, 2),
                "read_iops":                round(read_iops,  1),
                "write_iops":               round(write_iops, 1),
                "total_bytes_read_per_sec": total_bytes_read / total_time,
                "benchmark_duration_sec":   round(total_time, 2),
                "devices": dev_stats_out,
            }
        else:
            stats["disk_io"] = {"error": "Disk I/O statistics not available"}

        with open(os.path.join(output_dir, "statistics.json"), 'w') as f:
            json.dump(stats, f, indent=2)

        if args.json_output:
            print("\nBenchmark statistics as JSON:")
            print(json.dumps(stats))
        else:
            print("\n" + "=" * 60)
            print("BENCHMARK SUMMARY")
            print("=" * 60)
            print(f"Total Queries: {stats.get('total_queries', 0)}")
            print(f"Total Batches: {stats.get('batch_count', 0)}")
            print(f"Total Runtime: {stats.get('total_time_seconds', 0):.2f}s")

            print("\nQUERY STATISTICS")
            print("-" * 60)
            print(f"Mean Latency:      {stats.get('mean_latency_ms', 0):.2f} ms")
            print(f"Median Latency:    {stats.get('median_latency_ms', 0):.2f} ms")
            print(f"P95 Latency:       {stats.get('p95_latency_ms', 0):.2f} ms")
            print(f"P99 Latency:       {stats.get('p99_latency_ms', 0):.2f} ms")
            print(f"P99.9 Latency:     {stats.get('p999_latency_ms', 0):.2f} ms")
            print(f"P99.99 Latency:    {stats.get('p9999_latency_ms', 0):.2f} ms")
            print(f"Throughput:        {stats.get('throughput_qps', 0):.2f} queries/second")

            print("\nBATCH STATISTICS")
            print("-" * 60)
            mean_bms = stats.get('mean_batch_time_ms', 0)
            print(f"Mean Batch Time:   {mean_bms:.2f} ms")
            print(f"Median Batch Time: {stats.get('median_batch_time_ms', 0):.2f} ms")
            print(f"P95 Batch Time:    {stats.get('p95_batch_time_ms', 0):.2f} ms")
            print(f"P99 Batch Time:    {stats.get('p99_batch_time_ms', 0):.2f} ms")
            print(f"P99.9 Batch Time:  {stats.get('p999_batch_time_ms', 0):.2f} ms")
            print(f"P99.99 Batch Time: {stats.get('p9999_batch_time_ms', 0):.2f} ms")
            print(f"Max Batch Time:    {stats.get('max_batch_time_ms', 0):.2f} ms")
            bps = (1000.0 / mean_bms) if mean_bms > 0 else 0
            print(f"Batch Throughput:  {bps:.2f} batches/second")

            r = stats.get("recall", {}) or {}
            print(f"\nRECALL STATISTICS (recall@{r.get('k', recall_k)})")
            print("-" * 60)
            print(f"Mean Recall:       {r.get('mean_recall', 0):.4f}")
            print(f"Median Recall:     {r.get('median_recall', 0):.4f}")
            print(f"Min Recall:        {r.get('min_recall', 0):.4f}")
            print(f"Max Recall:        {r.get('max_recall', 0):.4f}")
            print(f"P95 Recall:        {r.get('p95_recall', 0):.4f}")
            print(f"P99 Recall:        {r.get('p99_recall', 0):.4f}")
            print(f"Queries Evaluated: {r.get('num_queries_evaluated', 0)}")

            print("\nDISK I/O DURING BENCHMARK")
            print("-" * 60)
            if disk_io_diff:
                di = stats.get("disk_io", {})
                print(f"Total Read:        {di.get('total_read_formatted', 'N/A')}"
                      f"  ({di.get('read_mbps', 0):.2f} MB/s,"
                      f"  {di.get('read_iops', 0):.0f} IOPS)")
                print(f"Total Write:       {di.get('total_write_formatted', 'N/A')}"
                      f"  ({di.get('write_mbps', 0):.2f} MB/s,"
                      f"  {di.get('write_iops', 0):.0f} IOPS)")
                if di.get("devices"):
                    print("\nPer-Device Breakdown:")
                    for device, ds in di["devices"].items():
                        print(f"  {device}:")
                        print(f"    Read:  {ds['read_formatted']}"
                              f"  ({ds['read_mbps']:.2f} MB/s, {ds['read_iops']:.0f} IOPS)")
                        print(f"    Write: {ds['write_formatted']}"
                              f"  ({ds['write_mbps']:.2f} MB/s, {ds['write_iops']:.0f} IOPS)")
            else:
                print("Disk I/O statistics not available")

            print(f"\nDetailed results: {output_dir}")
            print(f"Recall details:   {recall_output_file}")
            print("=" * 60)

        return  # End of simple_bench path

    # =========================================================================
    # PATH B: enhanced_bench execution (sweep / cache / budget)
    # =========================================================================
    gt_cache_dir = Path(args.gt_cache_dir) if args.gt_cache_dir else None

    connections.connect("default", host=args.host, port=args.port)

    if not utility.has_collection(args.collection):
        raise SystemExit(f"Collection not found: {args.collection}")
    col = Collection(args.collection)
    print(f"Loading collection {args.collection}...")
    try:
        col.load()
    except Exception as e:
        raise SystemExit(f"Failed to load collection {args.collection}: {e}")

    vector_field, dim, dtype_obj, dtype_name = get_vector_field_info(col)
    if not vector_field or not dim or dtype_obj is None:
        raise SystemExit(f"Could not detect vector field/dim for collection {args.collection}")

    if is_binary_vector_dtype(dtype_obj):
        raise SystemExit(
            f"Detected BINARY_VECTOR field '{vector_field}' in {args.collection} "
            f"(dtype={dtype_name}). This benchmark currently assumes FLOAT vectors.")

    index_type, metric_type, build_params = get_index_params(col)
    normalize = args.normalize_cosine and (metric_type.upper() == "COSINE")

    print(f"Detected: collection={args.collection} index_type={index_type} "
          f"metric={metric_type} vector_field={vector_field} dim={dim} dtype={dtype_name}")

    q_main = generate_queries(dim, args.queries, args.seed, normalize)

    # Optionally auto-create FLAT GT collection
    if args.auto_create_flat and not args.gt_collection:
        auto_gt_name = f"{args.collection}_flat_gt"
        print(f"\nAuto-creating FLAT GT collection: {auto_gt_name}")
        connections.disconnect("default")
        flat_ok = create_flat_collection(
            host=args.host, port=args.port,
            source_collection_name=args.collection,
            flat_collection_name=auto_gt_name,
            vector_dim=dim,
            metric_type=metric_type,
        )
        if not flat_ok:
            raise SystemExit("FLAT GT collection creation failed.")
        args.gt_collection = auto_gt_name
        connections.connect("default", host=args.host, port=args.port)
        col = Collection(args.collection)
        col.load()

    # GT collection
    if args.gt_collection:
        if not utility.has_collection(args.gt_collection):
            raise SystemExit(f"GT collection not found: {args.gt_collection}")
        gt_col = Collection(args.gt_collection)
        gt_col.load()
        gt_vector_field, gt_dim, gt_dtype_obj, gt_dtype_name = get_vector_field_info(gt_col)
        if gt_dim != dim:
            raise SystemExit(f"GT dim {gt_dim} != test dim {dim}")
        if not gt_vector_field:
            raise SystemExit("Could not detect vector field in GT collection")
        if is_binary_vector_dtype(gt_dtype_obj):
            raise SystemExit(f"GT collection is BINARY_VECTOR; expected FLOAT vectors.")
        gt_index_type, _, _ = get_index_params(gt_col)
        if gt_index_type != "FLAT":
            print(f"⚠️  GT collection uses {gt_index_type} index (FLAT recommended for accurate GT)")
        gt_vector_field_name = gt_vector_field
    else:
        print("⚠️  No --gt-collection provided. Recall computed against same collection/index.")
        gt_col = col
        gt_vector_field_name = vector_field

    recall_targets: List[float] = []
    if args.sweep:
        recall_targets = args.recall_targets if args.recall_targets else [args.target_recall]

    def maybe_restart_milvus():
        if args.restart_milvus_cmd:
            rc, _out, err = run_cmd(args.restart_milvus_cmd)
            if rc != 0:
                print(f"⚠️ restart-milvus-cmd failed rc={rc}: {err}")

    def do_drop_caches():
        rc, _out, err = run_cmd(args.drop_caches_cmd)
        if rc != 0:
            print(f"⚠️ drop-caches-cmd failed rc={rc}: {err}")

    def get_rss_bytes_now() -> Optional[int]:
        if args.milvus_container:
            return get_rss_bytes_for_containers(args.milvus_container)
        return None

    def maybe_enforce_budget_or_skip(
        host_before: HostMemSnapshot,
    ) -> Tuple[bool, Optional[bool], Optional[bool], Optional[str]]:
        rss = get_rss_bytes_now()
        rss_ok, host_ok, reason = check_budgets(
            rss_bytes=rss,
            host_before=host_before,
            mem_budget_gb=args.mem_budget_gb,
            host_mem_reserve_gb=args.host_mem_reserve_gb,
        )
        ok = rss_ok and host_ok
        if ok:
            return True, rss_ok, host_ok, None
        if args.budget_soft:
            print(f"⚠️ Budget violation (soft): {reason}")
            return False, rss_ok, host_ok, reason
        raise SystemExit(f"Budget violation (hard): {reason}")

    def run_one_cache_state(cache_state: str) -> Tuple[List[RunResult], List[Dict[str, Any]]]:
        if cache_state == "cold":
            maybe_restart_milvus()
            do_drop_caches()
        elif cache_state == "warm":
            warmup_params = default_search_params_for_index(index_type, build_params)
            warmup_queries = q_main[:min(10, len(q_main))]
            print(f"🔥 Warming up cache with {len(warmup_queries)} queries...")
            for qv in warmup_queries:
                try:
                    _ = col.search([qv.tolist()], vector_field,
                                   make_search_params_full(metric_type, warmup_params),
                                   limit=args.k)
                except Exception:
                    pass

        runs: List[RunResult] = []
        sweep_rows_all: List[Dict[str, Any]] = []

        rss_b = get_rss_bytes_now()
        chosen_params_by_target: Dict[Any, Dict[str, Any]] = {}

        if args.sweep:
            q_sweep_seed = args.seed + 999
            q_sweep = generate_queries(dim, args.sweep_queries, q_sweep_seed, normalize)

            for tgt in recall_targets:
                best_params, sweep_report = pick_best_by_target_recall(
                    collection=col, gt_collection=gt_col,
                    queries=q_sweep, vector_field=vector_field,
                    metric_type=metric_type, k=args.k,
                    index_type=index_type, target_recall=tgt,
                    optimize=args.optimize, rss_bytes=rss_b,
                    cache_state=cache_state, build_params=build_params,
                    gt_cache_dir=gt_cache_dir,
                    gt_cache_disable=args.gt_cache_disable,
                    gt_cache_force_refresh=args.gt_cache_force_refresh,
                    gt_query_seed=q_sweep_seed,
                    normalize_cosine=normalize,
                )
                chosen_params_by_target[tgt] = best_params

                for row in sweep_report:
                    row2 = dict(row)
                    row2["recall_target"] = tgt
                    row2["index_type"] = index_type
                    row2["optimize"] = args.optimize
                    sweep_rows_all.append(row2)

                print(f"✅ [{cache_state}] target={tgt:.3f} optimize={args.optimize} "
                      f"selected params: {best_params}")

            chosen_params_by_target["max_throughput"] = minimal_search_params_for_index(index_type)
            print(f"Max throughput params [{cache_state}]: {chosen_params_by_target['max_throughput']}")
        else:
            chosen_params_by_target["max_throughput"] = minimal_search_params_for_index(index_type)
            chosen_params_by_target[None] = default_search_params_for_index(index_type, build_params)
            print(f"Max throughput params [{cache_state}]: {chosen_params_by_target['max_throughput']}")
            print(f"Default params [{cache_state}]:        {chosen_params_by_target[None]}")

        gt_ids_main = compute_ground_truth(
            gt_col, q_main, gt_vector_field_name, metric_type, args.k,
            cache_dir=gt_cache_dir, cache_disable=args.gt_cache_disable,
            cache_force_refresh=args.gt_cache_force_refresh,
            query_seed=args.seed, normalize_cosine=normalize,
        )

        targets_to_run = (["max_throughput"] + recall_targets) if args.sweep else ["max_throughput", None]

        for tgt in targets_to_run:
            algo_params = chosen_params_by_target[tgt]
            is_max_throughput = (tgt == "max_throughput")

            host_before = HostMemSnapshot.from_proc_meminfo()
            should_run, rss_ok, host_ok, reason = maybe_enforce_budget_or_skip(host_before)
            if not should_run:
                annotated_params = dict(algo_params)
                if args.sweep and not is_max_throughput:
                    annotated_params["_recall_target"] = tgt
                    annotated_params["_optimize"] = args.optimize
                elif is_max_throughput:
                    annotated_params["_note"] = "max_throughput"

                rr = RunResult(
                    mode="skipped", index_type=index_type, metric_type=metric_type,
                    algo_params=annotated_params, k=args.k, queries=args.queries,
                    qps=0.0, lat_ms_avg=float("nan"), lat_ms_p50=float("nan"),
                    lat_ms_p95=float("nan"), lat_ms_p99=float("nan"),
                    recall=None, rss_bytes=get_rss_bytes_now(), cache_state=cache_state,
                    host_mem_avail_before=host_before.mem_available_bytes,
                    host_mem_cached_before=host_before.cached_bytes,
                    budget_rss_ok=rss_ok, budget_host_ok=host_ok,
                    budget_reason=reason, is_max_throughput=is_max_throughput,
                )
                runs.append(rr)
                continue

            rss_b_run = get_rss_bytes_now()

            if args.mode in ("single", "both"):
                host_before_s = HostMemSnapshot.from_proc_meminfo()
                r1 = bench_single(
                    collection=col, queries=q_main, vector_field=vector_field,
                    metric_type=metric_type, algo_params=algo_params,
                    k=args.k, gt_ids=gt_ids_main,
                    disk_devices=args.disk_dev, rss_bytes=rss_b_run,
                    cache_state=cache_state, host_before=host_before_s,
                    host_after=HostMemSnapshot.from_proc_meminfo(),
                )
                r1.index_type = index_type
                r1.algo_params = dict(r1.algo_params)
                r1.is_max_throughput = is_max_throughput
                if args.sweep and not is_max_throughput:
                    r1.algo_params["_recall_target"] = tgt
                    r1.algo_params["_optimize"] = args.optimize
                elif is_max_throughput:
                    r1.algo_params["_note"] = "max_throughput"
                r1.budget_rss_ok = rss_ok
                r1.budget_host_ok = host_ok
                r1.budget_reason = reason
                runs.append(r1)

            if args.mode in ("mp", "both"):
                host_before_m = HostMemSnapshot.from_proc_meminfo()
                mp_res = bench_multiprocess(
                    host=args.host, port=args.port,
                    collection_name=args.collection, vector_field=vector_field,
                    metric_type=metric_type, algo_params=algo_params,
                    k=args.k, queries=q_main, processes=args.processes,
                    disk_devices=args.disk_dev, gt_ids=gt_ids_main,
                )
                host_after_m = HostMemSnapshot.from_proc_meminfo()
                all_lat = mp_res["all_lat"]
                mp_dt   = mp_res["disk"]

                r2 = RunResult(
                    mode=f"mp({args.processes})", index_type=index_type,
                    metric_type=metric_type, algo_params=dict(algo_params),
                    k=args.k, queries=len(q_main), qps=mp_res["qps"],
                    lat_ms_avg=float(np.mean(all_lat)) if all_lat else float("nan"),
                    lat_ms_p50=percentile(all_lat, 50),
                    lat_ms_p95=percentile(all_lat, 95),
                    lat_ms_p99=percentile(all_lat, 99),
                    recall=mp_res["recall"],
                    recall_stats=mp_res["recall_stats"],
                    disk_read_bytes=mp_res["rd"]  if mp_dt["available"] else None,
                    disk_write_bytes=mp_res["wr"] if mp_dt["available"] else None,
                    read_bytes_per_query=mp_res["read_bpq"],
                    disk_read_iops=mp_dt["read_iops"]   if mp_dt["available"] else None,
                    disk_write_iops=mp_dt["write_iops"] if mp_dt["available"] else None,
                    disk_read_mbps=mp_dt["read_mbps"]   if mp_dt["available"] else None,
                    disk_write_mbps=mp_dt["write_mbps"] if mp_dt["available"] else None,
                    disk_duration_sec=mp_res["total_sec"] if mp_dt["available"] else None,
                    rss_bytes=rss_b_run,
                    cache_state=cache_state,
                    host_mem_avail_before=host_before_m.mem_available_bytes,
                    host_mem_avail_after=host_after_m.mem_available_bytes,
                    host_mem_cached_before=host_before_m.cached_bytes,
                    host_mem_cached_after=host_after_m.cached_bytes,
                    is_max_throughput=is_max_throughput,
                )
                if args.sweep and not is_max_throughput:
                    r2.algo_params["_recall_target"] = tgt
                    r2.algo_params["_optimize"] = args.optimize
                elif is_max_throughput:
                    r2.algo_params["_note"] = "max_throughput"

                r2.quality_score = r2.qps
                if r2.rss_bytes and r2.rss_bytes > 0:
                    r2.cost_score = r2.qps / (r2.rss_bytes / (1024 ** 3))
                r2.budget_rss_ok = rss_ok
                r2.budget_host_ok = host_ok
                r2.budget_reason = reason
                runs.append(r2)

        return runs, sweep_rows_all

    all_runs: List[RunResult] = []
    sweep_rows_global: List[Dict[str, Any]] = []

    cache_states = (["warm", "cold"] if args.cache_state == "both"
                    else [args.cache_state])
    for cs in cache_states:
        rs, sw = run_one_cache_state(cs)
        all_runs.extend(rs)
        sweep_rows_global.extend(sw)

    sweep_report = sweep_rows_global if args.sweep else None

    for r in all_runs:
        mode_label = "[MAX THROUGHPUT]" if r.is_max_throughput else ""
        label = f"{r.mode} {mode_label}".strip()
        if r.mode == "skipped":
            print(f"\n[SKIPPED — {label}] budget: {r.budget_reason}")
            continue
        print_bench_summary(r, label=label)
        if r.host_mem_avail_before is not None and r.host_mem_avail_after is not None:
            print(f"  Host MemAvail:     "
                  f"{bytes_to_gb(r.host_mem_avail_before):.2f} GB → "
                  f"{bytes_to_gb(r.host_mem_avail_after):.2f} GB")

    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = args.tag or args.collection
    base = f"combined_bench_{tag}_{ts}"
    out_dir = Path(args.out_dir)
    write_outputs(out_dir, base, all_runs, sweep_report)

    print(f"✅ Wrote: {out_dir / (base + '.json')}")
    print(f"✅ Wrote: {out_dir / (base + '.csv')}")
    if sweep_report is not None:
        print(f"✅ Wrote: {out_dir / (base + '.sweep.csv')}")
    if gt_cache_dir is not None and not args.gt_cache_disable:
        print(f"ℹ️  GT cache dir: {gt_cache_dir.resolve()} "
              f"(use --gt-cache-force-refresh if dataset changed)")


if __name__ == "__main__":
    main()
