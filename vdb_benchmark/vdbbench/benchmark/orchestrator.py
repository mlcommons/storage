"""Benchmark orchestrator -- producer / consumer pipeline.

Coordinates three concerns during the **load** phase:

1. **Producer** (:class:`VectorGenerator`) -- generates random vectors in
   blocks on a background thread.
2. **VDB consumer** (:class:`VectorDBBackend`) -- inserts each block into
   the target database (main thread, network I/O).
3. **Ground-truth consumer** (:class:`GroundTruthBuilder`) -- computes
   brute-force nearest neighbors for each block against the query set
   (background thread, runs in parallel with insert).

And during the **search** phase:

4. **SearchRunner** -- queries the VDB in batches, computes recall
   against the truth table, and logs QPS / latency percentiles.

Three runtime modes are supported via ``BenchmarkConfig.mode``:

* ``load``   -- generate vectors, ingest, compute ground truth.
* ``search`` -- run search queries against an already-loaded collection.
* ``both``   -- load then search.

After all blocks have been processed the orchestrator writes artifacts
to ``output_dir``:

* **Vectors in the database** -- already stored by the VDB consumer.
* **query_vectors.npy** -- the query-vector matrix.
* **ground_truth.npz** -- the truth table (``truth_table``) and the
  query vectors (``query_vectors``).  ``truth_table[q]`` is a length-K
  array of database IDs ordered closest-first to query *q*.
* **search_results.json** -- search benchmark results (search/both modes).

Usage::

    from benchmark.orchestrator import BenchmarkOrchestrator

    orch = BenchmarkOrchestrator(config, backend)
    orch.run()                # blocking -- runs load, search, or both
    orch.save(output_dir)     # write artifacts
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .backends.base import VectorDBBackend
from .generator import VectorBlock, VectorGenerator, generate_query_vectors
from .ground_truth import GroundTruthBuilder
from .search_runner import (
    SearchResult,
    SearchRunner,
    build_truth_from_flat,
    ensure_flat_collection,
)

logger = logging.getLogger(__name__)

# Valid mode values
MODES = ("load", "search", "both")
# Valid truth_mode values
TRUTH_MODES = ("precomputed", "flat_index")


@dataclass
class BenchmarkConfig:
    """All tunables for a single benchmark run."""

    # Run mode
    mode: str = "load"   # "load", "search", or "both"

    # Database vectors
    num_vectors: int = 1_000_000
    dimension: int = 1536
    distribution: str = "uniform"
    seed: int = 42
    block_size: int = 100_000
    batch_size: int = 10_000

    # Query vectors
    num_query_vectors: int = 10_000
    query_seed: int = 99

    # Ground truth
    truth_k: int = 100
    truth_mode: str = "precomputed"  # "precomputed" or "flat_index"

    # Index
    collection_name: str = "bench_vectors"
    metric_type: str = "COSINE"
    index_type: str = "HNSW"
    index_params: Dict[str, Any] = field(default_factory=dict)
    num_shards: int = 1
    force: bool = False

    # Connection (used by Milvus backend)
    host: str = "127.0.0.1"
    port: str = "19530"

    # Pipeline tuning
    max_queue_depth: int = 4

    # Post-load
    compact: bool = False
    monitor_interval: int = 5

    # Search benchmark
    search_k: int = 10
    search_params: Dict[str, Any] = field(default_factory=dict)
    num_search_rounds: int = 1
    search_batch_size: int = 1
    log_interval: int = 1000

    # Artifacts directory (for search mode -- where to load from)
    artifacts_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkConfig":
        """Build from a flat or sectioned dict (like the YAML configs).

        Nested dicts that correspond to known dict-typed fields
        (e.g. ``search_params``, ``index_params``) are preserved as-is.
        Other nested dicts (YAML sections like ``database``, ``dataset``)
        are flattened into the top level.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}
        # Fields that are Dict-typed and should stay as dicts
        dict_fields = {
            f.name for f in cls.__dataclass_fields__.values()
            if f.default_factory is dict  # type: ignore[comparison-overlap]
        }
        flat: Dict[str, Any] = {}
        for key, val in d.items():
            if isinstance(val, dict) and key not in dict_fields:
                # YAML section -- flatten its contents
                flat.update(val)
            else:
                flat[key] = val
        return cls(**{k: v for k, v in flat.items() if k in known})


class BenchmarkOrchestrator:
    """Wire everything together and drive the pipeline.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark tunables.
    backend : VectorDBBackend
        A connected backend instance (``connect()`` already called).
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        backend: VectorDBBackend,
    ) -> None:
        self.cfg = config
        self.backend = backend

        self.query_vectors: Optional[np.ndarray] = None
        self.truth_table: Optional[np.ndarray] = None
        self.search_result: Optional[SearchResult] = None

        # Timing bookkeeping
        self._timings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Execute the benchmark in the configured mode.

        Returns a summary dict with timings and counts.
        """
        mode = self.cfg.mode.lower()
        if mode not in MODES:
            raise ValueError(
                f"Invalid mode '{mode}'.  Must be one of {MODES}"
            )

        summary: Dict[str, Any] = {}

        if mode in ("load", "both"):
            summary.update(self._run_load())

        if mode in ("search", "both"):
            summary.update(self._run_search())

        logger.info("Pipeline complete (%s mode).  Summary: %s", mode, summary)
        return summary

    def save(self, output_dir: str) -> Dict[str, str]:
        """Persist artifacts to *output_dir*.

        Returns a dict mapping artifact name to file path.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        # Query vectors
        if self.query_vectors is not None:
            qpath = os.path.join(output_dir, "query_vectors.npy")
            np.save(qpath, self.query_vectors)
            paths["query_vectors"] = qpath

        # Ground-truth table
        if self.truth_table is not None:
            gtpath = os.path.join(output_dir, "ground_truth.npz")
            np.savez_compressed(
                gtpath,
                truth_table=self.truth_table,
                query_vectors=self.query_vectors,
            )
            paths["ground_truth"] = gtpath

        # Search results
        if self.search_result is not None:
            spath = os.path.join(output_dir, "search_results.json")
            with open(spath, "w") as f:
                json.dump(self.search_result.to_dict(), f, indent=2, default=str)
            paths["search_results"] = spath

        # Config + timings
        meta = {
            "config": self.cfg.to_dict(),
            "timings": self._timings,
        }
        mpath = os.path.join(output_dir, "benchmark_meta.json")
        with open(mpath, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        paths["meta"] = mpath

        logger.info("Artifacts saved to %s", output_dir)
        for name, p in paths.items():
            logger.info("  %s -> %s", name, p)
        return paths

    # ------------------------------------------------------------------
    # Load phase
    # ------------------------------------------------------------------
    def _run_load(self) -> Dict[str, Any]:
        """Execute the full load pipeline (blocking)."""
        cfg = self.cfg

        # ---- 1. Generate query vectors ---------------------------------
        logger.info(
            "Generating %s query vectors (%s-d, seed=%d) ...",
            f"{cfg.num_query_vectors:,}", f"{cfg.dimension:,}", cfg.query_seed,
        )
        t0 = time.time()
        self.query_vectors = generate_query_vectors(
            num_queries=cfg.num_query_vectors,
            dimension=cfg.dimension,
            distribution=cfg.distribution,
            seed=cfg.query_seed,
        )
        self._timings["query_gen_sec"] = time.time() - t0
        logger.info(
            "%s query vectors generated in %.2f s",
            f"{cfg.num_query_vectors:,}", self._timings["query_gen_sec"],
        )

        # ---- 2. Create the collection ----------------------------------
        logger.info(
            "Creating collection '%s' (%s / %s) ...",
            cfg.collection_name, cfg.index_type, cfg.metric_type,
        )
        t0 = time.time()
        self.backend.create_collection(
            name=cfg.collection_name,
            dimension=cfg.dimension,
            metric_type=cfg.metric_type,
            index_type=cfg.index_type,
            index_params=cfg.index_params,
            num_shards=cfg.num_shards,
            force=cfg.force,
        )
        self._timings["create_collection_sec"] = time.time() - t0

        # ---- 2b. Create FLAT companion (if flat_index truth mode) ------
        flat_name = f"{cfg.collection_name}_flat"
        if cfg.truth_mode == "flat_index":
            ensure_flat_collection(
                backend=self.backend,
                source_name=cfg.collection_name,
                flat_name=flat_name,
                dimension=cfg.dimension,
                metric_type=cfg.metric_type,
            )

        # ---- 3. Set up producer and ground-truth builder ---------------
        generator = VectorGenerator(
            total_vectors=cfg.num_vectors,
            dimension=cfg.dimension,
            block_size=cfg.block_size,
            distribution=cfg.distribution,
            seed=cfg.seed,
            max_queue_depth=cfg.max_queue_depth,
        )
        # Only build brute-force GT when in precomputed mode
        gt_builder: Optional[GroundTruthBuilder] = None
        if cfg.truth_mode == "precomputed":
            gt_builder = GroundTruthBuilder(
                query_vectors=self.query_vectors,
                k=cfg.truth_k,
                metric=cfg.metric_type,
            )

        # ---- 4. Run the pipeline ---------------------------------------
        # Insert (network I/O) and GT update (BLAS matmul) both release
        # the GIL, so they run truly in parallel when overlapped.
        logger.info(
            "Starting pipeline: %s vectors, block_size=%s, batch_size=%s",
            f"{cfg.num_vectors:,}", f"{cfg.block_size:,}", f"{cfg.batch_size:,}",
        )
        t_pipeline = time.time()
        total_inserted = 0
        blocks_consumed = 0

        def _timed_gt_update(builder, blk):
            """Run GT update and return its wall-clock time."""
            t0 = time.time()
            builder.update(blk)
            return time.time() - t0

        generator.start()

        with ThreadPoolExecutor(max_workers=1,
                                thread_name_prefix="gt") as gt_pool:
            while True:
                block: Optional[VectorBlock] = generator.queue.get()
                if block is None:
                    break  # sentinel

                n = len(block.ids)
                t_wall = time.time()

                # -- kick off GT in background thread --------------------
                gt_future = None
                if gt_builder is not None:
                    gt_future = gt_pool.submit(
                        _timed_gt_update, gt_builder, block,
                    )

                # -- consumer 1: insert into VDB (main thread) -----------
                t_insert = time.time()
                for off in range(0, n, cfg.batch_size):
                    end = min(off + cfg.batch_size, n)
                    self.backend.insert_batch(
                        name=cfg.collection_name,
                        ids=block.ids[off:end],
                        vectors=block.vectors[off:end],
                    )
                insert_elapsed = time.time() - t_insert
                total_inserted += n

                # -- consumer 1b: mirror into FLAT collection ------------
                if cfg.truth_mode == "flat_index":
                    for off in range(0, n, cfg.batch_size):
                        end = min(off + cfg.batch_size, n)
                        self.backend.insert_batch(
                            name=flat_name,
                            ids=block.ids[off:end],
                            vectors=block.vectors[off:end],
                        )

                # -- wait for GT to finish -------------------------------
                gt_elapsed = gt_future.result() if gt_future else 0.0
                wall_elapsed = time.time() - t_wall

                blocks_consumed += 1
                logger.info(
                    "Block %d/%d consumed: %s vectors "
                    "(insert=%.2fs | GT=%.2fs | wall=%.2fs).  "
                    "Total: %s / %s",
                    blocks_consumed, generator.num_blocks, f"{n:,}",
                    insert_elapsed, gt_elapsed, wall_elapsed,
                    f"{total_inserted:,}", f"{cfg.num_vectors:,}",
                )

        generator.join()  # propagate any producer error

        self._timings["pipeline_sec"] = time.time() - t_pipeline
        logger.info(
            "%s vectors inserted in %.2f s",
            f"{total_inserted:,}", self._timings["pipeline_sec"],
        )

        # ---- 5. Flush + optional compaction + wait for index --------------
        logger.info("Flushing collection ...")
        t0 = time.time()
        self.backend.flush(cfg.collection_name)
        if cfg.truth_mode == "flat_index":
            self.backend.flush(flat_name)
        self._timings["flush_sec"] = time.time() - t0
        logger.info("Flush completed in %.2f s", self._timings["flush_sec"])

        if cfg.compact:
            logger.info("Compacting segments ...")
            t0 = time.time()
            self.backend.compact(cfg.collection_name)
            self.backend.flush(cfg.collection_name)
            self._timings["compact_sec"] = time.time() - t0
            logger.info("Compaction completed in %.2f s", self._timings["compact_sec"])

        logger.info("Waiting for index build ...")
        t0 = time.time()
        self.backend.wait_for_index(
            cfg.collection_name, interval=cfg.monitor_interval,
            compacted=cfg.compact,
        )
        self._timings["index_build_sec"] = time.time() - t0

        # ---- 7. Finalize ground truth ----------------------------------
        if gt_builder is not None:
            logger.info("Building final truth table (k=%d) ...", cfg.truth_k)
            t0 = time.time()
            self.truth_table = gt_builder.build()
            self._timings["truth_build_sec"] = time.time() - t0
            logger.info(
                "Ground truth built in %.2f s  (%s queries x k=%s)",
                self._timings["truth_build_sec"],
                f"{cfg.num_query_vectors:,}", f"{cfg.truth_k:,}",
            )
        elif cfg.truth_mode == "flat_index":
            logger.info(
                "Building truth table from FLAT collection (k=%d) ...",
                cfg.truth_k,
            )
            t0 = time.time()
            self.truth_table = build_truth_from_flat(
                backend=self.backend,
                flat_collection_name=flat_name,
                query_vectors=self.query_vectors,
                truth_k=cfg.truth_k,
                metric_type=cfg.metric_type,
            )
            self._timings["truth_build_sec"] = time.time() - t0
            logger.info(
                "Ground truth (FLAT) built in %.2f s  (%s queries x k=%s)",
                self._timings["truth_build_sec"],
                f"{cfg.num_query_vectors:,}", f"{cfg.truth_k:,}",
            )

        return self._load_summary(total_inserted, blocks_consumed)

    # ------------------------------------------------------------------
    # Search phase
    # ------------------------------------------------------------------
    def _run_search(self) -> Dict[str, Any]:
        """Execute the search benchmark (blocking)."""
        cfg = self.cfg

        # ---- 1. Load query vectors + truth table -----------------------
        if self.query_vectors is None or self.truth_table is None:
            self._load_artifacts()

        # ---- 2. Build search params ------------------------------------
        search_params = cfg.search_params
        if not search_params:
            search_params = {
                "metric_type": cfg.metric_type,
                "params": {},
            }

        # ---- 3. Run the search benchmark -------------------------------
        runner = SearchRunner(
            backend=self.backend,
            collection_name=cfg.collection_name,
            query_vectors=self.query_vectors,
            truth_table=self.truth_table,
            search_k=cfg.search_k,
            search_params=search_params,
            metric_type=cfg.metric_type,
            num_rounds=cfg.num_search_rounds,
            batch_size=cfg.search_batch_size,
            log_interval=cfg.log_interval,
        )

        t0 = time.time()
        self.search_result = runner.run()
        self._timings["search_sec"] = time.time() - t0

        return self._search_summary()

    def _load_artifacts(self) -> None:
        """Load query vectors and truth table from a previous run."""
        d = self.cfg.artifacts_dir
        if not d:
            raise ValueError(
                "In 'search' mode, either run 'load' first (mode=both) "
                "or provide --artifacts-dir pointing to a previous run."
            )
        qpath = os.path.join(d, "query_vectors.npy")
        gtpath = os.path.join(d, "ground_truth.npz")

        if not os.path.isfile(qpath):
            raise FileNotFoundError(
                f"Expected artifacts not found in '{d}'.  "
                f"Looking for query_vectors.npy"
            )
        if self.cfg.truth_mode != "flat_index" and not os.path.isfile(gtpath):
            raise FileNotFoundError(
                f"Expected artifacts not found in '{d}'.  "
                f"Looking for ground_truth.npz (required when truth_mode != flat_index)"
            )

        self.query_vectors = np.load(qpath)

        if self.cfg.truth_mode == "flat_index":
            flat_name = f"{self.cfg.collection_name}_flat"
            self.truth_table = build_truth_from_flat(
                backend=self.backend,
                flat_collection_name=flat_name,
                query_vectors=self.query_vectors,
                truth_k=self.cfg.truth_k,
                metric_type=self.cfg.metric_type,
            )
        else:
            gt = np.load(gtpath)
            self.truth_table = gt["truth_table"]

        logger.info(
            "Loaded artifacts from '%s': queries=%s, truth=%s",
            d, self.query_vectors.shape, self.truth_table.shape,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_summary(self, total_inserted: int, blocks: int) -> Dict[str, Any]:
        return {
            "total_vectors_inserted": total_inserted,
            "blocks_processed": blocks,
            "num_query_vectors": self.cfg.num_query_vectors,
            "truth_k": self.cfg.truth_k,
            "truth_table_shape": list(self.truth_table.shape)
            if self.truth_table is not None
            else None,
            "timings": dict(self._timings),
        }

    def _search_summary(self) -> Dict[str, Any]:
        r = self.search_result
        if r is None:
            return {}
        return {
            "search_total_queries": r.total_queries,
            "search_qps": r.qps,
            "search_recall_at_k": r.recall_at_k,
            "search_latency_p50_ms": r.latency_p50_ms,
            "search_latency_p90_ms": r.latency_p90_ms,
            "search_latency_p99_ms": r.latency_p99_ms,
            "search_latency_mean_ms": r.latency_mean_ms,
            "search_wall_sec": r.total_wall_sec,
            "timings": dict(self._timings),
        }
