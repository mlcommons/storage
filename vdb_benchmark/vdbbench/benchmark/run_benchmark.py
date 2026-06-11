#!/usr/bin/env python3
"""CLI entry point for the producer-consumer vector-DB benchmark.

Usage examples::

    # List available backends
    python -m vdbbench.benchmark.run_benchmark help backends

    # Show detailed help for a specific backend
    python -m vdbbench.benchmark.run_benchmark help backend milvus

    # Run a benchmark (config-driven)
    python -m vdbbench.benchmark.run_benchmark --config configs/1m_hnsw.yaml

    # Override mode or backend on the CLI
    python -m vdbbench.benchmark.run_benchmark --config configs/1m_hnsw.yaml --mode both
    python -m vdbbench.benchmark.run_benchmark --config configs/1m_hnsw.yaml --backend pgvector

    # Dry-run (print resolved config and exit)
    python -m vdbbench.benchmark.run_benchmark --config configs/1m_hnsw.yaml --what-if

    # Direct script execution also works:
    python benchmark/run_benchmark.py help backend milvus

All dataset, index, search, and connection parameters are set in the YAML
config file.  The CLI is intentionally minimal -- only operational switches
(``--mode``, ``--backend``, ``--force``, ``--output-dir``, etc.) may be
given on the command line.
"""

from __future__ import annotations

import sys

# ------------------------------------------------------------------
# Direct-execution bootstrap.  When someone runs this file as a script
# (``python run_benchmark.py …``), Python sets __name__ = "__main__"
# and relative imports are impossible.  We detect that case *before*
# any relative imports, fix sys.path, re-import ourselves as a proper
# package member, and delegate to main().
# ------------------------------------------------------------------
if __name__ == "__main__":
    import importlib
    import pathlib

    _this = pathlib.Path(__file__).resolve()
    # …/vdb_benchmark/vdbbench/benchmark/run_benchmark.py
    # parent.parent.parent  →  …/vdb_benchmark   (contains vdbbench/)
    _pkg_root = str(_this.parent.parent.parent)
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)

    _mod = importlib.import_module("vdbbench.benchmark.run_benchmark")
    raise SystemExit(_mod.main())

# ------------------------------------------------------------------
# Normal imports (only reached when loaded as a package member).
# ------------------------------------------------------------------

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

import yaml

from .backends import registry, get_backend
from .backends._env import load_env_file, env_for_backend
from .backends._help import format_backend_help, format_backends_list
from .orchestrator import BenchmarkConfig, BenchmarkOrchestrator, MODES, TRUTH_MODES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# YAML helpers (mirrors existing config_loader.py pattern)
# ------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    """Try *path* directly, then under ``configs/``."""
    for candidate in [path, os.path.join("configs", path)]:
        if os.path.isfile(candidate):
            with open(candidate) as fh:
                cfg = yaml.safe_load(fh)
            logger.info("Loaded config from %s", candidate)
            return cfg or {}
    # Also try relative to this file's directory
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(pkg_dir, "configs", path)
    if os.path.isfile(candidate):
        with open(candidate) as fh:
            cfg = yaml.safe_load(fh)
        logger.info("Loaded config from %s", candidate)
        return cfg or {}
    logger.error("Config file not found: %s", path)
    return {}

# ------------------------------------------------------------------
# Help sub-commands
# ------------------------------------------------------------------

def _handle_help(argv: list[str]) -> bool:
    """If *argv* starts with ``help ...``, print the requested info
    and return ``True`` (meaning: handled, exit).  Otherwise return
    ``False``.
    """
    if not argv or argv[0].lower() != "help":
        return False

    rest = [a.lower() for a in argv[1:]]

    # help backends
    if rest == ["backends"]:
        print(format_backends_list(registry))
        return True

    # help backend <name>
    if len(rest) == 2 and rest[0] == "backend":
        print(format_backend_help(registry, rest[1]))
        return True

    # Bare "help" or unknown
    print("Usage:")
    print("  help backends            -- list all registered backends")
    print("  help backend <name>      -- show parameters for a backend")
    print()
    print(format_backends_list(registry))
    return True

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    available = ", ".join(registry.names()) or "(none)"
    p = argparse.ArgumentParser(
        description="Vector-DB benchmark: generate, ingest, build ground truth, and search",
        epilog=(
            "All dataset, index, search, and connection parameters live in "
            "the YAML config file.  Run 'help backends' or "
            "'help backend <name>' for backend-specific details."
        ),
    )

    # Config file (the primary input)
    p.add_argument("--config", type=str, required=False,
                    help="Path to YAML config file (required for benchmark runs)")

    # Operational overrides (take precedence over YAML values)
    p.add_argument(
        "--mode", type=str, dest="mode",
        choices=list(MODES),
        help="Override runtime mode: 'load', 'search', or 'both'",
    )
    p.add_argument(
        "--backend", type=str, dest="backend",
        help=f"Override backend ({available})",
    )
    p.add_argument("--force", action="store_true", default=None,
                    help="Drop collection if it already exists")
    p.add_argument("--output-dir", type=str, dest="output_dir",
                    help="Directory for artifacts (default: auto-timestamped)")
    p.add_argument("--artifacts-dir", type=str, dest="artifacts_dir",
                    help="Load query/truth artifacts from this directory "
                         "(required for --mode search without prior load)")

    # Introspection
    p.add_argument("--what-if", action="store_true",
                    help="Print resolved config and exit")
    p.add_argument("--plan", action="store_true",
                    help="Show the full execution plan (steps, sizes, "
                         "estimates) without running anything")
    p.add_argument("--debug", action="store_true",
                    help="Enable DEBUG logging")

    return p


def _merge_cli_over_yaml(yaml_cfg: dict, cli_ns: argparse.Namespace) -> dict:
    """Flatten YAML sections and overlay non-None CLI values."""
    flat: dict = {}
    for key, val in yaml_cfg.items():
        if isinstance(val, dict):
            flat.update(val)
        else:
            flat[key] = val

    skip = {"config", "what_if", "plan", "debug", "output_dir", "artifacts_dir"}
    for key, val in vars(cli_ns).items():
        if key in skip:
            continue
        if val is not None:
            flat[key] = val

    return flat


def _collect_index_params(flat: dict) -> dict:
    """Pull index-specific keys into the nested ``index_params`` dict."""
    ip = flat.get("index_params", {})
    if isinstance(ip, dict):
        ip = dict(ip)
    else:
        ip = {}
    for k in ("M", "efConstruction", "MaxDegree", "SearchListSize",
              "inline_pq", "max_degree", "search_list_size",
              "lists", "ef_search", "probes"):
        if k in flat and flat[k] is not None:
            ip[k] = flat[k]
    flat["index_params"] = ip
    return flat


def _resolve_backend_name(flat: dict, cli_ns: argparse.Namespace) -> str:
    """Determine which backend to use.

    Precedence: ``--backend`` CLI flag > ``backend`` key in YAML config
    > ``"milvus"`` (default).
    """
    if cli_ns.backend:
        return cli_ns.backend.lower()
    if "backend" in flat:
        return str(flat["backend"]).lower()
    return "milvus"


# ------------------------------------------------------------------
# Plan formatter
# ------------------------------------------------------------------

def _sizeof_fmt(num_bytes: float) -> str:
    """Human-readable byte size (e.g. ``5.86 GB``)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def _format_plan(cfg: BenchmarkConfig, desc) -> str:
    """Build a human-readable execution plan from *cfg* and the backend
    *desc* (:class:`BackendDescriptor`).  No database connection needed.
    """
    W = 64
    SEP = "-" * W
    lines: list[str] = []

    def heading(title: str) -> None:
        lines.append("")
        lines.append("=" * W)
        lines.append(f"  {title}")
        lines.append("=" * W)

    def step(num: int, title: str) -> None:
        lines.append("")
        lines.append(SEP)
        lines.append(f"  Step {num}: {title}")
        lines.append(SEP)

    def kv(key: str, val, indent: int = 4) -> None:
        pad = " " * indent
        lines.append(f"{pad}{key:<32s}: {val}")

    # -- Sizes -----------------------------------------------------------
    bytes_per_vector = cfg.dimension * 4            # float32
    db_vector_bytes = cfg.num_vectors * bytes_per_vector
    query_vector_bytes = cfg.num_query_vectors * bytes_per_vector
    # truth table: int64 ids per query
    truth_bytes = cfg.num_query_vectors * cfg.truth_k * 8
    num_blocks = math.ceil(cfg.num_vectors / cfg.block_size)
    inserts_per_block = math.ceil(cfg.block_size / cfg.batch_size)
    total_inserts = num_blocks * inserts_per_block

    # Ground-truth working memory: the builder keeps a running top-K
    # matrix of shape (num_queries, K) for IDs and distances (both float64).
    gt_working_bytes = cfg.num_query_vectors * cfg.truth_k * 8 * 2

    # Per-block GT compute: cosine/IP needs (num_queries x block_size)
    # distance matrix in float32.
    gt_block_bytes = cfg.num_query_vectors * cfg.block_size * 4

    # -- Header ----------------------------------------------------------
    heading("BENCHMARK EXECUTION PLAN")
    lines.append("")
    kv("Backend", f"{desc.display_name}  (--backend {desc.name})")
    kv("Mode", cfg.mode)
    kv("Collection", cfg.collection_name)
    kv("Force recreate", "yes" if cfg.force else "no")

    # -- Step 1: Query vector generation ---------------------------------
    step(1, "Generate query vectors")
    kv("Num query vectors", f"{cfg.num_query_vectors:,}")
    kv("Dimension", f"{cfg.dimension:,}")
    kv("Distribution", cfg.distribution)
    kv("Query seed", cfg.query_seed)
    kv("Memory", _sizeof_fmt(query_vector_bytes))
    kv("Output", "held in memory (saved to query_vectors.npy later)")

    # -- Step 2: Create collection + index -------------------------------
    step(2, "Create collection and index")
    kv("Index type", cfg.index_type)
    kv("Metric type", cfg.metric_type)
    kv("Num shards", cfg.num_shards)
    idx_desc = desc.get_index(cfg.index_type)
    if idx_desc and cfg.index_params:
        for p in idx_desc.build_params:
            val = cfg.index_params.get(p.name, p.default)
            kv(f"  {p.name}", val)
    elif idx_desc:
        for p in idx_desc.build_params:
            kv(f"  {p.name}", f"{p.default}  (default)")

    # -- Step 3: Vector generation + ingestion + GT ----------------------
    step(3, "Generate, ingest, and compute ground truth")
    lines.append("")
    lines.append("    Producer (background thread):")
    kv("Total database vectors", f"{cfg.num_vectors:,}")
    kv("Dimension", f"{cfg.dimension:,}")
    kv("Distribution", cfg.distribution)
    kv("Vector seed", cfg.seed)
    kv("Block size", f"{cfg.block_size:,} vectors")
    kv("Num blocks", f"{num_blocks:,}")
    kv("Queue depth", f"{cfg.max_queue_depth} blocks")
    kv("Per-block memory", _sizeof_fmt(cfg.block_size * bytes_per_vector))
    kv("Total vector data", _sizeof_fmt(db_vector_bytes))

    lines.append("")
    lines.append("    Consumer 1 -- Database ingestion:")
    kv("Batch size", f"{cfg.batch_size:,} vectors/insert")
    kv("Inserts per block", f"{inserts_per_block:,}")
    kv("Total insert calls", f"{total_inserts:,}")

    lines.append("")
    lines.append("    Consumer 2 -- Ground-truth builder:")
    kv("Query vectors", f"{cfg.num_query_vectors:,}")
    kv("K (neighbors)", f"{cfg.truth_k:,}")
    kv("Metric", cfg.metric_type)
    kv("Per-block distance matrix", _sizeof_fmt(gt_block_bytes))
    kv("Running top-K memory", _sizeof_fmt(gt_working_bytes))

    # -- Step 4: Flush ---------------------------------------------------
    step(4, "Flush collection")
    kv("Action", "commit pending writes to storage")

    # -- Step 5: Optional compaction -------------------------------------
    if cfg.compact:
        step(5, "Compact collection")
        kv("Action", "merge small segments before index build")
    else:
        lines.append("")
        lines.append(f"    (Step 5: Compact -- skipped, compact not set)")

    # -- Step 6: Wait for index build ------------------------------------
    step(6, "Wait for index build")
    kv("Poll interval", f"{cfg.monitor_interval}s")

    # -- Step 7: Finalize ground truth -----------------------------------
    step(7, "Finalize ground truth")
    kv("Truth table shape", f"({cfg.num_query_vectors:,}, {cfg.truth_k:,})")
    kv("Truth table size", _sizeof_fmt(truth_bytes))

    # -- Step 8: Save artifacts ------------------------------------------
    step(8, "Save artifacts")
    kv("query_vectors.npy", _sizeof_fmt(query_vector_bytes))
    kv("ground_truth.npz", f"~{_sizeof_fmt(truth_bytes + query_vector_bytes)}"
       "  (compressed)")
    kv("benchmark_meta.json", "config + timings")

    # -- Search steps (when mode is 'search' or 'both') ------------------
    mode = cfg.mode.lower()
    if mode in ("search", "both"):
        step(9, "Load collection into memory")
        kv("Collection", cfg.collection_name)
        kv("Action", "ensure collection is loaded for search")

        step(10, "Run search benchmark")
        kv("Search K (top-K)", cfg.search_k)
        kv("Query vectors", f"{cfg.num_query_vectors:,}")
        kv("Rounds", cfg.num_search_rounds)
        kv("Batch size", cfg.search_batch_size)
        kv("Log interval", f"every {cfg.log_interval} queries")
        kv("Truth K", cfg.truth_k)
        kv("Search params", cfg.search_params or "(backend defaults)")
        kv("Total queries", f"{cfg.num_query_vectors * cfg.num_search_rounds:,}")

    # -- Summary ---------------------------------------------------------
    heading("RESOURCE ESTIMATES")
    lines.append("")
    peak_mem = (
        query_vector_bytes                              # query vectors
        + cfg.max_queue_depth * cfg.block_size * bytes_per_vector  # queue
        + gt_working_bytes                              # GT top-K state
        + gt_block_bytes                                # GT distance matrix
    )
    kv("Peak memory (estimate)", _sizeof_fmt(peak_mem))
    kv("Total vector data generated", _sizeof_fmt(db_vector_bytes))
    kv("Disk artifacts (approx)", _sizeof_fmt(
        query_vector_bytes + truth_bytes + query_vector_bytes + 4096))
    lines.append("")

    return "\n".join(lines)
# Main
# ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    raw_argv = argv if argv is not None else sys.argv[1:]

    # No arguments at all → show usage and exit.
    if not raw_argv:
        _build_parser().print_help()
        print()
        print(format_backends_list(registry))
        return 0

    # Intercept "help" sub-commands before argparse runs.
    if _handle_help(raw_argv):
        return 0

    parser = _build_parser()
    args = parser.parse_args(raw_argv)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # A config file is required for any real work.
    if not args.config and not (args.what_if or args.plan):
        parser.error("--config is required (or use --what-if / --plan)")

    # Load .env file (if python-dotenv is installed and .env exists)
    load_env_file()

    # Build resolved config: defaults <- YAML <- CLI overrides
    yaml_cfg = _load_yaml(args.config) if args.config else {}
    flat = _merge_cli_over_yaml(yaml_cfg, args)
    flat = _collect_index_params(flat)

    # Inject CLI-only overrides that are not part of YAML sections
    if args.artifacts_dir is not None:
        flat["artifacts_dir"] = args.artifacts_dir

    # Resolve backend
    backend_name = _resolve_backend_name(flat, args)
    desc = registry.get(backend_name)
    if desc is None:
        available = ", ".join(registry.names()) or "(none)"
        parser.error(
            f"Unknown backend '{backend_name}'.  Available: {available}"
        )

    cfg = BenchmarkConfig.from_dict(flat)

    # --what-if: show config and exit
    if args.what_if:
        print(f"\nBackend: {desc.display_name}  (--backend {desc.name})")
        print("\nResolved benchmark configuration:")
        print("=" * 60)
        display = {k: v for k, v in cfg.to_dict().items()
                   if not (k == "compact" and v)}
        print(json.dumps(display, indent=2, default=str))
        print("=" * 60)

        # Show resolved connection parameters with sources
        _env = env_for_backend(backend_name, desc)
        if desc.connection_params:
            print("\nConnection parameters (source):")
            for p in desc.connection_params:
                k = p.name
                env_val = _env.get(k)
                yaml_val = flat.get(k)
                if env_val is not None:
                    print(f"  {k}: {env_val!r}  (env: {backend_name.upper()}__{k.upper()})")
                elif yaml_val is not None:
                    print(f"  {k}: {yaml_val!r}  (config)")
                else:
                    print(f"  {k}: {p.default!r}  (default)")
        return 0

    # --plan: show step-by-step execution plan and exit
    if args.plan:
        print(_format_plan(cfg, desc))
        return 0

    # Validate essentials
    mode = cfg.mode.lower()
    if mode in ("load", "both"):
        if not cfg.collection_name or not cfg.dimension or not cfg.num_vectors:
            parser.error(
                "collection_name, dimension, and num_vectors are required "
                "for load/both modes (set them in the config file)."
            )
    elif mode == "search":
        if not cfg.collection_name:
            parser.error(
                "collection_name is required for search mode "
                "(set it in the config file)."
            )
        if not cfg.artifacts_dir:
            parser.error(
                "--artifacts-dir is required for search mode to load "
                "query vectors and ground truth."
            )

    # Validate index type against backend capabilities
    if cfg.index_type and cfg.index_type.upper() not in (
        n.upper() for n in desc.index_names()
    ):
        parser.error(
            f"Backend '{desc.name}' does not support index type "
            f"'{cfg.index_type}'.  Supported: {', '.join(desc.index_names())}"
        )

    # Output directory
    output_dir = args.output_dir or os.path.join(
        "results",
        f"{cfg.collection_name}_{datetime.now():%Y%m%d_%H%M%S}",
    )

    # Connect backend.
    # Precedence: environment variables (.env / shell) > YAML config > defaults
    backend = desc.backend_class()
    env_kwargs = env_for_backend(backend_name, desc)
    conn_kwargs: dict = {}
    for p in desc.connection_params:
        k = p.name
        env_val = env_kwargs.get(k)              # env var / .env file
        yaml_val = flat.get(k)                   # YAML config
        if env_val is not None:
            conn_kwargs[k] = env_val
        elif yaml_val is not None:
            conn_kwargs[k] = yaml_val
        # else: omitted → backend.connect() uses its own default
    backend.connect(**conn_kwargs)

    try:
        orch = BenchmarkOrchestrator(config=cfg, backend=backend)
        summary = orch.run()
        paths = orch.save(output_dir)

        mode = cfg.mode.lower()

        print("\n" + "=" * 60)
        print(f"BENCHMARK COMPLETE  (backend: {desc.display_name}, mode: {mode})")
        print("=" * 60)

        if mode in ("load", "both"):
            print(f"  Vectors inserted : {summary.get('total_vectors_inserted', 'N/A'):,}")
            print(f"  Query vectors    : {cfg.num_query_vectors:,}")
            print(f"  Truth table      : {summary.get('truth_table_shape', 'N/A')}")
            print(f"  Truth mode       : {cfg.truth_mode}")

        if mode in ("search", "both"):
            print(f"\n  --- Search Results ---")
            print(f"  Total queries    : {summary.get('search_total_queries', 'N/A'):,}")
            print(f"  QPS              : {summary.get('search_qps', 0):.1f}")
            print(f"  Recall@{cfg.search_k:<9d}: {summary.get('search_recall_at_k', 0):.4f}")
            print(f"  Latency P50      : {summary.get('search_latency_p50_ms', 0):.2f} ms")
            print(f"  Latency P90      : {summary.get('search_latency_p90_ms', 0):.2f} ms")
            print(f"  Latency P99      : {summary.get('search_latency_p99_ms', 0):.2f} ms")
            print(f"  Latency mean     : {summary.get('search_latency_mean_ms', 0):.2f} ms")
            print(f"  Wall time        : {summary.get('search_wall_sec', 0):.2f} s")

        print(f"\n  Output dir       : {output_dir}")
        for name, p in paths.items():
            print(f"    {name:20s} -> {p}")
        print("=" * 60)
        print("\nTimings:")
        for k, v in summary.get("timings", {}).items():
            print(f"  {k:30s} : {v:>10.2f} s")
        print()

    finally:
        backend.disconnect()

    return 0
