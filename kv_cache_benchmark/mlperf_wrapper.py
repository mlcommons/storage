#!/usr/bin/env python3
"""MPI-rank-aware launcher for the MLPerf KV Cache benchmark.

Invoked by mpirun per-rank; reads OMPI_COMM_WORLD_RANK (OpenMPI) or PMI_RANK
(MPICH) to determine this rank's index. The wrapper does NOT encode any
workload parameters — it only computes per-rank values (seed, output file,
cache directory) and forwards everything else to kv-cache.py. The caller
(mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark) owns the per-option
workload parameter set and CLOSED/OPEN enforcement.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_SEED = 42
TEST_DELAY = 90


def get_rank() -> int:
    """Read global MPI rank from environment (no mpi4py).

    Returns:
        MPI rank (0-based). Falls back to 0 for non-MPI / single-process runs.
    """
    # Open MPI v4+ uses OMPI_COMM_WORLD_RANK
    rank_str = os.environ.get('OMPI_COMM_WORLD_RANK')
    if rank_str:
        try:
            return int(rank_str)
        except ValueError:
            pass

    # MPICH uses PMI_RANK
    rank_str = os.environ.get('PMI_RANK')
    if rank_str:
        try:
            return int(rank_str)
        except ValueError:
            pass

    return 0  # single-process / non-MPI execution


def main():
    # allow_abbrev=False so forwarded kv-cache.py flags like --seed are not
    # silently swallowed by a prefix match against --seed-base.
    parser = argparse.ArgumentParser(
        description="MLPerf KV Cache MPI-rank-aware launcher",
        allow_abbrev=False,
    )
    parser.add_argument(
        '--rank-output-base',
        type=str,
        required=True,
        dest='rank_output_base',
        help="Base output directory. Per-rank results written to <rank-output-base>/rank_<N>/.",
    )
    parser.add_argument(
        '--rank-cache-base',
        type=str,
        required=True,
        dest='rank_cache_base',
        help="Base cache directory. Per-rank cache written to <rank-cache-base>/rank_<N>/.",
    )
    parser.add_argument(
        '--seed-base',
        type=int,
        default=BASE_SEED,
        dest='seed_base',
        help=f"Base random seed (default: {BASE_SEED}). Effective seed = base + rank.",
    )
    parser.add_argument(
        '--start-delay',
        type=int,
        default=TEST_DELAY,
        dest='start_delay',
        help=f"Seconds to sleep before invoking kv-cache.py (default: {TEST_DELAY}).",
    )
    parser.add_argument(
        '--end-delay',
        type=int,
        default=TEST_DELAY,
        dest='end_delay',
        help=f"Seconds to sleep after kv-cache.py exits (default: {TEST_DELAY}).",
    )

    # Wrapper-specific flags are consumed; everything else is forwarded to
    # kv-cache.py verbatim (including --config, --model, --num-users, etc.).
    args, forwarded = parser.parse_known_args()

    rank = get_rank()
    effective_seed = args.seed_base + rank

    rank_output_dir = Path(args.rank_output_base) / f"rank_{rank}"
    rank_cache_dir = Path(args.rank_cache_base) / f"rank_{rank}"

    rank_output_dir.mkdir(parents=True, exist_ok=True)
    rank_cache_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = rank_output_dir / f"kvcache_results_{ts}.json"

    kvcache_script = Path(__file__).parent / 'kv-cache.py'

    # Per-rank seed/output/cache are appended last so argparse store-action
    # uses them over any duplicates that came in via --forwarded args.
    cmd = [
        sys.executable,
        str(kvcache_script),
        *forwarded,
        '--seed', str(effective_seed),
        '--output', str(output_file),
        '--cache-dir', str(rank_cache_dir),
    ]

    print(f"KV Cache Wrapper - Start delay for {args.start_delay} seconds")
    time.sleep(args.start_delay)
    print(f"KV Cache Wrapper - Starting benchmark pass...")

    result = subprocess.run(cmd)
    print(f"KV Cache Wrapper - End delay for {args.end_delay} seconds")
    time.sleep(args.end_delay)
    print(f"KV Cache Wrapper - Finished benchmark pass")

    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
