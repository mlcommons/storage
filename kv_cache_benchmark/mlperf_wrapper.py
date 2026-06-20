#!/usr/bin/env python3
"""MPI-rank-aware wrapper for MLPerf KV Cache benchmark.

Invoked by mpirun per-rank; reads OMPI_COMM_WORLD_RANK (OpenMPI) or PMI_RANK
(MPICH) to determine this rank's index. Computes per-rank seed, output
directory, and cache directory, then invokes kv-cache.py with the fixed MLPerf
v3.0 parameters for the specified option.
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

# MLPerf v3.0 fixed parameters per option.
# All numeric values stored as int/float; converted to str when building cmd.
# 'generation-mode' is ALWAYS 'none' for MLPerf compliance — do NOT rely on
# kv-cache.py defaults; the default may change in future versions.
WORKLOAD_PARAMS = {
    1: {
        'model': 'llama3.1-8b',
        'num-users': 200,
        'duration': 300,
        'gpu-mem-gb': 0,
        'cpu-mem-gb': 0,
        'max-concurrent-allocs': 16,
        'generation-mode': 'none',
    },
    2: {
        'model': 'llama3.1-8b',
        'num-users': 100,
        'duration': 300,
        'gpu-mem-gb': 0,
        'cpu-mem-gb': 4,
        'max-concurrent-allocs': 16,
        'generation-mode': 'none',
    },
    3: {
        'model': 'llama3.1-70b-instruct',
        'num-users': 70,
        'duration': 300,
        'gpu-mem-gb': 0,
        'cpu-mem-gb': 0,
        'max-concurrent-allocs': 4,
        'generation-mode': 'none',
    },
}


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
    parser = argparse.ArgumentParser(
        description="MLPerf KV Cache MPI-rank-aware wrapper"
    )
    parser.add_argument(
        '--option',
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="MLPerf v3.0 option (1=Max Storage Stress, 2=Storage Throughput, 3=Large Model 70B).",
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=BASE_SEED,
        help=f"Base random seed (default: {BASE_SEED}). Effective seed = base + rank.",
    )
    parser.add_argument(
        '--base-output-dir',
        type=str,
        required=True,
        dest='base_output_dir',
        help="Base output directory. Per-rank results written to <base_output_dir>/rank_<N>/.",
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        required=True,
        dest='cache_dir',
        help="Base cache directory. Per-rank cache written to <cache_dir>/rank_<N>/.",
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="Path to YAML config file. Defaults to config.yaml adjacent to this script (D-02).",
    )

    args = parser.parse_args()

    rank = get_rank()
    effective_seed = args.seed + rank

    rank_output_dir = Path(args.base_output_dir) / f"rank_{rank}"
    rank_cache_dir = Path(args.cache_dir) / f"rank_{rank}"

    rank_output_dir.mkdir(parents=True, exist_ok=True)
    rank_cache_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = rank_output_dir / f"kvcache_results_{ts}.json"

    # D-01: kv-cache.py located relative to this script; both share kv_cache_benchmark/
    kvcache_script = Path(__file__).parent / 'kv-cache.py'
    # D-02: config.yaml located relative to this script; user may override via --config
    config_path = args.config or str(Path(__file__).parent / 'config.yaml')

    params = WORKLOAD_PARAMS[args.option]

    cmd = [
        sys.executable,
        str(kvcache_script),
        '--config', config_path,
        '--seed', str(effective_seed),
        '--output', str(output_file),
        '--cache-dir', str(rank_cache_dir),
        '--model', params['model'],
        '--num-users', str(params['num-users']),
        '--duration', str(params['duration']),
        '--gpu-mem-gb', str(params['gpu-mem-gb']),
        '--cpu-mem-gb', str(params['cpu-mem-gb']),
        '--max-concurrent-allocs', str(params['max-concurrent-allocs']),
        # '--generation-mode' is passed EXPLICITLY — do NOT omit even though the
        # value is always 'none'. kv-cache.py defaults may change in future versions.
        '--generation-mode', params['generation-mode'],
    ]

    print(f"KV Cache Wrapper - Start delay for {TEST_DELAY} seconds")
    time.sleep(TEST_DELAY)
    print(f"KV Cache Wrapper - Starting benchmark pass...")

    result = subprocess.run(cmd)
    print(f"KV Cache Wrapper - End delay for {TEST_DELAY} seconds")
    time.sleep(TEST_DELAY)
    print(f"KV Cache Wrapper - Finished benchmark pass")

    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
