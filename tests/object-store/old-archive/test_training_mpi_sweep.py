#!/usr/bin/env python3
"""
Training MPI Process Count Sweep

For every (library, N) combination, runs a COMPLETE cycle:
  1. Cleanup — delete any leftover objects
  2. Datagen  — generate 100 × 128 MiB NPZ files with N parallel write processes
  3. Train    — read the dataset across 2 epochs with N MPI accelerators
  4. Cleanup  — delete the objects for this run

This means datagen is also under test at each N — both write (datagen) and read
(training) throughput are measured at the same process count.

Libraries:   s3dlio, minio, s3torchconnector  (or a subset via --library)
Process counts (N):  1, 2, 4                   (or custom via --process-counts)

Hypothesis being tested:
  Prior runs at 1 accelerator produced ~0.178 GB/s read throughput despite a
  ~1.2 GB/s network ceiling.  The question is whether:
    (a) More MPI processes help by adding independent read pipelines, OR
    (b) The per-process NPZ deserialise + DataLoader IPC pickle dominates regardless.

Usage:
  # All libraries, 1/2/4 process counts (default)
  python test_training_mpi_sweep.py

  # Single library
  python test_training_mpi_sweep.py --library s3dlio

  # Custom process count sweep
  python test_training_mpi_sweep.py --process-counts 1 2 4 8

  # Quick test: skip datagen phase (requires data already in bucket)
  python test_training_mpi_sweep.py --skip-datagen

  # Keep objects after run
  python test_training_mpi_sweep.py --skip-cleanup
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────────

DEFAULT_LIBRARIES      = ['s3dlio', 'minio', 's3torchconnector']
DEFAULT_PROCESS_COUNTS = [1, 2, 4]

LIBRARY_BUCKETS = {
    's3dlio':           'bucket-s3dlio',
    'minio':            'bucket-minio',
    's3torchconnector': 'bucket-s3torch',
}

# Training dataset parameters
TRAIN_MODEL        = 'unet3d'
TRAIN_ACCEL_TYPE   = 'a100'
TRAIN_NUM_FILES    = 100
TRAIN_SIZE_MiB     = 128
TRAIN_RECORD_BYTES = TRAIN_SIZE_MiB * 1024 * 1024   # 134,217,728
TRAIN_SAMPLES_PER  = 1
TRAIN_EPOCHS       = 2
TRAIN_PREFIX       = 'dlio-train'

# Per-training-run I/O settings (constant across sweep)
READ_THREADS   = 8
PREFETCH_SIZE  = 4
BATCH_SIZE     = 1

CLIENT_MEM_GB  = 32
RESULTS_DIR    = '/tmp/dlio_mpi_sweep'
PAUSE_SECONDS  = 30


# ── Credentials ──────────────────────────────────────────────────────────────────

def load_env_config() -> dict:
    env_path = None
    for candidate in [
        Path(__file__).parent.parent / '.env',
        Path(__file__).parent / '.env',
        Path.cwd() / '.env',
    ]:
        if candidate.exists():
            env_path = candidate
            break

    config = {}
    if env_path:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, val = line.partition('=')
                    config[key.strip()] = val.strip()
        print(f'Loaded credentials from: {env_path}')
    else:
        print('No .env file found — using environment variables only')

    for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_ENDPOINT_URL', 'AWS_REGION']:
        if key in os.environ:
            config[key] = os.environ[key]

    return config


def build_env(config: dict, library: str) -> dict:
    env = os.environ.copy()
    env.update(config)
    env['STORAGE_LIBRARY'] = library
    return env


# ── Subprocess helpers ────────────────────────────────────────────────────────────

def pause(seconds: int, reason: str):
    print(f'\n  Sleeping {seconds}s — {reason}')
    sys.stdout.flush()
    time.sleep(seconds)


def clean_prefix(bucket: str, prefix: str, env: dict):
    uri = f's3://{bucket}/{prefix}/'
    result = subprocess.run(
        ['s3-cli', 'delete', '-r', uri],
        env=env, capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f'    Cleaned s3://{bucket}/{prefix}/')
    else:
        print(f'    (nothing to clean at s3://{bucket}/{prefix}/)')


def list_prefix(bucket: str, prefix: str, env: dict, label: str = ''):
    uri = f's3://{bucket}/{prefix}/'
    result = subprocess.run(
        ['s3-cli', 'list', uri],
        env=env, capture_output=True, text=True,
    )
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    tag = f' [{label}]' if label else ''
    if lines:
        print(f'    s3-cli list {uri}{tag}: {len(lines)} object(s)')
        for l in lines[:5]:
            print(f'      {l}')
        if len(lines) > 5:
            print(f'      ... ({len(lines) - 5} more)')
    else:
        print(f'    s3-cli list {uri}{tag}: (empty)')


def run_phase(label: str, cmd: list, env: dict, timeout_s: int = 3600) -> tuple:
    """Stream subprocess output live. Returns (returncode, elapsed_seconds, captured_output)."""
    print(f'\n  $ {" ".join(cmd[:8])} {"..." if len(cmd) > 8 else ""}')
    t_start = time.perf_counter()
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    captured_lines = []
    try:
        for line in proc.stdout:
            sys.stdout.write(f'    {line}')
            sys.stdout.flush()
            captured_lines.append(line)
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        elapsed = time.perf_counter() - t_start
        print(f'\n  ❌ {label} timed out after {elapsed:.0f}s')
        return -1, elapsed, ''.join(captured_lines)

    elapsed = time.perf_counter() - t_start
    if proc.returncode == 0:
        print(f'  ✅ {label}: done in {elapsed:.1f}s')
    else:
        print(f'  ❌ {label}: FAILED (exit {proc.returncode}) after {elapsed:.1f}s')
    return proc.returncode, elapsed, ''.join(captured_lines)


# ── Storage params builder ────────────────────────────────────────────────────────

def build_storage_params(config: dict, library: str) -> list:
    bucket      = LIBRARY_BUCKETS[library]
    data_folder = f's3://{bucket}/{TRAIN_PREFIX}'
    region      = config.get('AWS_REGION', 'us-east-1')
    return [
        f'storage.storage_type=s3',
        f'storage.storage_root={bucket}',
        f'storage.storage_options.endpoint_url={config["AWS_ENDPOINT_URL"]}',
        f'storage.storage_options.access_key_id={config["AWS_ACCESS_KEY_ID"]}',
        f'storage.storage_options.secret_access_key={config["AWS_SECRET_ACCESS_KEY"]}',
        f'storage.storage_options.region={region}',
        f'storage.storage_options.s3_force_path_style=true',
        f'dataset.data_folder={data_folder}',
        f'dataset.num_files_train={TRAIN_NUM_FILES}',
        f'dataset.num_samples_per_file={TRAIN_SAMPLES_PER}',
        f'dataset.record_length={TRAIN_RECORD_BYTES}',
        f'dataset.format=npz',
    ]


# ── Single (library, N) cycle ────────────────────────────────────────────────────

def run_one_cycle(library: str, n: int, config: dict,
                  skip_datagen: bool, skip_cleanup: bool) -> dict:
    """
    Full cycle for one (library, process_count) pair:
      clean → datagen(N) → pause → train(N) → clean

    Returns a result dict with gen_gbps, run_gbps, gen_ok, run_ok.
    """
    bucket         = LIBRARY_BUCKETS[library]
    env            = build_env(config, library)
    total_gb       = TRAIN_NUM_FILES * TRAIN_SIZE_MiB / 1024.0
    read_total_gb  = total_gb * TRAIN_EPOCHS
    storage_params = build_storage_params(config, library)

    result = {
        'library':       library,
        'num_processes': n,
        'gen_ok':        False,
        'run_ok':        False,
        'gen_gbps':      None,
        'run_gbps':      None,
        'gen_time':      0.0,
        'run_time':      0.0,
        'dataset_gb':    total_gb,
        'epochs':        TRAIN_EPOCHS,
    }

    print(f'\n{"─"*72}')
    print(f'  [{library}]  N={n}  |  s3://{bucket}/{TRAIN_PREFIX}/')
    print(f'{"─"*72}')

    try:
        # ── Cleanup before ──────────────────────────────────────────────────
        if not skip_datagen:
            print('\n  Step 1: Cleanup (pre-run)')
            clean_prefix(bucket, TRAIN_PREFIX, env)

        # ── Datagen ─────────────────────────────────────────────────────────
        if skip_datagen:
            print(f'\n  Step 1: Skipping datagen — using existing data')
            list_prefix(bucket, TRAIN_PREFIX, env, 'existing')
            result['gen_ok'] = True
        else:
            print(f'\n  Step 2: datagen — {TRAIN_NUM_FILES} × {TRAIN_SIZE_MiB} MiB, '
                  f'{n} process(es)')
            datagen_flags = [
                '--model', TRAIN_MODEL,
                '--num-processes', str(n),
                '--open',
                '--skip-validation',
                '--results-dir', RESULTS_DIR,
            ]
            rc_gen, t_gen, _ = run_phase(
                f'datagen (N={n})',
                ['mlpstorage', 'training', 'datagen'] + datagen_flags
                    + ['--params'] + storage_params,
                env,
            )
            result['gen_ok']   = (rc_gen == 0)
            result['gen_time'] = t_gen
            if result['gen_ok']:
                result['gen_gbps'] = total_gb / t_gen if t_gen > 0 else None
                list_prefix(bucket, TRAIN_PREFIX, env, 'after datagen')
                pause(PAUSE_SECONDS, 'S3 eventual consistency before training read')
            else:
                print(f'  ❌ datagen failed — skipping training read for this cycle')
                return result

        # ── Training read ────────────────────────────────────────────────────
        print(f'\n  Step 3: training run — {TRAIN_EPOCHS} epochs × {total_gb:.2f} GiB, '
              f'{n} accelerator(s), {READ_THREADS} read threads each')
        run_flags = [
            '--model', TRAIN_MODEL,
            '--num-accelerators', str(n),
            '--accelerator-type', TRAIN_ACCEL_TYPE,
            '--client-host-memory-in-gb', str(CLIENT_MEM_GB),
            '--open',
            '--skip-validation',
            '--results-dir', RESULTS_DIR,
        ]
        rc_run, t_run, _ = run_phase(
            f'train (N={n})',
            ['mlpstorage', 'training', 'run'] + run_flags + ['--params'] + storage_params + [
                f'train.epochs={TRAIN_EPOCHS}',
                f'train.batch_size={BATCH_SIZE}',
                f'reader.batch_size={BATCH_SIZE}',
                f'reader.read_threads={READ_THREADS}',
                f'reader.prefetch_size={PREFETCH_SIZE}',
            ],
            env,
        )
        result['run_ok']   = (rc_run == 0)
        result['run_time'] = t_run
        if result['run_ok']:
            result['run_gbps'] = read_total_gb / t_run if t_run > 0 else None

    finally:
        # ── Cleanup after ───────────────────────────────────────────────────
        if not skip_cleanup:
            print(f'\n  Step 4: Cleanup (post-run)')
            clean_prefix(bucket, TRAIN_PREFIX, env)
            list_prefix(bucket, TRAIN_PREFIX, env, 'after cleanup')
        else:
            print(f'\n  Skipping cleanup (--skip-cleanup)')

    status = '✅' if result['run_ok'] else '❌'
    w_s = f"{result['gen_gbps']:.3f} GB/s write" if result.get('gen_gbps') else 'write skipped'
    r_s = f"{result['run_gbps']:.3f} GB/s read"  if result.get('run_gbps') else 'read FAILED'
    print(f'\n  {status}  [{library}] N={n}: {w_s}  |  {r_s}')
    return result


# ── Results tables ────────────────────────────────────────────────────────────────

def print_results(all_results: list, process_counts: list):
    print()
    print('=' * 100)
    print('TRAINING MPI PROCESS SWEEP — RESULTS')
    print('=' * 100)
    print()

    total_gb   = TRAIN_NUM_FILES * TRAIN_SIZE_MiB / 1024.0
    read_total = total_gb * TRAIN_EPOCHS
    print(f'Dataset : {TRAIN_NUM_FILES} × {TRAIN_SIZE_MiB} MiB = {total_gb:.2f} GiB per library')
    print(f'Reads   : {TRAIN_EPOCHS} epochs = {read_total:.2f} GiB total per cycle')
    print(f'I/O     : {READ_THREADS} read_threads per MPI process, prefetch {PREFETCH_SIZE}')
    print(f'Cycle   : clean → datagen(N) → train(N) → clean  (independent for each N)')
    print()

    libraries_seen = []
    by_lib = {}
    for r in all_results:
        lib = r['library']
        if lib not in by_lib:
            by_lib[lib] = {}
            libraries_seen.append(lib)
        by_lib[lib][r['num_processes']] = r

    count_headers = '  '.join(f'  N={n}' for n in process_counts)
    sep = '-' * (26 + len(process_counts) * 12)

    # ── Write throughput ───────────────────────────────────────────────────
    print(f'  Datagen write throughput (GB/s):')
    print(f'  {"Library":<24}  {count_headers}')
    print(f'  {sep}')
    for lib in libraries_seen:
        cols = []
        for n in process_counts:
            r = by_lib.get(lib, {}).get(n)
            if r is None:
                cols.append('    N/A')
            elif not r.get('gen_ok'):
                cols.append('   FAIL')
            elif r.get('gen_gbps') is None:
                cols.append('   skip')
            else:
                cols.append(f'{r["gen_gbps"]:>7.3f}')
        print(f'  {lib:<24}  ' + '        '.join(cols))
    print()

    # ── Read throughput ────────────────────────────────────────────────────
    print(f'  Training read throughput (GB/s):')
    print(f'  {"Library":<24}  {count_headers}')
    print(f'  {sep}')
    for lib in libraries_seen:
        cols = []
        for n in process_counts:
            r = by_lib.get(lib, {}).get(n)
            if r is None:
                cols.append('    N/A')
            elif not r.get('run_ok'):
                cols.append('   FAIL')
            else:
                cols.append(f'{r["run_gbps"]:>7.3f}' if r.get('run_gbps') else '    N/A')
        print(f'  {lib:<24}  ' + '        '.join(cols))
    print()

    # ── Scaling vs N=1 ─────────────────────────────────────────────────────
    if 1 in process_counts:
        print(f'  Read scaling relative to N=1:')
        print(f'  {"Library":<24}  {count_headers}')
        print(f'  {sep}')
        for lib in libraries_seen:
            lib_data = by_lib.get(lib, {})
            baseline = lib_data.get(1, {}).get('run_gbps')
            cols = []
            for n in process_counts:
                gbps = lib_data.get(n, {}).get('run_gbps')
                if gbps is None:
                    cols.append('    N/A')
                elif n == 1:
                    cols.append(f'{gbps:.3f}  ')
                elif baseline:
                    cols.append(f'{gbps / baseline:.2f}×   ')
                else:
                    cols.append(f'{gbps:.3f}  ')
            print(f'  {lib:<24}  ' + '        '.join(cols))
        print()

    print('  Interpretation:')
    print('  - ratio > 1.0×: more processes increase throughput (additional I/O pipelines)')
    print('  - ratio ≈ 1.0×: MPI process count is not the bottleneck')
    print('  - ratio < 1.0×: more processes hurt (contention or Python overhead dominates)')
    print()
    print('=' * 100)


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DLIO training sweep: process count for datagen + training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_training_mpi_sweep.py                               # all libs, N=1,2,4
  python test_training_mpi_sweep.py --library s3dlio              # one library
  python test_training_mpi_sweep.py --process-counts 1 2 4 8     # extended sweep
  python test_training_mpi_sweep.py --skip-datagen                # skip write phase
  python test_training_mpi_sweep.py --skip-cleanup                # keep objects
        """,
    )
    parser.add_argument(
        '--library', choices=['s3dlio', 'minio', 's3torchconnector'],
        nargs='+', dest='libraries', metavar='LIBRARY',
        help='Library/libraries to sweep (default: all three)',
    )
    parser.add_argument(
        '--process-counts', type=int, nargs='+', default=DEFAULT_PROCESS_COUNTS,
        metavar='N',
        help=f'N values to sweep for both datagen and training (default: {DEFAULT_PROCESS_COUNTS})',
    )
    parser.add_argument(
        '--skip-datagen', action='store_true',
        help='Skip datagen — use data already present in the bucket',
    )
    parser.add_argument(
        '--skip-cleanup', action='store_true',
        help='Do not delete training data after each cycle',
    )
    args = parser.parse_args()

    libraries      = args.libraries or DEFAULT_LIBRARIES
    process_counts = sorted(set(args.process_counts))

    config = load_env_config()
    for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_ENDPOINT_URL']:
        if not config.get(key):
            print(f'ERROR: {key} not set in .env or environment', file=sys.stderr)
            sys.exit(1)

    import shutil
    if not shutil.which('mlpstorage'):
        print('ERROR: mlpstorage not found in PATH. Activate the virtualenv first.',
              file=sys.stderr)
        sys.exit(1)

    total_gb   = TRAIN_NUM_FILES * TRAIN_SIZE_MiB / 1024.0
    n_cycles   = len(libraries) * len(process_counts)

    print()
    print('=' * 100)
    print('TRAINING MPI PROCESS SWEEP')
    print('=' * 100)
    print(f'  Endpoint:       {config["AWS_ENDPOINT_URL"]}')
    print(f'  Libraries:      {", ".join(libraries)}')
    print(f'  Process counts: {process_counts}')
    print(f'  Total cycles:   {n_cycles}  ({len(libraries)} libs × {len(process_counts)} N values)')
    print(f'  Dataset:        {TRAIN_NUM_FILES} × {TRAIN_SIZE_MiB} MiB = {total_gb:.2f} GiB/library')
    print(f'  Cycle:          {"datagen SKIPPED — existing data" if args.skip_datagen else "clean → datagen(N) → train(N) → clean"}')
    print(f'  I/O:            {READ_THREADS} read threads per process, prefetch {PREFETCH_SIZE}')
    print('=' * 100)

    all_results = []

    for lib in libraries:
        for n in process_counts:
            if all_results:
                pause(PAUSE_SECONDS, 'cooldown before next cycle')

            result = run_one_cycle(
                library      = lib,
                n            = n,
                config       = config,
                skip_datagen = args.skip_datagen,
                skip_cleanup = args.skip_cleanup,
            )
            all_results.append(result)

    print_results(all_results, process_counts)

    failed = [r for r in all_results if not r['run_ok']]
    if not failed:
        print('✅ All training runs succeeded.')
        sys.exit(0)
    else:
        names = [f'{r["library"]} N={r["num_processes"]}' for r in failed]
        print(f'❌ Failed: {", ".join(names)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
