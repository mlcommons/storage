#!/usr/bin/env python3
"""
DLIO Multi-Library Benchmark Demo

Demonstrates two DLIO-driven workloads across s3dlio, minio, and s3torchconnector.
I/O is handled by DLIO (via mlpstorage), NOT by the direct native APIs — this is
specifically to show how each library performs when used as DLIO's storage backend.

Workload 1 — TRAINING
  Phase 0: cleanup  — delete existing dlio-train/* objects from the library's bucket
  Phase 1: datagen  — DLIO generates 100 × 128 MiB NPZ objects and writes them to S3
  Phase 2: train    — DLIO reads all objects over 2 full epochs

Workload 2 — CHECKPOINT
  Model: llama3-8b, 8 simulated ranks, open mode → ~105 GB / ~97.8 GiB total.
  (Closest standard DLIO model configuration to the 128 GiB target.)
  Phase 0: cleanup  — delete existing dlio-ckpt/* objects from the library's bucket
  Phase 1: save     — DLIO writes 1 checkpoint (8 rank shards × ~13.12 GB each)
  Phase 2: restore  — DLIO reads the checkpoint back

Credentials are loaded from mlp-storage/.env (same as other test scripts in this folder).
Each library uses its own dedicated S3 bucket to avoid interference.

Usage:
  # All libraries, both workloads (default)
  python test_dlio_multilib_demo.py

  # Single workload
  python test_dlio_multilib_demo.py --workload training
  python test_dlio_multilib_demo.py --workload checkpoint

  # Specific library/libraries
  python test_dlio_multilib_demo.py --library s3dlio
  python test_dlio_multilib_demo.py --library s3dlio minio

  # Combine flags
  python test_dlio_multilib_demo.py --workload training --library s3dlio minio
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

# ── Configuration ───────────────────────────────────────────────────────────────

DEFAULT_LIBRARIES = ['s3dlio', 'minio', 's3torchconnector']

LIBRARY_BUCKETS = {
    's3dlio':           os.environ.get('BUCKET_S3DLIO', 'bucket-s3dlio'),
    'minio':            os.environ.get('BUCKET_MINIO', 'bucket-minio'),
    's3torchconnector': os.environ.get('BUCKET_S3TORCH', 'bucket-s3torch'),
}

# Workload 1 — Training
TRAIN_MODEL         = 'unet3d'
TRAIN_NUM_ACCEL     = 1
TRAIN_ACCEL_TYPE    = 'a100'
TRAIN_NUM_FILES     = 100
TRAIN_SIZE_MiB      = 128
TRAIN_RECORD_BYTES  = TRAIN_SIZE_MiB * 1024 * 1024   # 134,217,728
TRAIN_SAMPLES_PER   = 1                               # 1 sample = 1 file
TRAIN_EPOCHS        = 2
TRAIN_PREFIX        = 'dlio-train'

# Workload 2 — Checkpoint
# StreamingCheckpointing uses a fixed 128 MB buffer pool regardless of checkpoint size.
# ~100 GB single-object checkpoint per library.  At ~0.5 GB/s → ~200s per library.
CKPT_SIZE_GB        = 16.0           # single streaming object per library
CKPT_CHUNK_MB       = 32            # 32 MB chunks
CKPT_NUM_BUFFERS    = 4             # 4 buffers × 32 MB = 128 MB RAM max
CKPT_PREFIX         = 'dlio-ckpt'

# Per-library checkpoint size overrides.
# s3torchconnector fails at ~78 GB due to a CRT multipart bug.
# Re-add {'s3torchconnector': 75.0} here if CKPT_SIZE_GB is raised back toward 100 GB.
CKPT_SIZE_GB_OVERRIDE = {}

# Shared
CLIENT_MEM_GB   = 32
RESULTS_DIR     = '/tmp/dlio_multilib_demo'
PAUSE_SECONDS   = 30                # wait for S3 eventual consistency between phases


# ── Credentials ─────────────────────────────────────────────────────────────────

def load_env_config() -> dict:
    """Load .env file then let actual env vars override."""
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
    """Subprocess environment: current env + credentials + STORAGE_LIBRARY."""
    env = os.environ.copy()
    env.update(config)
    env['STORAGE_LIBRARY'] = library
    return env


# ── Subprocess helpers ───────────────────────────────────────────────────────────

def pause(seconds: int, reason: str):
    """Sleep with a simple one-line message."""
    print(f'\n  Sleeping {seconds}s — {reason}')
    sys.stdout.flush()
    time.sleep(seconds)


import contextlib

@contextlib.contextmanager
def _s3_env(config: dict):
    """Temporarily apply S3 credentials to os.environ for in-process s3dlio calls."""
    keys = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
            'AWS_ENDPOINT_URL', 'AWS_ENDPOINT_URL_S3', 'AWS_REGION']
    old = {k: os.environ.get(k) for k in keys}
    if config.get('AWS_ACCESS_KEY_ID'):
        os.environ['AWS_ACCESS_KEY_ID'] = config['AWS_ACCESS_KEY_ID']
    if config.get('AWS_SECRET_ACCESS_KEY'):
        os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS_SECRET_ACCESS_KEY']
    endpoint = config.get('AWS_ENDPOINT_URL')
    if endpoint:
        os.environ['AWS_ENDPOINT_URL']    = endpoint
        os.environ['AWS_ENDPOINT_URL_S3'] = endpoint
    if config.get('AWS_REGION'):
        os.environ['AWS_REGION'] = config['AWS_REGION']
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def clean_prefix(bucket: str, prefix: str, config: dict):
    """Delete all objects under s3://bucket/prefix/ using s3dlio Python API."""
    import s3dlio
    uri = f's3://{bucket}/{prefix}/'.rstrip('/') + '/'
    with _s3_env(config):
        try:
            full_uris = s3dlio.list(uri, recursive=True)
            if not full_uris:
                print(f'    (nothing to clean at {uri})')
                return
            for obj_uri in full_uris:
                s3dlio.delete(obj_uri)
            print(f'    Cleaned {len(full_uris)} object(s) at {uri}')
        except Exception as e:
            print(f'    (nothing to clean at {uri}: {e})')


def list_prefix(bucket: str, prefix: str, config: dict, label: str = '') -> int:
    """List & count objects under s3://bucket/prefix/ using s3dlio Python API.
    Returns the number of objects found."""
    import s3dlio
    uri = f's3://{bucket}/{prefix}/'.rstrip('/') + '/'
    tag = f' [{label}]' if label else ''
    with _s3_env(config):
        try:
            full_uris = s3dlio.list(uri, recursive=True)
            count = len(full_uris)
            if count:
                print(f'    s3dlio list {uri}{tag}: {count} object(s)')
                # Show up to 5 keys (strip the URI prefix for readability)
                for obj_uri in full_uris[:5]:
                    print(f'      {obj_uri}')
                if count > 5:
                    print(f'      ... ({count - 5} more)')
            else:
                print(f'    s3dlio list {uri}{tag}: (empty)')
            return count
        except Exception as e:
            print(f'    s3dlio list {uri}{tag}: error: {e}')
            return 0


def run_phase(label: str, cmd: list, env: dict, timeout_s: int = 3600) -> tuple:
    """
    Stream subprocess output live.
    Returns (returncode, elapsed_seconds, captured_output).
    Prints each output line indented for readability.
    """
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


# ── Workload 1: Training ─────────────────────────────────────────────────────────

def run_training(library: str, config: dict) -> dict:
    bucket = LIBRARY_BUCKETS[library]
    env    = build_env(config, library)
    data_folder = f's3://{bucket}/{TRAIN_PREFIX}'
    total_gb    = TRAIN_NUM_FILES * TRAIN_SIZE_MiB / 1024.0
    region      = config.get('AWS_REGION', 'us-east-1')

    print(f'\n── Training  [{library}]  s3://{bucket}/{TRAIN_PREFIX}/ ──')
    print(f'   {TRAIN_NUM_FILES} × {TRAIN_SIZE_MiB} MiB = {total_gb:.2f} GiB   '
          f'| {TRAIN_EPOCHS} epochs')

    # Phase 0: cleanup
    print('\n  Phase 0: Cleanup')
    clean_prefix(bucket, TRAIN_PREFIX, config)

    # Shared storage params (passed to both datagen and run)
    storage_params = [
        f'storage.storage_type=s3',
        f'storage.storage_root={bucket}',
        f'storage.storage_library={library}',
        f'storage.storage_options.endpoint_url={config["AWS_ENDPOINT_URL"]}',
        f'storage.storage_options.access_key_id={config["AWS_ACCESS_KEY_ID"]}',
        f'storage.storage_options.secret_access_key={config["AWS_SECRET_ACCESS_KEY"]}',
        f'storage.storage_options.region={region}',
        f'storage.storage_options.s3_force_path_style=true',
        f'dataset.data_folder={data_folder}',
        f'dataset.num_files_train={TRAIN_NUM_FILES}',
        f'dataset.num_samples_per_file={TRAIN_SAMPLES_PER}',
        f'dataset.record_length={TRAIN_RECORD_BYTES}',
        f'dataset.format=npz',          # required: S3+PyTorch only supports npz/npy
    ]

    # datagen uses --num-processes (NOT --num-accelerators / --accelerator-type)
    datagen_flags = [
        '--model', TRAIN_MODEL,
        '--num-processes', '8',
        '--open',
        '--skip-validation',
        '--results-dir', RESULTS_DIR,
    ]
    # training run uses --num-accelerators + --accelerator-type + --client-host-memory-in-gb
    run_flags = [
        '--model', TRAIN_MODEL,
        '--num-accelerators', str(TRAIN_NUM_ACCEL),
        '--accelerator-type', TRAIN_ACCEL_TYPE,
        '--client-host-memory-in-gb', str(CLIENT_MEM_GB),
        '--open',
        '--skip-validation',
        '--results-dir', RESULTS_DIR,
    ]

    # Phase 1: datagen (write)
    print(f'\n  Phase 1: datagen — write {TRAIN_NUM_FILES} × {TRAIN_SIZE_MiB} MiB objects')
    rc_gen = -1; t_gen = 0.0
    rc_run = -1; t_run = 0.0
    try:
        rc_gen, t_gen, _ = run_phase(
            'datagen',
            ['mlpstorage', 'training', 'datagen'] + datagen_flags + ['--params'] + storage_params,
            env,
        )

        gen_gbps = total_gb / t_gen if rc_gen == 0 and t_gen > 0 else None

        if rc_gen == 0:
            obj_count = list_prefix(bucket, TRAIN_PREFIX, config, 'after datagen')
            if obj_count < TRAIN_NUM_FILES:
                print(f'  ❌ datagen validation FAILED: bucket shows {obj_count} objects, '
                      f'expected {TRAIN_NUM_FILES}')
                rc_gen = 1
            else:
                pause(PAUSE_SECONDS, 'S3 eventual consistency — new objects must be visible before reads')

        # Phase 2: training run (read × epochs)
        print(f'\n  Phase 2: train — read {TRAIN_EPOCHS} epochs '
              f'({total_gb * TRAIN_EPOCHS:.2f} GiB total reads)')
        if rc_gen != 0:
            print('  ⚠ Skipping training run — datagen did not produce expected objects')
        else:
            rc_run, t_run, _ = run_phase(
                'training run',
                ['mlpstorage', 'training', 'run'] + run_flags + ['--params'] + storage_params + [
                    f'train.epochs={TRAIN_EPOCHS}',
                    f'train.batch_size=1',
                    f'reader.batch_size=1',
                    f'reader.read_threads=8',
                    f'reader.prefetch_size=4',
                ],
                env,
            )
    finally:
        # Always clean up — prevent filling storage between runs
        print(f'\n  Phase 3: Cleanup (post-run)')
        clean_prefix(bucket, TRAIN_PREFIX, config)
        list_prefix(bucket, TRAIN_PREFIX, config, 'after cleanup')

    read_total_gb = total_gb * TRAIN_EPOCHS
    gen_gbps  = total_gb     / t_gen if rc_gen == 0 and t_gen > 0 else None
    run_gbps  = read_total_gb / t_run if rc_run == 0 and t_run > 0 else None

    return {
        'library':    library,
        'workload':   'training',
        'dataset_gb': total_gb,
        'epochs':     TRAIN_EPOCHS,
        'gen_ok':     rc_gen == 0,
        'run_ok':     rc_run == 0,
        'gen_time':   t_gen,
        'run_time':   t_run,
        'gen_gbps':   gen_gbps,
        'run_gbps':   run_gbps,
    }


# ── Workload 2: Checkpoint ────────────────────────────────────────────────────────

def run_checkpoint(library: str, config: dict, network_gbps: float = None) -> dict:
    """
    Write a streaming checkpoint via StreamingCheckpointing.save(), then read it
    back via StreamingCheckpointing.load().  Cleanup happens only after both phases.

    StreamingCheckpointing uses a fixed producer-consumer pipeline:
      chunk_size × num_buffers = 32 MB × 4 = 128 MB RAM, regardless of checkpoint size.
    dgen-py generates data in parallel while the library uploads it — memory stays flat.
    """
    from mlpstorage.checkpointing import StreamingCheckpointing

    bucket      = LIBRARY_BUCKETS[library]
    env         = build_env(config, library)
    uri         = f's3://{bucket}/{CKPT_PREFIX}/checkpoint.dat'
    size_gb     = CKPT_SIZE_GB_OVERRIDE.get(library, CKPT_SIZE_GB)
    total_bytes = int(size_gb * 1024 ** 3)

    size_note = f'  (capped at {size_gb:.0f} GB for {library})' if library in CKPT_SIZE_GB_OVERRIDE else ''
    print(f'\n── Checkpoint  [{library}]  {uri} ──')
    print(f'   Size: {size_gb:.0f} GB  |  backend: {library}{size_note}')
    print(f'   RAM usage: streaming pipeline ({CKPT_CHUNK_MB} MB chunks '
          f'× {CKPT_NUM_BUFFERS} buffers = '
          f'{CKPT_CHUNK_MB * CKPT_NUM_BUFFERS} MB max regardless of checkpoint size)')

    # Apply credentials to os.environ so the storage backend writers can pick them up
    saved_env = {k: os.environ.get(k) for k in config}
    for k, v in config.items():
        os.environ[k] = v
    os.environ['STORAGE_LIBRARY'] = library

    ok_write = False
    ok_read  = False
    t_write  = 0.0
    t_read   = 0.0
    write_gbps = None
    read_gbps  = None
    try:
        # Phase 0: cleanup
        print('\n  Phase 0: Cleanup')
        clean_prefix(bucket, CKPT_PREFIX, config)
        list_prefix(bucket, CKPT_PREFIX, config, 'before save')
        pause(PAUSE_SECONDS, 'storage settling after cleanup')

        # Phase 1: streaming save
        print(f'\n  Phase 1: StreamingCheckpointing.save() → {uri}')
        if network_gbps:
            print(f'   {size_gb:.0f} GB at {network_gbps:.3f} GB/s ({network_gbps*8:.0f} Gbps) → expect ~'
                  f'{size_gb / network_gbps:.0f}s minimum')
        else:
            print(f'   {size_gb:.0f} GB  (no --network-gbits specified; no timing estimate)')
        checkpoint = StreamingCheckpointing(
            chunk_size   = CKPT_CHUNK_MB * 1024 * 1024,
            num_buffers  = CKPT_NUM_BUFFERS,
            use_dgen     = True,
            backend      = library,
            fadvise_mode = 'none',
        )
        t_start  = time.perf_counter()
        result   = checkpoint.save(uri, total_bytes)
        t_write  = time.perf_counter() - t_start

        io_time    = result.get('io_time', t_write)
        write_gbps = size_gb / io_time if io_time > 0 else size_gb / t_write
        gen_gbps   = result.get('gen_throughput_gbps', 0)
        bottleneck = result.get('bottleneck', '?')

        print(f'  ✅ checkpoint save done in {t_write:.1f}s  '
              f'({write_gbps:.3f} GB/s I/O  |  {gen_gbps:.1f} GB/s gen  '
              f'|  bottleneck: {bottleneck})')
        ok_write = True

        list_prefix(bucket, CKPT_PREFIX, config, 'after save')
        pause(PAUSE_SECONDS, 'S3 eventual consistency before read')

        # Phase 2: streaming load (read back)
        print(f'\n  Phase 2: StreamingCheckpointing.load() ← {uri}')
        if network_gbps:
            print(f'   {size_gb:.0f} GB at {network_gbps:.3f} GB/s → expect ~'
                  f'{size_gb / network_gbps:.0f}s minimum')
        r_start  = time.perf_counter()
        load_result = checkpoint.load(uri, total_bytes)
        t_read   = time.perf_counter() - r_start

        r_io_time  = load_result.get('io_time', t_read)
        read_gbps  = size_gb / r_io_time if r_io_time > 0 else size_gb / t_read
        print(f'  ✅ checkpoint load done in {t_read:.1f}s  ({read_gbps:.3f} GB/s)')
        ok_read = True

    except Exception as e:
        elapsed = time.perf_counter() - (t_start if 't_start' in dir() else time.perf_counter())
        print(f'  ❌ Checkpoint phase failed after {elapsed:.1f}s: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup runs after both write and read are done (or on error)
        print(f'\n  Phase 3: Cleanup (post-run)')
        clean_prefix(bucket, CKPT_PREFIX, config)
        list_prefix(bucket, CKPT_PREFIX, config, 'after cleanup')
        # Restore original env
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        os.environ.pop('STORAGE_LIBRARY', None)

    return {
        'library':    library,
        'workload':   'checkpoint',
        'size_gb':    size_gb,
        'ok_write':   ok_write,
        'ok_read':    ok_read,
        'ok':         ok_write and ok_read,
        't_write':    t_write,
        't_read':     t_read,
        'write_gbps': write_gbps,
        'read_gbps':  read_gbps,
    }


# ── Results table ─────────────────────────────────────────────────────────────────

def print_results(training_results: list, checkpoint_results: list):
    print()
    print('=' * 96)
    print('DLIO MULTI-LIBRARY BENCHMARK — RESULTS')
    print('=' * 96)

    if training_results:
        total_gb    = TRAIN_NUM_FILES * TRAIN_SIZE_MiB / 1024.0
        read_total  = total_gb * TRAIN_EPOCHS
        print()
        print(f'WORKLOAD 1: TRAINING')
        print(f'  {TRAIN_NUM_FILES} objects × {TRAIN_SIZE_MiB} MiB = '
              f'{total_gb:.2f} GiB dataset  |  {TRAIN_EPOCHS} epochs  |  '
              f'{read_total:.2f} GiB total reads per library')
        print(f'  {"Library":<22} {"Write GB/s":>12} {"Read GB/s":>12} '
              f'{"Gen s":>8} {"Train s":>9}  {"Status"}')
        print(f'  {"-"*22} {"-"*12} {"-"*12} {"-"*8} {"-"*9}  {"-"*6}')

        best_gen  = max((r['gen_gbps'] for r in training_results if r.get('gen_gbps')), default=0)
        best_read = max((r['run_gbps'] for r in training_results if r.get('run_gbps')), default=0)

        for r in training_results:
            gen_s  = f"{r['gen_gbps']:.3f}"  if r.get('gen_gbps')  else 'N/A  '
            read_s = f"{r['run_gbps']:.3f}"  if r.get('run_gbps')  else 'N/A  '
            gmark  = ' ◀W' if r.get('gen_gbps')  == best_gen  else '   '
            rmark  = ' ◀R' if r.get('run_gbps')  == best_read else '   '
            t_gen  = f"{r['gen_time']:.1f}s" if r.get('gen_time') else '-'
            t_run  = f"{r['run_time']:.1f}s" if r.get('run_time') else '-'
            status = ('✅' if (r['gen_ok'] and r['run_ok'])
                      else ('❌ datagen failed' if not r['gen_ok'] else '❌ train failed'))
            print(f"  {r['library']:<22} {gen_s+gmark:>15} {read_s+rmark:>15} "
                  f"{t_gen:>8} {t_run:>9}  {status}")

        print()
        print('  Write GB/s = DLIO datagen throughput (generate + write to S3)')
        print('  Read GB/s  = DLIO training read throughput (total read GiB / total read time)')
        print('  ◀W = fastest write   ◀R = fastest read')
        print()
        print('  Compare these numbers to the native API results in WRITE_READ_COMPARISON_RESULTS.md')
        print('  to quantify DLIO overhead vs raw library throughput.')

    if checkpoint_results:
        print()
        print(f'WORKLOAD 2: CHECKPOINT  (StreamingCheckpointing — fixed 128 MB RAM)')
        print(f'  Single object per library via streaming producer-consumer pipeline')
        print(f'  {CKPT_CHUNK_MB} MB chunks × {CKPT_NUM_BUFFERS} buffers = '
              f'{CKPT_CHUNK_MB * CKPT_NUM_BUFFERS} MB RAM max regardless of checkpoint size')
        print(f'  {"Library":<22} {"Size GB":>9} {"Write GB/s":>12} {"Read GB/s":>12}  {"Status"}')
        print(f'  {"-"*22} {"-"*9} {"-"*12} {"-"*12}  {"-"*6}')

        best_w = max((r['write_gbps'] for r in checkpoint_results if r.get('write_gbps')), default=0)
        best_r = max((r['read_gbps']  for r in checkpoint_results if r.get('read_gbps')),  default=0)

        for r in checkpoint_results:
            w_s   = f"{r['write_gbps']:.3f}" if r.get('write_gbps') else 'N/A  '
            rd_s  = f"{r['read_gbps']:.3f}"  if r.get('read_gbps')  else 'N/A  '
            wmark = ' ◀W' if r.get('write_gbps') == best_w else '   '
            rmark = ' ◀R' if r.get('read_gbps')  == best_r else '   '
            if not r.get('ok_write', r.get('ok')):
                status = '❌ write failed'
            elif not r.get('ok_read', True):
                status = '❌ read failed'
            else:
                status = '✅'
            print(f"  {r['library']:<22} {r['size_gb']:>9.0f} {w_s+wmark:>15} {rd_s+rmark:>15}  {status}")

        print()
        print('  Write GB/s = I/O throughput from StreamingCheckpointing.save()')
        print('  Read GB/s  = I/O throughput from StreamingCheckpointing.load() (byte-range GETs, data discarded)')
        print('  ◀W = fastest write   ◀R = fastest read')
        print('  dgen-py generates write data concurrently; bottleneck is always I/O, not generation')

    print()
    print('=' * 96)


# ── Preflight checks ──────────────────────────────────────────────────────────────

def preflight(do_checkpoint: bool):
    ok = True

    # mlpstorage
    import shutil
    if not shutil.which('mlpstorage'):
        print('ERROR: mlpstorage not found in PATH. Activate the virtualenv first.')
        ok = False

    # StreamingCheckpointing is in-process — no MPI required.
    # (mlpstorage.checkpointing import verified at import-time above)

    return ok


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DLIO multi-library benchmark demo (training + checkpoint)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_dlio_multilib_demo.py                                        # all libraries, both workloads
  python test_dlio_multilib_demo.py --workload training                    # training only
  python test_dlio_multilib_demo.py --workload checkpoint                  # checkpoint only
  python test_dlio_multilib_demo.py --library s3dlio                       # single library
  python test_dlio_multilib_demo.py --library s3dlio minio                 # two libraries
  python test_dlio_multilib_demo.py --workload training --library s3dlio minio
  python test_dlio_multilib_demo.py --workload checkpoint --network-gbits 10    # 10 Gbps link → ~80s estimate
        """,
    )
    parser.add_argument(
        '--workload', choices=['training', 'checkpoint', 'both'], default='both',
        help='Which workload to run (default: both)',
    )
    parser.add_argument(
        '--library', choices=['s3dlio', 'minio', 's3torchconnector'],
        nargs='+', dest='libraries', metavar='LIBRARY',
        help='Library/libraries to test (default: all three)',
    )
    parser.add_argument(
        '--network-gbits', type=float, default=None, metavar='N',
        help='Network link speed in Gbps (gigabits/s, e.g. 10 for a 10 Gbps link). '
             'Optional — used only for informational time estimates in the checkpoint '
             'phase. Does not affect test logic.',
    )
    args = parser.parse_args()

    libraries     = args.libraries or DEFAULT_LIBRARIES
    do_training   = args.workload in ('training', 'both')
    do_checkpoint = args.workload in ('checkpoint', 'both')
    # Convert Gbps → GB/s internally (1 byte = 8 bits)
    network_gbps  = args.network_gbits / 8.0 if args.network_gbits else None

    config = load_env_config()
    for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_ENDPOINT_URL']:
        if not config.get(key):
            print(f'ERROR: {key} not set in .env or environment', file=sys.stderr)
            sys.exit(1)

    if not preflight(do_checkpoint):
        sys.exit(1)

    # Header
    total_gb = TRAIN_NUM_FILES * TRAIN_SIZE_MiB / 1024.0
    print()
    print('=' * 96)
    print('DLIO MULTI-LIBRARY BENCHMARK DEMO')
    print('  I/O through DLIO (mlpstorage) — compares s3dlio, minio, s3torchconnector')
    print('=' * 96)
    print(f'  Endpoint:    {config["AWS_ENDPOINT_URL"]}')
    print(f'  Libraries:   {", ".join(libraries)}')
    print(f'  Workloads:   {args.workload}')
    if do_training:
        print(f'  Training:    {TRAIN_NUM_FILES} × {TRAIN_SIZE_MiB} MiB = '
              f'{total_gb:.2f} GiB/library  |  {TRAIN_EPOCHS} epochs')
    if do_checkpoint:
        net_hint = (f'  |  ~{CKPT_SIZE_GB / network_gbps:.0f}s at {args.network_gbits:.0f} Gbps'
                    if network_gbps else '')
        print(f'  Checkpoint:  {CKPT_SIZE_GB:.0f} GB streaming  |  '
              f'{CKPT_CHUNK_MB} MB chunks × {CKPT_NUM_BUFFERS} buffers = '
              f'{CKPT_CHUNK_MB * CKPT_NUM_BUFFERS} MB RAM  |  backend per library{net_hint}')
    print(f'  Buckets:     ' +
          '  '.join(f'{l}={LIBRARY_BUCKETS[l]}' for l in libraries if l in LIBRARY_BUCKETS))
    print('=' * 96)

    training_results   = []
    checkpoint_results = []

    for i, lib in enumerate(libraries):
        if i > 0:
            pause(PAUSE_SECONDS, f'cooldown between libraries ({libraries[i-1]} → {lib})')
        if do_training:
            result = run_training(lib, config)
            training_results.append(result)
        if do_checkpoint:
            if do_training:
                pause(PAUSE_SECONDS, 'cooldown between training and checkpoint workloads')
            result = run_checkpoint(lib, config, network_gbps=network_gbps)
            checkpoint_results.append(result)

    print_results(training_results, checkpoint_results)

    all_ok = (
        all(r['gen_ok'] and r['run_ok'] for r in training_results) and
        all(r['ok'] for r in checkpoint_results)
    )

    if all_ok:
        print('✅ All tests passed.')
        sys.exit(0)
    else:
        print('❌ Some tests failed — see output above.')
        sys.exit(1)


if __name__ == '__main__':
    main()
