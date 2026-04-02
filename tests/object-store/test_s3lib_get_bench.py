#!/usr/bin/env python3
"""S3 library GET benchmark — fair comparison across s3dlio, minio, and s3torchconnector.

Answers two questions:
  1. Per-request cost: how fast is a single async GET from one file, with no parallelism?
  2. Aggregate capacity: what throughput can each library achieve with identical concurrency?

Test modes
──────────
  serial      One file at a time, no parallelism.  Reveals per-request HTTP overhead
              of each library's underlying client (head latency, connection reuse, etc.).
              Reports p50/p95/p99/max latency and single-stream MB/s.

  parallel    ThreadPoolExecutor with the SAME worker count for all libraries.
              All three read the identical object list from the SAME bucket.
              Reports aggregate MB/s at each concurrency level; can be swept.

  native      s3dlio.get_many(uris, max_in_flight=N) — Rust Tokio async, not Python threads.
              Same max_in_flight sweep as parallel workers for a direct comparison.
              Shows whether Rust async outperforms Python ThreadPoolExecutor at matched N.

Key design choices
──────────────────
  • All three libraries read from THE SAME bucket and THE SAME objects.  No per-library
    buckets — this eliminates any data locality/ordering effects.
  • Object listing is done via minio (works on any S3-compatible endpoint).
  • Credential precedence: .env file < env vars < CLI flags.
  • Clients are created ONCE per library (not per-object) to share connection pools.
  • Read timing starts at first GET call and ends when the last byte is received.
    Data is consumed (len() counted) but not processed further.

Example usage
─────────────
  # Benchmark against existing training data (default bucket/prefix):
  python test_s3lib_get_bench.py

  # Restrict to 30 files, sweep concurrency 1/4/8/16/32:
  python test_s3lib_get_bench.py --num-files 30 --workers 1 4 8 16 32

  # Serial-only test with a different bucket:
  python test_s3lib_get_bench.py --mode serial --bucket my-bucket --prefix data/

  # Create 20 synthetic 128 MB objects first, then benchmark:
  python test_s3lib_get_bench.py --write --write-num-files 20 --write-size-mb 128

  # Test only minio and s3dlio:
  python test_s3lib_get_bench.py --libraries s3dlio minio

Credential precedence: .env file < environment variables < CLI options
"""

import os
import sys
import time
import argparse
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add mlp-storage root to path so shared utilities are importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_BUCKET      = os.environ.get('S3_BUCKET', 'mlp-s3dlio')
DEFAULT_PREFIX      = os.environ.get('S3_PREFIX', 'test-run/unet3d/train/')
DEFAULT_NUM_FILES   = 20
DEFAULT_WORKERS     = [1, 4, 8, 16]   # concurrency sweep for parallel + native tests
DEFAULT_MAX_LIST    = 1000            # max objects fetched from the prefix

WRITE_BUCKET        = DEFAULT_BUCKET
WRITE_PREFIX        = "bench-get/obj"
DEFAULT_WRITE_FILES = 20
DEFAULT_WRITE_MB    = 128


# ── Credential loading (mirrors test_direct_write_comparison.py) ──────────────

def load_env_config() -> dict:
    """Load config from .env file, then override with environment variables."""
    env_path = None
    for candidate in [
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
    ]:
        if candidate.exists():
            env_path = candidate
            break

    config: dict = {}
    if env_path:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, val = line.partition('=')
                    config[key.strip()] = val.strip()
        print(f"Loaded credentials from: {env_path}")
    else:
        print("No .env file found, using environment variables")

    for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
                'AWS_ENDPOINT_URL', 'AWS_REGION']:
        if key in os.environ:
            config[key] = os.environ[key]

    return config


def apply_config(config: dict) -> None:
    for key, val in config.items():
        os.environ[key] = val


# ── CA bundle helper ─────────────────────────────────────────────────────────

def _get_ca_bundle() -> str | None:
    """Return the CA bundle path from the AWS_CA_BUNDLE environment variable.

    This is the standard AWS SDK name, now also used by s3dlio.
    """
    return os.environ.get('AWS_CA_BUNDLE') or None


# ── S3 client factories ───────────────────────────────────────────────────────

def _make_minio_client():
    """Build a minio.Minio client from environment credentials (singleton caller).

    When the endpoint uses HTTPS and AWS_CA_BUNDLE is set,
    a custom urllib3 PoolManager is created with the specified CA cert so that
    self-signed certificates are accepted.  (Python's ssl module uses its own
    CA store and does not pick up AWS_CA_BUNDLE automatically.)
    """
    from minio import Minio
    endpoint_url = os.environ.get('AWS_ENDPOINT_URL', '')
    if endpoint_url.startswith('https://'):
        endpoint, secure = endpoint_url[8:], True
    elif endpoint_url.startswith('http://'):
        endpoint, secure = endpoint_url[7:], False
    else:
        endpoint = endpoint_url or 's3.amazonaws.com'
        secure = not bool(endpoint_url)

    http_client = None
    if secure:
        ca_bundle = _get_ca_bundle()
        if ca_bundle:
            import ssl
            import urllib3
            ctx = ssl.create_default_context(cafile=ca_bundle)
            http_client = urllib3.PoolManager(ssl_context=ctx)

    return Minio(
        endpoint,
        access_key=os.environ['AWS_ACCESS_KEY_ID'],
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        secure=secure,
        region=os.environ.get('AWS_REGION', 'us-east-1'),
        http_client=http_client,
    )


def _make_s3torch_client():
    """Build an s3torchconnector S3Client from environment credentials."""
    from s3torchconnector._s3client import S3Client, S3ClientConfig
    region   = os.environ.get('AWS_REGION', 'us-east-1')
    endpoint = os.environ.get('AWS_ENDPOINT_URL')
    cfg = S3ClientConfig(force_path_style=bool(endpoint), max_attempts=3)
    return S3Client(region=region, endpoint=endpoint, s3client_config=cfg)


# ── Object listing ────────────────────────────────────────────────────────────

def list_objects(bucket: str, prefix: str, max_count: int) -> list:
    """Return up to max_count object keys under prefix/ using minio."""
    client = _make_minio_client()
    norm_prefix = prefix.rstrip('/') + '/'
    objects = client.list_objects(bucket, prefix=norm_prefix, recursive=True)
    keys = []
    for obj in objects:
        keys.append(obj.object_name)
        if len(keys) >= max_count:
            break
    return keys


# ── Per-object GET workers ────────────────────────────────────────────────────

def _get_s3dlio(bucket: str, key: str) -> int:
    """Fetch one object via s3dlio.get(). Returns byte count."""
    import s3dlio
    data = s3dlio.get(f"s3://{bucket}/{key}")
    return len(memoryview(data))


def _get_minio(client, bucket: str, key: str) -> int:
    """Fetch one object via minio.get_object(). Returns byte count."""
    resp = client.get_object(bucket, key)
    try:
        data = resp.read()
        return len(data)
    finally:
        resp.close()
        resp.release_conn()


def _get_s3torch(client, bucket: str, key: str) -> int:
    """Fetch one object via S3Client.get_object() (direct, no S3IterableDataset). Returns byte count."""
    reader = client.get_object(bucket, key)
    data = reader.read()
    return len(data)


# ── Percentile helper ─────────────────────────────────────────────────────────

def _percentile(data: list, p: float) -> float:
    """Return the p-th percentile (0–100) of sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo = int(k)
    hi = lo + 1
    if hi >= len(sorted_data):
        return sorted_data[lo]
    frac = k - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


# ── Serial test ───────────────────────────────────────────────────────────────

def _run_serial(library: str, bucket: str, keys: list,
                minio_client, s3torch_client) -> dict:
    """Fetch all keys one at a time.  Returns latency list and totals."""
    latencies = []
    total_bytes = 0

    for key in keys:
        t0 = time.perf_counter()
        if library == 's3dlio':
            n = _get_s3dlio(bucket, key)
        elif library == 'minio':
            n = _get_minio(minio_client, bucket, key)
        else:
            n = _get_s3torch(s3torch_client, bucket, key)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        total_bytes += n

    total_elapsed = sum(latencies)
    return {
        'latencies':   latencies,
        'total_bytes': total_bytes,
        'total_time':  total_elapsed,
    }


def run_serial_test(libraries: list, bucket: str, keys: list,
                    minio_client, s3torch_client) -> dict:
    """Run serial test for all selected libraries.  Returns per-library results."""
    results = {}
    for lib in libraries:
        print(f"  [{lib:<20}] serial: {len(keys)} × 1 GET …", flush=True)
        t_wall = time.perf_counter()
        r = _run_serial(lib, bucket, keys, minio_client, s3torch_client)
        r['wall_time'] = time.perf_counter() - t_wall
        stream_mbps = (r['total_bytes'] / (1024**2)) / r['total_time'] if r['total_time'] else 0
        print(f"  [{lib:<20}]  done: {stream_mbps:.0f} MB/s (stream), "
              f"p50={_percentile(r['latencies'],50):.3f}s")
        results[lib] = r
    return results


# ── Parallel test (ThreadPoolExecutor — same for all libraries) ───────────────

def _run_parallel(library: str, bucket: str, keys: list, num_workers: int,
                  minio_client, s3torch_client) -> dict:
    """Fetch all keys in parallel via ThreadPoolExecutor(max_workers=num_workers)."""
    t_start = time.perf_counter()
    total_bytes = 0

    def _task(key):
        if library == 's3dlio':
            return _get_s3dlio(bucket, key)
        elif library == 'minio':
            return _get_minio(minio_client, bucket, key)
        else:
            return _get_s3torch(s3torch_client, bucket, key)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = [pool.submit(_task, k) for k in keys]
        for fut in as_completed(futs):
            total_bytes += fut.result()

    elapsed = time.perf_counter() - t_start
    return {'total_bytes': total_bytes, 'elapsed': elapsed}


def run_parallel_test(libraries: list, bucket: str, keys: list,
                      workers_sweep: list,
                      minio_client, s3torch_client) -> dict:
    """Run parallel test for all (library, workers) combinations.

    Returns: {library: {workers: {total_bytes, elapsed}}}
    """
    results: dict = {lib: {} for lib in libraries}
    for num_workers in workers_sweep:
        for lib in libraries:
            print(f"  [{lib:<20}] parallel workers={num_workers:>3}: …", end=' ', flush=True)
            r = _run_parallel(lib, bucket, keys, num_workers, minio_client, s3torch_client)
            mbps = (r['total_bytes'] / (1024**2)) / r['elapsed'] if r['elapsed'] else 0
            print(f"{mbps:>6.0f} MB/s")
            results[lib][num_workers] = r
    return results


# ── s3dlio native get_many test ───────────────────────────────────────────────

def run_native_test(bucket: str, keys: list, workers_sweep: list) -> dict:
    """Run s3dlio.get_many() with each max_in_flight value in workers_sweep.

    Returns: {max_in_flight: {total_bytes, elapsed}}
    """
    import s3dlio
    results = {}
    uris = [f"s3://{bucket}/{k}" for k in keys]

    for max_in_flight in workers_sweep:
        cap = min(max_in_flight, len(uris))
        print(f"  [s3dlio native       ] get_many max_in_flight={cap:>3}: …", end=' ', flush=True)
        t_start = time.perf_counter()
        pairs = s3dlio.get_many(uris, max_in_flight=cap)
        elapsed = time.perf_counter() - t_start
        total_bytes = sum(len(memoryview(data)) for _, data in pairs)
        mbps = (total_bytes / (1024**2)) / elapsed if elapsed else 0
        print(f"{mbps:>6.0f} MB/s")
        results[max_in_flight] = {'total_bytes': total_bytes, 'elapsed': elapsed}

    return results


# ── Write helper (optional: create synthetic test objects) ────────────────────

def write_test_objects(bucket: str, prefix: str, num_files: int, size_mb: int) -> list:
    """Write num_files synthetic objects of size_mb MB each, return their keys."""
    import io
    client = _make_minio_client()
    size_bytes = size_mb * 1024 * 1024
    keys = []

    # Ensure the bucket exists
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"  Created bucket: {bucket}")

    # Generate data once (simple repeating pattern — speed not measured)
    import random
    chunk = bytes(random.getrandbits(8) for _ in range(min(1024 * 1024, size_bytes)))
    data = (chunk * ((size_bytes // len(chunk)) + 1))[:size_bytes]

    print(f"  Writing {num_files} × {size_mb} MB objects to s3://{bucket}/{prefix}")
    t_start = time.perf_counter()
    for i in range(num_files):
        key = f"{prefix.rstrip('/')}/obj-{i:05d}.bin"
        client.put_object(bucket, key, io.BytesIO(data), size_bytes)
        keys.append(key)
        if (i + 1) % max(1, num_files // 5) == 0:
            elapsed = time.perf_counter() - t_start
            print(f"    {i+1}/{num_files} written  ({elapsed:.1f}s)")

    elapsed = time.perf_counter() - t_start
    total_mb = num_files * size_mb
    print(f"  Write done: {total_mb} MB in {elapsed:.1f}s ({total_mb/elapsed:.0f} MB/s)")
    return keys


# ── Result formatting ─────────────────────────────────────────────────────────

_W = 22   # library name column width


def print_header(title: str, separator: str = '═') -> None:
    print()
    print(separator * 72)
    print(title)
    print(separator * 72)


def print_serial_results(serial: dict, num_files: int) -> None:
    print_header(f"SERIAL GET — one file at a time (no parallelism)  [{num_files} files]")
    print(f"  {'Library':<{_W}}  {'p50':>7}  {'p95':>7}  {'p99':>7}  {'max':>7}  {'MB/s':>8}")
    print(f"  {'─'*_W}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}")

    best_mbps = max(
        (r['total_bytes'] / (1024**2)) / r['total_time']
        for r in serial.values() if r['total_time'] > 0
    )
    for lib, r in serial.items():
        lats = r['latencies']
        p50 = _percentile(lats, 50)
        p95 = _percentile(lats, 95)
        p99 = _percentile(lats, 99)
        mx  = max(lats)
        mbps = (r['total_bytes'] / (1024**2)) / r['total_time'] if r['total_time'] else 0
        mark = ' ◀' if abs(mbps - best_mbps) < 0.5 else ''
        print(f"  {lib:<{_W}}  {p50:>6.3f}s  {p95:>6.3f}s  {p99:>6.3f}s  {mx:>6.3f}s  "
              f"{mbps:>7.0f}{mark}")

    print()
    print("  p50/p95/p99/max — per-GET wall-clock latency (s)  |  "
          "MB/s — single-stream throughput (sum_bytes / sum_latency)")
    print("  ◀ = fastest library at this concurrency level")


def print_parallel_results(parallel: dict, workers_sweep: list, num_files: int) -> None:
    print_header(f"PARALLEL GET — ThreadPoolExecutor, same concurrency for all")
    print(f"  [{num_files} files, same bucket+objects for all libraries]\n")

    w_cols = [f"w={w:>2}" for w in workers_sweep]
    header = f"  {'Library':<{_W}}  " + "  ".join(f"{c:>9}" for c in w_cols)
    print(header)
    print(f"  {'─'*_W}  " + "  ".join("─" * 9 for _ in workers_sweep))

    # Compute best MB/s per workers value
    best: dict = {}
    for w in workers_sweep:
        vals = [
            (r[w]['total_bytes'] / (1024**2)) / r[w]['elapsed']
            for r in parallel.values() if w in r and r[w]['elapsed'] > 0
        ]
        best[w] = max(vals) if vals else 0

    for lib, by_w in parallel.items():
        cells = []
        for w in workers_sweep:
            if w not in by_w or by_w[w]['elapsed'] == 0:
                cells.append(f"{'—':>9}")
                continue
            mbps = (by_w[w]['total_bytes'] / (1024**2)) / by_w[w]['elapsed']
            mark = '◀' if abs(mbps - best[w]) < 0.5 else ' '
            cells.append(f"{mbps:>7.0f}{mark} ")
        print(f"  {lib:<{_W}}  " + "  ".join(cells))

    print()
    print("  All values in MB/s  |  ◀ = fastest library at that worker count")
    print("  All libraries use ThreadPoolExecutor(max_workers=N) — identical concurrency model")


def print_native_results(native: dict, workers_sweep: list, num_files: int,
                         parallel: dict) -> None:
    print_header("s3dlio NATIVE get_many() — Rust Tokio async  (s3dlio only)")
    print(f"  [{num_files} files]\n")

    print(f"  {'max_in_flight':<16}  {'MB/s':>9}  {'vs ThreadPoolExec':>20}")
    print(f"  {'─'*16}  {'─'*9}  {'─'*20}")

    for mif, r in native.items():
        if r['elapsed'] == 0:
            print(f"  {mif:<16}  {'—':>9}")
            continue
        mbps = (r['total_bytes'] / (1024**2)) / r['elapsed']
        # Compare to s3dlio parallel at same worker count (if measured)
        s3d_parallel = parallel.get('s3dlio', {}).get(mif)
        if s3d_parallel and s3d_parallel.get('elapsed', 0) > 0:
            tp_mbps = (s3d_parallel['total_bytes'] / (1024**2)) / s3d_parallel['elapsed']
            pct = (mbps - tp_mbps) / tp_mbps * 100 if tp_mbps else 0
            cmp = f"{pct:+.1f}% vs w={mif} ThreadPool"
        else:
            cmp = ""
        print(f"  {mif:<16}  {mbps:>9.0f}  {cmp}")

    print()
    print("  get_many() uses s3dlio's Rust Tokio async engine; all requests are scheduled")
    print("  in a single Rust thread pool — no Python GIL or thread creation overhead.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Source bucket/prefix/files
    parser.add_argument('--bucket',     default=DEFAULT_BUCKET,
                        help=f'Source bucket (default: {DEFAULT_BUCKET})')
    parser.add_argument('--prefix',     default=DEFAULT_PREFIX,
                        help=f'Object prefix to list from (default: {DEFAULT_PREFIX})')
    parser.add_argument('--num-files',  type=int, default=DEFAULT_NUM_FILES,
                        help=f'Max files to read from the prefix (default: {DEFAULT_NUM_FILES})')

    # Test mode
    parser.add_argument('--mode', choices=['all', 'serial', 'parallel', 'native'],
                        default='all',
                        help='Which test(s) to run (default: all)')

    # Concurrency sweep
    parser.add_argument('--workers', type=int, nargs='+', default=DEFAULT_WORKERS,
                        metavar='N',
                        help=f'Worker counts for parallel + native tests '
                             f'(default: {DEFAULT_WORKERS})')

    # Library selection
    parser.add_argument('--libraries', nargs='+',
                        choices=['s3dlio', 'minio', 's3torchconnector'],
                        default=['s3dlio', 'minio', 's3torchconnector'],
                        metavar='LIB',
                        help='Libraries to test (default: all three)')

    # Optional write phase
    parser.add_argument('--write', action='store_true',
                        help='Write synthetic test objects before benchmarking; '
                             'use --write-prefix/--write-num-files/--write-size-mb '
                             'to control. Objects are written to the same --bucket.')
    parser.add_argument('--write-prefix',    default=WRITE_PREFIX,
                        help=f'Key prefix for synthetic objects (default: {WRITE_PREFIX})')
    parser.add_argument('--write-num-files', type=int, default=DEFAULT_WRITE_FILES,
                        help=f'Number of synthetic objects to write (default: {DEFAULT_WRITE_FILES})')
    parser.add_argument('--write-size-mb',   type=int, default=DEFAULT_WRITE_MB,
                        help=f'Size of each synthetic object in MB (default: {DEFAULT_WRITE_MB})')

    # Credentials
    parser.add_argument('--endpoint',   default=None, help='S3 endpoint URL')
    parser.add_argument('--access-key', default=None, help='AWS/MinIO access key')
    parser.add_argument('--secret-key', default=None, help='AWS/MinIO secret key')
    parser.add_argument('--region',     default=None, help='AWS region (default: us-east-1)')

    args = parser.parse_args()

    # ── Apply credentials ─────────────────────────────────────────────────────
    config = load_env_config()
    if args.endpoint:   config['AWS_ENDPOINT_URL']     = args.endpoint
    if args.access_key: config['AWS_ACCESS_KEY_ID']    = args.access_key
    if args.secret_key: config['AWS_SECRET_ACCESS_KEY'] = args.secret_key
    if args.region:     config['AWS_REGION']            = args.region
    apply_config(config)

    libraries     = args.libraries
    workers_sweep = sorted(set(args.workers))
    run_serial    = args.mode in ('all', 'serial')
    run_parallel  = args.mode in ('all', 'parallel')
    run_native    = args.mode in ('all', 'native') and 's3dlio' in libraries

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print("═" * 72)
    print("S3 LIBRARY GET BENCHMARK")
    print("═" * 72)
    print(f"  Endpoint:   {os.environ.get('AWS_ENDPOINT_URL', '(AWS S3 default)')}")
    print(f"  Libraries:  {', '.join(libraries)}")
    print(f"  Mode:       {args.mode}")
    if run_parallel or run_native:
        print(f"  Workers:    {workers_sweep}  (concurrency sweep)")

    # ── Optional write phase ──────────────────────────────────────────────────
    bucket = args.bucket
    prefix = args.prefix

    if args.write:
        print(f"\n── Write phase ──────────────────────────────────────────────────────────")
        keys = write_test_objects(
            bucket, args.write_prefix, args.write_num_files, args.write_size_mb)
        prefix = args.write_prefix    # benchmark from the freshly-written objects
        print(f"  Using {len(keys)} written objects for benchmark\n")
    else:
        # ── List objects ──────────────────────────────────────────────────────
        print(f"\n── Listing objects ──────────────────────────────────────────────────────")
        print(f"  Bucket: {bucket}  Prefix: {prefix}  (max {args.num_files})")
        keys = list_objects(bucket, prefix, args.num_files)
        if not keys:
            print(f"\nERROR: No objects found at s3://{bucket}/{prefix}")
            print("  Use --bucket / --prefix to point to an existing dataset, or")
            print("  use --write to create synthetic test objects first.")
            sys.exit(1)
        print(f"  Found {len(keys)} objects  (first: {keys[0]})")

    # Limit to num_files after listing (applies even after write)
    keys = keys[:args.num_files]

    # Estimate object size from first object for the banner
    try:
        import s3dlio
        probe = s3dlio.get(f"s3://{bucket}/{keys[0]}")
        obj_bytes = len(memoryview(probe))
    except Exception:
        obj_bytes = 0

    total_mb = len(keys) * obj_bytes / (1024**2) if obj_bytes else 0
    per_mb   = obj_bytes / (1024**2) if obj_bytes else 0
    print(f"  Objects:  {len(keys)} × {per_mb:.1f} MB = {total_mb:.0f} MB total\n")

    # ── Build library clients (one per library, shared across all tests) ──────
    minio_client   = _make_minio_client()
    s3torch_client = _make_s3torch_client() if 's3torchconnector' in libraries else None

    serial_results   = {}
    parallel_results = {}
    native_results   = {}

    # ── Serial test ───────────────────────────────────────────────────────────
    if run_serial:
        print("── Serial GET ───────────────────────────────────────────────────────────")
        serial_results = run_serial_test(
            libraries, bucket, keys, minio_client, s3torch_client)

    # ── Parallel test ─────────────────────────────────────────────────────────
    if run_parallel:
        print("\n── Parallel GET (ThreadPoolExecutor) ────────────────────────────────────")
        parallel_results = run_parallel_test(
            libraries, bucket, keys, workers_sweep, minio_client, s3torch_client)

    # ── s3dlio native get_many ────────────────────────────────────────────────
    if run_native:
        print("\n── s3dlio native get_many() ─────────────────────────────────────────────")
        native_results = run_native_test(bucket, keys, workers_sweep)

    # ── Results ───────────────────────────────────────────────────────────────
    if serial_results:
        print_serial_results(serial_results, len(keys))

    if parallel_results:
        print_parallel_results(parallel_results, workers_sweep, len(keys))

    if native_results:
        print_native_results(native_results, workers_sweep, len(keys), parallel_results)

    print()
    print("═" * 72)
    print("DONE")
    print("═" * 72)


if __name__ == '__main__':
    main()
