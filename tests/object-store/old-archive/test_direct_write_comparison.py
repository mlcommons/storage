#!/usr/bin/env python3
"""Direct-API write + read comparison across s3dlio, minio, and s3torchconnector.

Phases (per library, each in its own dedicated bucket):
  0. SETUP  — delete every object under the test prefix in that library's bucket
  1. WRITE  — write N objects in parallel (ThreadPoolExecutor, --write-workers)
  2. READ   — read all N objects back in parallel (ThreadPoolExecutor, --read-workers)
  3. REPORT — write GB/s, read GB/s, per-object latency

Libraries and their native APIs tested:
  s3dlio           objects < 32 MiB: s3dlio.put_bytes() → single PUT (no multipart overhead)
                   objects ≥ 32 MiB: MultipartUploadWriter.from_uri() → .write() → .close()
                   s3dlio.get(uri) for reads
  minio            objects < 32 MiB: client.put_object() → single PUT (no multipart overhead)
                   objects ≥ 32 MiB: _create_multipart_upload / _upload_part / _complete_multipart_upload
                   client.get_object(bucket, key) → .read() for reads
  s3torchconnector S3Client.put_object() → .write() → .close()
                   S3Client.get_object(bucket, key) → .read() for reads

Write path selection:
  Objects below MULTIPART_THRESHOLD (32 MiB, matching s3dlio's DEFAULT_S3_MULTIPART_THRESHOLD)
  use a single PUT — no three-phase initiate/upload/complete overhead.  Larger objects use
  multipart for throughput.  This mirrors real-world library default behaviour.

Data generation:
  dgen-py (the Python wrapper for the dgen-rs crate) generates random data before each
  write phase.  Generation time is NOT included in write throughput figures; only the
  actual I/O time is measured.  This gives accurate, apples-to-apples write numbers.

Bucket cleanup uses the minio client uniformly — it works on any S3-compatible
endpoint, so all three buckets are emptied the same way before tests begin.

Credential precedence: .env file < environment variables < CLI options
"""

import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add mlp-storage root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Default configuration ────────────────────────────────────────────────────

DEFAULT_NUM_FILES      = 100
DEFAULT_SIZE_MB        = 128
DEFAULT_CHUNK_MB       = 32
DEFAULT_PREFIX         = "bench"
DEFAULT_WRITE_WORKERS  = 8   # parallel object writes (all libraries)
DEFAULT_READ_WORKERS   = 8   # parallel object reads  (all libraries)
DEFAULT_MAX_IN_FLIGHT  = 8   # s3dlio per-object concurrent multipart parts

# Match s3dlio's DEFAULT_S3_MULTIPART_THRESHOLD (src/constants.rs).
# Objects below this size use a single PUT; at or above use multipart.
MULTIPART_THRESHOLD = 32 * 1024 * 1024  # 32 MiB

_default_bucket = os.environ.get('BUCKET') or os.environ.get('S3_BUCKET') or ''
LIBRARY_BUCKETS = {
    's3dlio':            os.environ.get('BUCKET_S3DLIO', _default_bucket),
    'minio':             os.environ.get('BUCKET_MINIO', _default_bucket),
    's3torchconnector':  os.environ.get('BUCKET_S3TORCH', _default_bucket),
}


# ── Credential loading ────────────────────────────────────────────────────────

def load_env_config():
    """Load config: .env first, then env vars override (CLI applied by caller)."""
    env_path = None
    for candidate in [
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
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
        print(f"Loaded credentials from: {env_path}")
    else:
        print("No .env file found, using environment variables")

    for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_ENDPOINT_URL', 'AWS_REGION']:
        if key in os.environ:
            config[key] = os.environ[key]

    return config


def apply_config(config: dict):
    for key, val in config.items():
        os.environ[key] = val


# ── Bucket cleanup (minio client works on any S3-compatible endpoint) ─────────

def _make_minio_client():
    """Build a minio.Minio client from environment credentials."""
    from minio import Minio
    endpoint_url = os.environ.get('AWS_ENDPOINT_URL', '')
    if endpoint_url.startswith('https://'):
        endpoint, secure = endpoint_url[8:], True
    elif endpoint_url.startswith('http://'):
        endpoint, secure = endpoint_url[7:], False
    else:
        endpoint = endpoint_url or 's3.amazonaws.com'
        secure = not bool(endpoint_url)
    return Minio(
        endpoint,
        access_key=os.environ['AWS_ACCESS_KEY_ID'],
        secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        secure=secure,
        region=os.environ.get('AWS_REGION', 'us-east-1'),
    )


def empty_prefix(bucket: str, prefix: str, label: str) -> int:
    """Delete every object under prefix/ in bucket.  Returns count deleted."""
    from minio.deleteobjects import DeleteObject
    client = _make_minio_client()
    full_prefix = prefix.rstrip('/') + '/'
    objects = list(client.list_objects(bucket, prefix=full_prefix, recursive=True))
    if not objects:
        print(f"  [{label}] bucket {bucket}/{full_prefix} already empty")
        return 0
    delete_list = [DeleteObject(obj.object_name) for obj in objects]
    errors = list(client.remove_objects(bucket, iter(delete_list)))
    if errors:
        raise RuntimeError(f"Delete errors in {bucket}: {errors}")
    print(f"  [{label}] deleted {len(delete_list)} object(s) from s3://{bucket}/{full_prefix}")
    return len(delete_list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _progress(label: str, done: int, total: int, elapsed: float, total_bytes: int):
    gb = total_bytes / (1024 ** 3)
    gbps = gb / elapsed if elapsed > 0 else 0
    print(f"  [{label}]  {done:>4}/{total}  {gbps:.3f} GB/s  ({elapsed:.1f}s elapsed)")


def _object_key(prefix: str, index: int) -> str:
    return f"{prefix.rstrip('/')}/obj-{index:05d}.dat"


def _generate_data(size_bytes: int, label: str) -> bytes:
    """Generate random data with dgen-py. NOT counted in write timing."""
    try:
        import dgen_py
    except ImportError:
        raise ImportError(
            "dgen-py is required for data generation. "
            "Install with: pip install dgen-py"
        )
    t = time.perf_counter()
    data = bytes(dgen_py.generate_buffer(size_bytes))  # convert BytesView → bytes for sliceability
    elapsed = time.perf_counter() - t
    gbps = size_bytes / (1024 ** 3) / elapsed if elapsed > 0 else 0
    print(f"  [{label}] dgen-py: {size_bytes / (1024**2):.0f} MiB in {elapsed:.3f}s "
          f"({gbps:.2f} GB/s) — NOT included in write throughput")
    return data

# ── Per-object write workers (called from ThreadPoolExecutor) ────────────────

def _write_one_s3dlio(args):
    """Write one object via s3dlio. Data is pre-generated by the caller (not timed here).

    single PUT for objects < 32 MiB, multipart at or above.
    """
    import s3dlio
    bucket, key, data, chunk_size, max_in_flight = args
    uri = f"s3://{bucket}/{key}"
    size_bytes = len(data)
    if size_bytes < MULTIPART_THRESHOLD:
        # Single PUT — no three-phase overhead, matches library default behaviour.
        s3dlio.put_bytes(uri, data)
    else:
        writer = s3dlio.MultipartUploadWriter.from_uri(
            uri, part_size=chunk_size, max_in_flight=max_in_flight, abort_on_drop=True,
        )
        offset = 0
        while offset < size_bytes:
            n = min(chunk_size, size_bytes - offset)
            writer.write(data[offset:offset + n])  # bytes slice of pre-generated buffer
            offset += n
        writer.close()
    return size_bytes


def _write_one_minio(args):
    """Write one object via minio. Data is pre-generated by the caller (not timed here).

    single PUT for objects < 32 MiB, multipart at or above.
    """
    import io
    from minio.datatypes import Part
    client, bucket, key, data, chunk_size = args
    size_bytes = len(data)
    if size_bytes < MULTIPART_THRESHOLD:
        # Single PUT — no three-phase overhead.
        client.put_object(bucket, key, io.BytesIO(data), size_bytes)
    else:
        upload_id = client._create_multipart_upload(bucket, key, {})
        parts = []
        part_num = 1
        offset = 0
        try:
            while offset < size_bytes:
                n = min(chunk_size, size_bytes - offset)
                etag = client._upload_part(
                    bucket_name=bucket, object_name=key,
                    data=data[offset:offset + n],  # bytes slice of pre-generated buffer
                    headers=None,
                    upload_id=upload_id, part_number=part_num,
                )
                parts.append(Part(part_num, etag))
                offset += n
                part_num += 1
            client._complete_multipart_upload(bucket, key, upload_id, parts)
        except Exception:
            try:
                client._abort_multipart_upload(bucket, key, upload_id)
            except Exception:
                pass
            raise
    return size_bytes


def _write_one_s3torch(args):
    """Write one object via s3torchconnector. Data is pre-generated by the caller (not timed here)."""
    s3_client, bucket, key, data, chunk_size = args
    size_bytes = len(data)
    writer = s3_client.put_object(bucket, key)
    offset = 0
    while offset < size_bytes:
        n = min(chunk_size, size_bytes - offset)
        writer.write(data[offset:offset + n])  # bytes slice of pre-generated buffer
        offset += n
    writer.close()
    return size_bytes


# ── Write phases (parallel via ThreadPoolExecutor) ────────────────────────────

def _run_parallel_writes(label, worker_fn, args_list, num_workers):
    """Execute per-object writes in parallel; return (total_bytes, elapsed)."""
    import threading
    t_start = time.perf_counter()
    total_written = 0
    done = 0
    lock = threading.Lock()
    num_files = len(args_list)
    report_every = max(1, num_files // 10)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(worker_fn, a): a for a in args_list}
        for fut in as_completed(futs):
            n = fut.result()   # raises on error, propagates to caller
            with lock:
                total_written += n
                done += 1
                if done % report_every == 0:
                    _progress(label, done, num_files,
                              time.perf_counter() - t_start, total_written)

    return total_written, time.perf_counter() - t_start


def write_s3dlio(bucket: str, prefix: str, num_files: int,
                 size_bytes: int, chunk_size: int,
                 max_in_flight: int, num_workers: int) -> dict:
    """Write num_files objects in parallel using s3dlio.

    Data is generated once with dgen-py before the write timer starts.
    """
    import s3dlio
    version = s3dlio.__version__
    keys = [_object_key(prefix, i) for i in range(num_files)]
    data = _generate_data(size_bytes, 's3dlio')   # excluded from write timing
    args_list = [(bucket, k, data, chunk_size, max_in_flight) for k in keys]
    total_written, elapsed = _run_parallel_writes(
        's3dlio write', _write_one_s3dlio, args_list, num_workers)
    return {'library': 's3dlio', 'version': version, 'keys': keys,
            'write_bytes': total_written, 'write_time': elapsed, 'ok': True}


def write_minio(bucket: str, prefix: str, num_files: int,
                size_bytes: int, chunk_size: int, num_workers: int) -> dict:
    """Write num_files objects in parallel using minio.

    Data is generated once with dgen-py before the write timer starts.
    """
    import minio as minio_module
    try:
        version = minio_module.__version__
    except AttributeError:
        version = "unknown"
    client = _make_minio_client()
    keys = [_object_key(prefix, i) for i in range(num_files)]
    data = _generate_data(size_bytes, 'minio ')   # excluded from write timing
    args_list = [(client, bucket, k, data, chunk_size) for k in keys]
    total_written, elapsed = _run_parallel_writes(
        'minio  write', _write_one_minio, args_list, num_workers)
    return {'library': 'minio', 'version': version, 'keys': keys,
            'write_bytes': total_written, 'write_time': elapsed, 'ok': True}


def write_s3torch(bucket: str, prefix: str, num_files: int,
                  size_bytes: int, chunk_size: int, num_workers: int) -> dict:
    """Write num_files objects in parallel using s3torchconnector.S3Client.

    Data is generated once with dgen-py before the write timer starts.
    """
    from s3torchconnector._s3client import S3Client, S3ClientConfig
    import s3torchconnector as s3torch_module
    try:
        version = s3torch_module.__version__
    except AttributeError:
        version = "unknown"
    region = os.environ.get('AWS_REGION', 'us-east-1')
    endpoint = os.environ.get('AWS_ENDPOINT_URL')
    cfg = S3ClientConfig(force_path_style=bool(endpoint), max_attempts=3)
    s3_client = S3Client(region=region, endpoint=endpoint, s3client_config=cfg)
    keys = [_object_key(prefix, i) for i in range(num_files)]
    data = _generate_data(size_bytes, 's3torch')  # excluded from write timing
    args_list = [(s3_client, bucket, k, data, chunk_size) for k in keys]
    total_written, elapsed = _run_parallel_writes(
        's3torch write', _write_one_s3torch, args_list, num_workers)
    return {'library': 's3torchconnector', 'version': version, 'keys': keys,
            'write_bytes': total_written, 'write_time': elapsed, 'ok': True}


# ── Read phases (parallel via ThreadPoolExecutor) ─────────────────────────────

def _read_one_s3dlio(args):
    import s3dlio
    bucket, key = args
    uri = f"s3://{bucket}/{key}"
    data = s3dlio.get(uri)
    return len(memoryview(data))


def _read_one_minio(args):
    client, bucket, key = args
    resp = client.get_object(bucket, key)
    try:
        data = resp.read()
        return len(data)
    finally:
        resp.close()
        resp.release_conn()


def _read_one_s3torch(args):
    s3_client, bucket, key = args
    reader = s3_client.get_object(bucket, key)
    data = reader.read()
    return len(data)


def read_s3dlio(bucket: str, keys: list, num_workers: int) -> dict:
    """Read all objects in parallel using s3dlio.get()."""
    t_start = time.perf_counter()
    total_read = 0
    done = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(_read_one_s3dlio, (bucket, k)): k for k in keys}
        for fut in as_completed(futs):
            total_read += fut.result()
            done += 1
            if done % max(1, len(keys) // 10) == 0:
                _progress('s3dlio read ', done, len(keys),
                          time.perf_counter() - t_start, total_read)
    return {'read_bytes': total_read, 'read_time': time.perf_counter() - t_start}


def read_minio(bucket: str, keys: list, num_workers: int) -> dict:
    """Read all objects in parallel using minio client.get_object()."""
    client = _make_minio_client()
    t_start = time.perf_counter()
    total_read = 0
    done = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(_read_one_minio, (client, bucket, k)): k for k in keys}
        for fut in as_completed(futs):
            total_read += fut.result()
            done += 1
            if done % max(1, len(keys) // 10) == 0:
                _progress('minio  read ', done, len(keys),
                          time.perf_counter() - t_start, total_read)
    return {'read_bytes': total_read, 'read_time': time.perf_counter() - t_start}


def read_s3torch(bucket: str, keys: list, num_workers: int) -> dict:
    """Read all objects in parallel using s3torchconnector S3Client.get_object()."""
    from s3torchconnector._s3client import S3Client, S3ClientConfig
    region = os.environ.get('AWS_REGION', 'us-east-1')
    endpoint = os.environ.get('AWS_ENDPOINT_URL')
    cfg = S3ClientConfig(force_path_style=bool(endpoint), max_attempts=3)
    s3_client = S3Client(region=region, endpoint=endpoint, s3client_config=cfg)

    t_start = time.perf_counter()
    total_read = 0
    done = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(_read_one_s3torch, (s3_client, bucket, k)): k for k in keys}
        for fut in as_completed(futs):
            total_read += fut.result()
            done += 1
            if done % max(1, len(keys) // 10) == 0:
                _progress('s3torch read', done, len(keys),
                          time.perf_counter() - t_start, total_read)
    return {'read_bytes': total_read, 'read_time': time.perf_counter() - t_start}


# ── Results table ─────────────────────────────────────────────────────────────

def print_results(results: list, num_files: int, size_mb: float,
                  write_workers: int, read_workers: int):
    total_mb = num_files * size_mb

    def gbps(b, t):
        return (b / (1024**3)) / t if t > 0 else 0.0

    print()
    print("=" * 88)
    print("WRITE + READ COMPARISON — RESULTS")
    print(f"  {num_files} objects × {size_mb:.0f} MB = {total_mb:.0f} MB per library  |  "
          f"write workers: {write_workers}   read workers: {read_workers}")
    print("=" * 88)
    hdr = f"  {'Library':<22} {'Version':<12} {'Write GB/s':>11} {'Read GB/s':>11} "
    hdr += f"{'Wr s/obj':>9} {'Rd s/obj':>9}"
    print(hdr)
    print(f"  {'-'*22} {'-'*12} {'-'*11} {'-'*11} {'-'*9} {'-'*9}")

    ok_results = [r for r in results if r.get('ok')]
    if ok_results:
        best_write = max(ok_results, key=lambda r: gbps(r['write_bytes'], r['write_time']))
        best_read  = max(ok_results, key=lambda r: gbps(r['read_bytes'], r['read_time']))
    else:
        best_write = best_read = None

    for r in results:
        lib = r['library']
        if not r.get('ok'):
            print(f"  {lib:<22} {'':12}  {'FAILED':>11}")
            continue

        wgbps = gbps(r['write_bytes'], r['write_time'])
        rgbps = gbps(r['read_bytes'],  r['read_time'])
        ws    = r['write_time'] / num_files
        rs    = r['read_time']  / num_files

        wmark = ' ◀W' if r is best_write else '   '
        rmark = ' ◀R' if r is best_read  else '   '

        print(f"  {lib:<22} {r['version']:<12} "
              f"{wgbps:>10.3f}{wmark} {rgbps:>10.3f}{rmark} "
              f"{ws:>8.3f}s {rs:>8.3f}s")

    print()
    print("  Write GB/s — parallel write throughput (all objects, ThreadPoolExecutor)")
    print("  Read GB/s  — parallel read throughput (all objects, ThreadPoolExecutor)")
    print("  Wr s/obj   — average time to write one object (write + commit)")
    print("  Rd s/obj   — average time to read one object (wall-clock, under parallelism)")
    print("  ◀W = fastest write    ◀R = fastest read")
    print()
    print("  Notes:")
    print("   • Write GB/s = pure I/O only: data generated with dgen-py BEFORE the write timer")
    print("   • Write workers = parallel object uploads; Read workers = parallel object downloads")
    print("   • Objects < 32 MiB: s3dlio uses put_bytes() (single PUT); minio uses put_object()")
    print("   • Objects ≥ 32 MiB: both use multipart upload (s3dlio max_in_flight per object)")
    print("   • minio multipart part uploads are sequential within each object")
    print("   • s3torchconnector buffers writes internally and uploads at close()")
    print("=" * 88)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--bucket-s3dlio',   default=LIBRARY_BUCKETS['s3dlio'],
                        help='Bucket for s3dlio test (env: BUCKET_S3DLIO or BUCKET)')
    parser.add_argument('--bucket-minio',    default=LIBRARY_BUCKETS['minio'],
                        help='Bucket for minio test (env: BUCKET_MINIO or BUCKET)')
    parser.add_argument('--bucket-s3torch',  default=LIBRARY_BUCKETS['s3torchconnector'],
                        help='Bucket for s3torchconnector test (env: BUCKET_S3TORCH or BUCKET)')
    parser.add_argument('--num-files',  type=int,   default=DEFAULT_NUM_FILES,
                        help=f'Objects to write and read per library (default: {DEFAULT_NUM_FILES})')
    parser.add_argument('--size-mb',    type=float, default=DEFAULT_SIZE_MB,
                        help=f'Size of each object in MB (default: {DEFAULT_SIZE_MB})')
    parser.add_argument('--chunk-mb',   type=int,   default=DEFAULT_CHUNK_MB,
                        help=f'Multipart chunk/part size in MB (default: {DEFAULT_CHUNK_MB})')
    parser.add_argument('--prefix',     default=DEFAULT_PREFIX,
                        help=f'S3 key prefix for test objects (default: {DEFAULT_PREFIX})')
    parser.add_argument('--write-workers', type=int, default=DEFAULT_WRITE_WORKERS,
                        help=f'Parallel write threads per library (default: {DEFAULT_WRITE_WORKERS})')
    parser.add_argument('--read-workers', type=int, default=DEFAULT_READ_WORKERS,
                        help=f'Parallel read threads per library (default: {DEFAULT_READ_WORKERS})')
    parser.add_argument('--max-in-flight', type=int, default=DEFAULT_MAX_IN_FLIGHT,
                        help=f's3dlio per-object concurrent multipart parts (default: {DEFAULT_MAX_IN_FLIGHT})')
    parser.add_argument('--library', choices=['s3dlio', 'minio', 's3torchconnector'],
                        nargs='+', dest='libraries', metavar='LIBRARY',
                        help='Library/libraries to test: s3dlio minio s3torchconnector '
                             '(default: all three). Example: --library s3dlio minio')
    parser.add_argument('--endpoint',   default=None, help='S3 endpoint URL')
    parser.add_argument('--access-key', default=None, help='AWS/MinIO access key')
    parser.add_argument('--secret-key', default=None, help='AWS/MinIO secret key')
    parser.add_argument('--region',     default=None, help='AWS region (default: us-east-1)')
    args = parser.parse_args()

    config = load_env_config()
    if args.endpoint:   config['AWS_ENDPOINT_URL']      = args.endpoint
    if args.access_key: config['AWS_ACCESS_KEY_ID']      = args.access_key
    if args.secret_key: config['AWS_SECRET_ACCESS_KEY']  = args.secret_key
    if args.region:     config['AWS_REGION']             = args.region
    apply_config(config)

    libraries  = args.libraries or ['s3dlio', 'minio', 's3torchconnector']
    size_bytes = int(args.size_mb * 1024 * 1024)
    chunk_size = args.chunk_mb * 1024 * 1024
    total_gb   = args.num_files * args.size_mb / 1024

    buckets = {
        's3dlio':           args.bucket_s3dlio,
        'minio':            args.bucket_minio,
        's3torchconnector': args.bucket_s3torch,
    }

    # Validate that buckets are set for the libraries being tested
    import sys as _sys
    missing = [lib for lib in libraries if not buckets.get(lib)]
    if missing:
        print(
            f"ERROR: No bucket specified for: {', '.join(missing)}\n"
            "  Set BUCKET (or BUCKET_S3DLIO / BUCKET_MINIO / BUCKET_S3TORCH) in .env,\n"
            "  or pass --bucket-s3dlio / --bucket-minio / --bucket-s3torch on the CLI.",
            file=_sys.stderr,
        )
        _sys.exit(1)

    print()
    print("=" * 88)
    print("DIRECT API WRITE + READ COMPARISON")
    print("=" * 88)
    print(f"  Endpoint:     {os.environ.get('AWS_ENDPOINT_URL', '(AWS S3)')}")
    print(f"  Libraries:    {', '.join(libraries)}")
    print(f"  Objects:      {args.num_files} × {args.size_mb:.0f} MB = {total_gb:.1f} GB per library")
    print(f"  Chunk size:   {args.chunk_mb} MB  |  s3dlio max_in_flight: {args.max_in_flight}")
    print(f"  Write workers: {args.write_workers}  |  Read workers: {args.read_workers}  |  Prefix: {args.prefix}/")
    print(f"  Buckets:      s3dlio={buckets['s3dlio']}  "
          f"minio={buckets['minio']}  s3torch={buckets['s3torchconnector']}")
    print()
    print("  Each library uses its own dedicated bucket.")
    print("  Buckets are emptied before writing so every run starts from a clean state.")
    print("=" * 88)

    # ── Phase 0: empty each library's bucket prefix ───────────────────────────
    print("\n── Phase 0: Cleanup ──────────────────────────────────────────────────────────")
    for lib in libraries:
        try:
            empty_prefix(buckets[lib], args.prefix, lib)
        except Exception as e:
            print(f"  [{lib}] ⚠️  cleanup failed (continuing): {e}")
    print()

    results = []

    for lib in libraries:
        bucket = buckets[lib]
        print(f"\n── {lib}  →  s3://{bucket}/{args.prefix}/  ─────────────────────────────────")

        try:
            # Write
            print(f"  Writing {args.num_files} objects …")
            if lib == 's3dlio':
                wr = write_s3dlio(bucket, args.prefix, args.num_files,
                                  size_bytes, chunk_size,
                                  args.max_in_flight, args.write_workers)
            elif lib == 'minio':
                wr = write_minio(bucket, args.prefix, args.num_files,
                                 size_bytes, chunk_size, args.write_workers)
            else:
                wr = write_s3torch(bucket, args.prefix, args.num_files,
                                   size_bytes, chunk_size, args.write_workers)
            print(f"  Write done: {wr['write_bytes']/(1024**3):.2f} GB in "
                  f"{wr['write_time']:.1f}s  "
                  f"({wr['write_bytes']/(1024**3)/wr['write_time']:.3f} GB/s)")

            # Read
            print(f"  Reading {args.num_files} objects ({args.read_workers} workers) …")
            if lib == 's3dlio':
                rd = read_s3dlio(bucket, wr['keys'], args.read_workers)
            elif lib == 'minio':
                rd = read_minio(bucket, wr['keys'], args.read_workers)
            else:
                rd = read_s3torch(bucket, wr['keys'], args.read_workers)
            print(f"  Read done:  {rd['read_bytes']/(1024**3):.2f} GB in "
                  f"{rd['read_time']:.1f}s  "
                  f"({rd['read_bytes']/(1024**3)/rd['read_time']:.3f} GB/s)")

            results.append({**wr, **rd})

        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({'library': lib, 'ok': False})

    print_results(results, args.num_files, args.size_mb, args.write_workers, args.read_workers)

    failed = [r['library'] for r in results if not r.get('ok')]
    if failed:
        print(f"❌ Failed libraries: {', '.join(failed)}")
        sys.exit(1)
    print("✅ All tests passed.")
    print()


if __name__ == '__main__':
    main()
