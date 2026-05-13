#!/usr/bin/env python3
"""
StreamingCheckpointing with s3dlio backend.

Writes a configurable-size checkpoint to S3 using the streaming producer-consumer
pipeline: dgen-py generates data in parallel while s3dlio uploads it, keeping
memory usage constant at ~128 MB regardless of checkpoint size.

Configuration:
  32 MB chunks, 4 buffers (128 MB pool), fadvise=none
  300s SIGALRM timeout to detect hung S3 connections early

Credential precedence (lowest → highest):
  .env file  <  environment variables  <  CLI options

Usage:
  python test_s3dlio_checkpoint.py --bucket my-bucket
  python test_s3dlio_checkpoint.py --bucket my-bucket --size-gb 4.0
  python test_s3dlio_checkpoint.py --s3-uri s3://my-bucket/ckpt/test.dat --size-gb 8.0
"""

import os
import sys
import time
import signal
import argparse
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_env_config() -> dict:
    """Load config from .env, then let environment variables override."""
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

    # Environment variables override .env
    for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_ENDPOINT_URL', 'AWS_REGION']:
        if key in os.environ:
            config[key] = os.environ[key]

    return config


def apply_config(config: dict):
    for key, val in config.items():
        os.environ[key] = val


class TimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds: int, message: str = 'Operation timed out'):
    """SIGALRM-based timeout context manager (Unix only)."""
    def _handler(signum, frame):
        raise TimeoutException(message)

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def run(s3_uri: str, size_gb: float):
    from mlpstorage_py.checkpointing import StreamingCheckpointing

    total_bytes = int(size_gb * (1024 ** 3))
    endpoint = os.environ.get('AWS_ENDPOINT_URL', '(default)')
    access_key = os.environ.get('AWS_ACCESS_KEY_ID', '')

    print()
    print("=" * 80)
    print("S3DLIO STREAMING CHECKPOINT TEST")
    print("=" * 80)
    print(f"Endpoint: {endpoint}")
    print(f"URI:      {s3_uri}")
    print(f"Size:     {size_gb} GB  ({total_bytes:,} bytes)")
    print(f"Config:   32 MB chunks, 4 buffers (128 MB pool), fadvise=none")
    if access_key:
        print(f"Access:   {access_key[:8]}...{access_key[-4:]}")
    print("=" * 80)
    print()

    try:
        import s3dlio
        print(f"  s3dlio  {s3dlio.__version__}  ✅")
    except ImportError:
        print("  s3dlio  ❌  not installed — pip install s3dlio")
        sys.exit(1)

    try:
        import dgen_py
        print(f"  dgen-py {dgen_py.__version__}  ✅")
    except ImportError:
        print("  dgen-py ❌  not installed — pip install dgen-py")
        sys.exit(1)

    print()
    checkpoint = StreamingCheckpointing(
        chunk_size=32 * 1024 * 1024,
        num_buffers=4,
        use_dgen=True,
        backend='s3dlio',
        fadvise_mode='none',
    )
    print("StreamingCheckpointing ready  (backend=s3dlio, 32 MB chunks × 4 buffers)")
    print()
    print(f"Writing {size_gb} GB → {s3_uri}  [timeout: 300s]")
    print()

    start_time = time.perf_counter()
    try:
        with timeout(300, f"Write timed out after 300s  (size={size_gb:.2f} GB)"):
            result = checkpoint.save(s3_uri, total_bytes)
        elapsed = time.perf_counter() - start_time
    except TimeoutException as e:
        elapsed = time.perf_counter() - start_time
        print(f"\n❌ TIMEOUT after {elapsed:.0f}s: {e}")
        print("   Check S3 endpoint connectivity and credentials.")
        sys.exit(1)
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print(f"\n❌ Error after {elapsed:.1f}s: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)
    print("✅ COMPLETED")
    print("=" * 80)
    print(f"  Wall time:  {elapsed:.2f}s")

    if result:
        gen_time = result.get('gen_time', 0)
        io_time = result.get('io_time', 0)
        if gen_time:
            print(f"  Generation: {gen_time:.2f}s  ({result.get('gen_throughput_gbps', 0):.2f} GB/s)")
        if io_time:
            print(f"  I/O:        {io_time:.2f}s  ({result.get('io_throughput_gbps', 0):.2f} GB/s)")

    overall = (total_bytes / (1024 ** 3)) / elapsed
    print(f"  Overall:    {overall:.2f} GB/s")
    print(f"  URI:        {s3_uri}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='StreamingCheckpointing with s3dlio backend',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  python test_s3dlio_checkpoint.py --bucket my-bucket
  python test_s3dlio_checkpoint.py --bucket my-bucket --size-gb 4.0
  python test_s3dlio_checkpoint.py --s3-uri s3://my-bucket/ckpt/test.dat --size-gb 8.0
        """,
    )
    parser.add_argument('--bucket', default=os.environ.get('S3_BUCKET', 'bucket-s3dlio'),
                        help='S3 bucket name')
    parser.add_argument('--key', default=None,
                        help='Object key (default: auto-generated with timestamp)')
    parser.add_argument('--s3-uri', default=None,
                        help='Full S3 URI — overrides --bucket and --key')
    parser.add_argument('--size-gb', type=float, default=1.0,
                        help='Checkpoint size in GB')
    parser.add_argument('--endpoint', default=None,
                        help='S3 endpoint URL (e.g. http://minio-host:9000)')
    parser.add_argument('--access-key', default=None, help='AWS access key ID')
    parser.add_argument('--secret-key', default=None, help='AWS secret access key')
    parser.add_argument('--region', default=None, help='AWS region')
    args = parser.parse_args()

    # Credential precedence: .env < env vars < CLI
    config = load_env_config()
    if args.endpoint:
        config['AWS_ENDPOINT_URL'] = args.endpoint
    if args.access_key:
        config['AWS_ACCESS_KEY_ID'] = args.access_key
    if args.secret_key:
        config['AWS_SECRET_ACCESS_KEY'] = args.secret_key
    if args.region:
        config['AWS_REGION'] = args.region
    apply_config(config)

    if args.s3_uri:
        s3_uri = args.s3_uri
    else:
        key = args.key or f"test/checkpoint-{int(time.time())}.dat"
        s3_uri = f"s3://{args.bucket}/{key}"

    run(s3_uri, args.size_gb)


if __name__ == '__main__':
    main()
