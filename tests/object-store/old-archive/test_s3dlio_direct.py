#!/usr/bin/env python3
"""
Direct s3dlio write test.

Tests:
  1. Streaming writer via create_s3_writer (PyObjectWriter API: write_chunk + finalize)
  2. Multipart upload via MultipartUploadWriter (write + close)

Credential precedence: .env file < environment variables < CLI options
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add mlp-storage root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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

    # Environment variables override .env
    for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_ENDPOINT_URL', 'AWS_REGION']:
        if key in os.environ:
            config[key] = os.environ[key]

    return config


def apply_config(config: dict):
    for key, val in config.items():
        os.environ[key] = val


def run_tests(bucket: str):
    import s3dlio

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          DIRECT S3DLIO TEST - No Multiprocessing            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"s3dlio version: {s3dlio.__version__}")
    print(f"Endpoint:       {os.environ.get('AWS_ENDPOINT_URL', '(default)')}")
    print(f"Bucket:         {bucket}")
    print()

    data_size = 16 * 1024 * 1024  # 16 MB
    data = bytearray(data_size)

    # ── Test 1: Streaming writer (PyObjectWriter) ───────────────────────────
    print("=" * 70)
    print("TEST 1: Streaming Writer (create_s3_writer — write_chunk + finalize)")
    print("=" * 70)

    uri = f"s3://{bucket}/direct_test_16mb.dat"
    try:
        print(f"Creating writer for: {uri}")
        options = s3dlio.PyWriterOptions()
        options.with_buffer_size(4 * 1024 * 1024)
        writer = s3dlio.create_s3_writer(uri, options)
        print("✅ Writer created")

        print(f"Writing {data_size / (1024**2):.0f} MB...")
        start = time.perf_counter()
        writer.write_chunk(data)
        elapsed = time.perf_counter() - start
        print(f"✅ write_chunk in {elapsed:.3f}s  ({data_size / (1024**2) / elapsed:.1f} MB/s)")

        print("Finalizing...")
        fin_start = time.perf_counter()
        bytes_written, compressed = writer.finalize()
        fin_elapsed = time.perf_counter() - fin_start
        print(f"✅ finalize in {fin_elapsed:.3f}s")
        print(f"   bytes_written={bytes_written:,}  compressed={compressed:,}")

        print()
        print("✅ TEST 1 PASSED!")
        print()
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Test 2: MultipartUploadWriter ───────────────────────────────────────
    # NOTE: this test exercises the MultipartUploadWriter API directly to verify
    # the three-phase multipart protocol works.  In real workloads, objects below
    # 32 MiB should use s3dlio.put_bytes() (single PUT) — not MultipartUploadWriter
    # — to avoid the unnecessary create/upload/complete round-trip overhead.
    print("=" * 70)
    print("TEST 2: MultipartUploadWriter (write + close) — explicit API test")
    print("=" * 70)

    uri2 = f"s3://{bucket}/multipart_test_16mb.dat"
    try:
        print(f"Creating multipart writer for: {uri2}")
        writer2 = s3dlio.MultipartUploadWriter.from_uri(
            uri2,
            part_size=16 * 1024 * 1024,
            max_in_flight=1,
            abort_on_drop=True,
        )
        print("✅ Multipart writer created")

        print(f"Writing {data_size / (1024**2):.0f} MB...")
        start = time.perf_counter()
        bytes_written2 = writer2.write(data)
        elapsed = time.perf_counter() - start
        print(f"✅ write {bytes_written2:,} bytes in {elapsed:.3f}s  ({bytes_written2 / (1024**2) / elapsed:.1f} MB/s)")

        print("Closing multipart writer...")
        close_start = time.perf_counter()
        result2 = writer2.close()
        close_elapsed = time.perf_counter() - close_start
        print(f"✅ Closed in {close_elapsed:.3f}s")
        print(f"   Result: {result2}")

        print()
        print("✅ TEST 2 PASSED!")
        print()
    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 70)
    print("✅ ALL TESTS PASSED - S3DLIO WORKING!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Direct s3dlio write test (streaming + multipart)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--bucket', default=os.environ.get('S3_BUCKET', 'bucket-s3dlio'),
                        help='S3 bucket name')
    parser.add_argument('--endpoint', default=None,
                        help='S3 endpoint URL (e.g., http://minio-host:9000)')
    parser.add_argument('--access-key', default=None, help='AWS/MinIO access key')
    parser.add_argument('--secret-key', default=None, help='AWS/MinIO secret key')
    parser.add_argument('--region', default=None, help='AWS region')
    args = parser.parse_args()

    # Load config: .env < env vars < CLI
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

    run_tests(args.bucket)


if __name__ == '__main__':
    main()
