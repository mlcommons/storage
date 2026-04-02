#!/usr/bin/env python3
"""
S3 Connectivity Test — all 3 libraries
Tests minio, s3dlio, and s3torchconnector against a live S3 endpoint.

Credentials are loaded from a .env file (defaults to the repo root .env).

Usage:
    cd /home/eval/Documents/Code/mlp-storage
    source .venv/bin/activate

    # Use per-library buckets (recommended):
    python tests/integration/test_s3_connectivity.py \\
        --minio-bucket bucket-minio \\
        --s3dlio-bucket bucket-s3dlio \\
        --s3torch-bucket bucket-s3torch

    # Use a single bucket for all libraries:
    python tests/integration/test_s3_connectivity.py --bucket test-bucket

    # Override endpoint:
    python tests/integration/test_s3_connectivity.py \\
        --endpoint http://10.9.0.21 \\
        --minio-bucket bucket-minio \\
        --s3dlio-bucket bucket-s3dlio \\
        --s3torch-bucket bucket-s3torch

    # Test only specific libraries:
    python tests/integration/test_s3_connectivity.py \\
        --libraries minio s3dlio \\
        --minio-bucket bucket-minio \\
        --s3dlio-bucket bucket-s3dlio
"""

import argparse
import os
import sys
import time
from pathlib import Path


# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="S3 connectivity test for all 3 libraries",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Per-library buckets take precedence over --bucket.
If --bucket is given as the only bucket option, all libraries use it.

Examples:
  %(prog)s --minio-bucket bucket-minio --s3dlio-bucket bucket-s3dlio --s3torch-bucket bucket-s3torch
  %(prog)s --bucket test-bucket
  %(prog)s --libraries s3dlio --s3dlio-bucket bucket-s3dlio
    """
)
parser.add_argument("--endpoint",      metavar="URL",
                    help="Endpoint URL (overrides .env AWS_ENDPOINT_URL)")
parser.add_argument("--bucket",        metavar="BUCKET",
                    help="Default bucket for all libraries (overrides .env S3_BUCKET)")
parser.add_argument("--minio-bucket",  metavar="BUCKET",
                    help="Bucket for minio tests (overrides --bucket)")
parser.add_argument("--s3dlio-bucket", metavar="BUCKET",
                    help="Bucket for s3dlio tests (overrides --bucket)")
parser.add_argument("--s3torch-bucket", metavar="BUCKET",
                    help="Bucket for s3torchconnector tests (overrides --bucket)")
parser.add_argument("--libraries", nargs="+",
                    choices=["minio", "s3dlio", "s3torchconnector"],
                    default=["minio", "s3dlio", "s3torchconnector"],
                    help="Libraries to test (default: all 3)")
parser.add_argument("--env-file", metavar="PATH",
                    help="Path to .env credentials file (default: auto-detected)")
args = parser.parse_args()


# ── Load credentials from .env ────────────────────────────────────────────────
if args.env_file:
    env_path = Path(args.env_file)
else:
    # Search from script location upward, then CWD
    env_path = None
    for candidate in [
        Path(__file__).parent.parent.parent / ".env",  # repo root
        Path.cwd() / ".env",
    ]:
        if candidate.exists():
            env_path = candidate
            break
    if env_path is None:
        print("ERROR: No .env file found. Use --env-file to specify one.")
        sys.exit(1)

print(f"Loading credentials from: {env_path}")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

ENDPOINT = args.endpoint or os.environ.get("AWS_ENDPOINT_URL", "")
if not ENDPOINT:
    print("ERROR: No endpoint set. Use --endpoint or set AWS_ENDPOINT_URL in .env")
    sys.exit(1)

ACCESS  = os.environ.get("AWS_ACCESS_KEY_ID", "")
SECRET  = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
REGION  = os.environ.get("AWS_REGION", "us-east-1")
PREFIX  = "connectivity-test"

# Push endpoint override into env for libraries that read it directly
os.environ["AWS_ENDPOINT_URL"] = ENDPOINT

# Resolve per-library buckets (per-library > --bucket > .env > fallback)
_default_bucket = args.bucket or os.environ.get("S3_BUCKET", "")
BUCKETS = {
    "minio":            args.minio_bucket  or _default_bucket,
    "s3dlio":           args.s3dlio_bucket or _default_bucket,
    "s3torchconnector": args.s3torch_bucket or _default_bucket,
}

# Validate that every selected library has a bucket
missing = [lib for lib in args.libraries if not BUCKETS[lib]]
if missing:
    print(f"ERROR: No bucket specified for: {', '.join(missing)}")
    print("Use --minio-bucket / --s3dlio-bucket / --s3torch-bucket or --bucket")
    sys.exit(1)

print(f"Endpoint : {ENDPOINT}")
print(f"Region   : {REGION}")
for lib in args.libraries:
    print(f"Bucket ({lib:<16}): {BUCKETS[lib]}")
print()

TEST_DATA = b"s3-connectivity-test-payload-" + str(time.time()).encode()
results: dict[str, str] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def section(name: str):
    print("=" * 60)
    print(f"  {name}")
    print("=" * 60)

def ok(msg: str):   print(f"  \033[32m✓\033[0m {msg}")
def fail(msg: str): print(f"  \033[31m✗\033[0m {msg}")


# ── 1. minio ──────────────────────────────────────────────────────────────────
if "minio" in args.libraries:
    section(f"minio  →  s3://{BUCKETS['minio']}")
    try:
        from minio import Minio
        from urllib.parse import urlparse
        import io

        parsed = urlparse(ENDPOINT)
        secure = parsed.scheme == "https"
        host   = parsed.netloc  # strip scheme

        client = Minio(host, access_key=ACCESS, secret_key=SECRET, secure=secure)
        bucket = BUCKETS["minio"]
        key    = f"{PREFIX}/minio-test.bin"

        client.put_object(bucket, key, io.BytesIO(TEST_DATA), len(TEST_DATA))
        ok(f"put_object  → s3://{bucket}/{key}")

        resp = client.get_object(bucket, key)
        body = resp.read()
        resp.close()
        assert body == TEST_DATA, "Data mismatch!"
        ok(f"get_object  → {len(body)} bytes verified")

        objs = list(client.list_objects(bucket, prefix=PREFIX + "/"))
        ok(f"list_objects → {len(objs)} object(s) found")

        client.remove_object(bucket, key)
        ok("remove_object → deleted")

        results["minio"] = "PASS"
    except Exception as e:
        fail(str(e))
        results["minio"] = f"FAIL: {e}"
    print()


# ── 2. s3dlio ─────────────────────────────────────────────────────────────────
if "s3dlio" in args.libraries:
    section(f"s3dlio  →  s3://{BUCKETS['s3dlio']}")
    try:
        import s3dlio

        ok(f"s3dlio version: {s3dlio.__version__}")
        bucket = BUCKETS["s3dlio"]
        uri    = f"s3://{bucket}/{PREFIX}/s3dlio-test.bin"

        t0 = time.time()
        s3dlio.put_bytes(uri, TEST_DATA)
        ok(f"put_bytes   → {uri}  ({time.time()-t0:.3f}s)")

        t0 = time.time()
        body = s3dlio.get(uri)
        ok(f"get         → {len(body)} bytes  ({time.time()-t0:.3f}s)")
        assert bytes(body) == TEST_DATA, "Data mismatch!"
        ok("data verified ✓")

        t0 = time.time()
        uris = s3dlio.list_full_uris(f"s3://{bucket}/{PREFIX}/")
        ok(f"list_full_uris → {len(uris)} object(s)  ({time.time()-t0:.3f}s)")
        assert uri in uris, f"Expected '{uri}' in list_full_uris result {uris}"

        s3dlio.delete(uri)
        ok("delete      → deleted")

        results["s3dlio"] = "PASS"
    except Exception as e:
        fail(str(e))
        results["s3dlio"] = f"FAIL: {e}"
    print()


# ── 3. s3torchconnector ───────────────────────────────────────────────────────
if "s3torchconnector" in args.libraries:
    section(f"s3torchconnector  →  s3://{BUCKETS['s3torchconnector']}")
    try:
        from s3torchconnector import S3Checkpoint

        bucket = BUCKETS["s3torchconnector"]
        key    = f"s3://{bucket}/{PREFIX}/s3torch-test.bin"

        checkpoint = S3Checkpoint(region=REGION)

        t0 = time.time()
        with checkpoint.writer(key) as writer:
            writer.write(TEST_DATA)
        ok(f"writer.write → {key}  ({time.time()-t0:.3f}s)")

        t0 = time.time()
        with checkpoint.reader(key) as reader:
            body = reader.read()
        ok(f"reader.read  → {len(body)} bytes  ({time.time()-t0:.3f}s)")
        assert body == TEST_DATA, "Data mismatch!"
        ok("data verified ✓")

        results["s3torchconnector"] = "PASS"
    except Exception as e:
        fail(str(e))
        results["s3torchconnector"] = f"FAIL: {e}"
    print()


# ── Summary ───────────────────────────────────────────────────────────────────
section("SUMMARY")
all_pass = True
for lib in args.libraries:
    status = results.get(lib, "SKIPPED")
    if status == "PASS":
        ok(f"{lib:<22}  PASS  (s3://{BUCKETS[lib]})")
    else:
        fail(f"{lib:<22}  {status}")
        all_pass = False
print()
sys.exit(0 if all_pass else 1)
