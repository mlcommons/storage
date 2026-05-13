#!/usr/bin/env python3
"""
Zero-copy BytesView benchmark vs old bytes() path.
Tests whether the BytesView fast path in write() eliminates the GIL-held memcpy bottleneck.
"""
import s3dlio
import concurrent.futures
import os
import time

os.environ['AWS_ENDPOINT_URL'] = 'http://127.0.0.1:9101'
os.environ['AWS_ACCESS_KEY_ID'] = 'testkey'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'testsecret'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

SHAPE = [6053, 6053, 1]

print(f"s3dlio version: {s3dlio.__version__ if hasattr(s3dlio, '__version__') else 'unknown'}")

# --- Generate buffers once ---
print("Generating test buffers...")
t0 = time.perf_counter()
buf_bv = s3dlio.generate_npz_bytes(shape=SHAPE)   # BytesView (no bytes() conversion)
gen_time = time.perf_counter() - t0
file_mib = len(buf_bv) / (1024*1024)
print(f"  BytesView: {file_mib:.1f} MiB  generated in {gen_time*1000:.0f}ms")

t0 = time.perf_counter()
buf_bytes = bytes(buf_bv)                          # OLD: explicit bytes() copy
conv_time = time.perf_counter() - t0
print(f"  bytes() conversion: {conv_time*1000:.0f}ms")
print()

def upload_bv(i, prefix="zc"):
    with s3dlio.MultipartUploadWriter.from_uri(f's3://mlp-s3dlio/bench/{prefix}{i}.npz') as w:
        w.write(buf_bv)  # BytesView fast path

def upload_bytes(i, prefix="old"):
    with s3dlio.MultipartUploadWriter.from_uri(f's3://mlp-s3dlio/bench/{prefix}{i}.npz') as w:
        w.write(buf_bytes)  # old bytes path

def run_bench(fn, label, N=32):
    # warmup
    fn(0)
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
        list(pool.map(fn, range(N)))
    elapsed = time.perf_counter() - t0
    tput = N * file_mib / elapsed
    print(f"  n={N}: {tput:6.0f} MiB/s  ({elapsed:.2f}s)")
    return tput

# --- OLD path (bytes) ---
print("=== OLD PATH: bytes() ===")
for N in [8, 16, 32, 48]:
    run_bench(lambda i, N=N: upload_bytes(i), "bytes", N)

print()

# --- NEW path (BytesView zero-copy) ---
print("=== NEW PATH: BytesView zero-copy ===")
for N in [8, 16, 32, 48]:
    run_bench(lambda i, N=N: upload_bv(i), "zc", N)
