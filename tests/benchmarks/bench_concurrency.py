#!/usr/bin/env python3
"""
Test: configure_for_concurrency + setswitchinterval effects on throughput.
Must be the FIRST s3dlio-related script run (runtime not yet initialized).
"""
import sys
import os
import s3dlio
import concurrent.futures
import time

# MUST be called BEFORE any S3 I/O to affect runtime thread count
s3dlio.configure_for_concurrency(64)

os.environ['AWS_ENDPOINT_URL'] = 'http://127.0.0.1:9101'
os.environ['AWS_ACCESS_KEY_ID'] = 'testkey'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'testsecret'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

SHAPE = [6053, 6053, 1]
buf_bv = s3dlio.generate_npz_bytes(shape=SHAPE)
file_mib = len(buf_bv) / (1024*1024)
print(f"File size: {file_mib:.1f} MiB")
print(f"Python switch interval: {sys.getswitchinterval()*1000:.1f}ms  (default 5ms)")

def upload_mpu(i):
    with s3dlio.MultipartUploadWriter.from_uri(f's3://mlp-s3dlio/bench_cc/pt{i}.npz') as w:
        w.write(buf_bv)

# Warmup
upload_mpu(9999)

def run_bench(fn, label, n):
    nf = max(n, 32)
    fn(9999)  # warmup this n
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(fn, range(nf)))
    elapsed = time.perf_counter() - t0
    rate = nf * file_mib / elapsed
    print(f"  {label}  n={n:3d}  {rate:6.0f} MiB/s  ({elapsed:.2f}s)")
    return rate

print("\n=== Baseline (configure_for_concurrency=64, setswitchinterval=5ms) ===")
for n in [32, 48, 64]:
    run_bench(upload_mpu, "MPU", n)

print("\n=== With setswitchinterval(0.001) = 1ms ===")
sys.setswitchinterval(0.001)
print(f"Switch interval: {sys.getswitchinterval()*1000:.1f}ms")
for n in [32, 48, 64]:
    run_bench(upload_mpu, "MPU", n)

print("\n=== With setswitchinterval(0.0001) = 0.1ms ===")
sys.setswitchinterval(0.0001)
print(f"Switch interval: {sys.getswitchinterval()*1000:.2f}ms")
for n in [32, 48, 64]:
    run_bench(upload_mpu, "MPU", n)
