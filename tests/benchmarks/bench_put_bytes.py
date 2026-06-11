#!/usr/bin/env python3
"""
Compare MultipartUploadWriter vs put_bytes() throughput.
put_bytes() does entire upload in ONE py.detach() → only 1 GIL re-acquisition per file.
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
buf_bv = s3dlio.generate_npz_bytes(shape=SHAPE)
file_mib = len(buf_bv) / (1024*1024)
print(f"File size: {file_mib:.1f} MiB  (type={type(buf_bv).__name__})")

# === Verify put_bytes works at all ===
print("\nVerifying put_bytes...")
try:
    s3dlio.put_bytes('s3://mlp-s3dlio/bench_pb/verify.npz', buf_bv)
    print("  put_bytes: OK")
except Exception as e:
    print(f"  put_bytes FAILED: {e}")
    import sys; sys.exit(1)

# === MultipartUploadWriter (baseline) ===
def upload_mpu(i):
    with s3dlio.MultipartUploadWriter.from_uri(f's3://mlp-s3dlio/bench_pb/mpu_{i}.npz') as w:
        w.write(buf_bv)

# === put_bytes (single GIL release) ===
def upload_put(i):
    s3dlio.put_bytes(f's3://mlp-s3dlio/bench_pb/put_{i}.npz', buf_bv)

def run_bench(fn, label, n_workers, n_files):
    # warmup
    fn(9999)
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(fn, range(n_files)))
    elapsed = time.perf_counter() - t0
    total_mib = n_files * file_mib
    rate = total_mib / elapsed
    print(f"  {label:30s}  n={n_workers:3d}  {rate:6.0f} MiB/s  ({elapsed:.2f}s for {n_files} files)")
    return rate

print("\n=== Throughput comparison ===")
for n in [1, 8, 16, 32, 48, 64]:
    nf = max(n, 32)
    run_bench(upload_mpu, "MultipartUploadWriter", n, nf)
    run_bench(upload_put, "put_bytes()", n, nf)
    print()
