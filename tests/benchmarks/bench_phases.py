#!/usr/bin/env python3
"""
Phase timing: isolate where the wall-clock time goes per upload.
Measures from_uri, write, and close separately to find the GIL bottleneck.
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
print(f"File size: {file_mib:.1f} MiB")

# Measure phases: from_uri, write, close (manual context management)
times_from_uri = []
times_write = []
times_close = []

def upload_timed(i):
    t0 = time.perf_counter()
    w = s3dlio.MultipartUploadWriter.from_uri(f's3://mlp-s3dlio/bench/pt{i}.npz')
    t1 = time.perf_counter()
    w.write(buf_bv)
    t2 = time.perf_counter()
    w.close()
    t3 = time.perf_counter()
    return (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000

# warmup
upload_timed(0)

# === Single thread (no contention) ===
print("\n=== N=1 (no contention) ===")
results = [upload_timed(i) for i in range(4)]
for r in results:
    print(f"  from_uri={r[0]:.1f}ms  write={r[1]:.1f}ms  close={r[2]:.1f}ms  total={sum(r):.1f}ms")

# === N=32 (full contention) ===
print("\n=== N=32 (full contention) ===")
N = 32

all_results = []
t0 = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
    all_results = list(pool.map(upload_timed, range(N)))
elapsed = time.perf_counter() - t0
tput = N * file_mib / elapsed
print(f"  Wall time: {elapsed:.2f}s  Rate: {tput:.0f} MiB/s")

fu_times = [r[0] for r in all_results]
wr_times = [r[1] for r in all_results]
cl_times = [r[2] for r in all_results]
print(f"  from_uri: avg={sum(fu_times)/N:.1f}ms  max={max(fu_times):.1f}ms")
print(f"  write:    avg={sum(wr_times)/N:.1f}ms  max={max(wr_times):.1f}ms")
print(f"  close:    avg={sum(cl_times)/N:.1f}ms  max={max(cl_times):.1f}ms")
print(f"  total/thread avg: {sum(sum(r) for r in all_results)/N:.1f}ms")

# If bottleneck is pure serialized GIL:
# expected wall clock ≈ sum of GIL-held time / (1 thread holds GIL at a time)
# Effective GIL-serialized time per upload ≈ (wall_clock - non_GIL_overlap) / N
print(f"\n  Implied GIL-held per upload (upper bound): {elapsed/N*1000:.1f}ms")
