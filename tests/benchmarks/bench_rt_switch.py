#!/usr/bin/env python3
"""Test effect of RT_THREADS and setswitchinterval on throughput."""
import sys
import os
import s3dlio
import concurrent.futures
import time

os.environ['AWS_ENDPOINT_URL'] = 'http://127.0.0.1:9101'
os.environ['AWS_ACCESS_KEY_ID'] = 'testkey'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'testsecret'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

RT_THREADS = os.environ.get('S3DLIO_RT_THREADS', '28')
print(f"S3DLIO_RT_THREADS={RT_THREADS}, Python switchinterval={sys.getswitchinterval()*1000:.1f}ms")

SHAPE = [6053, 6053, 1]
buf_bv = s3dlio.generate_npz_bytes(shape=SHAPE)
file_mib = len(buf_bv) / (1024*1024)
print(f"File size: {file_mib:.1f} MiB")

def upload_mpu(i):
    with s3dlio.MultipartUploadWriter.from_uri(f's3://mlp-s3dlio/bench_rt/pt{i}.npz') as w:
        w.write(buf_bv)

# warmup
upload_mpu(9999)

for label, interval in [('5ms (default)', 0.005), ('1ms', 0.001), ('0.5ms', 0.0005)]:
    sys.setswitchinterval(interval)
    results = []
    for n in [8, 16, 32, 48, 64]:
        nf = max(n, 32)
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(upload_mpu, range(nf)))
        elapsed = time.perf_counter() - t0
        rate = nf * file_mib / elapsed
        results.append((n, rate))
    print(f"\n  switch={label}: " + "  ".join(f"n={n}:{rate:.0f}" for n, rate in results))
