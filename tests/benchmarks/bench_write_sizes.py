#!/usr/bin/env python3
"""
Test write() timing with different buffer sizes to find where 99ms comes from.
If fixed overhead: all sizes take ~99ms.
If data-size dependent: timing scales with size.
"""
import s3dlio
import os
import time

os.environ['AWS_ENDPOINT_URL'] = 'http://127.0.0.1:9101'
os.environ['AWS_ACCESS_KEY_ID'] = 'testkey'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'testsecret'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# Generate various buffer sizes
sizes = {
    '1_MiB': bytes(1 * 1024 * 1024),      # below part_size (16 MiB) - goes to buf, no blocking_send
    '16_MiB': bytes(16 * 1024 * 1024),     # exactly 1 part
    '32_MiB': bytes(32 * 1024 * 1024),     # exactly 2 parts
    '140_MiB': s3dlio.generate_npz_bytes(shape=[6053, 6053, 1]),  # full file, BytesView
}

for name, buf in sizes.items():
    is_bv = isinstance(buf, s3dlio.BytesView) if hasattr(s3dlio, 'BytesView') else False
    buf_type = "BytesView" if is_bv else "bytes"
    buf_len = len(buf)
    
    # Warmup
    w = s3dlio.MultipartUploadWriter.from_uri(f's3://mlp-s3dlio/bench_wt/warmup.npz')
    w.write(buf)
    w.close()
    
    # Measure write() times (5 runs)
    write_times = []
    for i in range(5):
        w = s3dlio.MultipartUploadWriter.from_uri(f's3://mlp-s3dlio/bench_wt/{name}_{i}.npz')
        t1 = time.perf_counter()
        w.write(buf)
        t2 = time.perf_counter()
        w.close()
        write_times.append((t2 - t1) * 1000)
    
    avg_write = sum(write_times) / len(write_times)
    mib = buf_len / (1024*1024)
    parts = max(0, buf_len // (16*1024*1024))
    print(f"{name:12s} ({mib:5.1f} MiB, ~{parts} full parts, {buf_type}): "
          f"write={avg_write:.1f}ms  times={[f'{t:.0f}' for t in write_times]}")
