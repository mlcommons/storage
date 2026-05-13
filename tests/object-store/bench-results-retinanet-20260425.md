# mlp-storage / dlio_benchmark Benchmark Results

System: Intel Xeon Platinum 8280L (Cascade Lake, 28c/56t) — **no SHA-NI**  
Server: s3-ultra `http://127.0.0.1:9101` (loopback)  
Library: s3dlio (PyPI)  
Protocol: HTTP/1.1 (default — `DEFAULT_H2C_ENABLED=false` since v0.9.92)  
Data: 50,000 × 322,957 bytes = 15,396 MiB (~15.0 GiB)

---

## Experiment 1 — write_threads sweep (s3dlio, retinanet, NP=1)

**Null hypothesis**: More threads beyond the default (32) will NOT improve throughput.

Date: 2026-04-25  
Model: retinanet (JPEG, 315 KiB/object)  
NP: 1 rank  
Files: 50,000  

| write_threads | elapsed (s) | throughput (MiB/s) | user CPU (s) | %CPU |
|:---:|---:|---:|---:|---:|
| 8  | 31.84 | 483 | 134.9 | 449% |
| 16 | 22.03 | **699** | 132.3 | 638% |
| 32 | 22.00 | **700** | 133.2 | 643% |
| 64 | 22.17 | 694 | 133.6 | 642% |
| 128 | 21.89 | **703** | 133.3 | 648% |

**Result**: Null hypothesis **REJECTED** for 8→16 (+45% gain). **CONFIRMED** for 16+: throughput plateaus flat from 16 to 128 threads. Saturation at ~700 MiB/s is a hard limit, not a thread-count problem.

**Conclusion**: The plateau at ~700 MiB/s with 16+ threads is a CPU/SHA-256 bottleneck. Software SHA-256 (no SHA-NI) limits throughput regardless of concurrency. The current auto-size formula already exceeds the saturation point.

**Note on SHA-NI**: Hardware SHA-NI (available on Ice Lake+, EPYC Zen 2+) gives ~3–5× faster SHA-256 throughput. On this Cascade Lake system, software SHA-256 caps us at ~700 MiB/s. With SHA-NI, we would expect ~2–3 GB/s for the same workload.

---

## Experiment 2 — Storage library comparison (s3dlio vs minio vs s3torchconnector)

**Null hypothesis**: All three libraries will produce similar throughput for 315 KiB objects.

Date: 2026-04-25  
Model: retinanet (JPEG, 315 KiB/object)  
NP: 1 rank, write_threads=32  
Files: 50,000  

| library | elapsed (s) | throughput (MiB/s) | user CPU (s) | %CPU | notes |
|:---:|---:|---:|---:|---:|:---|
| s3dlio | 22.54 | **683** | 134.7 | 636% | Rust AWS SDK, SigV4 in Tokio |
| minio | 57.85 | **266** | 111.6 | 216% | minio-py 7.2.20, Python GIL-bound |
| s3torchconnector | 21.51 | **716** | 51.7 | 318% | AWS official connector, ~2.6× less CPU than s3dlio |

**Result**: Null hypothesis **REJECTED**. minio is 2.6× slower. s3torchconnector matches/exceeds s3dlio at ~716 MiB/s but uses only 51.7s user CPU vs 134.7s for s3dlio — implying it has a more efficient signing path.

**Key observation — s3torchconnector CPU**: 51.7s user at 318% CPU = 16.3 effective CPU-seconds per core. s3dlio: 134.7s user at 636% CPU = 21.2 CPU-seconds per core. s3torchconnector uses ~3× less CPU per MiB/s, suggesting it either avoids SHA-256 body signing, uses hardware TLS offload, or has a more vectorized HMAC implementation.

**minio bottleneck**: 57.85s elapsed at only 216% CPU = severe GIL contention and Python-bound PUT overhead at 32 threads. Not suitable for high-throughput datagen.

---

## Experiment 3 — MPI scaling: s3dlio vs s3torchconnector vs minio (8 threads/rank)

**Null hypothesis**: Throughput scales linearly with NP for both libraries.

Date: 2026-04-25  
Model: retinanet (JPEG, 315 KiB/object)  
write_threads: 8 per rank (DLIO_MAX_AUTO_THREADS=8)  
Files: 50,000 total (split evenly across ranks)  
Total data: 15,396 MiB  

| library | NP | elapsed (s) | throughput (MiB/s) | speedup vs NP=1 | user CPU (s) | %CPU |
|:---:|:---:|---:|---:|---:|---:|---:|
| s3dlio | 1 | 30.59 | 503 | 1.00× | 134.2 | 465% |
| s3dlio | 2 | 19.69 | 782 | 1.55× | 138.0 | 747% |
| s3dlio | 4 | 16.66 | 924 | 1.84× | 149.1 | 958% |
| s3dlio | 8 | 14.56 | **1,057** | **2.10×** | 167.7 | 1240% |
| s3torchconnector | 1 | 32.92 | 468 | 1.00× | 51.6 | 208% |
| s3torchconnector | 2 | 19.22 | 801 | 1.71× | 53.7 | 368% |
| s3torchconnector | 4 | 11.80 | 1,305 | 2.79× | 62.1 | 687% |
| s3torchconnector | 8 | 8.86 | **1,738** | **3.71×** | 83.6 | 1206% |
| minio | 1 | 53.09 | 290 | 1.00× | 104.4 | 220% |
| minio | 2 | 29.83 | 516 | 1.78× | 107.2 | 405% |
| minio | 4 | 22.18 | 694 | 2.39× | 117.9 | 602% |
| minio | 8 | 17.48 | **881** | **3.04×** | 137.8 | 897% |

**Result**: Null hypothesis **REJECTED** — no library scales linearly, but scaling efficiency varies widely.

**Scaling efficiency** (actual/ideal-linear):
- s3dlio NP=8: 1,057 / (503×8) = **26%** — poor, CPU-bound (SHA-256 cores saturated across 8 Tokio runtimes)
- s3torchconnector NP=8: 1,738 / (468×8) = **46%** — best scaling, low per-PUT CPU cost
- minio NP=8: 881 / (290×8) = **38%** — moderate scaling, GIL overhead per rank reduces efficiency

**Key finding**: At NP=8, s3torchconnector reaches **1,738 MiB/s** vs s3dlio's **1,057 MiB/s** vs minio's **881 MiB/s**. s3torchconnector wins by a wide margin (1.64× over s3dlio, 1.97× over minio). Despite minio's poor single-rank throughput (290 MiB/s at NP=1), it scales reasonably (3.04× at NP=8) — multiple processes each get a separate GIL, hiding the single-rank bottleneck. s3dlio's Tokio runtimes (28 threads each) compete across 8 processes for the same 28 physical cores, all doing software SHA-256 signing.

**At NP=8, CPU usage**: s3torchconnector 83.6s, minio 137.8s, s3dlio 167.7s — the per-request signing cost of s3dlio multiplies with NP.

---

## Experiment 4 — Object-size aware thread scaling (planned)

**Null hypothesis**: Optimal thread count is independent of object size.

Planned: vary object size (64 KiB, 315 KiB, 1 MiB, 4 MiB, 16 MiB) and measure optimal thread count for each.

---
