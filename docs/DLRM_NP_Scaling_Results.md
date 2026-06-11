# DLRM Training — Compute Time & NP Scaling Study

---

## Test Environment

| Parameter | Value |
|-----------|-------|
| Host | 24 vCPU VM (with hyperthreading), 48 GB RAM |
| Object storage | s3-ultra (`http://127.0.0.1:9000`, co-located on test host) |
| Bucket / path | `mlp-dlrm / data/dlrm` |
| Dataset | 200 files × 1,536,000 samples/file |
| Record length | 761 bytes |
| Batch size | 12,288 |
| `decode_mode` | `none` |
| Epochs | 2 |
| Steps per epoch | 25,000 ÷ NP |
| Model config | `dlrm_b200.yaml` |
| MPI invocation | `mpirun -n NP -host 127.0.0.1:NP` |

> **⚠️ Co-located test configuration.** The s3-ultra storage server and all benchmark processes run on the
> **same** 24 vCPU / 48 GB RAM host, sharing CPU cores, memory, and the loopback network interface.
> In a real deployment the storage target would be a dedicated remote system, and the CPU/memory
> pressure that limits scaling here (particularly at NP ≥ 4) would not apply to the test processes.
> The resource constraints described in this document are a property of this co-located setup, not
> of the storage technology itself.

**AU (Accelerator Utilization)** — fraction of wall time the simulated GPU was computing rather than waiting for I/O. AU ≥ 70% is the target threshold for a "pass." Below that, the workload is I/O-bound and the storage system cannot keep the accelerator fed.

---

## Phase 1 — Compute Time Sweep (NP = 1)

Objective: find the `computation_time` at which the DLRM workload transitions from I/O-bound to
compute-bound on a single accelerator. Four values were tested: 375 µs, 1 ms, 5 ms, and 10 ms.

### Phase 1 — Summary Table

| `computation_time` | AU% (avg) | Samples/s | I/O MiB/s | AU ≥ 70%? |
|--------------------|-----------|-----------|-----------|-----------|
| 375 µs | 7.88% | 2,053,984 | 1,490.7 | ❌ FAIL |
| 1 ms | 19.59% | 2,178,529 | 1,581.1 | ❌ FAIL |
| 5 ms | 78.69% | 1,877,874 | 1,362.9 | ✅ PASS |
| 10 ms | 87.71% | 1,060,327 | 769.5 | ✅ PASS |

### Phase 1 — Per-Epoch Detail

| `computation_time` | Epoch | Wall (s) | Samples/s | AU% |
|--------------------|-------|----------|-----------|-----|
| 375 µs | 1 | 182.21 | 1,729,031 | 6.65% |
| 375 µs | 2 | 129.39 | 2,378,938 | 9.10% |
| 1 ms | 1 | 168.65 | 1,869,815 | 16.85% |
| 1 ms | 2 | 123.55 | 2,487,243 | 22.33% |
| 5 ms | 1 | 162.86 | 1,940,250 | 81.25% |
| 5 ms | 2 | 169.51 | 1,815,498 | 76.13% |
| 10 ms | 1 | 291.79 | 1,068,892 | 88.44% |
| 10 ms | 2 | 292.12 | 1,051,762 | 86.97% |

### Phase 1 — Key Observations

- **The AU knee lies between 1 ms and 5 ms.** At 1 ms the workload is severely I/O-bound (AU ≈ 20%);
  at 5 ms it passes the 70% threshold (AU ≈ 79%).
- **Peak I/O throughput occurs in the 375 µs – 1 ms range** (~1,500–1,580 MiB/s), where the
  simulated GPU is nearly always waiting and the pipeline is fully storage-saturated.
- **Epoch 2 is consistently faster than Epoch 1** at low compute times — page-cache warming and
  S3 connection reuse reduce cold-start overhead on the second pass.
- **At ct = 10 ms the workload is strongly compute-bound** (AU ≈ 88%) and I/O throughput drops to
  ~770 MiB/s because the GPU consumes data more slowly than storage can deliver it.

---

## Phase 2 — NP Scaling Sweep (ct = 1 ms and ct = 5 ms)

Objective: determine how aggregate throughput and per-accelerator AU scale as NP grows from 1 to 8,
at two operating points: one I/O-bound (ct = 1 ms) and one near the AU threshold (ct = 5 ms).

Each NP rank was mapped to the same host: `mpirun -n NP -host 127.0.0.1:NP`.

### Phase 2 — Summary Table

| ct | NP | AU% (avg) | Samples/s | I/O MiB/s | Scaling vs NP=1 | AU ≥ 70%? |
|----|----|-----------|-----------|-----------|-----------------|-----------|
| 1 ms | 1 | 17.77% | 1,972,511 | 1,431.5 | 1.00× | ❌ FAIL |
| 1 ms | 2 | 17.65% | 3,968,010 | 2,879.8 | 2.01× | ❌ FAIL |
| 1 ms | 4 | 15.02% | 6,784,287 | 4,923.7 | 3.44× | ❌ FAIL |
| 1 ms | 8 | — | — | — | — | 💥 CRASH (OOM) |
| 5 ms | 1 | 80.91% | 1,933,857 | 1,403.5 | 1.00× | ✅ PASS |
| 5 ms | 2 | 71.79% | 3,418,977 | 2,481.3 | 1.77× | ✅ PASS |
| 5 ms | 4 | 68.67% | 6,545,863 | 4,750.6 | 3.39× | ❌ FAIL |
| 5 ms | 8 | — | — | — | — | 💥 CRASH (OOM) |

**Scaling vs NP=1**: ratio of aggregate `samples/s` at NP=N to NP=1 within the same ct group.
Perfect linear scaling would yield 2.00×, 4.00×, 8.00× for NP=2, 4, 8.

### Phase 2 — Per-Epoch Detail

| ct | NP | Epoch | Wall (s) | Samples/s | AU% |
|----|----|-------|----------|-----------|-----|
| 1 ms | 1 | 1 | 179.15 | 1,754,308 | 15.66% |
| 1 ms | 1 | 2 | 140.46 | 2,190,715 | 19.88% |
| 5 ms | 1 | 1 | 165.13 | 1,911,922 | 80.19% |
| 5 ms | 1 | 2 | 157.51 | 1,955,793 | 81.63% |
| 1 ms | 2 | 1 | 95.23 | 3,384,832 | 14.97% |
| 1 ms | 2 | 2 | 67.83 | 4,567,957 | 20.90% |
| 5 ms | 2 | 1 | 94.48 | 3,414,248 | 71.64% |
| 5 ms | 2 | 2 | 89.93 | 3,421,878 | 71.80% |
| 1 ms | 4 | 1 | 50.28 | 6,716,084 | 14.77% |
| 1 ms | 4 | 2 | 45.27 | 6,891,347 | 16.23% |
| 5 ms | 4 | 1 | 52.55 | 6,424,380 | 67.64% |
| 5 ms | 4 | 2 | 46.49 | 6,708,777 | 70.49% |
| 1 ms | 8 | — | — | — | 💥 OOM (SIGKILL rank 4) |
| 5 ms | 8 | — | — | — | 💥 OOM (SIGKILL rank 3) |

---

## Scaling Analysis

### Aggregate Throughput Scaling (ct = 1 ms)

| NP | Samples/s | vs NP=1 | Efficiency |
|----|-----------|---------|------------|
| 1 | 1,972,511 | 1.00× | 100% |
| 2 | 3,968,010 | 2.01× | 100.5% |
| 4 | 6,784,287 | 3.44× | 86.0% |

Near-linear scaling to NP=2 (2.01× vs ideal 2.00×). At NP=4, efficiency drops to 86% — the storage
backend is saturating at ~4,924 MiB/s and cannot maintain linear per-rank delivery.

### Aggregate Throughput Scaling (ct = 5 ms)

| NP | Samples/s | vs NP=1 | Efficiency |
|----|-----------|---------|------------|
| 1 | 1,933,857 | 1.00× | 100% |
| 2 | 3,418,977 | 1.77× | 88.3% |
| 4 | 6,545,863 | 3.39× | 84.7% |
| 8 | — (CRASH) | — | — |

At ct = 5 ms the workload is already near-AU-threshold at NP=1, so adding ranks increases I/O
pressure while the per-rank compute budget remains fixed. AU degrades monotonically:
80.91% → 71.79% → 68.67%, crossing below the 70% pass threshold at NP=4.

### I/O Throughput Scaling

| NP | ct=1ms I/O (MiB/s) | ct=5ms I/O (MiB/s) |
|----|-------------------|-------------------|
| 1 | 1,431.5 | 1,403.5 |
| 2 | 2,879.8 | 2,481.3 |
| 4 | 4,923.7 | 4,750.6 |

I/O scales well through NP=4, with the two ct groups converging toward a similar ceiling near
~4,750–4,924 MiB/s. This suggests the loopback MinIO instance is approaching its throughput limit
at ~5 GB/s when 4 concurrent s3dlio processes are active.

### Per-Accelerator (per-rank) Samples/s

| ct | NP=1 | NP=2 | NP=4 | NP=8 |
|----|------|------|------|------|
| 1 ms | 1,972,511 | 1,984,005 | 1,696,072 | — |
| 5 ms | 1,933,857 | 1,709,489 | 1,636,466 | — |

At ct = 1 ms, per-rank throughput is nearly constant from NP=1 to NP=2, then drops ~15% at NP=4
as I/O contention grows. At ct = 5 ms, per-rank throughput drops earlier because the workload is
already closer to the storage saturation point at NP=1.

---

## NP = 8 Failure Analysis

Both ct = 1 ms and ct = 5 ms runs at NP = 8 crashed before completing any training steps.

**Root causes:**

1. **OOM — kernel SIGKILL.** Each MPI rank spawns a Python process. At NP = 8, the combined memory
   footprint (Python interpreter, DLIO data buffers, s3dlio connection pool, prefetch queues,
   MPI runtime) exceeded the 48 GB RAM limit. The kernel OOM killer sent SIGKILL to rank 3 or 4.
   - `mpirun noticed that process rank N exited on signal 9 (Killed)`

2. **S3 TCP connection exhaustion.** 8 concurrent s3dlio processes each attempted to open
   connection pools to s3-ultra on loopback. The aggregate connection demand — combined with
   s3-ultra itself consuming CPU on the same host — overwhelmed the server's listener backlog,
   causing TCP connection rejection errors on all ranks before the OOM fired on some runs.

**Conclusion:** NP = 8 is not viable on this co-located 24 vCPU / 48 GB RAM setup. Maximum usable
NP = 4. In a real deployment where s3-ultra runs on a dedicated remote system, NP = 8 would have
the full 48 GB and all 24 vCPUs available exclusively for the benchmark processes, making this
limitation irrelevant.

---

## Overall Key Findings

1. **The AU knee for DLRM on this storage stack is between ct = 1 ms and ct = 5 ms.**
   - At ct ≤ 1 ms: severely I/O-bound (AU ≈ 7–20%); storage cannot keep up regardless of NP.
   - At ct = 5 ms: marginally passes at NP=1 and NP=2 (AU ≈ 71–81%); fails at NP=4 (AU = 68.7%).
   - At ct = 10 ms: comfortably passes (AU ≈ 88%); workload is strongly compute-bound.

2. **Storage saturates near 5 GB/s on this co-located setup.** Both ct groups hit ~4.75–4.93 GB/s
   at NP=4, and AU begins degrading. This ceiling reflects the shared CPU/memory budget — s3-ultra
   and the benchmark processes are competing for the same resources. On a dedicated remote storage
   system, this throughput ceiling would be significantly higher.

3. **Aggregate throughput scales near-linearly to NP=4 in the I/O-bound regime (ct = 1 ms).**
   3.44× aggregate throughput at NP=4 (86% efficiency) reflects good parallelism up to the
   storage bandwidth limit.

4. **AU degrades with NP even when compute time is fixed.** Each additional rank increases
   per-step I/O demand without increasing the per-step compute budget, so the storage-to-compute
   ratio worsens. At ct = 5 ms, NP=4 drops just below the 70% threshold.

5. **Epoch 2 is consistently faster than Epoch 1** at low compute times. Page-cache warming and
   persistent S3 connections from epoch 1 reduce cold-start cost in epoch 2.

6. **NP = 8 is not viable on this VM** due to OOM and S3 TCP exhaustion. Maximum recommended
   NP for this host configuration: **4**.

---

---

*Benchmark date: May 12, 2026*  
*Host: loki-russ*  
*s3-ultra (localhost:9000, co-located on test host)*
