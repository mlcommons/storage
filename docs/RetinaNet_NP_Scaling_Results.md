# RetinaNet NP Scaling Results

**Sweep date**: 2026-05-12 17:39  
**dlio_benchmark commit**: `fc92d7f` (feat/parquet-dgen-streaming)  
**DataLoader path**: `TorchIterableDatasetSimple` + `_s3_stream_next()` pipelined chunking

---

## Test Environment

| Parameter | Value |
|-----------|-------|
| Host | 24 vCPU (Cascade Lake, no SHA-NI), 48 GB RAM |
| Object storage | s3-ultra (`http://127.0.0.1:9000`, co-located on test host) |
| Bucket / path | `mlp-retinanet/data/retinanet` |
| Dataset | 50,000 JPEG files × 1 sample/file (≈ 15,399 MiB / ~15 GiB) |
| Record length | 322,957 bytes (~315 KiB / file) |
| Batch size | 24 |
| Read threads | 8 |
| `computation_time` | 0.04755 s (B200) |
| DataLoader | `TorchIterableDatasetSimple` — pipelined chunked GETs via `_s3_stream_next()` |
| `prefetch_window` | 256 (default) — chunk N+1 fetched in background while yielding chunk N |
| Epochs | 8 |
| AU target | ≥ 85% |
| Model config | `retinanet_b200.yaml` |
| MPI invocation | `mpirun -n NP -host 127.0.0.1:NP` |

> **⚠️ Co-located test configuration.** The s3-ultra storage server and all benchmark
> processes run on the **same** 24 vCPU / 48 GB RAM host, sharing CPU cores, memory,
> and the loopback network interface. In a real deployment storage would be a dedicated
> remote system; the CPU/memory pressure that limits scaling here would not apply.
>
> **AU (Accelerator Utilization)** — fraction of wall time the simulated accelerator was
> computing rather than waiting for I/O. AU ≥ 85% is the MLPerf Storage target for
> retinanet.

---

## NP Scaling Results

| NP | AU% (mean ± σ) | Samples/s (mean ± σ) | I/O MiB/s (mean ± σ) | Wall (s) | AU ≥ 85%? |
|----|----------------|----------------------|----------------------|----------|-----------|
| 1 | 96.48 ± 0.08 | 485.0 ± 0.4 | 149.4 ± 0.1 | 864 | ✅ PASS |
| 2 | 95.88 ± 0.07 | 964.1 ± 0.8 | 296.9 ± 0.2 | 458 | ✅ PASS |
| 4 | 95.43 ± 0.20 | 1918.9 ± 4.5 | 591.0 ± 1.4 | 252 | ✅ PASS |

### Per-epoch AU% breakdown

| Epoch | NP=1 | NP=2 | NP=4 |
|-------|------|------|------|
| 1 | 96.42 | 95.83 | 94.93 |
| 2 | 96.41 | 96.00 | 95.65 |
| 3 | 96.56 | 95.94 | 95.49 |
| 4 | 96.60 | 95.84 | 95.54 |
| 5 | 96.51 | 95.84 | 95.40 |
| 6 | 96.53 | 95.94 | 95.45 |
| 7 | 96.38 | 95.89 | 95.44 |
| 8 | 96.41 | 95.79 | 95.53 |

AU is extremely stable across epochs (σ < 0.2% at all NP values), confirming the
pipelined I/O path is not accumulating latency or drift between epochs.

---

## Scaling Analysis

### Throughput Scaling Efficiency

| Transition | Samples/s | Ideal | Efficiency |
|------------|-----------|-------|------------|
| NP=1 → NP=2 | 485.0 → 964.1 | 970.0 | **99.4%** |
| NP=1 → NP=4 | 485.0 → 1918.9 | 1940.0 | **98.9%** |

Near-perfect linear scaling through NP=4. The small efficiency loss at NP=4 is
consistent with co-located SHA-256 signing load (no SHA-NI on this Cascade Lake
host) competing for CPU cores with the benchmark processes.

### I/O Throughput per NP

| NP | I/O MiB/s | Per-accelerator MiB/s |
|----|-----------|----------------------|
| 1 | 149.4 | 149.4 |
| 2 | 296.9 | 148.5 |
| 4 | 591.0 | 147.8 |

Per-accelerator I/O throughput is flat (within 1.1%) across all NP values —
the storage backend is not the bottleneck, and adding accelerators does not
degrade per-accelerator I/O bandwidth.

### DataLoader Architecture Note

RetinaNet (315 KiB × 50,000 files) is the most demanding small-object workload
in the suite. Key design decisions that enable the above results:

- **`TorchIterableDatasetSimple`** — file-sharded across workers, not map-style
  `__getitem__`, eliminating per-sample Python dispatch overhead.
- **`_s3_stream_next()` pipelined chunking** — chunk N+1 is submitted to a
  background thread (via `_PREFETCH_POOL`) the instant the yield loop for chunk
  N begins. Since s3dlio releases the GIL during Rust async I/O, fetch and
  Python compute overlap truly concurrently. Peak concurrent GETs per worker:
  `min(prefetch_window, 64) = 64`.
- **Worker stagger** — worker `k` delays `k × computation_time` seconds before
  its first chunk to spread startup I/O across one GPU-cycle window.

---

## Raw Results Location

```
results/retinanet_np_sweep/20260512_173956/
├── NP1/training/retinanet/run/20260512_173956/summary.json
├── NP2/training/retinanet/run/20260512_175421/summary.json
└── NP4/training/retinanet/run/20260512_180159/summary.json
```
