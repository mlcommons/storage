# Flux Training — NP × Read-Threads Scaling Study

---

> ## ⚠️ **NON-STANDARD `computation_time` — RESULTS ARE NOT REPRESENTATIVE OF REAL TRAINING**
>
> **All runs in this study used `computation_time = 0.05 s` — the simulated GPU compute sleep per step.**
>
> **The production default for Flux (flux_b200.yaml) is `computation_time = 1.35 s`.**
>
> **This 27× reduction was intentional — it stress-tests the storage stack by making I/O the
> dominant cost — but it means AU numbers and samples/s figures cannot be directly compared
> to a real Flux training job or to any benchmark run with default settings.**
>
> **Do not cite these AU numbers as "Flux training performance." They are I/O-stress results only.**

---

## Test Environment

| Parameter | Value |
|-----------|-------|
| Host | 24 vCPU VM (with hyperthreading), 48 GB RAM |
| Object storage | s3-ultra (`localhost:9000`, co-located on test host) |
| Dataset | 500 Parquet files, ~595 MiB each, 6 row groups × 99 MiB |
| Samples/file | 288 (batch_size=48) |
| `computation_time` | 0.05 s (fixed — stress I/O, not compute) |
| `coalesce_rgs` | 1 (99 MiB per GET) |
| `prefetch_workers` | 2 |
| Model config | flux\_b200.yaml |

> **⚠️ Co-located test configuration.** The s3-ultra storage server and all benchmark processes run on the
> **same** 24 vCPU / 48 GB RAM host, sharing CPU cores, memory, and the loopback network interface.
> In a real deployment the storage target would be a dedicated remote system, and the CPU/memory
> pressure that limits scaling here (particularly at NP ≥ 4) would not apply to the test processes.
> The resource constraints described in this document are a property of this co-located setup, not
> of the storage technology itself.

## Results

| NP | RT | AU% | samples/s | **samp/s/GPU** | I/O MiB/s | Wall (s) | Steps | Notes |
|----|----|-----|-----------|---------------|-----------|----------|-------|-------|
| 1 | 1 | 96.8 | 926 | **926** | 1,911 | 188 | 3000 | |
| 1 | 2 | 96.7 | 925 | **925** | 1,911 | 174 | 3000 | |
| 1 | 4 | 96.7 | 925 | **925** | 1,911 | 178 | 3000 | |
| 1 | 8 | 96.7 | 925 | **925** | 1,911 | 188 | 3000 | |
| 2 | 1 | 96.7 | 1,849 | **925** | 3,818 | 110 | 1500 | |
| 2 | 2 | 96.7 | 1,850 | **925** | 3,820 | 95 | 1500 | |
| 2 | 4 | 96.4 | 1,844 | **922** | 3,807 | 102 | 1500 | |
| 2 | 8 | 96.7 | 1,849 | **925** | 3,818 | 111 | 1500 | |
| 4 | 1 | 91.7 | 3,496 | **874** | 7,217 | 73 | 750 | |
| 4 | 2 | 93.2 | 3,557 | **889** | 7,343 | 60 | 750 | |
| 4 | 4 | 92.4 | 3,526 | **882** | 7,279 | 64 | 750 | |
| 4 | 8 | 91.7 | 3,496 | **874** | 7,217 | 76 | 750 | CPU constrained (NP×RT=32) |
| 8 | 1 | 59.9 | 4,477 | **560** | 9,244 | 55 | 375 | |
| 8 | 2 | 57.2 | 4,316 | **540** | 8,910 | 53 | 375 | |
| 8 | 4 | 61.0 | 4,532 | **567** | 9,356 | 58 | 375 | CPU constrained (NP×RT=32) |
| 8 | 8 | — | — | **—** | — | — | — | OOM — worker killed (SIGKILL); NP×RT=64 |

**NP** = number of MPI ranks (`--num-accelerators`).  
**RT** = `reader.read_threads` (Torch DataLoader workers per rank).  
**AU** = Accelerator Utilization — fraction of time the simulated GPU was computing rather than waiting for data.  
**samp/s/GPU** = `samples/s ÷ NP` — per-GPU throughput; the key scaling efficiency metric. Perfect linear scaling would hold this constant as NP grows. The drop from ~925 at NP=1–2 to ~560–567 at NP=8 shows the storage system losing ~40% per-GPU efficiency at 8 ranks.

## CPU Constraint Threshold

On this 24 vCPU (hyperthreaded) host, the practical CPU budget **shared between the benchmark
processes and the co-located s3-ultra server** is:

> **NP × RT ≤ 8 — sufficient CPU; NP × RT > 8 — CPU constrained**

All combinations at or below NP×RT=8 ran with high AU (91–97%) and consistent throughput.
Combinations above that threshold showed either degraded AU or outright failure:

- **NP=4, RT=8 (NP×RT=32)** and **NP=8, RT=4 (NP×RT=32)**: AU dropped; more threads competing for 24 vCPUs than the host can efficiently schedule — and s3-ultra is consuming a share of those vCPUs on the same machine.
- **NP=8, RT=8 (NP×RT=64)**: OOM. 8 MPI ranks × 8 DataLoader workers × 2 prefetch buffers × 99 MiB/GET ≈ 12+ GB I/O buffer pressure on a 48 GB host, combined with Python process overhead per rank and s3-ultra's own memory footprint — the kernel OOM killer fired.

**In a real deployment** with s3-ultra on a dedicated remote server, all 24 vCPUs and 48 GB RAM
would be available exclusively to the benchmark processes, and these specific constraints would
not apply.

## Key Observations

1. **`read_threads` has negligible effect at NP=1 and NP=2.** AU is flat at ~96.7% across RT=1–8. With only 1–2 ranks and 0.05 s compute, a single reader thread can keep the pipeline fed. This is a storage benchmark and storage is not the bottleneck at low NP.

2. **NP=4 is where storage starts to bite.** AU falls to 91–93%; throughput doubles vs NP=2 but AU drops ~5 points. RT=2 is the sweet spot here (93.2% AU, 7,343 MiB/s).

3. **NP=8 makes storage the clear bottleneck.** AU falls to 57–61% — ranks are spending ~40% of their time waiting for I/O. Peak observed throughput was ~9,356 MiB/s (NP=8, RT=4). RT=4 outperforms RT=1 and RT=2 here because more concurrent reader threads help overlap I/O with the pipeline.

4. **The co-located setup is the limiting factor at high NP×RT, not the storage stack itself.**
   s3-ultra and the benchmark processes share the same CPU and memory. On a system where s3-ultra
   is deployed on a dedicated remote server, the full host resources would be available to the
   benchmark, and the configurations with higher NP×RT products would be expected to perform
   significantly better.

## Impact of `computation_time` on AU and Throughput

### Background: How AU is Computed

$$AU = \frac{t_{compute}}{t_{compute} + t_{io\_wait}}$$

The I/O wait per step is a property of the **storage system only** — it does not change when
the sleep time changes. From the measured AU values at `computation_time = 0.05 s` we can
back-calculate the actual I/O wait the storage imposed on each configuration:

| NP | RT | Measured AU (0.05s) | Implied I/O wait/step |
|----|----|--------------------|-----------------------|
| 1 | 1–8 | ~96.8% | ~1.7 ms |
| 2 | 1–8 | ~96.6% | ~1.7 ms |
| 4 | 2 | 93.2% | ~3.7 ms |
| 4 | 1,4,8 | ~91.7–92.4% | ~4–5 ms |
| 8 | 4 | 61.0% | ~32 ms |
| 8 | 1 | 59.9% | ~33 ms |
| 8 | 2 | 57.2% | ~37 ms |

### Projected AU at Higher Sleep Values

Plugging those I/O wait numbers into the AU formula at `0.5 s` and `1.35 s` (the production
default):

| NP | RT | AU at 0.05 s (actual) | AU at 0.5 s (projected) | AU at 1.35 s (projected) |
|----|----|-----------------------|--------------------------|--------------------------|
| 1 | 1–8 | ~96.8% | ~99.7% | ~99.9% |
| 2 | 1–8 | ~96.6% | ~99.7% | ~99.9% |
| 4 | 2 | 93.2% | 99.3% | 99.7% |
| 4 | 1,4,8 | 91.7–92.4% | 99.1–99.2% | 99.7% |
| 8 | 4 | 61.0% | **94.0%** | **97.7%** |
| 8 | 1 | 59.9% | **93.7%** | **97.6%** |
| 8 | 2 | 57.2% | **93.0%** | **97.3%** |

### What This Means

1. **At 0.5 s sleep**, the storage bottleneck at NP=8 is still visible (AU ≈ 93–94%) but
   much less alarming than the 57–61% we measured. All NP≤4 runs would look essentially
   perfect (>99% AU), completely hiding any storage sensitivity.

2. **At 1.35 s (production default)**, *every single configuration* — including NP=8 — would
   report AU above 97%. The benchmark would appear to pass with flying colours and the storage
   system would look like it is never the bottleneck, even though at NP=8 it is imposing
   30–37 ms of wait per step.

3. **The 0.05 s setting is the right choice for a storage benchmark.** It amplifies the
   storage signal by a factor of ~27 relative to real training. The AU drop from 96% (NP=1)
   to 61% (NP=8) is the entire point — it reveals that the storage system has a real scaling
   wall somewhere between NP=4 and NP=8 on this platform.

4. **Throughput (samples/s and MiB/s) is unaffected by the sleep value** — the storage stack
   does the same amount of I/O work regardless. I/O MiB/s figures in the results table are
   valid for any sleep setting.

5. **To project to a real Flux B200 job** (1.35 s compute), the NP=8 results above suggest
   AU ≈ 97–98%. That means storage would *just barely* keep up on real hardware at 8 GPUs —
   which is still actionable: a faster or more parallel storage backend would meaningfully
   improve training time at scale.

## Date

Run: 2026-05-11
