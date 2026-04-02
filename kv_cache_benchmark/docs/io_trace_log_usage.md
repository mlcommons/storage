# Using `--io-trace-log` Trace Mode

**Branch**: `feature/io-trace-log` (`54d0135`)

---

## Overview

When `--io-trace-log <path>` is specified, the benchmark runs in **pure logical
trace mode**. The full LLM inference simulation (prefill, decode, multi-turn,
eviction, prefix caching) executes normally, but no real GPU/CPU/NVMe I/O is
performed. Instead, every KV cache operation is recorded to a structured CSV
file that can be replayed by an external storage benchmarking tool.

This cleanly separates **workload generation** from **storage validation**:

- The benchmark defines *what* operations happen and at *what rate* for a
  given model, request pattern, and hardware configuration.
- An external tool (`fio`, `sai3-bench`, `warp`, etc.) replays those
  operations against real hardware to measure actual storage performance.

---

## New Flags

### `--io-trace-log <path>`

Activates trace mode. Accepts any file path.

- Plain `.csv` path → uncompressed CSV, line-buffered.
- Path ending in `.zst` → streaming zstd-compressed CSV (strongly recommended
  for runs longer than a few minutes — see [Compression](#compression)).

```bash
--io-trace-log /tmp/kv_trace.csv          # plain CSV
--io-trace-log /tmp/kv_trace.csv.zst      # compressed (recommended)
```

Requires the `zstandard` package for `.zst` output:
```bash
uv pip install "kv-cache-benchmark[compression]"
# or
uv pip install zstandard
```

---

### `--num-gpus N`  *(default: 1)*

Total number of GPUs in the tensor-parallel group.  Effective GPU tier
capacity = `N × --gpu-mem-gb`.

```bash
--num-gpus 8 --gpu-mem-gb 141    # models an 8×H200 node: 1,128 GB HBM total
--num-gpus 4 --gpu-mem-gb 80     # models a 4×A100 node:    320 GB HBM total
```

---

### `--tensor-parallel N`  *(default: 1)*

Tensor-parallel (TP) degree. Each GPU rank stores `1/N` of each KV cache
entry, so the per-rank object size written/read — and recorded in the trace —
is divided by `N`.

Constraints:
- Must be ≥ 1 and ≤ `--num-gpus`.
- Values that are not a power of 2 emit a warning (unusual for real deployments).

```bash
--tensor-parallel 8    # TP=8: each rank stores 1/8 of the KV entry
```

The run banner shows the effective configuration:
```
System: 8× 141 GB GPU  (total 1128 GB HBM)  │  TP=8
```

---

## CSV Output Format

One row per KV cache I/O event.

| Column | Type | Description |
|--------|------|-------------|
| `Timestamp` | float | Unix epoch (6 decimal places) |
| `Operation` | string | `Write` or `Read` |
| `Object_Size_Bytes` | int | Exact byte size of the KV cache object for this rank (TP-adjusted) |
| `Tier` | string | `Tier-0` (GPU VRAM), `Tier-1` (CPU RAM), `Tier-2` (NVMe) |
| `Key` | string | Cache entry identifier — use as object name / path in replay tools |
| `Phase` | string | `Prefill` (initial write), `Decode` (per-token read), `Evict` (demotion) |

### Example rows

```
Timestamp,Operation,Object_Size_Bytes,Tier,Key,Phase
1740553426.194021,Write,131072,Tier-0,layer0/user0,Prefill
1740553426.194308,Read,131072,Tier-0,layer0/user0,Decode
1740553426.194521,Write,131072,Tier-2,layer0/user0,Evict
1740553426.194590,Read,131072,Tier-2,layer0/user0,Decode
```

### Tier mapping

| Tier label | Hardware |
|---|---|
| `Tier-0` | GPU VRAM (e.g. H200 HBM) |
| `Tier-1` | CPU / system DRAM |
| `Tier-2` | NVMe / persistent storage |

---

## Compression

For any run longer than a few minutes, using `.zst` output is strongly recommended.

| Run duration | Uncompressed size (est.) | Compressed (est.) |
|---|---|---|
| 1 minute | ~50 MB | ~3–5 MB |
| 1 hour | ~1–5 GB | ~50–250 MB |
| 8 hours | ~8–40 GB | ~400 MB–2 GB |

To inspect or decompress a `.zst` trace:
```bash
# Decompress in-place
zstd -d kv_trace.csv.zst

# Stream through head without full decompression
zstd -d --stdout kv_trace.csv.zst | head -20

# Count rows
zstd -d --stdout kv_trace.csv.zst | wc -l
```

---

## Usage Examples

### Minimal trace — default single GPU

```bash
cd kv_cache_benchmark
python -m kv_cache.cli \
  --model llama3.1-8b \
  --num-users 32 \
  --duration 60 \
  --io-trace-log /tmp/kv_trace_llama8b.csv.zst
```

---

### 8×H200 node, TP=8, Llama 70B — 5-minute trace

```bash
python -m kv_cache.cli \
  --model llama3.1-70b-instruct \
  --num-users 128 \
  --duration 300 \
  --num-gpus 8 \
  --gpu-mem-gb 141 \
  --tensor-parallel 8 \
  --io-trace-log /mnt/scratch/kv_trace_llama70b_tp8.csv.zst
```

Expected banner:
```
System: 8× 141 GB GPU  (total 1128 GB HBM)  │  TP=8
```

---

### Disaggregated prefill-only trace

Simulates a disaggregated prefill node (write-heavy, no decode reads):

```bash
python -m kv_cache.cli \
  --model llama3.1-70b-instruct \
  --num-users 64 \
  --duration 300 \
  --num-gpus 8 --gpu-mem-gb 141 \
  --tensor-parallel 8 \
  --prefill-only \
  --io-trace-log /tmp/kv_prefill_only.csv.zst
```

---

### Disaggregated decode-only trace

Simulates a decode node (read-heavy, assumes KV cache already exists on NVMe):

```bash
python -m kv_cache.cli \
  --model llama3.1-70b-instruct \
  --num-users 64 \
  --duration 300 \
  --num-gpus 8 --gpu-mem-gb 141 \
  --tensor-parallel 8 \
  --decode-only \
  --io-trace-log /tmp/kv_decode_only.csv.zst
```

---

### DeepSeek V3 — MLA attention model

```bash
python -m kv_cache.cli \
  --model deepseek-v3 \
  --num-users 64 \
  --duration 120 \
  --num-gpus 8 --gpu-mem-gb 141 \
  --tensor-parallel 8 \
  --io-trace-log /tmp/kv_deepseek_v3.csv.zst
```

---

## Available Models

| Model key | Description |
|---|---|
| `tiny-1b` | Tiny 1B (dev/test) |
| `mistral-7b` | Mistral 7B |
| `llama2-7b` | Llama 2 7B |
| `llama3.1-8b` | Llama 3.1 8B |
| `llama3.1-70b-instruct` | Llama 3.1 70B Instruct |
| `deepseek-v3` | DeepSeek V3 (MLA attention) |
| `qwen3-32b` | Qwen3 32B |
| `gpt-oss-120b` | GPT OSS 120B (MoE) |
| `gpt-oss-20b` | GPT OSS 20B (MoE) |

Custom models can be added via `config.yaml` — they are merged with and
override the defaults at runtime.

---

## Replaying a Trace

The `Key` column provides a stable object identifier across writes and reads,
enabling storage tools to correlate operations and build realistic object
stores.

### Example: sai3-bench (illustrative)

```bash
sai3-bench replay \
  --trace /tmp/kv_trace_llama70b_tp8.csv.zst \
  --endpoint s3://my-kv-cache-bucket
```

### Example: fio (illustrative)

Convert the trace to an fio job file using offset/size from
`Object_Size_Bytes` and replay against a block device or NFS path.

### Inspecting the trace first

```bash
# See the first 10 operations
zstd -d --stdout /tmp/kv_trace.csv.zst | head -11

# Count operations by tier
zstd -d --stdout /tmp/kv_trace.csv.zst \
  | awk -F, 'NR>1 {print $4}' \
  | sort | uniq -c | sort -rn

# Count reads vs writes
zstd -d --stdout /tmp/kv_trace.csv.zst \
  | awk -F, 'NR>1 {print $2}' \
  | sort | uniq -c

# Summarise phases
zstd -d --stdout /tmp/kv_trace.csv.zst \
  | awk -F, 'NR>1 {print $6}' \
  | sort | uniq -c
```

---

## Compatibility

All existing benchmark behaviour is **completely unchanged** when
`--io-trace-log` is not specified. There are no breaking changes to
existing CLI arguments, config files, or the Python API.

---

## Implementation Notes

| Component | Role |
|---|---|
| `kv_cache/tracer.py` | `IOTracer`: thread-safe CSV writer, optional zstd, context-manager support |
| `kv_cache/backends.py` | `NullBackend`: no-op write/read used for all tiers in trace mode |
| `kv_cache/cache.py` | Passes `io_tracer=` and `tensor_parallel=` into `MultiTierCache`; TP-adjusts `size_bytes` in all trace rows |
| `kv_cache/benchmark.py` | Manages `IOTracer` lifecycle; emits multi-GPU banner |
| `kv_cache/cli.py` | Exposes `--io-trace-log`, `--num-gpus`, `--tensor-parallel`; includes `Num GPUs`, `Tensor Parallel`, `Total GPU Memory` in XLSX export |
| `kv_cache/workload.py` | Validates TP ≤ num_gpus; warns on non-power-of-2 TP |
