# Simulated GPU Memory Tier — Problem Statement and Design

## 1. The Problem with the Current `GPUMemoryBackend`

### What it does today

`GPUMemoryBackend` is the implementation of the "GPU" tier in the three-tier KV cache
hierarchy (GPU VRAM → CPU DRAM → NVMe).  Its current code:

1. **Requires real GPU hardware** — it calls `torch.cuda.is_available()` and raises
   `RuntimeError("No GPU available for PyTorch backend")` if no CUDA device is present.
2. **Allocates real GPU memory** — every `write()` call pins a NumPy array on the host
   and DMA-transfers it to device VRAM via `torch.Tensor.to(device)`.
3. **Runs its own internal LRU eviction** — when VRAM is full it evicts its *own* oldest
   entries before the `MultiTierCache` waterfall logic has a chance to demote them
   gracefully to the CPU tier.
4. **Requires PyTorch or CuPy** — large ML framework installs just to simulate a tier
   that does not exist on the test machine.

### Why this is the wrong design for a storage simulator

The benchmark's purpose is to **simulate the I/O behaviour of a production LLM serving
system** and measure how different storage configurations affect latency and throughput.

The GPU tier in that system is where the *active working set* of KV cache lives in HBM.
For storage benchmarking purposes, we need to know:
- **How many bytes fit in GPU memory** (capacity)
- **What the effective read/write bandwidth to/from that tier is** (latency model)
- **When entries are evicted** to the CPU or NVMe tier (waterfall trigger)

We do **not** need:
- Actual tensor data in VRAM
- A real GPU
- PyTorch or CuPy installed

The current hard failure on machines without GPUs means the GPU tier is silently dropped,
every entry falls directly to the CPU tier, the benchmark produces misleading latency
numbers, and the three-tier simulation degenerates to a two-tier one.

### Concrete symptom observed

```
2026-02-25 - WARNING - Could not initialize GPU backend: No GPU available for PyTorch backend
```

Result: all entries go to CPU DRAM → CPU write P95 = 1810 ms because it is absorbing
the full write load that should be split across three tiers.

---

## 2. Proposed Solution: `SimulatedGPUBackend`

### Core idea

Replace `GPUMemoryBackend` with a pure-Python in-memory **metadata tracker** that:

- Stores only `{key → size_bytes}` — **no actual data bytes**.
- Models read/write latency by dividing `size_bytes` by a configurable **simulated
  bandwidth** (default: PCIe 5.0 host↔GPU, 64 GB/s; intra-GPU HBM reads, 3350 GB/s).
- Requires **zero GPU hardware, zero PyTorch, zero CuPy**.
- Is always available, never raises `RuntimeError`.

### Is this essentially an in-memory KV cache tracking what GPU memory would have used?

**Yes, exactly.**  The `SimulatedGPUBackend` is a `dict` keyed by cache entry ID, where
each value is the byte count of the entry.  It tracks:

```
{
    "seq_42_prefill": 536870912,   # 512 MB KV entry
    "seq_07_prefill": 134217728,   # 128 MB KV entry
    ...
}
```

The **`MultiTierCache`** already tracks total bytes used per tier in `gpu_memory_used`
and calls `_ensure_space_in_tier()` to enforce the limit.  The simulated backend does not
need to re-implement eviction — it just needs to respond to `write()` / `read()` /
`delete()` correctly and return plausible latency timings.

When an entry is evicted from the GPU tier by the waterfall, `_demote_entry()` calls
`read(key)` on this backend to get the data, then `write(key, data)` on the CPU backend.
Because the simulated GPU backend stores no actual bytes, `read()` regenerates fresh
random bytes of the correct size using `dgen_py.generate_buffer()` — which is correct for
the simulation (the bytes are synthetic data anyway; only the size and timing matter).

---

## 3. Architecture

```
MultiTierCache
│
├── backends['gpu']  = SimulatedGPUBackend(bandwidth_gb_s=64.0)
│       │
│       │  write(key, data)
│       │    → size = len(data)
│       │    → self._sizes[key] = size
│       │    → simulated_latency = size / bandwidth
│       │    → return IOTiming(total=simulated_latency)
│       │
│       │  read(key)
│       │    → size = self._sizes[key]
│       │    → raw = dgen_py.generate_buffer(size)   ← fresh random bytes, correct size
│       │    → simulated_latency = size / bandwidth
│       │    → return raw, IOTiming(total=simulated_latency)
│       │
│       └  delete(key) → del self._sizes[key]
│
├── backends['cpu']  = CPUMemoryBackend()        ← stores real bytes in DRAM
└── backends['nvme'] = NVMeBackend(...)          ← writes real bytes to disk
```

### Bandwidth model

| Configuration       | Simulated bandwidth  | Rationale                                      |
|---------------------|----------------------|------------------------------------------------|
| Default (PCIe 5.0)  | 64 GB/s              | PCIe 5.0 x16 host↔GPU DMA ceiling             |
| HBM3 intra-GPU      | 3350 GB/s            | H100/H200 HBM3 peak, for in-GPU reads          |
| Custom via CLI      | `--gpu-bandwidth-gbs`| Override for different GPU/interconnect configs |

For the initial implementation both read and write use the same `bandwidth_gb_s`
parameter (PCIe 5.0 default, 64 GB/s) since the dominant cost in LLM serving is the
host↔GPU transfer, not intra-HBM bandwidth.

### What stays the same

- `MultiTierCache` tier limits (`gpu_memory_limit`), waterfall eviction, and all
  statistics tracking are **unchanged**.
- `_handle_gpu_eviction` callback is kept for forward compatibility but is no longer
  triggered by the backend itself (waterfall handles all eviction).
- The `--gpu-mem-gb` and `--num-gpus` CLI flags continue to control the simulated
  capacity exactly as before.

---

## 4. Expected Impact

| Metric               | Before (no GPU hardware)   | After (SimulatedGPUBackend) |
|----------------------|----------------------------|-----------------------------|
| GPU tier available   | No (falls back to CPU)     | Yes (always)                |
| GPU write latency    | N/A                        | ~8 ms for 512 MB @ 64 GB/s  |
| CPU tier pressure    | 100% of entries            | Only entries > GPU capacity |
| NVMe tier used       | Only when CPU full         | Only when CPU full after GPU |
| Real RAM consumed    | All entry bytes in DRAM    | Only CPU-tier entries       |
| PyTorch required     | Yes                        | No                          |

---

## 5. Implementation Plan

1. **Add `SimulatedGPUBackend` to `backends.py`** — replaces `GPUMemoryBackend`
   in the non-trace path.

2. **Update `MultiTierCache.__init__`** in `cache.py` — always instantiate
   `SimulatedGPUBackend`; remove the `TORCH_AVAILABLE or CUPY_AVAILABLE` guard.

3. **Leave `GPUMemoryBackend`** in the file for any user who explicitly wants real GPU
   tensors and has hardware available — but it is no longer the default.

4. **Optional CLI flag** `--gpu-bandwidth-gbs` to override the simulated PCIe bandwidth
   (default 64.0).
