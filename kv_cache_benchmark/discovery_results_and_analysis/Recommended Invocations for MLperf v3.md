## Recommended Invocations by Model

### Why Two Invocations (cpu_mem=0 vs cpu_mem=4)?

| cpu_mem  | Purpose                          | Primary Metric                           | Why                                                                                                                                     |
| -------- | -------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **0 GB** | **Maximum Storage Stress**       | Decode Bytes Read, Wall-Clock Throughput | All I/O goes through NVMe. 4x more read traffic. True test of storage bandwidth.                                                        |
| **4 GB** | **Storage Throughput Benchmark** | Storage Throughput (tok/s)               | Some data cached in RAM. Storage Throughput metric works correctly (2.2x ratio). More representative of production inference workloads. |

---

### llama2-7b

| Parameter                 | cpu_mem=0 (Storage Stress) | cpu_mem=4 (Throughput) |
| ------------------------- | -------------------------- | ---------------------- |
| `--cpu-memory-gb`         | **0**                      | **4**                  |
| `--max-concurrent-allocs` | **0**                      | **4**                  |
| `--users`                 | **150**                    | **200**                |
| `--duration`              | **300**                    | **300**                |
| `--generation-mode`       | **none**                   | **none**               |
| **Expected Ratio**        | WC Tput: **4.64x**         | Stor Tput: **2.34x**   |

```bash
# llama2-7b: Storage Stress (cpu_mem=0)
python kv-cache.py --model llama2-7b --cpu-memory-gb 0 --max-concurrent-allocs 0 --users 150 --duration 300 --generation-mode none --output results/llama2-7b_stress_trial${N}.json

# llama2-7b: Throughput Benchmark (cpu_mem=4)
python kv-cache.py --model llama2-7b --cpu-memory-gb 4 --max-concurrent-allocs 4 --users 200 --duration 300 --generation-mode none --output results/llama2-7b_tput_trial${N}.json
```

---

### llama3.1-8b

| Parameter | cpu_mem=0 (Storage Stress) | cpu_mem=4 (Throughput) |
|-----------|---------------------------|------------------------|
| `--cpu-memory-gb` | **0** | **4** |
| `--max-concurrent-allocs` | **0** | **0** |
| `--users` | **200** | **150** |
| `--duration` | **300** | **300** |
| `--generation-mode` | **none** | **none** |
| **Expected Ratio** | WC Tput: **2.70x** | Stor Tput: **2.87x** |

```bash
# llama3.1-8b: Storage Stress (cpu_mem=0)
python kv-cache.py --model llama3.1-8b --cpu-memory-gb 0 --max-concurrent-allocs 0 --users 200 --duration 300 --generation-mode none --output results/llama3.1-8b_stress_trial${N}.json

# llama3.1-8b: Throughput Benchmark (cpu_mem=4)
python kv-cache.py --model llama3.1-8b --cpu-memory-gb 4 --max-concurrent-allocs 0 --users 150 --duration 300 --generation-mode none --output results/llama3.1-8b_tput_trial${N}.json
```

---

### llama3.1-70b-instruct

| Parameter | cpu_mem=0 (Storage Stress) | cpu_mem=4 (Throughput) |
|-----------|---------------------------|------------------------|
| `--cpu-memory-gb` | **0** | **4** |
| `--max-concurrent-allocs` | **0** | **4** |
| `--users` | **70** | **20** |
| `--duration` | **300** | **300** |
| `--generation-mode` | **none** | **none** |
| **Expected Ratio** | WC Tput: **2.44x** | Stor Tput: **3.25x** |

```bash
# llama3.1-70b: Storage Stress (cpu_mem=0)
python kv-cache.py --model llama3.1-70b-instruct --cpu-memory-gb 0 --max-concurrent-allocs 0 --users 70 --duration 300 --generation-mode none --output results/llama3.1-70b_stress_trial${N}.json

# llama3.1-70b: Throughput Benchmark (cpu_mem=4)
python kv-cache.py --model llama3.1-70b-instruct --cpu-memory-gb 4 --max-concurrent-allocs 4 --users 20 --duration 300 --generation-mode none --output results/llama3.1-70b_tput_trial${N}.json
```

---

## Summary Table

| Model | Invocation | cpu_mem | mca | users | Primary Metric | Expected Ratio |
|-------|------------|---------|-----|-------|----------------|----------------|
| **llama2-7b** | Stress | 0 | 0 | 150 | WC Throughput | 4.64x |
| **llama2-7b** | Tput | 4 | 4 | 200 | Stor Throughput | 2.34x |
| **llama3.1-8b** | Stress | 0 | 0 | 200 | WC Throughput | 2.70x |
| **llama3.1-8b** | Tput | 4 | 0 | 150 | Stor Throughput | 2.87x |
| **llama3.1-70b** | Stress | 0 | 0 | 70 | WC Throughput | 2.44x |
| **llama3.1-70b** | Tput | 4 | 4 | 20 | Stor Throughput | 3.25x |

**Notes:**
- **70b model uses fewer users** because larger KV cache = more memory per request
- **mca=0 often best at cpu_mem=0** (no allocation throttling when fully I/O-bound)
- **mca=4 often best at cpu_mem=4** (moderate throttling helps throughput)
- **gen_mode=none** for pure storage benchmark (no simulated token delays)
- **Run 3-5 trials** and report median