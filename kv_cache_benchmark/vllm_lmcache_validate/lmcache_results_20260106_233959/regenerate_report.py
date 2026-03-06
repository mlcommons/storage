#!/usr/bin/env python3
"""Regenerate comparison_report.txt with correct storage throughput metric."""

import json
from pathlib import Path
import numpy as np

results_dir = Path('.')

# Define configurations
configs = [
    ('vllm_baseline', 'vLLM Baseline (no LMCache)'),
    ('lmcache_gpu_only', 'LMCache GPU-only'),
    ('lmcache_cpu_offload', 'LMCache CPU Offload'),
    ('kvcache_gpu_only', 'kv-cache.py GPU-only (equal capacity)'),
    ('kvcache_gpu_cpu', 'kv-cache.py GPU+CPU (equal capacity)'),
    ('kvcache_gpu_cpu_nvme', 'kv-cache.py GPU+CPU+NVMe (equal capacity)'),
    ('kvcache_nvme_only', 'kv-cache.py NVMe-only (MLPerf Storage)'),
]

results_data = {}

for config_name, display_name in configs:
    trials = []
    for trial_file in sorted(results_dir.glob(f'{config_name}_trial*.json')):
        try:
            with open(trial_file) as f:
                data = json.load(f)
                if data.get('status') not in ['failed', 'skipped']:
                    trials.append(data)
        except Exception as e:
            pass

    if not trials:
        continue

    # Extract metrics based on backend type
    if 'vllm' in config_name or 'lmcache' in config_name:
        tok_per_sec = [t.get('tokens_per_second', 0) for t in trials]
        req_per_sec = [t.get('requests_per_second', 0) for t in trials]
        elapsed = [t.get('elapsed_time', 0) for t in trials]
        storage_tok_per_sec = tok_per_sec  # Same for real inference
    else:
        # kv-cache.py: Calculate storage throughput from raw fields
        storage_tok_per_sec = []
        tok_per_sec = []
        req_per_sec = []
        elapsed = []
        for t in trials:
            tokens = t.get('total_tokens_generated', 0)
            io_time = t.get('total_storage_io_latency', 0)
            requests = t.get('requests_completed', 0)
            # Storage throughput = tokens / storage_io_time
            st = tokens / io_time if io_time > 0 else 0
            storage_tok_per_sec.append(st)
            # Request rate based on storage time
            rps = requests / io_time if io_time > 0 else 0
            req_per_sec.append(rps)
            # Wall-clock throughput for reference
            wc_elapsed = t.get('elapsed_time', io_time)
            wc = tokens / wc_elapsed if wc_elapsed > 0 else 0
            tok_per_sec.append(wc)
            elapsed.append(io_time)

    results_data[config_name] = {
        'name': display_name,
        'trials': len(trials),
        'tok_per_sec_mean': np.mean(tok_per_sec),
        'tok_per_sec_std': np.std(tok_per_sec),
        'storage_tok_per_sec_mean': np.mean(storage_tok_per_sec),
        'storage_tok_per_sec_std': np.std(storage_tok_per_sec),
        'req_per_sec_mean': np.mean(req_per_sec),
        'req_per_sec_std': np.std(req_per_sec),
        'elapsed_mean': np.mean(elapsed),
        'elapsed_std': np.std(elapsed),
    }

# Build report
lines = []
lines.append('=' * 80)
lines.append('LMCACHE vs KV-CACHE COMPARISON RESULTS')
lines.append('=' * 80)
lines.append('')

# Real inference section
for cfg in ['vllm_baseline', 'lmcache_gpu_only', 'lmcache_cpu_offload']:
    if cfg not in results_data:
        continue
    d = results_data[cfg]
    lines.append(d['name'])
    lines.append('-' * 50)
    lines.append(f"  Trials:        {d['trials']}")
    lines.append(f"  Tokens/sec:    {d['tok_per_sec_mean']:8.2f} +/- {d['tok_per_sec_std']:7.2f}")
    lines.append(f"  Requests/sec:  {d['req_per_sec_mean']:8.2f} +/- {d['req_per_sec_std']:7.2f}")
    lines.append(f"  Elapsed time:  {d['elapsed_mean']:8.2f}s +/- {d['elapsed_std']:7.2f}s")
    lines.append('')

# kv-cache.py section with STORAGE THROUGHPUT
for cfg in ['kvcache_gpu_only', 'kvcache_gpu_cpu', 'kvcache_gpu_cpu_nvme', 'kvcache_nvme_only']:
    if cfg not in results_data:
        continue
    d = results_data[cfg]
    lines.append(d['name'])
    lines.append('-' * 50)
    lines.append(f"  Trials:                   {d['trials']}")
    lines.append(f"  Storage Throughput:       {d['storage_tok_per_sec_mean']:8.2f} +/- {d['storage_tok_per_sec_std']:7.2f} tok/s")
    lines.append(f"  Storage Requests/sec:     {d['req_per_sec_mean']:8.2f} +/- {d['req_per_sec_std']:7.2f}")
    lines.append(f"  Total I/O Time:           {d['elapsed_mean']:8.2f}s +/- {d['elapsed_std']:7.2f}s")
    lines.append('')

# Comparative analysis
lines.append('=' * 80)
lines.append('COMPARATIVE ANALYSIS')
lines.append('=' * 80)
lines.append('')
lines.append('Note: kv-cache.py tests use EQUAL total cache capacity for fair comparison.')
lines.append('      Storage Throughput = tokens / total_storage_io_latency (correct metric)')
lines.append('')

lines.append('kv-cache.py Storage Tier Comparison (Storage Throughput):')
for cfg in ['kvcache_gpu_only', 'kvcache_gpu_cpu', 'kvcache_gpu_cpu_nvme', 'kvcache_nvme_only']:
    if cfg not in results_data:
        continue
    d = results_data[cfg]
    tier_name = cfg.replace('kvcache_', '').upper().replace('_', ' ')
    lines.append(f"  {tier_name:20}: {d['storage_tok_per_sec_mean']:8.2f} tok/s")

lines.append('')

# Speedup calculation
if 'kvcache_nvme_only' in results_data:
    nvme_baseline = results_data['kvcache_nvme_only']['storage_tok_per_sec_mean']
    lines.append('  Speedup vs NVMe-only:')
    for cfg in ['kvcache_gpu_only', 'kvcache_gpu_cpu', 'kvcache_gpu_cpu_nvme']:
        if cfg not in results_data:
            continue
        d = results_data[cfg]
        speedup = d['storage_tok_per_sec_mean'] / nvme_baseline
        tier_name = cfg.replace('kvcache_', '').replace('_', ' ')
        lines.append(f"    {tier_name:16}: {speedup:.2f}x")

lines.append('')
lines.append('LMCache vs kv-cache.py (NOTE: different tools, different purposes):')
lines.append('  - LMCache: Real GPU inference with KV cache optimization')
lines.append('  - kv-cache.py: Storage I/O simulator for MLPerf Storage benchmark')
lines.append('')
if 'lmcache_cpu_offload' in results_data and 'kvcache_gpu_cpu' in results_data:
    lm = results_data['lmcache_cpu_offload']['tok_per_sec_mean']
    kv = results_data['kvcache_gpu_cpu']['storage_tok_per_sec_mean']
    lines.append(f"  LMCache CPU offload:      {lm:8.2f} tok/s (real inference)")
    lines.append(f"  kv-cache.py GPU+CPU:      {kv:8.2f} tok/s (storage I/O sim)")
    lines.append(f"  Ratio: {lm/kv:.2f}x (expected: LMCache faster due to GPU compute)")

output = '\n'.join(lines)
print(output)

with open('comparison_report.txt', 'w') as f:
    f.write(output)

print('\n\nSaved to comparison_report.txt')
