#!/usr/bin/env python3
"""
Analyze iostat files from kv-cache.py benchmark runs.
Goal: Find configurations that stress storage the most for MLPerf v3 submissions.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from collections import defaultdict

def parse_iostat_file(filepath):
    """Parse an iostat file and extract device metrics."""
    metrics = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find header line and parse subsequent data lines
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Device'):
            header_idx = i
            # Parse the data line after the header (if it exists and has nvme data)
            if i + 1 < len(lines):
                data_line = lines[i + 1].strip()
                if data_line.startswith('nvme'):
                    parts = data_line.split()
                    if len(parts) >= 21:
                        try:
                            metrics.append({
                                'device': parts[0],
                                'r_s': float(parts[1]),       # reads/sec
                                'rMB_s': float(parts[2]),     # read MB/s
                                'r_await': float(parts[5]),   # read latency ms
                                'rareq_sz': float(parts[6]),  # read request size KB
                                'w_s': float(parts[7]),       # writes/sec
                                'wMB_s': float(parts[8]),     # write MB/s
                                'w_await': float(parts[11]),  # write latency ms
                                'wareq_sz': float(parts[12]), # write request size KB
                                'aqu_sz': float(parts[20]),   # average queue size
                                'util': float(parts[21]),     # utilization %
                            })
                        except (ValueError, IndexError):
                            pass
    
    return metrics

def parse_filename(filename):
    """Extract configuration from filename."""
    # iostat_nvme3n1_llama2-7b_cpu0GB_qd32_gennone_users50.txt
    basename = os.path.basename(filename)
    
    m = re.search(r'(llama\d+\.?\d*-\d+b(?:-instruct)?|mistral-\d+b)', basename, re.I)
    model = m.group(1).lower().replace('-instruct', '') if m else None
    
    m = re.search(r'cpu(\d+)GB', basename, re.I)
    cpu_mem = int(m.group(1)) if m else None
    
    m = re.search(r'qd(\d+)', basename, re.I)
    mca = int(m.group(1)) if m else None
    
    m = re.search(r'gen(none|realistic)', basename, re.I)
    gen_mode = m.group(1).lower() if m else None
    
    m = re.search(r'users(\d+)', basename, re.I)
    users = int(m.group(1)) if m else None
    
    return {
        'model': model,
        'cpu_mem': cpu_mem,
        'mca': mca,
        'gen_mode': gen_mode,
        'users': users
    }

def analyze_iostat_files(directory):
    """Analyze all iostat files in a directory."""
    results = []
    
    pattern = os.path.join(directory, 'iostat_*.txt')
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} iostat files")
    
    for filepath in files:
        config = parse_filename(filepath)
        metrics = parse_iostat_file(filepath)
        
        if not metrics:
            continue
        
        # Filter out zero-activity samples (benchmark idle periods)
        active_metrics = [m for m in metrics if m['rMB_s'] > 0 or m['wMB_s'] > 0]
        
        if not active_metrics:
            continue
        
        # Calculate averages
        avg = {
            'r_s': np.mean([m['r_s'] for m in active_metrics]),
            'rMB_s': np.mean([m['rMB_s'] for m in active_metrics]),
            'r_await': np.mean([m['r_await'] for m in active_metrics]),
            'w_s': np.mean([m['w_s'] for m in active_metrics]),
            'wMB_s': np.mean([m['wMB_s'] for m in active_metrics]),
            'w_await': np.mean([m['w_await'] for m in active_metrics]),
            'aqu_sz': np.mean([m['aqu_sz'] for m in active_metrics]),
            'util': np.mean([m['util'] for m in active_metrics]),
            'total_MB_s': np.mean([m['rMB_s'] + m['wMB_s'] for m in active_metrics]),
            'total_IOPS': np.mean([m['r_s'] + m['w_s'] for m in active_metrics]),
            'samples': len(active_metrics),
        }
        
        results.append({**config, **avg})
    
    return pd.DataFrame(results)

def main():
    # Analyze fast system iostat files
    fast_dir = 'results_fast/results'
    
    print("=" * 80)
    print("IOSTAT ANALYSIS FOR KV-CACHE BENCHMARK")
    print("Goal: Find configurations that stress storage the most")
    print("=" * 80)
    print()
    
    df = analyze_iostat_files(fast_dir)
    
    if df.empty:
        print("No iostat data found!")
        return
    
    print(f"Parsed {len(df)} configurations with iostat data")
    print()
    
    # Sort by total throughput (storage stress indicator)
    df_sorted = df.sort_values('total_MB_s', ascending=False)
    
    print("=" * 80)
    print("TOP 20 CONFIGURATIONS BY TOTAL STORAGE THROUGHPUT (MB/s)")
    print("=" * 80)
    print()
    print("| Model | CPU | MCA | Gen | Users | Read MB/s | Write MB/s | Total MB/s | IOPS | Queue | Util% |")
    print("|-------|-----|-----|-----|-------|-----------|------------|------------|------|-------|-------|")
    
    for _, row in df_sorted.head(20).iterrows():
        model_short = str(row['model']).replace('llama', 'L').replace('mistral', 'M') if row['model'] else 'N/A'
        print(f"| {model_short} | {int(row['cpu_mem']) if pd.notna(row['cpu_mem']) else 'N/A'} | {int(row['mca']) if pd.notna(row['mca']) else 'N/A'} | {row['gen_mode'] or 'N/A'} | {int(row['users']) if pd.notna(row['users']) else 'N/A'} | {row['rMB_s']:.0f} | {row['wMB_s']:.0f} | {row['total_MB_s']:.0f} | {row['total_IOPS']:.0f} | {row['aqu_sz']:.1f} | {row['util']:.1f} |")
    
    print()
    print("=" * 80)
    print("ANALYSIS BY MODEL (Average across all configs)")
    print("=" * 80)
    print()
    
    model_agg = df.groupby('model').agg({
        'rMB_s': 'mean',
        'wMB_s': 'mean',
        'total_MB_s': 'mean',
        'total_IOPS': 'mean',
        'aqu_sz': 'mean',
        'util': 'mean',
        'samples': 'sum'
    }).sort_values('total_MB_s', ascending=False)
    
    print("| Model | Avg Read MB/s | Avg Write MB/s | Avg Total MB/s | Avg IOPS | Avg Queue | Avg Util% | Configs |")
    print("|-------|---------------|----------------|----------------|----------|-----------|-----------|---------|")
    
    for model, row in model_agg.iterrows():
        model_short = str(model).replace('llama', 'L').replace('mistral', 'M') if model else 'N/A'
        print(f"| {model_short} | {row['rMB_s']:.0f} | {row['wMB_s']:.0f} | {row['total_MB_s']:.0f} | {row['total_IOPS']:.0f} | {row['aqu_sz']:.1f} | {row['util']:.1f} | {int(row['samples'])} |")
    
    print()
    print("=" * 80)
    print("ANALYSIS BY CPU MEMORY (Critical for storage stress)")
    print("=" * 80)
    print()
    
    cpu_agg = df.groupby('cpu_mem').agg({
        'rMB_s': 'mean',
        'wMB_s': 'mean',
        'total_MB_s': 'mean',
        'total_IOPS': 'mean',
        'aqu_sz': 'mean',
        'util': 'mean',
        'r_await': 'mean',
        'w_await': 'mean',
        'samples': 'sum'
    }).sort_values('total_MB_s', ascending=False)
    
    print("| CPU Mem | Avg Read MB/s | Avg Write MB/s | Avg Total MB/s | Read Lat ms | Write Lat ms | Queue | Util% |")
    print("|---------|---------------|----------------|----------------|-------------|--------------|-------|-------|")
    
    for cpu_mem, row in cpu_agg.iterrows():
        print(f"| {int(cpu_mem)} GB | {row['rMB_s']:.0f} | {row['wMB_s']:.0f} | {row['total_MB_s']:.0f} | {row['r_await']:.2f} | {row['w_await']:.2f} | {row['aqu_sz']:.1f} | {row['util']:.1f} |")
    
    print()
    print("=" * 80)
    print("ANALYSIS BY MAX CONCURRENT ALLOCS (MCA / Queue Depth)")
    print("=" * 80)
    print()
    
    mca_agg = df.groupby('mca').agg({
        'rMB_s': 'mean',
        'wMB_s': 'mean',
        'total_MB_s': 'mean',
        'total_IOPS': 'mean',
        'aqu_sz': 'mean',
        'util': 'mean',
        'samples': 'sum'
    }).sort_values('mca')
    
    print("| MCA | Avg Read MB/s | Avg Write MB/s | Avg Total MB/s | Avg IOPS | Avg Queue | Avg Util% |")
    print("|-----|---------------|----------------|----------------|----------|-----------|-----------|")
    
    for mca, row in mca_agg.iterrows():
        print(f"| {int(mca)} | {row['rMB_s']:.0f} | {row['wMB_s']:.0f} | {row['total_MB_s']:.0f} | {row['total_IOPS']:.0f} | {row['aqu_sz']:.1f} | {row['util']:.1f} |")
    
    print()
    print("=" * 80)
    print("ANALYSIS BY USER COUNT")
    print("=" * 80)
    print()
    
    user_agg = df.groupby('users').agg({
        'rMB_s': 'mean',
        'wMB_s': 'mean',
        'total_MB_s': 'mean',
        'total_IOPS': 'mean',
        'aqu_sz': 'mean',
        'util': 'mean',
        'samples': 'sum'
    }).sort_values('users')
    
    print("| Users | Avg Read MB/s | Avg Write MB/s | Avg Total MB/s | Avg IOPS | Avg Queue | Avg Util% |")
    print("|-------|---------------|----------------|----------------|----------|-----------|-----------|")
    
    for users, row in user_agg.iterrows():
        print(f"| {int(users)} | {row['rMB_s']:.0f} | {row['wMB_s']:.0f} | {row['total_MB_s']:.0f} | {row['total_IOPS']:.0f} | {row['aqu_sz']:.1f} | {row['util']:.1f} |")
    
    print()
    print("=" * 80)
    print("ANALYSIS BY GENERATION MODE")
    print("=" * 80)
    print()
    
    gen_agg = df.groupby('gen_mode').agg({
        'rMB_s': 'mean',
        'wMB_s': 'mean',
        'total_MB_s': 'mean',
        'total_IOPS': 'mean',
        'aqu_sz': 'mean',
        'util': 'mean',
        'samples': 'sum'
    }).sort_values('total_MB_s', ascending=False)
    
    print("| Gen Mode | Avg Read MB/s | Avg Write MB/s | Avg Total MB/s | Avg IOPS | Avg Queue | Avg Util% |")
    print("|----------|---------------|----------------|----------------|----------|-----------|-----------|")
    
    for gen_mode, row in gen_agg.iterrows():
        print(f"| {gen_mode} | {row['rMB_s']:.0f} | {row['wMB_s']:.0f} | {row['total_MB_s']:.0f} | {row['total_IOPS']:.0f} | {row['aqu_sz']:.1f} | {row['util']:.1f} |")
    
    print()
    print("=" * 80)
    print("KEY FINDINGS FOR MAXIMUM STORAGE STRESS")
    print("=" * 80)
    print()
    
    # Find best config for each dimension
    best_throughput = df_sorted.iloc[0]
    best_util = df.sort_values('util', ascending=False).iloc[0]
    best_queue = df.sort_values('aqu_sz', ascending=False).iloc[0]
    
    print(f"HIGHEST THROUGHPUT CONFIG:")
    print(f"  Model: {best_throughput['model']}, cpu_mem: {best_throughput['cpu_mem']}GB, mca: {best_throughput['mca']}, users: {best_throughput['users']}")
    print(f"  Total: {best_throughput['total_MB_s']:.0f} MB/s (Read: {best_throughput['rMB_s']:.0f}, Write: {best_throughput['wMB_s']:.0f})")
    print()
    
    print(f"HIGHEST UTILIZATION CONFIG:")
    print(f"  Model: {best_util['model']}, cpu_mem: {best_util['cpu_mem']}GB, mca: {best_util['mca']}, users: {best_util['users']}")
    print(f"  Utilization: {best_util['util']:.1f}%, Throughput: {best_util['total_MB_s']:.0f} MB/s")
    print()
    
    print(f"HIGHEST QUEUE DEPTH CONFIG:")
    print(f"  Model: {best_queue['model']}, cpu_mem: {best_queue['cpu_mem']}GB, mca: {best_queue['mca']}, users: {best_queue['users']}")
    print(f"  Queue Depth: {best_queue['aqu_sz']:.1f}, Throughput: {best_queue['total_MB_s']:.0f} MB/s")
    print()
    
    # Best by cpu_mem
    print("BEST CPU_MEM FOR STORAGE STRESS:")
    best_cpu = cpu_agg['total_MB_s'].idxmax()
    print(f"  cpu_mem={int(best_cpu)}GB: {cpu_agg.loc[best_cpu, 'total_MB_s']:.0f} MB/s average")
    print()
    
    # Best by model
    print("BEST MODEL FOR STORAGE STRESS:")
    best_model = model_agg['total_MB_s'].idxmax()
    print(f"  {best_model}: {model_agg.loc[best_model, 'total_MB_s']:.0f} MB/s average")
    print()
    
    # Save to CSV for further analysis
    df.to_csv('iostat_analysis.csv', index=False)
    print("Full data saved to iostat_analysis.csv")

if __name__ == '__main__':
    main()
