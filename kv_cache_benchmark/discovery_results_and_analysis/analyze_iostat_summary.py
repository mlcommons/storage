#!/usr/bin/env python3
"""Summarize iostat analysis focusing on cpu_mem=0 configurations for maximum storage stress."""

import pandas as pd

df = pd.read_csv('iostat_analysis.csv')
# Rename columns for convenience
df = df.rename(columns={'rMB_s': 'read_mbs', 'wMB_s': 'write_mbs', 'total_MB_s': 'total_mbs', 'total_IOPS': 'iops'})

# Sort by read throughput
top_read = df.nlargest(30, 'read_mbs')
print('=' * 100)
print('TOP 30 CONFIGURATIONS BY READ THROUGHPUT (Maximum Storage Stress)')
print('=' * 100)
print(f"{'Model':<12} {'CPU':<5} {'MCA':<5} {'Gen':<10} {'Users':<6} {'Read MB/s':>10} {'Write MB/s':>11} {'Util%':>7}")
print('-' * 100)
for _, row in top_read.iterrows():
    print(f"{row['model']:<12} {int(row['cpu_mem']):<5} {int(row['mca']):<5} {row['gen_mode']:<10} {int(row['users']):<6} {row['read_mbs']:>10.0f} {row['write_mbs']:>11.0f} {row['util']:>7.1f}")

print()
print('=' * 100)
print('SUMMARY: Optimal Parameters for Maximum Storage Stress (cpu_mem=0 only)')
print('=' * 100)

# Filter to cpu_mem=0 (maximum storage stress)
cpu0 = df[df['cpu_mem'] == 0]

print()
print('BY MODEL (cpu_mem=0):')
model_avg = cpu0.groupby('model').agg({'read_mbs': 'mean', 'write_mbs': 'mean', 'total_mbs': 'mean', 'util': 'mean', 'model': 'count'})
model_avg.columns = ['Read MB/s', 'Write MB/s', 'Total MB/s', 'Util%', 'Configs']
print(model_avg.sort_values('Total MB/s', ascending=False).round(0).to_string())

print()
print('BY USERS (cpu_mem=0):')
users_avg = cpu0.groupby('users').agg({'read_mbs': 'mean', 'write_mbs': 'mean', 'total_mbs': 'mean', 'util': 'mean', 'model': 'count'})
users_avg.columns = ['Read MB/s', 'Write MB/s', 'Total MB/s', 'Util%', 'Configs']
print(users_avg.sort_values('Total MB/s', ascending=False).round(0).to_string())

print()
print('BY MCA (cpu_mem=0):')
mca_avg = cpu0.groupby('mca').agg({'read_mbs': 'mean', 'write_mbs': 'mean', 'total_mbs': 'mean', 'util': 'mean', 'model': 'count'})
mca_avg.columns = ['Read MB/s', 'Write MB/s', 'Total MB/s', 'Util%', 'Configs']
print(mca_avg.sort_values('Total MB/s', ascending=False).round(0).to_string())

print()
print('BY GEN_MODE (cpu_mem=0):')
gen_avg = cpu0.groupby('gen_mode').agg({'read_mbs': 'mean', 'write_mbs': 'mean', 'total_mbs': 'mean', 'util': 'mean', 'model': 'count'})
gen_avg.columns = ['Read MB/s', 'Write MB/s', 'Total MB/s', 'Util%', 'Configs']
print(gen_avg.sort_values('Total MB/s', ascending=False).round(0).to_string())

print()
print('=' * 100)
print('OPTIMAL INVOCATION PARAMETERS FOR MAXIMUM STORAGE STRESS')
print('=' * 100)

# Find best combination
best = cpu0.nlargest(1, 'total_mbs').iloc[0]
print(f"""
RECOMMENDED INVOCATION:
  --model: mistral-7b or llama3.1-8b (both show ~10 GB/s peak throughput)
  --cpu_mem: 0GB (forces all I/O to storage, 6.8x higher read throughput than cpu_mem=4GB)
  --max_concurrent_allocs: 16 or 32 (slight peak at 16)
  --users: 200 (highest throughput) or 150 (good balance)
  --gen_mode: none (slightly higher throughput than realistic)

PEAK CONFIGURATION OBSERVED:
  {best['model']}, cpu_mem={int(best['cpu_mem'])}GB, mca={int(best['mca'])}, gen={best['gen_mode']}, users={int(best['users'])}
  Read: {best['read_mbs']:.0f} MB/s, Write: {best['write_mbs']:.0f} MB/s, Total: {best['total_mbs']:.0f} MB/s
  
KEY INSIGHT: cpu_mem=0GB is THE critical parameter for storage stress:
  - cpu_mem=0GB: {cpu0['read_mbs'].mean():.0f} MB/s average read throughput
  - cpu_mem=4GB: {df[df['cpu_mem']==4]['read_mbs'].mean():.0f} MB/s average read throughput
  - Ratio: {cpu0['read_mbs'].mean() / df[df['cpu_mem']==4]['read_mbs'].mean():.1f}x more reads with cpu_mem=0
""")

# Cross-tab analysis: Model x Users for cpu_mem=0
print('=' * 100)
print('DETAILED: Model x Users (cpu_mem=0, averaged across MCA and gen_mode)')
print('=' * 100)
pivot = cpu0.pivot_table(values='total_mbs', index='model', columns='users', aggfunc='mean').round(0)
print(pivot.to_string())

print()
print('=' * 100)
print('VALIDATION: Comparing cpu_mem settings')
print('=' * 100)
cpu_comparison = df.groupby('cpu_mem').agg({
    'read_mbs': ['mean', 'max'],
    'write_mbs': ['mean', 'max'],
    'total_mbs': ['mean', 'max'],
    'util': 'mean'
}).round(0)
print(cpu_comparison.to_string())
