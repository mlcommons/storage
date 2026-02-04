#!/usr/bin/env python3
"""Compare all models at cpu_mem=0GB for apples-to-apples iostat comparison."""

import pandas as pd

df = pd.read_csv('iostat_analysis.csv')
df = df.rename(columns={'rMB_s': 'read_mbs', 'wMB_s': 'write_mbs', 'total_MB_s': 'total_mbs', 'total_IOPS': 'iops'})

# Filter to cpu_mem=0 only
cpu0 = df[df['cpu_mem'] == 0]

print('=' * 100)
print('ALL MODELS @ cpu_mem=0GB (Apples-to-Apples Comparison)')
print(f'Total configs: {len(cpu0)}')
print('=' * 100)

# Summary by model
print('\nSUMMARY BY MODEL:')
model_stats = cpu0.groupby('model').agg({
    'read_mbs': ['mean', 'std', 'max'],
    'write_mbs': ['mean', 'max'],
    'total_mbs': ['mean', 'max'],
    'iops': ['mean', 'max'],
    'util': 'mean',
    'model': 'count'
}).round(0)
model_stats.columns = ['Read Avg', 'Read Std', 'Read Max', 'Write Avg', 'Write Max', 'Total Avg', 'Total Max', 'IOPS Avg', 'IOPS Max', 'Util%', 'Configs']
print(model_stats.sort_values('Total Avg', ascending=False).to_string())

print('\n' + '=' * 100)
print('DETAILED: All Models x Users @ cpu_mem=0GB')
print('=' * 100)

# Pivot by model and users
pivot = cpu0.pivot_table(
    values=['read_mbs', 'write_mbs', 'total_mbs'], 
    index='model', 
    columns='users', 
    aggfunc='mean'
).round(0)

print('\nRead MB/s by Model x Users:')
print(pivot['read_mbs'].to_string())

print('\nWrite MB/s by Model x Users:')
print(pivot['write_mbs'].to_string())

print('\nTotal MB/s by Model x Users:')
print(pivot['total_mbs'].to_string())

print('\n' + '=' * 100)
print('TOP 5 CONFIGS PER MODEL @ cpu_mem=0GB')
print('=' * 100)

for model in ['mistral-7b', 'llama3.1-8b', 'llama2-7b', 'llama3.1-70b']:
    model_df = cpu0[cpu0['model'] == model].nlargest(5, 'total_mbs')
    print(f'\n{model}:')
    for _, row in model_df.iterrows():
        mca = int(row['mca'])
        users = int(row['users'])
        gen = row['gen_mode']
        read = row['read_mbs']
        write = row['write_mbs']
        total = row['total_mbs']
        print(f"  mca={mca:2d}, users={users:3d}, gen={gen:9s} => Read: {read:,.0f} MB/s, Write: {write:,.0f} MB/s, Total: {total:,.0f} MB/s")

print('\n' + '=' * 100)
print('MODEL COMPARISON @ SAME USER COUNT (cpu_mem=0GB)')
print('=' * 100)

# For each user count that all models have
common_users = [50]  # All models have 50 users
for users in common_users:
    print(f'\nUsers={users}:')
    user_df = cpu0[cpu0['users'] == users].groupby('model').agg({
        'read_mbs': 'mean',
        'write_mbs': 'mean', 
        'total_mbs': 'mean',
        'iops': 'mean',
        'util': 'mean'
    }).round(0)
    print(user_df.sort_values('total_mbs', ascending=False).to_string())

print('\n' + '=' * 100)
print('KEY INSIGHT: Which model stresses storage the most @ cpu_mem=0GB?')
print('=' * 100)

# Get best config per model
best_per_model = cpu0.loc[cpu0.groupby('model')['total_mbs'].idxmax()]
print('\nBest config per model:')
for _, row in best_per_model.sort_values('total_mbs', ascending=False).iterrows():
    print(f"  {row['model']:14s}: {row['total_mbs']:,.0f} MB/s (mca={int(row['mca'])}, users={int(row['users'])}, gen={row['gen_mode']})")

# Average across all configs
avg_per_model = cpu0.groupby('model')['total_mbs'].mean().sort_values(ascending=False)
print('\nAverage throughput per model (all configs):')
for model, avg in avg_per_model.items():
    print(f"  {model:14s}: {avg:,.0f} MB/s")
