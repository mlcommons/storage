"""
Command-line interface for KV Cache Benchmark.

Contains validate_args(), main(), and export_results_to_xlsx().
"""

import os
import sys
import json
import random
import logging
import argparse
from datetime import datetime
from dataclasses import is_dataclass, asdict
from typing import Dict

import numpy as np

from kv_cache._compat import (
    TORCH_AVAILABLE, CUPY_AVAILABLE, PANDAS_AVAILABLE, OPENPYXL_AVAILABLE,
)
from kv_cache.config import ConfigLoader, set_config, cfg
from kv_cache.models import (
    MODEL_CONFIGS, ModelConfig, GenerationMode, QoSLevel,
    QOS_PROFILES, get_qos_profiles,
)
from kv_cache.workload import validate_args
from kv_cache.benchmark import IntegratedBenchmark

if TORCH_AVAILABLE:
    import torch
if CUPY_AVAILABLE:
    import cupy as cp
if PANDAS_AVAILABLE:
    import pandas as pd

logger = logging.getLogger(__name__)


def export_results_to_xlsx(results: Dict, args, output_path: str):
    """
    Export benchmark results to an Excel file with run parameters embedded.
    Falls back to CSV if openpyxl is not available.
    """
    if not PANDAS_AVAILABLE:
        logger.warning("pandas not available, skipping XLSX export. Install with: pip install pandas")
        return

    summary = results.get('summary', {})
    if not summary:
        logger.warning("No summary data available for XLSX export")
        return

    def get_nested(d, keys, default=None):
        for key in keys:
            if isinstance(d, dict):
                d = d.get(key, default)
            else:
                return default
        return d

    run_params = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Model': args.model,
        'Num Users': args.num_users,
        'Duration (s)': args.duration,
        'GPU Memory per Card (GiB)': args.gpu_mem_gb,
        'Num GPUs': args.num_gpus,
        'Tensor Parallel': args.tensor_parallel,
        'Total GPU Memory (GiB)': args.gpu_mem_gb * args.num_gpus,
        'CPU Memory (GiB)': args.cpu_mem_gb,
        'Generation Mode': args.generation_mode,
        'Performance Profile': args.performance_profile,
        'Multi-turn': not args.disable_multi_turn,
        'Prefix Caching': not args.disable_prefix_caching,
        'RAG Enabled': args.enable_rag,
        'Autoscaling': args.enable_autoscaling,
        'Seed': args.seed,
        'Max Concurrent Allocs': args.max_concurrent_allocs,
        'Request Rate': args.request_rate,
        'Max Requests': args.max_requests,
        'Dataset Path': args.dataset_path or 'N/A',
        'Cache Dir': args.cache_dir or 'temp',
        'Storage Capacity (GiB)': args.storage_capacity_gb,
        'Precondition': args.precondition,
        'Precondition Size (GiB)': args.precondition_size_gb,
        'Precondition Threads': args.precondition_threads if args.precondition_threads > 0 else (os.cpu_count() or 4),
        'Trace Speedup': args.trace_speedup,
        'Replay Cycles': args.replay_cycles,
    }

    metrics = {
        'Total Requests': summary.get('total_requests'),
        'Total Tokens': summary.get('total_tokens'),
        'Elapsed Time (s)': summary.get('elapsed_time'),
        'Avg Throughput (tok/s)': summary.get('avg_throughput_tokens_per_sec'),
        'Storage Throughput (tok/s)': summary.get('storage_throughput_tokens_per_sec'),
        'Requests/sec': summary.get('requests_per_second'),

        'E2E Latency Mean (ms)': get_nested(summary, ['end_to_end_latency_ms', 'mean']),
        'E2E Latency P50 (ms)': get_nested(summary, ['end_to_end_latency_ms', 'p50']),
        'E2E Latency P95 (ms)': get_nested(summary, ['end_to_end_latency_ms', 'p95']),
        'E2E Latency P99 (ms)': get_nested(summary, ['end_to_end_latency_ms', 'p99']),
        'E2E Latency P99.9 (ms)': get_nested(summary, ['end_to_end_latency_ms', 'p999']),
        'E2E Latency P99.99 (ms)': get_nested(summary, ['end_to_end_latency_ms', 'p9999']),

        'Storage Latency Mean (ms)': get_nested(summary, ['storage_io_latency_ms', 'mean']),
        'Storage Latency P50 (ms)': get_nested(summary, ['storage_io_latency_ms', 'p50']),
        'Storage Latency P95 (ms)': get_nested(summary, ['storage_io_latency_ms', 'p95']),
        'Storage Latency P99 (ms)': get_nested(summary, ['storage_io_latency_ms', 'p99']),
        'Storage Latency P99.9 (ms)': get_nested(summary, ['storage_io_latency_ms', 'p999']),
        'Storage Latency P99.99 (ms)': get_nested(summary, ['storage_io_latency_ms', 'p9999']),

        'Gen Latency Mean (ms)': get_nested(summary, ['generation_latency_ms', 'mean']),
        'Gen Latency P50 (ms)': get_nested(summary, ['generation_latency_ms', 'p50']),
        'Gen Latency P95 (ms)': get_nested(summary, ['generation_latency_ms', 'p95']),
        'Gen Latency P99 (ms)': get_nested(summary, ['generation_latency_ms', 'p99']),

        'Storage Tier Read Total P50 (ms)': get_nested(summary, ['cache_stats', 'storage_read_p50_ms']),
        'Storage Tier Read Total P95 (ms)': get_nested(summary, ['cache_stats', 'storage_read_p95_ms']),
        'Storage Tier Read Total P99 (ms)': get_nested(summary, ['cache_stats', 'storage_read_p99_ms']),
        'Storage Tier Read Total P99.9 (ms)': get_nested(summary, ['cache_stats', 'storage_read_p999_ms']),
        'Storage Tier Read Total P99.99 (ms)': get_nested(summary, ['cache_stats', 'storage_read_p9999_ms']),
        'Storage Tier Write Total P50 (ms)': get_nested(summary, ['cache_stats', 'storage_write_p50_ms']),
        'Storage Tier Write Total P95 (ms)': get_nested(summary, ['cache_stats', 'storage_write_p95_ms']),
        'Storage Tier Write Total P99 (ms)': get_nested(summary, ['cache_stats', 'storage_write_p99_ms']),
        'Storage Tier Write Total P99.9 (ms)': get_nested(summary, ['cache_stats', 'storage_write_p999_ms']),
        'Storage Tier Write Total P99.99 (ms)': get_nested(summary, ['cache_stats', 'storage_write_p9999_ms']),

        'Storage Tier Read Device P50 (ms)': get_nested(summary, ['cache_stats', 'storage_read_device_p50_ms']),
        'Storage Tier Read Device P95 (ms)': get_nested(summary, ['cache_stats', 'storage_read_device_p95_ms']),
        'Storage Tier Read Device P99 (ms)': get_nested(summary, ['cache_stats', 'storage_read_device_p99_ms']),
        'Storage Tier Read Device P99.9 (ms)': get_nested(summary, ['cache_stats', 'storage_read_device_p999_ms']),
        'Storage Tier Read Device P99.99 (ms)': get_nested(summary, ['cache_stats', 'storage_read_device_p9999_ms']),
        'Storage Tier Write Device P50 (ms)': get_nested(summary, ['cache_stats', 'storage_write_device_p50_ms']),
        'Storage Tier Write Device P95 (ms)': get_nested(summary, ['cache_stats', 'storage_write_device_p95_ms']),
        'Storage Tier Write Device P99 (ms)': get_nested(summary, ['cache_stats', 'storage_write_device_p99_ms']),
        'Storage Tier Write Device P99.9 (ms)': get_nested(summary, ['cache_stats', 'storage_write_device_p999_ms']),
        'Storage Tier Write Device P99.99 (ms)': get_nested(summary, ['cache_stats', 'storage_write_device_p9999_ms']),

        'Storage Tier Read Host P50 (ms)': get_nested(summary, ['cache_stats', 'storage_read_host_p50_ms']),
        'Storage Tier Read Host P95 (ms)': get_nested(summary, ['cache_stats', 'storage_read_host_p95_ms']),
        'Storage Tier Read Host P99 (ms)': get_nested(summary, ['cache_stats', 'storage_read_host_p99_ms']),
        'Storage Tier Read Host P99.9 (ms)': get_nested(summary, ['cache_stats', 'storage_read_host_p999_ms']),
        'Storage Tier Read Host P99.99 (ms)': get_nested(summary, ['cache_stats', 'storage_read_host_p9999_ms']),
        'Storage Tier Write Host P50 (ms)': get_nested(summary, ['cache_stats', 'storage_write_host_p50_ms']),
        'Storage Tier Write Host P95 (ms)': get_nested(summary, ['cache_stats', 'storage_write_host_p95_ms']),
        'Storage Tier Write Host P99 (ms)': get_nested(summary, ['cache_stats', 'storage_write_host_p99_ms']),
        'Storage Tier Write Host P99.9 (ms)': get_nested(summary, ['cache_stats', 'storage_write_host_p999_ms']),
        'Storage Tier Write Host P99.99 (ms)': get_nested(summary, ['cache_stats', 'storage_write_host_p9999_ms']),

        'Cache Hit Rate': get_nested(summary, ['cache_stats', 'cache_hit_rate']),
        'Read/Write Ratio': get_nested(summary, ['cache_stats', 'read_write_ratio']),
        'Total Read (GiB)': get_nested(summary, ['cache_stats', 'total_read_gb']),
        'Total Write (GiB)': get_nested(summary, ['cache_stats', 'total_write_gb']),

        'Tier GPU KV Bytes Written (GiB)': get_nested(summary, ['cache_stats', 'tier_gpu_kv_bytes_written_gb']),
        'Tier CPU KV Bytes Written (GiB)': get_nested(summary, ['cache_stats', 'tier_cpu_kv_bytes_written_gb']),
        'Tier Storage KV Bytes Written (GiB)': get_nested(summary, ['cache_stats', 'tier_storage_kv_bytes_written_gb']),

        'Tier GPU KV Bytes Read (GiB)': get_nested(summary, ['cache_stats', 'tier_gpu_kv_bytes_read_gb']),
        'Tier CPU KV Bytes Read (GiB)': get_nested(summary, ['cache_stats', 'tier_cpu_kv_bytes_read_gb']),
        'Tier Storage KV Bytes Read (GiB)': get_nested(summary, ['cache_stats', 'tier_storage_kv_bytes_read_gb']),

        'Tier GPU Read Bandwidth (GiB/s)': get_nested(summary, ['cache_stats', 'tier_gpu_read_bandwidth_gbps']),
        'Tier GPU Write Bandwidth (GiB/s)': get_nested(summary, ['cache_stats', 'tier_gpu_write_bandwidth_gbps']),
        'Tier CPU Read Bandwidth (GiB/s)': get_nested(summary, ['cache_stats', 'tier_cpu_read_bandwidth_gbps']),
        'Tier CPU Write Bandwidth (GiB/s)': get_nested(summary, ['cache_stats', 'tier_cpu_write_bandwidth_gbps']),
        'Tier Storage Read Bandwidth (GiB/s)': get_nested(summary, ['cache_stats', 'tier_storage_read_bandwidth_gbps']),
        'Tier Storage Write Bandwidth (GiB/s)': get_nested(summary, ['cache_stats', 'tier_storage_write_bandwidth_gbps']),

        'GPU Entries': get_nested(summary, ['cache_stats', 'gpu_entries']),
        'CPU Entries': get_nested(summary, ['cache_stats', 'cpu_entries']),
        'Storage Entries': get_nested(summary, ['cache_stats', 'storage_entries']),

        'Multi-turn Hit Rate': get_nested(summary, ['multi_turn_stats', 'hit_rate']),
    }

    combined_row = {**run_params, **metrics}

    df = pd.DataFrame([combined_row])

    use_excel = OPENPYXL_AVAILABLE and output_path.endswith('.xlsx')

    try:
        if use_excel:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Summary', index=False)

                params_df = pd.DataFrame(list(run_params.items()), columns=['Parameter', 'Value'])
                params_df.to_excel(writer, sheet_name='Run Parameters', index=False)

                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)

                qos_metrics = summary.get('qos_metrics', {})
                if qos_metrics:
                    is_throughput = args.performance_profile == 'throughput'
                    qos_rows = []
                    for level, data in qos_metrics.items():
                        if isinstance(data, dict) and not data.get('no_data'):
                            qos_rows.append({
                                'QoS Level': level,
                                'Total Requests': data.get('total_requests'),
                                'Latency P95 (ms)': get_nested(data, ['latency_ms', 'p95']),
                                'Latency P99 (ms)': get_nested(data, ['latency_ms', 'p99']),
                                'SLA Met': 'N/A (throughput mode)' if is_throughput else get_nested(data, ['sla', 'met']),
                                'SLA Compliance': 'N/A (throughput mode)' if is_throughput else get_nested(data, ['sla', 'compliance']),
                            })
                    if qos_rows:
                        qos_df = pd.DataFrame(qos_rows)
                        qos_df.to_excel(writer, sheet_name='QoS Metrics', index=False)

                # Device tracing sheet (when --enable-latency-tracing is used)
                trace_data = results.get('device_latency_tracing', {})
                if trace_data:
                    trace_rows = []
                    display_order = [
                        ('d2c_read_us', 'D2C Read (us)', 'Device hardware time'),
                        ('d2c_write_us', 'D2C Write (us)', 'Device hardware time'),
                        ('q2d_read_us', 'Q2D Read (us)', 'I/O scheduler queue'),
                        ('q2d_write_us', 'Q2D Write (us)', 'I/O scheduler queue'),
                        ('vfs_read_us', 'VFS Read (us)', 'Application-visible'),
                        ('vfs_write_us', 'VFS Write (us)', 'Application-visible'),
                        ('fsync_us', 'fsync (us)', 'Device flush'),
                        ('write_to_fsync_us', 'Write-to-fsync (us)', 'CPU serialization gap'),
                        ('fadvise_to_read_us', 'fadvise-to-read (us)', 'Cache drop overhead'),
                        ('bssplit_read_kb', 'Block Size Read (KiB)', 'I/O size distribution'),
                        ('bssplit_write_kb', 'Block Size Write (KiB)', 'I/O size distribution'),
                        ('qd_read', 'Queue Depth Read', 'Instantaneous QD at dispatch'),
                        ('qd_write', 'Queue Depth Write', 'Instantaneous QD at dispatch'),
                        ('lba_read_gb', 'LBA Heatmap Read (GiB)', 'Spatial I/O distribution'),
                        ('lba_write_gb', 'LBA Heatmap Write (GiB)', 'Spatial I/O distribution'),
                    ]

                    def hist_pct(buckets, pct):
                        total = sum(b['count'] for b in buckets)
                        if total == 0:
                            return 0
                        target = total * pct / 100.0
                        cum = 0
                        for b in buckets:
                            cum += b['count']
                            if cum >= target:
                                return b['range_us'][0]
                        return buckets[-1]['range_us'][0]

                    for key, label, description in display_order:
                        if key not in trace_data or not trace_data[key].get('buckets'):
                            continue
                        buckets = trace_data[key]['buckets']
                        total_count = sum(b['count'] for b in buckets)
                        if total_count == 0:
                            continue
                        trace_rows.append({
                            'Metric': label,
                            'Description': description,
                            'Samples': total_count,
                            'P50': hist_pct(buckets, 50),
                            'P95': hist_pct(buckets, 95),
                            'P99': hist_pct(buckets, 99),
                            'Min Bucket': buckets[0]['range_us'][0],
                            'Max Bucket': buckets[-1]['range_us'][1],
                        })

                    if trace_rows:
                        trace_df = pd.DataFrame(trace_rows)
                        trace_df.to_excel(writer, sheet_name='Device Tracing', index=False)

                        # Raw histograms sheet
                        raw_rows = []
                        for key, label, _ in display_order:
                            if key not in trace_data or not trace_data[key].get('buckets'):
                                continue
                            for b in trace_data[key]['buckets']:
                                raw_rows.append({
                                    'Histogram': label,
                                    'Bucket Low': b['range_us'][0],
                                    'Bucket High': b['range_us'][1],
                                    'Count': b['count'],
                                })
                        if raw_rows:
                            raw_df = pd.DataFrame(raw_rows)
                            raw_df.to_excel(writer, sheet_name='Trace Histograms', index=False)

            logger.info(f"XLSX results saved to {output_path}")
        else:
            csv_path = output_path.replace('.xlsx', '.csv') if output_path.endswith('.xlsx') else output_path
            if not csv_path.endswith('.csv'):
                csv_path += '.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"CSV results saved to {csv_path} (openpyxl not available for XLSX)")

    except Exception as e:
        logger.error(f"Error saving XLSX/CSV: {e}")
        try:
            csv_path = output_path.replace('.xlsx', '.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Fallback CSV saved to {csv_path}")
        except Exception as e2:
            logger.error(f"Failed to save results: {e2}")


def main():
    """Main entry point for running the benchmark from the command line."""
    parser = argparse.ArgumentParser(description="Integrated Multi-User KV Cache Benchmark")
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: INFO)')
    parser.add_argument('--model', type=str, default='llama3.1-8b',
                        help='The model configuration to use. Models are loaded from config.yaml.')
    parser.add_argument('--num-users', type=int, default=100,
                        help='The number of concurrent users to simulate.')
    parser.add_argument('--duration', type=int, default=60,
                        help='The duration of the benchmark in seconds.')
    parser.add_argument('--gpu-mem-gb', type=float, default=16,
                        help='Per-GPU VRAM to allocate for the KV cache tier in GiB. '
                             'When --num-gpus > 1 the effective GPU pool = num_gpus × gpu-mem-gb.')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs in the tensor-parallel group. '
                             'Sets total GPU tier = num_gpus × gpu-mem-gb. '
                             'Example: --num-gpus 8 --gpu-mem-gb 141 models 8×H200.')
    parser.add_argument('--tensor-parallel', type=int, default=1,
                        help='Tensor-parallel degree (TP). '
                             'Each GPU rank stores 1/TP of each KV cache entry, '
                             'so per-rank I/O object sizes are divided by TP. '
                             'Must be >= 1 and <= --num-gpus. '
                             'Example: --tensor-parallel 8 models TP=8 for Llama 70B on 8×H200.')
    parser.add_argument('--cpu-mem-gb', type=float, default=32,
                        help='Total CPU DRAM to allocate for the KV cache spill tier in GiB.')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='The directory to use for the NVMe cache tier.')
    parser.add_argument('--generation-mode', type=str, default='realistic', choices=[g.value for g in GenerationMode],
                        help='The token generation speed simulation mode.')
    parser.add_argument('--performance-profile', type=str, default='latency', choices=['latency', 'throughput'],
                        help='The performance profile to use for pass/fail criteria.')
    parser.add_argument('--disable-multi-turn', action='store_true',
                        help='Disable multi-turn conversation caching.')
    parser.add_argument('--disable-prefix-caching', action='store_true',
                        help='Disable prefix caching.')
    parser.add_argument('--enable-rag', action='store_true',
                        help='Enable the RAG workload simulation.')
    parser.add_argument('--rag-num-docs', type=int, default=10, help='Number of RAG documents to ingest')
    parser.add_argument('--enable-autoscaling', action='store_true',
                        help='Enable workload autoscaling.')
    parser.add_argument('--autoscaler-mode', type=str, default='qos', choices=['qos', 'capacity'],
                        help='The autoscaling strategy.')
    parser.add_argument('--target-saturation', type=float, default=0.8, help='Target storage saturation (0.0-1.0)')
    parser.add_argument('--use-burst-trace', action='store_true',
                        help='Use BurstGPT trace for workload generation.')
    parser.add_argument('--burst-trace-path', type=str, default='BurstGPT/data/BurstGPT_1.csv',
                        help='Path to the BurstGPT trace file.')
    parser.add_argument('--validation-trace', type=str, default=None,
                        help='Path to a real-world trace file for validation.')
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to ShareGPT dataset JSON file.')
    parser.add_argument('--max-conversations', type=int, default=500,
                        help='Maximum number of conversations from ShareGPT dataset.')
    parser.add_argument('--output', type=str, default=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", help='Output file for results')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for random number generators.')
    parser.add_argument('--max-concurrent-allocs', type=int, default=0,
                        help='Limit concurrent allocations. 0 = unlimited.')
    parser.add_argument('--request-rate', type=float, default=0,
                        help='Target request arrival rate (requests/sec). 0 = unlimited.')
    parser.add_argument('--max-requests', type=int, default=0,
                        help='Stop after completing N requests (0 = use duration instead).')
    parser.add_argument('--xlsx-output', type=str, default=None,
                        help='Optional: Output Excel file path.')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file.')
    parser.add_argument('--storage-capacity-gb', type=float, default=0,
                        help='NVMe/storage tier capacity in GiB. 0 = auto-detect.')
    parser.add_argument('--precondition', action='store_true',
                        help='Enable SSD preconditioning phase before benchmark.')
    parser.add_argument('--precondition-size-gb', type=float, default=0,
                        help='Preconditioning data volume in GiB. 0 = 2x NVMe capacity.')
    parser.add_argument('--precondition-threads', type=int, default=0,
                        help='Number of threads for preconditioning writes. 0 = os.cpu_count().')
    parser.add_argument('--trace-speedup', type=float, default=1.0,
                        help='Speedup factor for BurstGPT trace replay timestamps.')
    parser.add_argument('--replay-cycles', type=int, default=0,
                        help='Number of complete passes through the trace dataset. 0 = infinite.')
    parser.add_argument('--prefill-only', action='store_true',
                        help='Simulate disaggregated prefill node (write-heavy, no decode reads).')
    parser.add_argument('--decode-only', action='store_true',
                        help='Simulate disaggregated decode node (read-heavy, assumes KV cache exists).')
    parser.add_argument('--io-trace-log', type=str, default=None,
                        help=(
                            'Path for the I/O trace CSV output file. '
                            'When set, activates trace mode: no real GPU/CPU/NVMe I/O is performed. '
                            'Instead every KV cache operation is logged as a row: '
                            'Timestamp,Operation,Object_Size_Bytes,Tier (Tier-0=GPU, Tier-1=CPU, Tier-2=NVMe). '
                            'The resulting trace can be replayed by an external storage benchmark tool.'
                        ))
    parser.add_argument('--enable-latency-tracing', action='store_true',
                        help='Enable bpftrace device latency tracing (requires sudo, bpftrace).')

    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.prefill_only and args.decode_only:
        parser.error("--prefill-only and --decode-only are mutually exclusive")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    args = validate_args(args)

    if args.io_trace_log:
        logger.info(f"Trace mode active: I/O operations will be logged to {args.io_trace_log} (no real hardware I/O)")

    if args.config:
        config = ConfigLoader(args.config)
        set_config(config)
        logger.info(f"Loaded configuration from {args.config}")

        # Refresh MODEL_CONFIGS and QOS_PROFILES with config values
        import kv_cache.models as _models
        _models.MODEL_CONFIGS = _models.get_model_configs()
        _models.QOS_PROFILES = get_qos_profiles()

    # Re-import MODEL_CONFIGS after potential config reload
    from kv_cache.models import MODEL_CONFIGS as CURRENT_MODEL_CONFIGS

    # Validate model choice
    if args.model not in CURRENT_MODEL_CONFIGS:
        available = ', '.join(sorted(CURRENT_MODEL_CONFIGS.keys()))
        logger.error(f"Unknown model '{args.model}'. Available models: {available}")
        sys.exit(1)

    if args.seed is not None:
        logger.info(f"Using random seed: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(args.seed)
        if CUPY_AVAILABLE:
            cp.random.seed(args.seed)

    model_config = CURRENT_MODEL_CONFIGS[args.model]
    gen_mode = GenerationMode(args.generation_mode)

    benchmark = IntegratedBenchmark(
        model_config=model_config,
        num_users=args.num_users,
        gpu_memory_gb=args.gpu_mem_gb,
        num_gpus=args.num_gpus,
        tensor_parallel=args.tensor_parallel,
        cpu_memory_gb=args.cpu_mem_gb,
        duration_seconds=args.duration,
        cache_dir=args.cache_dir,
        enable_autoscaling=args.enable_autoscaling,
        autoscaler_mode=args.autoscaler_mode,
        target_saturation=args.target_saturation,
        enable_multi_turn=not args.disable_multi_turn,
        enable_prefix_caching=not args.disable_prefix_caching,
        enable_rag=args.enable_rag,
        rag_num_docs=args.rag_num_docs,
        validation_trace=args.validation_trace,
        generation_mode=gen_mode,
        performance_profile=args.performance_profile,
        use_burst_trace=args.use_burst_trace,
        burst_trace_path=args.burst_trace_path,
        dataset_path=args.dataset_path,
        max_conversations=args.max_conversations,
        seed=args.seed,
        max_concurrent_allocs=args.max_concurrent_allocs,
        request_rate=args.request_rate,
        max_requests=args.max_requests,
        storage_capacity_gb=args.storage_capacity_gb,
        precondition=args.precondition,
        precondition_size_gb=args.precondition_size_gb,
        precondition_threads=args.precondition_threads,
        trace_speedup=args.trace_speedup,
        replay_cycles=args.replay_cycles,
        prefill_only=args.prefill_only,
        decode_only=args.decode_only,
        io_trace_log=args.io_trace_log,
        enable_latency_tracing=args.enable_latency_tracing
    )

    results = benchmark.run()

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if is_dataclass(obj):
            return asdict(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4, default=convert_numpy)

    logger.info(f"Results saved to {args.output}")

    if args.xlsx_output:
        export_results_to_xlsx(results, args, args.xlsx_output)

    # Save fio workload file when latency tracing produced one
    fio_config = results.get('fio_workload')
    if fio_config:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fio_filename = f"fio_kv_cache_workload_{timestamp}.ini"
        with open(fio_filename, 'w') as f:
            f.write(fio_config)
        logger.info(f"fio workload saved to {fio_filename}")


if __name__ == "__main__":
    main()
