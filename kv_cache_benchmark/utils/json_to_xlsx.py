import os
import json
import pandas as pd
import glob
import argparse

def process_json_files(input_dir='.', output_file='mlperf_storage_summary.xlsx'):
    # Find all json files in the specified directory
    json_pattern = os.path.join(input_dir, '*.json')
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    data_list = []

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract summary data
            summary = data.get('summary', {})
            if not summary:
                print(f"Warning: No 'summary' key found in {json_file}")
                continue

            # Helper to safely get nested keys
            def get_nested(d, keys, default=None):
                for key in keys:
                    if isinstance(d, dict):
                        d = d.get(key, default)
                    else:
                        return default
                return d

            # Calculate storage throughput from root-level fields
            # This is the correct metric: tokens / total_storage_io_latency
            total_tokens = data.get('total_tokens_generated', 0)
            total_io_latency = data.get('total_storage_io_latency', 0)
            storage_throughput = total_tokens / total_io_latency if total_io_latency > 0 else None
            
            # Also get requests completed for storage requests/sec
            requests_completed = data.get('requests_completed', 0)
            storage_requests_per_sec = requests_completed / total_io_latency if total_io_latency > 0 else None

            # Build the row for this file
            row = {
                'Filename': json_file,
                
                # === THROUGHPUT METRICS ===
                # Storage throughput is the PRIMARY metric for MLPerf Storage benchmark
                'Storage Throughput (tok/s)': storage_throughput,
                'Storage Requests/sec': storage_requests_per_sec,
                'Total I/O Time (s)': total_io_latency,
                # Wall-clock throughput (for reference only - NOT for tier comparison)
                'Avg Throughput (tok/s)': summary.get('avg_throughput_tokens_per_sec'),
                'Requests/sec': summary.get('requests_per_second'),
                'Total Tokens': summary.get('total_tokens') or total_tokens,
                'Total Requests': summary.get('total_requests') or requests_completed,
                'Elapsed Time (s)': summary.get('elapsed_time'),
                
                # === END-TO-END LATENCY ===
                'E2E Latency Mean (ms)': get_nested(summary, ['end_to_end_latency_ms', 'mean']),
                'E2E Latency P50 (ms)': get_nested(summary, ['end_to_end_latency_ms', 'p50']),
                'E2E Latency P95 (ms)': get_nested(summary, ['end_to_end_latency_ms', 'p95']),
                'E2E Latency P99 (ms)': get_nested(summary, ['end_to_end_latency_ms', 'p99']),
                'E2E Latency P99.9 (ms)': get_nested(summary, ['end_to_end_latency_ms', 'p999']),
                'E2E Latency P99.99 (ms)': get_nested(summary, ['end_to_end_latency_ms', 'p9999']),
                
                # === STORAGE I/O LATENCY (aggregate) ===
                'Storage Latency Mean (ms)': get_nested(summary, ['storage_io_latency_ms', 'mean']),
                'Storage Latency P50 (ms)': get_nested(summary, ['storage_io_latency_ms', 'p50']),
                'Storage Latency P95 (ms)': get_nested(summary, ['storage_io_latency_ms', 'p95']),
                'Storage Latency P99 (ms)': get_nested(summary, ['storage_io_latency_ms', 'p99']),
                'Storage Latency P99.9 (ms)': get_nested(summary, ['storage_io_latency_ms', 'p999']),
                'Storage Latency P99.99 (ms)': get_nested(summary, ['storage_io_latency_ms', 'p9999']),
                
                # === GENERATION LATENCY (simulated GPU work) ===
                'Gen Latency Mean (ms)': get_nested(summary, ['generation_latency_ms', 'mean']),
                'Gen Latency P50 (ms)': get_nested(summary, ['generation_latency_ms', 'p50']),
                'Gen Latency P95 (ms)': get_nested(summary, ['generation_latency_ms', 'p95']),
                'Gen Latency P99 (ms)': get_nested(summary, ['generation_latency_ms', 'p99']),

                # === STORAGE TIER TOTAL LATENCY (Host + Device) ===
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

                # === STORAGE TIER DEVICE LATENCY (actual disk I/O - PRIMARY METRIC) ===
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

                # === STORAGE TIER HOST LATENCY (CPU serialization/deserialization) ===
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
                
                # === CACHE STATS ===
                'Cache Hit Rate': get_nested(summary, ['cache_stats', 'cache_hit_rate']),
                'Read/Write Ratio': get_nested(summary, ['cache_stats', 'read_write_ratio']),
                'Total Read (GiB)': get_nested(summary, ['cache_stats', 'total_read_gb']),
                'Total Write (GiB)': get_nested(summary, ['cache_stats', 'total_write_gb']),
                
                # === PER-TIER KV BYTES (MLPerf v3.0) ===
                'Tier GPU KV Bytes Written (GiB)': get_nested(summary, ['cache_stats', 'tier_gpu_kv_bytes_written_gb']),
                'Tier GPU KV Bytes Read (GiB)': get_nested(summary, ['cache_stats', 'tier_gpu_kv_bytes_read_gb']),
                'Tier CPU KV Bytes Written (GiB)': get_nested(summary, ['cache_stats', 'tier_cpu_kv_bytes_written_gb']),
                'Tier CPU KV Bytes Read (GiB)': get_nested(summary, ['cache_stats', 'tier_cpu_kv_bytes_read_gb']),
                'Tier Storage KV Bytes Written (GiB)': get_nested(summary, ['cache_stats', 'tier_storage_kv_bytes_written_gb']),
                'Tier Storage KV Bytes Read (GiB)': get_nested(summary, ['cache_stats', 'tier_storage_kv_bytes_read_gb']),

                # === PER-TIER BANDWIDTH (GiB/s) - PRIMARY METRICS ===
                'Tier GPU Read Bandwidth (GiB/s)': get_nested(summary, ['cache_stats', 'tier_gpu_read_bandwidth_gbps']),
                'Tier GPU Write Bandwidth (GiB/s)': get_nested(summary, ['cache_stats', 'tier_gpu_write_bandwidth_gbps']),
                'Tier CPU Read Bandwidth (GiB/s)': get_nested(summary, ['cache_stats', 'tier_cpu_read_bandwidth_gbps']),
                'Tier CPU Write Bandwidth (GiB/s)': get_nested(summary, ['cache_stats', 'tier_cpu_write_bandwidth_gbps']),
                'Tier Storage Read Bandwidth (GiB/s)': get_nested(summary, ['cache_stats', 'tier_storage_read_bandwidth_gbps']),
                'Tier Storage Write Bandwidth (GiB/s)': get_nested(summary, ['cache_stats', 'tier_storage_write_bandwidth_gbps']),

                # === TIER ENTRY DISTRIBUTION ===
                'GPU Entries': get_nested(summary, ['cache_stats', 'gpu_entries']),
                'CPU Entries': get_nested(summary, ['cache_stats', 'cpu_entries']),
                'Storage Entries': get_nested(summary, ['cache_stats', 'storage_entries']),

                # === MULTI-TURN STATS ===
                'Multi-turn Hit Rate': get_nested(summary, ['multi_turn_stats', 'hit_rate']),
            }
            
            data_list.append(row)
            print(f"Processed {json_file}")

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    if not data_list:
        print("No valid data extracted.")
        return

    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Sort by Filename
    df = df.sort_values('Filename')

    # Save to Excel
    try:
        df.to_excel(output_file, index=False)
        print(f"\nSuccessfully created {output_file} with {len(df)} records.")
        print("\nColumns included:")
        print(df.columns.tolist())
        print(f"\nPreview of data (Storage Throughput is the correct metric for tier comparison):")
        preview_cols = ['Filename', 'Storage Throughput (tok/s)', 'Tier Storage Read Bandwidth (GiB/s)', 'Total Tokens']
        available_cols = [c for c in preview_cols if c in df.columns]
        print(df[available_cols].to_string())
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        # Fallback to CSV if Excel fails (e.g. missing openpyxl)
        csv_file = output_file.replace('.xlsx', '.csv')
        print(f"Attempting to save as CSV to {csv_file}...")
        df.to_csv(csv_file, index=False)
        print(f"Successfully created {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert JSON benchmark results to Excel')
    parser.add_argument('--input-dir', '-i', default='.', help='Directory containing JSON files')
    parser.add_argument('--output', '-o', default='mlperf_storage_summary.xlsx', help='Output Excel filename')
    args = parser.parse_args()
    
    process_json_files(input_dir=args.input_dir, output_file=args.output)
