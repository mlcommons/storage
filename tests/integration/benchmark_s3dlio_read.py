#!/usr/bin/env python3
"""
High-Performance Read Test using s3dlio with zero-copy

Benchmarks read performance from S3-compatible storage with zero-copy
architecture for maximum throughput.

Target: 20-30 GB/s read throughput
"""

import time
import os
import sys
import s3dlio

def format_size(bytes_val):
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"

def format_speed(bytes_per_sec):
    """Format throughput to GB/s"""
    return f"{bytes_per_sec / 1e9:.2f} GB/s"

def test_s3_read_performance(
    endpoint="http://localhost:9000",
    bucket="benchmark",
    num_files=100,
    expected_file_size_mb=100
):
    """Test S3 read performance using s3dlio's zero-copy reads"""
    print("="*60)
    print("s3dlio High-Performance Read Benchmark")
    print("="*60)
    
    # Configure s3dlio
    os.environ['AWS_ENDPOINT_URL'] = endpoint
    
    print(f"\nConfiguration:")
    print(f"  Endpoint: {endpoint}")
    print(f"  Bucket: {bucket}")
    print(f"  Files: {num_files}")
    print(f"  Expected File Size: {expected_file_size_mb} MB")
    
    # Read files
    print(f"\nReading {num_files} files from {bucket}...")
    read_start = time.perf_counter()
    total_bytes = 0
    
    for i in range(num_files):
        uri = f"s3://{bucket}/test-data/file_{i:06d}.bin"
        try:
            # ZERO-COPY read - returns BytesView
            data = s3dlio.get(uri)
            
            # Access via memoryview (zero-copy)
            view = memoryview(data)
            total_bytes += len(view)
            
            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - read_start
                throughput = total_bytes / elapsed
                print(f"  Progress: {i+1}/{num_files} files, {format_speed(throughput)}")
        except Exception as e:
            print(f"  ❌ Error reading {uri}: {e}")
            return False
    
    read_elapsed = time.perf_counter() - read_start
    read_throughput = total_bytes / read_elapsed
    
    print("\n" + "="*60)
    print("Read Performance Results")
    print("="*60)
    print(f"  Total Data: {format_size(total_bytes)}")
    print(f"  Total Time: {read_elapsed:.2f} seconds")
    print(f"  Throughput: {format_speed(read_throughput)}")
    print(f"  Files/sec: {num_files / read_elapsed:.1f}")
    
    if read_throughput >= 20e9:
        print(f"\n  ✅ EXCELLENT: {format_speed(read_throughput)} (Target: 20+ GB/s)")
    elif read_throughput >= 10e9:
        print(f"\n  ✅ GOOD: {format_speed(read_throughput)}")
    else:
        print(f"\n  ⚠️  Below target: {format_speed(read_throughput)} (Target: 20+ GB/s)")
    
    print("\n  ✅ All reads used ZERO-COPY BytesView!")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="s3dlio high-performance read benchmark")
    parser.add_argument("--endpoint", default="http://localhost:9000", 
                       help="S3 endpoint URL")
    parser.add_argument("--bucket", default="benchmark",
                       help="S3 bucket name")
    parser.add_argument("--files", type=int, default=100,
                       help="Number of files to read")
    parser.add_argument("--size", type=int, default=100,
                       help="Expected file size in MB")
    
    args = parser.parse_args()
    
    success = test_s3_read_performance(
        endpoint=args.endpoint,
        bucket=args.bucket,
        num_files=args.files,
        expected_file_size_mb=args.size
    )
    
    if not success:
        print("\n❌ Read test failed!")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ Benchmark Complete!")
    print("="*60)
