#!/usr/bin/env python3
"""Compare all 3 S3 storage libraries for checkpoint writing.

Tests s3dlio, minio, and s3torchconnector backends with identical workloads
to demonstrate multi-library support in StreamingCheckpointing.
"""

import sys
import os
import time
import argparse

# Set AWS credentials from environment
os.environ['AWS_ACCESS_KEY_ID'] = 'bqVnJNb1wvrFe5Opo08y'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'psM7Whx9dpOeNFBbErf7gabRhpdvNCUskBqwG38A'
os.environ['AWS_ENDPOINT_URL'] = 'http://172.16.1.40:9000'
os.environ['AWS_REGION'] = 'us-east-1'

from mlpstorage.checkpointing import StreamingCheckpointing


def test_backend(backend: str, uri: str, size_gb: float, max_in_flight: int):
    """Test a specific backend.
    
    Args:
        backend: Backend name (s3dlio, minio, s3torchconnector)
        uri: S3 URI for checkpoint
        size_gb: Checkpoint size in GB
        max_in_flight: Number of concurrent uploads/parts
        
    Returns:
        Tuple of (success, elapsed, io_throughput) or (False, 0, 0) on failure
    """
    total_bytes = int(size_gb * (1024**3))
    
    try:
        # Backend-specific configuration
        if backend == 's3dlio':
            kwargs = {
                'part_size': 32 * 1024 * 1024,      # 32 MB parts (dgen-aligned)
                'max_in_flight': max_in_flight
            }
        elif backend == 'minio':
            kwargs = {
                'part_size': 32 * 1024 * 1024,      # 32 MB parts
                'num_parallel_uploads': max_in_flight
            }
        else:  # s3torchconnector
            kwargs = {}  # Auto-managed multipart
        
        # Create checkpoint with specified backend
        checkpoint = StreamingCheckpointing(
            chunk_size=32 * 1024 * 1024,  # 32 MB chunks
            num_buffers=4,                 # 128 MB memory
            use_dgen=True,
            backend=backend,
            **kwargs
        )
        
        start = time.perf_counter()
        result = checkpoint.save(uri, total_bytes)
        elapsed = time.perf_counter() - start
        
        io_throughput = result['io_throughput_gbps']
        
        return (True, elapsed, io_throughput)
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return (False, 0, 0)


def main():
    """Compare specified backends with customizable parameters."""
    parser = argparse.ArgumentParser(
        description='Compare S3 storage libraries for checkpoint writing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all backends with default size (32 GB) and concurrency (16)
  %(prog)s
  
  # Test only s3dlio with 1 GB
  %(prog)s --backends s3dlio --size 1
  
  # Test s3dlio and minio with 64 GB and 32 concurrent uploads
  %(prog)s --backends s3dlio minio --size 64 --max-in-flight 32
  
  # Test minio only with 0.1 GB (100 MB) for quick validation
  %(prog)s --backends minio --size 0.1 --max-in-flight 8
        """
    )
    
    parser.add_argument(
        '--backends', 
        nargs='*',
        choices=['s3dlio', 'minio', 's3torchconnector'],
        default=['s3dlio', 'minio', 's3torchconnector'],
        help='Backends to test (default: all 3)'
    )
    parser.add_argument(
        '--size',
        type=float,
        default=32.0,
        help='Checkpoint size in GB (default: 32.0)'
    )
    parser.add_argument(
        '--max-in-flight',
        type=int,
        default=16,
        help='Number of concurrent uploads/parts (default: 16)'
    )
    
    args = parser.parse_args()
    
    size_gb = args.size
    max_in_flight = args.max_in_flight
    selected_backends = args.backends
    
    print("="*80)
    print("MULTI-LIBRARY S3 STORAGE COMPARISON")
    print("="*80)
    print(f"Test size: {size_gb:.2f} GB")
    print(f"Endpoint: http://172.16.1.40:9000")
    print(f"Bucket: chckpt-test1")
    print(f"Buffer alignment: 32 MB (dgen-py optimized)")
    print(f"Max in-flight: {max_in_flight}")
    print(f"Testing backends: {', '.join(selected_backends)}")
    print("="*80)
    print()
    
    # Define all backends with their URIs and config descriptions
    all_backends = [
        ('s3dlio', 's3://chckpt-test1/compare_s3dlio.dat',
         f'32 MB parts, {max_in_flight} concurrent'),
        ('minio', 's3://chckpt-test1/compare_minio.dat',
         f'32 MB parts, {max_in_flight} concurrent'),
        ('s3torchconnector', 's3://chckpt-test1/compare_s3torch.dat',
         'Auto-managed multipart'),
    ]
    
    # Filter to only selected backends
    backends = [b for b in all_backends if b[0] in selected_backends]
    
    results = []
    
    for backend, uri, config in backends:
        print(f"Testing {backend}...")
        print(f"  Config: {config}")
        
        success, elapsed, io_throughput = test_backend(backend, uri, size_gb, max_in_flight)
        
        if success:
            total_throughput = size_gb / elapsed
            print(f"  ✅ Time: {elapsed:.2f}s")
            print(f"  ✅ I/O: {io_throughput:.2f} GB/s")
            print(f"  ✅ Total: {total_throughput:.2f} GB/s")
            results.append((backend, elapsed, io_throughput, total_throughput))
        
        print()
    
    # Summary
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Backend':<20} {'Time (s)':<10} {'I/O (GB/s)':<12} {'Total (GB/s)':<12}")
    print("-"*80)
    
    for backend, elapsed, io_throughput, total_throughput in results:
        print(f"{backend:<20} {elapsed:>8.2f}   {io_throughput:>10.2f}   {total_throughput:>10.2f}")
    
    print("="*80)
    
    if results:
        best = min(results, key=lambda x: x[1])  # Fastest time
        print(f"🏆 FASTEST: {best[0]} @ {best[3]:.2f} GB/s")
        print("="*80)
        
        if len(results) > 1:
            print()
            print(f"✅ {len(results)} storage libraries tested successfully!")
        else:
            print()
            print(f"✅ {results[0][0]} backend working correctly!")
        
        if len(selected_backends) == 3:
            print("   - s3dlio: Zero-copy multi-protocol (fastest)")
            print("   - minio: MinIO native SDK (good performance)")
            print("   - s3torchconnector: AWS official connector (auto-tuned)")
    else:
        print("❌ No backends succeeded")
        return 1


if __name__ == '__main__':
    sys.exit(main())
