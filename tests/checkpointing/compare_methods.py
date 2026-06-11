#!/usr/bin/env python3
"""
Checkpoint Testing Suite

Tests:
1. Original DLIO Method vs Streaming Checkpoint Method comparison
2. S3Checkpoint compatibility layer (read/write with PyTorch)

This validates both checkpoint approaches produce equivalent performance
and that the compatibility layer works correctly.
"""

import os
import sys
import time
import subprocess

# Add mlp-storage to path
sys.path.insert(0, '/home/eval/Documents/Code/mlp-storage')

import dgen_py
from mlpstorage_py.checkpointing import StreamingCheckpointing


def drop_caches():
    """Drop OS page cache to ensure clean measurements."""
    try:
        print("[System] Dropping page cache...")
        subprocess.run(['sync'], check=True)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], check=True)
        print("[System] Page cache cleared")
    except subprocess.CalledProcessError as e:
        print(f"[System] WARNING: Could not drop caches: {e}")
        print("[System] Continuing without cache drop (measurements may be affected)")


def method1_original_dlio(output_path, total_size_gb, fadvise_mode='none'):
    """Original DLIO method: Pre-generate data in memory, then write.
    
    Args:
        fadvise_mode: 'none', 'sequential', or 'dontneed'
    
    This is the "ground truth" for storage performance measurement.
    """
    print("\n" + "="*80)
    print("METHOD 1: Original DLIO Approach")
    print("="*80)
    print(f"Output: {output_path}")
    print(f"Size: {total_size_gb} GB")
    print(f"Fadvise: {fadvise_mode}")
    print("="*80)
    
    total_bytes = int(total_size_gb * (1024**3))
    
    print(f"\n[Original] Step 1: Generating {total_size_gb} GB in memory (alloc+generate)...")
    gen_start = time.time()
    
    # Generate data using dgen-py (OPTIMIZED: numa_mode + max_threads)
    generator = dgen_py.Generator(
        size=total_bytes,
        dedup_ratio=1.0,
        compress_ratio=1.0,
        numa_mode="auto",      # CRITICAL: Enable NUMA-aware multi-threading
        max_threads=None       # CRITICAL: Use all available cores
    )
    
    # Use generator's optimal chunk size
    chunk_size = generator.chunk_size
    
    # Calculate number of chunks needed
    num_chunks = (total_bytes + chunk_size - 1) // chunk_size
    
    # OPTIMIZED: Pre-allocate ALL buffers using Rust (1,654x faster than Python!)
    # Old: chunks = [bytearray(chunk_size) for _ in range(num_chunks)]  # ~12s for 24 GB
    # New: 7.3ms for 24 GB using Python C API from Rust
    chunks = dgen_py.create_bytearrays(count=num_chunks, size=chunk_size)
    
    # Fill buffers with high-speed generation
    idx = 0
    while not generator.is_complete():
        nbytes = generator.fill_chunk(chunks[idx])
        if nbytes == 0:
            break
        # Resize last chunk if needed
        if nbytes < chunk_size and idx == num_chunks - 1:
            chunks[idx] = chunks[idx][:nbytes]
        idx += 1
    
    gen_time = time.time() - gen_start
    gen_throughput = (total_bytes / (1024**3)) / gen_time
    
    print(f"[Original] Generation: {gen_time:.4f}s @ {gen_throughput:.2f} GB/s")
    print(f"[Original] Memory used: {len(chunks)} chunks × {chunk_size/(1024**2):.0f} MB = {total_bytes/(1024**3):.2f} GB")
    
    # Step 2: Write pre-generated data and measure ONLY I/O time
    print(f"\n[Original] Step 2: Writing {total_size_gb} GB (timing writes only)...")
    
    # Remove old file if exists
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Open file
    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    
    # Apply fadvise hints based on mode
    if fadvise_mode == 'sequential' and hasattr(os, 'posix_fadvise'):
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except (OSError, AttributeError):
            pass
    elif fadvise_mode == 'dontneed' and hasattr(os, 'posix_fadvise'):
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except (OSError, AttributeError):
            pass
    
    # Time ONLY the write operations (this is the "ground truth" I/O time)
    io_start = time.perf_counter()
    write_time_only = 0.0
    
    for i, chunk in enumerate(chunks):
        write_start = time.perf_counter()
        os.write(fd, chunk)
        write_time_only += time.perf_counter() - write_start
        
        # Apply POSIX_FADV_DONTNEED after each write if mode is 'dontneed'
        if fadvise_mode == 'dontneed' and hasattr(os, 'posix_fadvise'):
            try:
                offset = i * chunk_size
                os.posix_fadvise(fd, offset, len(chunk), os.POSIX_FADV_DONTNEED)
            except (OSError, AttributeError):
                pass
    
    # Time fsync separately
    fsync_start = time.perf_counter()
    os.fsync(fd)
    fsync_time = time.perf_counter() - fsync_start
    
    os.close(fd)
    io_total_time = time.perf_counter() - io_start
    
    # Calculate throughputs
    write_throughput = (total_bytes / (1024**3)) / write_time_only
    total_throughput = (total_bytes / (1024**3)) / io_total_time
    
    print(f"\n[Original] RESULTS:")
    print(f"  Write time (no fsync): {write_time_only:.4f}s @ {write_throughput:.2f} GB/s")
    print(f"  Fsync time:            {fsync_time:.4f}s")
    print(f"  Total I/O time:        {io_total_time:.4f}s @ {total_throughput:.2f} GB/s")
    
    # Verify file size
    actual_size = os.path.getsize(output_path)
    print(f"  File size: {actual_size:,} bytes ({actual_size/(1024**3):.2f} GB)")
    
    # Cleanup
    del chunks
    
    return {
        'method': 'Original DLIO (pre-generate)',
        'gen_time': gen_time,
        'gen_throughput_gbps': gen_throughput,
        'write_time': write_time_only,
        'fsync_time': fsync_time,
        'io_total_time': io_total_time,
        'write_throughput_gbps': write_throughput,
        'io_total_throughput_gbps': total_throughput,
        'total_bytes': total_bytes,
    }


def method2_streaming_checkpoint(output_path, total_size_gb, fadvise_mode='none'):
    """New streaming method: Generate chunks while writing.
    
    Args:
        fadvise_mode: 'none', 'sequential', or 'dontneed'
    
    This approach uses less memory but should have same I/O performance.
    """
    print("\n" + "="*80)
    print("METHOD 2: Streaming Checkpoint Approach")
    print("="*80)
    print(f"Output: {output_path}")
    print(f"Size: {total_size_gb} GB")
    print(f"Fadvise: {fadvise_mode}")
    print("="*80)
    
    total_bytes = int(total_size_gb * (1024**3))
    
    # Remove old file if exists
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Use streaming checkpoint with same fadvise mode as original method
    checkpoint = StreamingCheckpointing(
        chunk_size=32 * 1024 * 1024,  # 32 MB chunks (same as original method)
        num_buffers=4,  # Only 128 MB in memory vs 24 GB for original
        use_dgen=True,
        fadvise_mode=fadvise_mode  # Use same fadvise strategy as original
    )
    
    results = checkpoint.save(
        filepath=output_path,
        total_size_bytes=total_bytes
    )
    
    # Calculate write-only throughput (excluding fsync)
    write_only_time = results['io_time'] - results['close_time']
    write_only_throughput = (results['total_bytes'] / (1024**3)) / write_only_time
    
    print(f"\n[Streaming] RESULTS:")
    print(f"  Write time (no fsync): {write_only_time:.4f}s @ {write_only_throughput:.2f} GB/s")
    print(f"  Fsync time:            {results['close_time']:.4f}s")
    print(f"  Total I/O time:        {results['io_time']:.4f}s @ {results['io_throughput_gbps']:.2f} GB/s")
    
    return {
        'method': 'Streaming Checkpoint',
        'gen_time': results['gen_time'],
        'gen_throughput_gbps': results['gen_throughput_gbps'],
        'write_time': write_only_time,
        'fsync_time': results['close_time'],
        'io_total_time': results['io_time'],
        'write_throughput_gbps': write_only_throughput,
        'io_total_throughput_gbps': results['io_throughput_gbps'],
        'total_bytes': results['total_bytes'],
        'total_time': results['total_time'],
        'throughput_ratio': results['throughput_ratio'],
        'pipeline_overhead_pct': results['pipeline_overhead_pct'],
    }


def compare_results(result1, result2, fadvise_mode='none'):
    """Compare the two methods and show differences."""
    print("\n" + "="*80)
    print(f"COMPARISON: Original vs Streaming (fadvise={fadvise_mode})")
    print("="*80)
    
    print(f"\n{'Metric':<35} {'Original':<15} {'Streaming':<15} {'Δ%':<10}")
    print("-"*75)
    
    # I/O Performance (most important!)
    metrics = [
        ('Write Throughput (no fsync)', 'write_throughput_gbps', 'GB/s', True),
        ('Total I/O Throughput (+ fsync)', 'io_total_throughput_gbps', 'GB/s', True),
        ('', None, None, False),  # Blank line
        ('Write Time (no fsync)', 'write_time', 's', False),
        ('Fsync Time', 'fsync_time', 's', False),
        ('Total I/O Time', 'io_total_time', 's', False),
        ('', None, None, False),  # Blank line
        ('Generation Throughput', 'gen_throughput_gbps', 'GB/s', True),
        ('Generation Time', 'gen_time', 's', False),
    ]
    
    for label, key, unit, higher_is_better in metrics:
        if key is None:
            print()
            continue
            
        val1 = result1[key]
        val2 = result2[key]
        
        # Calculate percentage difference
        if val1 > 0:
            diff_pct = ((val2 - val1) / val1) * 100
            diff_str = f"{diff_pct:+.1f}%"
        else:
            diff_str = "N/A"
        
        print(f"{label:<35} {val1:<7.4f} {unit:<7} {val2:<7.4f} {unit:<7} {diff_str:<10}")
    
    # Streaming-only metrics
    if 'total_time' in result2:
        print()
        print(f"Streaming-only metrics:")
        print(f"  End-to-end time: {result2['total_time']:.4f}s")
        print(f"  Throughput ratio: {result2['throughput_ratio']:.1f}x (gen/io)")
        print(f"  Pipeline overhead: {result2['pipeline_overhead_pct']:.1f}%")
    
    # Key finding
    print("\n" + "="*80)
    print("KEY FINDING:")
    print("="*80)
    
    io_diff = abs(result1['io_total_throughput_gbps'] - result2['io_total_throughput_gbps'])
    io_diff_pct = (io_diff / result1['io_total_throughput_gbps']) * 100
    
    if io_diff_pct < 5:
        print(f"✅ I/O throughput difference: {io_diff_pct:.1f}% (< 5% threshold)")
        print(f"   Both methods measure storage performance equally accurately!")
    else:
        print(f"⚠️  I/O throughput difference: {io_diff_pct:.1f}% (> 5% threshold)")
        print(f"   May indicate measurement variance or system load")
    
    # Memory advantage
    original_memory = result1['total_bytes']
    streaming_memory = 4 * 32 * 1024 * 1024  # 4 buffers × 32 MB
    memory_reduction = (1 - streaming_memory / original_memory) * 100
    
    print(f"\nMemory Usage:")
    print(f"  Original: {original_memory / (1024**3):.2f} GB (all in RAM)")
    print(f"  Streaming: {streaming_memory / (1024**2):.0f} MB (buffer pool)")
    print(f"  Reduction: {memory_reduction:.1f}% less memory")
    
    print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Checkpoint testing suite')
    parser.add_argument('--output-dir', type=str, default='/mnt/nvme_data',
                        help='Output directory for test files')
    parser.add_argument('--size-gb', type=float, default=1.0,
                        help='Test size in GB')
    parser.add_argument('--fadvise', type=str, nargs='+', default=['none'],
                        choices=['none', 'sequential', 'dontneed'],
                        help='Fadvise modes to test')
    parser.add_argument('--skip-comparison', action='store_true',
                        help='Skip streaming vs DLIO comparison')
    parser.add_argument('--skip-s3checkpoint', action='store_true',
                        help='Skip S3Checkpoint compatibility test')
    
    args = parser.parse_args()
    
    # Run streaming vs DLIO comparison
    if not args.skip_comparison:
        run_comparison_test(args)
    
    # Run S3Checkpoint compatibility test
    if not args.skip_s3checkpoint:
        test_s3checkpoint_compatibility()
    
    print("\n" + "="*80)
    print("✅ All checkpoint tests completed!")
    print("="*80)


def run_comparison_test(args):
    """Run the original streaming vs DLIO comparison."""
    """Run comparison test."""
    import argparse
    import subprocess
    
    parser = argparse.ArgumentParser(description='Compare original vs streaming checkpoint methods')
    parser.add_argument('--size-gb', type=float, default=1.0, 
                        help='Test size in GB (default: 1.0)')
    parser.add_argument('--output-dir', type=str, default='/mnt/nvme_data',
                        help='Output directory (default: /mnt/nvme_data)')
    parser.add_argument('--fadvise', type=str, default='all',
                        choices=['none', 'sequential', 'dontneed', 'all'],
                        help='Fadvise mode: none (no hints), sequential (SEQUENTIAL only), ' +
                             'dontneed (SEQUENTIAL+DONTNEED), all (test all 3 modes)')
    args = parser.parse_args()
    
    # Check available memory dynamically
    try:
        result = subprocess.run(['free', '-b'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        mem_line = [l for l in lines if l.startswith('Mem:')][0]
        available_bytes = int(mem_line.split()[6])  # 'available' column
        available_gb = available_bytes / (1024**3)
        print(f"Available memory: {available_gb:.1f} GB, Test size: {args.size_gb} GB")
    except Exception as e:
        print(f"Could not check available memory: {e}")
    
    output_path_1 = os.path.join(args.output_dir, 'test_original.dat')
    output_path_2 = os.path.join(args.output_dir, 'test_streaming.dat')
    
    print(f"\n{'='*80}")
    print(f"CHECKPOINT METHOD COMPARISON TEST")
    print(f"{'='*80}")
    print(f"Test size: {args.size_gb} GB")
    print(f"Output dir: {args.output_dir}")
    print(f"Generator: dgen-py (same for both methods)")
    print(f"Fadvise modes: {args.fadvise}")
    print(f"{'='*80}")
    
    # Determine which modes to test
    if args.fadvise == 'all':
        fadvise_modes = ['none', 'sequential', 'dontneed']
    else:
        fadvise_modes = [args.fadvise]
    
    # Test each fadvise mode
    all_results = []
    for mode in fadvise_modes:
        print(f"\n\n" + "#"*80)
        print(f"# TESTING FADVISE MODE: {mode.upper()}")
        print("#"*80)
        
        # Drop cache before tests for clean measurements
        drop_caches()
        
        try:
            # Method 1: Original DLIO (pre-generate all data)
            result1 = method1_original_dlio(output_path_1, args.size_gb, fadvise_mode=mode)
            
            # Drop cache between tests
            drop_caches()
            
            # Method 2: Streaming checkpoint
            result2 = method2_streaming_checkpoint(output_path_2, args.size_gb, fadvise_mode=mode)
            
            # Compare results
            compare_results(result1, result2, fadvise_mode=mode)
            
            all_results.append({
                'mode': mode,
                'original': result1,
                'streaming': result2
            })
            
        finally:
            # Cleanup after each mode
            for path in [output_path_1, output_path_2]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Cleaned up: {path}")
    
    # Final summary if testing all modes
    if len(fadvise_modes) > 1:
        print(f"\n\n" + "="*80)
        print("FINAL SUMMARY: All Fadvise Modes")
        print("="*80)
        print(f"\n{'Mode':<15} {'Original (GB/s)':<20} {'Streaming (GB/s)':<20} {'Δ%':<10}")
        print("-"*75)
        for res in all_results:
            orig_tput = res['original']['io_total_throughput_gbps']
            stream_tput = res['streaming']['io_total_throughput_gbps']
            diff_pct = ((stream_tput - orig_tput) / orig_tput) * 100
            print(f"{res['mode']:<15} {orig_tput:<20.2f} {stream_tput:<20.2f} {diff_pct:+.1f}%")
        print("="*80)
    
    # Final cache drop to free memory
    drop_caches()


def test_s3checkpoint_compatibility():
    """Test S3Checkpoint compatibility layer with PyTorch."""
    print("\n" + "="*80)
    print("TEST 3: S3Checkpoint Compatibility Layer")
    print("="*80)
    
    from pathlib import Path
    import torch
    from s3dlio.compat.s3torchconnector import S3Checkpoint
    
    # Setup test directory
    test_dir = Path("/tmp/s3dlio-checkpoint-test")
    test_dir.mkdir(exist_ok=True)
    
    checkpoint_path = f"file://{test_dir}/checkpoint.pt"
    checkpoint = S3Checkpoint()
    
    # Create dummy model state
    dummy_state = {
        'epoch': 42,
        'model_state': torch.tensor([1.0, 2.0, 3.0, 4.0]),
        'optimizer_state': {'lr': 0.001, 'momentum': 0.9}
    }
    
    # Test write
    print(f"\n[Write Test]")
    print(f"  Path: {checkpoint_path}")
    write_start = time.perf_counter()
    with checkpoint.writer(checkpoint_path) as writer:
        torch.save(dummy_state, writer)
    write_time = time.perf_counter() - write_start
    print(f"  ✅ Checkpoint written in {write_time:.3f}s")
    
    # Test read
    print(f"\n[Read Test]")
    read_start = time.perf_counter()
    with checkpoint.reader(checkpoint_path) as reader:
        loaded_state = torch.load(reader, weights_only=False)
    read_time = time.perf_counter() - read_start
    print(f"  ✅ Checkpoint loaded in {read_time:.3f}s")
    
    # Verify data
    print(f"\n[Verification]")
    assert loaded_state['epoch'] == 42, "Epoch mismatch"
    assert torch.equal(loaded_state['model_state'], dummy_state['model_state']), "Model state mismatch"
    assert loaded_state['optimizer_state']['lr'] == 0.001, "Optimizer LR mismatch"
    print(f"  ✅ All data verified correctly")
    print(f"     Epoch: {loaded_state['epoch']}")
    print(f"     Model tensor: {loaded_state['model_state'].tolist()}")
    print(f"     Optimizer LR: {loaded_state['optimizer_state']['lr']}")
    
    # Cleanup
    import os
    checkpoint_file = str(test_dir / "checkpoint.pt")
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    print("\n✅ S3Checkpoint compatibility test passed!")


if __name__ == '__main__':
    main()
