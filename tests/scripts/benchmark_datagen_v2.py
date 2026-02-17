#!/usr/bin/env python3
"""
Data Generation Benchmark V2 - Testing fill_chunk() buffer reuse patterns.

This version focuses on fill_chunk() with buffer pooling to achieve:
- High throughput (>20 GB/s from fill_chunk vs ~1.5 GB/s from get_chunk+bytes)
- Low memory usage (<2GB for 3000×16MB objects via buffer reuse)
- Compatibility with upload libraries (bytearray works with s3dlio buffer protocol)

NEW Approaches (V2):
6. fill_chunk() + Single Buffer - ONE reusable buffer (16MB RAM for 16MB objects)
7. fill_chunk() + Buffer Pool (N buffers) - Pool of N buffers (N×16MB RAM)

Comparison against V1 approaches:
1. Streaming + NO COPY (reuse bytearray buffer) - baseline, already uses fill_chunk()
2. Streaming + COPY to bytes() (queue safety) 
3. Large chunks split (32MB → multiple smaller chunks)
4. BytesView + get_chunk() - SINGLE producer (dgen-py handles parallelism)
5. BytesView + get_chunk() - MULTIPLE producers (4 concurrent producers)

KEY INSIGHT from FAST tests:
- get_chunk() + bytes() conversion: 1.55 GB/s (bottleneck!)
- fill_chunk() with buffer: 23.82 GB/s (15x faster)
- All Python libraries PUT at 1.45-1.71 GB/s (data gen limited)
- Rust s3-cli PUT: 6.5 GB/s (proves network capable)
→ SOLUTION: Use fill_chunk() to eliminate bytes() conversion bottleneck

Tests multiple object sizes: 1MB, 8MB, 16MB, 32MB
Can test with 100 or 1000+ objects to validate buffer reuse.

Usage:
    python3 benchmark_datagen_v2.py --count 100 --sizes 16
    python3 benchmark_datagen_v2.py --count 3000 --sizes 16  # Test 3000×16MB with <2GB RAM
    python3 benchmark_datagen_v2.py --quick  # Quick test (100 objects, all sizes)
    python3 benchmark_datagen_v2.py --full   # Full test (1000 objects, all sizes)
"""

import argparse
import time
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# dgen_py is REQUIRED - no fallback is fast enough
try:
    import dgen_py
    HAS_DGEN = True
except ImportError:
    print("ERROR: dgen_py not available. This benchmark requires dgen_py.")
    print("Install with: pip install dgen-py")
    print("")
    print("NOTE: There is NO viable fallback. dgen_py is 50-200x faster than")
    print("      alternatives like os.urandom(). Data generation speed is critical.")
    sys.exit(1)


def benchmark_no_copy(num_objects, chunk_size_mb):
    """
    APPROACH 1: Streaming with NO COPY (reuse buffer directly)
    Fastest but requires careful handling - buffer gets overwritten.
    """
    chunk_size = chunk_size_mb * 1024 * 1024
    total_size = num_objects * chunk_size
    
    print(f"  → No Copy (reuse buffer): {chunk_size_mb}MB × {num_objects:,} objects...", end=" ", flush=True)
    
    # Create generator for total dataset
    gen = dgen_py.Generator(
        size=total_size,
        dedup_ratio=1.0,
        compress_ratio=1.0,
        numa_mode="auto",
        max_threads=None,
        seed=12345
    )
    
    # ONE reusable buffer (constant memory)
    buffer = bytearray(chunk_size)
    
    start = time.perf_counter()
    
    for i in range(num_objects):
        # Fill buffer with generated data (OVERWRITES previous data)
        nbytes = gen.fill_chunk(buffer)
        if nbytes == 0:
            print(f"\n  Warning: Generator exhausted at object {i}")
            break
        
        # In real usage: must consume buffer IMMEDIATELY before next iteration
        # e.g., f.write(buffer) or upload(buffer)
    
    elapsed = time.perf_counter() - start
    throughput = (total_size / (1024**3)) / elapsed
    
    print(f"{throughput:.2f} GB/s in {elapsed:.3f}s")
    
    return elapsed, throughput


def benchmark_with_copy(num_objects, chunk_size_mb):
    """
    APPROACH 2: Streaming WITH COPY to bytes() (queue safety)
    Safer for async queues but has copy overhead.
    """
    chunk_size = chunk_size_mb * 1024 * 1024
    total_size = num_objects * chunk_size
    
    print(f"  → With Copy (bytes()): {chunk_size_mb}MB × {num_objects:,} objects...", end=" ", flush=True)
    
    # Create generator for total dataset
    gen = dgen_py.Generator(
        size=total_size,
        dedup_ratio=1.0,
        compress_ratio=1.0,
        numa_mode="auto",
        max_threads=None,
        seed=12345
    )
    
    # ONE reusable buffer
    buffer = bytearray(chunk_size)
    
    start = time.perf_counter()
    
    for i in range(num_objects):
        # Fill buffer
        nbytes = gen.fill_chunk(buffer)
        if nbytes == 0:
            print(f"\n  Warning: Generator exhausted at object {i}")
            break
        
        # Copy to bytes (queue safety) - THIS IS THE KEY DIFFERENCE
        data = bytes(buffer[:nbytes])
    
    elapsed = time.perf_counter() - start
    throughput = (total_size / (1024**3)) / elapsed
    
    print(f"{throughput:.2f} GB/s in {elapsed:.3f}s")
    
    return elapsed, throughput


def benchmark_large_split(num_objects, chunk_size_mb):
    """
    APPROACH 3: Large chunks split (32MB → multiple smaller chunks)
    Generate larger chunks then split - tests if larger gen chunks help.
    """
    if chunk_size_mb >= 32:
        # Only makes sense for objects smaller than 32MB
        return 0.0, 0.0
    
    large_chunk_size = 32 * 1024 * 1024  # Always use 32MB for generation
    target_chunk_size = chunk_size_mb * 1024 * 1024
    chunks_per_large = large_chunk_size // target_chunk_size
    
    # Adjust num_objects for splitting
    num_large_chunks = (num_objects + chunks_per_large - 1) // chunks_per_large
    total_size = num_objects * target_chunk_size
    
    print(f"  → Large Split (32MB→{chunks_per_large}×{chunk_size_mb}MB): {num_objects:,} objects...", end=" ", flush=True)
    
    # Create generator for total dataset
    gen_size = num_large_chunks * large_chunk_size
    gen = dgen_py.Generator(
        size=gen_size,
        dedup_ratio=1.0,
        compress_ratio=1.0,
        numa_mode="auto",
        max_threads=None,
        seed=12345
    )
    
    # ONE large reusable buffer
    buffer = bytearray(large_chunk_size)
    
    start = time.perf_counter()
    
    objects_generated = 0
    for i in range(num_large_chunks):
        # Fill large buffer
        nbytes = gen.fill_chunk(buffer)
        if nbytes == 0:
            print(f"\n  Warning: Generator exhausted at large chunk {i}")
            break
        
        # Split into target-sized chunks with copy
        for offset in range(0, nbytes, target_chunk_size):
            if objects_generated >= num_objects:
                break
            remaining = min(target_chunk_size, nbytes - offset)
            chunk_data = bytes(buffer[offset:offset + remaining])
            objects_generated += 1
        
        if objects_generated >= num_objects:
            break
    
    elapsed = time.perf_counter() - start
    throughput = (total_size / (1024**3)) / elapsed
    
    print(f"{throughput:.2f} GB/s in {elapsed:.3f}s")
    
    return elapsed, throughput


def benchmark_bytesview_single_producer(num_objects, chunk_size_mb):
    """
    APPROACH 4: Single producer using get_chunk() with BytesView (PROPOSED OPTIMAL)
    - ONE producer calls get_chunk() sequentially
    - dgen-py uses max_threads=None (all cores via Rayon)
    - No threading coordination overhead
    - Let dgen-py's optimized Rayon pool handle all parallelism
    """
    chunk_size = chunk_size_mb * 1024 * 1024
    total_size = num_objects * chunk_size
    
    print(f"  → BytesView Single Producer (Rayon parallel): {chunk_size_mb}MB × {num_objects:,} objects...", end=" ", flush=True)
    
    # Create ONE generator for total dataset
    gen = dgen_py.Generator(
        size=total_size,
        dedup_ratio=1.0,
        compress_ratio=1.0,
        numa_mode="auto",
        max_threads=None,  # Let dgen-py use all cores
        seed=12345
    )
    
    start = time.perf_counter()
    
    # Single producer loop - dgen-py parallelizes internally
    for i in range(num_objects):
        # get_chunk() returns BytesView (zero-copy, immutable)
        # Rayon parallelizes the internal data generation
        data = gen.get_chunk(chunk_size)
        
        # Convert to bytes (simulating what we do for upload libs)
        data_bytes = bytes(data)
    
    elapsed = time.perf_counter() - start
    throughput = (total_size / (1024**3)) / elapsed
    
    print(f"{throughput:.2f} GB/s in {elapsed:.3f}s")
    
    return elapsed, throughput


def benchmark_bytesview_multi_producer(num_objects, chunk_size_mb, num_producers=4):
    """
    APPROACH 5: Multiple producers using get_chunk() with BytesView (CURRENT APPROACH)
    - MULTIPLE producers (4) call get_chunk() concurrently
    - Each generator uses max_threads=None (tries to use all cores)
    - Thread coordination overhead + Rayon pool contention
    - Tests if multiple producers add value or overhead
    """
    chunk_size = chunk_size_mb * 1024 * 1024
    total_size = num_objects * chunk_size
    
    print(f"  → BytesView {num_producers} Producers (each Rayon parallel): {chunk_size_mb}MB × {num_objects:,} objects...", end=" ", flush=True)
    
    # Shared state for work distribution
    next_obj_id = 0
    lock = threading.Lock()
    results = []
    
    def producer_worker(worker_id):
        nonlocal next_obj_id
        
        # Each producer gets its own generator
        gen = dgen_py.Generator(
            size=total_size,  # Each generator sized for full dataset
            dedup_ratio=1.0,
            compress_ratio=1.0,
            numa_mode="auto",
            max_threads=None,  # Each generator tries to use all cores
            seed=12345 + worker_id
        )
        
        worker_results = []
        
        while True:
            # Get next object ID
            with lock:
                if next_obj_id >= num_objects:
                    break
                obj_id = next_obj_id
                next_obj_id += 1
            
            # get_chunk() returns BytesView
            # With max_threads=None, each call tries to use all cores
            # Multiple concurrent calls = Rayon pool contention
            data = gen.get_chunk(chunk_size)
            
            # Convert to bytes (simulating what we do for upload libs)
            data_bytes = bytes(data)
            worker_results.append((obj_id, data_bytes))
        
        return worker_results
    
    start = time.perf_counter()
    
    # Run multiple producer threads
    with ThreadPoolExecutor(max_workers=num_producers) as executor:
        futures = [executor.submit(producer_worker, i) for i in range(num_producers)]
        
        for future in as_completed(futures):
            worker_data = future.result()
            results.extend(worker_data)
    
    elapsed = time.perf_counter() - start
    throughput = (total_size / (1024**3)) / elapsed
    
    print(f"{throughput:.2f} GB/s in {elapsed:.3f}s")
    
    return elapsed, throughput


def benchmark_fill_chunk_single_buffer(num_objects, chunk_size_mb):
    """
    APPROACH 6 (V2): fill_chunk() with SINGLE buffer reuse (LOWEST MEMORY)
    - ONE bytearray buffer reused for all objects
    - Memory: 1 × chunk_size (16MB for 16MB objects)
    - Use fill_chunk() → 23.82 GB/s (vs get_chunk+bytes 1.55 GB/s)
    - Simulates immediate consumption pattern (upload before next generation)
    - Perfect for streaming/queue pattern with tight producer-consumer coupling
    """
    chunk_size = chunk_size_mb * 1024 * 1024
    total_size = num_objects * chunk_size
    
    print(f"  → fill_chunk() Single Buffer (reuse): {chunk_size_mb}MB × {num_objects:,} objects...", end=" ", flush=True)
    
    # Create generator for total dataset
    gen = dgen_py.Generator(
        size=total_size,
        dedup_ratio=1.0,
        compress_ratio=1.0,
        numa_mode="auto",
        max_threads=None,  # Let dgen-py use all cores
        seed=12345
    )
    
    # ONE reusable buffer (constant memory - 16MB for 16MB objects)
    buffer = bytearray(chunk_size)
    
    start = time.perf_counter()
    
    for i in range(num_objects):
        # Fill buffer with generated data (OVERWRITES previous data)
        # This is FAST - no bytes() conversion overhead
        nbytes = gen.fill_chunk(buffer)
        if nbytes == 0:
            print(f"\n  Warning: Generator exhausted at object {i}")
            break
        
        # In real usage: must consume buffer IMMEDIATELY before next iteration
        # Simulating consumption (in real code: upload(buffer) or queue.put(buffer))
        _ = buffer  # Simulate work without actual memory allocation
    
    elapsed = time.perf_counter() - start
    throughput = (total_size / (1024**3)) / elapsed
    
    print(f"{throughput:.2f} GB/s in {elapsed:.3f}s (RAM: {chunk_size_mb}MB)")
    
    return elapsed, throughput


def benchmark_fill_chunk_buffer_pool(num_objects, chunk_size_mb, pool_size=64):
    """
    APPROACH 7 (V2): fill_chunk() with BUFFER POOL (QUEUE PATTERN)
    - Pool of N pre-allocated buffers (default: 64 to match QUEUE_SIZE)
    - Memory: N × chunk_size (64 × 16MB = 1024MB for 16MB objects)
    - Use fill_chunk() → 23.82 GB/s (vs get_chunk+bytes 1.55 GB/s)
    - Simulates producer filling queue while consumers drain it
    - Buffers rotate through pool (producer->queue->consumer->pool)
    - Realistic for async producer/consumer pattern
    """
    chunk_size = chunk_size_mb * 1024 * 1024
    total_size = num_objects * chunk_size
    pool_ram_mb = (pool_size * chunk_size) // (1024 * 1024)
    
    print(f"  → fill_chunk() Buffer Pool ({pool_size} buffers): {chunk_size_mb}MB × {num_objects:,} objects...", end=" ", flush=True)
    
    # Create generator for total dataset
    gen = dgen_py.Generator(
        size=total_size,
        dedup_ratio=1.0,
        compress_ratio=1.0,
        numa_mode="auto",
        max_threads=None,  # Let dgen-py use all cores
        seed=12345
    )
    
    # Pre-allocate buffer pool
    buffer_pool = [bytearray(chunk_size) for _ in range(pool_size)]
    
    start = time.perf_counter()
    
    for i in range(num_objects):
        # Get buffer from pool (round-robin)
        buffer = buffer_pool[i % pool_size]
        
        # Fill buffer with generated data
        nbytes = gen.fill_chunk(buffer)
        if nbytes == 0:
            print(f"\n  Warning: Generator exhausted at object {i}")
            break
        
        # Simulate queue put + consumer processing
        # In real code: queue.put(buffer), consumer uploads it, returns to pool
        _ = buffer
    
    elapsed = time.perf_counter() - start
    throughput = (total_size / (1024**3)) / elapsed
    
    print(f"{throughput:.2f} GB/s in {elapsed:.3f}s (RAM: {pool_ram_mb}MB)")
    
    return elapsed, throughput


def run_size_test(num_objects, chunk_size_mb):
    """Run all approaches for a given object size."""
    print(f"\n{'='*80}")
    print(f"Testing {chunk_size_mb}MB objects ({num_objects:,} objects = {num_objects * chunk_size_mb / 1024:.2f} GB)")
    print(f"{'='*80}")
    
    results = {}
    
    # Approach 1: No copy (fastest, requires care)
    t1, bw1 = benchmark_no_copy(num_objects, chunk_size_mb)
    results['no_copy'] = {'time': t1, 'throughput': bw1}
    
    # Approach 2: With copy (safer, overhead)
    t2, bw2 = benchmark_with_copy(num_objects, chunk_size_mb)
    results['with_copy'] = {'time': t2, 'throughput': bw2}
    
    # Calculate copy overhead
    if bw1 > 0 and bw2 > 0:
        copy_overhead_pct = ((bw1 - bw2) / bw1) * 100
        slowdown = bw1 / bw2
        print(f"\n  📊 Copy overhead: {slowdown:.2f}x slower ({bw1:.2f} → {bw2:.2f} GB/s, {copy_overhead_pct:.1f}% loss)")
    
    # Approach 3: Large split (only for <32MB objects)
    if chunk_size_mb < 32:
        t3, bw3 = benchmark_large_split(num_objects, chunk_size_mb)
        if bw3 > 0:
            results['large_split'] = {'time': t3, 'throughput': bw3}
            if bw1 > 0:
                vs_no_copy = bw3 / bw1
                print(f"  📊 Large split vs no-copy: {vs_no_copy:.2f}x ({bw1:.2f} → {bw3:.2f} GB/s)")
    
    # Approach 4: BytesView Single Producer (PROPOSED - dgen-py handles all parallelism)
    t4, bw4 = benchmark_bytesview_single_producer(num_objects, chunk_size_mb)
    results['bytesview_single'] = {'time': t4, 'throughput': bw4}
    
    # Approach 5: BytesView Multi Producer (CURRENT - 4 producers with coordination overhead)
    t5, bw5 = benchmark_bytesview_multi_producer(num_objects, chunk_size_mb, num_producers=4)
    results['bytesview_multi'] = {'time': t5, 'throughput': bw5}
    
    # Compare single vs multi producer approaches
    if bw4 > 0 and bw5 > 0:
        ratio = bw4 / bw5
        if ratio > 1.0:
            print(f"\n  📊 Single producer is {ratio:.2f}x FASTER ({bw5:.2f} → {bw4:.2f} GB/s)")
            print(f"      → Multiple producers add coordination overhead with max_threads=None")
        else:
            print(f"\n  📊 Multi producer is {1/ratio:.2f}x faster ({bw4:.2f} → {bw5:.2f} GB/s)")
            print(f"      → Multiple producers beneficial despite coordination")
    
    # Approach 6 (V2): fill_chunk() Single Buffer (LOWEST MEMORY)
    t6, bw6 = benchmark_fill_chunk_single_buffer(num_objects, chunk_size_mb)
    results['fill_single'] = {'time': t6, 'throughput': bw6}
    
    # Approach 7 (V2): fill_chunk() Buffer Pool (QUEUE PATTERN)
    t7, bw7 = benchmark_fill_chunk_buffer_pool(num_objects, chunk_size_mb, pool_size=64)
    results['fill_pool'] = {'time': t7, 'throughput': bw7}
    
    # Compare fill_chunk approaches vs get_chunk + bytes()
    print(f"\n  🔥 KEY COMPARISON: fill_chunk() vs get_chunk()+bytes()")
    if bw6 > 0 and bw4 > 0:
        improvement = bw6 / bw4
        print(f"     fill_chunk (single): {improvement:.2f}x FASTER than get_chunk+bytes ({bw4:.2f} → {bw6:.2f} GB/s)")
    if bw7 > 0 and bw4 > 0:
        improvement = bw7 / bw4
        print(f"     fill_chunk (pool):   {improvement:.2f}x FASTER than get_chunk+bytes ({bw4:.2f} → {bw7:.2f} GB/s)")
    if bw1 > 0 and bw6 > 0:
        compare = bw6 / bw1  
        print(f"     fill_chunk matches no_copy: {compare:.2f}x ({bw1:.2f} vs {bw6:.2f} GB/s) - SAME METHOD!")
    
    # Determine winner
    best_approach = max(results.items(), key=lambda x: x[1]['throughput'])
    print(f"\n  🏆 WINNER for {chunk_size_mb}MB: {best_approach[0]} @ {best_approach[1]['throughput']:.2f} GB/s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark dgen_py data generation approaches')
    parser.add_argument('--count', type=int, default=100,
                        help='Number of objects to generate per test (default: 100)')
    parser.add_argument('--sizes', type=str, default='1,8,16,32',
                        help='Comma-separated object sizes in MB (default: 1,8,16,32)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 100 objects, all sizes')
    parser.add_argument('--full', action='store_true',
                        help='Full test: 1000 objects, all sizes')
    
    args = parser.parse_args()
    
    # Handle presets
    if args.quick:
        num_objects = 100
    elif args.full:
        num_objects = 1000
    else:
        num_objects = args.count
    
    # Parse sizes
    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    
    print(f"\n{'#'*80}")
    print(f"# Data Generation Benchmark V2 - Finding Optimal Approach")
    print(f"{'#'*80}")
    print(f"Testing {num_objects:,} objects per size")
    print(f"Object sizes: {sizes} MB")
    print(f"dgen_py version: {dgen_py.__version__ if hasattr(dgen_py, '__version__') else 'unknown'}")
    print(f"\nV1 Approaches (baseline):")
    print(f"  1. No Copy - fill_chunk() reuse bytearray (fastest, requires immediate consumption)")
    print(f"  2. With Copy - fill_chunk() + bytes() copy (safer for queues, has overhead)")
    print(f"  3. Large Split - 32MB chunks split (only for <32MB objects)")
    print(f"  4. BytesView Single Producer - get_chunk() + bytes(), ONE producer")
    print(f"  5. BytesView Multi Producer - get_chunk() + bytes(), FOUR producers")
    print(f"")
    print(f"V2 Approaches (NEW - testing fill_chunk buffer strategies):")
    print(f"  6. fill_chunk() Single Buffer - Reuse ONE buffer (lowest memory: {sizes[0] if sizes else 16}MB)")
    print(f"  7. fill_chunk() Buffer Pool - Pool of 64 buffers (queue pattern: ~1GB for 16MB objects)")
    
    # Run tests for each size
    all_results = {}
    for size_mb in sizes:
        all_results[size_mb] = run_size_test(num_objects, size_mb)
    
    # Print summary
    print(f"\n\n{'='*80}")
    print(f"SUMMARY - Best approach for each object size")
    print(f"{'='*80}")
    
    for size_mb in sizes:
        results = all_results[size_mb]
        best = max(results.items(), key=lambda x: x[1]['throughput'])
        print(f"  {size_mb:2d} MB: {best[0]:15s} @ {best[1]['throughput']:6.2f} GB/s")
    
    # Overall recommendations
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS FOR BENCHMARK_STANDALONE_5K_V7.PY")
    print(f"{'='*80}")
    
    # Check if no-copy is consistently fastest
    no_copy_wins = sum(1 for size_mb in sizes 
                       if max(all_results[size_mb].items(), key=lambda x: x[1]['throughput'])[0] == 'no_copy')
    
    if no_copy_wins == len(sizes):
        print(f"  ✓ NO COPY approach wins for ALL tested sizes")
        print(f"    → Recommendation: Use bytearray buffer without bytes() copy")
        print(f"    → Pattern: buffer = bytearray(size); gen.fill_chunk(buffer); use buffer directly")
        print(f"    ⚠️  CRITICAL: Must consume buffer BEFORE next fill_chunk() call")
        print(f"    ⚠️  For queues: Queue must handle bytearray OR ensure immediate consumption")
    elif no_copy_wins > len(sizes) // 2:
        print(f"  ⚠️  NO COPY wins for MOST sizes ({no_copy_wins}/{len(sizes)})")
        print(f"    → Consider using no-copy if queue can handle bytearray")
        print(f"    → Fall back to with-copy if queue safety is critical")
    else:
        print(f"  ℹ️  Mixed results - check per-size recommendations above")
    
    # Check copy overhead
    avg_copy_overhead = []
    for size_mb in sizes:
        if 'no_copy' in all_results[size_mb] and 'with_copy' in all_results[size_mb]:
            bw1 = all_results[size_mb]['no_copy']['throughput']
            bw2 = all_results[size_mb]['with_copy']['throughput']
            overhead = ((bw1 - bw2) / bw1) * 100 if bw1 > 0 else 0
            avg_copy_overhead.append(overhead)
    
    if avg_copy_overhead:
        avg = sum(avg_copy_overhead) / len(avg_copy_overhead)
        print(f"\n  📊 Average bytes() copy overhead: {avg:.1f}% slower")
        if avg > 50:
            print(f"    → CRITICAL overhead - MUST use no-copy approach")
        elif avg > 20:
            print(f"    → SIGNIFICANT overhead - strongly prefer no-copy approach")
        elif avg > 10:
            print(f"    → Moderate overhead - prefer no-copy where practical")
        else:
            print(f"    → Minimal overhead - either approach acceptable")
    
    # Analyze single vs multi producer (KEY FINDING for v7 optimization)
    print(f"\n{'='*80}")
    print(f"PRODUCER PARALLELISM ANALYSIS (Single vs Multi Producer)")
    print(f"{'='*80}")
    
    single_wins = 0
    multi_wins = 0
    avg_single_advantage = []
    
    for size_mb in sizes:
        if 'bytesview_single' in all_results[size_mb] and 'bytesview_multi' in all_results[size_mb]:
            bw_single = all_results[size_mb]['bytesview_single']['throughput']
            bw_multi = all_results[size_mb]['bytesview_multi']['throughput']
            ratio = bw_single / bw_multi if bw_multi > 0 else 0
            
            if ratio > 1.0:
                single_wins += 1
                advantage = ((ratio - 1.0) * 100)
                avg_single_advantage.append(advantage)
                print(f"  {size_mb:2d} MB: Single producer {ratio:.2f}x faster ({bw_multi:.2f} → {bw_single:.2f} GB/s, +{advantage:.1f}%)")
            else:
                multi_wins += 1
                advantage = ((1.0/ratio - 1.0) * 100)
                print(f"  {size_mb:2d} MB: Multi producer {1/ratio:.2f}x faster ({bw_single:.2f} → {bw_multi:.2f} GB/s, +{advantage:.1f}%)")
    
    if single_wins == len(sizes):
        avg_adv = sum(avg_single_advantage) / len(avg_single_advantage) if avg_single_advantage else 0
        print(f"\n  ✅ SINGLE producer wins for ALL sizes (avg +{avg_adv:.1f}%)")
        print(f"     → RECOMMENDATION: Use 1 producer with max_threads=None")
        print(f"     → Let dgen-py's Rayon pool handle ALL parallelism")
        print(f"     → Avoids thread coordination overhead")
        print(f"     → Simpler architecture, better performance")
    elif multi_wins == len(sizes):
        print(f"\n  ⚠️  MULTI producer wins for ALL sizes")
        print(f"     → Keep current 4-producer approach")
        print(f"     → Benefits outweigh coordination overhead")
    else:
        print(f"\n  ℹ️  Mixed results: {single_wins} single wins, {multi_wins} multi wins")
        print(f"     → Size-dependent optimization may be needed")
    
    # V2 KEY ANALYSIS: fill_chunk() buffer approaches vs get_chunk()+bytes()
    print(f"\n{'='*80}")
    print(f"V2 CRITICAL FINDING: fill_chunk() BUFFER APPROACHES")
    print(f"{'='*80}")
    print(f"Problem: get_chunk() + bytes() conversion creates bottleneck")
    print(f"Solution: Use fill_chunk() with buffer reuse (no bytes() conversion)")
    print(f"")
    
    for size_mb in sizes:
        if 'bytesview_single' in all_results[size_mb] and 'fill_single' in all_results[size_mb]:
            bw_getchunk = all_results[size_mb]['bytesview_single']['throughput']
            bw_fill_single = all_results[size_mb]['fill_single']['throughput']
            bw_fill_pool = all_results[size_mb].get('fill_pool', {}).get('throughput', 0)
            
            if bw_getchunk > 0 and bw_fill_single > 0:
                improvement_single = bw_fill_single / bw_getchunk
                print(f"  {size_mb:2d} MB: fill_chunk(single) {improvement_single:.2f}x faster than get_chunk+bytes")
                print(f"         ({bw_getchunk:.2f} GB/s → {bw_fill_single:.2f} GB/s)")
                
                if bw_fill_pool > 0:
                    improvement_pool = bw_fill_pool / bw_getchunk  
                    print(f"         fill_chunk(pool)   {improvement_pool:.2f}x faster than get_chunk+bytes")
                    print(f"         ({bw_getchunk:.2f} GB/s → {bw_fill_pool:.2f} GB/s)")
                print()
    
    print(f"  🎯 RECOMMENDATION for benchmark_standalone_5k_v7.py:")
    print(f"     ❌ REMOVE: get_chunk() + bytes() conversion (SLOW: ~1.55 GB/s)")
    print(f"     ✅ USE: fill_chunk() with buffer pool (FAST: ~23-37 GB/s)")
    print(f"     ✅ Memory: 64-buffer pool = 1GB for 16MB objects (acceptable)")
    print(f"     ✅ Pattern: producer fills buffers → queue → consumer uploads → return to pool")
    print(f"     ✅ Expected: PUT throughput 1.45 GB/s → 5-6 GB/s (closer to s3-cli 6.5 GB/s)")
    
    # Check against target PUT performance
    print(f"\n{'='*80}")
    print(f"TARGET PUT PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    target_put_gbps = 6.5  # Based on s3-cli results
    print(f"Target PUT performance: {target_put_gbps} GB/s (s3-cli on FAST)")
    print(f"\nData generation throughput by size:")
    
    for size_mb in sizes:
        best = max(all_results[size_mb].items(), key=lambda x: x[1]['throughput'])
        bw = best[1]['throughput']
        ratio = bw / target_put_gbps
        status = "✅" if ratio >= 2.0 else "⚠️" if ratio >= 1.5 else "❌"
        print(f"  {status} {size_mb:2d} MB: {bw:6.2f} GB/s ({ratio:.1f}x target)")
    
    print(f"\n{'='*80}")
    print(f"✓ Benchmark complete")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
