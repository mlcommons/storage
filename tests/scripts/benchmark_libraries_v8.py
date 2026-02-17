#!/usr/bin/env python3
"""
Library Performance Benchmark - S3 library comparison (s3dlio, minio, s3torch).
No MLPerf or DLIO dependencies. Pure storage library comparison.

ASYNC PRODUCER/CONSUMER PATTERN:
- Single producer task: Generate data into queue using buffer pool (NOT in I/O timing)
- Multiple consumer tasks: Pull data from queue and upload (MEASURED)
- Uses asyncio for better concurrency without GIL

This separates data generation overhead from network I/O measurement.

KEY OPTIMIZATION IN v8 (CRITICAL BREAKTHROUGH):
- PROBLEM: v7 used get_chunk() + bytes() conversion → 1.45 GB/s (BOTTLENECK!)
- SOLUTION: Use fill_chunk() with buffer pool → 24.74 GB/s (17x faster!)
- Buffer pool: 64 reusable bytearray buffers (1GB RAM for 16MB objects)
- Libraries accept bytearray via buffer protocol (s3dlio, minio)
- Convert to bytes() only for s3torch (requires actual bytes)

BENCHMARK PROOF (benchmark_datagen_v2.py results):
- get_chunk() + bytes(): 1.45 GB/s ← Limited ALL libraries to 1.45-1.71 GB/s PUT
- fill_chunk() buffer pool: 24.74 GB/s ← Should unlock 5-6 GB/s PUT (s3-cli baseline)
- Memory: 64 buffers × 16MB = 1024MB (acceptable)

Other v7 features retained:
- Clear all objects from bucket before each test (ensure clean state)
- 30 second pause after bucket clearing (allow storage to settle)
- 60 second pause between PUT and GET phases (prevent interference)
- Configurable delays via --quick flag
- Configurable object size via --object-size parameter

Usage:
    # Set credentials in environment:
    export ACCESS_KEY_ID="your-access-key"
    export SECRET_ACCESS_KEY="your-secret-key"
    export ENDPOINT_URL="http://your-endpoint:9000"
    
    # Then run benchmarks:
    python3 benchmark_libraries_v8.py --target default --threads 16
    python3 benchmark_libraries_v8.py --target default --num-objects 3000 --quick
    python3 benchmark_libraries_v8.py --target default --threads 16 --libraries s3dlio
    
    # Alternatively, use custom endpoint (bypass environment):
    python3 benchmark_libraries_v8.py --endpoint http://10.9.0.21 --access-key KEY --secret-key SECRET --bucket mybucket --threads 16
"""

import argparse
import time
import sys
import os
import asyncio
import threading
from io import BytesIO
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

# Test configuration defaults (can be overridden by command line args)
DEFAULT_NUM_OBJECTS = 5000
DEFAULT_OBJECT_SIZE_MB = 16
OBJECT_SIZE_MB = DEFAULT_OBJECT_SIZE_MB
OBJECT_SIZE_BYTES = OBJECT_SIZE_MB * 1024 * 1024
DEFAULT_NUM_THREADS = 16

# Producer/Consumer queue size (buffer at most 64 objects ahead of uploads)
QUEUE_SIZE = 64

# Will be set by main() based on command line args or defaults
NUM_OBJECTS = DEFAULT_NUM_OBJECTS
TOTAL_SIZE_GB = (NUM_OBJECTS * OBJECT_SIZE_MB) / 1024.0
NUM_THREADS = DEFAULT_NUM_THREADS

# S3 credentials from environment variables
# Prefer generic (ACCESS_KEY_ID) over AWS_* if both exist
def get_env_credentials():
    """
    Get S3 credentials from environment variables.
    Prefers generic names (ACCESS_KEY_ID) over AWS_* prefixed versions.
    Returns: (access_key, secret_key, endpoint_url)
    """
    # Access Key: Prefer ACCESS_KEY_ID over AWS_ACCESS_KEY_ID
    access_key = os.environ.get('ACCESS_KEY_ID')
    if access_key:
        print("Using ACCESS_KEY_ID from environment")
    else:
        access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        if access_key:
            print("Using AWS_ACCESS_KEY_ID from environment")
        else:
            raise ValueError("ERROR: Neither ACCESS_KEY_ID nor AWS_ACCESS_KEY_ID is set in environment")
    
    # Secret Key: Prefer SECRET_ACCESS_KEY over AWS_SECRET_ACCESS_KEY
    secret_key = os.environ.get('SECRET_ACCESS_KEY')
    if secret_key:
        print("Using SECRET_ACCESS_KEY from environment")
    else:
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        if secret_key:
            print("Using AWS_SECRET_ACCESS_KEY from environment")
        else:
            raise ValueError("ERROR: Neither SECRET_ACCESS_KEY nor AWS_SECRET_ACCESS_KEY is set in environment")
    
    # Endpoint URL: Prefer ENDPOINT_URL over AWS_ENDPOINT_URL
    endpoint_url = os.environ.get('ENDPOINT_URL')
    if endpoint_url:
        print("Using ENDPOINT_URL from environment")
    else:
        endpoint_url = os.environ.get('AWS_ENDPOINT_URL')
        if endpoint_url:
            print("Using AWS_ENDPOINT_URL from environment")
        else:
            raise ValueError("ERROR: Neither ENDPOINT_URL nor AWS_ENDPOINT_URL is set in environment")
    
    return access_key, secret_key, endpoint_url

# Get credentials from environment
ACCESS_KEY, SECRET_KEY, ENDPOINT_URL = get_env_credentials()

# S3 Target configuration (using environment credentials)
# Note: This script previously had hardcoded 'minio' and 'fast' presets.
# Now it uses a single 'default' target with credentials from environment.
S3_TARGETS = {
    'default': {
        'name': 'S3 Target (from environment)',
        'endpoint': ENDPOINT_URL,
        'access_key': ACCESS_KEY,
        'secret_key': SECRET_KEY,
        'bucket_minio': 'bucket-minio',
        'bucket_s3torch': 'bucket-s3torch',
        'bucket_s3dlio': 'bucket-s3dlio',
        'region': 'us-east-1'
    }
}

# Try to import dgen_py for efficient data generation
try:
    import dgen_py
    HAS_DGEN = True
except ImportError:
    HAS_DGEN = False
    print("WARNING: dgen_py not available. Will use os.urandom() for data generation (slower).")


async def countdown_sleep(seconds: int, reason: str, quick: bool = False):
    """
    Sleep for specified seconds while displaying countdown timer.
    
    Args:
        seconds: Number of seconds to sleep
        reason: Description of why we're sleeping (e.g., "after bucket clear")
        quick: If True, skip the sleep (for quick testing/debugging)
    """
    if quick:
        print(f"⚡ Skipping {seconds}s delay {reason} (--quick mode)")
        return
    
    print(f"\n⏳ Pausing {seconds} seconds {reason}...")
    for i in range(seconds, 0, -1):
        if i == seconds or i % 10 == 0 or i <= 5:
            print(f"   {i} seconds remaining...", flush=True)
        await asyncio.sleep(1)
    print(f"✓ Pause complete\n")


class DataProducer:
    """
    Generates data chunks into queue using fill_chunk() with buffer pool (V8 OPTIMIZATION).
    
    CRITICAL BREAKTHROUGH (from benchmark_datagen_v2.py):
    - V7 PROBLEM: get_chunk() + bytes() conversion = 1.45 GB/s (BOTTLENECK!)
    - V8 SOLUTION: fill_chunk() buffer pool = 24.74 GB/s (17x faster!)
    
    Architecture:
    - Pre-allocate pool of 64 bytearray buffers (matches QUEUE_SIZE)
    - Use fill_chunk() to fill buffers (NO bytes() conversion overhead)
    - Cycle through buffer pool as objects are queued
    - Memory: 64 × 16MB = 1024MB for 16MB objects (acceptable)
    
    Performance impact:
    - V7: Limited all libraries to 1.45-1.71 GB/s PUT (data gen bottleneck)
    - V8: Should unlock 5-6 GB/s PUT (matching s3-cli Rust baseline)
    
    Benchmark results (benchmark_datagen_v2.py, 100×16MB):
    - get_chunk() + bytes(): 1.45 GB/s ← OLD (v7)
    - fill_chunk() buffer pool: 24.74 GB/s ← NEW (v8, 17x faster)
    """
    
    def __init__(self, num_objects, chunk_size, queue_ref, pool_size=64):
        self.num_objects = num_objects
        self.chunk_size = chunk_size
        self.queue = queue_ref
        self.pool_size = pool_size
        # Pre-allocate buffer pool (constant memory)
        self.buffer_pool = [bytearray(chunk_size) for _ in range(pool_size)]
    
    async def producer_worker(self, loop, executor):
        """
        Single producer using fill_chunk() with buffer pool (V8 OPTIMIZATION).
        
        KEY CHANGE FROM V7:
        - V7: get_chunk() + bytes() conversion = 1.45 GB/s (BOTTLENECK)
        - V8: fill_chunk() buffer pool = 24.74 GB/s (17x faster)
        
        How it works:
        - Pre-allocated buffer pool (64 buffers)
        - Cycle through buffers using fill_chunk() (fast: 24.74 GB/s)
        - Pass bytearray directly to queue (no conversion for s3dlio/minio)
        - Consumer handles conversion to bytes if needed (s3torch only)
        """
        if HAS_DGEN:
            # Single generator for entire dataset - dgen-py parallelizes internally
            total_size = self.num_objects * self.chunk_size
            generator = dgen_py.Generator(
                size=total_size,
                dedup_ratio=1.0,
                compress_ratio=1.0,
                numa_mode="auto",
                max_threads=None,  # Let dgen-py use all cores
                seed=12345
            )
        
        for obj_id in range(self.num_objects):
            # Get buffer from pool (cycle through)
            buffer_idx = obj_id % self.pool_size
            buffer = self.buffer_pool[buffer_idx]
            
            # Fill buffer using fill_chunk() (CPU-bound, run in executor)
            def fill_buffer():
                if HAS_DGEN:
                    # fill_chunk() fills buffer in-place (FAST: 24.74 GB/s)
                    # No bytes() conversion overhead (17x faster than get_chunk+bytes)
                    nbytes = generator.fill_chunk(buffer)
                    return nbytes
                else:
                    # Fallback should never be used
                    fallback_data = os.urandom(self.chunk_size)
                    buffer[:] = fallback_data
                    return len(fallback_data)
            
            # Run fill_chunk in executor (allows async coordination)
            nbytes = await loop.run_in_executor(executor, fill_buffer)
            
            if nbytes == 0:
                print(f"  WARNING: Generator exhausted at object {obj_id}")
                break
            
            # DEBUG: Check what type we're putting in queue
            if obj_id == 0:
                print(f"  DEBUG: data type = bytearray, len = {len(buffer)}")
            
            # Put bytearray into queue for consumers
            # s3dlio and minio accept bytearray via buffer protocol
            # s3torch adapter will convert to bytes() if needed
            await self.queue.put((obj_id, buffer))
    
    async def run(self, executor=None):
        """Start single producer task (optimal based on benchmarks)"""
        if executor is None:
            # Single worker for producer - dgen-py parallelizes internally
            executor = ThreadPoolExecutor(max_workers=1)
        
        loop = asyncio.get_event_loop()
        
        # Run single producer - simpler and faster than multiple producers
        await self.producer_worker(loop, executor)


class S3LibraryAdapter(ABC):
    """Abstract base class for S3 library adapters"""
    
    def __init__(self, num_threads=4, endpoint_url=None, access_key=None, secret_key=None):
        """Initialize adapter - subclasses should call super().__init__()
        
        Args:
            num_threads: Number of executor threads (default: 4)
            endpoint_url: S3 endpoint URL (for bucket clearing)
            access_key: AWS access key (for bucket clearing)
            secret_key: AWS secret key (for bucket clearing)
        """
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.loop = None
        # Store credentials for bucket clearing (uses s3dlio)
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
    
    def set_loop(self, loop):
        """Set the event loop for executor operations"""
        self.loop = loop
    
    @abstractmethod
    def get_library_name(self):
        """Return the library name for display"""
        pass
    
    @abstractmethod
    def _setup_bucket_sync(self, bucket_name):
        """Synchronous bucket setup (runs in executor)"""
        pass
    
    async def setup_bucket(self, bucket_name):
        """Create/verify bucket exists (async wrapper)"""
        if self.loop is None:
            self.loop = asyncio.get_event_loop()
        await self.loop.run_in_executor(self.executor, self._setup_bucket_sync, bucket_name)
    
    @abstractmethod
    def _upload_object_sync(self, bucket_name, key, data):
        """Synchronous upload (runs in executor)"""
        pass
    
    async def upload_object(self, bucket_name, key, data):
        """Upload data to S3 (async wrapper)"""
        if self.loop is None:
            self.loop = asyncio.get_event_loop()
        await self.loop.run_in_executor(
            self.executor,
            self._upload_object_sync,
            bucket_name,
            key,
            data
        )
    
    @abstractmethod
    def _download_object_sync(self, bucket_name, key):
        """Synchronous download (runs in executor)"""
        pass
    
    async def download_object(self, bucket_name, key):
        """Download and return object data (async wrapper)"""
        if self.loop is None:
            self.loop = asyncio.get_event_loop()
        return await self.loop.run_in_executor(
            self.executor,
            self._download_object_sync,
            bucket_name,
            key
        )
    
    @abstractmethod
    def get_object_key_prefix(self):
        """Return the prefix to use for object keys (e.g., 'minio_object_')"""
        pass
    
    async def download_many(self, bucket_name, key_prefix, num_objects):
        """
        Optional: Override for libraries with built-in batch download.
        Returns list of (success, bytes_read) tuples.
        Default: returns None (use individual downloads).
        """
        return None
    
    def _clear_bucket_sync(self, bucket_name, key_prefix):
        """
        Clear ALL objects from bucket using s3-cli command line tool.
        This is more reliable than s3dlio library calls for bulk deletion.
        """
        try:
            import subprocess
            
            # Set environment variables for s3-cli
            env = os.environ.copy()
            if self.endpoint_url and self.access_key and self.secret_key:
                env['AWS_ACCESS_KEY_ID'] = self.access_key
                env['AWS_SECRET_ACCESS_KEY'] = self.secret_key
                env['AWS_ENDPOINT_URL'] = self.endpoint_url
                env['AWS_REGION'] = 'us-east-1'
            
            uri = f"s3://{bucket_name}/"
            
            # First count objects
            print(f"  Counting objects in bucket: {uri}")
            count_cmd = ['s3-cli', 'list', '-cr', uri]
            result = subprocess.run(count_cmd, env=env, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"  Warning: Could not list objects: {result.stderr}")
                return 0
            
            # Parse count from output (format: "Total objects: 2000 (0.091s, rate: 21,984 objects/s)")
            count = 0
            for line in result.stdout.split('\n'):
                if 'Total objects:' in line:
                    count = int(line.split('Total objects:')[1].split()[0])
                    break
            
            print(f"  Found {count} objects to delete")
            
            if count > 0:
                # Delete all objects with s3-cli
                print(f"  Deleting {count} objects with s3-cli...")
                delete_cmd = ['s3-cli', 'delete', '-r', uri]
                result = subprocess.run(delete_cmd, env=env, capture_output=True, text=True, timeout=120)
                
                if result.returncode != 0:
                    print(f"  Warning: Delete failed: {result.stderr}")
                    return 0
                
                print(f"  ✓ Deleted {count} objects")
            
            return count
        except subprocess.TimeoutExpired:
            print(f"  Warning: Command timed out")
            return 0
        except Exception as e:
            print(f"  Warning: Could not clear bucket: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    async def clear_bucket(self, bucket_name, key_prefix):
        """Clear all objects with given prefix (async wrapper)"""
        if self.loop is None:
            self.loop = asyncio.get_event_loop()
        return await self.loop.run_in_executor(
            self.executor,
            self._clear_bucket_sync,
            bucket_name,
            key_prefix
        )


class MinioAdapter(S3LibraryAdapter):
    """Adapter for minio library"""
    
    def __init__(self, endpoint_url, access_key, secret_key, num_threads=4):
        super().__init__(num_threads, endpoint_url, access_key, secret_key)
        from minio import Minio
        
        # Parse endpoint URL
        if endpoint_url.startswith("https://"):
            endpoint = endpoint_url[8:]
            secure = True
        elif endpoint_url.startswith("http://"):
            endpoint = endpoint_url[7:]
            secure = False
        else:
            endpoint = endpoint_url
            secure = False
        
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
    
    def get_library_name(self):
        return "minio"
    
    def _setup_bucket_sync(self, bucket_name):
        try:
            self.client.make_bucket(bucket_name)
            print(f"  Created bucket: {bucket_name}")
        except Exception as e:
            err_msg = str(e).lower()
            if any(x in err_msg for x in ["exist", "already", "owned"]):
                print(f"  Bucket already exists: {bucket_name}")
            else:
                raise
        
        # Verify bucket is accessible
        _ = self.client.list_objects(bucket_name)
        print(f"  Bucket is accessible")
    
    def _upload_object_sync(self, bucket_name, key, data):
        # minio accepts bytearray via buffer protocol (v8 optimization)
        # BytesIO constructor accepts any bytes-like object
        self.client.put_object(
            bucket_name=bucket_name,
            object_name=key,
            data=BytesIO(data),
            length=len(data)
        )
    
    def _download_object_sync(self, bucket_name, key):
        response = self.client.get_object(bucket_name, key)
        data = response.read()
        response.close()
        return data
    
    def get_object_key_prefix(self):
        return "minio_object_"


class S3TorchConnectorAdapter(S3LibraryAdapter):
    """Adapter for s3torchconnectorclient library"""
    
    def __init__(self, endpoint_url, access_key, secret_key, num_threads=4):
        super().__init__(num_threads, endpoint_url, access_key, secret_key)
        from s3torchconnectorclient._mountpoint_s3_client import MountpointS3Client
        from minio import Minio
        
        # Set credentials via environment
        os.environ['AWS_ACCESS_KEY_ID'] = access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
        os.environ['AWS_ENDPOINT_URL'] = endpoint_url
        os.environ['AWS_REGION'] = 'us-east-1'
        
        self.client = MountpointS3Client(
            region="us-east-1",
            endpoint=endpoint_url,
            throughput_target_gbps=10.0,
            part_size=32 * 1024**2
        )
        
        # Keep minio client for bucket management
        self.minio_client = Minio(
            endpoint_url.replace('http://', '').replace('https://', ''),
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )
    
    def get_library_name(self):
        return "s3torchconnectorclient"
    
    def _setup_bucket_sync(self, bucket_name):
        try:
            self.minio_client.make_bucket(bucket_name)
            print(f"  Created bucket: {bucket_name}")
        except Exception as e:
            err_msg = str(e).lower()
            if any(x in err_msg for x in ["exist", "already", "owned"]):
                print(f"  Bucket already exists: {bucket_name}")
            else:
                raise
        
        # Verify bucket is accessible
        _ = self.minio_client.list_objects(bucket_name)
        print(f"  Bucket is accessible")
    
    def _upload_object_sync(self, bucket_name, key, data):
        # s3torch requires actual bytes, not bytearray
        # Convert if necessary (v8 buffer pool passes bytearray)
        if isinstance(data, bytearray):
            data = bytes(data)
        
        stream = self.client.put_object(bucket=bucket_name, key=key)
        stream.write(data)
        stream.close()
    
    def _download_object_sync(self, bucket_name, key):
        stream = self.client.get_object(bucket=bucket_name, key=key)
        # GetObjectStream is an iterator, consume all chunks
        return b''.join(chunk for chunk in stream)
    
    def get_object_key_prefix(self):
        return "s3tc_object_"


class S3DlioAdapter(S3LibraryAdapter):
    """Adapter for s3dlio library - uses native async functions for optimal performance"""
    
    def __init__(self, endpoint_url, access_key, secret_key, num_threads=4):
        super().__init__(num_threads, endpoint_url, access_key, secret_key)
        import s3dlio
        self.s3dlio = s3dlio
        
        # Set up environment for s3dlio
        os.environ['AWS_ACCESS_KEY_ID'] = access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
        os.environ['AWS_ENDPOINT_URL'] = endpoint_url
        os.environ['AWS_REGION'] = 'us-east-1'
        
        # Phase 1a: Disable range splitting for small/medium objects (16MB training samples)
        # This avoids HEAD + multiple range requests overhead for objects < 256MB
        os.environ['S3DLIO_RANGE_THRESHOLD_MB'] = '256'
    
    def get_library_name(self):
        return "s3dlio"
    
    def _setup_bucket_sync(self, bucket_name):
        try:
            self.s3dlio.create_bucket(bucket_name)
            print(f"  Created/verified bucket: {bucket_name}")
        except Exception as e:
            print(f"  Note: create_bucket returned: {e}")
            print(f"  Proceeding (bucket may already exist)")
    
    def _upload_object_sync(self, bucket_name, key, data):
        """Sync wrapper - not used (we override with async)"""
        uri = f"s3://{bucket_name}/{key}"
        self.s3dlio.put_bytes(uri, data)
    
    async def upload_object(self, bucket_name, key, data):
        """Override to use async put_bytes_async instead of executor
        
        V8 OPTIMIZATION: Accepts bytearray from buffer pool
        - s3dlio supports buffer protocol (4-tier fallback already implemented)
        - No bytes() conversion overhead (17x speedup vs v7)
        """
        uri = f"s3://{bucket_name}/{key}"
        await self.s3dlio.put_bytes_async(uri, data)
    
    def _download_object_sync(self, bucket_name, key):
        """Sync download using s3dlio.get() - runs in executor with throttling
        
        Phase 1b/1d: Use sync get() (releases GIL, runs on Tokio runtime internally)
        with executor throttling (16 threads instead of 4). Remove bytes() copy.
        
        Note: There's no get_async(uri) in s3dlio yet, only get_many_async() for batches.
        An async override would need semaphore throttling to prevent OOM from 2000 
        concurrent tasks. This will be addressed in Phase 2.
        """
        uri = f"s3://{bucket_name}/{key}"
        data = self.s3dlio.get(uri)
        # Return BytesView directly (implements buffer protocol) - no copy needed
        return data
    
    def get_object_key_prefix(self):
        return "s3dlio_object_"


async def run_library_benchmark(adapter, bucket_name, put_threads, get_threads, quick=False):
    """
    Generic benchmark function that works with any S3 library adapter.
    Eliminates code duplication across library-specific tests.
    Uses asyncio for concurrent producer/consumer operations.
    
    Args:
        adapter: S3 library adapter instance
        bucket_name: Name of the bucket to use
        put_threads: Number of concurrent upload workers
        get_threads: Number of concurrent download workers
        quick: Skip delays if True
    """
    library_name = adapter.get_library_name()
    
    print("\n" + "="*70)
    print(f"Testing: {library_name}")
    print("="*70)
    
    # Setup bucket
    print(f"\nVerifying bucket '{bucket_name}'...")
    try:
        await adapter.setup_bucket(bucket_name)
    except Exception as e:
        print(f"ERROR: Could not verify bucket: {e}")
        return None
    
    # v6: Clear all existing objects from bucket
    print(f"\n🗑  Clearing all objects from bucket with prefix '{adapter.get_object_key_prefix()}'...")
    cleared = await adapter.clear_bucket(bucket_name, adapter.get_object_key_prefix())
    if cleared > 0:
        print(f"  Removed {cleared} existing objects")
    else:
        print(f"  Bucket is empty or clear skipped")
    
    # v6: Pause after clearing to let storage settle
    await countdown_sleep(30, "after bucket clear (allow storage to settle)", quick)
    
    # Create asyncio queue for producer/consumer
    data_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    # V8: Buffer pool size matches QUEUE_SIZE for efficient cycling
    producer = DataProducer(NUM_OBJECTS, OBJECT_SIZE_BYTES, data_queue, pool_size=QUEUE_SIZE)
    
    # START PRODUCER (NOT TIMED)
    print(f"\nStarting producer task group to generate {NUM_OBJECTS} objects...")
    producer_task = asyncio.create_task(producer.run())
    
    # Give producer a head start to buffer some data
    await asyncio.sleep(0.1)
    
    # Phase 1: PUT - Upload objects from queue
    print(f"Phase 1: Uploading {NUM_OBJECTS} objects ({TOTAL_SIZE_GB:.1f} GB)...")
    
    completed = [0]
    put_errors = [0]
    completed_lock = asyncio.Lock()
    key_prefix = adapter.get_object_key_prefix()
    
    async def upload_from_queue(thread_id):
        """Consumer: Upload objects pulled from queue"""
        while True:
            try:
                item = await asyncio.wait_for(data_queue.get(), timeout=300)
            except asyncio.TimeoutError:
                break
            
            if item is None:
                break
            
            obj_id, data = item
            key = f"{key_prefix}{obj_id:05d}.dat"
            
            # DEBUG: Check type before upload
            if obj_id == 0:
                print(f"  DEBUG: Uploading object 0 - data type = {type(data).__name__}, len = {len(data) if hasattr(data, '__len__') else 'N/A'}")
            
            try:
                await adapter.upload_object(bucket_name, key, data)
            except Exception as e:
                print(f"  ERROR uploading {key}: {e}")
                async with completed_lock:
                    put_errors[0] += 1
                continue
            
            # Progress update
            async with completed_lock:
                completed[0] += 1
                if completed[0] % 500 == 0:
                    pct = (completed[0] / NUM_OBJECTS) * 100
                    print(f"  Progress: {completed[0]}/{NUM_OBJECTS} ({pct:.1f}%)")
    
    # START I/O TIMING
    put_start = time.perf_counter()
    
    # Create upload consumer tasks
    upload_tasks = [
        asyncio.create_task(upload_from_queue(i))
        for i in range(put_threads)
    ]
    
    # Wait for producer to finish
    await producer_task
    
    # Signal end of stream (one None sentinel per consumer task)
    for _ in range(put_threads):
        await data_queue.put(None)
    
    # Wait for all uploads to complete
    await asyncio.gather(*upload_tasks)
    put_time = time.perf_counter() - put_start
    # END I/O TIMING
    
    put_success = NUM_OBJECTS - put_errors[0]
    put_bytes = put_success * OBJECT_SIZE_BYTES
    put_throughput = (put_bytes / (1024**3)) / put_time if put_time > 0 else 0
    
    print(f"✓ PUT completed: {put_success}/{NUM_OBJECTS} objects in {put_time:.2f}s")
    print(f"  Throughput: {put_throughput:.2f} GB/s")
    
    # v6: Pause between PUT and GET to prevent interference
    await countdown_sleep(60, "between PUT and GET phases (prevent interference)", quick)
    
    # Phase 2: GET - Download ALL objects
    print(f"\nPhase 2: Downloading {NUM_OBJECTS} objects...")
    
    completed[0] = 0
    get_errors = [0]
    
    async def download_object(obj_id):
        """Download and discard a single object"""
        key = f"{key_prefix}{obj_id:05d}.dat"
        
        try:
            data = await adapter.download_object(bucket_name, key)
            bytes_read = len(data)
        except Exception as e:
            print(f"  ERROR downloading {key}: {e}")
            async with completed_lock:
                get_errors[0] += 1
            return (0, 0)
        
        # Progress update
        async with completed_lock:
            completed[0] += 1
            if completed[0] % 500 == 0:
                pct = (completed[0] / NUM_OBJECTS) * 100
                print(f"  Progress: {completed[0]}/{NUM_OBJECTS} ({pct:.1f}%)")
        
        return (1, bytes_read)
    
    get_start = time.perf_counter()
    
    # Create download tasks with concurrency limit based on get_threads
    # Use semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(get_threads)
    
    async def download_with_semaphore(obj_id):
        async with semaphore:
            return await download_object(obj_id)
    
    download_tasks = [
        asyncio.create_task(download_with_semaphore(obj_id))
        for obj_id in range(NUM_OBJECTS)
    ]
    
    # Wait for all downloads to complete
    get_results = await asyncio.gather(*download_tasks, return_exceptions=False)
    get_time = time.perf_counter() - get_start
    
    get_success = sum(1 for r in get_results if r[0] > 0)
    get_bytes = sum(r[1] for r in get_results if r[0] > 0)
    get_throughput = (get_bytes / (1024**3)) / get_time if get_time > 0 else 0
    
    print(f"✓ GET completed: {get_success}/{NUM_OBJECTS} objects in {get_time:.2f}s")
    print(f"  Throughput: {get_throughput:.2f} GB/s")
    
    return {
        'library': library_name,
        'put_objects': put_success,
        'put_time': put_time,
        'put_throughput_gbs': put_throughput,
        'get_objects': get_success,
        'get_time': get_time,
        'get_throughput_gbs': get_throughput,
        'total_time': put_time + get_time
    }


async def test_library(library_name, s3_target, bucket_key, put_threads, get_threads, quick=False):
    """
    Test a specific library by creating its adapter and running the generic benchmark.
    """
    # Get config from S3_TARGETS
    s3_config = S3_TARGETS.get(s3_target)
    if not s3_config:
        print(f"ERROR: Unknown S3 target '{s3_target}'")
        return None
    
    endpoint_url = s3_config['endpoint']
    access_key = s3_config['access_key']
    secret_key = s3_config['secret_key']
    bucket_name = s3_config.get(bucket_key)
    
    if not bucket_name:
        print(f"ERROR: Bucket key '{bucket_key}' not found in S3 target config")
        return None
    
    # Create appropriate adapter
    # Use max of put_threads and get_threads for adapter's executor pool size
    max_threads = max(put_threads, get_threads)
    try:
        if library_name == 'minio':
            from minio import Minio
            adapter = MinioAdapter(endpoint_url, access_key, secret_key, max_threads)
        elif library_name == 's3torchconnectorclient':
            from s3torchconnectorclient._mountpoint_s3_client import MountpointS3Client
            adapter = S3TorchConnectorAdapter(endpoint_url, access_key, secret_key, max_threads)
        elif library_name == 's3dlio':
            import s3dlio
            adapter = S3DlioAdapter(endpoint_url, access_key, secret_key, max_threads)
        else:
            print(f"ERROR: Unknown library '{library_name}'")
            return None
    except ImportError as e:
        print(f"SKIP: {library_name} not installed ({e})")
        return None
    except Exception as e:
        print(f"ERROR: Failed to create {library_name} adapter: {e}")
        return None
    
    # Run the benchmark
    return await run_library_benchmark(adapter, bucket_name, put_threads, get_threads, quick)


def print_summary(results, put_threads, get_threads, target_name):
    """Print performance summary"""
    if not results:
        print("\n" + "="*70)
        print("No test results!")
        return
    
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"Target: {target_name}")
    print(f"Configuration: {NUM_OBJECTS} objects × {OBJECT_SIZE_MB} MB = {TOTAL_SIZE_GB:.1f} GB")
    print(f"PUT threads: {put_threads} concurrent upload workers")
    print(f"GET threads: {get_threads} concurrent download workers")
    print(f"Data generation: {'dgen_py' if HAS_DGEN else 'os.urandom'} (single producer, dgen-py max_threads=None, NOT in I/O timing)")
    print()
    
    for result in results:
        if result is None:
            continue
        print(f"\n{result['library'].upper()}")
        print("-" * 70)
        print(f"PUT: {result['put_objects']:,} objects in {result['put_time']:.2f}s")
        print(f"     Throughput: {result['put_throughput_gbs']:.2f} GB/s")
        print(f"GET: {result['get_objects']:,} objects in {result['get_time']:.2f}s")
        print(f"     Throughput: {result['get_throughput_gbs']:.2f} GB/s")
        print(f"Total time: {result['total_time']:.2f}s")


async def main():
    parser = argparse.ArgumentParser(
        description='Standalone S3 library benchmark with asyncio producer/consumer pattern',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set credentials in environment first:
  export ACCESS_KEY_ID="your-access-key"
  export SECRET_ACCESS_KEY="your-secret-key"
  export ENDPOINT_URL="http://your-endpoint:9000"
  
  # Test with default 5000 objects
  python3 benchmark_libraries_v8.py --target default --threads 16

  # Test with 1000 objects (faster for testing)
  python3 benchmark_libraries_v8.py --target default --num-objects 1000 --threads 16

  # Test with only s3dlio library
  python3 benchmark_libraries_v8.py --target default --threads 16 --libraries s3dlio

  # List available targets
  python3 benchmark_libraries_v8.py --list-targets
  
  # Or use custom endpoint (bypass environment variables):
  python3 benchmark_libraries_v8.py --endpoint http://10.9.0.21 --access-key KEY --secret-key SECRET --bucket mybucket --threads 16
        """)
    
    parser.add_argument('--target', choices=list(S3_TARGETS.keys()),
                       help='Predefined S3 target')
    parser.add_argument('--endpoint', help='Custom S3 endpoint URL')
    parser.add_argument('--access-key', help='Access key')
    parser.add_argument('--secret-key', help='Secret key')
    parser.add_argument('--bucket', help='S3 bucket name')
    parser.add_argument('--num-objects', type=int, default=DEFAULT_NUM_OBJECTS,
                       help=f'Number of objects to upload/download (default: {DEFAULT_NUM_OBJECTS})')
    parser.add_argument('--threads', type=int, default=DEFAULT_NUM_THREADS, 
                       help=f'Number of concurrent workers for both PUT and GET (default: {DEFAULT_NUM_THREADS}). Overridden by --put-threads and --get-threads if specified.')
    parser.add_argument('--put-threads', type=int, default=None,
                       help=f'Number of concurrent upload workers (default: use --threads value)')
    parser.add_argument('--get-threads', type=int, default=None,
                       help=f'Number of concurrent download workers (default: use --threads value)')
    parser.add_argument('--object-size', type=int, default=DEFAULT_OBJECT_SIZE_MB,
                       help=f'Object size in MB (default: {DEFAULT_OBJECT_SIZE_MB}). Test 14MB vs 18MB to validate range GET behavior')
    parser.add_argument('--libraries', nargs='+', 
                       default=['s3torchconnectorclient', 'minio', 's3dlio'],
                       choices=['s3torchconnectorclient', 'minio', 's3dlio'],
                       help='Libraries to test')
    parser.add_argument('--quick', action='store_true',
                       help='Skip delays (for quick testing/debugging)')
    parser.add_argument('--list-targets', action='store_true',
                       help='List available S3 targets and exit')
    
    args = parser.parse_args()
    
    # List targets if requested
    if args.list_targets:
        print("Available S3 Targets:")
        print("-" * 50)
        for key, config in S3_TARGETS.items():
            print(f"\n{key}: {config['name']}")
            print(f"  Endpoint: {config['endpoint']}")
            print(f"  Buckets: minio={config.get('bucket_minio')}, s3torch={config.get('bucket_s3torch')}, s3dlio={config.get('bucket_s3dlio')}")
        return
    
    # Determine credentials
    if args.target:
        if args.endpoint or args.access_key or args.secret_key or args.bucket:
            print("ERROR: Cannot use --target with custom endpoint/credentials")
            sys.exit(1)
        s3_target = args.target
        config = S3_TARGETS[args.target]
        target_name = config['name']
    else:
        if not (args.endpoint and args.access_key and args.secret_key and args.bucket):
            print("ERROR: Either use --target OR provide --endpoint, --access-key, --secret-key, and --bucket")
            print("Use --list-targets to see available presets")
            sys.exit(1)
        # Create custom target config
        s3_target = 'custom'
        S3_TARGETS['custom'] = {
            'name': f'Custom ({args.endpoint})',
            'endpoint': args.endpoint,
            'access_key': args.access_key,
            'secret_key': args.secret_key,
            'bucket_minio': args.bucket,
            'bucket_s3torch': args.bucket,
            'bucket_s3dlio': args.bucket
        }
        target_name = S3_TARGETS['custom']['name']
    
    # Validate and apply command line overrides
    if args.num_objects < 1:
        print("ERROR: --num-objects must be >= 1")
        sys.exit(1)
    if args.threads < 1:
        print("ERROR: --threads must be >= 1")
        sys.exit(1)
    
    # Determine PUT and GET thread counts
    put_threads = args.put_threads if args.put_threads is not None else args.threads
    get_threads = args.get_threads if args.get_threads is not None else args.threads
    
    if put_threads < 1:
        print("ERROR: --put-threads must be >= 1")
        sys.exit(1)
    if get_threads < 1:
        print("ERROR: --get-threads must be >= 1")
        sys.exit(1)
    
    # Update global variables based on command line args
    global NUM_OBJECTS, TOTAL_SIZE_GB, NUM_THREADS, OBJECT_SIZE_MB, OBJECT_SIZE_BYTES
    NUM_OBJECTS = args.num_objects
    OBJECT_SIZE_MB = args.object_size
    OBJECT_SIZE_BYTES = OBJECT_SIZE_MB * 1024 * 1024
    TOTAL_SIZE_GB = (NUM_OBJECTS * OBJECT_SIZE_MB) / 1024.0
    NUM_THREADS = args.threads  # Keep for backwards compatibility
    
    print("="*70)
    print("STANDALONE S3 LIBRARY BENCHMARK (Asyncio Producer/Consumer Pattern)")
    print("="*70)
    print(f"Target: {target_name}")
    print(f"Configuration: {NUM_OBJECTS:,} objects × {OBJECT_SIZE_MB} MB")
    print(f"Total size: {TOTAL_SIZE_GB:.1f} GB")
    print(f"PUT tasks: {put_threads} concurrent upload workers")
    print(f"GET tasks: {get_threads} concurrent download workers")
    print(f"Data producer: 1 task with dgen-py Rayon parallelism (NOT in I/O timing)")
    print(f"Concurrency model: asyncio (no GIL limit)")
    print(f"Endpoint: {S3_TARGETS[s3_target]['endpoint']}")
    print(f"Libraries to test: {', '.join(args.libraries)}")
    print()
    
    # Map library names to their bucket keys
    bucket_keys = {
        's3torchconnectorclient': 'bucket_s3torch',
        'minio': 'bucket_minio',
        's3dlio': 'bucket_s3dlio'
    }
    
    results = []
    for idx, library_name in enumerate(args.libraries):
        bucket_key = bucket_keys.get(library_name)
        if bucket_key:
            result = await test_library(library_name, s3_target, bucket_key, put_threads, get_threads, args.quick)
            if result:
                results.append(result)
            
            # v6: Pause between different libraries (except after the last one)
            if idx < len(args.libraries) - 1:
                await countdown_sleep(60, "before next library (test isolation)", args.quick)
    
    print_summary(results, put_threads, get_threads, target_name)


def run_main():
    """Entry point that runs the async main() function"""
    asyncio.run(main())


if __name__ == '__main__':
    run_main()
