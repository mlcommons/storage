"""Streaming checkpoint implementation with producer-consumer pattern.

This module implements efficient checkpoint I/O that maximizes training throughput
by isolating data generation from storage operations using shared memory buffers.
"""

import os
import time
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Optional, Dict, Any

from .storage_writers import StorageWriterFactory

# Try to import dgen-py for high-performance data generation
try:
    import dgen_py
    HAS_DGEN = True
except ImportError:
    HAS_DGEN = False


class StreamingCheckpointing:
    """Producer-consumer streaming checkpoint with buffer pool.
    
    This class implements a two-process pipeline:
    1. Producer (main process): Generates checkpoint data into shared memory buffers
    2. Consumer (writer process): Writes buffers to storage backend
    
    The buffer pool allows overlapping generation and I/O for maximum throughput.
    Accurate I/O timing is maintained by isolating the writer in a separate process.
    
    Attributes:
        chunk_size: Size of each buffer chunk in bytes (default: 32 MB)
        num_buffers: Number of buffers in the pool (default: 64 = 2 GB pool)
        use_dgen: Whether to use dgen-py for parallel data generation
        backend: Storage backend ('file', 's3dlio', etc.)
        backend_kwargs: Backend-specific configuration
        
    Examples:
        >>> # Simple local file checkpoint
        >>> checkpoint = StreamingCheckpointing(
        ...     chunk_size=32 * 1024 * 1024,  # 32 MB chunks  
        ...     num_buffers=64,  # 2 GB buffer pool
        ...     backend='file'
        ... )
        >>> results = checkpoint.save('/tmp/checkpoint.dat', total_size_bytes=10*1024**3)
        >>> print(f"I/O throughput: {results['io_throughput_gbps']:.2f} GB/s")
        
        >>> # S3 checkpoint via s3dlio
        >>> checkpoint = StreamingCheckpointing(backend='s3dlio')
        >>> results = checkpoint.save(
        ...     's3://my-bucket/checkpoints/ckpt_epoch_10.dat',
        ...     total_size_bytes=100*1024**3
        ... )
    """
    
    def __init__(
        self,
        chunk_size: int = 32 * 1024 * 1024,
        num_buffers: int = 64,
        use_dgen: bool = True,
        backend: Optional[str] = None,
        use_direct_io: bool = False,
        fadvise_mode: str = 'none',
        **backend_kwargs
    ):
        """Initialize streaming checkpoint configuration.
        
        Args:
            chunk_size: Size of each buffer in bytes (default: 32 MB)
            num_buffers: Number of buffers in pool (default: 64 for 2 GB total)
            use_dgen: Use dgen-py for fast parallel generation (default: True)
            backend: Explicit backend name ('file', 's3dlio', etc.) or None for auto-detect
            use_direct_io: Enable O_DIRECT for file backend (requires aligned buffers)
            fadvise_mode: Fadvise strategy - 'none', 'sequential', or 'dontneed' (default: 'none')
            **backend_kwargs: Additional backend-specific options
        """
        self.chunk_size = chunk_size
        self.num_buffers = num_buffers
        self.use_dgen = use_dgen and HAS_DGEN
        self.backend = backend
        self.use_direct_io = use_direct_io
        self.fadvise_mode = fadvise_mode
        self.backend_kwargs = backend_kwargs
        
        # dgen-py is REQUIRED if no custom generator will be provided
        if use_dgen and not HAS_DGEN:
            raise ImportError(
                "dgen-py is required for data generation. "
                "Install with: pip install dgen-py"
            )
    
    def save(
        self,
        filepath: str,
        total_size_bytes: int,
        data_generator: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Save checkpoint using streaming producer-consumer pattern.
        
        Args:
            filepath: Output path or URI (file://, s3://, az://, etc.)
            total_size_bytes: Total checkpoint size in bytes
            data_generator: Optional custom generator function(buffer, size) -> None
                           If None, uses dgen-py (must be installed)
                           Custom generators MUST use efficient buffer operations (no byte-by-byte)
                           
        Returns:
            Dictionary containing:
                - gen_time: Time spent generating data (seconds)
                - io_time: Time spent in I/O operations (seconds)
                - close_time: Time spent in finalize/fsync (seconds)
                - total_time: End-to-end elapsed time (seconds)
                - total_bytes: Total bytes written
                - chunks: Number of chunks written
                - gen_throughput_gbps: Generation throughput (GB/s)
                - io_throughput_gbps: I/O throughput (GB/s)
                - throughput_ratio: Generation/I/O speed ratio (should be > 2x)
                - pipeline_overhead_pct: Pipeline coordination overhead (should be < 10%)
                - bottleneck: "I/O" or "Generation" (should always be "I/O")
                - backend_stats: Backend-specific statistics
                
        Raises:
            RuntimeError: If writer process fails or times out
            ValueError: If parameters are invalid
        """
        if total_size_bytes <= 0:
            raise ValueError(f"Invalid total_size_bytes: {total_size_bytes}")
        
        if total_size_bytes < self.chunk_size:
            import warnings
            warnings.warn(
                f"total_size_bytes ({total_size_bytes}) < chunk_size ({self.chunk_size}). "
                f"Consider reducing chunk_size for better efficiency.",
                RuntimeWarning
            )
        
        print("=" * 80)
        print("STREAMING CHECKPOINT - Producer-Consumer Pattern")
        print("=" * 80)
        print(f"Output:      {filepath}")
        print(f"Backend:     {self.backend or 'auto-detect'}")
        print(f"Total size:  {total_size_bytes / (1024**3):.2f} GB")
        print(f"Buffer size: {self.chunk_size / (1024**2):.0f} MB")
        print(f"Buffer pool: {self.num_buffers} × {self.chunk_size / (1024**2):.0f} MB = {(self.num_buffers * self.chunk_size) / (1024**3):.2f} GB")
        print(f"Direct I/O:  {self.use_direct_io}")
        print(f"Use dgen-py: {self.use_dgen}")
        print("=" * 80)
        
        start_time = time.time()
        
        # Create buffer pool
        buffers, buffer_names = self._create_buffer_pool()
        
        # Initialize data generator
        generator = self._init_generator(total_size_bytes) if data_generator is None else None
        
        # Disable O_DIRECT for shared_memory (not page-aligned)
        actual_direct_io = False
        if self.use_direct_io:
            print(f"[Main] ⚠ Disabling O_DIRECT (shared_memory buffers not page-aligned)")
        
        # Setup IPC
        buffer_queue = mp.Queue(maxsize=self.num_buffers)
        stop_event = mp.Event()
        stats_queue = mp.Queue()
        
        # Start writer process
        writer_proc = mp.Process(
            target=self._writer_process,
            args=(buffer_names, self.chunk_size, filepath, total_size_bytes,
                  buffer_queue, stop_event, stats_queue, self.backend, actual_direct_io, self.fadvise_mode),
            kwargs=self.backend_kwargs
        )
        writer_proc.start()
        print(f"\n[Main] Writer process started (PID={writer_proc.pid})")
        
        try:
            # Producer loop
            gen_time = self._run_producer(
                buffers, buffer_queue, total_size_bytes,
                generator, data_generator
            )
            
            # Signal completion and wait for writer
            buffer_queue.put(None)
            writer_proc.join(timeout=300)
            
            if writer_proc.is_alive():
                print("[Main] WARNING: Writer timeout!")
                writer_proc.terminate()
                raise RuntimeError("Writer process timed out after 300 seconds")
        
        except Exception as e:
            # Ensure writer process is terminated on any error
            print(f"[Main] Error during checkpoint: {e}")
            buffer_queue.put(None)  # Signal writer to stop
            writer_proc.terminate()
            writer_proc.join(timeout=5)
            raise
        
        finally:
            # Cleanup buffers
            for shm in buffers:
                shm.close()
                shm.unlink()
        
        # Collect results
        if stats_queue.empty():
            raise RuntimeError("Writer process failed to return statistics")
        
        stats = stats_queue.get()
        if 'error' in stats:
            raise RuntimeError(f"Writer process error: {stats['error']}")
        
        return self._format_results(stats, gen_time, time.time() - start_time, total_size_bytes)
    
    def _create_buffer_pool(self):
        """Create shared memory buffer pool."""
        print(f"\n[Main] Creating {self.num_buffers} buffers...")
        buffers = []
        buffer_names = []
        
        for i in range(self.num_buffers):
            shm_name = f"ckpt_{os.getpid()}_{i}_{int(time.time() * 1e6)}"
            shm = shared_memory.SharedMemory(create=True, size=self.chunk_size, name=shm_name)
            buffers.append(shm)
            buffer_names.append(shm_name)
        
        print(f"[Main] Buffer pool ready: {self.num_buffers * self.chunk_size / (1024**3):.2f} GB")
        return buffers, buffer_names
    
    def _init_generator(self, total_size_bytes):
        """Initialize dgen-py generator (required if no custom generator)."""
        if not self.use_dgen:
            return None
        
        if not HAS_DGEN:
            raise ImportError(
                "dgen-py is required but not installed. "
                "Install with: pip install dgen-py"
            )
        
        print(f"[Main] Initializing dgen-py...")
        try:
            generator = dgen_py.Generator(
                size=total_size_bytes,
                chunk_size=self.chunk_size,  # Match our buffer size
                dedup_ratio=1.0,
                compress_ratio=1.0,
                numa_mode="auto",      # CRITICAL: Enable NUMA-aware multi-threading
                max_threads=None       # CRITICAL: Use all available cores
            )
            print(f"[Main] Generator ready")
            return generator
        except Exception as e:
            raise RuntimeError(f"Failed to initialize dgen-py generator: {e}")
    
    def _run_producer(self, buffers, buffer_queue, total_size_bytes, generator, custom_generator):
        """Run producer loop to fill buffers."""
        print(f"[Main] Starting producer (buffer pool reuse pattern)...")
        gen_start = time.time()
        generated = 0
        buffer_idx = 0
        
        # Validate we have a generator BEFORE starting loop
        if not custom_generator and not generator:
            raise RuntimeError(
                "No data generator available. Either provide data_generator parameter "
                "or ensure dgen-py is installed and use_dgen=True."
            )
        
        while generated < total_size_bytes:
            current_chunk_size = min(self.chunk_size, total_size_bytes - generated)
            shm = buffers[buffer_idx]
            
            # Generate data directly into buffer (zero-copy)
            if custom_generator:
                # Custom generator MUST use efficient buffer operations
                custom_generator(shm.buf, current_chunk_size)
            elif generator:
                # dgen-py high-performance parallel generation
                generator.fill_chunk(shm.buf)
            
            # Signal writer (pass buffer index and size)
            buffer_queue.put((buffer_idx, current_chunk_size))
            
            generated += current_chunk_size
            buffer_idx = (buffer_idx + 1) % self.num_buffers  # Round-robin reuse
        
        gen_time = time.time() - gen_start
        print(f"[Main] Generation complete: {gen_time:.2f}s, {(total_size_bytes / (1024**3)) / gen_time:.2f} GB/s")
        return gen_time
    
    @staticmethod
    def _writer_process(buffer_names, chunk_size, filepath, total_size,
                       buffer_queue, stop_event, stats_queue, backend, use_direct_io, fadvise_mode, **backend_kwargs):
        """Writer process entry point - isolated I/O timing."""
        print(f"[Writer] Starting (PID={os.getpid()})")
        
        # Attach to shared memory buffers
        buffers = []
        for name in buffer_names:
            shm = shared_memory.SharedMemory(name=name)
            buffers.append(shm)
        
        print(f"[Writer] Attached to {len(buffers)} buffers ({chunk_size / (1024**2):.0f} MB each)")
        
        # Create storage writer
        try:
            writer = StorageWriterFactory.create(
                filepath,
                backend=backend,
                use_direct_io=use_direct_io,
                fadvise_mode=fadvise_mode,
                **backend_kwargs
            )
            writer_info = f"{backend or 'auto'} backend"
            if hasattr(writer, 'direct_io') and writer.direct_io:
                writer_info += " (O_DIRECT enabled)"
            print(f"[Writer] Using {writer_info}")
        except Exception as e:
            print(f"[Writer] ERROR: Failed to create storage writer: {e}")
            stats_queue.put({'error': str(e)})
            for shm in buffers:
                shm.close()
            return
        
        written = 0
        total_io_time = 0.0
        chunks_written = 0
        
        try:
            while written < total_size:
                item = buffer_queue.get()
                if item is None:
                    break
                
                buffer_idx, nbytes = item
                shm = buffers[buffer_idx]
                
                # Time ONLY the I/O operation
                io_start = time.perf_counter()
                bytes_written = writer.write_chunk(shm.buf, nbytes)
                total_io_time += time.perf_counter() - io_start
                
                written += bytes_written
                chunks_written += 1
                
                if chunks_written % 10 == 0:
                    throughput = (written / (1024**3)) / total_io_time if total_io_time > 0 else 0
                    print(f"[Writer] {written / (1024**3):.2f} GB, {throughput:.2f} GB/s")
        
        except Exception as e:
            print(f"[Writer] ERROR during write: {e}")
            stats_queue.put({'error': str(e)})
            return
        
        finally:
            # Close writer and get stats
            try:
                close_start = time.perf_counter()
                writer_stats = writer.close()
                close_time = time.perf_counter() - close_start
                total_io_time += close_time
                print(f"[Writer] Closed: {writer_stats} (close time: {close_time:.4f}s)")
            except Exception as e:
                print(f"[Writer] ERROR closing writer: {e}")
                writer_stats = {'backend': backend or 'auto', 'total_bytes': written}
                close_time = 0.0
            
            # Report stats
            stats_queue.put({
                'io_time': total_io_time,
                'close_time': close_time,
                'total_bytes': written,
                'chunks_written': chunks_written,
                'backend_stats': writer_stats,
            })
            
            for shm in buffers:
                shm.close()
            
            print(f"[Writer] Finished")
    
    def _format_results(self, stats, gen_time, total_time, total_size_bytes):
        """Format results for return."""
        gen_throughput = (total_size_bytes / (1024**3)) / gen_time
        io_throughput = (stats['total_bytes'] / (1024**3)) / stats['io_time']
        
        # Calculate improved metrics
        throughput_ratio = gen_throughput / io_throughput
        pipeline_overhead = ((total_time - max(gen_time, stats['io_time'])) / total_time) * 100
        bottleneck = "I/O" if stats['io_time'] > gen_time else "Generation"
        
        results = {
            'gen_time': gen_time,
            'io_time': stats['io_time'],
            'close_time': stats.get('close_time', 0.0),
            'total_time': total_time,
            'total_bytes': stats['total_bytes'],
            'chunks': stats['chunks_written'],
            'gen_throughput_gbps': gen_throughput,
            'io_throughput_gbps': io_throughput,
            'throughput_ratio': throughput_ratio,
            'pipeline_overhead_pct': pipeline_overhead,
            'bottleneck': bottleneck,
            'backend_stats': stats.get('backend_stats', {})
        }
        
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Generation:  {results['gen_time']:.4f}s @ {results['gen_throughput_gbps']:.2f} GB/s")
        print(f"I/O:         {results['io_time']:.4f}s @ {results['io_throughput_gbps']:.2f} GB/s")
        print(f"  - write:   {results['io_time'] - results['close_time']:.4f}s")
        print(f"  - close:   {results['close_time']:.4f}s (fsync/finalize)")
        print(f"Total:       {results['total_time']:.4f}s")
        print(f"")
        print(f"Throughput ratio: {results['throughput_ratio']:.1f}x (gen/io)")
        print(f"Pipeline overhead: {results['pipeline_overhead_pct']:.1f}%")
        print(f"Bottleneck: {results['bottleneck']}")
        print(f"Chunks: {results['chunks']}")
        print("=" * 80)
        
        return results
