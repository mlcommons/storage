"""Streaming checkpoint implementation with producer-consumer pattern.

This module implements efficient checkpoint I/O that maximizes training throughput
by isolating data generation from storage operations using shared memory buffers.
"""

import os
import time
import multiprocessing as mp
from contextlib import contextmanager


# Per-node MPI rank count, used to size the dgen-py generator's thread pool.
#
# Issue #689: dgen-py's threads must be throttled so that N ranks sharing a
# node don't each spin up threads = os.cpu_count() and oversubscribe the
# host. The throttle divisor must be the number of ranks sharing THIS node
# (local size) — os.cpu_count() is already a per-node figure. Dividing it by
# the GLOBAL world size instead (the bug prior to this fix) starves
# generation on any multi-node run: e.g. 192 local CPUs / 64 global ranks
# (16 nodes x 4 ranks/node) = 3 threads/rank, when only 4 ranks actually
# share the node and 192 / 4 = 48 threads/rank is correct. That exact
# 3-thread figure is what issue #689's profiling observed, with generation
# throughput consequently landing close to the storage write rate instead
# of comfortably ahead of it.
#
# This mirrors DLIO_local_changes' storage#671 fix, which hit the identical
# global-vs-local mismatch for per-node accounting (there, OpenMPI's
# COMM_TYPE_SHARED split collapsing to size 1 on remote nodes; here, using
# the wrong env var entirely) and fixed it the same way: prefer the
# launcher's LOCAL size env var over any global/world figure.
_LOCAL_SIZE_ENV_VARS = (
    'OMPI_COMM_WORLD_LOCAL_SIZE',   # OpenMPI
    'MPI_LOCALNRANKS',              # MPICH / Hydra
    'MV2_COMM_WORLD_LOCAL_SIZE',    # MVAPICH2
)


def _local_ranks_per_node(env=None):
    """Return the number of MPI ranks sharing this node (>= 1).

    Checks the launcher-specific LOCAL size env vars in ``_LOCAL_SIZE_ENV_VARS``
    (first present, parseable, positive value wins). Falls back to 1 — i.e.
    "assume this rank has the node to itself" — when none are set (single-node
    / non-MPI launches) or all are malformed. 1 is deliberately the safe
    default: it never reproduces the #689 under-threading bug, and a rank
    that turns out not to be alone simply shares CPU time with its
    node-mates rather than being starved down to a handful of threads.
    """
    if env is None:
        env = os.environ
    for _env_var in _LOCAL_SIZE_ENV_VARS:
        _v = env.get(_env_var)
        if _v:
            try:
                parsed = int(_v)
            except (TypeError, ValueError):
                continue
            if parsed >= 1:
                return parsed
    return 1


# Selecting the streaming-checkpoint writer subprocess start method.
#
# The right method depends on the backend, so it is a *choice* with a
# backend-aware default (issues #642 and #682):
#
#   * POSIX file backend (``backend='file'`` or a scheme-less / ``file://``
#     path): the parent never starts s3dlio's Tokio runtime, so ``fork`` is
#     safe *and* ~40% faster — it keeps the parent's copy-on-write warm state
#     and CPU/NUMA locality with the shared-memory buffers (#682). Default:
#     ``fork``.
#   * Object-storage / s3dlio path (``s3dlio``, ``direct_fs``,
#     ``s3torchconnector``, ``minio``, or any remote-scheme URI): the parent
#     has already started s3dlio's Tokio runtime via
#     ``ObjStoreLibStorage._preflight()`` → ``s3dlio.list()`` by the time the
#     writer is created. ``fork()`` copies that mutex state without the
#     threads holding the mutexes and the writer futex-deadlocks the first
#     time it re-enters s3dlio (#642). Default: ``forkserver`` — the clean
#     "no live Tokio" property of ``spawn`` without ``spawn``'s ``__main__``
#     re-import (``spawn`` also hangs MPI ranks re-joining the OpenMPI
#     communicator, the historical reason fork was chosen originally).
#
# Users may override the default with the ``mp_start_method`` constructor
# argument or the ``MLPS_CHECKPOINT_MP_START_METHOD`` environment variable
# (constructor arg wins). Explicitly requesting ``fork`` on the object-storage
# path is refused rather than silently honored, because it deadlocks (#642).
MP_START_METHOD_ENV = "MLPS_CHECKPOINT_MP_START_METHOD"
_VALID_MP_START_METHODS = ("auto", "fork", "forkserver", "spawn")


def _is_posix_file_backend(backend=None, uri_or_path=None):
    """True when the writer targets a local POSIX file — the only path where
    forking the writer is safe (no parent-side s3dlio Tokio runtime).

    Mirrors ``StorageWriterFactory``'s own resolution: an explicit
    ``backend='file'`` is POSIX; with no explicit backend, a scheme-less or
    ``file://`` path auto-detects to the local file writer, anything else
    (``s3://``, ``az://``, ``gs://``, ``direct://``, …) is object storage.
    """
    if backend:
        return backend == "file"
    uri = uri_or_path or ""
    if "://" not in uri:
        return True
    return uri.startswith("file://")


def _resolve_mp_start_method(start_method="auto", *, backend=None, uri_or_path=None):
    """Resolve a requested start method to a concrete ``fork`` / ``forkserver``
    / ``spawn``, applying the backend-aware default and the #642 guardrail.

    ``start_method='auto'`` (the default) picks ``fork`` on the POSIX file
    backend and ``forkserver`` on the object-storage path. An explicit value
    is honored, except ``fork`` on the object-storage path which raises rather
    than deadlock.
    """
    method = (start_method or "auto").lower()
    if method not in _VALID_MP_START_METHODS:
        raise ValueError(
            f"invalid checkpoint writer start method {start_method!r}; choose "
            f"one of {_VALID_MP_START_METHODS[1:]} or 'auto' "
            f"(via mp_start_method= or ${MP_START_METHOD_ENV})"
        )

    posix_file = _is_posix_file_backend(backend, uri_or_path)

    if method == "auto":
        return "fork" if posix_file else "forkserver"

    if method == "fork" and not posix_file:
        raise ValueError(
            f"checkpoint writer start method 'fork' is unsafe on the "
            f"object-storage backend ({(backend or uri_or_path)!r}): fork() "
            f"after s3dlio's Tokio runtime is live deadlocks the writer "
            f"(issue #642). Use 'forkserver' (the default there) or 'auto'."
        )
    return method


def _mp_availability_fallback(method):
    """Platform-availability fallback order for a resolved method (NOT a policy
    choice): degrade only when the OS can't provide the requested context —
    fork→forkserver→spawn, forkserver→spawn, spawn→spawn."""
    if method == "fork":
        return ("fork", "forkserver", "spawn")
    if method == "forkserver":
        return ("forkserver", "spawn")
    return ("spawn",)


def _writer_mp_context(start_method="auto", *, backend=None, uri_or_path=None):
    """Return the multiprocessing context for the streaming-checkpoint writer.

    ``start_method`` selects the policy (see ``_resolve_mp_start_method``);
    the resolved method then degrades gracefully if the platform cannot
    provide it (e.g. no ``fork``/``forkserver`` on Windows/macOS).
    """
    method = _resolve_mp_start_method(
        start_method, backend=backend, uri_or_path=uri_or_path
    )
    for candidate in _mp_availability_fallback(method):
        try:
            return mp.get_context(candidate)
        except ValueError:
            continue
    return mp.get_context("spawn")
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from typing import Optional, Dict, Any

from .storage_writers import StorageWriterFactory


# Issue #768 / #777: pre-3.13 CPython resource_tracker double-unlink workaround.
#
# The tracker daemon's cache is a `set`; REGISTER = idempotent add,
# UNREGISTER = set.remove (KeyError on absent). SharedMemory(name=...) attach
# unconditionally sends REGISTER to the tracker, so a writer subprocess that
# borrows Main-owned buffers ends up double-registered. Without any
# workaround, the tracker at exit unlinks whatever's left in cache and races
# Main's own shm.unlink() — one wins, the other raises FileNotFoundError
# and (on the shm side) killed the owning MPI rank in Main's finally block
# (#768).
#
# The initial #768 fix used the "immediate unregister in child" community
# pattern (register then unregister right after attach). That pattern is
# correct only when the parent NEVER explicitly calls shm.unlink() /
# SemLock finalizer itself — because SharedMemory.unlink() and
# SemLock._cleanup both end in resource_tracker.unregister() on the same
# name. Our parent DOES call shm.unlink() per save() cycle (a fresh pool
# per save() means the previous pool must be released or /dev/shm leaks
# gigabytes per checkpoint), and Main's SemLock Finalize inevitably fires
# on Queue/Event GC. The child's immediate UNREGISTER empties the set
# entry; Main's later UNREGISTER then hits set.remove on an absent name
# -> KeyError storm at 64 ranks x N save() calls (#777). On the semaphore
# side the same race + SemLock._cleanup's own sem_unlink produced a
# 700+/run "sem_unlink -> FileNotFoundError" cascade, correlated with a
# cluster-wide livelock right after checkpoint 1.
#
# The correct shape for our lifecycle is the OTHER well-known variant:
# skip registration in the child entirely, so the tracker cache holds
# exactly one entry (Main's) and Main's later unlink -> unregister finds
# it and removes it cleanly. Functionally equivalent to Python 3.13's
# SharedMemory(track=False).
#
# We only do this for shared_memory. SemLock's __setstate__ registers with
# the tracker BEFORE our target function runs (it happens during pickle
# unpickling in the forkserver bootstrap), so a context-manager scope
# can't catch it. That's OK: pre-#768 semaphore behavior was "alarming
# SemLock._rebuild warnings but functionally correct" — strictly better
# than the KeyError + double-sem_unlink cascade #777 reports.
@contextmanager
def _child_skips_shm_registration():
    """Suppress ``resource_tracker.register`` for ``shared_memory`` within
    the block.

    Used in the writer subprocess so ``SharedMemory(name=...)`` attach
    does NOT double-register Main-owned buffer names in the tracker
    daemon's cache. See the module comment above for why this is the
    correct pattern for our lifecycle and why the "immediate unregister
    after attach" variant (initial #768 fix) is not.

    Only the ``register`` shim is installed — ``unregister`` is left
    alone, so the child cannot accidentally send an UNREGISTER for a
    name it never REGISTERed. The patch is scoped to the ``with`` block
    and restored in ``finally`` even on exception.
    """
    from multiprocessing import resource_tracker
    orig_register = resource_tracker.register

    def _register_skip_shm(name, rtype):
        if rtype == "shared_memory":
            return
        return orig_register(name, rtype)

    resource_tracker.register = _register_skip_shm
    try:
        yield
    finally:
        resource_tracker.register = orig_register

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
        num_parallel_readers: int = 8,
        read_chunk_size: Optional[int] = None,
        mp_start_method: Optional[str] = None,
        **backend_kwargs
    ):
        """Initialize streaming checkpoint configuration.
        
        Args:
            chunk_size: Size of each write buffer in bytes (default: 32 MB)
            num_buffers: Number of buffers in pool (default: 64 for 2 GB total)
            use_dgen: Use dgen-py for fast parallel generation (default: True)
            backend: Explicit backend name ('file', 's3dlio', etc.) or None for auto-detect
            use_direct_io: Enable O_DIRECT for file backend (requires aligned buffers)
            fadvise_mode: Fadvise strategy - 'none', 'sequential', or 'dontneed' (default: 'none')
            num_parallel_readers: Number of parallel range-GET threads for load() (default: 8)
            read_chunk_size: Chunk size for read range-GETs in bytes (default: 4 × chunk_size).
                             Larger values reduce per-request HTTP overhead at the cost of
                             more RAM per reader thread (peak = num_parallel_readers × read_chunk_size).
            mp_start_method: Writer subprocess start method — 'auto' (default),
                             'fork', 'forkserver', or 'spawn'. 'auto' picks 'fork'
                             on the POSIX file backend (fast, #682) and 'forkserver'
                             on the object-storage / s3dlio path (fork deadlocks
                             there, #642). Overrides the ``MLPS_CHECKPOINT_MP_START_METHOD``
                             environment variable; if both are None, defaults to 'auto'.
                             Requesting 'fork' on the object-storage path is refused.
            **backend_kwargs: Additional backend-specific options
        """
        self.chunk_size = chunk_size
        self.num_buffers = num_buffers
        self.use_dgen = use_dgen and HAS_DGEN
        self.backend = backend
        self.use_direct_io = use_direct_io
        self.fadvise_mode = fadvise_mode
        self.num_parallel_readers = num_parallel_readers
        self.read_chunk_size = read_chunk_size if read_chunk_size is not None else chunk_size * 4
        # Writer start-method policy: explicit arg > env var > 'auto'
        # (backend-aware). Resolved to a concrete method at save() time, when
        # the target filepath/backend is known. See _writer_mp_context.
        self.mp_start_method = (
            mp_start_method
            or os.environ.get(MP_START_METHOD_ENV)
            or "auto"
        )
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
        
        # Writer subprocess context — see the module-scope helpers for the
        # backend-aware default: 'fork' on the POSIX file backend (fast, #682),
        # 'forkserver' on the object-storage / s3dlio path (fork after live
        # Tokio deadlocks, #642). self.mp_start_method (constructor arg or
        # $MLPS_CHECKPOINT_MP_START_METHOD) overrides the default; 'fork' on
        # the object-storage path is refused rather than silently deadlocked.
        # The backend is resolved from self.backend, falling back to the
        # target filepath's URI scheme when backend is auto-detect (None).
        #
        # CRITICAL: IPC objects (Queue, Event) MUST be created from the SAME
        # context as the child process — mixing contexts (e.g. fork-context
        # semaphores passed to a spawn-context child) causes:
        #   RuntimeError: A SemLock created in a fork context is being shared
        #                 with a process in a spawn context.
        # Always create IPC objects from ctx BEFORE ctx.Process().
        ctx = _writer_mp_context(
            self.mp_start_method, backend=self.backend, uri_or_path=filepath
        )
        print(f"[Main] Writer start method: {ctx.get_start_method()} "
              f"(policy={self.mp_start_method!r}, backend={self.backend or 'auto'})")

        # Setup IPC using the same context as the child process.
        buffer_queue = ctx.Queue(maxsize=self.num_buffers)
        stop_event = ctx.Event()
        stats_queue = ctx.Queue()
        
        writer_proc = ctx.Process(
            target=self._writer_process,
            args=(buffer_names, self.chunk_size, filepath, total_size_bytes,
                  buffer_queue, stop_event, stats_queue, self.backend, actual_direct_io, self.fadvise_mode),
            kwargs=self.backend_kwargs
        )
        writer_proc.start()
        print(f"\n[Main] Writer process started (PID={writer_proc.pid})")
        
        try:
            # Producer loop
            print(f"[Main] Starting producer at {time.perf_counter():.3f}s")
            gen_time = self._run_producer(
                buffers, buffer_queue, total_size_bytes,
                generator, data_generator
            )
            print(f"[Main] Producer finished at {time.perf_counter():.3f}s")
            
            # Signal completion and wait for writer
            print(f"[Main] Signaling writer to stop at {time.perf_counter():.3f}s")
            buffer_queue.put(None)
            print(f"[Main] Waiting for writer to join at {time.perf_counter():.3f}s")
            writer_proc.join(timeout=300)
            print(f"[Main] Writer joined at {time.perf_counter():.3f}s")
            
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
            # Cleanup buffers — see _release_buffer_pool for the #768 defensive
            # FileNotFoundError swallow that keeps a raced writer-side unlink
            # from crashing the owning MPI rank in Main's finally.
            self._release_buffer_pool(buffers)
        
        # Collect results
        if stats_queue.empty():
            raise RuntimeError("Writer process failed to return statistics")
        
        stats = stats_queue.get()
        if 'error' in stats:
            raise RuntimeError(f"Writer process error: {stats['error']}")
        
        return self._format_results(stats, gen_time, time.time() - start_time, total_size_bytes)
    
    @staticmethod
    def _release_buffer_pool(buffers):
        """Release Main's per-save() shared-memory buffer pool.

        Defensive against #768: swallows ``FileNotFoundError`` on
        ``shm.unlink()`` because a second unlink of an already-unlinked
        segment is exactly the desired end state (segment gone), not a real
        failure — that spurious exception is what crashed the owning MPI
        rank and killed the whole job in the reporter's traceback.

        Other exceptions (``PermissionError``, ``OSError`` from the IO layer)
        still propagate: they indicate real bugs, not the resource_tracker
        double-unlink race, and must not be masked by this shim.
        Non-``FileNotFoundError`` ``close()`` failures on one buffer do not
        stop cleanup of the rest — segments are independent POSIX names and
        one bad segment shouldn't strand N-1 others as leaks.
        """
        for shm in buffers:
            try:
                shm.close()
            except Exception:
                pass  # keep unwinding — the remaining unlink still matters
            try:
                shm.unlink()
            except FileNotFoundError:
                # #768: writer's resource_tracker won the race. Segment is
                # gone, which is what we wanted. Not a real error.
                pass

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
        
        # Throttle dgen-py threads when running under MPI to avoid
        # overloading the host with N ranks x all-CPU threads simultaneously.
        # Divide by the ranks sharing THIS node (#689), not the global world
        # size — see _local_ranks_per_node for why that distinction matters.
        local_ranks = _local_ranks_per_node()
        total_cpus = os.cpu_count() or 4
        max_threads = max(1, total_cpus // local_ranks)
        print(f"[Main] Initializing dgen-py (local_ranks_per_node={local_ranks}, threads={max_threads}/{total_cpus} CPUs)...")
        try:
            generator = dgen_py.Generator(
                size=total_size_bytes,
                chunk_size=self.chunk_size,  # Match our buffer size
                dedup_ratio=1.0,
                compress_ratio=1.0,
                numa_mode="auto",
                max_threads=max_threads,  # Throttled by local ranks-per-node (#689)
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
        import os
        import sys
        
        print(f"[Writer] Starting (PID={os.getpid()})")
        
        # DEBUG: Check if environment variables are inherited
        aws_key = os.environ.get('AWS_ACCESS_KEY_ID', 'NOT SET')
        aws_endpoint = os.environ.get('AWS_ENDPOINT_URL', 'NOT SET')
        print(f"[Writer] DEBUG: AWS_ACCESS_KEY_ID = {aws_key[:4] if aws_key != 'NOT SET' else 'NOT SET'}***")
        print(f"[Writer] DEBUG: AWS_ENDPOINT_URL = {aws_endpoint}")
        
        # Attach to shared memory buffers.
        #
        # #768/#777: suppress this child's tracker REGISTER for each
        # SharedMemory attach. Main owns these segments and will call
        # shm.unlink() itself per save() cycle; if we let the child ALSO
        # register with the tracker daemon, either the tracker unlinks at
        # exit and races Main's unlink -> FileNotFoundError (#768), or we
        # "immediate unregister" and Main's later unregister trips
        # KeyError on an already-empty set (#777). Skipping the register
        # entirely leaves the daemon holding exactly one entry (Main's),
        # so Main's unlink -> unregister finds and removes it cleanly.
        # Equivalent to SharedMemory(track=False) which lands in 3.13.
        buffers = []
        with _child_skips_shm_registration():
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
            stats_queue.close()
            stats_queue.join_thread()
            for shm in buffers:
                shm.close()
            sys.stdout.flush()
            os._exit(1)
        
        written = 0
        total_io_time = 0.0
        chunks_written = 0
        _write_error = None  # Error from write loop, if any

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
            # Record the error; let the finally block handle the single stats_queue.put().
            _write_error = str(e)
            print(f"[Writer] ERROR during write: {e}")

        finally:
            # Close writer and collect stats — runs whether write succeeded or failed.
            close_time = 0.0
            writer_stats = {'backend': backend or 'auto', 'total_bytes': written}
            try:
                close_start = time.perf_counter()
                writer_stats = writer.close()
                close_time = time.perf_counter() - close_start
                total_io_time += close_time
                print(f"[Writer] Closed: {writer_stats} (close time: {close_time:.4f}s)")
            except Exception as e:
                print(f"[Writer] ERROR closing writer: {e}")
                if _write_error is None:
                    _write_error = f"close() failed: {e}"

            # Force cleanup of storage-library resources.
            try:
                del writer
                print(f"[Writer] Deleted writer object")
            except Exception:
                pass

            # Build result dict — single put to stats_queue.
            result = {
                'io_time': total_io_time,
                'close_time': close_time,
                'total_bytes': written,
                'chunks_written': chunks_written,
                'backend_stats': writer_stats,
            }
            if _write_error is not None:
                result['error'] = _write_error

            # CRITICAL: flush stats_queue's background sender thread BEFORE
            # os._exit() kills it.  mp.Queue.put() is non-blocking — it queues
            # the item locally and a background thread sends it over the pipe.
            # os._exit() bypasses Python cleanup, killing that thread and losing
            # the data.  close() + join_thread() drain the pipe synchronously.
            stats_queue.put(result)
            print(f"[Writer] Flushing stats queue...")
            stats_queue.close()
            stats_queue.join_thread()

            for shm in buffers:
                shm.close()

            exit_code = 1 if _write_error is not None else 0
            print(f"[Writer] Finished (exit_code={exit_code})")
            print(f"[Writer] Exiting (PID={os.getpid()})")
            sys.stdout.flush()
            os._exit(exit_code)
    
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

    def load(
        self,
        filepath: str,
        total_size_bytes: int,
    ) -> dict:
        """Load (restore) a checkpoint using streaming byte-range GETs.

        Reads the object in chunk_size pieces and discards each chunk
        immediately after receipt.  Peak RAM = chunk_size bytes (one chunk
        in flight at a time) — the same constant footprint as save().

        For s3dlio backend this uses s3dlio.get_range(uri, offset, length)
        which returns a BytesView (zero-copy); for minio it uses a Range-GET
        via the minio SDK; for s3torchconnector it reads sequentially via
        S3Reader.read(chunk_size).

        Args:
            filepath:         URI of the checkpoint written by save().
            total_size_bytes: Exact size in bytes (same value passed to save()).

        Returns:
            Dictionary containing:
                - io_time (float): Seconds spent in I/O calls.
                - total_time (float): Wall-clock seconds for the entire load.
                - total_bytes (int): Bytes received.
                - chunks (int): Number of chunk reads issued.
                - io_throughput_gbps (float): total_bytes / io_time.
                - backend_stats (dict): Backend-specific counters.
        """
        from .storage_readers import StorageReaderFactory

        if total_size_bytes <= 0:
            raise ValueError(f"Invalid total_size_bytes: {total_size_bytes}")

        print("=" * 80)
        print("STREAMING CHECKPOINT LOAD - Byte-Range GETs")
        print("=" * 80)
        print(f"Input:       {filepath}")
        print(f"Backend:     {self.backend or 'auto-detect'}")
        print(f"Total size:  {total_size_bytes / (1024**3):.2f} GB")
        print(f"Chunk size:  {self.chunk_size / (1024**2):.0f} MB  (peak RAM = one chunk)")
        print("=" * 80)

        # All three backends support offset-based read_chunk():
        # - s3dlio: get_range(uri, offset, length)
        # - minio: Range-GET via SDK
        # - s3torchconnector: range_based(buffer_size=0) reader + seek(offset)
        # All can run in parallel with multiple independent reader instances.
        # Use read_chunk_size (default 4× write chunk_size) to reduce per-request
        # HTTP overhead: fewer, larger range-GETs are more efficient than many small ones.
        n_workers = self.num_parallel_readers
        effective_chunk = self.read_chunk_size
        print(f"Read chunks: {effective_chunk // (1024**2)} MB × {n_workers} workers  "
              f"(peak RAM ≤ {effective_chunk * n_workers // (1024**2)} MB)")
        print("=" * 80)

        total_read = 0
        io_time    = 0.0
        chunks     = 0
        wall_start = time.time()
        backend_stats = {}

        if n_workers <= 1:
            # ----------------------------------------------------------------
            # Serial path (fallback for n_workers=1)
            # ----------------------------------------------------------------
            reader = StorageReaderFactory.create(
                filepath,
                backend=self.backend,
                fadvise_mode=self.fadvise_mode,
                chunk_size=effective_chunk,
            )
            try:
                while total_read < total_size_bytes:
                    size = min(effective_chunk, total_size_bytes - total_read)

                    t0 = time.perf_counter()
                    nbytes = reader.read_chunk(total_read, size)
                    io_time += time.perf_counter() - t0

                    total_read += nbytes
                    chunks += 1

                    if chunks % 10 == 0:
                        tp = (total_read / 1024**3) / io_time if io_time > 0 else 0
                        print(f"[Load] {total_read / (1024**3):.2f} GB  {tp:.2f} GB/s")

                    if nbytes == 0:
                        raise RuntimeError(
                            f"Reader returned 0 bytes at offset {total_read} "
                            f"(expected {size} more bytes)"
                        )
            finally:
                backend_stats = reader.close()

        else:
            # ----------------------------------------------------------------
            # Parallel path — n_workers concurrent streaming threads.
            #
            # Each worker is assigned a contiguous byte block [block_start,
            # block_end) of the object and reads it with a single HTTP
            # connection.  Two strategies are used depending on the reader:
            #
            #   stream_block (s3torchconnector and any reader that implements
            #   it): opens ONE CRT GetObjectStream for the full block and
            #   iterates native CRT chunks (~8 MB each).  Peak RAM per worker
            #   ≈ one CRT chunk; total RAM ≈ n × 32 MB, constant for any
            #   object size.
            #
            #   read_chunk loop (s3dlio, minio, any StorageReader without
            #   stream_block): calls read_chunk(offset, effective_chunk)
            #   sequentially within the block.  Peak RAM per worker =
            #   effective_chunk (128 MB by default); total ≈ n × 128 MB.
            #
            # Block boundaries are byte-aligned (not chunk-aligned) so the
            # scheme works for any total_size_bytes regardless of chunk size.
            # ----------------------------------------------------------------
            block_size = (total_size_bytes + n_workers - 1) // n_workers
            blocks = []
            pos = 0
            while pos < total_size_bytes:
                blocks.append((pos, min(pos + block_size, total_size_bytes)))
                pos += block_size

            n = min(n_workers, len(blocks))
            blocks = blocks[:n]

            readers = [
                StorageReaderFactory.create(
                    filepath, backend=self.backend,
                    fadvise_mode=self.fadvise_mode,
                    chunk_size=effective_chunk
                )
                for _ in range(n)
            ]

            def _read_block(reader, block_start, block_end, worker_id):
                t0 = time.perf_counter()

                if hasattr(reader, 'stream_block'):
                    # ONE streaming range-GET for the full block.
                    nb = reader.stream_block(block_start, block_end)
                    io_secs = time.perf_counter() - t0
                    gb = nb / 1024**3
                    rate = gb / io_secs if io_secs > 0 else 0
                    print(f"[Load w{worker_id}] {gb:.2f} GB  {rate:.2f} GB/s", flush=True)
                    return nb, io_secs, 1

                # Chunk-based fallback (s3dlio, minio).
                local_bytes  = 0
                local_io_time = 0.0
                local_chunks = 0
                off = block_start
                while off < block_end:
                    sz = min(effective_chunk, block_end - off)
                    t1 = time.perf_counter()
                    nb = reader.read_chunk(off, sz)
                    local_io_time += time.perf_counter() - t1
                    if nb == 0:
                        raise RuntimeError(
                            f"Reader returned 0 bytes at offset {off} "
                            f"(expected {sz} more bytes)"
                        )
                    off          += nb
                    local_bytes  += nb
                    local_chunks += 1
                    if local_chunks % 16 == 0:
                        gb_done = local_bytes / 1024**3
                        rate = gb_done / local_io_time if local_io_time > 0 else 0
                        print(f"[Load w{worker_id}] {gb_done:.2f} GB  {rate:.2f} GB/s",
                              flush=True)
                return local_bytes, local_io_time, local_chunks

            try:
                with ThreadPoolExecutor(max_workers=n) as pool:
                    futs = [
                        pool.submit(_read_block, readers[i],
                                    blocks[i][0], blocks[i][1], i)
                        for i in range(n)
                    ]
                    results = [f.result() for f in futs]  # re-raises on error

                total_read = sum(nb for nb, _, _ in results)
                io_time    = max(t  for _,  t, _ in results)
                chunks     = sum(c  for _, _, c in results)
            finally:
                for r in readers:
                    try:
                        backend_stats = r.close()
                    except Exception:
                        pass

        total_time  = time.time() - wall_start
        io_gbps     = (total_read / 1024**3) / io_time if io_time > 0 else 0.0

        print("\n" + "=" * 80)
        print("LOAD RESULTS")
        print("=" * 80)
        print(f"I/O:        {io_time:.4f}s @ {io_gbps:.2f} GB/s")
        print(f"Total:      {total_time:.4f}s")
        print(f"Chunks:     {chunks}")
        print("=" * 80)

        return {
            'io_time':            io_time,
            'total_time':         total_time,
            'total_bytes':        total_read,
            'chunks':             chunks,
            'io_throughput_gbps': io_gbps,
            'backend_stats':      backend_stats,
        }
