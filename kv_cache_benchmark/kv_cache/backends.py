"""
Storage backend classes for KV Cache Benchmark.

Provides the abstract StorageBackend interface and concrete implementations
for GPU VRAM, CPU RAM, and NVMe/SSD storage tiers.
"""

import os
import gc
import time
import logging
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from kv_cache._compat import (
    HAS_TORCH, TORCH_AVAILABLE,
    HAS_CUPY, CUPY_AVAILABLE,
)
from kv_cache.config import cfg

if HAS_TORCH:
    import torch
if HAS_CUPY:
    import cupy as cp

logger = logging.getLogger(__name__)


# ============================================================================
# STORAGE BACKEND CLASSES
# ============================================================================

class StorageBackend:
    """Abstract base class for all storage backends (GPU, CPU, NVMe)."""

    from dataclasses import dataclass

    @dataclass
    class IOTiming:
        """Captures total latency along with host and device components."""
        total: float
        device: float
        host: float

    def write(self, key: str, data: np.ndarray) -> 'StorageBackend.IOTiming':
        """Writes data to the backend and returns latency breakdown."""
        raise NotImplementedError

    def read(self, key: str) -> Tuple[np.ndarray, 'StorageBackend.IOTiming']:
        """Reads data from the backend and returns the data and latency."""
        raise NotImplementedError

    def delete(self, key: str):
        """Deletes data from the backend."""
        raise NotImplementedError

    def clear(self):
        """Clears all data from the backend."""
        raise NotImplementedError


class GPUMemoryBackend(StorageBackend):
    """
    GPU VRAM storage backend.
    Uses PyTorch or CuPy for GPU operations. This is the fastest tier.
    """

    def __init__(self, use_torch=True, on_eviction_callback=None):
        self.on_eviction_callback = on_eviction_callback

        if use_torch and TORCH_AVAILABLE:
            self.backend = 'torch'
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cpu':
                raise RuntimeError("No GPU available for PyTorch backend")
            memory_fraction = cfg('gpu_backend', 'memory_fraction', default=0.8)
            torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
            torch.cuda.empty_cache()
        elif CUPY_AVAILABLE:
            self.backend = 'cupy'
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        else:
            raise RuntimeError("No GPU backend (PyTorch or CuPy) available.")

        self.cache = {}
        self.pinned_memory = {}

    def write(self, key: str, data: np.ndarray) -> StorageBackend.IOTiming:
        """Writes a NumPy array from CPU to GPU VRAM."""
        if self.backend == 'torch' and torch.cuda.is_available():
            required_bytes = data.nbytes
            max_eviction_attempts = cfg('gpu_backend', 'max_eviction_attempts', default=100)
            eviction_count = 0
            free_memory_threshold = cfg('gpu_backend', 'free_memory_threshold', default=0.1)
            usable_fraction = 1.0 - free_memory_threshold

            while eviction_count < max_eviction_attempts:
                free_memory = torch.cuda.mem_get_info()[0]
                if required_bytes <= free_memory * usable_fraction:
                    break

                torch.cuda.empty_cache()
                free_memory = torch.cuda.mem_get_info()[0]
                if required_bytes <= free_memory * usable_fraction:
                    break

                if len(self.cache) == 0:
                    logger.warning(
                        f"GPU OOM: Need {required_bytes / 1024**2:.1f}MB, "
                        f"have {free_memory / 1024**2:.1f}MB, no entries to evict"
                    )
                    break

                oldest_key = next(iter(self.cache))
                evicted_tensor = self.cache.pop(oldest_key)
                evicted_size = evicted_tensor.element_size() * evicted_tensor.nelement()
                del evicted_tensor

                if oldest_key in self.pinned_memory:
                    del self.pinned_memory[oldest_key]

                if self.on_eviction_callback:
                    try:
                        self.on_eviction_callback(oldest_key, 'gpu', evicted_size)
                    except Exception as e:
                        logger.warning(f"GPU eviction callback failed for {oldest_key}: {e}")

                eviction_count += 1
                logger.debug(
                    f"GPU eviction #{eviction_count}: evicted {oldest_key} "
                    f"({evicted_size / 1024**2:.1f}MB)"
                )

            if eviction_count > 0:
                torch.cuda.empty_cache()
                logger.debug(f"GPU: evicted {eviction_count} entries to make room for {key}")

        start = time.perf_counter()

        if self.backend == 'torch':
            if key not in self.pinned_memory:
                self.pinned_memory[key] = torch.from_numpy(data).pin_memory()
            gpu_tensor = self.pinned_memory[key].to(self.device, non_blocking=True)
            torch.cuda.synchronize()
            self.cache[key] = gpu_tensor
            del self.pinned_memory[key]
        else:
            self.cache[key] = cp.asarray(data)
            cp.cuda.Stream.null.synchronize()

        total = time.perf_counter() - start
        return StorageBackend.IOTiming(total=total, device=total, host=total)

    def read(self, key: str) -> Tuple[np.ndarray, StorageBackend.IOTiming]:
        """Reads a tensor from GPU VRAM back to a NumPy array on the CPU."""
        if key not in self.cache:
            raise KeyError(f"Key {key} not found in GPU cache")

        start = time.perf_counter()

        if self.backend == 'torch':
            gpu_tensor = self.cache[key]
            cpu_tensor = gpu_tensor.to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            data = cpu_tensor.numpy()
        else:
            data = cp.asnumpy(self.cache[key])
            cp.cuda.Stream.null.synchronize()

        total = time.perf_counter() - start
        return data, StorageBackend.IOTiming(total=total, device=total, host=total)

    def delete(self, key: str):
        if key in self.cache:
            del self.cache[key]
        if key in self.pinned_memory:
            del self.pinned_memory[key]

    def clear(self):
        """Clears all tensors from the GPU cache and frees memory."""
        for key in list(self.cache.keys()):
            del self.cache[key]
        self.cache.clear()
        for key in list(self.pinned_memory.keys()):
            del self.pinned_memory[key]
        self.pinned_memory.clear()

        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.backend == 'cupy':
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()


class CPUMemoryBackend(StorageBackend):
    """CPU RAM storage backend. This is the second tier in the cache hierarchy."""

    def __init__(self):
        self.cache = {}

    def write(self, key: str, data: np.ndarray) -> StorageBackend.IOTiming:
        """Writes data by copying it into the cache dictionary."""
        start = time.perf_counter()
        self.cache[key] = np.copy(data)
        total = time.perf_counter() - start
        return StorageBackend.IOTiming(total=total, device=total, host=total)

    def read(self, key: str) -> Tuple[np.ndarray, StorageBackend.IOTiming]:
        """Reads data by copying it from the cache dictionary."""
        if key not in self.cache:
            raise KeyError(f"Key {key} not found in CPU cache")
        start = time.perf_counter()
        data = np.copy(self.cache[key])
        total = time.perf_counter() - start
        return data, StorageBackend.IOTiming(total=total, device=total, host=total)

    def delete(self, key: str):
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        for key in list(self.cache.keys()):
            del self.cache[key]
        self.cache.clear()
        gc.collect()


class NVMeBackend(StorageBackend):
    """
    NVMe/SSD storage backend using memory-mapped files.
    This is the third and slowest tier, used for offloading from CPU RAM.
    """

    def __init__(self, base_path: str = None):
        self.temp_dir = None
        if base_path is None:
            self.temp_dir = tempfile.TemporaryDirectory(prefix="kv_cache_")
            self.base_path = Path(self.temp_dir.name)
        else:
            self.base_path = Path(base_path)
            if self.base_path.exists():
                if not self.base_path.is_dir():
                    raise NotADirectoryError(f"Cache path {self.base_path} exists but is not a directory.")
                for entry in self.base_path.glob("*.npy"):
                    try:
                        entry.unlink()
                    except OSError:
                        pass
            else:
                self.base_path.mkdir(parents=True, exist_ok=True)

        if not self.base_path.exists():
            raise OSError(f"Cache directory {self.base_path} does not exist and could not be created.")

        self.metadata = {}

    def _get_path(self, key: str) -> Path:
        """Constructs the file path for a given cache key."""
        return self.base_path / f"{key}.npy"

    def write(self, key: str, data: np.ndarray) -> StorageBackend.IOTiming:
        """Writes a NumPy array to a binary .npy file on disk."""
        start = time.perf_counter()
        path = self._get_path(key)

        with open(path, 'wb') as f:
            np.save(f, data, allow_pickle=False)
            post_save = time.perf_counter()
            f.flush()
            os.fsync(f.fileno())
            post_fsync = time.perf_counter()

        self.metadata[key] = {'shape': data.shape, 'dtype': str(data.dtype), 'size': data.nbytes}

        host_time = post_save - start
        device_time = post_fsync - post_save
        total = post_fsync - start
        return StorageBackend.IOTiming(total=total, device=device_time, host=host_time)

    def read(self, key: str) -> Tuple[np.ndarray, StorageBackend.IOTiming]:
        """Reads a .npy file from disk, dropping page cache first for accurate benchmarking."""
        start = time.perf_counter()
        path = self._get_path(key)

        if not path.exists():
            raise KeyError(f"Key {key} not found in NVMe cache")

        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, 4)  # POSIX_FADV_DONTNEED
            except AttributeError:
                pass
            finally:
                os.close(fd)
        except Exception:
            pass

        pre_load = time.perf_counter()
        data = np.load(path, allow_pickle=False)
        load_done = time.perf_counter()
        data = np.array(data)
        copy_done = time.perf_counter()

        device_time = load_done - pre_load
        host_time = (pre_load - start) + (copy_done - load_done)
        total = copy_done - start
        return data, StorageBackend.IOTiming(total=total, device=device_time, host=host_time)

    def delete(self, key: str):
        path = self._get_path(key)
        path.unlink(missing_ok=True)
        self.metadata.pop(key, None)

    def clear(self):
        """Deletes all .npy files from the cache directory."""
        for file in self.base_path.glob("*.npy"):
            file.unlink()
        self.metadata.clear()

    def __del__(self):
        """Cleans up the temporary directory when the object is destroyed."""
        if self.temp_dir:
            self.temp_dir.cleanup()
