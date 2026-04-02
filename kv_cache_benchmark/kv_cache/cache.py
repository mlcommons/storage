"""
Core multi-tier cache engine for KV Cache Benchmark.

Contains KVCacheGenerator (data generation with pre-allocated buffers)
and MultiTierCache (3-tier LRU cache with waterfall eviction).
"""

import os
import time
import hashlib
import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from kv_cache._compat import TORCH_AVAILABLE, CUPY_AVAILABLE
from kv_cache.config import cfg
from kv_cache.models import ModelConfig, InferencePhase
from kv_cache.backends import (
    StorageBackend, GPUMemoryBackend, CPUMemoryBackend, NVMeBackend, NullBackend,
)
from kv_cache.tracer import IOTracer

logger = logging.getLogger(__name__)


class KVCacheGenerator:
    """Generates realistic-looking KV cache data for testing."""

    def __init__(self, model_config: ModelConfig, global_seed: Optional[int] = None):
        self.model_config = model_config
        self.global_seed = 0 if global_seed is None else int(global_seed)

        self.buffer_size_elements = 128 * 1024 * 1024  # 128 million elements (~256MB for float16)
        self.dtype = np.float16 if 'float16' in self.model_config.dtype else np.float32

        logger.info(f"Pre-generating {self.buffer_size_elements * 2 / 1024**2:.0f} MB noise buffer...")
        rng = np.random.default_rng(self.global_seed)
        self.precomputed_buffer = rng.uniform(-1.0, 1.0, size=self.buffer_size_elements).astype(self.dtype)

    def _seed_from_key(self, key: str) -> int:
        """Derives a deterministic seed from the key string, combined with the global seed."""
        # Use SHA-256 to get a consistent hash of the key, then combine with global_seed.
        h = hashlib.sha256(key.encode('utf-8')).digest()
        # Take the first 8 bytes of the hash to form a 64-bit integer, then XOR with global_seed.
        key_hash64 = int.from_bytes(h[:8], 'little')
        # Mask to 64 bits to ensure it stays within the range of uint64.       
        return (key_hash64 ^ self.global_seed) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def _apply_xor_stamp(data: np.ndarray, seed: int) -> None:
        """XOR-stamp data IN-PLACE so every 4 KB block is unique on disk.

        Problem solved:
            All entries are sliced from the same 256 MB precomputed buffer.
            A repeating stamp fixes cross-entry dedup (different keys →
            different stamps) but NOT intra-entry dedup: a 2.5 GB entry
            reuses each buffer block ~10×, and a repeating XOR leaves
            those copies identical — measured at **95-97% dedup ratio**.

        Solution (two-layer XOR):
            1. XOR every block with a key-derived 4 KB base stamp.
               → eliminates cross-entry duplicates.
            2. XOR the first 8 bytes of each block with its block index.
               → eliminates intra-entry duplicates (same buffer content
                 at different positions becomes different on disk).

            Properties preserved:
              - Same key  → same output → reproducible  ✓
              - Diff keys → diff output → no cross-entry dedup  ✓
              - Diff positions → diff output → no intra-entry dedup  ✓

        Performance:
            Layer 1 (full XOR) is the same cost as before: ~15-20 GB/s,
            limited by memcpy bandwidth.  Layer 2 touches only 8 bytes
            per 4 KB block (0.2% of data) with one small allocation
            (8 × n_blocks bytes ≈ 5 MB per 2.5 GB entry).  Net overhead
            of Layer 2 vs. original single-layer stamp: <1%.
        """
        STAMP_BYTES = 4096  # one 4 KB disk block
        stamp_rng = np.random.default_rng(seed)
        stamp = stamp_rng.integers(0, 0xFF, size=STAMP_BYTES,
                                   dtype=np.uint8, endpoint=True)

        raw = data.view(np.uint8)
        n = raw.shape[0]

        full = (n // STAMP_BYTES) * STAMP_BYTES
        if full > 0:
            blocks = raw[:full].reshape(-1, STAMP_BYTES)

            # Layer 1: base stamp — same pattern per key (cross-entry dedup)
            blocks ^= stamp

            # Layer 2: block index in first 8 bytes (intra-entry dedup)
            # Each block gets its positional index XOR'd in, so identical
            # buffer regions at different offsets produce different disk blocks.
            # Allocation: 8 × n_blocks bytes (~5 MB per 2.5 GB entry).
            n_blocks = blocks.shape[0]
            idx_bytes = (np.arange(n_blocks, dtype=np.uint64)
                         .view(np.uint8)
                         .reshape(n_blocks, 8))
            blocks[:, :8] ^= idx_bytes

        remainder = n - full
        if remainder > 0:
            raw[full:] ^= stamp[:remainder]

    def generate(self, sequence_length: int, key: Optional[str] = None) -> np.ndarray:
        """
        Generates a NumPy array with the correct shape and dtype for a KV cache.
        Uses a pre-computed buffer to avoid CPU bottlenecks during benchmarking.

        Anti-dedup guarantee:
            Every entry is XOR-stamped with a key-derived base pattern
            (cross-entry uniqueness) AND a per-block positional index
            (intra-entry uniqueness).  This ensures no two 4 KB blocks
            are identical, even when buffer regions repeat within one
            large entry.
        """
        if self.model_config.attention_type == 'mla':
            # MLA: compressed latent (kv_lora_rank) + decoupled RoPE key (qk_rope_head_dim)
            # No separate K and V — jointly compressed into single latent vector per layer
            kv_shape = (
                self.model_config.num_layers,
                int(sequence_length),
                self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim,
            )
        else:
            kv_shape = (
                self.model_config.num_layers,
                2,
                int(sequence_length),
                self.model_config.kv_heads,
                self.model_config.kv_dim_per_head,
            )

        total_elements = int(np.prod(kv_shape))

        # Derive a deterministic seed for this entry (used for offset + XOR stamp).
        entry_seed = self._seed_from_key(key) if key else self.global_seed

        if total_elements <= self.buffer_size_elements:
            divisor = self.buffer_size_elements - total_elements
            start_idx = int(entry_seed % divisor) if divisor > 0 else 0

            # COPY (not view) so we can XOR-stamp without mutating the
            # shared precomputed buffer.  Cost: ~6 ms per 64 MB on modern
            # CPUs — negligible vs. NVMe write latency.
            data = self.precomputed_buffer[start_idx : start_idx + total_elements].copy()

            # XOR-stamp to break 4 KB block-level deduplication.
            # Zero-allocation: reshape-broadcast over a 4 KB pattern.
            self._apply_xor_stamp(data, entry_seed)

            return data.reshape(kv_shape)
        else:
            # Large entry: exceeds the 256 MB precomputed buffer.
            # Each chunk gets a unique random offset (with wraparound) so that:
            #   - Different keys produce different data  (no cross-entry dedup)
            #   - Each chunk within one entry differs    (no intra-entry dedup)
            #
            # Performance note: for a 10 GB entry this copies ~40 × 256 MB chunks.
            # Using np.concatenate of pre-sliced views would not help because the
            # wraparound means each chunk is up to 2 non-contiguous slices.
            # The bottleneck is memcpy bandwidth (~12 GB/s), not Python overhead.
            large_data = np.empty(total_elements, dtype=self.dtype)

            # Always initialise rng so there is no UnboundLocalError risk.
            # When key is None, seed=0 gives a fixed (but still varied per-chunk)
            # offset sequence; when key is provided, each key gets its own sequence.
            rng = np.random.default_rng(entry_seed)

            buf = self.precomputed_buffer          # local alias — avoids repeated attr lookup
            buf_n = self.buffer_size_elements

            pos = 0
            while pos < total_elements:
                chunk_size = min(buf_n, total_elements - pos)
                offset = int(rng.integers(0, buf_n))

                # Copy with wraparound: buffer[offset:] then buffer[:remainder]
                first_part = min(chunk_size, buf_n - offset)
                large_data[pos:pos + first_part] = buf[offset:offset + first_part]
                remaining = chunk_size - first_part
                if remaining > 0:
                    large_data[pos + first_part:pos + chunk_size] = buf[:remaining]
                pos += chunk_size

            # XOR-stamp the entire large entry to break dedup.
            # Zero-allocation: reshape-broadcast over a 4 KB pattern.
            self._apply_xor_stamp(large_data, entry_seed)

            return large_data.reshape(kv_shape)


# ============================================================================
# ENHANCED MULTI-TIER CACHE
# ============================================================================

class MultiTierCache:
    """
    Manages KV cache data across GPU, CPU, and NVMe tiers.

    This class is the heart of the benchmark. It orchestrates where cache data is
    written to and read from based on available space and access patterns.
    """

    def __init__(self,
                 model_config: ModelConfig,
                 gpu_memory_gb: float,
                 cpu_memory_gb: float,
                 cache_dir: str = None,
                 eviction_policy: str = 'lru',
                 performance_profile: str = 'latency',
                 seed: Optional[int] = None,
                 max_concurrent_allocs: int = 0,
                 storage_capacity_gb: float = 0,
                 tensor_parallel: int = 1,
                 io_tracer: Optional['IOTracer'] = None):

        self.model_config = model_config
        self.gpu_memory_limit = gpu_memory_gb * 1024**3
        self.cpu_memory_limit = cpu_memory_gb * 1024**3
        self.eviction_policy = eviction_policy
        self.performance_profile = performance_profile
        self.seed = seed
        self.max_concurrent_allocs = max_concurrent_allocs
        self.tensor_parallel = max(1, tensor_parallel)
        self.io_tracer = io_tracer

        # Initialize storage backends for each tier.
        # In trace mode all backends are NullBackend — no real hardware I/O.
        self.backends = {}
        if self.io_tracer is not None:
            logger.info("MultiTierCache: trace mode active — using NullBackend for all tiers")
            self.backends['gpu'] = NullBackend()
            self.backends['cpu'] = NullBackend()
            self.backends['nvme'] = NullBackend()
        else:
            try:
                if TORCH_AVAILABLE or CUPY_AVAILABLE:
                    self.backends['gpu'] = GPUMemoryBackend(
                        use_torch=TORCH_AVAILABLE,
                        on_eviction_callback=self._handle_gpu_eviction
                    )
            except Exception as e:
                logger.warning(f"Could not initialize GPU backend: {e}")

            self.backends['cpu'] = CPUMemoryBackend()
            self.backends['nvme'] = NVMeBackend(base_path=cache_dir)

        self.generator = KVCacheGenerator(model_config, global_seed=self.seed)

        self.cache_entries = {}
        self.entry_locks: Dict[str, threading.Lock] = {}
        if storage_capacity_gb > 0:
            self.nvme_memory_limit = storage_capacity_gb * 1024**3
        else:
            try:
                nvme_base = self.backends['nvme'].base_path
                st = os.statvfs(str(nvme_base))
                self.nvme_memory_limit = float(st.f_bavail * st.f_frsize) * 0.95
            except Exception:
                self.nvme_memory_limit = float('inf')

        self.gpu_memory_used = 0
        self.cpu_memory_used = 0
        self.nvme_memory_used = 0

        self.metadata_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        self.stats_lock = threading.Lock()

        if self.max_concurrent_allocs and self.max_concurrent_allocs > 0:
            self.allocation_semaphore = threading.Semaphore(self.max_concurrent_allocs)
        else:
            self.allocation_semaphore = None

        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'offloads_cpu': 0,
            'offloads_storage': 0,

            'gpu_read_latencies': [], 'cpu_read_latencies': [], 'storage_read_latencies': [],
            'gpu_write_latencies': [], 'cpu_write_latencies': [], 'storage_write_latencies': [],
            'storage_read_device_latencies': [], 'storage_read_host_latencies': [],
            'storage_write_device_latencies': [], 'storage_write_host_latencies': [],

            'prefill_writes': 0, 'decode_reads': 0,

            'tier_gpu_kv_bytes_written': 0, 'tier_cpu_kv_bytes_written': 0, 'tier_storage_kv_bytes_written': 0,
            'tier_gpu_kv_bytes_read': 0, 'tier_cpu_kv_bytes_read': 0, 'tier_storage_kv_bytes_read': 0,

            'system_prompt_hits': 0, 'common_phrase_hits': 0,
            'user_cache_hits': 0, 'multi_turn_hits': 0,

            'total_read_bytes': 0, 'total_write_bytes': 0,
            'read_operations': 0, 'write_operations': 0,

            'storage_tokens_processed': 0,
        }

    def _get_entry_lock(self, key: str) -> threading.Lock:
        """Get or create a lock for a specific cache entry."""
        with self.metadata_lock:
            if key not in self.entry_locks:
                self.entry_locks[key] = threading.Lock()
            return self.entry_locks[key]

    def _handle_gpu_eviction(self, key: str, tier: str, evicted_size: int) -> None:
        """Callback invoked by GPUMemoryBackend when it evicts entries during OOM handling."""
        with self.metadata_lock:
            if key in self.cache_entries:
                del self.cache_entries[key]
            if key in self.entry_locks:
                del self.entry_locks[key]

        with self.memory_lock:
            self.gpu_memory_used = max(0, self.gpu_memory_used - evicted_size)

        with self.stats_lock:
            self.stats['evictions'] += 1

        logger.debug(f"GPU eviction synced: removed {key} from cache metadata")

    # ========================================================================
    # WATERFALL LRU EVICTION METHODS
    # ========================================================================

    def _get_tier_order(self) -> List[str]:
        """Returns the tier hierarchy from fastest to slowest."""
        tiers = []
        if 'gpu' in self.backends:
            tiers.append('gpu')
        tiers.extend(['cpu', 'nvme'])
        return tiers

    def _get_tier_limit(self, tier: str) -> float:
        """Get the memory limit for a tier in bytes."""
        if tier == 'gpu':
            return self.gpu_memory_limit
        elif tier == 'cpu':
            return self.cpu_memory_limit
        else:
            return self.nvme_memory_limit

    def _get_tier_usage(self, tier: str) -> float:
        """Get the current memory usage for a tier in bytes."""
        if tier == 'gpu':
            return self.gpu_memory_used
        elif tier == 'cpu':
            return self.cpu_memory_used
        else:
            return self.nvme_memory_used

    def _update_tier_usage(self, tier: str, delta: int):
        """Update the memory usage tracking for a tier."""
        if tier == 'gpu':
            self.gpu_memory_used = max(0, self.gpu_memory_used + delta)
        elif tier == 'cpu':
            self.cpu_memory_used = max(0, self.cpu_memory_used + delta)
        elif tier == 'nvme':
            self.nvme_memory_used = max(0, self.nvme_memory_used + delta)

    def _get_lru_entries_in_tier(self, tier: str) -> List[Tuple[str, dict]]:
        """Get all cache entries in a specific tier, sorted by LRU order."""
        with self.metadata_lock:
            entries = [
                (k, dict(v))
                for k, v in self.cache_entries.items()
                if v['location'] == tier
            ]
        entries.sort(key=lambda x: (x[1]['last_access'], x[1].get('access_count', 0)))
        return entries

    def _demote_entry(self, key: str, from_tier: str, to_tier: str) -> Tuple[bool, float]:
        """Move a cache entry from one tier to a lower (slower) tier."""
        entry_lock = self._get_entry_lock(key)

        with entry_lock:
            with self.metadata_lock:
                if key not in self.cache_entries:
                    return False, 0.0
                entry = self.cache_entries[key]
                current_location = entry['location']
                if current_location != from_tier:
                    return True, 0.0
                size = entry['size']

            try:
                data, read_timing = self.backends[from_tier].read(key)
                write_timing = self.backends[to_tier].write(key, data)
                self.backends[from_tier].delete(key)

                if self.io_tracer is not None:
                    self.io_tracer.log('Read',  size, from_tier, key=key, phase='Evict')
                    self.io_tracer.log('Write', size, to_tier,   key=key, phase='Evict')

                with self.metadata_lock:
                    if key in self.cache_entries:
                        self.cache_entries[key]['location'] = to_tier

                with self.memory_lock:
                    self._update_tier_usage(from_tier, -size)

                with self.stats_lock:
                    self.stats['evictions'] += 1
                    if to_tier == 'cpu':
                        self.stats['offloads_cpu'] += 1
                    elif to_tier == 'nvme':
                        self.stats['offloads_storage'] += 1
                        bytes_per_token = (self.model_config.kv_cache_size_per_token
                                          // max(1, self.tensor_parallel))
                        if bytes_per_token > 0:
                            tokens = size // bytes_per_token
                            self.stats['storage_tokens_processed'] += tokens
                        else:
                            logger.warning("bytes_per_token is 0, skipping token count update")

                total_latency = read_timing.total + write_timing.total
                return True, total_latency

            except Exception as e:
                logger.error(f"Failed to demote {key} from {from_tier} to {to_tier}: {e}")
                return False, 0.0

    def _ensure_space_in_tier(self, tier: str, required_bytes: int, recursion_depth: int = 0) -> bool:
        """Ensure there's enough space in a tier by evicting LRU entries."""
        if tier == 'nvme' and self.nvme_memory_limit == float('inf'):
            # Still track usage even when unlimited, for accurate metrics
            with self.memory_lock:
                self._update_tier_usage('nvme', required_bytes)
            return True

        max_recursion = cfg('eviction', 'max_recursion_depth', default=10)
        if recursion_depth > max_recursion:
            logger.warning("Hit recursion limit in _ensure_space_in_tier")
            return False

        tier_order = self._get_tier_order()
        try:
            tier_idx = tier_order.index(tier)
        except ValueError:
            return False

        next_tier = tier_order[tier_idx + 1] if tier_idx + 1 < len(tier_order) else None
        if next_tier is None and tier != 'nvme':
            return False

        # When NVMe is the terminal tier (no tier after it), the entry MUST
        # be written here — relax capacity guards and evict to full limit.
        is_last_tier = (next_tier is None)

        limit = self._get_tier_limit(tier)
        target_usage_ratio = cfg('eviction', 'target_usage_ratio', default=0.8)
        target_usage = limit * target_usage_ratio

        large_entry_limit_ratio = cfg('eviction', 'large_entry_limit_ratio', default=0.95)
        # Only reject oversized entries on non-terminal tiers (they can cascade).
        # On the last tier, we must accommodate the entry regardless of size.
        if not is_last_tier and required_bytes > limit * large_entry_limit_ratio:
            return False

        # On the last tier, evict to full capacity (not 80%) since there's
        # no next tier that needs a buffer for cascading entries.
        effective_target = limit if is_last_tier else target_usage

        # ────────────────────────────────────────────────────────────────
        # SNAPSHOT-BASED LRU EVICTION
        #
        # Performance context:
        #   _get_lru_entries_in_tier() copies every entry in cache_entries
        #   that belongs to this tier, then sorts by last_access time.
        #   At 15 TB with 60k entries, that's ~60k dict copies + sort.
        #
        # Old behavior (O(n²)):
        #   The while loop called _get_lru_entries_in_tier() on EVERY
        #   iteration, but only used lru_entries[0] — the single oldest
        #   entry.  Evicting 100 entries meant 100 full scans.
        #
        # New behavior (O(n)):
        #   Take ONE sorted snapshot before the loop.  Walk through it
        #   with an index.  Each entry is either:
        #     - Still valid → evict it (delete or demote)
        #     - Already gone (another thread got it) → skip, advance index
        #   If we exhaust the snapshot without freeing enough space,
        #   refresh it ONCE (new entries may have been written since the
        #   snapshot).  Worst case: 2 scans instead of thousands.
        #
        # Why stale snapshots are safe:
        #   - DELETE path: the existence check under metadata_lock already
        #     skips entries that another thread evicted.  A stale snapshot
        #     just means we hit more skips — no double-decrement.
        #   - DEMOTE path: _demote_entry() checks that the entry still
        #     exists in from_tier before moving it.  If it's gone, it
        #     returns False and we advance to the next entry.
        #   - New entries added after the snapshot are NEWER than
        #     everything in it (higher last_access time), so LRU order
        #     says evict them last.  Not including them is correct.
        #
        # Impact on MLPerf metrics:
        #   Storage device latencies (write_device_p50, read_device_p50)
        #   are timed INSIDE the backend — after eviction has already
        #   freed space.  This optimization only reduces the untimed CPU
        #   overhead between I/O operations.  Throughput (req/s) improves
        #   because the benchmark can push I/O faster; device-level
        #   numbers stay the same.
        # ────────────────────────────────────────────────────────────────

        lru_entries = self._get_lru_entries_in_tier(tier)
        lru_idx = 0

        max_evictions_hard_cap = cfg('eviction', 'max_evictions_hard_cap', default=5000)
        max_evictions_min = cfg('eviction', 'max_evictions_min', default=1000)
        max_evictions_per_call = min(max_evictions_hard_cap, max(max_evictions_min, len(lru_entries) + 100))
        eviction_count = 0

        while eviction_count < max_evictions_per_call:
            # ── Check 1: Is there already enough space? ──
            with self.memory_lock:
                current_usage = self._get_tier_usage(tier)
                if current_usage + required_bytes <= effective_target:
                    self._update_tier_usage(tier, required_bytes)
                    return True

                # Near-empty tier: usage tracking may have drifted from
                # accumulated rounding.  Trust it and allow the write.
                if current_usage < limit * 0.05:
                    self._update_tier_usage(tier, required_bytes)
                    return True

            # ── Check 2: Advance through the LRU snapshot ──
            # If we've walked past the end of the snapshot, try one
            # refresh — concurrent threads may have evicted most of our
            # snapshot, or new entries may have landed in this tier.
            if lru_idx >= len(lru_entries):
                lru_entries = self._get_lru_entries_in_tier(tier)
                lru_idx = 0

                if not lru_entries:
                    # Tier is truly empty.  Recount actual usage from
                    # cache_entries to correct any drift, then decide.
                    with self.metadata_lock:
                        actual_usage = sum(
                            entry['size'] for entry in self.cache_entries.values()
                            if entry['location'] == tier
                        )
                    with self.memory_lock:
                        if tier == 'gpu':
                            self.gpu_memory_used = actual_usage
                        elif tier == 'cpu':
                            self.cpu_memory_used = actual_usage
                        elif tier == 'nvme':
                            self.nvme_memory_used = actual_usage

                    with self.memory_lock:
                        current_usage = self._get_tier_usage(tier)
                        if current_usage + required_bytes <= effective_target:
                            self._update_tier_usage(tier, required_bytes)
                            return True

                    # Last tier with nothing left to evict — allow the
                    # write and let the OS enforce disk space.
                    if is_last_tier:
                        with self.memory_lock:
                            self._update_tier_usage(tier, required_bytes)
                        return True

                    return False

            # On non-terminal tiers, bail out if there's little data to
            # evict relative to what we need.  On the last tier, keep
            # going — there's nowhere else to send the entry.
            # (Only check on first pass through the snapshot to avoid
            # re-summing on every iteration.)
            if lru_idx == 0 and not is_last_tier:
                total_size_in_tier = sum(e['size'] for _, e in lru_entries)
                if total_size_in_tier < limit * 0.2 and required_bytes > target_usage * 0.5:
                    return False

            # ── Pick the next LRU entry from the snapshot ──
            lru_key, lru_entry = lru_entries[lru_idx]
            lru_size = lru_entry['size']
            lru_idx += 1

            # ── Evict: DELETE (terminal tier) or DEMOTE (non-terminal) ──
            if next_tier is None and tier == 'nvme':
                # Terminal tier: delete the .npy file from disk.
                # The existence check prevents double-decrementing when
                # multiple threads race on the same stale snapshot entry.
                entry_lock = self._get_entry_lock(lru_key)
                with entry_lock:
                    with self.metadata_lock:
                        existing = self.cache_entries.get(lru_key)
                        if existing is None or existing['location'] != 'nvme':
                            # Another thread already evicted this entry.
                            # Safe to skip — just advance to the next one.
                            eviction_count += 1
                            continue
                        actual_size = existing['size']
                        del self.cache_entries[lru_key]
                        self.entry_locks.pop(lru_key, None)
                    try:
                        self.backends['nvme'].delete(lru_key)
                    except Exception as e:
                        logger.warning(f"Failed to delete NVMe entry {lru_key}: {e}")
                    with self.memory_lock:
                        self.nvme_memory_used = max(0, self.nvme_memory_used - actual_size)
                with self.stats_lock:
                    self.stats['evictions'] += 1
            else:
                # Non-terminal tier: demote entry to the next tier down.
                # Recursively ensure space in next_tier first.
                if not self._ensure_space_in_tier(next_tier, lru_size, recursion_depth + 1):
                    logger.warning(f"Could not make space in {next_tier} for demotion")
                    return False

                success, _ = self._demote_entry(lru_key, tier, next_tier)
                if not success:
                    # Entry was deleted/moved by another thread between
                    # the snapshot and now.  Skip to the next one.
                    eviction_count += 1
                    continue

            eviction_count += 1

        # Exhausted eviction budget.  On the last tier, allow the write
        # anyway — we've freed as much as we can.
        if is_last_tier:
            with self.memory_lock:
                self._update_tier_usage(tier, required_bytes)
            return True

        return False

    def allocate_cache(self, key: str, num_tokens: int, phase: InferencePhase = InferencePhase.PREFILL) -> Tuple[bool, str, float]:
        """Allocates and writes a new KV cache entry to the most appropriate tier."""
        with self.metadata_lock:
            if key in self.cache_entries:
                return True, self.cache_entries[key]['location'], 0.0

        if self.allocation_semaphore:
            self.allocation_semaphore.acquire()

        try:
            return self._allocate_cache_inner(key, num_tokens, phase)
        finally:
            if self.allocation_semaphore:
                self.allocation_semaphore.release()

    def _allocate_cache_inner(self, key: str, num_tokens: int, phase: InferencePhase) -> Tuple[bool, str, float]:
        """Inner implementation of allocate_cache, called within semaphore."""
        if self.io_tracer is not None:
            # Trace mode: compute size from model config — no numpy allocation needed.
            # Divide by tensor_parallel: each TP rank stores only its 1/TP shard.
            size_bytes = (self.model_config.kv_cache_size_per_token * num_tokens
                          ) // self.tensor_parallel
            data = None
        else:
            try:
                data = self.generator.generate(sequence_length=num_tokens, key=key)
            except MemoryError:
                logger.error(f"MemoryError generating cache for key {key} ({num_tokens} tokens)")
                return False, 'none', 0.0
            except Exception as exc:
                logger.error(f"Failed to generate cache for key {key}: {exc}")
                return False, 'none', 0.0
            if self.tensor_parallel > 1:
                # Each TP rank owns 1/tensor_parallel of the KV heads.
                # Take the first shard of the flat buffer as this rank's share.
                tp_elements = data.size // self.tensor_parallel
                data = data.ravel()[:tp_elements]
            size_bytes = data.nbytes

        with self.stats_lock:
            if phase == InferencePhase.PREFILL:
                self.stats['prefill_writes'] += 1
            self.stats['write_operations'] += 1
            self.stats['total_write_bytes'] += size_bytes

        tier_order = self._get_tier_order()
        allocated_tier = None

        for tier in tier_order:
            if self._ensure_space_in_tier(tier, size_bytes):
                allocated_tier = tier
                break

        if allocated_tier is None:
            logger.warning("All tiers full — eviction could not free space, forcing write to NVMe")
            allocated_tier = 'nvme'
            with self.memory_lock:
                self._update_tier_usage('nvme', size_bytes)

        try:
            if self.io_tracer is not None:
                # Trace mode: record the operation with no actual data movement
                timing = self.backends[allocated_tier].write_size(key, size_bytes)
                self.io_tracer.log('Write', size_bytes, allocated_tier,
                                   key=key, phase=phase.value.capitalize())
            elif allocated_tier == 'gpu':
                timing = self.backends['gpu'].write(key, data)
            elif allocated_tier == 'cpu':
                timing = self.backends['cpu'].write(key, data)
            else:
                timing = self.backends['nvme'].write(key, data)

            with self.metadata_lock:
                self.cache_entries[key] = {
                    'location': allocated_tier,
                    'size': size_bytes,
                    'last_access': time.time(),
                    'access_count': 1
                }

            with self.stats_lock:
                tier_stats_name = 'storage' if allocated_tier == 'nvme' else allocated_tier

                self.stats[f'tier_{tier_stats_name}_kv_bytes_written'] += size_bytes

                if allocated_tier == 'cpu':
                    self.stats['offloads_cpu'] += 1
                    self.stats['cpu_write_latencies'].append(timing.total)
                elif allocated_tier == 'nvme':
                    self.stats['offloads_storage'] += 1
                    self.stats['storage_write_latencies'].append(timing.total)
                    self.stats['storage_write_device_latencies'].append(timing.device)
                    self.stats['storage_write_host_latencies'].append(timing.host)
                    self.stats['storage_tokens_processed'] += num_tokens
                elif allocated_tier == 'gpu':
                    self.stats['gpu_write_latencies'].append(timing.total)

            del data
            return True, allocated_tier, timing.total

        except Exception as e:
            with self.memory_lock:
                self._update_tier_usage(allocated_tier, -size_bytes)
            del data
            return False, 'none', 0.0

    def check_cache_exists(self, key: str) -> Tuple[Optional[str], int]:
        """Metadata-only existence check. No I/O, no LRU update.

        Used by multi-turn virtual context checks: determines whether a
        previous turn's KV cache entry survived LRU eviction without
        loading data from disk (no np.load, no memory allocation).

        Returns:
            (location, size_bytes) if the entry exists, (None, 0) otherwise.
        """
        with self.metadata_lock:
            entry = self.cache_entries.get(key)
            if entry is None:
                return None, 0
            return entry['location'], entry['size']

    def access_cache(self, key: str, phase: InferencePhase = InferencePhase.DECODE,
                     cache_type: str = 'user') -> Tuple[Optional[str], float]:
        """Accesses an existing cached entry and records the read performance."""
        with self.metadata_lock:
            if key not in self.cache_entries:
                with self.stats_lock:
                    self.stats['cache_misses'] += 1
                return None, 0.0

         #   try:
            entry = self.cache_entries[key]
            location = entry['location']
            entry_size = entry['size']
         #   except KeyError:
         #      with self.stats_lock:
         #         self.stats['cache_misses'] += 1
         #    return None, 0.0

        entry_lock = self._get_entry_lock(key)

        with entry_lock:
            with self.metadata_lock:
                if key not in self.cache_entries:
                    with self.stats_lock:
                        self.stats['cache_misses'] += 1
                    return None, 0.0

                entry = self.cache_entries[key]
                entry['last_access'] = time.time()
                entry['access_count'] += 1

            with self.stats_lock:
                self.stats['cache_hits'] += 1

                if cache_type == 'system': self.stats['system_prompt_hits'] += 1
                elif cache_type == 'common': self.stats['common_phrase_hits'] += 1
                elif cache_type == 'multi_turn': self.stats['multi_turn_hits'] += 1
                else: self.stats['user_cache_hits'] += 1

                tier_stats_name = 'storage' if location == 'nvme' else location

                self.stats[f'tier_{tier_stats_name}_kv_bytes_read'] += entry_size

                if phase == InferencePhase.DECODE:
                    self.stats['decode_reads'] += 1

                self.stats['read_operations'] += 1
                self.stats['total_read_bytes'] += entry_size

            try:
                _, timing = self.backends[location].read(key)

                if self.io_tracer is not None:
                    self.io_tracer.log('Read', entry_size, location,
                                       key=key, phase=phase.value.capitalize())

                with self.stats_lock:
                    if location == 'gpu':
                        self.stats['gpu_read_latencies'].append(timing.total)
                    elif location == 'cpu':
                        self.stats['cpu_read_latencies'].append(timing.total)
                    else:
                        self.stats['storage_read_latencies'].append(timing.total)
                        self.stats['storage_read_device_latencies'].append(timing.device)
                        self.stats['storage_read_host_latencies'].append(timing.host)

                        if self.model_config.kv_cache_size_per_token > 0:
                            num_tokens = entry_size / self.model_config.kv_cache_size_per_token
                            self.stats['storage_tokens_processed'] += num_tokens

                return location, timing.total
            except Exception as e:
                return location, 0.0

    def _evaluate_storage_performance(self, duration: float) -> Dict:
        """Evaluates storage performance against MLPerf Storage WG criteria."""
        criteria = []
        all_passed = True

        if self.performance_profile == 'throughput':
            read_bytes = self.stats.get('tier_storage_kv_bytes_read', 0)
            write_bytes = self.stats.get('tier_storage_kv_bytes_written', 0)
            read_bw_gbps = (read_bytes / 1024**3) / duration if duration > 0 else 0
            write_bw_gbps = (write_bytes / 1024**3) / duration if duration > 0 else 0

            # Only check read bandwidth if there were reads (skip for prefill-only mode)
            if read_bytes > 0 or write_bytes == 0:
                read_passed = read_bw_gbps > 0
                criteria.append({
                    'name': 'Storage KV Read Bandwidth',
                    'target': '>0', 'actual': f"{read_bw_gbps:.2f}", 'unit': 'GB/s', 'passed': read_passed
                })
                all_passed = all_passed and read_passed

            # Only check write bandwidth if there were writes (skip for decode-only mode)
            if write_bytes > 0 or read_bytes == 0:
                write_passed = write_bw_gbps > 0
                criteria.append({
                    'name': 'Storage KV Write Bandwidth',
                    'target': '>0', 'actual': f"{write_bw_gbps:.2f}", 'unit': 'GB/s', 'passed': write_passed
                })
                all_passed = all_passed and write_passed

            return {
                'overall_status': 'PASS' if all_passed else 'FAIL',
                'criteria': criteria,
                'passed_count': sum(1 for c in criteria if c['passed']),
                'total_count': len(criteria)
            }

        # Latency-focused profile (default)
        storage_write_device = self.stats.get('storage_write_device_latencies', [])
        storage_write_total = self.stats.get('storage_write_latencies', [])
        storage_write_basis = storage_write_device if storage_write_device else storage_write_total
        latency_type = 'Device' if storage_write_device else 'Total'
        if storage_write_basis:
            storage_write_p95 = np.percentile(storage_write_basis, 95) * 1000
            passed = storage_write_p95 < 500
            criteria.append({
                'name': f'Storage Tier Write {latency_type} P95 < 500ms',
                'target': 500, 'actual': storage_write_p95, 'unit': 'ms', 'passed': passed
            })
            all_passed = all_passed and passed

        storage_read_device = self.stats.get('storage_read_device_latencies', [])
        storage_read_total = self.stats.get('storage_read_latencies', [])
        storage_read_basis = storage_read_device if storage_read_device else storage_read_total
        latency_type = 'Device' if storage_read_device else 'Total'
        if storage_read_basis:
            storage_read_p95 = np.percentile(storage_read_basis, 95) * 1000
            passed = storage_read_p95 < 200
            criteria.append({
                'name': f'Storage Tier Read {latency_type} P95 < 200ms',
                'target': 200, 'actual': storage_read_p95, 'unit': 'ms', 'passed': passed
            })
            all_passed = all_passed and passed

        cpu_read_lats = self.stats.get('cpu_read_latencies', [])
        cpu_write_lats = self.stats.get('cpu_write_latencies', [])
        if cpu_read_lats or cpu_write_lats:
            all_cpu_lats = cpu_read_lats + cpu_write_lats
            cpu_p95 = np.percentile(all_cpu_lats, 95) * 1000
            passed = cpu_p95 < 150
            criteria.append({
                'name': 'CPU RAM P95 < 150ms',
                'target': 150, 'actual': cpu_p95, 'unit': 'ms', 'passed': passed
            })
            all_passed = all_passed and passed

        total_accesses = self.stats['cache_hits'] + self.stats['cache_misses']
        if total_accesses > 0:
            hit_rate = self.stats['cache_hits'] / total_accesses
            passed = hit_rate > 0.3
            criteria.append({
                'name': 'Cache Hit Rate > 30%',
                'target': 0.3, 'actual': hit_rate, 'unit': 'ratio', 'passed': passed
            })
            all_passed = all_passed and passed

        return {
            'overall_status': 'PASS' if all_passed else 'FAIL',
            'criteria': criteria,
            'passed_count': sum(1 for c in criteria if c['passed']),
            'total_count': len(criteria)
        }

    def get_stats(self, duration: float) -> Dict:
        """Gathers and returns a comprehensive dictionary of all performance statistics."""
        with self.stats_lock:
            total_accesses = self.stats['cache_hits'] + self.stats['cache_misses']
            hit_rate = self.stats['cache_hits'] / total_accesses if total_accesses > 0 else 0
            stats_snapshot = self.stats.copy()

        with self.metadata_lock:
            gpu_entries = sum(1 for e in self.cache_entries.values() if e['location'] == 'gpu')
            cpu_entries = sum(1 for e in self.cache_entries.values() if e['location'] == 'cpu')
            nvme_entries = sum(1 for e in self.cache_entries.values() if e['location'] == 'nvme')

        with self.memory_lock:
            gpu_mem_used = self.gpu_memory_used
            cpu_mem_used = self.cpu_memory_used

        storage_health = self._evaluate_storage_performance(duration)

        tier_gpu_read_bytes = self.stats['tier_gpu_kv_bytes_read']
        tier_gpu_write_bytes = self.stats['tier_gpu_kv_bytes_written']
        tier_cpu_read_bytes = self.stats['tier_cpu_kv_bytes_read']
        tier_cpu_write_bytes = self.stats['tier_cpu_kv_bytes_written']
        tier_storage_read_bytes = self.stats['tier_storage_kv_bytes_read']
        tier_storage_write_bytes = self.stats['tier_storage_kv_bytes_written']

        stats = {
            'cache_hit_rate': hit_rate,
            'cache_hits': stats_snapshot['cache_hits'],
            'cache_misses': stats_snapshot['cache_misses'],
            'gpu_entries': gpu_entries,
            'cpu_entries': cpu_entries,
            'storage_entries': nvme_entries,
            'gpu_memory_used_gb': gpu_mem_used / 1024**3,
            'cpu_memory_used_gb': cpu_mem_used / 1024**3,
            'offloads_cpu': stats_snapshot['offloads_cpu'],
            'offloads_storage': stats_snapshot['offloads_storage'],
            'storage_health': storage_health,
            'prefill_writes': self.stats['prefill_writes'],
            'decode_reads': self.stats['decode_reads'],

            'tier_gpu_kv_bytes_written_gb': tier_gpu_write_bytes / 1024**3,
            'tier_cpu_kv_bytes_written_gb': tier_cpu_write_bytes / 1024**3,
            'tier_storage_kv_bytes_written_gb': tier_storage_write_bytes / 1024**3,
            'tier_gpu_kv_bytes_read_gb': tier_gpu_read_bytes / 1024**3,
            'tier_cpu_kv_bytes_read_gb': tier_cpu_read_bytes / 1024**3,
            'tier_storage_kv_bytes_read_gb': tier_storage_read_bytes / 1024**3,

            'tier_gpu_read_bandwidth_gbps': (tier_gpu_read_bytes / 1024**3) / duration if duration > 0 else 0,
            'tier_gpu_write_bandwidth_gbps': (tier_gpu_write_bytes / 1024**3) / duration if duration > 0 else 0,
            'tier_cpu_read_bandwidth_gbps': (tier_cpu_read_bytes / 1024**3) / duration if duration > 0 else 0,
            'tier_cpu_write_bandwidth_gbps': (tier_cpu_write_bytes / 1024**3) / duration if duration > 0 else 0,
            'tier_storage_read_bandwidth_gbps': (tier_storage_read_bytes / 1024**3) / duration if duration > 0 else 0,
            'tier_storage_write_bandwidth_gbps': (tier_storage_write_bytes / 1024**3) / duration if duration > 0 else 0,

            'system_prompt_hits': self.stats['system_prompt_hits'],
            'common_phrase_hits': self.stats['common_phrase_hits'],
            'user_cache_hits': self.stats['user_cache_hits'],
            'multi_turn_hits': self.stats['multi_turn_hits'],
            'total_read_bytes': self.stats['total_read_bytes'],
            'total_write_bytes': self.stats['total_write_bytes'],
            'total_read_gb': self.stats['total_read_bytes'] / 1024**3,
            'total_write_gb': self.stats['total_write_bytes'] / 1024**3,
            'read_write_ratio': self.stats['total_read_bytes'] / max(self.stats['total_write_bytes'], 1),
            'read_iops': self.stats['read_operations'],
            'write_iops': self.stats['write_operations'],
            'storage_tokens_processed': self.stats['storage_tokens_processed'],
        }

       # tier_mapping = {'gpu': 'gpu', 'cpu': 'cpu', 'nvme': 'storage'}
        for internal_tier, output_tier in [('gpu', 'gpu'), ('cpu', 'cpu'), ('storage', 'storage')]:
            for op in ['read', 'write']:
                latencies = self.stats.get(f'{internal_tier}_{op}_latencies', [])
                if latencies:
                    lat_array = np.array(latencies)
                    stats[f'{output_tier}_{op}_p50_ms'] = np.percentile(lat_array, 50) * 1000
                    stats[f'{output_tier}_{op}_p95_ms'] = np.percentile(lat_array, 95) * 1000
                    stats[f'{output_tier}_{op}_p99_ms'] = np.percentile(lat_array, 99) * 1000
                    stats[f'{output_tier}_{op}_p999_ms'] = np.percentile(lat_array, 99.9) * 1000
                    stats[f'{output_tier}_{op}_p9999_ms'] = np.percentile(lat_array, 99.99) * 1000

        for op in ['read', 'write']:
            device_latencies = self.stats.get(f'storage_{op}_device_latencies', [])
            host_latencies = self.stats.get(f'storage_{op}_host_latencies', [])
            if device_latencies:
                device_array = np.array(device_latencies)
                stats[f'storage_{op}_device_p50_ms'] = np.percentile(device_array, 50) * 1000
                stats[f'storage_{op}_device_p95_ms'] = np.percentile(device_array, 95) * 1000
                stats[f'storage_{op}_device_p99_ms'] = np.percentile(device_array, 99) * 1000
                stats[f'storage_{op}_device_p999_ms'] = np.percentile(device_array, 99.9) * 1000
                stats[f'storage_{op}_device_p9999_ms'] = np.percentile(device_array, 99.99) * 1000
            if host_latencies:
                host_array = np.array(host_latencies)
                stats[f'storage_{op}_host_p50_ms'] = np.percentile(host_array, 50) * 1000
                stats[f'storage_{op}_host_p95_ms'] = np.percentile(host_array, 95) * 1000
                stats[f'storage_{op}_host_p99_ms'] = np.percentile(host_array, 99) * 1000
                stats[f'storage_{op}_host_p999_ms'] = np.percentile(host_array, 99.9) * 1000
                stats[f'storage_{op}_host_p9999_ms'] = np.percentile(host_array, 99.99) * 1000

        return stats

    def reset_stats(self):
        """Reset all performance counters (used after preconditioning)."""
        with self.stats_lock:
            for key, value in self.stats.items():
                if isinstance(value, list):
                    self.stats[key] = []
                elif isinstance(value, (int, float)):
                    self.stats[key] = 0
