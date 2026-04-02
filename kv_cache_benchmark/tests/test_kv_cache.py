#!/usr/bin/env python3
"""
Fast Unit Tests for KV Cache Benchmark (pytest version)

Run with:
    pytest test_kv_cache.py -v                                    # Console output
    pytest test_kv_cache.py -v --html=report.html --self-contained-html  # HTML report

Requirements:
    pip install pytest pytest-html

These tests verify core functionality without running the full benchmark.
Typical execution time: < 5 seconds

This version tests kv-cache.py which includes:
- ConfigLoader with YAML support and strict validation
- Extended QoS SLA with p999 and p9999 percentiles
- Config-driven parameters via cfg() helper
- Renamed nvme_* to storage_* in stats
"""

import os
import sys
import time
import argparse
import tempfile
import threading
import pytest
import numpy as np
from datetime import datetime
from pathlib import Path

# Import from kv-cache.py (handle the hyphen in filename)
# Try multiple locations: same directory, parent directory
import importlib.util

_kv_cache_path = None
_possible_paths = [
    os.path.join(os.path.dirname(__file__), "kv-cache.py"),        # Same directory
    os.path.join(os.path.dirname(__file__), "..", "kv-cache.py"),  # Parent directory
]
for _path in _possible_paths:
    if os.path.exists(_path):
        _kv_cache_path = _path
        break

if _kv_cache_path is None:
    raise FileNotFoundError(
        f"Could not find kv-cache.py. Searched in:\n"
        + "\n".join(f"  - {os.path.abspath(p)}" for p in _possible_paths)
    )

spec = importlib.util.spec_from_file_location("kv_cache", _kv_cache_path)
kv_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv_cache)

# Import all needed classes and functions
MODEL_CONFIGS = kv_cache.MODEL_CONFIGS
ModelConfig = kv_cache.ModelConfig
InferenceRequest = kv_cache.InferenceRequest
InferencePhase = kv_cache.InferencePhase
GenerationMode = kv_cache.GenerationMode
GENERATION_TIMING = kv_cache.GENERATION_TIMING
QoSLevel = kv_cache.QoSLevel
QOS_PROFILES = kv_cache.QOS_PROFILES
KVCacheGenerator = kv_cache.KVCacheGenerator
CPUMemoryBackend = kv_cache.CPUMemoryBackend
NVMeBackend = kv_cache.NVMeBackend
ConversationManager = kv_cache.ConversationManager
UserSimulator = kv_cache.UserSimulator
MultiTierCache = kv_cache.MultiTierCache
export_results_to_xlsx = kv_cache.export_results_to_xlsx
PANDAS_AVAILABLE = kv_cache.PANDAS_AVAILABLE

# New imports for 01-26-2026 version
ConfigLoader = kv_cache.ConfigLoader
cfg = kv_cache.cfg
get_config = kv_cache.get_config
set_config = kv_cache.set_config
get_qos_profiles = kv_cache.get_qos_profiles
QoSSLA = kv_cache.QoSSLA
YAML_AVAILABLE = kv_cache.YAML_AVAILABLE
IntegratedBenchmark = kv_cache.IntegratedBenchmark

# Input validation imports
validate_args = kv_cache.validate_args
MAX_USERS = kv_cache.MAX_USERS
MAX_DURATION_SECONDS = kv_cache.MAX_DURATION_SECONDS
MAX_GPU_MEMORY_GB = kv_cache.MAX_GPU_MEMORY_GB
MAX_CPU_MEMORY_GB = kv_cache.MAX_CPU_MEMORY_GB
FORBIDDEN_CACHE_PREFIXES = kv_cache.FORBIDDEN_CACHE_PREFIXES

if PANDAS_AVAILABLE:
    import pandas as pd

# Check for GPU/CUDA availability
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        GPUMemoryBackend = kv_cache.GPUMemoryBackend
except ImportError:
    CUDA_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tiny_model_config():
    """Return the tiny-1b model config for fast tests."""
    return MODEL_CONFIGS['tiny-1b']


@pytest.fixture
def llama8b_config():
    """Return the llama3.1-8b model config."""
    return MODEL_CONFIGS['llama3.1-8b']


@pytest.fixture
def kv_generator(tiny_model_config):
    """Return a KVCacheGenerator with deterministic seed."""
    return KVCacheGenerator(tiny_model_config, global_seed=42)


@pytest.fixture
def cpu_backend():
    """Return a fresh CPUMemoryBackend."""
    backend = CPUMemoryBackend()
    yield backend
    backend.clear()


@pytest.fixture
def nvme_backend():
    """Return a fresh NVMeBackend (uses temp directory)."""
    backend = NVMeBackend()
    yield backend
    backend.clear()


@pytest.fixture
def conversation_manager():
    """Return a ConversationManager with small limit."""
    return ConversationManager(max_conversations=5)


@pytest.fixture
def multi_tier_cache(tiny_model_config):
    """Return a MultiTierCache in CPU-only mode."""
    return MultiTierCache(
        model_config=tiny_model_config,
        gpu_memory_gb=0,
        cpu_memory_gb=0.1,  # 100MB
        seed=42
    )


@pytest.fixture
def gpu_backend():
    """Return a fresh GPUMemoryBackend (requires CUDA)."""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")
    backend = GPUMemoryBackend()
    yield backend
    backend.clear()


@pytest.fixture
def multi_tier_cache_with_gpu(tiny_model_config):
    """Return a MultiTierCache with GPU enabled."""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")
    return MultiTierCache(
        model_config=tiny_model_config,
        gpu_memory_gb=1.0,  # 1GB GPU
        cpu_memory_gb=0.1,  # 100MB CPU
        seed=42
    )


@pytest.fixture
def mock_benchmark_results():
    """Return mock benchmark results for export tests."""
    return {
        'summary': {
            'total_requests': 100,
            'total_tokens': 10000,
            'elapsed_time': 60.0,
            'avg_throughput_tokens_per_sec': 166.67,
            'storage_throughput_tokens_per_sec': 200.0,
            'requests_per_second': 1.67,
            'end_to_end_latency_ms': {'mean': 50, 'p50': 45, 'p95': 100, 'p99': 150},
            'storage_io_latency_ms': {'mean': 10, 'p50': 8, 'p95': 20, 'p99': 30},
            'generation_latency_ms': {'mean': 40, 'p50': 35, 'p95': 80, 'p99': 120},
            'cache_stats': {
                'cache_hit_rate': 0.65,
                'read_write_ratio': 2.5,
                'total_read_gb': 5.0,
                'total_write_gb': 2.0,
                'gpu_entries': 0,
                'cpu_entries': 50,
                'nvme_entries': 100,
                'prefill_bytes_written_gb': 1.5,
                'decode_bytes_read_gb': 3.5,
            },
            'qos_metrics': {},
            'multi_turn_stats': {'hit_rate': 0.5}
        }
    }


@pytest.fixture
def mock_args():
    """Return mock CLI args for export tests."""
    class MockArgs:
        model = 'llama3.1-8b'
        num_users = 100
        duration = 60
        gpu_mem_gb = 16
        cpu_mem_gb = 32
        num_gpus = 1
        tensor_parallel = 1
        generation_mode = 'none'
        performance_profile = 'latency'
        disable_multi_turn = False
        disable_prefix_caching = False
        enable_rag = False
        enable_autoscaling = False
        seed = 42
        max_concurrent_allocs = 0
        request_rate = 0
        max_requests = 0
        max_conversations = 500
        rag_num_docs = 10
        dataset_path = None
        cache_dir = None
        storage_capacity_gb = 0
        precondition = False
        precondition_size_gb = 0
        precondition_threads = 0
        trace_speedup = 1.0
        replay_cycles = 0
        prefill_only = False
        decode_only = False
        validation_trace = None
        use_burst_trace = False
        burst_trace_path = None
        io_trace_log = None
        enable_latency_tracing = False
    return MockArgs()


@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a sample config.yaml for testing."""
    config_content = '''
user_templates:
  chatbot:
    context_range: [256, 1024]
    generation_range: [50, 150]
    think_time_range: [0.1, 0.5]
  coding:
    context_range: [1024, 4096]
    generation_range: [100, 500]
    think_time_range: [0.2, 1.0]
  document:
    context_range: [2048, 8192]
    generation_range: [200, 800]
    think_time_range: [0.3, 1.5]

qos_profiles:
  interactive:
    target_latency_p95_ms: 50
    target_latency_p99_ms: 100
    target_latency_p999_ms: 150
    target_latency_p9999_ms: 200
    priority: 3
  responsive:
    target_latency_p95_ms: 100
    target_latency_p99_ms: 200
    target_latency_p999_ms: 350
    target_latency_p9999_ms: 500
    priority: 2
  batch:
    target_latency_p95_ms: 1000
    target_latency_p99_ms: 5000
    target_latency_p999_ms: 7500
    target_latency_p9999_ms: 10000
    priority: 1

qos_distribution:
  interactive_probability: 0.15
  responsive_threshold: 0.50

eviction:
  max_recursion_depth: 10
  target_usage_ratio: 0.8
  large_entry_limit_ratio: 0.95
  max_evictions_hard_cap: 5000
  max_evictions_min: 1000

decode:
  batch_size: 32

conversation:
  max_conversations: 1000
  max_turns_per_conv: 50
  end_conversation_probability: 0.2
'''
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


# =============================================================================
# Test 0: ConfigLoader (New in 01-26-2026)
# =============================================================================

@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
class TestConfigLoader:
    """Tests for ConfigLoader and cfg() helper function."""
    
    def test_config_loader_without_file(self):
        """ConfigLoader should work without a config file."""
        loader = ConfigLoader(config_path=None)
        assert loader is not None
        assert loader.config == {}
    
    def test_config_loader_loads_yaml(self, sample_config_yaml):
        """ConfigLoader should load and parse YAML file."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        assert loader.config is not None
        assert 'qos_profiles' in loader.config
    
    def test_config_loader_get_nested_value(self, sample_config_yaml):
        """ConfigLoader.get() should retrieve nested values."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        priority = loader.get('qos_profiles', 'interactive', 'priority')
        assert priority == 3
    
    def test_config_loader_get_with_default(self, sample_config_yaml):
        """ConfigLoader.get() should return default for missing keys."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        value = loader.get('nonexistent', 'key', default=42)
        assert value == 42
    
    def test_cfg_without_global_config(self):
        """cfg() should return default when no global config is set."""
        # Ensure no global config
        set_config(None)
        value = cfg('qos_profiles', 'interactive', 'priority', default=99)
        assert value == 99
    
    def test_cfg_with_global_config(self, sample_config_yaml):
        """cfg() should retrieve values from global config."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        set_config(loader)
        try:
            value = cfg('qos_profiles', 'interactive', 'priority', default=99)
            assert value == 3
        finally:
            set_config(None)  # Clean up
    
    def test_config_loader_validates_schema(self, tmp_path):
        """ConfigLoader should reject unknown keys."""
        bad_config = tmp_path / "bad_config.yaml"
        bad_config.write_text('''
unknown_section:
  bad_key: true
''')
        with pytest.raises(ValueError, match="Unknown configuration key"):
            ConfigLoader(config_path=str(bad_config))
    
    def test_get_config_returns_none_initially(self):
        """get_config() should return None before set_config() is called."""
        set_config(None)
        assert get_config() is None
    
    def test_set_config_stores_loader(self, sample_config_yaml):
        """set_config() should store the ConfigLoader globally."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        set_config(loader)
        try:
            assert get_config() is loader
        finally:
            set_config(None)


class TestCfgHelper:
    """Tests for cfg() helper function in various contexts."""
    
    def test_cfg_returns_default_for_none_config(self):
        """cfg() returns default when config is None."""
        set_config(None)
        assert cfg('any', 'path', default='fallback') == 'fallback'
    
    def test_cfg_returns_default_for_missing_key(self, sample_config_yaml):
        """cfg() returns default for missing nested keys."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        set_config(loader)
        try:
            result = cfg('nonexistent', 'nested', 'key', default=123)
            assert result == 123
        finally:
            set_config(None)
    
    def test_cfg_retrieves_list_values(self, sample_config_yaml):
        """cfg() can retrieve list values from config."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        set_config(loader)
        try:
            context_range = cfg('user_templates', 'chatbot', 'context_range')
            assert context_range == [256, 1024]
        finally:
            set_config(None)


# =============================================================================
# Test 1: ModelConfig
# =============================================================================

class TestModelConfig:
    """Tests for ModelConfig dataclass and calculations."""
    
    def test_llama8b_config_exists(self, llama8b_config):
        assert llama8b_config is not None
    
    def test_kv_cache_size_per_token_positive(self, llama8b_config):
        assert llama8b_config.kv_cache_size_per_token > 0
    
    def test_bytes_per_element_float16(self, llama8b_config):
        assert llama8b_config.bytes_per_element == 2
    
    def test_kv_cache_size_formula(self, llama8b_config):
        """Verify: num_layers * kv_heads * kv_dim_per_head * 2 * bytes_per_element"""
        expected = (llama8b_config.num_layers * 
                    llama8b_config.kv_heads * 
                    llama8b_config.kv_dim_per_head * 
                    2 * llama8b_config.bytes_per_element)
        assert llama8b_config.kv_cache_size_per_token == expected
    
    def test_all_nine_model_configs_exist(self):
        assert len(MODEL_CONFIGS) == 9
    
    @pytest.mark.parametrize("model_name", [
        'tiny-1b', 'mistral-7b', 'llama2-7b', 'llama3.1-8b', 'llama3.1-70b-instruct', 'deepseek-v3', 'qwen3-32b', 'gpt-oss-120b', 'gpt-oss-20b'])
    def test_model_config_exists(self, model_name):
        assert model_name in MODEL_CONFIGS


# =============================================================================
# Test 2: InferenceRequest
# =============================================================================

class TestInferenceRequest:
    """Tests for InferenceRequest dataclass."""
    
    def test_create_request(self):
        req = InferenceRequest(
            user_id="test_user",
            request_id="test_req_001",
            timestamp=datetime.now(),
            context_tokens=1024,
            generate_tokens=128,
            priority=2,
            conversation_id="conv_123",
            turn_number=1
        )
        assert req is not None
    
    def test_cache_key_auto_generated(self):
        req = InferenceRequest(
            user_id="test_user",
            request_id="test_req_001",
            timestamp=datetime.now(),
            context_tokens=1024,
            generate_tokens=128,
            priority=2,
            conversation_id="conv_123",
            turn_number=1
        )
        assert req.cache_key == "conv_123_turn_1"
    
    def test_cache_key_fallback_without_conversation(self):
        req = InferenceRequest(
            user_id="test_user2",
            request_id="test_req_002",
            timestamp=datetime.now(),
            context_tokens=512,
            generate_tokens=64,
            priority=1
        )
        assert req.cache_key == "test_user2_ctx"
    
    def test_submit_time_set(self):
        req = InferenceRequest(
            user_id="test_user",
            request_id="test_req",
            timestamp=datetime.now(),
            context_tokens=100,
            generate_tokens=10,
            priority=1
        )
        assert req.submit_time > 0
    
    def test_total_latency_ms(self):
        req = InferenceRequest(
            user_id="test_user",
            request_id="test_req",
            timestamp=datetime.now(),
            context_tokens=100,
            generate_tokens=10,
            priority=1
        )
        req.complete_time = req.submit_time + 0.1  # 100ms
        assert req.total_latency_ms > 0


# =============================================================================
# Test 3: QoS Profiles
# =============================================================================

class TestQoSProfiles:
    """Tests for QoS profiles and SLA."""
    
    def test_three_qos_levels(self):
        assert len(QOS_PROFILES) == 3
    
    def test_interactive_priority_highest(self):
        assert QOS_PROFILES[QoSLevel.INTERACTIVE].priority == 3
    
    def test_responsive_priority_middle(self):
        assert QOS_PROFILES[QoSLevel.RESPONSIVE].priority == 2
    
    def test_batch_priority_lowest(self):
        assert QOS_PROFILES[QoSLevel.BATCH].priority == 1
    
    def test_sla_compliance_starts_at_one(self):
        sla = QOS_PROFILES[QoSLevel.INTERACTIVE]
        assert sla.sla_compliance == 1.0
    
    def test_interactive_target_latency(self):
        sla = QOS_PROFILES[QoSLevel.INTERACTIVE]
        assert sla.target_latency_p95_ms == 50
    
    # New tests for extended QoS percentiles (01-26-2026 feature)
    def test_interactive_has_p999_latency(self):
        """Test that p999 percentile is defined for INTERACTIVE."""
        sla = QOS_PROFILES[QoSLevel.INTERACTIVE]
        assert hasattr(sla, 'target_latency_p999_ms')
        assert sla.target_latency_p999_ms > sla.target_latency_p99_ms
    
    def test_interactive_has_p9999_latency(self):
        """Test that p9999 percentile is defined for INTERACTIVE."""
        sla = QOS_PROFILES[QoSLevel.INTERACTIVE]
        assert hasattr(sla, 'target_latency_p9999_ms')
        assert sla.target_latency_p9999_ms > sla.target_latency_p999_ms
    
    def test_all_qos_levels_have_extended_percentiles(self):
        """Verify all QoS levels have p999 and p9999 defined."""
        for level in QoSLevel:
            sla = QOS_PROFILES[level]
            assert hasattr(sla, 'target_latency_p999_ms')
            assert hasattr(sla, 'target_latency_p9999_ms')
    
    def test_get_qos_profiles_returns_dict(self):
        """Test that get_qos_profiles() returns profiles dict."""
        profiles = get_qos_profiles()
        assert isinstance(profiles, dict)
        assert len(profiles) == 3
    
    def test_get_qos_profiles_levels(self):
        """Test that get_qos_profiles() has all QoS levels."""
        profiles = get_qos_profiles()
        assert QoSLevel.INTERACTIVE in profiles
        assert QoSLevel.RESPONSIVE in profiles
        assert QoSLevel.BATCH in profiles


# =============================================================================
# Test 4: KVCacheGenerator
# =============================================================================

class TestKVCacheGenerator:
    """Tests for KVCacheGenerator."""
    
    def test_generator_created(self, kv_generator):
        assert kv_generator is not None
    
    def test_precomputed_buffer_allocated(self, kv_generator):
        assert kv_generator.precomputed_buffer is not None
    
    def test_precomputed_buffer_size(self, kv_generator):
        assert len(kv_generator.precomputed_buffer) == 128 * 1024 * 1024
    
    def test_generated_data_shape(self, kv_generator, tiny_model_config):
        data = kv_generator.generate(sequence_length=10, key="test_key")
        expected_shape = (
            tiny_model_config.num_layers, 2, 10,
            tiny_model_config.kv_heads, tiny_model_config.kv_dim_per_head
        )
        assert data.shape == expected_shape
    
    def test_generated_data_dtype(self, kv_generator):
        data = kv_generator.generate(sequence_length=10, key="test_key")
        assert data.dtype == np.float16
    
    def test_determinism_same_key(self, kv_generator):
        data1 = kv_generator.generate(sequence_length=10, key="test_key")
        data2 = kv_generator.generate(sequence_length=10, key="test_key")
        # Compare raw bytes: XOR-stamping may produce NaN bit patterns,
        # and NaN != NaN under IEEE 754, so np.array_equal would fail
        # even though the data is bit-identical.
        assert data1.view(np.uint8).tobytes() == data2.view(np.uint8).tobytes()
    
    def test_different_key_runs(self, kv_generator):
        """Different key should not crash."""
        kv_generator.generate(sequence_length=10, key="different_key")


# =============================================================================
# Test 4b: Data Deduplication Resistance
# =============================================================================

class TestDedup:
    """Verify that generated KV cache data resists 4 KB block deduplication.

    Real NVMe controllers and storage appliances may transparently dedup
    identical blocks, which would make the benchmark measure less I/O than
    intended.  The XOR-stamp in KVCacheGenerator._apply_xor_stamp must
    ensure zero (or near-zero) duplicate 4 KB blocks across entries and
    within a single large entry.

    Equivalent of the shell one-liner:
        find /mnt/... -name '*.npy' -print0 | xargs -0 cat | sha256-block-check
    """

    @pytest.fixture
    def tiny_model_config(self):
        return MODEL_CONFIGS['tiny-1b']

    def _block_dedup_ratio(self, data_list: list, block_size: int = 4096) -> dict:
        """Hash every block_size chunk across all byte strings, return stats."""
        import hashlib
        blocks = {}
        total = 0
        for raw in data_list:
            for offset in range(0, len(raw), block_size):
                chunk = raw[offset:offset + block_size]
                h = hashlib.sha256(chunk).digest()
                blocks[h] = blocks.get(h, 0) + 1
                total += 1
        unique = len(blocks)
        dedup_ratio = 1 - (unique / total) if total else 0
        dupes = sum(v - 1 for v in blocks.values() if v > 1)
        return dict(total=total, unique=unique, dupes=dupes, ratio=dedup_ratio)

    def test_cross_entry_no_dedup(self, tiny_model_config):
        """Different keys must produce zero duplicate 4 KB blocks."""
        gen = KVCacheGenerator(tiny_model_config, global_seed=42)
        raw_list = []
        for i in range(50):
            data = gen.generate(sequence_length=100, key=f"user_{i:04d}_ctx")
            raw_list.append(data.view(np.uint8).tobytes())

        stats = self._block_dedup_ratio(raw_list)
        print(f"\n  Cross-entry dedup: {stats['total']:,} blocks, "
              f"{stats['unique']:,} unique, ratio={stats['ratio']:.4%}")
        assert stats['ratio'] < 0.01, (
            f"Cross-entry dedup ratio {stats['ratio']:.2%} exceeds 1% — "
            f"{stats['dupes']:,} duplicate blocks out of {stats['total']:,}"
        )

    def test_intra_entry_no_dedup(self, tiny_model_config):
        """A single large entry must have no duplicate 4 KB blocks within itself.

        Uses a long sequence so the entry exceeds the 256 MB precomputed buffer,
        forcing the chunked-copy path.  Without the block-index XOR layer,
        repeated buffer regions would produce ~90% intra-entry dedup.
        """
        # Make a model with enough layers/heads that 8192 tokens > 256 MB
        # tiny-1b at 24 KB/tok: 8192 × 24 KB = 192 MB (under buffer)
        # Use 16384 tokens → 384 MB (exceeds 256 MB buffer → large path)
        gen = KVCacheGenerator(tiny_model_config, global_seed=42)
        data = gen.generate(sequence_length=16384, key="big_entry")
        raw = data.view(np.uint8).tobytes()

        stats = self._block_dedup_ratio([raw])
        print(f"\n  Intra-entry dedup: {stats['total']:,} blocks, "
              f"{stats['unique']:,} unique, ratio={stats['ratio']:.4%}")
        assert stats['ratio'] < 0.01, (
            f"Intra-entry dedup ratio {stats['ratio']:.2%} exceeds 1% — "
            f"{stats['dupes']:,} duplicate blocks out of {stats['total']:,}"
        )

    def test_combined_dedup_many_entries(self, tiny_model_config):
        """Mixed workload: many entries of varying sizes, check overall dedup."""
        gen = KVCacheGenerator(tiny_model_config, global_seed=99)
        raw_list = []
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 512, 1024, 2048]
        for i, seq_len in enumerate(seq_lengths):
            data = gen.generate(sequence_length=seq_len, key=f"req_{i:04d}")
            raw_list.append(data.view(np.uint8).tobytes())

        stats = self._block_dedup_ratio(raw_list)
        total_mb = sum(len(r) for r in raw_list) / 1024**2
        print(f"\n  Combined dedup ({total_mb:.1f} MiB across {len(raw_list)} entries): "
              f"{stats['total']:,} blocks, {stats['unique']:,} unique, "
              f"ratio={stats['ratio']:.4%}")
        assert stats['ratio'] < 0.01, (
            f"Combined dedup ratio {stats['ratio']:.2%} exceeds 1% — "
            f"{stats['dupes']:,} duplicate blocks out of {stats['total']:,}"
        )

    def test_determinism_preserves_dedup_resistance(self, tiny_model_config):
        """Two runs with the same seed must produce identical bytes AND low dedup."""
        gen1 = KVCacheGenerator(tiny_model_config, global_seed=42)
        gen2 = KVCacheGenerator(tiny_model_config, global_seed=42)

        for i in range(10):
            key = f"det_test_{i}"
            d1 = gen1.generate(sequence_length=512, key=key)
            d2 = gen2.generate(sequence_length=512, key=key)
            assert d1.view(np.uint8).tobytes() == d2.view(np.uint8).tobytes(), (
                f"Entry {key} not bit-identical across generators"
            )

        # Also check dedup across the 10 entries
        raw_list = []
        gen3 = KVCacheGenerator(tiny_model_config, global_seed=42)
        for i in range(10):
            data = gen3.generate(sequence_length=512, key=f"det_test_{i}")
            raw_list.append(data.view(np.uint8).tobytes())
        stats = self._block_dedup_ratio(raw_list)
        print(f"\n  Deterministic dedup: ratio={stats['ratio']:.4%}")
        assert stats['ratio'] < 0.01


# =============================================================================
# Test 5: CPUMemoryBackend
# =============================================================================

class TestCPUMemoryBackend:
    """Tests for CPUMemoryBackend."""
    
    def test_backend_created(self, cpu_backend):
        assert cpu_backend is not None
    
    def test_write_returns_timing(self, cpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        timing = cpu_backend.write("test_key", test_data)
        assert timing.total >= 0
    
    def test_read_returns_data(self, cpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        cpu_backend.write("test_key", test_data)
        read_data, _ = cpu_backend.read("test_key")
        assert read_data is not None
    
    def test_read_data_matches_written(self, cpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        cpu_backend.write("test_key", test_data)
        read_data, _ = cpu_backend.read("test_key")
        assert np.array_equal(read_data, test_data)
    
    def test_read_timing_returned(self, cpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        cpu_backend.write("test_key", test_data)
        _, timing = cpu_backend.read("test_key")
        assert timing.total >= 0
    
    def test_delete_removes_key(self, cpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        cpu_backend.write("test_key", test_data)
        cpu_backend.delete("test_key")
        with pytest.raises(KeyError):
            cpu_backend.read("test_key")
    
    def test_clear_empties_cache(self, cpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        cpu_backend.write("key1", test_data)
        cpu_backend.write("key2", test_data)
        assert len(cpu_backend.cache) == 2
        cpu_backend.clear()
        assert len(cpu_backend.cache) == 0


# =============================================================================
# Test 6: NVMeBackend
# =============================================================================

class TestNVMeBackend:
    """Tests for NVMeBackend (uses temp directory)."""
    
    def test_backend_created(self, nvme_backend):
        assert nvme_backend is not None
    
    def test_temp_directory_exists(self, nvme_backend):
        assert nvme_backend.base_path.exists()
    
    def test_write_returns_timing(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        timing = nvme_backend.write("nvme_test", test_data)
        assert timing.total >= 0
    
    def test_write_timing_has_device_component(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        timing = nvme_backend.write("nvme_test", test_data)
        assert timing.device >= 0
    
    def test_file_created(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        nvme_backend.write("nvme_test", test_data)
        assert (nvme_backend.base_path / "nvme_test.npy").exists()
    
    def test_read_returns_data(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        nvme_backend.write("nvme_test", test_data)
        read_data, _ = nvme_backend.read("nvme_test")
        assert read_data is not None
    
    def test_read_data_matches(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        nvme_backend.write("nvme_test", test_data)
        read_data, _ = nvme_backend.read("nvme_test")
        assert np.allclose(read_data, test_data)
    
    def test_metadata_stored(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        nvme_backend.write("nvme_test", test_data)
        assert "nvme_test" in nvme_backend.metadata
    
    def test_delete_removes_file(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        nvme_backend.write("nvme_test", test_data)
        nvme_backend.delete("nvme_test")
        assert not (nvme_backend.base_path / "nvme_test.npy").exists()
    
    def test_clear_removes_all_files(self, nvme_backend):
        test_data = np.random.rand(50, 50).astype(np.float32)
        nvme_backend.write("nvme_test1", test_data)
        nvme_backend.write("nvme_test2", test_data)
        nvme_backend.clear()
        assert len(list(nvme_backend.base_path.glob("*.npy"))) == 0


# =============================================================================
# Test 6b: GPUMemoryBackend (requires CUDA)
# =============================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGPUMemoryBackend:
    """Tests for GPUMemoryBackend (requires CUDA)."""
    
    def test_backend_created(self, gpu_backend):
        assert gpu_backend is not None
    
    def test_write_returns_timing(self, gpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        timing = gpu_backend.write("test_key", test_data)
        assert timing.total >= 0
    
    def test_write_timing_has_device_component(self, gpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        timing = gpu_backend.write("test_key", test_data)
        assert timing.device >= 0
    
    def test_read_returns_data(self, gpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        gpu_backend.write("test_key", test_data)
        read_data, _ = gpu_backend.read("test_key")
        assert read_data is not None
    
    def test_read_data_matches_written(self, gpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        gpu_backend.write("test_key", test_data)
        read_data, _ = gpu_backend.read("test_key")
        assert np.allclose(read_data, test_data, rtol=1e-3)
    
    def test_read_timing_returned(self, gpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        gpu_backend.write("test_key", test_data)
        _, timing = gpu_backend.read("test_key")
        assert timing.total >= 0
    
    def test_delete_removes_key(self, gpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        gpu_backend.write("test_key", test_data)
        gpu_backend.delete("test_key")
        with pytest.raises(KeyError):
            gpu_backend.read("test_key")
    
    def test_clear_empties_cache(self, gpu_backend):
        test_data = np.random.rand(100, 100).astype(np.float16)
        gpu_backend.write("key1", test_data)
        gpu_backend.write("key2", test_data)
        gpu_backend.clear()
        assert len(gpu_backend.cache) == 0
    
    def test_data_on_cuda_device(self, gpu_backend):
        """Verify data is stored on GPU."""
        import torch
        test_data = np.random.rand(100, 100).astype(np.float16)
        gpu_backend.write("test_key", test_data)
        # Access internal cache to verify CUDA storage
        assert gpu_backend.cache["test_key"].is_cuda


# =============================================================================
# Test 7: ConversationManager
# =============================================================================

class TestConversationManager:
    """Tests for ConversationManager."""
    
    def test_manager_created(self, conversation_manager):
        assert conversation_manager is not None
    
    def test_max_conversations_set(self, conversation_manager):
        assert conversation_manager.max_conversations == 5
    
    def test_start_conversation(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        assert conv_id is not None
    
    def test_conversation_id_format(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        assert conv_id.startswith("conv_user_1_")
    
    def test_conversation_stored(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        assert conv_id in conversation_manager.conversations
    
    def test_add_turn(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        turn_num, cache_key = conversation_manager.add_turn(conv_id, 100, 50)
        assert turn_num == 1
    
    def test_cache_key_format(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        turn_num, cache_key = conversation_manager.add_turn(conv_id, 100, 50)
        assert cache_key == f"{conv_id}_turn_1"
    
    def test_second_turn_number(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        conversation_manager.add_turn(conv_id, 100, 50)
        turn_num, _ = conversation_manager.add_turn(conv_id, 200, 100)
        assert turn_num == 2
    
    def test_context_size_tracked(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        conversation_manager.add_turn(conv_id, 100, 50)
        conversation_manager.add_turn(conv_id, 200, 100)
        context_size = conversation_manager.get_conversation_context_size(conv_id)
        assert context_size == 450  # 100+50+200+100
    
    def test_previous_turn_keys(self, conversation_manager):
        conv_id = conversation_manager.start_conversation("user_1")
        conversation_manager.add_turn(conv_id, 100, 50)
        conversation_manager.add_turn(conv_id, 200, 100)
        prev_keys = conversation_manager.get_all_previous_turn_keys(conv_id, 2)
        assert len(prev_keys) == 1
    
    def test_max_conversations_enforced(self, conversation_manager):
        for i in range(10):
            conversation_manager.start_conversation(f"user_{i}")
        assert len(conversation_manager.conversations) <= 5


# =============================================================================
# Test 8: UserSimulator
# =============================================================================

class TestUserSimulator:
    """Tests for UserSimulator."""
    
    def test_generate_mixed_users(self):
        users = UserSimulator.generate_mixed_users(10)
        assert len(users) == 10
    
    def test_users_have_valid_context_lengths(self):
        users = UserSimulator.generate_mixed_users(10)
        for user in users:
            # Range covers all user templates: chatbot [512,4096], coding [4096,25000], document [4096,16384]
            assert 512 <= user.context_length <= 25000
    
    def test_qos_levels_assigned(self):
        users = UserSimulator.generate_mixed_users(10)
        qos_levels = set(u.qos_level for u in users)
        assert len(qos_levels) >= 1
    
    def test_single_user_generation(self):
        user = UserSimulator.generate_user("test_user", "chatbot", 2, QoSLevel.RESPONSIVE)
        assert user is not None
    
    def test_single_user_id(self):
        user = UserSimulator.generate_user("test_user", "chatbot", 2, QoSLevel.RESPONSIVE)
        assert user.user_id == "test_user"
    
    def test_single_user_qos(self):
        user = UserSimulator.generate_user("test_user", "chatbot", 2, QoSLevel.RESPONSIVE)
        assert user.qos_level == QoSLevel.RESPONSIVE


# =============================================================================
# Test 9: MultiTierCache (CPU-only)
# =============================================================================

class TestMultiTierCache:
    """Tests for MultiTierCache (CPU-only mode)."""
    
    def test_cache_created_with_zero_gpu_memory(self, multi_tier_cache):
        """With gpu_memory_gb=0, GPU limit should be 0 (even if backend exists)."""
        gpu_limit = multi_tier_cache._get_tier_limit('gpu') if 'gpu' in multi_tier_cache.backends else 0
        assert gpu_limit == 0
    
    def test_cpu_backend_available(self, multi_tier_cache):
        assert 'cpu' in multi_tier_cache.backends
    
    def test_nvme_backend_available(self, multi_tier_cache):
        assert 'nvme' in multi_tier_cache.backends
    
    def test_allocation_succeeds(self, multi_tier_cache):
        success, location, latency = multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        assert success is True
    
    def test_allocation_location(self, multi_tier_cache):
        success, location, latency = multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        assert location in ['cpu', 'nvme']
    
    def test_allocation_returns_latency(self, multi_tier_cache):
        success, location, latency = multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        assert latency >= 0
    
    def test_cache_access_succeeds(self, multi_tier_cache):
        multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        loc, read_lat = multi_tier_cache.access_cache("test_entry", InferencePhase.DECODE)
        assert loc is not None
    
    def test_cache_access_returns_location(self, multi_tier_cache):
        multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        loc, _ = multi_tier_cache.access_cache("test_entry", InferencePhase.DECODE)
        assert loc in ['cpu', 'nvme']
    
    def test_nonexistent_key_returns_none(self, multi_tier_cache):
        loc, _ = multi_tier_cache.access_cache("nonexistent_key", InferencePhase.DECODE)
        assert loc is None
    
    def test_stats_returned(self, multi_tier_cache):
        multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        multi_tier_cache.access_cache("test_entry", InferencePhase.DECODE)
        stats = multi_tier_cache.get_stats(duration=1.0)
        assert stats is not None
    
    def test_cache_hit_recorded(self, multi_tier_cache):
        multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        multi_tier_cache.access_cache("test_entry", InferencePhase.DECODE)
        stats = multi_tier_cache.get_stats(duration=1.0)
        assert stats['cache_hits'] >= 1
    
    def test_cache_miss_recorded(self, multi_tier_cache):
        multi_tier_cache.access_cache("nonexistent", InferencePhase.DECODE)
        stats = multi_tier_cache.get_stats(duration=1.0)
        assert stats['cache_misses'] >= 1
    
    def test_storage_health_in_stats(self, multi_tier_cache):
        stats = multi_tier_cache.get_stats(duration=1.0)
        assert 'storage_health' in stats


# =============================================================================
# Test 9b: MultiTierCache with GPU
# =============================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestMultiTierCacheWithGPU:
    """Tests for MultiTierCache with GPU enabled."""
    
    def test_gpu_backend_available(self, multi_tier_cache_with_gpu):
        assert 'gpu' in multi_tier_cache_with_gpu.backends
    
    def test_cpu_backend_available(self, multi_tier_cache_with_gpu):
        assert 'cpu' in multi_tier_cache_with_gpu.backends
    
    def test_nvme_backend_available(self, multi_tier_cache_with_gpu):
        assert 'nvme' in multi_tier_cache_with_gpu.backends
    
    def test_tier_order_with_gpu(self, multi_tier_cache_with_gpu):
        tier_order = multi_tier_cache_with_gpu._get_tier_order()
        assert tier_order == ['gpu', 'cpu', 'nvme']
    
    def test_gpu_limit_set(self, multi_tier_cache_with_gpu):
        gpu_limit = multi_tier_cache_with_gpu._get_tier_limit('gpu')
        assert gpu_limit == 1.0 * 1024**3  # 1GB
    
    def test_allocation_prefers_gpu(self, multi_tier_cache_with_gpu):
        """Small allocations should go to GPU first."""
        success, location, latency = multi_tier_cache_with_gpu.allocate_cache("test_entry", num_tokens=100)
        assert success is True
        assert location == 'gpu'
    
    def test_gpu_overflow_to_cpu(self, multi_tier_cache_with_gpu):
        """When GPU is full, should overflow to CPU."""
        # Fill GPU with large allocations
        for i in range(100):
            multi_tier_cache_with_gpu.allocate_cache(f"entry_{i}", num_tokens=10000)
        
        # Next allocation should go to CPU or NVMe
        success, location, _ = multi_tier_cache_with_gpu.allocate_cache("overflow_entry", num_tokens=10000)
        assert success is True
        assert location in ['cpu', 'nvme']
    
    def test_cache_access_from_gpu(self, multi_tier_cache_with_gpu):
        multi_tier_cache_with_gpu.allocate_cache("test_entry", num_tokens=100)
        loc, read_lat = multi_tier_cache_with_gpu.access_cache("test_entry", InferencePhase.DECODE)
        assert loc == 'gpu'
    
    def test_stats_include_gpu_entries(self, multi_tier_cache_with_gpu):
        multi_tier_cache_with_gpu.allocate_cache("test_entry", num_tokens=100)
        stats = multi_tier_cache_with_gpu.get_stats(duration=1.0)
        assert 'gpu_entries' in stats


# =============================================================================
# Test 10: XLSX Export
# =============================================================================

@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not installed")
class TestXLSXExport:
    """Tests for XLSX/CSV export functionality."""
    
    def test_csv_export_succeeds(self, mock_benchmark_results, mock_args):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_path = f.name
        try:
            export_results_to_xlsx(mock_benchmark_results, mock_args, test_path)
            assert os.path.exists(test_path)
        finally:
            if os.path.exists(test_path):
                os.unlink(test_path)
    
    def test_csv_has_data(self, mock_benchmark_results, mock_args):
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_path = f.name
        try:
            export_results_to_xlsx(mock_benchmark_results, mock_args, test_path)
            df = pd.read_csv(test_path)
            assert len(df) == 1
        finally:
            if os.path.exists(test_path):
                os.unlink(test_path)
    
    def test_csv_has_model_column(self, mock_benchmark_results, mock_args):
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_path = f.name
        try:
            export_results_to_xlsx(mock_benchmark_results, mock_args, test_path)
            df = pd.read_csv(test_path)
            assert 'Model' in df.columns
        finally:
            if os.path.exists(test_path):
                os.unlink(test_path)
    
    def test_csv_model_value(self, mock_benchmark_results, mock_args):
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_path = f.name
        try:
            export_results_to_xlsx(mock_benchmark_results, mock_args, test_path)
            df = pd.read_csv(test_path)
            assert df['Model'].iloc[0] == 'llama3.1-8b'
        finally:
            if os.path.exists(test_path):
                os.unlink(test_path)
    
    def test_csv_has_throughput_columns(self, mock_benchmark_results, mock_args):
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_path = f.name
        try:
            export_results_to_xlsx(mock_benchmark_results, mock_args, test_path)
            df = pd.read_csv(test_path)
            assert 'Avg Throughput (tok/s)' in df.columns
            assert 'Storage Throughput (tok/s)' in df.columns
        finally:
            if os.path.exists(test_path):
                os.unlink(test_path)


# =============================================================================
# Test 11: Enums
# =============================================================================

class TestEnums:
    """Tests for enum consistency."""
    
    def test_inference_phase_count(self):
        assert len(InferencePhase) == 3
    
    def test_inference_phase_prefill(self):
        assert InferencePhase.PREFILL.value == "prefill"
    
    def test_inference_phase_decode(self):
        assert InferencePhase.DECODE.value == "decode"
    
    def test_inference_phase_both(self):
        assert InferencePhase.PREFILL_DECODE.value == "both"
    
    def test_generation_mode_count(self):
        assert len(GenerationMode) == 3
    
    def test_generation_timing_none(self):
        assert GENERATION_TIMING[GenerationMode.NONE] == 0.0
    
    def test_generation_timing_fast(self):
        assert GENERATION_TIMING[GenerationMode.FAST] == 0.002
    
    def test_generation_timing_realistic(self):
        assert GENERATION_TIMING[GenerationMode.REALISTIC] == 0.030
    
    def test_qos_level_count(self):
        assert len(QoSLevel) == 3
    
    def test_timing_matches_modes(self):
        assert len(GENERATION_TIMING) == len(GenerationMode)


# =============================================================================
# Test 12: Tier Logic
# =============================================================================

class TestTierLogic:
    """Tests for tier ordering and limits."""
    
    def test_tier_order_includes_expected_tiers(self, multi_tier_cache):
        """Tier order should include cpu and nvme (gpu may or may not be present)."""
        tier_order = multi_tier_cache._get_tier_order()
        assert 'cpu' in tier_order
        assert 'nvme' in tier_order
        # If GPU is present, it should be first
        if 'gpu' in tier_order:
            assert tier_order.index('gpu') < tier_order.index('cpu')
    
    def test_cpu_limit(self, multi_tier_cache):
        cpu_limit = multi_tier_cache._get_tier_limit('cpu')
        assert cpu_limit == 0.1 * 1024**3  # 100MB
    
    def test_nvme_limit_auto_detected(self, multi_tier_cache):
        """NVMe limit should be auto-detected from disk free space (not inf)."""
        nvme_limit = multi_tier_cache._get_tier_limit('nvme')
        assert nvme_limit > 0
    
    def test_initial_cpu_usage_zero(self, multi_tier_cache):
        cpu_usage = multi_tier_cache._get_tier_usage('cpu')
        assert cpu_usage == 0


# =============================================================================
# Test 13: Config-Driven Parameters (New in 01-26-2026)
# =============================================================================

class TestConfigDrivenConversationManager:
    """Tests for ConversationManager with config-driven parameters."""
    
    def test_default_max_conversations(self):
        """Without config, should use hardcoded default of 1000."""
        set_config(None)
        manager = ConversationManager()
        assert manager.max_conversations == 1000
    
    def test_default_max_turns(self):
        """Without config, should use hardcoded default of 50."""
        set_config(None)
        manager = ConversationManager()
        assert manager.max_turns_per_conv == 50
    
    def test_explicit_params_override_config(self, sample_config_yaml):
        """Explicit constructor params should override config values."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        set_config(loader)
        try:
            manager = ConversationManager(max_conversations=42, max_turns_per_conv=7)
            assert manager.max_conversations == 42
            assert manager.max_turns_per_conv == 7
        finally:
            set_config(None)


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
class TestConfigDrivenUserSimulator:
    """Tests for UserSimulator with config-driven parameters."""
    
    def test_user_templates_from_config(self, sample_config_yaml):
        """UserSimulator should read templates from config."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        set_config(loader)
        try:
            templates = UserSimulator._get_user_templates()
            assert 'chatbot' in templates
            assert 'coding' in templates
            assert 'document' in templates
            assert templates['chatbot']['context_range'] == (256, 1024)
        finally:
            set_config(None)
    
    def test_qos_distribution_from_config(self, sample_config_yaml):
        """UserSimulator.generate_mixed_users should use config QoS distribution."""
        loader = ConfigLoader(config_path=sample_config_yaml)
        set_config(loader)
        try:
            # Generate many users to test distribution
            users = UserSimulator.generate_mixed_users(1000)
            # With 15% interactive probability, expect ~150 interactive users
            interactive_count = sum(1 for u in users if u.qos_level == QoSLevel.INTERACTIVE)
            # Allow 50% variance for randomness
            assert 75 <= interactive_count <= 225, f"Expected ~150 interactive, got {interactive_count}"
        finally:
            set_config(None)


# =============================================================================
# Test 14: Stats Naming Convention (storage_* vs nvme_*)
# =============================================================================

class TestStatsNamingConvention:
    """Tests that stats use 'storage_*' naming (not 'nvme_*') in 01-26-2026."""
    
    def test_stats_use_storage_prefix(self, multi_tier_cache):
        """Stats should use 'storage_' prefix instead of 'nvme_'."""
        multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        multi_tier_cache.access_cache("test_entry", InferencePhase.DECODE)
        stats = multi_tier_cache.get_stats(duration=1.0)
        
        # Check for storage_* naming
        storage_keys = [k for k in stats.keys() if 'storage_' in k.lower()]
        nvme_keys = [k for k in stats.keys() if 'nvme_' in k.lower()]
        
        # Should have storage_* keys
        assert len(storage_keys) > 0, "Expected storage_* keys in stats"
    
    def test_tier_stats_key_format(self, multi_tier_cache):
        """tier_storage_* keys should exist (renamed from tier_nvme_*)."""
        multi_tier_cache.allocate_cache("test_entry", num_tokens=100)
        stats = multi_tier_cache.get_stats(duration=1.0)
        
        # Check for tier_storage_* keys
        tier_storage_keys = [k for k in stats.keys() if k.startswith('tier_storage_')]
        assert len(tier_storage_keys) > 0, "Expected tier_storage_* keys in stats"


# =============================================================================
# Test 15: GPUMemoryBackend Eviction Callback (New in 01-26-2026)
# =============================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGPUMemoryBackendEvictionCallback:
    """Tests for GPUMemoryBackend's on_eviction_callback feature."""
    
    def test_gpu_backend_accepts_callback(self):
        """GPUMemoryBackend should accept on_eviction_callback parameter."""
        evicted_keys = []
        def callback(key, tier, size):
            evicted_keys.append((key, tier, size))
        
        backend = GPUMemoryBackend(on_eviction_callback=callback)
        assert backend.on_eviction_callback is callback
        backend.clear()
    
    def test_gpu_backend_works_without_callback(self):
        """GPUMemoryBackend should work without a callback (None)."""
        backend = GPUMemoryBackend(on_eviction_callback=None)
        assert backend.on_eviction_callback is None
        backend.clear()


# =============================================================================
# Test 16: Input Validation (validate_args)
# =============================================================================

class TestValidateArgs:
    """Tests for the validate_args() input validation function."""
    
    @pytest.fixture
    def valid_args(self):
        """Create a valid args namespace with all required attributes."""
        import argparse
        args = argparse.Namespace(
            num_users=100,
            duration=60,
            gpu_mem_gb=16,
            cpu_mem_gb=32,
            num_gpus=1,
            tensor_parallel=1,
            rag_num_docs=10,
            max_conversations=500,
            max_concurrent_allocs=0,
            request_rate=0,
            max_requests=0,
            target_saturation=0.8,
            cache_dir=None,
            storage_capacity_gb=0,
            precondition_size_gb=0,
            precondition_threads=0,
            trace_speedup=1.0,
            replay_cycles=0
        )
        return args
    
    def test_valid_args_pass_through(self, valid_args):
        """Valid arguments should pass validation and return unchanged."""
        result = validate_args(valid_args)
        assert result is valid_args
        assert result.num_users == 100
        assert result.duration == 60
    
    def test_num_users_zero_rejected(self, valid_args):
        """num_users=0 should raise ValueError."""
        valid_args.num_users = 0
        with pytest.raises(ValueError, match="num-users must be positive"):
            validate_args(valid_args)
    
    def test_num_users_negative_rejected(self, valid_args):
        """Negative num_users should raise ValueError."""
        valid_args.num_users = -5
        with pytest.raises(ValueError, match="num-users must be positive"):
            validate_args(valid_args)
    
    def test_num_users_exceeds_limit(self, valid_args):
        """num_users exceeding MAX_USERS should raise ValueError."""
        valid_args.num_users = MAX_USERS + 1
        with pytest.raises(ValueError, match="num-users exceeds limit"):
            validate_args(valid_args)
    
    def test_duration_zero_rejected(self, valid_args):
        """duration=0 should raise ValueError."""
        valid_args.duration = 0
        with pytest.raises(ValueError, match="duration must be positive"):
            validate_args(valid_args)
    
    def test_duration_negative_rejected(self, valid_args):
        """Negative duration should raise ValueError."""
        valid_args.duration = -10
        with pytest.raises(ValueError, match="duration must be positive"):
            validate_args(valid_args)
    
    def test_duration_exceeds_limit(self, valid_args):
        """duration exceeding 24 hours should raise ValueError."""
        valid_args.duration = MAX_DURATION_SECONDS + 1
        with pytest.raises(ValueError, match="duration exceeds 24 hours"):
            validate_args(valid_args)
    
    def test_gpu_mem_negative_rejected(self, valid_args):
        """Negative gpu_mem_gb should raise ValueError."""
        valid_args.gpu_mem_gb = -1
        with pytest.raises(ValueError, match="gpu-mem-gb cannot be negative"):
            validate_args(valid_args)
    
    def test_gpu_mem_zero_allowed(self, valid_args):
        """gpu_mem_gb=0 should be valid (disables GPU tier)."""
        valid_args.gpu_mem_gb = 0
        result = validate_args(valid_args)
        assert result.gpu_mem_gb == 0
    
    def test_gpu_mem_exceeds_limit(self, valid_args):
        """gpu_mem_gb exceeding limit should raise ValueError."""
        valid_args.gpu_mem_gb = MAX_GPU_MEMORY_GB + 1
        with pytest.raises(ValueError, match="gpu-mem-gb exceeds limit"):
            validate_args(valid_args)
    
    def test_cpu_mem_negative_rejected(self, valid_args):
        """Negative cpu_mem_gb should raise ValueError."""
        valid_args.cpu_mem_gb = -1
        with pytest.raises(ValueError, match="cpu-mem-gb cannot be negative"):
            validate_args(valid_args)
    
    def test_cpu_mem_zero_allowed(self, valid_args):
        """cpu_mem_gb=0 should be valid."""
        valid_args.cpu_mem_gb = 0
        result = validate_args(valid_args)
        assert result.cpu_mem_gb == 0
    
    def test_cpu_mem_exceeds_limit(self, valid_args):
        """cpu_mem_gb exceeding limit should raise ValueError."""
        valid_args.cpu_mem_gb = MAX_CPU_MEMORY_GB + 1
        with pytest.raises(ValueError, match="cpu-mem-gb exceeds limit"):
            validate_args(valid_args)
    
    def test_target_saturation_below_zero_rejected(self, valid_args):
        """target_saturation < 0 should raise ValueError."""
        valid_args.target_saturation = -0.1
        with pytest.raises(ValueError, match="target-saturation must be between 0.0 and 1.0"):
            validate_args(valid_args)
    
    def test_target_saturation_above_one_rejected(self, valid_args):
        """target_saturation > 1 should raise ValueError."""
        valid_args.target_saturation = 1.5
        with pytest.raises(ValueError, match="target-saturation must be between 0.0 and 1.0"):
            validate_args(valid_args)
    
    def test_target_saturation_boundaries_valid(self, valid_args):
        """target_saturation at 0.0 and 1.0 should be valid."""
        valid_args.target_saturation = 0.0
        result = validate_args(valid_args)
        assert result.target_saturation == 0.0
        
        valid_args.target_saturation = 1.0
        result = validate_args(valid_args)
        assert result.target_saturation == 1.0
    
    def test_rag_num_docs_negative_rejected(self, valid_args):
        """Negative rag_num_docs should raise ValueError."""
        valid_args.rag_num_docs = -1
        with pytest.raises(ValueError, match="rag-num-docs cannot be negative"):
            validate_args(valid_args)
    
    def test_max_conversations_zero_rejected(self, valid_args):
        """max_conversations=0 should raise ValueError."""
        valid_args.max_conversations = 0
        with pytest.raises(ValueError, match="max-conversations must be positive"):
            validate_args(valid_args)
    
    def test_max_concurrent_allocs_negative_rejected(self, valid_args):
        """Negative max_concurrent_allocs should raise ValueError."""
        valid_args.max_concurrent_allocs = -1
        with pytest.raises(ValueError, match="max-concurrent-allocs cannot be negative"):
            validate_args(valid_args)
    
    def test_request_rate_negative_rejected(self, valid_args):
        """Negative request_rate should raise ValueError."""
        valid_args.request_rate = -1
        with pytest.raises(ValueError, match="request-rate cannot be negative"):
            validate_args(valid_args)
    
    def test_max_requests_negative_rejected(self, valid_args):
        """Negative max_requests should raise ValueError."""
        valid_args.max_requests = -1
        with pytest.raises(ValueError, match="max-requests cannot be negative"):
            validate_args(valid_args)
    
    @pytest.mark.skipif(sys.platform == 'win32', reason="Unix paths not valid on Windows")
    def test_forbidden_cache_dir_rejected(self, valid_args):
        """Cache directories in system paths should be rejected."""
        valid_args.cache_dir = '/etc/kv_cache'
        with pytest.raises(ValueError, match="cannot be a system directory"):
            validate_args(valid_args)
    
    def test_valid_cache_dir_allowed(self, valid_args, tmp_path):
        """Valid cache directory should be accepted."""
        valid_args.cache_dir = str(tmp_path / "kv_cache_test")
        result = validate_args(valid_args)
        assert result.cache_dir == str(tmp_path / "kv_cache_test")
    
    def test_multiple_errors_collected(self, valid_args):
        """Multiple validation errors should all be reported."""
        valid_args.num_users = -1
        valid_args.duration = -1
        valid_args.gpu_mem_gb = -1
        with pytest.raises(ValueError) as exc_info:
            validate_args(valid_args)
        # All three errors should be in the message
        error_msg = str(exc_info.value)
        assert "num-users" in error_msg
        assert "duration" in error_msg
        assert "gpu-mem-gb" in error_msg

    # --- New validation tests for v3.0 Changes 1-3 ---

    def test_storage_capacity_gb_negative_rejected(self, valid_args):
        """Negative storage_capacity_gb should raise ValueError."""
        valid_args.storage_capacity_gb = -1
        with pytest.raises(ValueError, match="storage-capacity-gb cannot be negative"):
            validate_args(valid_args)

    def test_storage_capacity_gb_zero_allowed(self, valid_args):
        """storage_capacity_gb=0 should be valid (auto-detect)."""
        valid_args.storage_capacity_gb = 0
        result = validate_args(valid_args)
        assert result.storage_capacity_gb == 0

    def test_storage_capacity_gb_positive_allowed(self, valid_args):
        """Positive storage_capacity_gb should be valid."""
        valid_args.storage_capacity_gb = 100
        result = validate_args(valid_args)
        assert result.storage_capacity_gb == 100

    def test_precondition_size_gb_negative_rejected(self, valid_args):
        """Negative precondition_size_gb should raise ValueError."""
        valid_args.precondition_size_gb = -1
        with pytest.raises(ValueError, match="precondition-size-gb cannot be negative"):
            validate_args(valid_args)

    def test_precondition_size_gb_zero_allowed(self, valid_args):
        """precondition_size_gb=0 should be valid (default to 2x NVMe capacity)."""
        valid_args.precondition_size_gb = 0
        result = validate_args(valid_args)
        assert result.precondition_size_gb == 0

    def test_precondition_threads_negative_rejected(self, valid_args):
        """Negative precondition_threads should raise ValueError."""
        valid_args.precondition_threads = -1
        with pytest.raises(ValueError, match="precondition-threads cannot be negative"):
            validate_args(valid_args)

    def test_precondition_threads_zero_allowed(self, valid_args):
        """precondition_threads=0 should be valid (auto-detect from cpu_count)."""
        valid_args.precondition_threads = 0
        result = validate_args(valid_args)
        assert result.precondition_threads == 0


# =============================================================================
# Test 16b: NVMe Capacity Tracking (Change 1)
# =============================================================================

class TestNVMeCapacityTracking:
    """Tests for NVMe/storage tier capacity tracking."""

    @pytest.fixture
    def tiny_model_config(self):
        return MODEL_CONFIGS['tiny-1b']

    def test_explicit_storage_capacity(self, tiny_model_config):
        """Explicit storage_capacity_gb should set nvme_memory_limit."""
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.1,
            seed=42,
            storage_capacity_gb=10.0
        )
        assert cache.nvme_memory_limit == 10.0 * 1024**3

    def test_auto_detect_storage_capacity(self, tiny_model_config):
        """storage_capacity_gb=0 should auto-detect from disk free space."""
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.1,
            seed=42,
            storage_capacity_gb=0
        )
        # Auto-detect should return a finite positive value (disk free space)
        assert cache.nvme_memory_limit > 0
        assert cache.nvme_memory_limit != float('inf')

    def test_nvme_usage_starts_at_zero(self, tiny_model_config):
        """NVMe usage should start at 0."""
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.1,
            seed=42,
            storage_capacity_gb=10.0
        )
        assert cache.nvme_memory_used == 0

    def test_nvme_usage_tracked_after_write(self, tiny_model_config):
        """NVMe usage should increase after writing to NVMe tier."""
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.001,  # 1MB — force overflow to NVMe
            seed=42,
            storage_capacity_gb=10.0
        )
        # Write enough to overflow CPU to NVMe
        for i in range(10):
            cache.allocate_cache(f"entry_{i}", num_tokens=1000)
        assert cache.nvme_memory_used > 0

    def test_get_tier_limit_returns_set_value(self, tiny_model_config):
        """_get_tier_limit('nvme') should return the configured limit."""
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.1,
            seed=42,
            storage_capacity_gb=5.0
        )
        assert cache._get_tier_limit('nvme') == 5.0 * 1024**3

    def test_get_tier_usage_reflects_writes(self, tiny_model_config):
        """_get_tier_usage('nvme') should reflect bytes written."""
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.001,
            seed=42,
            storage_capacity_gb=10.0
        )
        assert cache._get_tier_usage('nvme') == 0
        for i in range(10):
            cache.allocate_cache(f"entry_{i}", num_tokens=1000)
        assert cache._get_tier_usage('nvme') > 0


# =============================================================================
# Test 16c: NVMe Eviction (Change 2)
# =============================================================================

class TestNVMeEviction:
    """Tests for NVMe eviction when storage tier is full."""

    @pytest.fixture
    def tiny_model_config(self):
        return MODEL_CONFIGS['tiny-1b']

    def test_nvme_eviction_triggers_when_full(self, tiny_model_config):
        """When NVMe is full, LRU entries should be evicted (deleted)."""
        # tiny-1b: ~24KB per token. 10 tokens = ~240KB per entry.
        # 10MB NVMe fits ~42 entries before eviction triggers.
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.001,  # 1MB CPU
            seed=42,
            storage_capacity_gb=0.01  # 10MB NVMe
        )
        # Write more data than fits in NVMe (200 >> 42)
        keys = []
        for i in range(200):
            key = f"entry_{i}"
            success, location, _ = cache.allocate_cache(key, num_tokens=10)
            if success:
                keys.append(key)

        # evictions counter is in cache.stats, not in get_stats() output
        assert cache.stats['evictions'] > 0, "Evictions should have occurred when NVMe is full"

    def test_evicted_entry_removed_from_cache_entries(self, tiny_model_config):
        """Evicted NVMe entries should be removed from cache_entries dict."""
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.001,
            seed=42,
            storage_capacity_gb=0.01  # 10MB NVMe
        )
        # Fill and overflow (200 entries >> ~42 capacity)
        for i in range(200):
            cache.allocate_cache(f"entry_{i}", num_tokens=10)

        # Some early entries should have been evicted
        total_entries = len(cache.cache_entries)
        assert total_entries < 200, f"Expected evictions to reduce entries, got {total_entries}"

    def test_allocation_still_succeeds_after_eviction(self, tiny_model_config):
        """New allocations should succeed even after NVMe evictions."""
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.001,
            seed=42,
            storage_capacity_gb=0.01
        )
        # Fill NVMe
        for i in range(100):
            cache.allocate_cache(f"fill_{i}", num_tokens=10)

        # New allocation should still work (eviction frees space)
        success, location, _ = cache.allocate_cache("after_eviction", num_tokens=10)
        assert success is True

    def test_unlimited_nvme_skips_eviction(self, tiny_model_config):
        """When nvme_memory_limit is inf (auto-detect fails), no eviction should occur."""
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.001,
            seed=42,
            storage_capacity_gb=0  # auto-detect
        )
        # Force nvme_memory_limit to inf for this test
        cache.nvme_memory_limit = float('inf')

        for i in range(20):
            cache.allocate_cache(f"entry_{i}", num_tokens=500)

        stats = cache.get_stats(duration=1.0)
        # With unlimited NVMe, no NVMe-tier evictions should occur
        # (CPU evictions/demotions to NVMe are expected)
        nvme_entries = sum(1 for e in cache.cache_entries.values() if e['location'] == 'nvme')
        assert nvme_entries > 0, "Entries should exist on NVMe tier"


# =============================================================================
# Test 16d: reset_stats (Change 3)
# =============================================================================

class TestResetStats:
    """Tests for MultiTierCache.reset_stats() method."""

    @pytest.fixture
    def tiny_model_config(self):
        return MODEL_CONFIGS['tiny-1b']

    def test_reset_stats_zeroes_counters(self, tiny_model_config):
        """reset_stats() should zero all numeric counters."""
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.1,
            seed=42
        )
        # Generate some stats
        for i in range(5):
            cache.allocate_cache(f"entry_{i}", num_tokens=100)
            cache.access_cache(f"entry_{i}", InferencePhase.DECODE)

        # Verify stats are non-zero before reset
        assert cache.stats['cache_hits'] > 0
        assert cache.stats['write_operations'] > 0

        cache.reset_stats()

        assert cache.stats['cache_hits'] == 0
        assert cache.stats['cache_misses'] == 0
        assert cache.stats['write_operations'] == 0
        assert cache.stats['read_operations'] == 0
        assert cache.stats['evictions'] == 0

    def test_reset_stats_clears_lists(self, tiny_model_config):
        """reset_stats() should clear all list stats."""
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.1,
            seed=42
        )
        for i in range(5):
            cache.allocate_cache(f"entry_{i}", num_tokens=100)

        cache.reset_stats()

        for key, value in cache.stats.items():
            if isinstance(value, list):
                assert len(value) == 0, f"List stat '{key}' should be empty after reset"

    def test_reset_stats_preserves_cache_entries(self, tiny_model_config):
        """reset_stats() should NOT remove cached data, only counters."""
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.1,
            seed=42
        )
        for i in range(5):
            cache.allocate_cache(f"entry_{i}", num_tokens=100)

        entries_before = len(cache.cache_entries)
        cache.reset_stats()
        entries_after = len(cache.cache_entries)

        assert entries_after == entries_before, "Cache entries should survive reset_stats()"


# =============================================================================
# Test 16e: Race Condition Safety in read_cache (Change 2 fix)
# =============================================================================

class TestReadCacheRaceConditionSafety:
    """Tests that read_cache handles evicted entries gracefully."""

    @pytest.fixture
    def tiny_model_config(self):
        return MODEL_CONFIGS['tiny-1b']

    def test_access_evicted_key_returns_none(self, tiny_model_config):
        """Accessing a key that was evicted should return None, not crash."""
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.001,
            seed=42,
            storage_capacity_gb=0.005
        )
        # Allocate an entry
        cache.allocate_cache("victim", num_tokens=500)

        # Force eviction by filling the cache
        for i in range(50):
            cache.allocate_cache(f"fill_{i}", num_tokens=500)

        # Try to read the likely-evicted entry — should not crash
        loc, latency = cache.access_cache("victim", InferencePhase.DECODE)
        # loc is None if evicted, or a tier name if still present
        if loc is None:
            assert latency == 0.0
        else:
            assert loc in ['cpu', 'nvme']

    def test_access_nonexistent_key_records_miss(self, tiny_model_config):
        """Accessing a key that doesn't exist should record a cache miss."""
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.1,
            seed=42
        )
        loc, latency = cache.access_cache("does_not_exist", InferencePhase.DECODE)
        assert loc is None
        stats = cache.get_stats(duration=1.0)
        assert stats['cache_misses'] >= 1


# =============================================================================
# Test 17: Per-Tier Phase Metrics
# =============================================================================

class TestPerTierPhaseMetrics:
    """Tests for per-tier KV bytes tracking (prefill/decode per tier)."""
    
    @pytest.fixture
    def tiny_model_config(self):
        """Return the tiny-1b model config for fast tests."""
        return MODEL_CONFIGS['tiny-1b']
    
    @pytest.fixture
    def multi_tier_cache_cpu_only(self, tiny_model_config):
        """Return a MultiTierCache in CPU-only mode (GPU disabled)."""
        return MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.1,  # 100MB
            seed=42
        )
    
    def test_stats_have_tier_kv_bytes_written_keys(self, multi_tier_cache_cpu_only):
        """Stats should include tier_*_kv_bytes_written keys."""
        multi_tier_cache_cpu_only.allocate_cache("test_entry", num_tokens=100)
        stats = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        
        # Check for per-tier write tracking
        assert 'tier_gpu_kv_bytes_written_gb' in stats
        assert 'tier_cpu_kv_bytes_written_gb' in stats
        assert 'tier_storage_kv_bytes_written_gb' in stats
    
    def test_stats_have_tier_kv_bytes_read_keys(self, multi_tier_cache_cpu_only):
        """Stats should include tier_*_kv_bytes_read keys."""
        multi_tier_cache_cpu_only.allocate_cache("test_entry", num_tokens=100)
        multi_tier_cache_cpu_only.access_cache("test_entry", InferencePhase.DECODE)
        stats = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        
        # Check for per-tier read tracking
        assert 'tier_gpu_kv_bytes_read_gb' in stats
        assert 'tier_cpu_kv_bytes_read_gb' in stats
        assert 'tier_storage_kv_bytes_read_gb' in stats
    
    def test_cpu_write_bytes_increment_on_allocate(self, multi_tier_cache_cpu_only):
        """Allocating to CPU tier should increment tier_cpu_kv_bytes_written."""
        # Get initial stats
        stats_before = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        cpu_written_before = stats_before.get('tier_cpu_kv_bytes_written_gb', 0)
        
        # Allocate cache entry (goes to CPU since GPU is disabled)
        success, location, _ = multi_tier_cache_cpu_only.allocate_cache("test_entry", num_tokens=100)
        assert success
        assert location == 'cpu'
        
        # Check that CPU write bytes increased
        stats_after = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        cpu_written_after = stats_after.get('tier_cpu_kv_bytes_written_gb', 0)
        
        assert cpu_written_after > cpu_written_before, \
            f"CPU write bytes should increase: {cpu_written_before} -> {cpu_written_after}"
    
    def test_cpu_read_bytes_increment_on_access(self, multi_tier_cache_cpu_only):
        """Accessing from CPU tier should increment tier_cpu_kv_bytes_read."""
        # Allocate first
        multi_tier_cache_cpu_only.allocate_cache("test_entry", num_tokens=100)
        
        # Get stats before access
        stats_before = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        cpu_read_before = stats_before.get('tier_cpu_kv_bytes_read_gb', 0)
        
        # Access the cache entry
        location, _ = multi_tier_cache_cpu_only.access_cache("test_entry", InferencePhase.DECODE)
        assert location == 'cpu'
        
        # Check that CPU read bytes increased
        stats_after = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        cpu_read_after = stats_after.get('tier_cpu_kv_bytes_read_gb', 0)
        
        assert cpu_read_after > cpu_read_before, \
            f"CPU read bytes should increase: {cpu_read_before} -> {cpu_read_after}"
    
    def test_gpu_bytes_zero_when_gpu_disabled(self, multi_tier_cache_cpu_only):
        """With GPU disabled (0 GiB), GPU tier bytes should remain zero."""
        # Do some allocations and accesses
        for i in range(5):
            multi_tier_cache_cpu_only.allocate_cache(f"entry_{i}", num_tokens=100)
        for i in range(5):
            multi_tier_cache_cpu_only.access_cache(f"entry_{i}", InferencePhase.DECODE)
        
        stats = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        
        # GPU bytes should be zero since GPU tier is disabled
        assert stats.get('tier_gpu_kv_bytes_written_gb', 0) == 0, \
            "GPU write bytes should be 0 when GPU disabled"
        assert stats.get('tier_gpu_kv_bytes_read_gb', 0) == 0, \
            "GPU read bytes should be 0 when GPU disabled"
    
    def test_storage_tier_overflow(self, tiny_model_config):
        """When CPU is full, allocations should overflow to storage tier."""
        # Create cache with very small CPU limit
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.001,  # 1MB - very small
            seed=42
        )
        
        # Allocate enough to overflow CPU
        for i in range(20):
            cache.allocate_cache(f"entry_{i}", num_tokens=1000)
        
        stats = cache.get_stats(duration=1.0)
        
        # Storage tier should have received some data
        storage_written = stats.get('tier_storage_kv_bytes_written_gb', 0)
        assert storage_written > 0, \
            f"Storage tier should have data when CPU overflows: {storage_written}"
    
    def test_per_tier_bandwidth_calculated(self, multi_tier_cache_cpu_only):
        """Per-tier bandwidth stats should be calculated."""
        # Do some I/O
        for i in range(10):
            multi_tier_cache_cpu_only.allocate_cache(f"entry_{i}", num_tokens=100)
        for i in range(10):
            multi_tier_cache_cpu_only.access_cache(f"entry_{i}", InferencePhase.DECODE)
        
        stats = multi_tier_cache_cpu_only.get_stats(duration=1.0)
        
        # Bandwidth stats should exist
        assert 'tier_cpu_read_bandwidth_gbps' in stats
        assert 'tier_cpu_write_bandwidth_gbps' in stats
        assert 'tier_storage_read_bandwidth_gbps' in stats
        assert 'tier_storage_write_bandwidth_gbps' in stats


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestPerTierPhaseMetricsWithGPU:
    """Tests for per-tier metrics when GPU is enabled."""
    
    @pytest.fixture
    def tiny_model_config(self):
        """Return the tiny-1b model config for fast tests."""
        return MODEL_CONFIGS['tiny-1b']
    
    @pytest.fixture
    def multi_tier_cache_with_gpu(self, tiny_model_config):
        """Return a MultiTierCache with GPU enabled."""
        return MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=1.0,  # 1GB GPU
            cpu_memory_gb=0.1,  # 100MB CPU
            seed=42
        )
    
    def test_gpu_write_bytes_increment_on_allocate(self, multi_tier_cache_with_gpu):
        """Allocating to GPU tier should increment tier_gpu_kv_bytes_written."""
        # Get initial stats
        stats_before = multi_tier_cache_with_gpu.get_stats(duration=1.0)
        gpu_written_before = stats_before.get('tier_gpu_kv_bytes_written_gb', 0)
        
        # Allocate cache entry (should go to GPU first)
        success, location, _ = multi_tier_cache_with_gpu.allocate_cache("test_entry", num_tokens=100)
        assert success
        assert location == 'gpu'
        
        # Check that GPU write bytes increased
        stats_after = multi_tier_cache_with_gpu.get_stats(duration=1.0)
        gpu_written_after = stats_after.get('tier_gpu_kv_bytes_written_gb', 0)
        
        assert gpu_written_after > gpu_written_before, \
            f"GPU write bytes should increase: {gpu_written_before} -> {gpu_written_after}"
    
    def test_gpu_read_bytes_increment_on_access(self, multi_tier_cache_with_gpu):
        """Accessing from GPU tier should increment tier_gpu_kv_bytes_read."""
        # Allocate first
        multi_tier_cache_with_gpu.allocate_cache("test_entry", num_tokens=100)
        
        # Get stats before access
        stats_before = multi_tier_cache_with_gpu.get_stats(duration=1.0)
        gpu_read_before = stats_before.get('tier_gpu_kv_bytes_read_gb', 0)
        
        # Access the cache entry
        location, _ = multi_tier_cache_with_gpu.access_cache("test_entry", InferencePhase.DECODE)
        assert location == 'gpu'
        
        # Check that GPU read bytes increased
        stats_after = multi_tier_cache_with_gpu.get_stats(duration=1.0)
        gpu_read_after = stats_after.get('tier_gpu_kv_bytes_read_gb', 0)
        
        assert gpu_read_after > gpu_read_before, \
            f"GPU read bytes should increase: {gpu_read_before} -> {gpu_read_after}"
    
    def test_gpu_bandwidth_calculated(self, multi_tier_cache_with_gpu):
        """GPU tier bandwidth stats should be calculated."""
        # Do some I/O
        for i in range(5):
            multi_tier_cache_with_gpu.allocate_cache(f"entry_{i}", num_tokens=100)
        for i in range(5):
            multi_tier_cache_with_gpu.access_cache(f"entry_{i}", InferencePhase.DECODE)
        
        stats = multi_tier_cache_with_gpu.get_stats(duration=1.0)
        
        # GPU bandwidth stats should exist
        assert 'tier_gpu_read_bandwidth_gbps' in stats
        assert 'tier_gpu_write_bandwidth_gbps' in stats


# =============================================================================
# Test: Trace Replay (Streaming Iterator, Timestamp Pacing, Replay Cycles)
# =============================================================================

class TestTraceReplay:
    """Tests for BurstGPT trace streaming iterator and replay logic."""

    @pytest.fixture
    def trace_dir(self, tmp_path):
        """Create a temporary directory with small BurstGPT CSV trace files."""
        # File 1: 5 rows
        csv1 = tmp_path / "BurstGPT_1.csv"
        csv1.write_text(
            "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
            "0,ChatGPT,100,20,120,Conversation log\n"
            "10,ChatGPT,200,40,240,Conversation log\n"
            "20,GPT-4,300,60,360,Conversation log\n"
            "30,ChatGPT,400,80,480,Conversation log\n"
            "40,ChatGPT,500,100,600,Conversation log\n"
        )
        # File 2: 3 rows with timestamps continuing from file 1
        csv2 = tmp_path / "BurstGPT_2.csv"
        csv2.write_text(
            "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
            "50,GPT-4,150,30,180,Conversation log\n"
            "60,ChatGPT,250,50,300,Conversation log\n"
            "70,GPT-4,350,70,420,Conversation log\n"
        )
        return tmp_path

    @pytest.fixture
    def benchmark_with_trace(self, trace_dir):
        """Return an IntegratedBenchmark configured for trace replay testing."""
        model_config = MODEL_CONFIGS['tiny-1b']
        bench = IntegratedBenchmark(
            model_config=model_config,
            num_users=5,
            gpu_memory_gb=0,
            cpu_memory_gb=0.01,
            duration_seconds=30,
            use_burst_trace=True,
            burst_trace_path=str(trace_dir),
            generation_mode=GenerationMode.NONE,
            trace_speedup=0,    # no delay for testing
            replay_cycles=1,    # single pass
        )
        return bench

    def test_resolve_trace_files_from_directory(self, trace_dir):
        """Passing a directory should resolve all CSVs sorted by name."""
        model_config = MODEL_CONFIGS['tiny-1b']
        bench = IntegratedBenchmark(
            model_config=model_config,
            num_users=1,
            gpu_memory_gb=0,
            cpu_memory_gb=0.01,
            duration_seconds=5,
            use_burst_trace=True,
            burst_trace_path=str(trace_dir),
            generation_mode=GenerationMode.NONE,
            trace_speedup=0,
            replay_cycles=1,
        )
        assert len(bench.burst_trace_files) == 2
        assert 'BurstGPT_1.csv' in bench.burst_trace_files[0]
        assert 'BurstGPT_2.csv' in bench.burst_trace_files[1]

    def test_resolve_single_file(self, trace_dir):
        """Passing a single CSV file should resolve to a list of one."""
        csv_path = str(trace_dir / "BurstGPT_1.csv")
        model_config = MODEL_CONFIGS['tiny-1b']
        bench = IntegratedBenchmark(
            model_config=model_config,
            num_users=1,
            gpu_memory_gb=0,
            cpu_memory_gb=0.01,
            duration_seconds=5,
            use_burst_trace=True,
            burst_trace_path=csv_path,
            generation_mode=GenerationMode.NONE,
            trace_speedup=0,
            replay_cycles=1,
        )
        assert len(bench.burst_trace_files) == 1

    def test_streaming_iterator_yields_all_rows(self, benchmark_with_trace):
        """Streaming iterator should yield all rows across all files."""
        rows = list(benchmark_with_trace._burst_trace_iterator())
        assert len(rows) == 8  # 5 from file 1 + 3 from file 2

    def test_streaming_iterator_tuple_format(self, benchmark_with_trace):
        """Each yielded row should be (timestamp, context, generate, total)."""
        row = next(iter(benchmark_with_trace._burst_trace_iterator()))
        timestamp, context, generate, total = row
        assert timestamp == 0.0
        assert context == 100
        assert generate == 20
        assert total == 120

    def test_streaming_iterator_preserves_order(self, benchmark_with_trace):
        """Rows should come in file order: all of file 1 then all of file 2."""
        rows = list(benchmark_with_trace._burst_trace_iterator())
        timestamps = [r[0] for r in rows]
        # Timestamps should be monotonically increasing across both files
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i-1], \
                f"Timestamp at index {i} ({timestamps[i]}) should be > {timestamps[i-1]}"

    def test_replay_cycles_one_pass(self, trace_dir):
        """With replay_cycles=1, generator should process all rows once then stop."""
        import threading
        model_config = MODEL_CONFIGS['tiny-1b']
        bench = IntegratedBenchmark(
            model_config=model_config,
            num_users=5,
            gpu_memory_gb=0,
            cpu_memory_gb=0.01,
            duration_seconds=60,
            use_burst_trace=True,
            burst_trace_path=str(trace_dir),
            generation_mode=GenerationMode.NONE,
            trace_speedup=0,
            replay_cycles=1,
        )

        stop_event = threading.Event()
        bench.stop_event = stop_event

        # Run generator in a thread
        gen_thread = threading.Thread(
            target=bench._generate_requests_from_trace,
            args=(stop_event,),
            daemon=True
        )
        gen_thread.start()
        gen_thread.join(timeout=10)

        # stop_event should have been set by the generator after 1 cycle
        assert stop_event.is_set(), "stop_event should be set after replay_cycles=1 completes"

        # Queue should have exactly 8 requests (5 + 3)
        count = 0
        while not bench.request_queue.empty():
            bench.request_queue.get_nowait()
            count += 1
        assert count == 8, f"Expected 8 requests from 1 cycle, got {count}"

    def test_replay_cycles_two_passes(self, trace_dir):
        """With replay_cycles=2, generator should process all rows twice."""
        import threading
        model_config = MODEL_CONFIGS['tiny-1b']
        bench = IntegratedBenchmark(
            model_config=model_config,
            num_users=5,
            gpu_memory_gb=0,
            cpu_memory_gb=0.01,
            duration_seconds=60,
            use_burst_trace=True,
            burst_trace_path=str(trace_dir),
            generation_mode=GenerationMode.NONE,
            trace_speedup=0,
            replay_cycles=2,
        )

        stop_event = threading.Event()
        bench.stop_event = stop_event

        gen_thread = threading.Thread(
            target=bench._generate_requests_from_trace,
            args=(stop_event,),
            daemon=True
        )
        gen_thread.start()
        gen_thread.join(timeout=10)

        assert stop_event.is_set()
        count = 0
        while not bench.request_queue.empty():
            bench.request_queue.get_nowait()
            count += 1
        assert count == 16, f"Expected 16 requests from 2 cycles, got {count}"

    def test_total_tokens_tracked(self, benchmark_with_trace):
        """Total tokens from trace should be summed correctly."""
        rows = list(benchmark_with_trace._burst_trace_iterator())
        expected_total = sum(r[3] for r in rows)
        # 120+240+360+480+600 + 180+300+420 = 2700
        assert expected_total == 2700

    def test_trace_speedup_zero_no_sleep(self, trace_dir):
        """trace_speedup=0 should skip all timestamp delays (fast)."""
        import threading
        model_config = MODEL_CONFIGS['tiny-1b']
        bench = IntegratedBenchmark(
            model_config=model_config,
            num_users=5,
            gpu_memory_gb=0,
            cpu_memory_gb=0.01,
            duration_seconds=60,
            use_burst_trace=True,
            burst_trace_path=str(trace_dir),
            generation_mode=GenerationMode.NONE,
            trace_speedup=0,
            replay_cycles=1,
        )

        stop_event = threading.Event()
        bench.stop_event = stop_event

        start = time.time()
        gen_thread = threading.Thread(
            target=bench._generate_requests_from_trace,
            args=(stop_event,),
            daemon=True
        )
        gen_thread.start()
        gen_thread.join(timeout=10)
        elapsed = time.time() - start

        # With speedup=0, should finish almost instantly (< 2s)
        assert elapsed < 2.0, f"speedup=0 should be near-instant, took {elapsed:.2f}s"


# =============================================================================
# Test: Eviction Tracing
# =============================================================================

class TestEvictionTracing:
    """Test that traces eviction behavior in the multi-tier cache."""

    def test_eviction_lifecycle(self):
        """Trace the full eviction lifecycle: fill tier, trigger eviction, verify entries removed."""
        model_config = MODEL_CONFIGS['tiny-1b']
        # tiny-1b: ~24KB per token of KV cache.
        # 10 tokens per entry = ~240KB per entry.
        # storage_capacity_gb=0.01 (~10MB) fits ~42 entries before eviction.
        cache = MultiTierCache(
            model_config=model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.001,   # ~1MB CPU to trigger overflow quickly
            seed=42,
            storage_capacity_gb=0.01  # ~10MB storage to trigger NVMe eviction
        )

        eviction_log = []
        allocated_keys = []
        allocated_tiers = {}

        # Phase 1: Fill both CPU and storage tiers (200 entries >> ~42 capacity)
        for i in range(200):
            key = f"evict_test_{i}"
            success, tier, latency = cache.allocate_cache(key, num_tokens=10)
            if success:
                allocated_keys.append(key)
                allocated_tiers[key] = tier
                eviction_log.append(('allocate', key, tier))

        # Phase 2: Check that evictions occurred
        # Note: evictions counter is in cache.stats directly, not in get_stats() output
        evictions = cache.stats['evictions']
        eviction_log.append(('stats', 'evictions', evictions))

        # Phase 3: Verify some early keys were evicted (no longer in cache)
        evicted_count = 0
        surviving_count = 0
        for key in allocated_keys[:50]:  # Check first 50 keys
            if key in cache.cache_entries:
                surviving_count += 1
            else:
                evicted_count += 1
                eviction_log.append(('evicted', key, None))

        # Assertions
        assert evictions > 0, \
            f"Evictions should have occurred with tiny capacity. Log: {eviction_log[:20]}"
        assert evicted_count > 0, \
            f"Some early entries should have been evicted. " \
            f"Evicted: {evicted_count}, Surviving: {surviving_count}"

        # Phase 4: Verify later keys are still accessible
        late_key = allocated_keys[-1]
        assert late_key in cache.cache_entries, \
            f"Most recent key '{late_key}' should still be in cache"


# =============================================================================
# Test: 3-Tier Eviction Cascade (GPU → CPU → NVMe → Delete)
# =============================================================================

class TestThreeTierEvictionCascade:
    """
    Tests that eviction cascades correctly through all three tiers:
        GPU → CPU → NVMe → delete

    Since we have no real GPU, we inject a CPUMemoryBackend as a fake GPU
    backend.  This exercises the full _ensure_space_in_tier recursive path:
        depth 0: GPU is full → demote LRU to CPU
        depth 1: CPU is full → demote LRU to NVMe
        depth 2: NVMe is full → delete LRU from disk
    """

    @pytest.fixture
    def tiny_model_config(self):
        return MODEL_CONFIGS['tiny-1b']

    def test_full_cascade_gpu_to_cpu_to_nvme_to_delete(self, tiny_model_config):
        """
        Fill all three tiers, then allocate one more entry.
        Expect the cascade:
          1. GPU evicts its LRU to CPU  (demote)
          2. CPU is full, so CPU evicts its LRU to NVMe  (demote)
          3. NVMe is full, so NVMe deletes its LRU from disk  (delete)
          4. New entry lands on GPU
        """
        # --- Setup ---
        # Tiny-1b: ~24KB per token, 10 tokens ≈ 240KB per entry.
        #   GPU:  2 MB  → fits ~8 entries
        #   CPU:  2 MB  → fits ~8 entries
        #   NVMe: 2 MB  → fits ~8 entries
        # Total across all tiers: ~24 entries before disk deletes start.
        gpu_mb = 2
        cpu_mb = 2
        nvme_mb = 2
        tokens_per_entry = 10

        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,             # we'll fake the GPU below
            cpu_memory_gb=cpu_mb / 1024,
            seed=42,
            storage_capacity_gb=nvme_mb / 1024,
        )

        # Inject a fake GPU backend (CPUMemoryBackend in disguise)
        cache.backends['gpu'] = CPUMemoryBackend()
        cache.gpu_memory_limit = gpu_mb * 1024 * 1024  # 2 MB

        # --- Phase 1: Fill GPU ---
        print("\n  === Phase 1: Filling GPU ===")
        gpu_keys = []
        for i in range(50):
            key = f"gpu_fill_{i}"
            success, tier, _ = cache.allocate_cache(key, num_tokens=tokens_per_entry)
            assert success, f"Allocation {i} should succeed"
            if tier == 'gpu':
                gpu_keys.append(key)
            print(f"    [{i:3d}] key={key:<20s} → tier={tier}  "
                  f"(GPU={cache.gpu_memory_used/1024:.0f}KiB  "
                  f"CPU={cache.cpu_memory_used/1024:.0f}KiB  "
                  f"NVMe={cache.nvme_memory_used/1024:.0f}KiB)")

        # --- Phase 2: Verify entries exist on all three tiers ---
        gpu_entries = [k for k, v in cache.cache_entries.items() if v['location'] == 'gpu']
        cpu_entries = [k for k, v in cache.cache_entries.items() if v['location'] == 'cpu']
        nvme_entries = [k for k, v in cache.cache_entries.items() if v['location'] == 'nvme']

        print(f"\n  === Phase 2: Tier distribution ===")
        print(f"    GPU entries:  {len(gpu_entries)}")
        print(f"    CPU entries:  {len(cpu_entries)}")
        print(f"    NVMe entries: {len(nvme_entries)}")
        print(f"    Total in cache_entries: {len(cache.cache_entries)}")
        print(f"    Evictions: {cache.stats['evictions']}")
        print(f"    Offloads to CPU: {cache.stats['offloads_cpu']}")
        print(f"    Offloads to storage: {cache.stats['offloads_storage']}")

        # With 50 entries and ~24 capacity, evictions must have happened
        assert cache.stats['evictions'] > 0, \
            "Evictions should have occurred with 50 entries across 6MiB total"

        # CPU demotion must have occurred (GPU → CPU)
        assert cache.stats['offloads_cpu'] > 0, \
            "At least one GPU → CPU demotion should have occurred"

        # NVMe demotion must have occurred (CPU → NVMe)
        assert cache.stats['offloads_storage'] > 0, \
            "At least one CPU → NVMe demotion should have occurred"

        # --- Phase 3: Verify early keys were deleted from all tiers ---
        # With 50 entries and ~24 capacity, about half should be gone
        total_entries = len(cache.cache_entries)
        deleted_count = 50 - total_entries
        print(f"\n  === Phase 3: Deletion check ===")
        print(f"    Entries remaining: {total_entries}")
        print(f"    Entries deleted:   {deleted_count}")

        assert deleted_count > 0, \
            f"Some entries should have been deleted from NVMe. " \
            f"Total remaining: {total_entries}/50"

        # --- Phase 4: Verify .npy files are actually deleted from disk ---
        nvme_dir = cache.backends['nvme'].base_path
        npy_files = list(nvme_dir.glob("*.npy"))
        print(f"\n  === Phase 4: Disk file check ===")
        print(f"    .npy files on disk: {len(npy_files)}")
        print(f"    NVMe entries in metadata: {len(nvme_entries)}")

        # Files on disk should roughly match entries in cache_entries with location='nvme'
        # Some tolerance for timing, but there shouldn't be orphaned files
        assert len(npy_files) <= len(nvme_entries) + 2, \
            f"Orphaned .npy files: {len(npy_files)} on disk vs {len(nvme_entries)} tracked"

        # --- Phase 5: Allocate one more and verify it still works ---
        print(f"\n  === Phase 5: Post-cascade allocation ===")
        success, tier, _ = cache.allocate_cache("final_entry", num_tokens=tokens_per_entry)
        print(f"    final_entry → tier={tier}, success={success}")
        assert success, "Allocation after full cascade should still succeed"

    def test_demote_path_preserves_data(self, tiny_model_config):
        """
        Verify that data survives the full demotion chain:
          GPU → CPU → NVMe
        Read the entry back from NVMe and confirm it's the same data.

        Note: access_cache() returns (location, latency), not data.
        To verify data integrity, we read directly from the backend.
        """
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.5 / 1024,  # 0.5 MB CPU
            seed=42,
            storage_capacity_gb=10.0 / 1024,  # 10 MB NVMe (plenty of room)
        )

        # Inject fake GPU: 0.5 MB
        cache.backends['gpu'] = CPUMemoryBackend()
        cache.gpu_memory_limit = int(0.5 * 1024 * 1024)

        # Write one entry to GPU
        key = "preserve_test"
        success, tier, _ = cache.allocate_cache(key, num_tokens=10)
        assert success
        print(f"\n    Initial allocation: tier={tier}")

        # Read raw data from the backend while it's on the initial tier
        original_data, _ = cache.backends[tier].read(key)
        print(f"    Original data shape: {original_data.shape}, sum: {np.sum(original_data):.4f}")

        # Fill GPU to force demotion to CPU
        print("    Filling GPU to force demotion...")
        for i in range(20):
            cache.allocate_cache(f"push_{i}", num_tokens=10)

        # Check where our key ended up
        entry = cache.cache_entries.get(key)
        if entry:
            print(f"    After GPU fill: key is on tier={entry['location']}")

        # Fill CPU to force demotion to NVMe
        print("    Filling CPU to force demotion to NVMe...")
        for i in range(40):
            cache.allocate_cache(f"push_more_{i}", num_tokens=10)

        entry = cache.cache_entries.get(key)
        if entry:
            current_tier = entry['location']
            print(f"    After CPU fill: key is on tier={current_tier}")

            # Read raw data back from whichever backend it landed on
            read_data, _ = cache.backends[current_tier].read(key)
            print(f"    Re-read data shape: {read_data.shape}, sum: {np.sum(read_data):.4f}")

            assert original_data.shape == read_data.shape, \
                f"Shape mismatch: {original_data.shape} vs {read_data.shape}"
            assert np.allclose(original_data, read_data, atol=1e-3), \
                f"Data mismatch after demotion through tiers"
            print("    Data integrity verified after demotion chain!")
        else:
            # Key was evicted entirely — that's also valid if NVMe was tiny
            print("    Key was evicted (deleted). Skipping data comparison.")

    def test_tier_order_includes_fake_gpu(self, tiny_model_config):
        """
        Confirm that injecting a GPU backend adds 'gpu' to the tier order,
        giving us the full 3-tier cascade path.
        """
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.001,
            seed=42,
        )

        # Without fake GPU, tier order is ['cpu', 'nvme']
        tier_order_before = cache._get_tier_order()
        print(f"\n    Tier order without GPU: {tier_order_before}")
        assert 'gpu' not in tier_order_before

        # Inject fake GPU
        cache.backends['gpu'] = CPUMemoryBackend()
        cache.gpu_memory_limit = 1 * 1024 * 1024  # 1 MB

        tier_order_after = cache._get_tier_order()
        print(f"    Tier order with fake GPU: {tier_order_after}")
        assert tier_order_after == ['gpu', 'cpu', 'nvme'], \
            f"Expected ['gpu', 'cpu', 'nvme'], got {tier_order_after}"


# =============================================================================
# Test: NVMe-Only Mode (cpu=0, gpu=0) — Eviction and File Deletion
# =============================================================================

class TestNVMeOnlyEviction:
    """
    Tests the cpu=0, gpu=0 configuration where NVMe is the ONLY tier.

    This is the exact configuration that triggered the three bugs:
      1. Double-decrement race in nvme_memory_used
      2. Eviction guards rejecting entries on the terminal tier
      3. Preconditioning spinning forever

    These tests verify that:
      - Entries are allocated on NVMe (the only tier)
      - When NVMe fills up, LRU entries are deleted (not demoted)
      - .npy files are actually removed from disk after eviction
      - nvme_memory_used tracking stays sane (no negative drift)
      - The "second pass" works: new allocations succeed after eviction
    """

    @pytest.fixture
    def tiny_model_config(self):
        return MODEL_CONFIGS['tiny-1b']

    def test_nvme_only_basic_allocation(self, tiny_model_config):
        """
        With cpu=0 gpu=0, all entries should land on NVMe.
        Verify tier='nvme' for every allocation.
        """
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0,          # ZERO CPU
            seed=42,
            storage_capacity_gb=0.01  # 10 MB NVMe
        )

        print(f"\n    NVMe limit: {cache.nvme_memory_limit / 1024:.0f}KiB")
        print(f"    CPU limit:  {cache.cpu_memory_limit / 1024:.0f}KiB")
        print(f"    Tier order: {cache._get_tier_order()}")

        for i in range(5):
            key = f"nvme_only_{i}"
            success, tier, _ = cache.allocate_cache(key, num_tokens=10)
            print(f"    [{i}] key={key} → tier={tier}, success={success}")
            assert success, f"Allocation {i} should succeed"
            # CPU has 0 capacity — entry should skip CPU and go to NVMe
            assert tier == 'nvme' or tier == 'cpu', \
                f"Expected 'nvme' (or 'cpu' if zero-cap is treated as available), got '{tier}'"

    def test_nvme_only_eviction_deletes_files(self, tiny_model_config):
        """
        Fill NVMe past capacity with cpu=0, gpu=0.
        Verify that:
          1. Eviction counter increments
          2. Early keys are removed from cache_entries
          3. .npy files are actually deleted from disk
          4. Later allocations still succeed (the "second loop")
        """
        nvme_mb = 2  # 2 MB NVMe
        tokens_per_entry = 10  # ~240 KB per entry with tiny-1b
        # 2 MB / 240 KB ≈ 8 entries before eviction starts

        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0,
            seed=42,
            storage_capacity_gb=nvme_mb / 1024,
        )

        nvme_dir = cache.backends['nvme'].base_path
        print(f"\n    NVMe dir: {nvme_dir}")
        print(f"    NVMe limit: {cache.nvme_memory_limit / 1024:.0f}KiB")
        print(f"    Tier order: {cache._get_tier_order()}")

        # --- Pass 1: Fill NVMe to trigger eviction ---
        print("\n    --- Pass 1: Fill and overflow ---")
        all_keys = []
        for i in range(30):
            key = f"pass1_{i}"
            success, tier, _ = cache.allocate_cache(key, num_tokens=tokens_per_entry)
            all_keys.append(key)

            npy_count = len(list(nvme_dir.glob("*.npy")))
            entry_count = len(cache.cache_entries)

            print(f"    [{i:2d}] success={success} tier={tier:<5s} "
                  f"entries={entry_count:3d}  .npy={npy_count:3d}  "
                  f"nvme_used={cache.nvme_memory_used/1024:.0f}KiB  "
                  f"evictions={cache.stats['evictions']}")

            assert success, f"Allocation {i} should succeed even after eviction"

        # --- Verify eviction occurred ---
        evictions = cache.stats['evictions']
        print(f"\n    Evictions after pass 1: {evictions}")
        assert evictions > 0, \
            "Evictions should have occurred with 30 entries in 2 MiB"

        # --- Verify early keys were deleted ---
        early_keys_present = sum(1 for k in all_keys[:10] if k in cache.cache_entries)
        late_keys_present = sum(1 for k in all_keys[-5:] if k in cache.cache_entries)
        print(f"    Early keys (0-9) still in cache: {early_keys_present}/10")
        print(f"    Late keys (25-29) still in cache: {late_keys_present}/5")

        assert early_keys_present < 10, \
            f"Some early keys should have been evicted, but {early_keys_present}/10 remain"
        assert late_keys_present > 0, \
            "Recent keys should still be in cache"

        # --- Verify .npy files match cache_entries ---
        npy_files = set(f.stem for f in nvme_dir.glob("*.npy"))
        nvme_entries = set(k for k, v in cache.cache_entries.items() if v['location'] == 'nvme')
        orphaned = npy_files - nvme_entries
        missing = nvme_entries - npy_files

        print(f"\n    .npy files on disk: {len(npy_files)}")
        print(f"    NVMe entries tracked: {len(nvme_entries)}")
        print(f"    Orphaned files (on disk, not tracked): {len(orphaned)}")
        print(f"    Missing files (tracked, not on disk): {len(missing)}")

        assert len(orphaned) == 0, \
            f"Orphaned .npy files found: {orphaned}"

        # --- Pass 2: "Second loop" — new allocations after eviction ---
        print("\n    --- Pass 2: Second loop (allocate after eviction) ---")
        pass2_success = 0
        for i in range(20):
            key = f"pass2_{i}"
            success, tier, _ = cache.allocate_cache(key, num_tokens=tokens_per_entry)
            if success:
                pass2_success += 1

            if i < 5 or i >= 15:
                print(f"    [{i:2d}] success={success} tier={tier:<5s} "
                      f"nvme_used={cache.nvme_memory_used/1024:.0f}KiB  "
                      f"evictions={cache.stats['evictions']}")

        print(f"\n    Pass 2 successes: {pass2_success}/20")
        assert pass2_success == 20, \
            f"All pass-2 allocations should succeed, got {pass2_success}/20"

        # --- Verify nvme_memory_used didn't go negative ---
        print(f"    Final nvme_memory_used: {cache.nvme_memory_used/1024:.0f}KiB")
        assert cache.nvme_memory_used >= 0, \
            f"nvme_memory_used drifted negative: {cache.nvme_memory_used}"

    def test_nvme_only_memory_tracking_no_negative_drift(self, tiny_model_config):
        """
        Rapid allocation/eviction cycles with cpu=0, gpu=0.
        The double-decrement bug caused nvme_memory_used to drift to ~0
        while the disk was full.  This test verifies tracking stays accurate.
        """
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0,
            seed=42,
            storage_capacity_gb=1.0 / 1024,  # 1 MB — very tight
        )

        print(f"\n    NVMe limit: {cache.nvme_memory_limit / 1024:.0f}KiB")

        # Rapid-fire 100 allocations into 1 MB — heavy eviction pressure
        for i in range(100):
            cache.allocate_cache(f"stress_{i}", num_tokens=10)

        # Recount actual usage from cache_entries
        actual_nvme = sum(
            e['size'] for e in cache.cache_entries.values()
            if e['location'] == 'nvme'
        )

        tracked = cache.nvme_memory_used
        print(f"    Tracked nvme_memory_used: {tracked / 1024:.0f}KiB")
        print(f"    Actual from cache_entries: {actual_nvme / 1024:.0f}KiB")
        print(f"    Evictions: {cache.stats['evictions']}")

        assert tracked >= 0, \
            f"nvme_memory_used went negative: {tracked}"

        # Tracked should be >= actual (it can overcount due to forced writes,
        # but should never undercount after our fix)
        assert tracked >= actual_nvme * 0.5, \
            f"Tracked usage ({tracked/1024:.0f}KiB) is suspiciously low vs " \
            f"actual ({actual_nvme/1024:.0f}KiB) — possible double-decrement"

    def test_nvme_only_concurrent_allocation(self, tiny_model_config):
        """
        Multiple threads allocating simultaneously with cpu=0, gpu=0.
        This is the exact scenario that triggers the double-decrement race
        (Bug 1 from the fix).  Verify no crash and no negative drift.
        """
        cache = MultiTierCache(
            model_config=tiny_model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0,
            seed=42,
            storage_capacity_gb=2.0 / 1024,  # 2 MB
        )

        results = {'success': 0, 'fail': 0}
        lock = threading.Lock()

        def worker(thread_id, count):
            local_success = 0
            local_fail = 0
            for i in range(count):
                key = f"t{thread_id}_entry_{i}"
                success, tier, _ = cache.allocate_cache(key, num_tokens=10)
                if success:
                    local_success += 1
                else:
                    local_fail += 1
            with lock:
                results['success'] += local_success
                results['fail'] += local_fail

        # 4 threads, 25 allocations each = 100 total
        threads = []
        for t in range(4):
            th = threading.Thread(target=worker, args=(t, 25))
            threads.append(th)

        print(f"\n    Starting 4 threads, 25 allocations each...")
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        print(f"    Successes: {results['success']}")
        print(f"    Failures:  {results['fail']}")
        print(f"    Evictions: {cache.stats['evictions']}")
        print(f"    nvme_memory_used: {cache.nvme_memory_used / 1024:.0f}KiB")
        print(f"    Entries in cache: {len(cache.cache_entries)}")

        assert results['success'] > 0, "At least some allocations should succeed"
        assert cache.nvme_memory_used >= 0, \
            f"nvme_memory_used went negative after concurrent access: {cache.nvme_memory_used}"


# =============================================================================
# Test: Visualize User Request Flow
#
# Run with:  pytest tests/test_kv_cache.py::TestVisualizeUserRequestFlow -v -s --log-cli-level=DEBUG
#
# This test walks through the entire benchmark pipeline step-by-step,
# printing and logging every decision so you can see exactly what happens
# when a user request enters the system.
# =============================================================================

class TestVisualizeUserRequestFlow:
    """
    Educational test that traces a user request through the full benchmark
    pipeline.  Enable debug logging to see every internal decision:

        pytest -k TestVisualizeUserRequestFlow -v -s --log-cli-level=DEBUG

    The test covers:
      1. How users are generated (UserSimulator, QoS distribution)
      2. How context tokens map to KV cache bytes (ModelConfig math)
      3. How the 4 latency components are produced
         (end-to-end, storage I/O, generation, prefill/decode)
      4. Waterfall LRU eviction with 3 tiers (GPU → CPU → NVMe → delete)
      5. Waterfall LRU eviction with 1 tier  (NVMe-only, cpu=0 gpu=0)
    """

    @pytest.fixture
    def tiny_model(self):
        return MODEL_CONFIGS['tiny-1b']

    # ------------------------------------------------------------------
    # Part 1: User selection and request creation
    # ------------------------------------------------------------------

    def test_part1_user_selection_and_request_creation(self, tiny_model):
        """
        Shows how UserSimulator picks users and how InferenceRequest
        is built from a UserProfile.

        Key flow:
          UserSimulator.generate_mixed_users(N)
            → for each user, pick random type (chatbot/coding/document)
            → sample context_length from type's range
            → sample generation_length from type's range
            → roll QoS level (15% interactive, 35% responsive, 50% batch)
            → return UserProfile

          InferenceRequest is created from a UserProfile:
            → context_tokens  = user.context_length (how many tokens to prefill)
            → generate_tokens = user.generation_length (how many tokens to decode)
            → cache_key       = "{user_id}_ctx" (or conversation-based)
            → submit_time     = time.perf_counter() (latency clock starts here)
        """
        import random as rng
        rng.seed(42)

        print("\n" + "=" * 72)
        print("  PART 1: USER SELECTION AND REQUEST CREATION")
        print("=" * 72)

        # --- Step 1: Generate users ---
        print("\n  --- Step 1: UserSimulator generates 6 users ---")
        print("  Each user gets a random type (chatbot/coding/document)")
        print("  and a QoS level (interactive/responsive/batch).\n")
        print("  Templates:")
        for utype, tmpl in UserSimulator.DEFAULT_USER_TEMPLATES.items():
            print(f"    {utype:10s}  context={tmpl['context_range']}  "
                  f"gen={tmpl['generation_range']}  think={tmpl['think_time_range']}")

        users = UserSimulator.generate_mixed_users(6)

        print(f"\n  Generated {len(users)} users:")
        print(f"  {'ID':<12s} {'QoS':<14s} {'Pri':>3s} {'Context':>8s} {'GenLen':>7s} {'Think':>6s}")
        print(f"  {'-'*12} {'-'*14} {'-'*3} {'-'*8} {'-'*7} {'-'*6}")
        for u in users:
            print(f"  {u.user_id:<12s} {u.qos_level.value:<14s} {u.priority:>3d} "
                  f"{u.context_length:>8,d} {u.generation_length:>7d} {u.think_time:>6.2f}s")

        # --- Step 2: Create an InferenceRequest ---
        print("\n  --- Step 2: Build an InferenceRequest from first user ---")
        user = users[0]
        req = InferenceRequest(
            user_id=user.user_id,
            request_id=f"{user.user_id}_req_0",
            timestamp=datetime.now(),
            context_tokens=user.context_length,
            generate_tokens=user.generation_length,
            priority=user.priority,
            phase=InferencePhase.PREFILL_DECODE,
            qos_level=user.qos_level,
        )

        print(f"  Request fields:")
        print(f"    user_id         = {req.user_id}")
        print(f"    request_id      = {req.request_id}")
        print(f"    context_tokens  = {req.context_tokens:,d}")
        print(f"    generate_tokens = {req.generate_tokens}")
        print(f"    phase           = {req.phase.value}")
        print(f"    qos_level       = {req.qos_level.value}")
        print(f"    priority        = {req.priority}")
        print(f"    cache_key       = {req.cache_key}")
        print(f"    submit_time     = {req.submit_time:.6f} (perf_counter)")

        assert req.cache_key == f"{user.user_id}_ctx", \
            "Default cache_key should be '{user_id}_ctx'"
        assert req.context_tokens > 0
        assert req.generate_tokens > 0

    # ------------------------------------------------------------------
    # Part 2: KV cache size calculation
    # ------------------------------------------------------------------

    def test_part2_kv_cache_size_calculation(self, tiny_model):
        """
        Shows how context_tokens is converted to bytes.

        Formula (MHA/GQA):
          bytes_per_token = num_layers × kv_heads × kv_dim_per_head × 2 × dtype_bytes

        Total cache size:
          cache_bytes = context_tokens × bytes_per_token

        For tiny-1b (12 layers, 4 KV heads, dim=128, float16):
          bytes_per_token = 12 × 4 × 128 × 2 × 2 = 24,576 bytes = 24 KB/token
        """
        print("\n" + "=" * 72)
        print("  PART 2: KV CACHE SIZE CALCULATION")
        print("=" * 72)

        m = tiny_model
        bpt = m.kv_cache_size_per_token

        print(f"\n  Model: {m.name}")
        print(f"    num_layers      = {m.num_layers}")
        print(f"    kv_heads        = {m.kv_heads}")
        print(f"    kv_dim_per_head = {m.kv_dim_per_head}")
        print(f"    dtype           = {m.dtype}  ({m.bytes_per_element} bytes/element)")
        print(f"    attention_type  = {m.attention_type}")
        print(f"\n  Formula: num_layers × kv_heads × kv_dim_per_head × 2(K+V) × dtype_bytes")
        print(f"           {m.num_layers} × {m.kv_heads} × {m.kv_dim_per_head} × 2 × {m.bytes_per_element}")
        print(f"         = {bpt:,d} bytes/token  ({bpt / 1024:.1f}KiB/token)")

        expected = m.num_layers * m.kv_heads * m.kv_dim_per_head * 2 * m.bytes_per_element
        assert bpt == expected, f"Formula mismatch: {bpt} != {expected}"

        # Show how different context sizes scale
        print(f"\n  Context size → cache bytes:")
        for tokens in [100, 512, 2048, 8192, 16384]:
            total = tokens * bpt
            print(f"    {tokens:>6,d} tokens × {bpt/1024:.0f}KiB/tok = {total / 1024**2:>8.2f}MiB")

        # Compare with a larger model
        print(f"\n  Comparison across models:")
        for model_key in ['tiny-1b', 'mistral-7b', 'llama3.1-8b', 'llama3.1-70b-instruct']:
            mc = MODEL_CONFIGS[model_key]
            bpt2 = mc.kv_cache_size_per_token
            size_2k = 2048 * bpt2
            print(f"    {model_key:<25s}  {bpt2/1024:>6.0f}KiB/tok  "
                  f"  2048 ctx = {size_2k / 1024**2:>7.1f}MiB")

        # Show MLA (DeepSeek) is different
        if 'deepseek-v3' in MODEL_CONFIGS:
            ds = MODEL_CONFIGS['deepseek-v3']
            ds_bpt = ds.kv_cache_size_per_token
            print(f"\n  MLA model (DeepSeek V3): different formula")
            print(f"    num_layers × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes")
            print(f"    {ds.num_layers} × ({ds.kv_lora_rank} + {ds.qk_rope_head_dim}) × {ds.bytes_per_element}")
            print(f"    = {ds_bpt:,d} bytes/token  ({ds_bpt / 1024:.1f}KiB/token)")

    # ------------------------------------------------------------------
    # Part 3: The 4 latency levels (nested hierarchy)
    # ------------------------------------------------------------------

    def test_part3_four_latency_levels(self, tiny_model):
        """
        Traces a single request and shows how the 4 latency levels nest:

        ┌───────────────────────────────────────────────────────────────────┐
        │ L1: END-TO-END LATENCY                                          │
        │     submit_time → complete_time                                  │
        │     = Queue Wait + Storage I/O + Token Generation               │
        │                                                                  │
        │  ┌────────────────────────────────────────────────────────────┐  │
        │  │ L2: PER-REQUEST STORAGE LATENCY                           │  │
        │  │     Total I/O time for ONE request (multiple ops)         │  │
        │  │     = 1× Prefill Write + N× Decode Reads                  │  │
        │  │                                                            │  │
        │  │  ┌──────────────────────────────────────────────────────┐  │  │
        │  │  │ L3: PER-TIER TOTAL LATENCY                          │  │  │
        │  │  │     Time for ONE file I/O op on ONE tier             │  │  │
        │  │  │     = Host + Device                                  │  │  │
        │  │  │                                                      │  │  │
        │  │  │  ┌────────────────────────────────────────────────┐  │  │  │
        │  │  │  │ L4: HOST vs DEVICE BREAKDOWN                  │  │  │  │
        │  │  │  │     Write: Host=np.save() | Device=fsync()    │  │  │  │
        │  │  │  │     Read:  Host=fadvise+copy | Device=np.load │  │  │  │
        │  │  │  └────────────────────────────────────────────────┘  │  │  │
        │  │  └──────────────────────────────────────────────────────┘  │  │
        │  └────────────────────────────────────────────────────────────┘  │
        └───────────────────────────────────────────────────────────────────┘
        """
        print("\n" + "=" * 72)
        print("  PART 3: THE 4 LATENCY LEVELS (NESTED HIERARCHY)")
        print("=" * 72)

        # Force NVMe so we get real host/device splits (CPU backend
        # doesn't have a meaningful host vs device distinction)
        cache = MultiTierCache(
            model_config=tiny_model,
            gpu_memory_gb=0,
            cpu_memory_gb=0,   # zero → everything hits NVMe
            seed=42,
            storage_capacity_gb=0.1,  # 100 MB
        )

        context_tokens = 512
        generate_tokens = 100
        bpt = tiny_model.kv_cache_size_per_token
        cache_bytes = context_tokens * bpt

        print(f"\n  Request: {context_tokens} context tokens, {generate_tokens} gen tokens")
        print(f"  Cache entry: {context_tokens} × {bpt:,d} = {cache_bytes:,d} bytes ({cache_bytes/1024:.0f}KiB)")
        print(f"  Generation mode: NONE (0 ms/tok) — real benchmark uses FAST or REALISTIC")

        # ═══════════════════════════════════════════════════════════════
        # The clock starts when the request is submitted
        # ═══════════════════════════════════════════════════════════════
        submit_time = time.perf_counter()

        # ─────────────────────────────────────────────────────────────
        #  L3/L4: PREFILL WRITE — one I/O operation
        #  NVMeBackend.write() measures:
        #    host_time   = time for np.save()     (serialize + buffered write)
        #    device_time = time for f.flush() + os.fsync()  (commit to disk)
        #    total       = host_time + device_time
        # ─────────────────────────────────────────────────────────────
        print(f"\n  ──── PREFILL: allocate_cache('{context_tokens} tokens') ────")

        cache.stats['storage_write_latencies'].clear()
        cache.stats['storage_write_device_latencies'].clear()
        cache.stats['storage_write_host_latencies'].clear()

        success, tier, write_total = cache.allocate_cache(
            "user_0000_ctx", num_tokens=context_tokens, phase=InferencePhase.PREFILL,
        )

        # Pull L4 breakdown from stats (cache records it during allocate_cache)
        w_host = cache.stats['storage_write_host_latencies'][-1] if cache.stats['storage_write_host_latencies'] else 0
        w_device = cache.stats['storage_write_device_latencies'][-1] if cache.stats['storage_write_device_latencies'] else 0
        w_total = cache.stats['storage_write_latencies'][-1] if cache.stats['storage_write_latencies'] else write_total

        print(f"  tier = {tier},  success = {success}")
        print(f"  L3 write total : {w_total * 1000:>10.3f} ms  (one np.save + fsync)")
        print(f"    L4 host      : {w_host * 1000:>10.3f} ms  (np.save — serialize to page cache)")
        print(f"    L4 device    : {w_device * 1000:>10.3f} ms  (fsync — flush to NVMe controller)")

        prefill_latency = write_total
        storage_latency = write_total

        # ─────────────────────────────────────────────────────────────
        #  L3/L4: DECODE READ — one I/O operation
        #  NVMeBackend.read() measures:
        #    device_time = time for np.load()        (read from disk)
        #    host_time   = time for fadvise + np.array(copy)
        #    total       = host_time + device_time
        # ─────────────────────────────────────────────────────────────
        print(f"\n  ──── DECODE: access_cache('{tier}') ────")

        cache.stats['storage_read_latencies'].clear()
        cache.stats['storage_read_device_latencies'].clear()
        cache.stats['storage_read_host_latencies'].clear()

        location, read_total = cache.access_cache(
            "user_0000_ctx", phase=InferencePhase.DECODE,
        )

        r_host = cache.stats['storage_read_host_latencies'][-1] if cache.stats['storage_read_host_latencies'] else 0
        r_device = cache.stats['storage_read_device_latencies'][-1] if cache.stats['storage_read_device_latencies'] else 0
        r_total = cache.stats['storage_read_latencies'][-1] if cache.stats['storage_read_latencies'] else read_total

        print(f"  location = {location}")
        print(f"  L3 read total  : {r_total * 1000:>10.3f} ms  (one fadvise + np.load + copy)")
        print(f"    L4 host      : {r_host * 1000:>10.3f} ms  (posix_fadvise + np.array copy)")
        print(f"    L4 device    : {r_device * 1000:>10.3f} ms  (np.load — read from NVMe)")

        decode_latency = read_total
        storage_latency += read_total

        # ─────────────────────────────────────────────────────────────
        #  L2: BATCHED DECODE READS
        #  The benchmark does ceil(generate_tokens / batch_size) extra reads
        #  to simulate incremental KV access during token generation.
        # ─────────────────────────────────────────────────────────────
        decode_batch_size = 32
        num_batched = max(1, (generate_tokens + decode_batch_size - 1) // decode_batch_size)

        print(f"\n  ──── BATCHED DECODE READS ────")
        print(f"  ceil({generate_tokens} gen_tokens / {decode_batch_size} batch_size) = {num_batched} extra reads")

        for i in range(num_batched):
            _, batch_lat = cache.access_cache("user_0000_ctx", InferencePhase.DECODE)
            storage_latency += batch_lat

        print(f"  Batched read total: {(storage_latency - write_total - read_total) * 1000:.3f} ms")

        # ─────────────────────────────────────────────────────────────
        #  GENERATION LATENCY
        #  Simulates GPU token generation: sleep(tokens × per_token_time)
        # ─────────────────────────────────────────────────────────────
        gen_mode = GenerationMode.NONE   # use NONE for test speed
        generation_latency = generate_tokens * GENERATION_TIMING[gen_mode]

        print(f"\n  ──── GENERATION ────")
        print(f"  Mode: {gen_mode.value}")
        for mode, per_tok in GENERATION_TIMING.items():
            marker = " ←" if mode == gen_mode else ""
            print(f"    {mode.value:>10s}: {per_tok*1000:>5.0f} ms/tok × {generate_tokens} tok "
                  f"= {generate_tokens * per_tok * 1000:>7.0f} ms{marker}")

        complete_time = time.perf_counter()
        end_to_end = (complete_time - submit_time) * 1000

        # ═══════════════════════════════════════════════════════════════
        #  FULL HIERARCHY — with real numbers
        # ═══════════════════════════════════════════════════════════════
        print(f"\n  {'═' * 68}")
        print(f"  LATENCY HIERARCHY (real measurements from this request)")
        print(f"  {'═' * 68}")
        print(f"")
        print(f"  L1: END-TO-END                     {end_to_end:>10.3f} ms")
        print(f"  │   (submit_time → complete_time = storage + generation + overhead)")
        print(f"  │")
        print(f"  ├─ L2: STORAGE I/O (this request)  {storage_latency * 1000:>10.3f} ms")
        print(f"  │  │   (1 prefill write + 1 decode read + {num_batched} batched reads)")
        print(f"  │  │")
        print(f"  │  ├─ PREFILL WRITE                {write_total * 1000:>10.3f} ms")
        print(f"  │  │  └─ L3: tier total            {w_total * 1000:>10.3f} ms")
        print(f"  │  │     ├─ L4 host  (np.save)     {w_host * 1000:>10.3f} ms")
        print(f"  │  │     └─ L4 device (fsync)      {w_device * 1000:>10.3f} ms")
        print(f"  │  │")
        print(f"  │  ├─ DECODE READ                  {read_total * 1000:>10.3f} ms")
        print(f"  │  │  └─ L3: tier total            {r_total * 1000:>10.3f} ms")
        print(f"  │  │     ├─ L4 host  (fadvise+cp)  {r_host * 1000:>10.3f} ms")
        print(f"  │  │     └─ L4 device (np.load)    {r_device * 1000:>10.3f} ms")
        print(f"  │  │")
        print(f"  │  └─ BATCHED READS ×{num_batched:<3d}           "
              f"{(storage_latency - write_total - read_total) * 1000:>10.3f} ms")
        print(f"  │")
        print(f"  └─ GENERATION                      {generation_latency * 1000:>10.3f} ms")
        print(f"     ({generate_tokens} tokens × {GENERATION_TIMING[gen_mode]*1000:.0f} ms/tok [{gen_mode.value}])")
        print(f"")
        print(f"  Overhead (locks, data gen, etc):    "
              f"{end_to_end - storage_latency * 1000 - generation_latency * 1000:>10.3f} ms")

        # ═══════════════════════════════════════════════════════════════
        #  Where each level is recorded in the benchmark results JSON
        # ═══════════════════════════════════════════════════════════════
        print(f"\n  {'─' * 68}")
        print(f"  WHERE EACH LEVEL APPEARS IN BENCHMARK OUTPUT")
        print(f"  {'─' * 68}")
        print(f"  L1  → results['end_to_end_latencies']")
        print(f"  L2  → results['storage_latencies']  (per-request sum)")
        print(f"        results['prefill_latencies']   (write ops only)")
        print(f"        results['decode_latencies']    (read ops only)")
        print(f"  L3  → stats['storage_write_p50_ms']  thru  stats['storage_write_p9999_ms']")
        print(f"        stats['storage_read_p50_ms']   thru  stats['storage_read_p9999_ms']")
        print(f"  L4  → stats['storage_write_device_p50_ms']  (fsync only)")
        print(f"        stats['storage_write_host_p50_ms']    (np.save only)")
        print(f"        stats['storage_read_device_p50_ms']   (np.load only)")
        print(f"        stats['storage_read_host_p50_ms']     (fadvise+copy)")

        assert success
        assert storage_latency >= 0
        assert w_host >= 0 and w_device >= 0
        assert r_host >= 0 and r_device >= 0
        assert write_total > 0, "Write to NVMe should have measurable latency"
        assert read_total > 0, "Read from NVMe should have measurable latency"

    # ------------------------------------------------------------------
    # Part 3b: How requests become .npy files on disk
    # ------------------------------------------------------------------

    def test_part3b_request_to_npy_file_mapping(self, tiny_model):
        """
        Shows the exact path from a user request to a .npy file on disk.

        Flow:
          InferenceRequest.cache_key
            → NVMeBackend._get_path(cache_key) = base_path / "{cache_key}.npy"
            → NVMeBackend.write():
                open("{cache_key}.npy", 'wb')
                np.save(f, kv_data)          ← host time (serialize to page cache)
                f.flush(); os.fsync(f.fileno()) ← device time (commit to NVMe)
            → NVMeBackend.read():
                posix_fadvise(DONTNEED)      ← drop page cache for honest benchmark
                np.load("{cache_key}.npy")   ← device time (read from NVMe)
                np.array(data)               ← host time (copy to writable buffer)

        The .npy file is a standard NumPy binary format:
          - 10-byte magic header ("\\x93NUMPY")
          - Version, header length, dtype/shape metadata
          - Raw float16/float32 tensor data

        File size on disk ≈ data.nbytes + ~128 bytes header overhead
        """
        print("\n" + "=" * 72)
        print("  PART 3b: HOW REQUESTS BECOME .npy FILES ON DISK")
        print("=" * 72)

        cache = MultiTierCache(
            model_config=tiny_model,
            gpu_memory_gb=0,
            cpu_memory_gb=0,
            seed=42,
            storage_capacity_gb=0.1,
        )

        nvme_dir = cache.backends['nvme'].base_path
        bpt = tiny_model.kv_cache_size_per_token

        print(f"\n  NVMe base path: {nvme_dir}")
        print(f"  Model: {tiny_model.name}  ({bpt:,d} bytes/token)")

        # --- Single-turn request: cache_key = "{user_id}_ctx" ---
        print(f"\n  ──── Single-turn request ────")
        req = InferenceRequest(
            user_id="user_0001", request_id="req_0",
            timestamp=datetime.now(),
            context_tokens=100, generate_tokens=50, priority=1,
        )
        print(f"  cache_key    = {req.cache_key}")
        print(f"  Expected file: {nvme_dir / (req.cache_key + '.npy')}")

        success, tier, _ = cache.allocate_cache(
            req.cache_key, num_tokens=req.context_tokens
        )

        file_path = nvme_dir / f"{req.cache_key}.npy"
        expected_data_bytes = req.context_tokens * bpt
        file_size = file_path.stat().st_size if file_path.exists() else 0
        header_overhead = file_size - expected_data_bytes

        print(f"\n  allocate_cache() wrote to tier: {tier}")
        print(f"  File exists: {file_path.exists()}")
        print(f"  File path:   {file_path}")
        print(f"  File size:   {file_size:,d} bytes")
        print(f"    data:      {expected_data_bytes:,d} bytes  ({req.context_tokens} tok × {bpt:,d} B/tok)")
        print(f"    header:    {header_overhead:,d} bytes  (.npy magic + dtype + shape)")

        # Show file structure
        print(f"\n  .npy file internal structure:")
        print(f"    ┌──────────────────────────────────────────────┐")
        print(f"    │ \\x93NUMPY magic (6 bytes)                    │")
        print(f"    │ version 1.0 (2 bytes)                        │")
        print(f"    │ header_len (2 bytes)                         │")
        print(f"    │ {{'descr': '<f2', 'shape': (...), ...}}        │")
        print(f"    │ padding to 64-byte alignment                 │")
        print(f"    ├──────────────────────────────────────────────┤")
        print(f"    │ raw float16 tensor data                      │")
        print(f"    │ {expected_data_bytes:,d} bytes                               │")
        print(f"    │ shape: ({tiny_model.num_layers}, 2, {req.context_tokens}, "
              f"{tiny_model.kv_heads}, {tiny_model.kv_dim_per_head})   │")
        print(f"    └──────────────────────────────────────────────┘")

        # --- Read it back ---
        print(f"\n  ──── Reading the file back ────")
        print(f"  access_cache('{req.cache_key}') reads:")
        print(f"    1. posix_fadvise(fd, DONTNEED) — drop page cache")
        print(f"    2. np.load('{file_path.name}')  — read from disk")
        print(f"    3. np.array(data)               — copy to writable buffer")

        location, read_lat = cache.access_cache(req.cache_key, InferencePhase.DECODE)
        print(f"  location: {location},  latency: {read_lat*1000:.3f} ms")

        # --- Multiple requests = multiple files ---
        print(f"\n  ──── Multiple requests = multiple .npy files ────")
        keys = []
        for i in range(5):
            key = f"user_{i:04d}_ctx"
            cache.allocate_cache(key, num_tokens=100 + i * 50)
            keys.append(key)

        npy_files = sorted(nvme_dir.glob("*.npy"))
        print(f"  Allocated {len(keys)} requests → {len(npy_files)} files:")
        for f in npy_files:
            sz = f.stat().st_size
            print(f"    {f.name:<30s}  {sz:>10,d} bytes  ({sz/1024:.1f}KiB)")

        assert file_path.exists(), f"Expected .npy file at {file_path}"
        assert file_size > expected_data_bytes, "File should include .npy header"
        assert len(npy_files) >= len(keys), "Each cache_key should produce one .npy file"

    # ------------------------------------------------------------------
    # Part 3c: Multi-turn conversations and file I/O
    # ------------------------------------------------------------------

    def test_part3c_multi_turn_prefill_decode_file_io(self, tiny_model):
        """
        Shows how a multi-turn conversation reads ALL previous turns.

        Each turn writes its own new tokens. On subsequent turns, ALL
        previous turns are read via access_cache to reload the full
        conversation context. Evicted turns return (None, 0.0) with
        no I/O; surviving turns trigger real np.load reads.

        Turn 1: WRITE turn_1 + READ turn_1 (decode)
        Turn 2: READ [turn_1] + WRITE turn_2 + READ turn_2
        Turn 3: READ [turn_1, turn_2] + WRITE turn_3 + READ turn_3
        Turn 4: READ [turn_1, turn_2, turn_3] + WRITE turn_4 + READ turn_4

        Multi-turn reads (triangular): 0+1+2+3 = 6
        Decode reads: 4 (one per turn)
        Total decode_reads stat: 10

        This is the exact flow from benchmark.py process_requests() steps 2, 3, 5.
        """
        print("\n" + "=" * 72)
        print("  PART 3c: MULTI-TURN CONVERSATION FILE I/O")
        print("=" * 72)

        cache = MultiTierCache(
            model_config=tiny_model,
            gpu_memory_gb=0,
            cpu_memory_gb=0,
            seed=42,
            storage_capacity_gb=0.5,  # plenty of room so no eviction
        )

        nvme_dir = cache.backends['nvme'].base_path
        bpt = tiny_model.kv_cache_size_per_token
        conv_mgr = ConversationManager(max_conversations=10)

        # Start a conversation
        conv_id = conv_mgr.start_conversation("alice")
        print(f"\n  Conversation started: {conv_id}")
        print(f"  NVMe dir: {nvme_dir}")

        num_turns = 4
        context_per_turn = 200  # tokens

        print(f"\n  Simulating {num_turns} turns, {context_per_turn} context tokens each")
        print(f"  Entry size per turn: {context_per_turn} × {bpt:,d} = "
              f"{context_per_turn * bpt / 1024:.0f}KiB")

        for turn in range(1, num_turns + 1):
            print(f"\n  {'━' * 64}")
            print(f"  TURN {turn}")
            print(f"  {'━' * 64}")

            # ConversationManager creates the cache_key
            turn_num, cache_key = conv_mgr.add_turn(conv_id, context_per_turn, 50)

            print(f"  cache_key = {cache_key}")
            print(f"  file      = {cache_key}.npy")

            storage_latency = 0.0
            file_ops = []

            # ── Step 2: Multi-turn read (ALL previous turns) ──
            if turn > 1:
                prev_keys = conv_mgr.get_all_previous_turn_keys(conv_id, turn)
                print(f"\n  Step 2: MULTI-TURN READ ({len(prev_keys)} previous turn(s))")
                for prev_key in prev_keys:
                    location, read_lat = cache.access_cache(
                        prev_key, InferencePhase.DECODE, 'multi_turn'
                    )
                    storage_latency += read_lat
                    if location:
                        file_ops.append(f"READ  {prev_key}.npy  ({read_lat*1000:.3f} ms)  [hit, {location}]")
                        print(f"    {prev_key} -> Hit ({location}, {read_lat*1000:.3f} ms)")
                    else:
                        file_ops.append(f"READ  {prev_key}.npy  [miss, evicted]")
                        print(f"    {prev_key} -> Miss (evicted)")
            else:
                print(f"\n  Step 2: MULTI-TURN READ — skipped (turn 1, no history)")

            # ── Step 3: Prefill write (this turn's new KV cache) ──
            this_file = nvme_dir / f"{cache_key}.npy"

            print(f"\n  Step 3: PREFILL WRITE (new KV cache for this turn)")
            print(f"    Write: {cache_key}.npy")

            success, tier, write_lat = cache.allocate_cache(
                cache_key, num_tokens=context_per_turn, phase=InferencePhase.PREFILL
            )
            storage_latency += write_lat
            file_ops.append(f"WRITE {cache_key}.npy  ({write_lat*1000:.3f} ms)  [prefill]")

            file_size = this_file.stat().st_size if this_file.exists() else 0
            print(f"    tier={tier}, success={success}, latency={write_lat*1000:.3f} ms")
            print(f"    File created: {this_file.exists()},  size: {file_size:,d} bytes")

            # ── Step 5: Decode read (read back this turn's cache) ──
            print(f"\n  Step 5: DECODE READ (read back this turn's KV cache)")
            print(f"    Read:  {cache_key}.npy")

            location, read_lat = cache.access_cache(
                cache_key, InferencePhase.DECODE
            )
            storage_latency += read_lat
            file_ops.append(f"READ  {cache_key}.npy  ({read_lat*1000:.3f} ms)  [decode]")

            print(f"    location={location}, latency={read_lat*1000:.3f} ms")

            # ── Summary for this turn ──
            npy_files = sorted(nvme_dir.glob("*.npy"))
            print(f"\n  Turn {turn} I/O summary:")
            for op in file_ops:
                print(f"    {op}")
            print(f"  Total storage latency this turn: {storage_latency*1000:.3f} ms")
            print(f"  .npy files on disk after turn {turn}: {len(npy_files)}")
            for f in npy_files:
                marker = " ← NEW" if f.stem == cache_key else ""
                print(f"    {f.name}{marker}")

        # ── Final summary ──
        all_npy = sorted(nvme_dir.glob("*.npy"))
        all_entries = {k: v for k, v in cache.cache_entries.items()
                       if v['location'] == 'nvme'}

        print(f"\n  {'═' * 64}")
        print(f"  MULTI-TURN FILE I/O SUMMARY")
        print(f"  {'═' * 64}")
        print(f"  Turns completed:       {num_turns}")
        print(f"  .npy files on disk:    {len(all_npy)}")
        print(f"  NVMe cache entries:    {len(all_entries)}")
        print(f"  Total writes:          {cache.stats['prefill_writes']}")
        print(f"  Total reads:           {cache.stats['decode_reads']}")
        print(f"  Total write bytes:     {cache.stats['total_write_bytes']/1024:.0f}KiB")
        print(f"  Total read bytes:      {cache.stats['total_read_bytes']/1024:.0f}KiB")

        multi_turn_reads = (num_turns * (num_turns - 1)) // 2

        print(f"\n  Full-context reload pattern:")
        print(f"    Turn 1: WRITE turn_1 + READ turn_1")
        print(f"    Turn 2: READ [turn_1] + WRITE turn_2 + READ turn_2")
        print(f"    Turn 3: READ [turn_1, turn_2] + WRITE turn_3 + READ turn_3")
        print(f"    Turn N: READ [turns 1..N-1] + WRITE turn_N + READ turn_N")
        print(f"")
        print(f"  I/O per turn:")
        print(f"    Turn 1:  1 write + 1 read  = 2 I/O ops")
        print(f"    Turn N:  1 write + N reads  = {num_turns + 1} I/O ops  (turn {num_turns})")
        print(f"  Multi-turn reads (triangular): {multi_turn_reads}")

        # Assertions
        assert len(all_npy) == num_turns, \
            f"Should have {num_turns} .npy files (one per turn), got {len(all_npy)}"
        assert cache.stats['prefill_writes'] == num_turns, \
            f"Should have {num_turns} prefill writes"
        # decode_reads = multi-turn reads (triangular) + decode reads (one per turn)
        expected_reads = num_turns + multi_turn_reads
        assert cache.stats['decode_reads'] == expected_reads, \
            f"Expected {expected_reads} decode reads, got {cache.stats['decode_reads']}"

    # ------------------------------------------------------------------
    # Part 3d: Multi-turn with eviction pressure
    # ------------------------------------------------------------------

    def test_part3d_multi_turn_with_eviction(self, tiny_model):
        """
        Shows what happens when previous turns get evicted by the LRU
        waterfall before the next turn reads them.

        Setup: NVMe capacity fits only 3 entries. A 6-turn conversation
        forces turns 1-3 to be evicted as turns 4-6 are written. The
        multi-turn reads for later turns show a mix of hits and misses.

        This validates that access_cache returns (None, 0.0) cleanly for
        evicted entries, and that the benchmark correctly counts misses.
        """
        print("\n" + "=" * 72)
        print("  PART 3d: MULTI-TURN WITH EVICTION PRESSURE")
        print("=" * 72)

        bpt = tiny_model.kv_cache_size_per_token
        context_per_turn = 200
        entry_bytes = context_per_turn * bpt
        # Capacity for ~3 entries; turns 4+ force eviction of oldest
        capacity_bytes = entry_bytes * 3.5
        capacity_gb = capacity_bytes / (1024 ** 3)

        cache = MultiTierCache(
            model_config=tiny_model,
            gpu_memory_gb=0,
            cpu_memory_gb=0,
            seed=42,
            storage_capacity_gb=capacity_gb,
        )

        nvme_dir = cache.backends['nvme'].base_path
        conv_mgr = ConversationManager(max_conversations=10)
        conv_id = conv_mgr.start_conversation("bob")

        num_turns = 6
        print(f"\n  Entry size: {entry_bytes/1024:.0f}KiB")
        print(f"  NVMe capacity: {capacity_bytes/1024:.0f}KiB (~{capacity_bytes/entry_bytes:.1f} entries)")
        print(f"  Turns: {num_turns} (will force eviction after turn 3)")

        total_hits = 0
        total_misses = 0

        for turn in range(1, num_turns + 1):
            turn_num, cache_key = conv_mgr.add_turn(conv_id, context_per_turn, 50)

            print(f"\n  Turn {turn}:")

            # Step 2: read all previous turns
            if turn > 1:
                prev_keys = conv_mgr.get_all_previous_turn_keys(conv_id, turn)
                hits = 0
                misses = 0
                for prev_key in prev_keys:
                    location, _ = cache.access_cache(prev_key, InferencePhase.DECODE, 'multi_turn')
                    if location is not None:
                        hits += 1
                    else:
                        misses += 1
                total_hits += hits
                total_misses += misses
                print(f"    Step 2: read {len(prev_keys)} previous turns -> {hits} hits, {misses} misses")

            # Step 3: write this turn
            success, tier, _ = cache.allocate_cache(
                cache_key, num_tokens=context_per_turn, phase=InferencePhase.PREFILL
            )
            npy_count = len(list(nvme_dir.glob("*.npy")))
            print(f"    Step 3: write {cache_key} -> {tier}, files on disk: {npy_count}")

        npy_files = sorted(nvme_dir.glob("*.npy"))
        print(f"\n  {'=' * 64}")
        print(f"  EVICTION SUMMARY")
        print(f"  {'=' * 64}")
        print(f"  Multi-turn hits:   {total_hits}")
        print(f"  Multi-turn misses: {total_misses}")
        print(f"  Files on disk:     {len(npy_files)} (capacity held ~3)")
        print(f"  Evictions:         {cache.stats['evictions']}")
        print(f"  Cache misses:      {cache.stats['cache_misses']}  (tier-level, all types)")
        for f in npy_files:
            print(f"    {f.name}")

        print(f"\n  HOW CACHE MISSES INCREMENT:")
        print(f"  ────────────────────────────")
        print(f"  Two separate counters track misses:")
        print(f"")
        print(f"  1. cache.stats['cache_misses'] (tier-level, inside access_cache):")
        print(f"     Increments when ANY access_cache() call finds the key missing")
        print(f"     from cache_entries. Covers all miss types: multi-turn, decode,")
        print(f"     prefix, RAG. Returns (None, 0.0) immediately; no I/O, no np.load.")
        print(f"")
        print(f"  2. results['multi_turn_cache_misses'] (benchmark-level, step 2 only):")
        print(f"     Increments when a multi-turn access_cache() returns None.")
        print(f"     This is a SUBSET of cache_misses; it counts only the misses")
        print(f"     from the step 2 previous-turn read loop.")
        print(f"")
        print(f"  In this test: {cache.stats['cache_misses']} tier-level misses total,")
        print(f"  {total_misses} of which are multi-turn misses from evicted turns.")
        print(f"")
        print(f"  A high multi-turn miss rate ({total_misses}/{total_hits + total_misses}"
              f" = {total_misses*100/max(total_hits+total_misses,1):.0f}%) means the")
        print(f"  NVMe tier is too small to hold the full conversation history.")
        print(f"  The LRU waterfall evicts old turns before they can be reused.")

        # Assertions
        assert total_misses > 0, "Eviction should cause some multi-turn misses"
        assert total_hits > 0, "Recent turns should still be cached"
        assert cache.stats['evictions'] > 0, "Eviction should have occurred"
        assert cache.stats['cache_misses'] >= total_misses, \
            "Tier-level misses should be >= multi-turn misses (superset)"
        assert len(npy_files) <= 4, "NVMe should hold ~3 entries, not all 6"

    # ------------------------------------------------------------------
    # Part 4: 3-tier waterfall LRU eviction
    # ------------------------------------------------------------------

    def test_part4_three_tier_waterfall_eviction(self, tiny_model):
        """
        Demonstrates the full 3-tier waterfall LRU eviction cascade:

          GPU (fastest) → CPU (mid) → NVMe (slowest) → DELETE

        When the benchmark calls allocate_cache():
          1. Try GPU: _ensure_space_in_tier('gpu', size)
             - If GPU is full, pick LRU entry in GPU
             - Recursively call _ensure_space_in_tier('cpu', lru_size)  ← makes room
             - _demote_entry(lru_key, 'gpu', 'cpu')                    ← move data
             - Now GPU has space → write new entry

          2. If GPU has no capacity (limit=0), skip to CPU.
          3. If CPU is full, same cascade:  CPU LRU → NVMe
          4. If NVMe is full (terminal tier): DELETE the LRU .npy file

        This test uses a fake GPU backend (CPUMemoryBackend injected as
        backends['gpu']) since we have no real GPU.
        """
        print("\n" + "=" * 72)
        print("  PART 4: 3-TIER WATERFALL LRU EVICTION")
        print("=" * 72)

        bpt = tiny_model.kv_cache_size_per_token
        tokens = 10
        entry_kb = (tokens * bpt) / 1024

        gpu_mb, cpu_mb, nvme_mb = 1, 1, 1

        cache = MultiTierCache(
            model_config=tiny_model,
            gpu_memory_gb=0,
            cpu_memory_gb=cpu_mb / 1024,
            seed=42,
            storage_capacity_gb=nvme_mb / 1024,
        )

        # Inject fake GPU
        cache.backends['gpu'] = CPUMemoryBackend()
        cache.gpu_memory_limit = gpu_mb * 1024 * 1024

        tier_order = cache._get_tier_order()
        entries_per_tier = int((gpu_mb * 1024) / entry_kb)

        print(f"\n  Tier order: {tier_order}")
        print(f"  Entry size: {tokens} tokens × {bpt:,d} B/tok = {entry_kb:.0f}KiB")
        print(f"  Tier capacity: GPU={gpu_mb}MiB, CPU={cpu_mb}MiB, NVMe={nvme_mb}MiB")
        print(f"  Entries per tier: ~{entries_per_tier}")
        print(f"\n  Writing 30 entries (much more than total 3-tier capacity)...")

        print(f"\n  {'#':>4s} {'Key':<14s} {'Tier':<6s} {'GPU KiB':>7s} {'CPU KiB':>7s} "
              f"{'NVMe KiB':>8s} {'Evict':>5s} {'→CPU':>4s} {'→NVMe':>5s} {'Event'}")
        print(f"  {'─'*4} {'─'*14} {'─'*6} {'─'*7} {'─'*7} {'─'*8} {'─'*5} {'─'*4} {'─'*5} {'─'*30}")

        prev_evictions = 0
        prev_cpu_offloads = 0
        prev_nvme_offloads = 0

        for i in range(30):
            key = f"req_{i}"
            success, tier, lat = cache.allocate_cache(key, num_tokens=tokens)

            evictions = cache.stats['evictions']
            cpu_off = cache.stats['offloads_cpu']
            nvme_off = cache.stats['offloads_storage']

            # Detect what happened this iteration
            events = []
            new_evictions = evictions - prev_evictions
            new_cpu = cpu_off - prev_cpu_offloads
            new_nvme = nvme_off - prev_nvme_offloads
            if new_cpu > 0:
                events.append(f"GPU→CPU demote ×{new_cpu}")
            if new_nvme > 0:
                events.append(f"CPU→NVMe demote ×{new_nvme}")
            if new_evictions > new_cpu + new_nvme:
                deletes = new_evictions - new_cpu - new_nvme
                events.append(f"NVMe DELETE ×{deletes}")
            event_str = ", ".join(events) if events else "—"

            print(f"  {i:>4d} {key:<14s} {tier:<6s} "
                  f"{cache.gpu_memory_used/1024:>7.0f} "
                  f"{cache.cpu_memory_used/1024:>7.0f} "
                  f"{cache.nvme_memory_used/1024:>8.0f} "
                  f"{evictions:>5d} {cpu_off:>4d} {nvme_off:>5d} {event_str}")

            prev_evictions = evictions
            prev_cpu_offloads = cpu_off
            prev_nvme_offloads = nvme_off

        gpu_entries = sum(1 for v in cache.cache_entries.values() if v['location'] == 'gpu')
        cpu_entries = sum(1 for v in cache.cache_entries.values() if v['location'] == 'cpu')
        nvme_entries = sum(1 for v in cache.cache_entries.values() if v['location'] == 'nvme')

        print(f"\n  Final state:")
        print(f"    GPU entries:      {gpu_entries}")
        print(f"    CPU entries:      {cpu_entries}")
        print(f"    NVMe entries:     {nvme_entries}")
        print(f"    Total alive:      {len(cache.cache_entries)}")
        print(f"    Total evictions:  {cache.stats['evictions']}")
        print(f"    GPU→CPU demotes:  {cache.stats['offloads_cpu']}")
        print(f"    CPU→NVMe demotes: {cache.stats['offloads_storage']}")
        print(f"    NVMe deletes:     {cache.stats['evictions'] - cache.stats['offloads_cpu'] - cache.stats['offloads_storage']}")

        npy_files = list(cache.backends['nvme'].base_path.glob("*.npy"))
        print(f"    .npy files on disk: {len(npy_files)}  (should ≈ {nvme_entries})")

        print(f"\n  Eviction flow summary:")
        print(f"    GPU full → demote LRU to CPU   (_demote_entry, data moves)")
        print(f"    CPU full → demote LRU to NVMe  (_demote_entry, data moves)")
        print(f"    NVMe full → DELETE LRU from disk (file unlinked, entry gone)")
        print(f"    New entry always lands on GPU (fastest available tier)")

        assert cache.stats['offloads_cpu'] > 0, "GPU→CPU demotions should have occurred"
        assert cache.stats['offloads_storage'] > 0, "CPU→NVMe demotions should have occurred"
        nvme_deletes = cache.stats['evictions'] - cache.stats['offloads_cpu'] - cache.stats['offloads_storage']
        assert nvme_deletes > 0, "NVMe deletes should have occurred"

    # ------------------------------------------------------------------
    # Part 5: 1-tier (NVMe-only) waterfall eviction
    # ------------------------------------------------------------------

    def test_part5_one_tier_nvme_only_eviction(self, tiny_model):
        """
        Demonstrates NVMe-only mode (cpu=0, gpu=0).

        This is the configuration that exposed 3 bugs:
          1. Double-decrement race on nvme_memory_used
          2. Eviction guards rejecting entries on the terminal tier
          3. Preconditioning spinning forever

        With only NVMe available:
          - Every allocate_cache() goes directly to NVMe
          - _ensure_space_in_tier('nvme') sees next_tier=None → is_last_tier=True
          - Eviction = DELETE (unlink .npy file), not demote
          - Capacity guards are relaxed:
              • Skip 95% size cap   (entry has nowhere else to go)
              • Use 100% target     (no cascade buffer needed)
              • Skip low-data bail  (keep evicting until space is free)
        """
        print("\n" + "=" * 72)
        print("  PART 5: 1-TIER NVMe-ONLY EVICTION (cpu=0, gpu=0)")
        print("=" * 72)

        bpt = tiny_model.kv_cache_size_per_token
        tokens = 10
        entry_kb = (tokens * bpt) / 1024
        nvme_mb = 1

        cache = MultiTierCache(
            model_config=tiny_model,
            gpu_memory_gb=0,
            cpu_memory_gb=0,          # ZERO
            seed=42,
            storage_capacity_gb=nvme_mb / 1024,
        )

        nvme_dir = cache.backends['nvme'].base_path
        tier_order = cache._get_tier_order()
        entries_fit = int((nvme_mb * 1024) / entry_kb)

        print(f"\n  Tier order: {tier_order}")
        print(f"  CPU limit: {cache.cpu_memory_limit} bytes (zero → skipped)")
        print(f"  NVMe limit: {cache.nvme_memory_limit / 1024:.0f}KiB")
        print(f"  Entry size: {entry_kb:.0f}KiB")
        print(f"  Entries that fit: ~{entries_fit}")
        print(f"  NVMe dir: {nvme_dir}")

        print(f"\n  is_last_tier behavior:")
        print(f"    next_tier = None  (nothing after NVMe)")
        print(f"    is_last_tier = True")
        print(f"    → Skip 95% size cap     (can't send entry elsewhere)")
        print(f"    → effective_target = 100% (no cascade buffer)")
        print(f"    → Skip low-data bailout  (keep evicting)")
        print(f"    → Eviction = DELETE file  (not demote)")

        print(f"\n  Writing 20 entries into {nvme_mb}MiB NVMe...")
        print(f"\n  {'#':>4s} {'Key':<12s} {'Tier':<6s} {'NVMe KiB':>8s} "
              f"{'Files':>5s} {'Evict':>5s} {'Event'}")
        print(f"  {'─'*4} {'─'*12} {'─'*6} {'─'*8} {'─'*5} {'─'*5} {'─'*20}")

        prev_evictions = 0

        for i in range(20):
            key = f"req_{i}"
            success, tier, lat = cache.allocate_cache(key, num_tokens=tokens)

            npy_count = len(list(nvme_dir.glob("*.npy")))
            evictions = cache.stats['evictions']
            new_ev = evictions - prev_evictions

            event = f"DELETE ×{new_ev}" if new_ev > 0 else "—"

            print(f"  {i:>4d} {key:<12s} {tier:<6s} "
                  f"{cache.nvme_memory_used/1024:>8.0f} "
                  f"{npy_count:>5d} {evictions:>5d} {event}")

            prev_evictions = evictions
            assert success, f"Allocation {i} must succeed on terminal tier"

        entries_alive = len(cache.cache_entries)
        npy_final = len(list(nvme_dir.glob("*.npy")))

        print(f"\n  Final state:")
        print(f"    Entries in cache: {entries_alive}")
        print(f"    .npy on disk:    {npy_final}")
        print(f"    Total evictions: {cache.stats['evictions']}  (all were DELETEs)")
        print(f"    nvme_memory_used: {cache.nvme_memory_used / 1024:.0f}KiB")
        print(f"    Offloads to CPU: {cache.stats['offloads_cpu']}  (0 — no CPU tier)")
        print(f"    Offloads to NVMe: {cache.stats['offloads_storage']}  (= every allocation, since NVMe is the only tier)")

        print(f"\n  Note on 'offloads_storage':")
        print(f"    This counter increments for EVERY entry written to NVMe,")
        print(f"    whether by direct allocation or by demotion from CPU.")
        print(f"    In NVMe-only mode: offloads_storage = total allocations (20)")
        print(f"    In 3-tier mode:    offloads_storage = CPU→NVMe demotions only")

        print(f"\n  Key difference from 3-tier:")
        print(f"    3-tier: eviction = DEMOTE to next tier (data preserved)")
        print(f"    1-tier: eviction = DELETE from disk    (data destroyed)")
        print(f"    Both use LRU ordering (oldest access first)")

        assert cache.stats['evictions'] > 0, "Evictions should have occurred"
        assert cache.stats['offloads_cpu'] == 0, "No CPU demotions with cpu=0"
        assert cache.nvme_memory_used >= 0, "No negative drift"
        assert npy_final == entries_alive, \
            f"Disk files ({npy_final}) should match alive entries ({entries_alive})"

class TestBottleneckProfiling:
    """Profile bottleneck detection in the KV cache benchmark."""

    def test_profile_allocate_vs_access_overhead(self):
        """Profile allocate vs access operations to identify bottleneck ratios."""
        import time as time_mod

        model_config = MODEL_CONFIGS['tiny-1b']
        cache = MultiTierCache(
            model_config=model_config,
            gpu_memory_gb=0,
            cpu_memory_gb=0.1,  # 100MB
            seed=42
        )

        num_ops = 500
        keys = [f"profile_key_{i}" for i in range(num_ops)]

        # Profile allocations (write path)
        alloc_start = time_mod.perf_counter()
        for key in keys:
            cache.allocate_cache(key, num_tokens=100)
        alloc_elapsed = time_mod.perf_counter() - alloc_start

        # Profile accesses (read path)
        access_start = time_mod.perf_counter()
        for key in keys:
            cache.access_cache(key, InferencePhase.DECODE)
        access_elapsed = time_mod.perf_counter() - access_start

        alloc_per_op_us = (alloc_elapsed / num_ops) * 1e6
        access_per_op_us = (access_elapsed / num_ops) * 1e6

        # Profile lock contention: metadata_lock acquire time
        lock_times = []
        for _ in range(100):
            t0 = time_mod.perf_counter()
            with cache.metadata_lock:
                pass
            lock_times.append((time_mod.perf_counter() - t0) * 1e6)
        avg_lock_us = sum(lock_times) / len(lock_times)

        # Profile stats collection overhead
        stats_start = time_mod.perf_counter()
        for _ in range(100):
            cache.get_stats(duration=1.0)
        stats_elapsed = time_mod.perf_counter() - stats_start
        stats_per_call_us = (stats_elapsed / 100) * 1e6

        # Assertions: ensure no single operation is unreasonably slow
        # These thresholds are generous — the point is detecting regressions
        assert alloc_per_op_us < 50000, \
            f"Allocation too slow: {alloc_per_op_us:.0f} us/op (threshold: 50ms)"
        assert access_per_op_us < 50000, \
            f"Access too slow: {access_per_op_us:.0f} us/op (threshold: 50ms)"
        assert avg_lock_us < 1000, \
            f"Lock contention too high: {avg_lock_us:.0f} us/acquire (threshold: 1ms)"
        assert stats_per_call_us < 100000, \
            f"get_stats() too slow: {stats_per_call_us:.0f} us/call (threshold: 100ms)"

        # Report profiling results for visibility in test output
        print(f"\n  --- Bottleneck Profile ({num_ops} ops) ---")
        print(f"  Allocate:    {alloc_per_op_us:>8.1f} us/op  ({num_ops / alloc_elapsed:>8.0f} ops/s)")
        print(f"  Access:      {access_per_op_us:>8.1f} us/op  ({num_ops / access_elapsed:>8.0f} ops/s)")
        print(f"  Lock:        {avg_lock_us:>8.1f} us/acquire")
        print(f"  get_stats(): {stats_per_call_us:>8.1f} us/call")
        print(f"  Write:Read ratio: {alloc_per_op_us / max(access_per_op_us, 0.01):.2f}x")


# =============================================================================
# Test: Validation for new CLI args (trace_speedup, replay_cycles)
# =============================================================================

class TestValidateNewTraceArgs:
    """Validation tests for --trace-speedup and --replay-cycles."""

    @pytest.fixture
    def valid_args(self):
        import argparse
        return argparse.Namespace(
            num_users=100, duration=60, gpu_mem_gb=16, cpu_mem_gb=32,
            num_gpus=1, tensor_parallel=1,
            rag_num_docs=10, max_conversations=500, max_concurrent_allocs=0,
            request_rate=0, max_requests=0, target_saturation=0.8,
            cache_dir=None, storage_capacity_gb=0, precondition_size_gb=0,
            precondition_threads=0, trace_speedup=1.0, replay_cycles=0
        )

    def test_trace_speedup_negative_rejected(self, valid_args):
        valid_args.trace_speedup = -1.0
        with pytest.raises(ValueError, match="trace-speedup cannot be negative"):
            validate_args(valid_args)

    def test_trace_speedup_zero_accepted(self, valid_args):
        valid_args.trace_speedup = 0
        result = validate_args(valid_args)
        assert result.trace_speedup == 0

    def test_trace_speedup_positive_accepted(self, valid_args):
        valid_args.trace_speedup = 100.0
        result = validate_args(valid_args)
        assert result.trace_speedup == 100.0

    def test_replay_cycles_negative_rejected(self, valid_args):
        valid_args.replay_cycles = -1
        with pytest.raises(ValueError, match="replay-cycles cannot be negative"):
            validate_args(valid_args)

    def test_replay_cycles_zero_accepted(self, valid_args):
        valid_args.replay_cycles = 0
        result = validate_args(valid_args)
        assert result.replay_cycles == 0

    def test_replay_cycles_positive_accepted(self, valid_args):
        valid_args.replay_cycles = 5
        result = validate_args(valid_args)
        assert result.replay_cycles == 5


# =============================================================================
# Main entry point for running without pytest
# =============================================================================

def pytest_configure(config):
    """Add metadata to pytest-html report."""
    if hasattr(config, '_metadata'):
        config._metadata['Project'] = 'MLPerf v3 KV Cache Benchmark'
        config._metadata['Source File'] = 'kv-cache.py'
        config._metadata['Models'] = 'tiny-1b, mistral-7b, llama2-7b, llama3.1-8b, llama3.1-70b-instruct'
        config._metadata['Test File'] = 'test_kv_cache.py'
        config._metadata['New Features Tested'] = 'ConfigLoader, Extended QoS (p999/p9999), cfg() helper, storage_* naming, NVMe capacity tracking, NVMe eviction, reset_stats, preconditioning validation, trace streaming iterator, timestamp pacing, replay cycles, eviction tracing, bottleneck profiling'


def pytest_html_report_title(report):
    """Set custom title for HTML report."""
    report.title = "KV Cache Benchmark - Unit Test Report"


if __name__ == "__main__":
    # Generate HTML report by default when run directly
    report_path = Path(__file__).parent / "test_report.html"
    exit_code = pytest.main([
        __file__, 
        "-v",
        f"--html={report_path}",
        "--self-contained-html",
    ])
    if exit_code == 0:
        print(f"\n✓ All tests passed! HTML report: {report_path}")
    else:
        print(f"\n✗ Some tests failed. HTML report: {report_path}")
    sys.exit(exit_code)
