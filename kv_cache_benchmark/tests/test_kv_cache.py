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
"""

import os
import sys
import tempfile
import pytest
import numpy as np
from datetime import datetime
from pathlib import Path

# Import from kv-cache.py (handle the hyphen in filename)
import importlib.util
spec = importlib.util.spec_from_file_location("kv_cache", os.path.join(os.path.dirname(__file__), "kv-cache.py"))
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
        dataset_path = None
        cache_dir = None
    return MockArgs()


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
    
    def test_all_five_model_configs_exist(self):
        assert len(MODEL_CONFIGS) == 5
    
    @pytest.mark.parametrize("model_name", [
        'tiny-1b', 'mistral-7b', 'llama2-7b', 'llama3.1-8b', 'llama3.1-70b-instruct'
    ])
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
        assert np.array_equal(data1, data2)
    
    def test_different_key_runs(self, kv_generator):
        """Different key should not crash."""
        kv_generator.generate(sequence_length=10, key="different_key")


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
            assert 256 <= user.context_length <= 8192
    
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
    
    def test_nvme_limit_infinite(self, multi_tier_cache):
        nvme_limit = multi_tier_cache._get_tier_limit('nvme')
        assert nvme_limit == float('inf')
    
    def test_initial_cpu_usage_zero(self, multi_tier_cache):
        cpu_usage = multi_tier_cache._get_tier_usage('cpu')
        assert cpu_usage == 0


# =============================================================================
# Main entry point for running without pytest
# =============================================================================

def pytest_configure(config):
    """Add metadata to pytest-html report."""
    if hasattr(config, '_metadata'):
        config._metadata['Project'] = 'MLPerf v3 KV Cache Benchmark'
        config._metadata['Models'] = 'tiny-1b, mistral-7b, llama2-7b, llama3.1-8b, llama3.1-70b-instruct'
        config._metadata['Test File'] = 'test_kv_cache.py'


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
