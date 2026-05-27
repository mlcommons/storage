"""
Core data models for KV Cache Benchmark.

Defines enums, dataclasses, and model configurations used throughout
the benchmark: ModelConfig, InferencePhase, GenerationMode, QoSLevel,
QoSSLA, UserProfile, InferenceRequest, etc.
"""

import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
from datetime import datetime

from kv_cache.config import cfg


# ============================================================================
# CORE DATA MODELS
# ============================================================================

@dataclass
class ModelConfig:
    """
    Configuration for a model's KV cache requirements.

    This dataclass holds the architectural parameters of an LLM that are essential
    for calculating the size of its KV cache.
    """
    name: str
    num_layers: int         # Number of transformer layers in the model.
    hidden_dim: int         # The size of the main hidden state vector.
    num_heads: int          # Number of attention heads for queries (Q).
    kv_heads: int           # Number of attention heads for keys/values (K/V). For GQA, kv_heads < num_heads.
    dtype: str = 'float16'  # Data type used for cache tensors (e.g., float16, bfloat16).
    _kv_dim_override: int = 0  # Optional override for kv_dim_per_head (e.g., DeepSeek MLA uses 56)
    attention_type: str = 'mha'  # 'mha', 'gqa', or 'mla'
    kv_lora_rank: int = 0       # MLA: compressed KV latent dimension (d_c)
    qk_rope_head_dim: int = 0   # MLA: decoupled RoPE key dimension (d_R^h)

    @property
    def bytes_per_element(self) -> int:
        """Returns the size in bytes of a single element based on the data type."""
        dtype_map = {'float32': 4, 'float16': 2, 'bfloat16': 2, 'int8': 1}
        return dtype_map.get(self.dtype, 2)

    @property
    def kv_dim_per_head(self) -> int:
        """Calculates the dimension of each Key/Value attention head."""
        if self._kv_dim_override > 0:
            return self._kv_dim_override
        return self.hidden_dim // self.num_heads

    @property
    def kv_cache_size_per_token(self) -> int:
        """
        Calculates the total memory in bytes required to store the KV cache for a single token.

        For MHA/GQA: num_layers * kv_heads * head_dim * 2 (K+V) * dtype_bytes
        For MLA:     num_layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
                     MLA jointly compresses K and V into a single latent vector (no ×2),
                     plus a shared RoPE key that must also be cached.
        """
        if self.attention_type == 'mla':
            return self.num_layers * (self.kv_lora_rank + self.qk_rope_head_dim) * self.bytes_per_element
        return self.num_layers * self.kv_heads * self.kv_dim_per_head * 2 * self.bytes_per_element


_DEFAULT_MODEL_CONFIGS = {
    'tiny-1b': {'name': 'Tiny 1B', 'num_layers': 12, 'hidden_dim': 1024, 'num_heads': 8, 'kv_heads': 4, 'dtype': 'float16'},
    'mistral-7b': {'name': 'Mistral 7B', 'num_layers': 32, 'hidden_dim': 4096, 'num_heads': 32, 'kv_heads': 8, 'dtype': 'float16'},
    'llama2-7b': {'name': 'Llama 2 7B', 'num_layers': 32, 'hidden_dim': 4096, 'num_heads': 32, 'kv_heads': 32, 'dtype': 'float16'},
    'llama3.1-8b': {'name': 'Llama 3.1 8B', 'num_layers': 32, 'hidden_dim': 4096, 'num_heads': 32, 'kv_heads': 8, 'dtype': 'float16'},
    'llama3.1-70b-instruct': {'name': 'Llama 3.1 70B Instruct', 'num_layers': 80, 'hidden_dim': 8192, 'num_heads': 64, 'kv_heads': 8, 'dtype': 'float16'},
    'deepseek-v3': {'name': 'DeepSeek V3', 'num_layers': 61, 'hidden_dim': 7168, 'num_heads': 128, 'kv_heads': 128, 'dtype': 'float16',
                     'attention_type': 'mla', 'kv_lora_rank': 512, 'qk_rope_head_dim': 64},
    'qwen3-32b': {'name': 'Qwen3 32B', 'num_layers': 64, 'hidden_dim': 5120, 'num_heads': 64, 'kv_heads': 8, 'kv_dim_per_head': 128, 'dtype': 'float16'},
    'gpt-oss-120b': {'name': 'GPT OSS 120B (MoE)', 'num_layers': 36, 'hidden_dim': 2880, 'num_heads': 64, 'kv_heads': 8, 'kv_dim_per_head': 64, 'dtype': 'float16'},
    'gpt-oss-20b': {'name': 'GPT OSS 20B (MoE)', 'num_layers': 24, 'hidden_dim': 2880, 'num_heads': 64, 'kv_heads': 8, 'kv_dim_per_head': 64, 'dtype': 'float16'},
}


def get_model_configs() -> Dict[str, ModelConfig]:
    """
    Returns model configurations, merging config.yaml values with defaults.
    Models defined in YAML are added to/override the defaults.
    """
    configs = {}
    
    # Get models from config.yaml (empty dict if not defined)
    yaml_models = cfg('model_configs', default={})
    
    # Merge: defaults + yaml (yaml overrides defaults)
    all_model_keys = set(_DEFAULT_MODEL_CONFIGS.keys()) | set(yaml_models.keys())
    
    for model_key in all_model_keys:
        defaults = _DEFAULT_MODEL_CONFIGS.get(model_key, {})
        
        configs[model_key] = ModelConfig(
            name=cfg('model_configs', model_key, 'name', default=defaults.get('name', model_key)),
            num_layers=cfg('model_configs', model_key, 'num_layers', default=defaults.get('num_layers', 32)),
            hidden_dim=cfg('model_configs', model_key, 'hidden_dim', default=defaults.get('hidden_dim', 4096)),
            num_heads=cfg('model_configs', model_key, 'num_heads', default=defaults.get('num_heads', 32)),
            kv_heads=cfg('model_configs', model_key, 'kv_heads', default=defaults.get('kv_heads', 8)),
            dtype=cfg('model_configs', model_key, 'dtype', default=defaults.get('dtype', 'float16')),
            _kv_dim_override=cfg('model_configs', model_key, 'kv_dim_per_head', default=defaults.get('kv_dim_per_head', 0)),
            attention_type=cfg('model_configs', model_key, 'attention_type', default=defaults.get('attention_type', 'mha')),
            kv_lora_rank=cfg('model_configs', model_key, 'kv_lora_rank', default=defaults.get('kv_lora_rank', 0)),
            qk_rope_head_dim=cfg('model_configs', model_key, 'qk_rope_head_dim', default=defaults.get('qk_rope_head_dim', 0)),
        )
    
    return configs


# For backward compatibility
MODEL_CONFIGS = get_model_configs()


# ============================================================================
# PHASE-AWARE PROCESSING
# ============================================================================

class InferencePhase(Enum):
    """Enumeration for the two main phases of LLM inference."""
    PREFILL = "prefill"
    DECODE = "decode"
    PREFILL_DECODE = "both"


class GenerationMode(Enum):
    """Enumeration for token generation simulation modes."""
    NONE = "none"
    FAST = "fast"
    REALISTIC = "realistic"

# Defines the sleep time per token to simulate GPU work for each mode.
GENERATION_TIMING = {
    GenerationMode.NONE: 0.0,
    GenerationMode.FAST: 0.002,
    GenerationMode.REALISTIC: 0.030,
}


# ============================================================================
# QOS SUPPORT
# ============================================================================

class QoSLevel(Enum):
    """Enumeration for Quality of Service (QoS) levels, defining user priority."""
    INTERACTIVE = "interactive"
    RESPONSIVE = "responsive"
    BATCH = "batch"


@dataclass
class QoSSLA:
    """
    Represents a Service Level Agreement (SLA) for a given QoS level.
    Defines the performance targets and tracks violations.
    """
    qos_level: QoSLevel
    target_latency_p95_ms: float
    target_latency_p99_ms: float
    target_latency_p999_ms: float
    target_latency_p9999_ms: float
    priority: int

    # SLA violation tracking
    violations: int = 0
    total_requests: int = 0

    @property
    def sla_compliance(self) -> float:
        """Calculates the percentage of requests that met the SLA target."""
        if self.total_requests == 0:
            return 1.0
        return 1.0 - (self.violations / self.total_requests)


# Default QoS profile values (overridden by config.yaml when loaded)
_DEFAULT_QOS_PROFILES = {
    'interactive': {'target_latency_p95_ms': 50, 'target_latency_p99_ms': 100,
                   'target_latency_p999_ms': 150, 'target_latency_p9999_ms': 200, 'priority': 3},
    'responsive': {'target_latency_p95_ms': 100, 'target_latency_p99_ms': 200,
                  'target_latency_p999_ms': 350, 'target_latency_p9999_ms': 500, 'priority': 2},
    'batch': {'target_latency_p95_ms': 1000, 'target_latency_p99_ms': 5000,
             'target_latency_p999_ms': 7500, 'target_latency_p9999_ms': 10000, 'priority': 1},
}


def get_qos_profiles() -> Dict[QoSLevel, QoSSLA]:
    """
    Returns QoS profiles, using config.yaml values if loaded, otherwise defaults.
    """
    profiles = {}
    for level in QoSLevel:
        level_key = level.value
        defaults = _DEFAULT_QOS_PROFILES[level_key]

        profiles[level] = QoSSLA(
            qos_level=level,
            target_latency_p95_ms=cfg('qos_profiles', level_key, 'target_latency_p95_ms',
                                     default=defaults['target_latency_p95_ms']),
            target_latency_p99_ms=cfg('qos_profiles', level_key, 'target_latency_p99_ms',
                                     default=defaults['target_latency_p99_ms']),
            target_latency_p999_ms=cfg('qos_profiles', level_key, 'target_latency_p999_ms',
                                      default=defaults['target_latency_p999_ms']),
            target_latency_p9999_ms=cfg('qos_profiles', level_key, 'target_latency_p9999_ms',
                                       default=defaults['target_latency_p9999_ms']),
            priority=cfg('qos_profiles', level_key, 'priority', default=defaults['priority']),
        )
    return profiles


# For backward compatibility, QOS_PROFILES can still be used as a dict
# but code should prefer get_qos_profiles() to pick up config changes
QOS_PROFILES = get_qos_profiles()


# ============================================================================
# USER AND REQUEST MODELS
# ============================================================================

@dataclass
class UserProfile:
    """Represents a simulated user with specific behavior patterns."""
    user_id: str
    context_length: int
    generation_length: int
    think_time: float
    priority: int
    qos_level: QoSLevel
    session_start: datetime = field(default_factory=datetime.now)
    total_latency: float = 0.0
    request_count: int = 0


@dataclass
class InferenceRequest:
    """Represents a single, atomic inference request sent to the benchmark."""
    user_id: str
    request_id: str
    timestamp: datetime
    context_tokens: int
    generate_tokens: int
    priority: int
    phase: InferencePhase = InferencePhase.PREFILL_DECODE
    qos_level: QoSLevel = QoSLevel.BATCH
    cache_key: Optional[str] = None

    # Timing fields to track latency at different stages.
    submit_time: float = field(default_factory=time.perf_counter)
    start_time: float = 0
    complete_time: float = 0
    storage_complete_time: float = 0

    # Conversation tracking for stateful workloads.
    conversation_id: Optional[str] = None
    turn_number: int = 0

    def __post_init__(self):
        if self.cache_key is None:
            if self.conversation_id:
                self.cache_key = f"{self.conversation_id}_turn_{self.turn_number}"
            else:
                self.cache_key = f"{self.user_id}_ctx"

    @property
    def total_latency_ms(self) -> float:
        """Calculates the total end-to-end latency for the request in milliseconds."""
        if self.complete_time == 0:
            return 0
        return (self.complete_time - self.submit_time) * 1000

    @property
    def storage_e2e_latency_ms(self) -> float:
        """Calculates storage-only end-to-end latency (excludes GPU simulation time)."""
        if self.storage_complete_time == 0:
            return 0
        return (self.storage_complete_time - self.submit_time) * 1000

    @property
    def service_latency_ms(self) -> float:
        """Calculates service latency (time from worker dequeue to completion, excludes queue wait)."""
        if self.start_time == 0 or self.complete_time == 0:
            return 0
        return (self.complete_time - self.start_time) * 1000
