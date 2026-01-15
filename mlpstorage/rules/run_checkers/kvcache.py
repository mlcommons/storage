"""
KV Cache benchmark run rules checker.

Validates KV Cache benchmark parameters for individual runs.
Note: KV Cache is currently a preview benchmark, so validation
rules are less strict than training/checkpointing.
"""

from typing import Optional, List

from mlpstorage.config import (
    BENCHMARK_TYPES,
    PARAM_VALIDATION,
    KVCACHE_MODELS,
    KVCACHE_PERFORMANCE_PROFILES,
    KVCACHE_GENERATION_MODES,
)
from mlpstorage.rules.issues import Issue
from mlpstorage.rules.run_checkers.base import RunRulesChecker


class KVCacheRunRulesChecker(RunRulesChecker):
    """Rules checker for KV Cache benchmarks.

    KV Cache benchmark validates LLM inference storage performance including:
    - Multi-tier cache configurations (GPU → CPU → NVMe)
    - Multi-user simulation
    - Phase-aware processing (prefill/decode)

    Currently in preview mode - rules are informational rather than strict.
    """

    # Minimum requirements for valid KV Cache runs
    MIN_DURATION_SECONDS = 30
    MIN_NUM_USERS = 1
    MIN_GPU_MEM_GB = 1.0
    MIN_CPU_MEM_GB = 1.0

    # Recommended values for meaningful benchmarks
    RECOMMENDED_DURATION_SECONDS = 60
    RECOMMENDED_MIN_USERS = 10

    def check_benchmark_type(self) -> Optional[Issue]:
        """Verify this is a KV Cache benchmark."""
        if self.benchmark_run.benchmark_type != BENCHMARK_TYPES.kv_cache:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid benchmark type: {self.benchmark_run.benchmark_type}",
                parameter="benchmark_type",
                expected=BENCHMARK_TYPES.kv_cache,
                actual=self.benchmark_run.benchmark_type
            )
        return None

    def check_model(self) -> Optional[Issue]:
        """Verify the model configuration is valid."""
        model = self.benchmark_run.parameters.get('model')
        if model is None:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="Missing model parameter",
                parameter="model"
            )

        if model not in KVCACHE_MODELS:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid KV Cache model: {model}",
                parameter="model",
                expected=f"one of {KVCACHE_MODELS}",
                actual=model
            )
        return None

    def check_num_users(self) -> Optional[Issue]:
        """Verify num_users is valid."""
        num_users = self.benchmark_run.parameters.get('num_users', 100)

        if num_users < self.MIN_NUM_USERS:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"num_users must be at least {self.MIN_NUM_USERS}",
                parameter="num_users",
                expected=f">= {self.MIN_NUM_USERS}",
                actual=num_users
            )

        if num_users < self.RECOMMENDED_MIN_USERS:
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message=f"num_users below recommended minimum of {self.RECOMMENDED_MIN_USERS}",
                parameter="num_users",
                expected=f">= {self.RECOMMENDED_MIN_USERS}",
                actual=num_users
            )

        return None

    def check_duration(self) -> Optional[Issue]:
        """Verify benchmark duration is valid."""
        duration = self.benchmark_run.parameters.get('duration', 60)

        if duration < self.MIN_DURATION_SECONDS:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Duration must be at least {self.MIN_DURATION_SECONDS} seconds",
                parameter="duration",
                expected=f">= {self.MIN_DURATION_SECONDS}",
                actual=duration
            )

        if duration < self.RECOMMENDED_DURATION_SECONDS:
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message=f"Duration below recommended minimum of {self.RECOMMENDED_DURATION_SECONDS}s",
                parameter="duration",
                expected=f">= {self.RECOMMENDED_DURATION_SECONDS}",
                actual=duration
            )

        return None

    def check_cache_configuration(self) -> Optional[Issue]:
        """Verify cache tier configuration is valid."""
        gpu_mem_gb = self.benchmark_run.parameters.get('gpu_mem_gb', 16.0)
        cpu_mem_gb = self.benchmark_run.parameters.get('cpu_mem_gb', 32.0)

        if gpu_mem_gb < self.MIN_GPU_MEM_GB:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"GPU memory must be at least {self.MIN_GPU_MEM_GB} GB",
                parameter="gpu_mem_gb",
                expected=f">= {self.MIN_GPU_MEM_GB}",
                actual=gpu_mem_gb
            )

        if cpu_mem_gb < self.MIN_CPU_MEM_GB:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"CPU memory must be at least {self.MIN_CPU_MEM_GB} GB",
                parameter="cpu_mem_gb",
                expected=f">= {self.MIN_CPU_MEM_GB}",
                actual=cpu_mem_gb
            )

        return None

    def check_generation_mode(self) -> Optional[Issue]:
        """Verify generation mode is valid."""
        generation_mode = self.benchmark_run.parameters.get('generation_mode', 'realistic')

        if generation_mode not in KVCACHE_GENERATION_MODES:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid generation mode: {generation_mode}",
                parameter="generation_mode",
                expected=f"one of {KVCACHE_GENERATION_MODES}",
                actual=generation_mode
            )

        # 'none' mode is allowed but considered OPEN only
        if generation_mode == 'none':
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message="Generation mode 'none' skips token simulation",
                parameter="generation_mode",
                expected="'fast' or 'realistic'",
                actual=generation_mode
            )

        return None

    def check_performance_profile(self) -> Optional[Issue]:
        """Verify performance profile is valid."""
        performance_profile = self.benchmark_run.parameters.get('performance_profile', 'latency')

        if performance_profile not in KVCACHE_PERFORMANCE_PROFILES:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid performance profile: {performance_profile}",
                parameter="performance_profile",
                expected=f"one of {KVCACHE_PERFORMANCE_PROFILES}",
                actual=performance_profile
            )

        return None

    def check_preview_status(self) -> Optional[Issue]:
        """
        Return informational issue that KV Cache is in preview.

        This is always returned to inform users that this benchmark
        is not yet accepted for closed submissions.
        """
        return Issue(
            validation=PARAM_VALIDATION.OPEN,
            message="KV Cache benchmark is in preview status - not accepted for closed submissions",
            parameter="benchmark_status"
        )
