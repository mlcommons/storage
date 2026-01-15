"""
Benchmark implementations for MLPerf Storage.

This module exports benchmark classes and registers them with
the BenchmarkRegistry for dynamic CLI construction.
"""

from mlpstorage.benchmarks.dlio import TrainingBenchmark, CheckpointingBenchmark
from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
from mlpstorage.registry import BenchmarkRegistry
from mlpstorage.cli import (
    PROGRAM_DESCRIPTIONS,
    add_training_arguments,
    add_checkpointing_arguments,
    add_vectordb_arguments,
    add_kvcache_arguments,
)


def register_benchmarks():
    """Register all benchmark types with the BenchmarkRegistry.

    This function is called at module import time to ensure benchmarks
    are available for dynamic CLI construction.
    """
    BenchmarkRegistry.register(
        name='training',
        benchmark_class=TrainingBenchmark,
        cli_builder=add_training_arguments,
        description=PROGRAM_DESCRIPTIONS['training'],
        help_text="Training benchmark options"
    )

    BenchmarkRegistry.register(
        name='checkpointing',
        benchmark_class=CheckpointingBenchmark,
        cli_builder=add_checkpointing_arguments,
        description=PROGRAM_DESCRIPTIONS['checkpointing'],
        help_text="Checkpointing benchmark options"
    )

    BenchmarkRegistry.register(
        name='vectordb',
        benchmark_class=VectorDBBenchmark,
        cli_builder=add_vectordb_arguments,
        description=PROGRAM_DESCRIPTIONS['vectordb'],
        help_text="VectorDB benchmark options"
    )

    BenchmarkRegistry.register(
        name='kvcache',
        benchmark_class=KVCacheBenchmark,
        cli_builder=add_kvcache_arguments,
        description=PROGRAM_DESCRIPTIONS['kvcache'],
        help_text="KV Cache benchmark options"
    )


# Register benchmarks at import time
register_benchmarks()

__all__ = [
    'TrainingBenchmark',
    'CheckpointingBenchmark',
    'VectorDBBenchmark',
    'KVCacheBenchmark',
    'register_benchmarks',
]
