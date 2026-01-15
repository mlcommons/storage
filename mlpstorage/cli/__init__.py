"""
CLI argument builders for MLPerf Storage benchmarks.

This package provides modular CLI argument builders that can be
registered with the BenchmarkRegistry for dynamic CLI construction.

Modules:
    - common_args: Shared help messages and universal arguments
    - training_args: Training benchmark arguments
    - checkpointing_args: Checkpointing benchmark arguments
    - vectordb_args: VectorDB benchmark arguments
    - utility_args: Reports and history arguments

Usage:
    from mlpstorage.cli import add_training_arguments
    from mlpstorage.registry import BenchmarkRegistry

    BenchmarkRegistry.register(
        'training',
        TrainingBenchmark,
        add_training_arguments
    )
"""

from mlpstorage.cli.common_args import (
    HELP_MESSAGES,
    PROGRAM_DESCRIPTIONS,
    add_universal_arguments,
    add_mpi_arguments,
    add_host_arguments,
    add_dlio_arguments,
)

from mlpstorage.cli.training_args import add_training_arguments
from mlpstorage.cli.checkpointing_args import add_checkpointing_arguments
from mlpstorage.cli.vectordb_args import add_vectordb_arguments
from mlpstorage.cli.kvcache_args import add_kvcache_arguments
from mlpstorage.cli.utility_args import add_reports_arguments, add_history_arguments

__all__ = [
    # Common
    'HELP_MESSAGES',
    'PROGRAM_DESCRIPTIONS',
    'add_universal_arguments',
    'add_mpi_arguments',
    'add_host_arguments',
    'add_dlio_arguments',
    # Benchmark argument builders
    'add_training_arguments',
    'add_checkpointing_arguments',
    'add_vectordb_arguments',
    'add_kvcache_arguments',
    # Utility argument builders
    'add_reports_arguments',
    'add_history_arguments',
]
