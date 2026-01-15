"""
Interface definitions for mlpstorage.

This package defines the abstract interfaces (contracts) that components
must implement. Using interfaces enables:

- **Testability**: Components can be mocked/stubbed for unit testing
- **Extensibility**: New implementations can be added without modifying core code
- **Consistency**: All implementations follow the same contract
- **Documentation**: Interfaces clearly document expected behavior

Available Interfaces:

Benchmark Interfaces:
    - BenchmarkInterface: Core interface for all benchmark implementations
    - BenchmarkConfig: Configuration dataclass for benchmarks
    - BenchmarkCommand: Enum of standard benchmark commands

Validator Interfaces:
    - ValidatorInterface: Interface for parameter/result validators
    - SubmissionValidatorInterface: Interface for multi-run submission validators
    - ValidationResult: Result container for validation checks
    - ValidationCategory: Enum for validation outcomes (CLOSED, OPEN, INVALID)
    - ClosedRequirements: Requirements for CLOSED submission

Collector Interfaces:
    - ClusterCollectorInterface: Interface for distributed system info collection
    - LocalCollectorInterface: Interface for local-only system info collection
    - CollectionResult: Result container for collection operations

Example Usage:
    from mlpstorage.interfaces import BenchmarkInterface, BenchmarkConfig

    class MyBenchmark(BenchmarkInterface):
        @property
        def config(self) -> BenchmarkConfig:
            return BenchmarkConfig(
                name="My Benchmark",
                benchmark_type="my_benchmark"
            )
        # ... implement other abstract methods
"""

from mlpstorage.interfaces.benchmark import (
    BenchmarkInterface,
    BenchmarkConfig,
    BenchmarkCommand,
)

from mlpstorage.interfaces.validator import (
    ValidatorInterface,
    SubmissionValidatorInterface,
    ValidationResult,
    ValidationCategory,
    ClosedRequirements,
)

from mlpstorage.interfaces.collector import (
    ClusterCollectorInterface,
    LocalCollectorInterface,
    CollectionResult,
)

__all__ = [
    # Benchmark interfaces
    'BenchmarkInterface',
    'BenchmarkConfig',
    'BenchmarkCommand',
    # Validator interfaces
    'ValidatorInterface',
    'SubmissionValidatorInterface',
    'ValidationResult',
    'ValidationCategory',
    'ClosedRequirements',
    # Collector interfaces
    'ClusterCollectorInterface',
    'LocalCollectorInterface',
    'CollectionResult',
]
