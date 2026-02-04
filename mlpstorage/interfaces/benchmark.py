"""
Benchmark interface definitions for mlpstorage.

This module defines the abstract interface that all benchmarks must implement,
providing a consistent contract for benchmark behavior across different
benchmark types (training, checkpointing, vector database, KV cache).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum


class BenchmarkCommand(Enum):
    """Standard benchmark commands that all benchmarks should support."""
    RUN = "run"
    DATAGEN = "datagen"
    DATASIZE = "datasize"
    VALIDATE = "validate"


@dataclass
class BenchmarkConfig:
    """Configuration container for benchmark initialization.

    Attributes:
        name: Human-readable name of the benchmark.
        benchmark_type: The type identifier (e.g., 'training', 'checkpointing').
        config_path: Path to the benchmark's configuration file.
        supported_commands: List of commands this benchmark supports.
        requires_cluster_info: Whether cluster information collection is needed.
        requires_mpi: Whether MPI is required for execution.
        default_params: Default parameters for the benchmark.
    """
    name: str
    benchmark_type: str
    config_path: Optional[str] = None
    supported_commands: List[BenchmarkCommand] = field(default_factory=lambda: [BenchmarkCommand.RUN])
    requires_cluster_info: bool = True
    requires_mpi: bool = False
    default_params: Dict[str, Any] = field(default_factory=dict)


class BenchmarkInterface(ABC):
    """Abstract interface that all benchmarks must implement.

    This interface defines the contract for benchmark implementations,
    ensuring consistent behavior across different benchmark types.
    Implementing this interface enables:
    - Consistent CLI integration
    - Unified validation
    - Standardized result collection
    - Dependency injection for testing

    Example:
        class MyBenchmark(BenchmarkInterface):
            @property
            def config(self) -> BenchmarkConfig:
                return BenchmarkConfig(
                    name="My Benchmark",
                    benchmark_type="my_benchmark",
                    supported_commands=[BenchmarkCommand.RUN, BenchmarkCommand.DATAGEN]
                )

            def validate_args(self, args) -> List[str]:
                errors = []
                if not args.data_dir:
                    errors.append("--data-dir is required")
                return errors

            # ... implement other abstract methods
    """

    @property
    @abstractmethod
    def config(self) -> BenchmarkConfig:
        """Return benchmark configuration.

        Returns:
            BenchmarkConfig containing benchmark metadata and capabilities.
        """
        pass

    @abstractmethod
    def validate_args(self, args) -> List[str]:
        """Validate command-line arguments before execution.

        Args:
            args: Parsed command-line arguments (argparse.Namespace).

        Returns:
            List of error messages. Empty list indicates valid arguments.
        """
        pass

    @abstractmethod
    def get_command_handler(self, command: str) -> Optional[Callable]:
        """Return handler function for the given command.

        Args:
            command: Command string (e.g., 'run', 'datagen').

        Returns:
            Callable that handles the command, or None if not supported.
        """
        pass

    @abstractmethod
    def generate_command(self, command: str) -> str:
        """Generate the shell command to execute.

        Args:
            command: Command string (e.g., 'run', 'datagen').

        Returns:
            Shell command string ready for execution.
        """
        pass

    @abstractmethod
    def collect_results(self) -> Dict[str, Any]:
        """Collect and return benchmark results.

        Called after benchmark execution to gather results, metrics,
        and any other relevant data.

        Returns:
            Dictionary containing benchmark results and metadata.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get benchmark metadata for recording.

        Returns:
            Dictionary containing benchmark configuration, parameters,
            and system information for result recording.
        """
        pass
