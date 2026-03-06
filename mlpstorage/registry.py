"""
Benchmark registry for dynamic benchmark registration and CLI building.

This module provides the BenchmarkRegistry class that enables:
- Dynamic registration of benchmark types
- Dynamic CLI argument building
- Extensibility without modifying core CLI code
"""

from typing import Dict, Type, Callable, List, Optional, Any


class BenchmarkRegistry:
    """Registry for benchmark types and their CLI configurations.

    This class provides a central registry for benchmark types, allowing
    new benchmarks to be registered without modifying the core CLI code.

    Usage:
        # Register a benchmark
        BenchmarkRegistry.register(
            name='training',
            benchmark_class=TrainingBenchmark,
            cli_builder=add_training_arguments,
            description='Training benchmark'
        )

        # Get a benchmark class
        benchmark_class = BenchmarkRegistry.get_benchmark_class('training')

        # Build CLI arguments dynamically
        for name in BenchmarkRegistry.get_all_names():
            BenchmarkRegistry.build_cli_args(name, subparser)
    """

    _benchmarks: Dict[str, Type] = {}
    _cli_builders: Dict[str, Callable] = {}
    _descriptions: Dict[str, str] = {}
    _help_texts: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, benchmark_class: Type,
                 cli_builder: Callable = None,
                 description: str = "",
                 help_text: str = "") -> None:
        """Register a benchmark type.

        Args:
            name: Unique name for the benchmark (e.g., 'training').
            benchmark_class: The benchmark class implementing BenchmarkInterface.
            cli_builder: Function that adds CLI arguments for this benchmark.
            description: Long description for the benchmark.
            help_text: Short help text for CLI.
        """
        cls._benchmarks[name] = benchmark_class
        if cli_builder:
            cls._cli_builders[name] = cli_builder
        if description:
            cls._descriptions[name] = description
        if help_text:
            cls._help_texts[name] = help_text

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a benchmark type.

        Args:
            name: Name of the benchmark to unregister.
        """
        cls._benchmarks.pop(name, None)
        cls._cli_builders.pop(name, None)
        cls._descriptions.pop(name, None)
        cls._help_texts.pop(name, None)

    @classmethod
    def get_benchmark_class(cls, name: str) -> Type:
        """Get benchmark class by name.

        Args:
            name: Name of the benchmark.

        Returns:
            The benchmark class.

        Raises:
            ValueError: If benchmark is not registered.
        """
        if name not in cls._benchmarks:
            raise ValueError(f"Unknown benchmark type: {name}. "
                           f"Available types: {list(cls._benchmarks.keys())}")
        return cls._benchmarks[name]

    @classmethod
    def get_all_names(cls) -> List[str]:
        """Get all registered benchmark names.

        Returns:
            List of benchmark names.
        """
        return list(cls._benchmarks.keys())

    @classmethod
    def get_description(cls, name: str) -> str:
        """Get description for a benchmark.

        Args:
            name: Name of the benchmark.

        Returns:
            Description string.
        """
        return cls._descriptions.get(name, "")

    @classmethod
    def get_help_text(cls, name: str) -> str:
        """Get help text for a benchmark.

        Args:
            name: Name of the benchmark.

        Returns:
            Help text string.
        """
        return cls._help_texts.get(name, f"{name} benchmark options")

    @classmethod
    def has_cli_builder(cls, name: str) -> bool:
        """Check if a benchmark has a CLI builder.

        Args:
            name: Name of the benchmark.

        Returns:
            True if CLI builder is registered.
        """
        return name in cls._cli_builders

    @classmethod
    def build_cli_args(cls, name: str, parser) -> None:
        """Build CLI arguments for a benchmark.

        Args:
            name: Name of the benchmark.
            parser: Argparse parser/subparser to add arguments to.
        """
        if name in cls._cli_builders:
            cls._cli_builders[name](parser)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a benchmark is registered.

        Args:
            name: Name of the benchmark.

        Returns:
            True if benchmark is registered.
        """
        return name in cls._benchmarks

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Useful for testing."""
        cls._benchmarks.clear()
        cls._cli_builders.clear()
        cls._descriptions.clear()
        cls._help_texts.clear()

    @classmethod
    def get_registry_info(cls) -> Dict[str, Any]:
        """Get information about all registered benchmarks.

        Returns:
            Dictionary with benchmark info.
        """
        return {
            name: {
                'class': cls._benchmarks[name].__name__,
                'has_cli_builder': name in cls._cli_builders,
                'description': cls._descriptions.get(name, ""),
                'help': cls._help_texts.get(name, ""),
            }
            for name in cls._benchmarks
        }
