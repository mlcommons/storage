# Coding Conventions

**Analysis Date:** 2026-01-23

## Naming Patterns

**Files:**
- Modules use `snake_case.py`: `cli_parser.py`, `cluster_collector.py`, `validation_helpers.py`
- Test files prefixed with `test_`: `test_cli.py`, `test_benchmarks_base.py`
- Fixture files use descriptive names: `mock_collector.py`, `mock_executor.py`, `sample_data.py`
- CLI argument builders: `{benchmark}_args.py` pattern (e.g., `training_args.py`, `checkpointing_args.py`)

**Functions:**
- Use `snake_case`: `calculate_training_data_size()`, `generate_output_location()`, `is_valid_datetime_format()`
- Private methods prefixed with underscore: `_run()`, `_execute_command()`, `_collect_cluster_information()`
- Factory functions prefixed with `create_`: `create_sample_cluster_info()`, `create_mock_logger()`
- Checker methods prefixed with `check_`: `check_dataset_size()`, `check_memory_requirements()`

**Variables:**
- Use `snake_case`: `num_processes`, `client_host_memory_in_gb`, `run_datetime`
- Constants use `UPPER_SNAKE_CASE`: `MLPS_DEBUG`, `DATETIME_STR`, `DEFAULT_RESULTS_DIR`
- Boolean variables often prefixed with `is_`, `has_`, `should_`: `is_valid_datetime_format`, `should_fail`

**Classes:**
- Use `PascalCase`: `Benchmark`, `CommandExecutor`, `ClusterInformation`, `BenchmarkVerifier`
- Abstract base classes: `RulesChecker`, `BenchmarkInterface`
- Mock classes prefixed with `Mock`: `MockCommandExecutor`, `MockClusterCollector`, `MockLogger`
- Exceptions suffixed with `Error`: `ConfigurationError`, `BenchmarkExecutionError`, `DependencyError`

**Enums:**
- Use `UPPER_SNAKE_CASE` for class names: `BENCHMARK_TYPES`, `PARAM_VALIDATION`, `EXEC_TYPE`
- Enum values are lowercase strings: `BENCHMARK_TYPES.training = "training"`

## Code Style

**Formatting:**
- No explicit formatter configured (no `.prettierrc`, `.flake8`, `ruff.toml`)
- Indentation: 4 spaces
- Line length: Generally 80-120 characters
- Strings: Single quotes preferred for identifiers, double quotes for user-facing messages

**Linting:**
- No explicit linter configuration detected
- Follow standard PEP 8 conventions

**Type Hints:**
- Use type hints throughout, especially in function signatures
- Import from `typing` for complex types: `Dict`, `Any`, `Optional`, `List`, `Tuple`, `Set`
- Use `TYPE_CHECKING` guard for import-only types to avoid circular imports:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging
```

## Import Organization

**Order:**
1. Standard library imports (alphabetical)
2. Third-party imports (alphabetical)
3. Local application imports (alphabetical)

**Example from `mlpstorage/utils.py`:**
```python
import concurrent.futures
import enum
import io
import json
import logging
import math
import os
import pprint
import psutil
import subprocess
import shlex
import select
import signal
import sys
import threading
import yaml
from datetime import datetime
from typing import Any, List, Union, Optional, Dict, Tuple, Set

from mlpstorage.config import CONFIGS_ROOT_DIR, MPIRUN, MPIEXEC, MPI_RUN_BIN, MPI_EXEC_BIN
```

**Path Aliases:**
- No path aliases configured (no `pyproject.toml` `[tool.ruff]` or `tsconfig.json` paths)
- Use relative imports within packages: `from mlpstorage.config import ...`
- Import from package `__init__.py` for public APIs: `from tests.fixtures import MockLogger`

## Error Handling

**Custom Exception Hierarchy:**
```python
# Base exception in mlpstorage/errors.py
MLPStorageException  # Base class with structured error info
├── ConfigurationError      # Configuration/parameter issues
├── BenchmarkExecutionError # Runtime execution failures
├── ValidationError         # Validation failures
├── FileSystemError         # File/directory issues
├── MPIError                # MPI/cluster issues
└── DependencyError         # Missing dependencies
```

**Pattern - Structured Errors:**
```python
raise ConfigurationError(
    message="Invalid model specified",
    parameter="model",
    expected=MODELS,
    actual=args.model,
    suggestion="Use one of: cosmoflow, resnet50, unet3d"
)
```

**Pattern - sys.exit for CLI:**
```python
if self.verification == PARAM_VALIDATION.INVALID:
    self.logger.error(f'Invalid configuration found. Aborting benchmark run.')
    sys.exit(1)
```

## Logging

**Framework:** Custom logger with extended levels in `mlpstorage/mlps_logging.py`

**Extended Log Levels:**
- `debug`, `info`, `warning`, `error`, `critical` (standard)
- `status` - High-visibility status updates
- `verbose` - Detailed operational info
- `verboser` - Even more detail
- `ridiculous` - Extremely detailed debug info
- `result` - Benchmark results output

**Pattern:**
```python
self.logger = setup_logging(name=f"{self.BENCHMARK_TYPE}_benchmark", stream_log_level=args.stream_log_level)
apply_logging_options(self.logger, args)

self.logger.status(f'Benchmark results directory: {self.run_result_output}')
self.logger.debug(f'Collecting cluster information via MPI...')
self.logger.warning(f'MPI cluster info collection failed: {e}')
```

## Comments

**Module Docstrings:**
- Every module has a docstring explaining its purpose
- List main classes/functions at module level

```python
"""
Utility Functions for MLPerf Storage Benchmarks.

This module provides shared utility functions used throughout the mlpstorage
framework, including:

- JSON encoding with custom type handling
- Configuration file loading and manipulation
- Dictionary operations (nesting, flattening, updates)
- Command execution with signal handling
- MPI command generation

Classes:
    MLPSJsonEncoder: Custom JSON encoder for mlpstorage types.
    CommandExecutor: Execute shell commands with live output streaming.

Functions:
    read_config_from_file: Load YAML configuration files.
    ...
"""
```

**Class and Method Docstrings:**
- Use Google-style docstrings with Args, Returns, Raises sections
- Include Examples where helpful

```python
def is_valid_datetime_format(datetime_str: str) -> bool:
    """Check if a string is a valid datetime in the format "YYYYMMDD_HHMMSS".

    Args:
        datetime_str: String to validate.

    Returns:
        True if the string matches the datetime format, False otherwise.

    Example:
        >>> is_valid_datetime_format("20250115_143022")
        True
        >>> is_valid_datetime_format("invalid")
        False
    """
```

## Function Design

**Size:** Functions are generally focused and under 50 lines. Complex logic is split into helper methods.

**Parameters:**
- Use type hints for all parameters
- Use `Optional[Type]` for parameters that can be None
- Provide default values where sensible
- Use keyword-only arguments after `*` for clarity in complex functions

```python
def __init__(
    self,
    args: Namespace,
    logger: Optional['logging.Logger'] = None,
    run_datetime: Optional[str] = None,
    run_number: int = 0,
    cluster_collector: Optional[Any] = None,
    validator: Optional[Any] = None
) -> None:
```

**Return Values:**
- Use explicit return type hints
- Return tuples for multiple values: `Tuple[str, str, int]`
- Use `Optional` when None is a valid return
- Prefer returning data over modifying in place

## Module Design

**Exports:**
- Use `__init__.py` to define public API
- Export commonly used classes and functions at package level

```python
# tests/fixtures/__init__.py
from tests.fixtures.mock_logger import MockLogger, create_mock_logger
from tests.fixtures.mock_executor import MockCommandExecutor
from tests.fixtures.mock_collector import MockClusterCollector
from tests.fixtures.sample_data import (
    SAMPLE_MEMINFO,
    SAMPLE_CPUINFO,
    ...
)
```

**Barrel Files:**
- `mlpstorage/cli/__init__.py` exports all argument builders
- `mlpstorage/rules/__init__.py` exports all rule-related classes
- `tests/fixtures/__init__.py` exports all mock utilities

## Patterns

**Abstract Base Classes:**
```python
class Benchmark(BenchmarkInterface, abc.ABC):
    BENCHMARK_TYPE = None  # Class attribute to be set by subclass

    @abc.abstractmethod
    def _run(self) -> int:
        """Run the actual benchmark execution."""
        raise NotImplementedError
```

**Dependency Injection:**
```python
def __init__(
    self,
    args: Namespace,
    logger: Optional['logging.Logger'] = None,
    cluster_collector: Optional[Any] = None,  # For testing without MPI
    validator: Optional[Any] = None           # For testing validation
) -> None:
```

**Factory Functions:**
```python
def create_sample_cluster_info(
    num_hosts: int = 2,
    memory_gb_per_host: int = 256,
    cpu_cores_per_host: int = 64,
    logger: Optional[Any] = None
) -> Any:
```

**Dataclasses for Configuration:**
```python
@dataclass
class BenchmarkConfig:
    name: str
    benchmark_type: str
    config_path: Optional[str] = None
    supported_commands: List[BenchmarkCommand] = field(default_factory=lambda: [BenchmarkCommand.RUN])
    requires_cluster_info: bool = True
    requires_mpi: bool = False
    default_params: Dict[str, Any] = field(default_factory=dict)
```

**Dynamic Method Discovery:**
```python
# In RulesChecker
self.check_methods = [
    getattr(self, method) for method in dir(self)
    if callable(getattr(self, method)) and method.startswith('check_')
]
```

---

*Convention analysis: 2026-01-23*
