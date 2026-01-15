# MLPerf Storage Code Improvement Plan

This document outlines a comprehensive, phased improvement plan for the mlpstorage codebase, focusing on testability, documentation, code organization, and modularity for future benchmark additions (Vector Database and KV Cache benchmarks).

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Phase 1: Foundation - Interfaces and Abstractions](#phase-1-foundation---interfaces-and-abstractions)
4. [Phase 2: Modular Rules Engine](#phase-2-modular-rules-engine)
5. [Phase 3: CLI Refactoring](#phase-3-cli-refactoring)
6. [Phase 4: Test Infrastructure Enhancement](#phase-4-test-infrastructure-enhancement)
7. [Phase 5: Documentation and Type Annotations](#phase-5-documentation-and-type-annotations)
8. [Phase 6: KV Cache Benchmark Integration](#phase-6-kv-cache-benchmark-integration)
9. [Phase 7: Plugin Architecture](#phase-7-plugin-architecture)

---

## Executive Summary

The mlpstorage codebase (~7,000 LOC) is a well-structured benchmarking framework, but requires improvements to:

1. **Support new benchmarks** (KV Cache, future workloads) without modifying core code
2. **Improve testability** through dependency injection and better separation of concerns
3. **Enhance maintainability** by splitting large modules and improving documentation
4. **Enable code reuse** through shared interfaces and utility abstractions

---

## Current State Analysis

### Strengths
- Clear separation between benchmarks, validation, and utilities
- MPI-based cluster collection provides rich system information
- Extensible validation framework with rule-based system
- Comprehensive logging with 8+ log levels

### Issues Identified

| Issue | Impact | Files Affected |
|-------|--------|----------------|
| Large monolithic files | Hard to navigate and test | `rules.py` (1,434 LOC), `cluster_collector.py` (1,196 LOC) |
| Tight coupling in CLI | Adding benchmarks requires CLI changes | `cli.py` (464 LOC) |
| Inconsistent initialization patterns | Benchmarks initialize differently | `dlio.py`, `vectordbbench.py` |
| Missing interfaces | No clear contracts for benchmarks | `benchmarks/` |
| Limited dependency injection | Hard to mock for tests | `base.py`, `dlio.py` |
| Embedded scripts | MPI script in string literal | `cluster_collector.py` |
| Code duplication | Parameter handling repeated | `dlio.py`, `cli.py` |

### Current Module Dependency Graph

```
main.py
  └── cli.py
      └── benchmarks/{Training,Checkpointing,Vector}Benchmark
          └── base.py (Benchmark ABC)
              ├── rules.py (1,434 LOC - validation + data classes)
              ├── cluster_collector.py (1,196 LOC)
              └── utils.py (388 LOC)
```

---

## Phase 1: Foundation - Interfaces and Abstractions

**Goal:** Create clear interfaces that define contracts for benchmarks, validators, and collectors.

**Duration:** 3-5 files created/modified

### Step 1.1: Create Benchmark Protocol/Interface

Create `mlpstorage/interfaces/benchmark.py`:

```python
from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

class BenchmarkCommand(Enum):
    """Standard benchmark commands that all benchmarks should support."""
    RUN = "run"
    DATAGEN = "datagen"
    DATASIZE = "datasize"
    VALIDATE = "validate"

@dataclass
class BenchmarkConfig:
    """Configuration container for benchmark initialization."""
    name: str
    benchmark_type: str
    config_path: str
    supported_commands: List[BenchmarkCommand]
    requires_cluster_info: bool = True
    requires_mpi: bool = False

class BenchmarkInterface(ABC):
    """Abstract interface that all benchmarks must implement."""

    @property
    @abstractmethod
    def config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        pass

    @abstractmethod
    def validate_args(self, args) -> List[str]:
        """Validate arguments, return list of error messages."""
        pass

    @abstractmethod
    def get_command_handler(self, command: str):
        """Return handler function for the given command."""
        pass

    @abstractmethod
    def generate_command(self, command: str) -> str:
        """Generate the shell command to execute."""
        pass

    @abstractmethod
    def collect_results(self) -> Dict[str, Any]:
        """Collect and return benchmark results."""
        pass
```

### Step 1.2: Create Validator Protocol

Create `mlpstorage/interfaces/validator.py`:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    is_closed: bool
    issues: List['Issue']
    warnings: List[str]

class ValidatorInterface(ABC):
    """Interface for benchmark parameter validators."""

    @abstractmethod
    def validate_pre_run(self, benchmark, args) -> ValidationResult:
        """Validate before benchmark execution."""
        pass

    @abstractmethod
    def validate_post_run(self, benchmark, results: Dict[str, Any]) -> ValidationResult:
        """Validate after benchmark execution."""
        pass

    @abstractmethod
    def get_closed_requirements(self) -> Dict[str, Any]:
        """Return requirements for closed submission."""
        pass
```

### Step 1.3: Create Cluster Collector Protocol

Create `mlpstorage/interfaces/collector.py`:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class ClusterCollectorInterface(ABC):
    """Interface for cluster information collectors."""

    @abstractmethod
    def collect(self, hosts: List[str], timeout: int = 60) -> Dict[str, Any]:
        """Collect information from all hosts."""
        pass

    @abstractmethod
    def collect_local(self) -> Dict[str, Any]:
        """Collect information from local host only."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this collector is available (e.g., MPI installed)."""
        pass
```

### Step 1.4: Create Interfaces Package

Create `mlpstorage/interfaces/__init__.py`:

```python
from .benchmark import BenchmarkInterface, BenchmarkConfig, BenchmarkCommand
from .validator import ValidatorInterface, ValidationResult
from .collector import ClusterCollectorInterface

__all__ = [
    'BenchmarkInterface',
    'BenchmarkConfig',
    'BenchmarkCommand',
    'ValidatorInterface',
    'ValidationResult',
    'ClusterCollectorInterface',
]
```

### Step 1.5: Update Base Benchmark

Modify `mlpstorage/benchmarks/base.py` to implement the interface:

1. Import the interface
2. Make `Benchmark` class implement `BenchmarkInterface`
3. Add default implementations for interface methods
4. Add dependency injection for collectors and validators

```python
# Add to base.py
from mlpstorage.interfaces import BenchmarkInterface, BenchmarkConfig

class Benchmark(BenchmarkInterface, abc.ABC):

    def __init__(self, args, logger=None, run_datetime=None, run_number=0,
                 cluster_collector=None, validator=None):
        # ... existing code ...

        # Dependency injection for testability
        self._cluster_collector = cluster_collector
        self._validator = validator
```

---

## Phase 2: Modular Rules Engine

**Goal:** Split the monolithic `rules.py` (1,434 LOC) into focused modules.

### Step 2.1: Create Rules Package Structure

```
mlpstorage/rules/
├── __init__.py           # Public API exports
├── base.py               # RulesChecker ABC, RuleState enum
├── issues.py             # Issue dataclass
├── models.py             # Data classes (HostInfo, ClusterInformation, etc.)
├── training.py           # TrainingRunRulesChecker, TrainingSubmissionRulesChecker
├── checkpointing.py      # CheckpointingRunRulesChecker, CheckpointSubmissionRulesChecker
├── vectordb.py           # VectorDBRulesChecker (new)
├── kvcache.py            # KVCacheRulesChecker (placeholder)
├── verifier.py           # BenchmarkVerifier
└── utils.py              # calculate_training_data_size, generate_output_location
```

### Step 2.2: Extract Data Classes to models.py

Move to `mlpstorage/rules/models.py`:
- `HostMemoryInfo`
- `HostCPUInfo`
- `HostInfo`
- `ClusterInformation`
- `RunID`
- `ProcessedRun`
- `BenchmarkResult`
- `BenchmarkRun`

### Step 2.3: Extract Training Rules

Create `mlpstorage/rules/training.py`:

```python
from mlpstorage.rules.base import RulesChecker, RunRulesChecker, MultiRunRulesChecker
from mlpstorage.rules.issues import Issue

class TrainingRunRulesChecker(RunRulesChecker):
    """Validate training benchmark parameters."""

    def check_model(self) -> Optional[Issue]:
        # Move from rules.py
        pass

    def check_accelerator(self) -> Optional[Issue]:
        # Move from rules.py
        pass

    # ... other training-specific checks
```

### Step 2.4: Extract Checkpointing Rules

Create `mlpstorage/rules/checkpointing.py`:

```python
class CheckpointingRunRulesChecker(RunRulesChecker):
    """Validate checkpointing benchmark parameters."""

    def check_model(self) -> Optional[Issue]:
        # LLM model validation
        pass

    def check_zero_level(self) -> Optional[Issue]:
        # DeepSpeed Zero level validation
        pass
```

### Step 2.5: Create VectorDB Rules (New)

Create `mlpstorage/rules/vectordb.py`:

```python
class VectorDBRunRulesChecker(RunRulesChecker):
    """Validate vector database benchmark parameters."""

    def check_dimension(self) -> Optional[Issue]:
        pass

    def check_index_type(self) -> Optional[Issue]:
        pass

    def check_metric_type(self) -> Optional[Issue]:
        pass
```

### Step 2.6: Update __init__.py for Backwards Compatibility

Create `mlpstorage/rules/__init__.py`:

```python
# Maintain backwards compatibility
from mlpstorage.rules.models import (
    HostMemoryInfo,
    HostCPUInfo,
    HostInfo,
    ClusterInformation,
    RunID,
    ProcessedRun,
    BenchmarkResult,
    BenchmarkRun,
)
from mlpstorage.rules.base import RulesChecker, RuleState
from mlpstorage.rules.issues import Issue
from mlpstorage.rules.training import TrainingRunRulesChecker, TrainingSubmissionRulesChecker
from mlpstorage.rules.checkpointing import CheckpointingRunRulesChecker, CheckpointSubmissionRulesChecker
from mlpstorage.rules.vectordb import VectorDBRunRulesChecker
from mlpstorage.rules.verifier import BenchmarkVerifier
from mlpstorage.rules.utils import calculate_training_data_size, generate_output_location, get_runs_files

__all__ = [
    # Models
    'HostMemoryInfo', 'HostCPUInfo', 'HostInfo', 'ClusterInformation',
    'RunID', 'ProcessedRun', 'BenchmarkResult', 'BenchmarkRun',
    # Base
    'RulesChecker', 'RuleState', 'Issue',
    # Validators
    'TrainingRunRulesChecker', 'TrainingSubmissionRulesChecker',
    'CheckpointingRunRulesChecker', 'CheckpointSubmissionRulesChecker',
    'VectorDBRunRulesChecker',
    'BenchmarkVerifier',
    # Utils
    'calculate_training_data_size', 'generate_output_location', 'get_runs_files',
]
```

---

## Phase 3: CLI Refactoring

**Goal:** Make CLI extensible for new benchmarks without modifying core CLI code.

### Step 3.1: Create Benchmark Registry

Create `mlpstorage/registry.py`:

```python
from typing import Dict, Type, Callable
from mlpstorage.interfaces import BenchmarkInterface

class BenchmarkRegistry:
    """Registry for benchmark types and their CLI configurations."""

    _benchmarks: Dict[str, Type[BenchmarkInterface]] = {}
    _cli_builders: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, benchmark_class: Type[BenchmarkInterface],
                 cli_builder: Callable = None):
        """Register a benchmark type."""
        cls._benchmarks[name] = benchmark_class
        if cli_builder:
            cls._cli_builders[name] = cli_builder

    @classmethod
    def get_benchmark_class(cls, name: str) -> Type[BenchmarkInterface]:
        """Get benchmark class by name."""
        if name not in cls._benchmarks:
            raise ValueError(f"Unknown benchmark type: {name}")
        return cls._benchmarks[name]

    @classmethod
    def get_all_names(cls) -> list:
        """Get all registered benchmark names."""
        return list(cls._benchmarks.keys())

    @classmethod
    def build_cli_args(cls, name: str, subparser):
        """Build CLI arguments for a benchmark."""
        if name in cls._cli_builders:
            cls._cli_builders[name](subparser)
```

### Step 3.2: Create CLI Argument Builders per Benchmark

Create `mlpstorage/cli/training_args.py`:

```python
def add_training_arguments(parser):
    """Add training-specific CLI arguments."""
    subparsers = parser.add_subparsers(dest='command')

    # datasize command
    datasize = subparsers.add_parser('datasize')
    add_common_training_args(datasize)
    datasize.add_argument('--max-accelerators', type=int, required=True)

    # datagen command
    datagen = subparsers.add_parser('datagen')
    add_common_training_args(datagen)

    # run command
    run = subparsers.add_parser('run')
    add_common_training_args(run)

def add_common_training_args(parser):
    """Add arguments common to all training commands."""
    parser.add_argument('--model', choices=['cosmoflow', 'resnet50', 'unet3d'])
    parser.add_argument('--accelerator-type', choices=['h100', 'a100'])
    parser.add_argument('--hosts', nargs='+')
    parser.add_argument('--num-accelerators', type=int)
    # ... etc
```

### Step 3.3: Create CLI Argument Builders for Other Benchmarks

Create similar files:
- `mlpstorage/cli/checkpointing_args.py`
- `mlpstorage/cli/vectordb_args.py`
- `mlpstorage/cli/kvcache_args.py`

### Step 3.4: Refactor Main CLI

Modify `mlpstorage/cli.py`:

```python
from mlpstorage.registry import BenchmarkRegistry

def build_parser():
    """Build argument parser dynamically from registry."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='program')

    # Add universal arguments
    add_universal_arguments(parser)

    # Dynamically add benchmark-specific parsers
    for name in BenchmarkRegistry.get_all_names():
        benchmark_parser = subparsers.add_parser(name)
        BenchmarkRegistry.build_cli_args(name, benchmark_parser)

    # Add non-benchmark commands (reports, history)
    add_utility_commands(subparsers)

    return parser
```

### Step 3.5: Register Benchmarks at Import Time

Update `mlpstorage/benchmarks/__init__.py`:

```python
from mlpstorage.registry import BenchmarkRegistry
from mlpstorage.benchmarks.dlio import TrainingBenchmark, CheckpointingBenchmark
from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
from mlpstorage.cli.training_args import add_training_arguments
from mlpstorage.cli.checkpointing_args import add_checkpointing_arguments
from mlpstorage.cli.vectordb_args import add_vectordb_arguments

# Register benchmarks
BenchmarkRegistry.register('training', TrainingBenchmark, add_training_arguments)
BenchmarkRegistry.register('checkpointing', CheckpointingBenchmark, add_checkpointing_arguments)
BenchmarkRegistry.register('vectordb', VectorDBBenchmark, add_vectordb_arguments)
```

---

## Phase 4: Test Infrastructure Enhancement

**Goal:** Improve testability and test organization.

### Step 4.1: Create Test Fixtures Package

Create `mlpstorage/tests/fixtures/__init__.py`:

```python
from .mock_logger import MockLogger, create_mock_logger
from .sample_data import (
    SAMPLE_MEMINFO,
    SAMPLE_CPUINFO,
    SAMPLE_DISKSTATS,
    SAMPLE_HOSTS,
    create_sample_cluster_info,
    create_sample_benchmark_args,
)
from .mock_executor import MockCommandExecutor
from .mock_collector import MockClusterCollector

__all__ = [
    'MockLogger',
    'create_mock_logger',
    'SAMPLE_MEMINFO',
    'SAMPLE_CPUINFO',
    'SAMPLE_DISKSTATS',
    'SAMPLE_HOSTS',
    'create_sample_cluster_info',
    'create_sample_benchmark_args',
    'MockCommandExecutor',
    'MockClusterCollector',
]
```

### Step 4.2: Create MockCommandExecutor

Create `mlpstorage/tests/fixtures/mock_executor.py`:

```python
from typing import Tuple, List, Dict, Any

class MockCommandExecutor:
    """Mock command executor for testing without subprocess calls."""

    def __init__(self, responses: Dict[str, Tuple[str, str, int]] = None):
        self.responses = responses or {}
        self.executed_commands: List[str] = []

    def execute(self, command: str, **kwargs) -> Tuple[str, str, int]:
        """Record command and return mock response."""
        self.executed_commands.append(command)

        # Check for pattern matches
        for pattern, response in self.responses.items():
            if pattern in command:
                return response

        # Default success response
        return ("", "", 0)

    def assert_command_executed(self, pattern: str):
        """Assert that a command matching pattern was executed."""
        for cmd in self.executed_commands:
            if pattern in cmd:
                return True
        raise AssertionError(f"No command matching '{pattern}' was executed")
```

### Step 4.3: Create MockClusterCollector

Create `mlpstorage/tests/fixtures/mock_collector.py`:

```python
from mlpstorage.interfaces import ClusterCollectorInterface

class MockClusterCollector(ClusterCollectorInterface):
    """Mock cluster collector for testing."""

    def __init__(self, mock_data: dict = None, should_fail: bool = False):
        self.mock_data = mock_data or self._default_data()
        self.should_fail = should_fail
        self.collect_calls = []

    def collect(self, hosts, timeout=60):
        self.collect_calls.append({'hosts': hosts, 'timeout': timeout})
        if self.should_fail:
            raise RuntimeError("Mock collector failure")
        return self.mock_data

    def collect_local(self):
        return self.mock_data.get('hosts', [{}])[0]

    def is_available(self):
        return not self.should_fail

    def _default_data(self):
        return {
            'hosts': [
                {
                    'hostname': 'test-host-1',
                    'memory': {'total': 64 * 1024**3},
                    'cpu': {'num_cores': 32, 'num_logical_cores': 64},
                }
            ],
            '_metadata': {'collection_method': 'mock'}
        }
```

### Step 4.4: Split Test Files by Category

Reorganize `mlpstorage/tests/`:

```
mlpstorage/tests/
├── fixtures/
│   ├── __init__.py
│   ├── mock_logger.py
│   ├── mock_executor.py
│   ├── mock_collector.py
│   └── sample_data.py
├── unit/
│   ├── test_parsers.py         # /proc file parsers
│   ├── test_models.py          # Data classes
│   ├── test_validators.py      # Rules validators
│   └── test_utils.py           # Utility functions
├── integration/
│   ├── test_benchmark_flow.py  # Full benchmark execution
│   ├── test_cluster_collection.py  # MPI collection
│   └── test_cli.py             # CLI parsing and routing
├── benchmarks/
│   ├── test_training.py
│   ├── test_checkpointing.py
│   ├── test_vectordb.py
│   └── test_kvcache.py
└── conftest.py                 # Shared pytest fixtures
```

### Step 4.5: Create Shared Pytest Fixtures

Create `mlpstorage/tests/conftest.py`:

```python
import pytest
from mlpstorage.tests.fixtures import (
    MockLogger,
    MockCommandExecutor,
    MockClusterCollector,
    create_sample_benchmark_args,
)

@pytest.fixture
def mock_logger():
    return MockLogger()

@pytest.fixture
def mock_executor():
    return MockCommandExecutor()

@pytest.fixture
def mock_collector():
    return MockClusterCollector()

@pytest.fixture
def training_args():
    return create_sample_benchmark_args('training')

@pytest.fixture
def checkpointing_args():
    return create_sample_benchmark_args('checkpointing')

@pytest.fixture
def temp_results_dir(tmp_path):
    """Create a temporary results directory."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir
```

### Step 4.6: Add Integration Test Template

Create `mlpstorage/tests/integration/test_benchmark_flow.py`:

```python
import pytest
from mlpstorage.benchmarks.dlio import TrainingBenchmark
from mlpstorage.tests.fixtures import MockCommandExecutor, MockClusterCollector

class TestBenchmarkFlow:
    """Integration tests for full benchmark execution flow."""

    def test_training_benchmark_dry_run(self, training_args, mock_logger, tmp_path):
        """Test training benchmark in what-if mode."""
        training_args.what_if = True
        training_args.results_dir = str(tmp_path)

        benchmark = TrainingBenchmark(training_args, logger=mock_logger)
        result = benchmark.run()

        assert result == 0
        # Verify metadata was written
        assert (tmp_path / "training" / "metadata.json").exists()

    def test_benchmark_with_mock_executor(self, training_args, mock_logger):
        """Test benchmark using mock command executor."""
        mock_exec = MockCommandExecutor({
            'dlio_benchmark': ('stdout', '', 0)
        })

        # Inject mock executor
        benchmark = TrainingBenchmark(training_args, logger=mock_logger)
        benchmark.cmd_executor = mock_exec

        benchmark.run()

        mock_exec.assert_command_executed('dlio_benchmark')
```

---

## Phase 5: Documentation and Type Annotations

**Goal:** Add comprehensive documentation and type hints for better IDE support and maintainability.

### Step 5.1: Add Type Annotations to Core Modules

Priority files for type annotations:
1. `mlpstorage/interfaces/*.py` - Already typed in Phase 1
2. `mlpstorage/rules/models.py` - Data classes
3. `mlpstorage/benchmarks/base.py` - Base class
4. `mlpstorage/utils.py` - Utility functions

Example for `utils.py`:

```python
from typing import Dict, Any, List, Optional, Tuple, Union

def read_config_from_file(file_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        file_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the parsed configuration.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    ...

def generate_mpi_prefix_cmd(
    mpi_bin: str,
    hosts: List[str],
    num_processes: int,
    oversubscribe: bool = False,
    allow_run_as_root: bool = False,
    mpi_params: Optional[List[str]] = None,
    logger: Optional['Logger'] = None
) -> str:
    """Generate MPI command prefix for distributed execution.

    Args:
        mpi_bin: Path to mpirun/mpiexec binary.
        hosts: List of hostnames to run on.
        num_processes: Number of MPI processes.
        oversubscribe: Allow more processes than cores.
        allow_run_as_root: Allow running as root user.
        mpi_params: Additional MPI parameters.
        logger: Logger instance for debug output.

    Returns:
        MPI command prefix string (e.g., "mpirun -np 4 -H host1,host2").
    """
    ...
```

### Step 5.2: Create Module-Level Docstrings

Add module docstrings to each file explaining its purpose:

```python
# mlpstorage/benchmarks/base.py
"""
Base Benchmark Classes
======================

This module defines the abstract base class for all benchmarks in the
MLPerf Storage suite. It provides common functionality for:

- Command execution with signal handling
- Metadata generation and persistence
- Cluster information collection
- Benchmark verification

Classes:
    Benchmark: Abstract base class for all benchmark implementations.

Example:
    class MyBenchmark(Benchmark):
        BENCHMARK_TYPE = BENCHMARK_TYPES.my_benchmark

        def _run(self):
            cmd = self.generate_command()
            self._execute_command(cmd)
"""
```

### Step 5.3: Create Architecture Documentation

Create `mlpstorage/docs/ARCHITECTURE.md`:

```markdown
# MLPerf Storage Architecture

## Module Overview

### Core Modules

| Module | Purpose |
|--------|---------|
| `main.py` | Entry point, signal handling, command routing |
| `cli.py` | Argument parsing, validation |
| `config.py` | Constants, enumerations, configuration |
| `utils.py` | Shared utilities (command execution, JSON, YAML) |

### Benchmark Framework

```
benchmarks/
├── base.py          # Benchmark ABC with common functionality
├── dlio.py          # DLIO-based benchmarks (Training, Checkpointing)
├── vectordbbench.py # Vector database benchmark
└── kvcache.py       # KV cache benchmark (planned)
```

### Validation Framework

```
rules/
├── base.py          # RulesChecker ABC
├── models.py        # Data classes (HostInfo, ClusterInformation)
├── training.py      # Training validation rules
├── checkpointing.py # Checkpointing validation rules
└── verifier.py      # BenchmarkVerifier orchestrator
```

## Data Flow

1. CLI parses arguments
2. Benchmark class instantiated with args
3. Cluster information collected (MPI or args-based)
4. Benchmark parameters validated
5. Command generated and executed
6. Results collected and metadata written
7. Post-run validation (optional)

## Adding a New Benchmark

See [Adding Benchmarks](./ADDING_BENCHMARKS.md)
```

### Step 5.4: Create "Adding Benchmarks" Guide

Create `mlpstorage/docs/ADDING_BENCHMARKS.md`:

```markdown
# Adding a New Benchmark

This guide explains how to add a new benchmark to the MLPerf Storage suite.

## Step 1: Create Benchmark Class

Create `mlpstorage/benchmarks/mybenchmark.py`:

```python
from mlpstorage.benchmarks.base import Benchmark
from mlpstorage.config import BENCHMARK_TYPES

class MyBenchmark(Benchmark):
    BENCHMARK_TYPE = BENCHMARK_TYPES.my_benchmark

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.command_method_map = {
            "run": self.execute_run,
            "datagen": self.execute_datagen,
        }

    def _run(self):
        handler = self.command_method_map.get(self.args.command)
        if handler:
            return handler()
        raise ValueError(f"Unknown command: {self.args.command}")

    def execute_run(self):
        cmd = self.generate_command()
        self._execute_command(cmd)
```

## Step 2: Create CLI Arguments

Create `mlpstorage/cli/mybenchmark_args.py`:

```python
def add_mybenchmark_arguments(parser):
    subparsers = parser.add_subparsers(dest='command')

    run = subparsers.add_parser('run')
    run.add_argument('--my-param', required=True)

    datagen = subparsers.add_parser('datagen')
    # ...
```

## Step 3: Create Validation Rules

Create `mlpstorage/rules/mybenchmark.py`:

```python
from mlpstorage.rules.base import RunRulesChecker

class MyBenchmarkRunRulesChecker(RunRulesChecker):
    def check_my_param(self):
        # Validate my-param
        pass
```

## Step 4: Register the Benchmark

Update `mlpstorage/benchmarks/__init__.py`:

```python
from mlpstorage.registry import BenchmarkRegistry
from mlpstorage.benchmarks.mybenchmark import MyBenchmark
from mlpstorage.cli.mybenchmark_args import add_mybenchmark_arguments

BenchmarkRegistry.register('mybenchmark', MyBenchmark, add_mybenchmark_arguments)
```

## Step 5: Add Tests

Create `mlpstorage/tests/benchmarks/test_mybenchmark.py`
```

---

## Phase 6: KV Cache Benchmark Integration

**Goal:** Integrate the existing `kv_cache_benchmark/` code into the mlpstorage framework.

### Step 6.1: Analyze Existing KV Cache Code

The `kv_cache_benchmark/` directory contains:
- `kv-cache.py` (168K) - Main benchmark implementation
- `kv-cache_sharegpt_replay.py` (141K) - ShareGPT replay variant
- `kv-cache-wrapper.sh` (51K) - Shell wrapper
- `validate.sh` - Validation script

### Step 6.2: Create KVCacheBenchmark Class

Create `mlpstorage/benchmarks/kvcache.py`:

```python
from mlpstorage.benchmarks.base import Benchmark
from mlpstorage.config import BENCHMARK_TYPES

class KVCacheBenchmark(Benchmark):
    """KV Cache benchmark for LLM inference storage."""

    BENCHMARK_TYPE = BENCHMARK_TYPES.kv_cache
    KVCACHE_BIN = "kv-cache.py"

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.command_method_map = {
            "run": self.execute_run,
            "validate": self.execute_validate,
        }

        # Load KV cache specific config
        self.config_path = self._find_config_path()

    def _run(self):
        handler = self.command_method_map.get(self.args.command)
        if handler:
            return handler()
        raise ValueError(f"Unknown command: {self.args.command}")

    def execute_run(self):
        """Execute KV cache benchmark."""
        cmd = self._build_kvcache_command()
        self._execute_command(cmd, output_file_prefix="kvcache_run")

    def execute_validate(self):
        """Run validation on KV cache results."""
        # Call validate.sh or implement validation
        pass

    def _build_kvcache_command(self) -> str:
        """Build the kv-cache.py command with parameters."""
        cmd = f"python {self.KVCACHE_BIN}"

        # Add common parameters
        if hasattr(self.args, 'model') and self.args.model:
            cmd += f" --model {self.args.model}"
        if hasattr(self.args, 'batch_size') and self.args.batch_size:
            cmd += f" --batch-size {self.args.batch_size}"
        if hasattr(self.args, 'sequence_length') and self.args.sequence_length:
            cmd += f" --sequence-length {self.args.sequence_length}"

        # Add output directory
        cmd += f" --output-dir {self.run_result_output}"

        return cmd
```

### Step 6.3: Add KV Cache CLI Arguments

Create `mlpstorage/cli/kvcache_args.py`:

```python
def add_kvcache_arguments(parser):
    """Add KV cache benchmark CLI arguments."""
    subparsers = parser.add_subparsers(dest='command')

    # Run command
    run = subparsers.add_parser('run', help='Run KV cache benchmark')
    add_common_kvcache_args(run)
    run.add_argument('--runtime', type=int, default=60,
                    help='Benchmark runtime in seconds')
    run.add_argument('--mode', choices=['prefill', 'decode', 'mixed'],
                    default='mixed', help='Benchmark mode')

    # Validate command
    validate = subparsers.add_parser('validate', help='Validate results')
    validate.add_argument('--results-path', required=True,
                         help='Path to results to validate')

def add_common_kvcache_args(parser):
    """Add arguments common to KV cache commands."""
    parser.add_argument('--model', required=True,
                       choices=['llama3-8b', 'llama3-70b', 'llama3-405b'],
                       help='LLM model to benchmark')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--sequence-length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--cache-backend',
                       choices=['memory', 'disk', 'redis', 'lmcache'],
                       default='memory', help='KV cache storage backend')
    parser.add_argument('--cache-size-gb', type=float, default=10.0,
                       help='Cache size in GB')
```

### Step 6.4: Add KV Cache Validation Rules

Create `mlpstorage/rules/kvcache.py`:

```python
from mlpstorage.rules.base import RunRulesChecker
from mlpstorage.rules.issues import Issue
from mlpstorage.config import PARAM_VALIDATION

class KVCacheRunRulesChecker(RunRulesChecker):
    """Validation rules for KV cache benchmark."""

    # Closed submission requirements
    CLOSED_REQUIREMENTS = {
        'min_runtime': 60,  # seconds
        'min_batch_size': 1,
        'allowed_backends': ['disk', 'lmcache'],
    }

    def check_runtime(self):
        """Check if runtime meets closed submission requirements."""
        runtime = getattr(self.benchmark.args, 'runtime', 0)
        if runtime < self.CLOSED_REQUIREMENTS['min_runtime']:
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message=f"Runtime {runtime}s is below minimum {self.CLOSED_REQUIREMENTS['min_runtime']}s",
                parameter='runtime',
                expected=self.CLOSED_REQUIREMENTS['min_runtime'],
                actual=runtime
            )
        return None

    def check_cache_backend(self):
        """Check if cache backend is allowed for closed submission."""
        backend = getattr(self.benchmark.args, 'cache_backend', 'memory')
        if backend not in self.CLOSED_REQUIREMENTS['allowed_backends']:
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message=f"Cache backend '{backend}' not allowed for closed submission",
                parameter='cache_backend',
                expected=self.CLOSED_REQUIREMENTS['allowed_backends'],
                actual=backend
            )
        return None
```

### Step 6.5: Add KV Cache to BENCHMARK_TYPES

Update `mlpstorage/config.py`:

```python
class BENCHMARK_TYPES(enum.Enum):
    training = "training"
    checkpointing = "checkpointing"
    vector_database = "vector_database"
    kv_cache = "kv_cache"  # Add this
```

### Step 6.6: Register KV Cache Benchmark

Update `mlpstorage/benchmarks/__init__.py`:

```python
from mlpstorage.benchmarks.kvcache import KVCacheBenchmark
from mlpstorage.cli.kvcache_args import add_kvcache_arguments

BenchmarkRegistry.register('kvcache', KVCacheBenchmark, add_kvcache_arguments)
```

---

## Phase 7: Plugin Architecture

**Goal:** Enable third-party benchmarks to be added without modifying core code.

### Step 7.1: Create Plugin Discovery

Create `mlpstorage/plugins/__init__.py`:

```python
import importlib
import pkgutil
from pathlib import Path
from typing import List, Type

from mlpstorage.interfaces import BenchmarkInterface
from mlpstorage.registry import BenchmarkRegistry

def discover_plugins(plugin_dirs: List[str] = None) -> List[str]:
    """Discover and load benchmark plugins.

    Args:
        plugin_dirs: Additional directories to search for plugins.

    Returns:
        List of loaded plugin names.
    """
    loaded = []

    # Default plugin locations
    search_paths = [
        Path(__file__).parent / "builtin",  # Built-in plugins
        Path.home() / ".mlpstorage" / "plugins",  # User plugins
    ]

    if plugin_dirs:
        search_paths.extend(Path(d) for d in plugin_dirs)

    for path in search_paths:
        if not path.exists():
            continue

        for finder, name, ispkg in pkgutil.iter_modules([str(path)]):
            try:
                module = importlib.import_module(f"mlpstorage.plugins.{name}")
                if hasattr(module, 'register_benchmark'):
                    module.register_benchmark(BenchmarkRegistry)
                    loaded.append(name)
            except ImportError as e:
                print(f"Warning: Failed to load plugin {name}: {e}")

    return loaded
```

### Step 7.2: Create Plugin Template

Create `mlpstorage/plugins/template.py`:

```python
"""
Template for creating a new benchmark plugin.

To create a plugin:
1. Copy this file to ~/.mlpstorage/plugins/my_plugin.py
2. Implement the required classes and functions
3. The plugin will be auto-discovered on next run
"""

from mlpstorage.benchmarks.base import Benchmark
from mlpstorage.interfaces import BenchmarkInterface
from mlpstorage.registry import BenchmarkRegistry

class MyPluginBenchmark(Benchmark):
    """Example plugin benchmark implementation."""

    BENCHMARK_TYPE = None  # Will be set dynamically

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def _run(self):
        # Implement benchmark logic
        pass

def add_cli_arguments(parser):
    """Add CLI arguments for this benchmark."""
    subparsers = parser.add_subparsers(dest='command')
    run = subparsers.add_parser('run')
    run.add_argument('--my-option', help='Plugin-specific option')

def register_benchmark(registry: BenchmarkRegistry):
    """Called by plugin discovery to register this benchmark."""
    registry.register(
        name='my-plugin',
        benchmark_class=MyPluginBenchmark,
        cli_builder=add_cli_arguments
    )
```

### Step 7.3: Update Main Entry Point for Plugin Loading

Modify `mlpstorage/main.py`:

```python
from mlpstorage.plugins import discover_plugins

def main():
    # Discover and load plugins before parsing arguments
    plugin_dirs = os.environ.get('MLPS_PLUGIN_DIRS', '').split(':')
    loaded_plugins = discover_plugins(plugin_dirs)

    if loaded_plugins:
        logging.debug(f"Loaded plugins: {loaded_plugins}")

    # Continue with normal argument parsing
    args = parse_arguments()
    # ...
```

---

## Implementation Priority

| Phase | Priority | Effort | Impact |
|-------|----------|--------|--------|
| Phase 1: Interfaces | High | Medium | Foundation for all other phases |
| Phase 2: Rules Modularization | High | High | Better organization, testability |
| Phase 4: Test Infrastructure | High | Medium | Enables safe refactoring |
| Phase 6: KV Cache Integration | High | Medium | Immediate user value |
| Phase 3: CLI Refactoring | Medium | High | Extensibility |
| Phase 5: Documentation | Medium | Low | Long-term maintainability |
| Phase 7: Plugin Architecture | Low | Medium | Future extensibility |

## Recommended Order

1. **Phase 1** (Interfaces) - Create foundation
2. **Phase 4** (Tests) - Ensure safety net for refactoring
3. **Phase 2** (Rules) - Modularize largest file
4. **Phase 6** (KV Cache) - Add new benchmark using new patterns
5. **Phase 3** (CLI) - Improve extensibility
6. **Phase 5** (Docs) - Document everything
7. **Phase 7** (Plugins) - Enable external extensions

---

## Success Metrics

After completing all phases:

- [ ] All benchmarks implement `BenchmarkInterface`
- [ ] `rules.py` split into <300 LOC modules
- [ ] New benchmarks can be added without modifying `cli.py`
- [ ] Test coverage >80%
- [ ] All public functions have docstrings and type annotations
- [ ] KV Cache benchmark integrated and working
- [ ] Plugin system allows external benchmark registration

---

## Appendix: File Changes Summary

### New Files Created
- `mlpstorage/interfaces/__init__.py`
- `mlpstorage/interfaces/benchmark.py`
- `mlpstorage/interfaces/validator.py`
- `mlpstorage/interfaces/collector.py`
- `mlpstorage/registry.py`
- `mlpstorage/rules/__init__.py` (package)
- `mlpstorage/rules/base.py`
- `mlpstorage/rules/issues.py`
- `mlpstorage/rules/models.py`
- `mlpstorage/rules/training.py`
- `mlpstorage/rules/checkpointing.py`
- `mlpstorage/rules/vectordb.py`
- `mlpstorage/rules/kvcache.py`
- `mlpstorage/rules/verifier.py`
- `mlpstorage/rules/utils.py`
- `mlpstorage/cli/__init__.py` (package)
- `mlpstorage/cli/training_args.py`
- `mlpstorage/cli/checkpointing_args.py`
- `mlpstorage/cli/vectordb_args.py`
- `mlpstorage/cli/kvcache_args.py`
- `mlpstorage/benchmarks/kvcache.py`
- `mlpstorage/plugins/__init__.py`
- `mlpstorage/plugins/template.py`
- `mlpstorage/tests/fixtures/__init__.py`
- `mlpstorage/tests/fixtures/mock_*.py`
- `mlpstorage/tests/conftest.py`
- `mlpstorage/docs/ARCHITECTURE.md`
- `mlpstorage/docs/ADDING_BENCHMARKS.md`

### Files Modified
- `mlpstorage/benchmarks/base.py` - Implement interface
- `mlpstorage/benchmarks/__init__.py` - Registry integration
- `mlpstorage/cli.py` - Dynamic parser building
- `mlpstorage/config.py` - Add kv_cache benchmark type
- `mlpstorage/main.py` - Plugin discovery
- `mlpstorage/rules.py` - Converted to package (deprecated)

### Files Deprecated
- `mlpstorage/rules.py` - Replaced by `mlpstorage/rules/` package
