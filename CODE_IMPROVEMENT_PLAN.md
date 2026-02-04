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
9. [Phase 7: Reporting System Refactoring](#phase-7-reporting-system-refactoring)
10. [Phase 8: Error Handling and User Messaging](#phase-8-error-handling-and-user-messaging)

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

**Goal:** Split the monolithic `rules.py` (1,434 LOC) into focused modules with a clear separation between single-run rule checkers (`RunRulesChecker`) and multi-run submission checkers (`MultiRunRulesChecker`).

### Step 2.1: Create Rules Package Structure

The rules package should separate concerns into distinct subpackages:

```
mlpstorage/rules/
├── __init__.py              # Public API exports
├── base.py                  # RulesChecker ABC, RuleState enum, RunRulesChecker, MultiRunRulesChecker
├── issues.py                # Issue dataclass
├── models.py                # Data classes (HostInfo, ClusterInformation, etc.)
├── run_checkers/            # Single-run validation checkers
│   ├── __init__.py
│   ├── base.py              # RunRulesChecker base class
│   ├── training.py          # TrainingRunRulesChecker
│   ├── checkpointing.py     # CheckpointingRunRulesChecker
│   ├── vectordb.py          # VectorDBRunRulesChecker
│   └── kvcache.py           # KVCacheRunRulesChecker
├── submission_checkers/     # Multi-run submission validation checkers
│   ├── __init__.py
│   ├── base.py              # MultiRunRulesChecker base class
│   ├── training.py          # TrainingSubmissionRulesChecker
│   └── checkpointing.py     # CheckpointSubmissionRulesChecker
├── verifier.py              # BenchmarkVerifier orchestrator
└── utils.py                 # calculate_training_data_size, generate_output_location
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
- `BenchmarkRunData`

### Step 2.3: Define Base Classes with Clear Contracts

Create `mlpstorage/rules/base.py`:

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from mlpstorage.rules.issues import Issue

class RulesChecker(ABC):
    """Base class for all rule checkers."""

    def __init__(self, logger, *args, **kwargs):
        self.logger = logger
        self.issues = []
        self.check_methods = [
            getattr(self, method) for method in dir(self)
            if callable(getattr(self, method)) and method.startswith('check_')
        ]

    def run_checks(self) -> List[Issue]:
        """Run all check methods and return issues."""
        self.issues = []
        for check_method in self.check_methods:
            try:
                result = check_method()
                if result:
                    if isinstance(result, list):
                        self.issues.extend(result)
                    else:
                        self.issues.append(result)
            except Exception as e:
                self.logger.error(f"Check {check_method.__name__} failed: {e}")
                self.issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Check failed: {e}",
                    severity="error"
                ))
        return self.issues
```

### Step 2.4: Create RunRulesChecker Base Class

Create `mlpstorage/rules/run_checkers/base.py`:

```python
from mlpstorage.rules.base import RulesChecker
from mlpstorage.rules.models import BenchmarkRun

class RunRulesChecker(RulesChecker):
    """
    Base class for single-run validation.

    RunRulesCheckers validate individual benchmark runs for:
    - Parameter correctness (model, accelerator, batch size, etc.)
    - System requirements (memory, CPU, etc.)
    - Configuration validity (allowed parameter overrides)

    Each check method should:
    1. Return None if the check passes for CLOSED submission
    2. Return Issue with PARAM_VALIDATION.OPEN if the check fails but is allowed for OPEN
    3. Return Issue with PARAM_VALIDATION.INVALID if the check fails completely
    """

    def __init__(self, benchmark_run: BenchmarkRun, logger, *args, **kwargs):
        super().__init__(logger, *args, **kwargs)
        self.benchmark_run = benchmark_run

    @property
    def parameters(self):
        """Access benchmark run parameters."""
        return self.benchmark_run.parameters

    @property
    def override_parameters(self):
        """Access override parameters."""
        return self.benchmark_run.override_parameters

    @property
    def system_info(self):
        """Access system information."""
        return self.benchmark_run.system_info
```

### Step 2.5: Create MultiRunRulesChecker Base Class

Create `mlpstorage/rules/submission_checkers/base.py`:

```python
from typing import List
from mlpstorage.rules.base import RulesChecker
from mlpstorage.rules.models import BenchmarkRun
from mlpstorage.rules.issues import Issue
from mlpstorage.config import PARAM_VALIDATION

class MultiRunRulesChecker(RulesChecker):
    """
    Base class for multi-run submission validation.

    MultiRunRulesCheckers validate groups of benchmark runs as a submission:
    - Required number of runs per workload
    - Consistency across runs (same model, accelerator, etc.)
    - Aggregate metrics requirements

    These checkers operate on already-validated individual runs.
    """

    def __init__(self, benchmark_runs: List[BenchmarkRun], logger, *args, **kwargs):
        super().__init__(logger, *args, **kwargs)
        if not isinstance(benchmark_runs, (list, tuple)):
            raise TypeError("benchmark_runs must be a list or tuple")
        self.benchmark_runs = benchmark_runs

    def check_runs_valid(self) -> Optional[Issue]:
        """Verify all individual runs passed validation."""
        category_set = {run.category for run in self.benchmark_runs}

        if PARAM_VALIDATION.INVALID in category_set:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="Submission contains INVALID runs",
                parameter="run_categories",
                expected="All runs OPEN or CLOSED",
                actual=[cat.value.upper() for cat in category_set]
            )
        elif PARAM_VALIDATION.OPEN in category_set:
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message="Submission contains OPEN runs - qualifies for OPEN category only",
                parameter="run_categories",
                actual=[cat.value.upper() for cat in category_set]
            )
        return None  # All runs are CLOSED

    def check_run_consistency(self) -> Optional[Issue]:
        """Verify all runs have consistent configuration."""
        models = {run.model for run in self.benchmark_runs}
        if len(models) > 1:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="Inconsistent models across runs in submission",
                parameter="model",
                expected="Single model",
                actual=list(models)
            )
        return None
```

### Step 2.6: Extract Training Run Rules

Create `mlpstorage/rules/run_checkers/training.py`:

```python
from typing import Optional, List
from mlpstorage.rules.run_checkers.base import RunRulesChecker
from mlpstorage.rules.issues import Issue
from mlpstorage.config import PARAM_VALIDATION, BENCHMARK_TYPES

class TrainingRunRulesChecker(RunRulesChecker):
    """Validate training benchmark parameters for a single run."""

    # Parameters allowed for CLOSED submission
    CLOSED_ALLOWED_PARAMS = [
        'dataset.num_files_train',
        'dataset.num_subfolders_train',
        'dataset.data_folder',
        'reader.read_threads',
        'reader.computation_threads',
        'reader.transfer_size',
        'reader.odirect',
        'reader.prefetch_size',
        'checkpoint.checkpoint_folder',
        'storage.storage_type',
        'storage.storage_root',
    ]

    # Parameters allowed for OPEN submission (but not CLOSED)
    OPEN_ALLOWED_PARAMS = [
        'framework',
        'dataset.format',
        'dataset.num_samples_per_file',
        'reader.data_loader',
    ]

    def check_benchmark_type(self) -> Optional[Issue]:
        """Verify this is a training benchmark."""
        if self.benchmark_run.benchmark_type != BENCHMARK_TYPES.training:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Expected training benchmark, got {self.benchmark_run.benchmark_type}",
                parameter="benchmark_type",
                expected=BENCHMARK_TYPES.training.name,
                actual=self.benchmark_run.benchmark_type.name if self.benchmark_run.benchmark_type else None
            )
        return None

    def check_allowed_params(self) -> List[Issue]:
        """Check if parameter overrides are allowed for CLOSED submission."""
        issues = []
        for param, value in self.override_parameters.items():
            if param.startswith("workflow"):
                continue  # Handled separately

            if param in self.CLOSED_ALLOWED_PARAMS:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.CLOSED,
                    message=f"Parameter override allowed for CLOSED: {param}",
                    parameter=param,
                    actual=value
                ))
            elif param in self.OPEN_ALLOWED_PARAMS:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.OPEN,
                    message=f"Parameter override only allowed for OPEN: {param}",
                    parameter=param,
                    actual=value
                ))
            else:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Parameter override not allowed: {param}",
                    parameter=param,
                    expected="Allowed parameter",
                    actual=value
                ))
        return issues

    def check_num_files_train(self) -> Optional[Issue]:
        """Check if training file count meets requirements."""
        # Implementation moved from rules.py
        pass
```

### Step 2.7: Extract Training Submission Rules

Create `mlpstorage/rules/submission_checkers/training.py`:

```python
from typing import Optional
from mlpstorage.rules.submission_checkers.base import MultiRunRulesChecker
from mlpstorage.rules.issues import Issue
from mlpstorage.config import PARAM_VALIDATION, MODELS

class TrainingSubmissionRulesChecker(MultiRunRulesChecker):
    """Validate training benchmark submissions (multiple runs)."""

    REQUIRED_RUNS = 5  # Required number of runs for closed submission
    supported_models = MODELS

    def check_num_runs(self) -> Optional[Issue]:
        """Require 5 runs for training benchmark closed submission."""
        num_runs = len(self.benchmark_runs)
        if num_runs < self.REQUIRED_RUNS:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Training submission requires {self.REQUIRED_RUNS} runs",
                parameter="num_runs",
                expected=self.REQUIRED_RUNS,
                actual=num_runs
            )
        return Issue(
            validation=PARAM_VALIDATION.CLOSED,
            message=f"Training submission has required {self.REQUIRED_RUNS} runs",
            parameter="num_runs",
            expected=self.REQUIRED_RUNS,
            actual=num_runs
        )
```

### Step 2.8: Extract Checkpointing Run Rules

Create `mlpstorage/rules/run_checkers/checkpointing.py`:

```python
from typing import Optional
from mlpstorage.rules.run_checkers.base import RunRulesChecker
from mlpstorage.rules.issues import Issue
from mlpstorage.config import PARAM_VALIDATION, BENCHMARK_TYPES, LLM_MODELS

class CheckpointingRunRulesChecker(RunRulesChecker):
    """Validate checkpointing benchmark parameters for a single run."""

    def check_benchmark_type(self) -> Optional[Issue]:
        """Verify this is a checkpointing benchmark."""
        if self.benchmark_run.benchmark_type != BENCHMARK_TYPES.checkpointing:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Expected checkpointing benchmark",
                parameter="benchmark_type",
                expected=BENCHMARK_TYPES.checkpointing.name,
                actual=self.benchmark_run.benchmark_type.name if self.benchmark_run.benchmark_type else None
            )
        return None

    def check_model(self) -> Optional[Issue]:
        """Verify model is a valid LLM model."""
        if self.benchmark_run.model not in [m.value for m in LLM_MODELS]:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid model for checkpointing benchmark",
                parameter="model",
                expected=[m.value for m in LLM_MODELS],
                actual=self.benchmark_run.model
            )
        return None
```

### Step 2.9: Extract Checkpointing Submission Rules

Create `mlpstorage/rules/submission_checkers/checkpointing.py`:

```python
from typing import Optional, List
from mlpstorage.rules.submission_checkers.base import MultiRunRulesChecker
from mlpstorage.rules.issues import Issue
from mlpstorage.config import PARAM_VALIDATION, BENCHMARK_TYPES, LLM_MODELS

class CheckpointSubmissionRulesChecker(MultiRunRulesChecker):
    """Validate checkpointing benchmark submissions."""

    REQUIRED_WRITES = 10
    REQUIRED_READS = 10
    supported_models = LLM_MODELS

    def check_num_operations(self) -> List[Issue]:
        """
        Require 10 total writes and 10 total reads for checkpointing submissions.

        Operations can be spread across multiple runs.
        """
        issues = []
        num_writes = num_reads = 0

        for run in self.benchmark_runs:
            if run.benchmark_type == BENCHMARK_TYPES.checkpointing:
                checkpoint_params = run.parameters.get('checkpoint', {})
                num_writes += checkpoint_params.get('num_checkpoints_write', 0)
                num_reads += checkpoint_params.get('num_checkpoints_read', 0)

        # Check writes
        if num_writes < self.REQUIRED_WRITES:
            issues.append(Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Insufficient checkpoint write operations",
                parameter="num_checkpoints_write",
                expected=self.REQUIRED_WRITES,
                actual=num_writes
            ))
        else:
            issues.append(Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=f"Checkpoint write operations meet requirement",
                parameter="num_checkpoints_write",
                expected=self.REQUIRED_WRITES,
                actual=num_writes
            ))

        # Check reads
        if num_reads < self.REQUIRED_READS:
            issues.append(Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Insufficient checkpoint read operations",
                parameter="num_checkpoints_read",
                expected=self.REQUIRED_READS,
                actual=num_reads
            ))
        else:
            issues.append(Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=f"Checkpoint read operations meet requirement",
                parameter="num_checkpoints_read",
                expected=self.REQUIRED_READS,
                actual=num_reads
            ))

        return issues
```

### Step 2.10: Create KV Cache Rules

Create `mlpstorage/rules/run_checkers/kvcache.py`:

```python
from typing import Optional
from mlpstorage.rules.run_checkers.base import RunRulesChecker
from mlpstorage.rules.issues import Issue
from mlpstorage.config import PARAM_VALIDATION

class KVCacheRunRulesChecker(RunRulesChecker):
    """Validation rules for KV cache benchmark runs."""

    CLOSED_REQUIREMENTS = {
        'min_runtime': 60,  # seconds
        'allowed_backends': ['disk', 'lmcache'],
    }

    def check_runtime(self) -> Optional[Issue]:
        """Check runtime meets CLOSED requirements."""
        runtime = getattr(self.benchmark_run.args, 'runtime', 0)
        min_runtime = self.CLOSED_REQUIREMENTS['min_runtime']

        if runtime < min_runtime:
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message=f"Runtime below CLOSED requirement",
                parameter='runtime',
                expected=f">= {min_runtime}s",
                actual=f"{runtime}s"
            )
        return None

    def check_cache_backend(self) -> Optional[Issue]:
        """Check cache backend is allowed for CLOSED submission."""
        backend = getattr(self.benchmark_run.args, 'cache_backend', 'memory')
        allowed = self.CLOSED_REQUIREMENTS['allowed_backends']

        if backend not in allowed:
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message=f"Cache backend not allowed for CLOSED submission",
                parameter='cache_backend',
                expected=allowed,
                actual=backend
            )
        return None
```

### Step 2.11: Update __init__.py for Backwards Compatibility

Create `mlpstorage/rules/__init__.py`:

```python
# Maintain backwards compatibility
from mlpstorage.rules.models import (
    HostMemoryInfo, HostCPUInfo, HostInfo, ClusterInformation,
    RunID, ProcessedRun, BenchmarkResult, BenchmarkRun, BenchmarkRunData,
)
from mlpstorage.rules.base import RulesChecker, RuleState
from mlpstorage.rules.issues import Issue

# Run checkers (single-run validation)
from mlpstorage.rules.run_checkers import (
    RunRulesChecker,
    TrainingRunRulesChecker,
    CheckpointingRunRulesChecker,
    VectorDBRunRulesChecker,
    KVCacheRunRulesChecker,
)

# Submission checkers (multi-run validation)
from mlpstorage.rules.submission_checkers import (
    MultiRunRulesChecker,
    TrainingSubmissionRulesChecker,
    CheckpointSubmissionRulesChecker,
)

from mlpstorage.rules.verifier import BenchmarkVerifier
from mlpstorage.rules.utils import calculate_training_data_size, generate_output_location, get_runs_files

__all__ = [
    # Models
    'HostMemoryInfo', 'HostCPUInfo', 'HostInfo', 'ClusterInformation',
    'RunID', 'ProcessedRun', 'BenchmarkResult', 'BenchmarkRun', 'BenchmarkRunData',
    # Base
    'RulesChecker', 'RuleState', 'Issue',
    # Run Checkers
    'RunRulesChecker', 'TrainingRunRulesChecker', 'CheckpointingRunRulesChecker',
    'VectorDBRunRulesChecker', 'KVCacheRunRulesChecker',
    # Submission Checkers
    'MultiRunRulesChecker', 'TrainingSubmissionRulesChecker', 'CheckpointSubmissionRulesChecker',
    # Verifier
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

The `kv_cache_benchmark/` directory contains several files, but only the main benchmark script should be wrapped:

**Main Entrypoint (to be wrapped):**

- `kv-cache.py` - Main benchmark implementation for KV cache benchmarking

**Test/Development Scripts (NOT to be wrapped):**

- `kv-cache_sharegpt_replay.py` - ShareGPT replay variant (for testing only)
- `kv-cache-wrapper.sh` - Shell wrapper (for development/testing)
- `validate.sh` - Validation script (for testing)

The KVCacheBenchmark class should only invoke `kv-cache.py`. The other scripts in the directory are for development, testing, and validation purposes and should not be exposed through the mlpstorage CLI.

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

## Phase 7: Reporting System Refactoring

**Goal:** Improve the reporting system to provide clear validation feedback, better error handling for malformed directory structures, and explicit OPEN vs CLOSED submission messaging.

### Step 7.1: Analyze Current Reporting Issues

The current `reporting.py` has several areas for improvement:

**Current Architecture:**

- `ReportGenerator` class handles all reporting logic
- `accumulate_results()` walks directory tree and validates runs
- `print_results()` outputs to console
- Limited error handling for malformed directories

**Issues Identified:**

1. **Directory Structure Validation**: No explicit validation of expected directory structure
2. **Error Recovery**: Errors during directory walking can halt the entire process
3. **Message Clarity**: OPEN vs CLOSED distinction not always clear to users
4. **Workload Grouping**: Implicit grouping by (model, accelerator) without validation

### Step 7.2: Create Directory Structure Validator

Create `mlpstorage/reporting/directory_validator.py`:

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import os

@dataclass
class DirectoryValidationError:
    """Represents an error in the results directory structure."""
    path: str
    error_type: str  # 'missing', 'malformed', 'unexpected'
    message: str
    suggestion: str  # How to fix the issue

class ResultsDirectoryValidator:
    """
    Validates the structure of a results directory.

    Expected structure:
    results_dir/
        <benchmark_type>/           # training, checkpointing, vector_database
            <model>/                # unet3d, resnet50, llama3-8b, etc.
                <command>/          # run, datagen (for training)
                    <datetime>/     # YYYYMMDD_HHMMSS format
                        *_metadata.json
                        summary.json (for DLIO runs)
    """

    EXPECTED_BENCHMARK_TYPES = ['training', 'checkpointing', 'vector_database', 'kv_cache']

    def __init__(self, results_dir: str, logger=None):
        self.results_dir = Path(results_dir)
        self.logger = logger
        self.errors: List[DirectoryValidationError] = []
        self.warnings: List[str] = []

    def validate(self) -> bool:
        """
        Validate the directory structure.

        Returns:
            True if structure is valid, False otherwise.
        """
        self.errors = []
        self.warnings = []

        if not self.results_dir.exists():
            self.errors.append(DirectoryValidationError(
                path=str(self.results_dir),
                error_type='missing',
                message=f"Results directory does not exist: {self.results_dir}",
                suggestion="Create the directory or specify a different --results-dir path"
            ))
            return False

        if not self.results_dir.is_dir():
            self.errors.append(DirectoryValidationError(
                path=str(self.results_dir),
                error_type='malformed',
                message=f"Results path is not a directory: {self.results_dir}",
                suggestion="Specify a directory path, not a file"
            ))
            return False

        # Check for benchmark type directories
        found_benchmark_dirs = False
        for entry in self.results_dir.iterdir():
            if entry.is_dir():
                if entry.name in self.EXPECTED_BENCHMARK_TYPES:
                    found_benchmark_dirs = True
                    self._validate_benchmark_type_dir(entry)
                else:
                    self.warnings.append(
                        f"Unexpected directory '{entry.name}' in results root. "
                        f"Expected benchmark types: {self.EXPECTED_BENCHMARK_TYPES}"
                    )

        if not found_benchmark_dirs:
            self.errors.append(DirectoryValidationError(
                path=str(self.results_dir),
                error_type='malformed',
                message="No benchmark type directories found",
                suggestion=f"Results should contain directories named: {self.EXPECTED_BENCHMARK_TYPES}"
            ))

        return len(self.errors) == 0

    def _validate_benchmark_type_dir(self, benchmark_dir: Path):
        """Validate a benchmark type directory (e.g., training/)."""
        for model_dir in benchmark_dir.iterdir():
            if model_dir.is_dir():
                self._validate_model_dir(model_dir, benchmark_dir.name)

    def _validate_model_dir(self, model_dir: Path, benchmark_type: str):
        """Validate a model directory."""
        has_valid_runs = False

        for entry in model_dir.iterdir():
            if entry.is_dir():
                # Check if this is a datetime directory or a command directory
                if self._is_datetime_dir(entry.name):
                    self._validate_run_dir(entry)
                    has_valid_runs = True
                elif entry.name in ['run', 'datagen', 'datasize']:
                    # Training has command subdirectories
                    for datetime_dir in entry.iterdir():
                        if datetime_dir.is_dir() and self._is_datetime_dir(datetime_dir.name):
                            self._validate_run_dir(datetime_dir)
                            has_valid_runs = True

        if not has_valid_runs:
            self.warnings.append(
                f"No valid run directories found in {model_dir}"
            )

    def _validate_run_dir(self, run_dir: Path):
        """Validate a single run directory."""
        files = list(run_dir.iterdir())
        file_names = [f.name for f in files if f.is_file()]

        # Check for metadata file
        metadata_files = [f for f in file_names if f.endswith('_metadata.json')]
        if not metadata_files:
            self.errors.append(DirectoryValidationError(
                path=str(run_dir),
                error_type='malformed',
                message=f"Missing metadata file in {run_dir.name}",
                suggestion="Run directory should contain a *_metadata.json file"
            ))

        # Check for summary.json (required for completed DLIO runs)
        if 'summary.json' not in file_names:
            self.warnings.append(
                f"Missing summary.json in {run_dir} - run may be incomplete"
            )

    def _is_datetime_dir(self, name: str) -> bool:
        """Check if directory name matches expected datetime format."""
        # Expected format: YYYYMMDD_HHMMSS or similar
        import re
        return bool(re.match(r'^\d{8}_\d{6}', name))

    def get_error_report(self) -> str:
        """Generate a human-readable error report."""
        lines = []

        if self.errors:
            lines.append("=== Directory Structure Errors ===\n")
            for error in self.errors:
                lines.append(f"ERROR: {error.message}")
                lines.append(f"  Path: {error.path}")
                lines.append(f"  Fix: {error.suggestion}")
                lines.append("")

        if self.warnings:
            lines.append("=== Warnings ===\n")
            for warning in self.warnings:
                lines.append(f"WARNING: {warning}")

        return "\n".join(lines)
```

### Step 7.3: Refactor ReportGenerator with Better Error Handling

Update `mlpstorage/reporting.py`:

```python
from mlpstorage.reporting.directory_validator import ResultsDirectoryValidator, DirectoryValidationError

class ReportGenerator:
    """Generate validation reports for benchmark results."""

    def __init__(self, results_dir, args=None, logger=None):
        self.args = args
        self.logger = logger or setup_logging(name="mlpstorage_reporter")
        self.results_dir = results_dir

        # Validate directory structure first
        self.validator = ResultsDirectoryValidator(results_dir, logger=self.logger)
        if not self.validator.validate():
            self._report_directory_errors()
            raise ValueError(f"Invalid results directory structure: {results_dir}")

        self.run_results = dict()
        self.workload_results = dict()

    def _report_directory_errors(self):
        """Report directory structure errors to user."""
        self.logger.error("Results directory structure validation failed:")
        self.logger.error(self.validator.get_error_report())

        # Provide specific guidance
        self.logger.error("\nExpected directory structure:")
        self.logger.error("  results_dir/")
        self.logger.error("    training/")
        self.logger.error("      <model>/")
        self.logger.error("        run/")
        self.logger.error("          <datetime>/")
        self.logger.error("            *_metadata.json")
        self.logger.error("            summary.json")

    def accumulate_results(self):
        """Accumulate results with improved error handling."""
        try:
            benchmark_runs = get_runs_files(self.results_dir, logger=self.logger)
        except Exception as e:
            self.logger.error(f"Failed to scan results directory: {e}")
            raise

        if not benchmark_runs:
            self.logger.warning(
                f"No valid benchmark runs found in {self.results_dir}. "
                "Ensure runs have completed and contain summary.json files."
            )
            return

        self.logger.info(f"Found {len(benchmark_runs)} benchmark runs")

        # Process each run with error isolation
        for benchmark_run in benchmark_runs:
            try:
                self._process_single_run(benchmark_run)
            except Exception as e:
                self.logger.error(
                    f"Failed to process run {benchmark_run.run_id}: {e}. Skipping."
                )
                continue

        # Process workload groups
        self._process_workload_groups(benchmark_runs)

    def _process_single_run(self, benchmark_run):
        """Process a single benchmark run with clear error handling."""
        verifier = BenchmarkVerifier(benchmark_run, logger=self.logger)
        category = verifier.verify()
        issues = verifier.issues

        result = Result(
            multi=False,
            benchmark_run=benchmark_run,
            benchmark_type=benchmark_run.benchmark_type,
            benchmark_command=benchmark_run.command,
            benchmark_model=benchmark_run.model,
            issues=issues,
            category=category,
            metrics=benchmark_run.metrics
        )
        self.run_results[benchmark_run.run_id] = result
```

### Step 7.4: Add Clear OPEN vs CLOSED Messaging

Create `mlpstorage/reporting/formatters.py`:

```python
from typing import List
from mlpstorage.rules.issues import Issue
from mlpstorage.config import PARAM_VALIDATION

class ValidationMessageFormatter:
    """Format validation results with clear OPEN vs CLOSED messaging."""

    @staticmethod
    def format_category_summary(category: PARAM_VALIDATION, issues: List[Issue]) -> str:
        """Generate a clear summary of why a run is in a particular category."""
        if category == PARAM_VALIDATION.CLOSED:
            return (
                "This run qualifies for CLOSED submission.\n"
                "All parameters meet the strict requirements for closed division."
            )

        elif category == PARAM_VALIDATION.OPEN:
            open_issues = [i for i in issues if i.validation == PARAM_VALIDATION.OPEN]
            reasons = []
            for issue in open_issues:
                reasons.append(f"  - {issue.parameter}: {issue.message}")

            return (
                "This run qualifies for OPEN submission only.\n"
                "The following parameters do not meet CLOSED requirements:\n"
                + "\n".join(reasons) +
                "\n\nTo qualify for CLOSED submission, modify these parameters."
            )

        else:  # INVALID
            invalid_issues = [i for i in issues if i.validation == PARAM_VALIDATION.INVALID]
            reasons = []
            for issue in invalid_issues:
                reasons.append(f"  - {issue.parameter}: {issue.message}")
                if issue.expected and issue.actual:
                    reasons.append(f"    Expected: {issue.expected}")
                    reasons.append(f"    Actual: {issue.actual}")

            return (
                "This run is INVALID and cannot be submitted.\n"
                "The following issues must be resolved:\n"
                + "\n".join(reasons) +
                "\n\nFix these issues and re-run the benchmark."
            )

    @staticmethod
    def format_closed_requirements_checklist(benchmark_type: str) -> str:
        """Generate a checklist of CLOSED submission requirements."""
        if benchmark_type == "training":
            return """
CLOSED Submission Requirements for Training:
  [ ] Dataset size >= 5x total cluster memory
  [ ] At least 500 steps per epoch
  [ ] 5 complete runs required
  [ ] Only allowed parameter overrides used
  [ ] No modifications to core workload parameters

Allowed Parameter Overrides (CLOSED):
  - dataset.num_files_train
  - dataset.num_subfolders_train
  - dataset.data_folder
  - reader.read_threads
  - reader.computation_threads
  - reader.transfer_size
  - storage.storage_type
  - storage.storage_root
"""
        elif benchmark_type == "checkpointing":
            return """
CLOSED Submission Requirements for Checkpointing:
  [ ] 10 checkpoint write operations
  [ ] 10 checkpoint read operations
  [ ] Valid LLM model (llama3-8b, llama3-70b, llama3-405b)
  [ ] Only allowed parameter overrides used
"""
        return ""
```

### Step 7.5: Update Print Methods with Better Formatting

```python
def print_results(self):
    """Print results with clear OPEN/CLOSED distinction."""
    formatter = ValidationMessageFormatter()

    print("\n" + "=" * 60)
    print("BENCHMARK VALIDATION REPORT")
    print("=" * 60)

    # Summary counts
    closed_count = sum(1 for r in self.run_results.values()
                       if r.category == PARAM_VALIDATION.CLOSED)
    open_count = sum(1 for r in self.run_results.values()
                     if r.category == PARAM_VALIDATION.OPEN)
    invalid_count = sum(1 for r in self.run_results.values()
                        if r.category == PARAM_VALIDATION.INVALID)

    print(f"\nSummary: {len(self.run_results)} runs analyzed")
    print(f"  CLOSED: {closed_count} runs")
    print(f"  OPEN:   {open_count} runs")
    print(f"  INVALID: {invalid_count} runs")

    # Print INVALID runs first (most important)
    if invalid_count > 0:
        print("\n" + "-" * 60)
        print("INVALID RUNS - These runs cannot be submitted")
        print("-" * 60)
        for result in self.run_results.values():
            if result.category == PARAM_VALIDATION.INVALID:
                self._print_run_details(result, formatter)

    # Print OPEN runs
    if open_count > 0:
        print("\n" + "-" * 60)
        print("OPEN RUNS - These runs qualify for OPEN division only")
        print("-" * 60)
        for result in self.run_results.values():
            if result.category == PARAM_VALIDATION.OPEN:
                self._print_run_details(result, formatter)

    # Print CLOSED runs
    if closed_count > 0:
        print("\n" + "-" * 60)
        print("CLOSED RUNS - These runs qualify for CLOSED division")
        print("-" * 60)
        for result in self.run_results.values():
            if result.category == PARAM_VALIDATION.CLOSED:
                self._print_run_details(result, formatter)
```

---

## Phase 8: Error Handling and User Messaging

**Goal:** Improve error handling throughout the project to provide clear, actionable messages when users make mistakes.

### Step 8.1: Create Error Handling Framework

Create `mlpstorage/errors.py`:

```python
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class MLPSError:
    """Base error class with user-friendly messaging."""
    code: str           # Machine-readable error code
    message: str        # User-facing error message
    details: str        # Technical details
    suggestion: str     # How to fix the issue
    related_docs: Optional[str] = None  # Link to documentation

class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, parameter: str = None,
                 expected: str = None, actual: str = None,
                 suggestion: str = None):
        self.parameter = parameter
        self.expected = expected
        self.actual = actual
        self.suggestion = suggestion

        full_message = f"Configuration Error: {message}"
        if parameter:
            full_message += f"\n  Parameter: {parameter}"
        if expected:
            full_message += f"\n  Expected: {expected}"
        if actual:
            full_message += f"\n  Actual: {actual}"
        if suggestion:
            full_message += f"\n  Suggestion: {suggestion}"

        super().__init__(full_message)

class BenchmarkExecutionError(Exception):
    """Raised when benchmark execution fails."""

    def __init__(self, message: str, command: str = None,
                 exit_code: int = None, stderr: str = None,
                 suggestion: str = None):
        self.command = command
        self.exit_code = exit_code
        self.stderr = stderr
        self.suggestion = suggestion

        full_message = f"Benchmark Execution Failed: {message}"
        if command:
            full_message += f"\n  Command: {command}"
        if exit_code is not None:
            full_message += f"\n  Exit Code: {exit_code}"
        if stderr:
            full_message += f"\n  Error Output: {stderr[:500]}..."
        if suggestion:
            full_message += f"\n  Suggestion: {suggestion}"

        super().__init__(full_message)

class ValidationError(Exception):
    """Raised when validation fails."""

    def __init__(self, message: str, issues: List = None,
                 category: str = None, suggestion: str = None):
        self.issues = issues or []
        self.category = category
        self.suggestion = suggestion

        full_message = f"Validation Failed: {message}"
        if category:
            full_message += f"\n  Category: {category}"
        if issues:
            full_message += "\n  Issues:"
            for issue in issues[:5]:  # Limit to first 5
                full_message += f"\n    - {issue}"
            if len(issues) > 5:
                full_message += f"\n    ... and {len(issues) - 5} more"
        if suggestion:
            full_message += f"\n  Suggestion: {suggestion}"

        super().__init__(full_message)

class FileSystemError(Exception):
    """Raised when file system operations fail."""

    def __init__(self, message: str, path: str = None,
                 operation: str = None, suggestion: str = None):
        self.path = path
        self.operation = operation
        self.suggestion = suggestion

        full_message = f"File System Error: {message}"
        if path:
            full_message += f"\n  Path: {path}"
        if operation:
            full_message += f"\n  Operation: {operation}"
        if suggestion:
            full_message += f"\n  Suggestion: {suggestion}"

        super().__init__(full_message)
```

### Step 8.2: Create Error Message Templates

Create `mlpstorage/error_messages.py`:

```python
"""
Centralized error message templates for consistent user messaging.
"""

ERROR_MESSAGES = {
    # Configuration Errors
    'CONFIG_MISSING_REQUIRED': (
        "Required parameter '{param}' is missing.\n"
        "Please provide this parameter via command line or configuration file.\n"
        "Example: mlpstorage {benchmark} run --{param} <value>"
    ),

    'CONFIG_INVALID_VALUE': (
        "Invalid value for parameter '{param}': {actual}\n"
        "Expected: {expected}\n"
        "Please correct this value and try again."
    ),

    'CONFIG_FILE_NOT_FOUND': (
        "Configuration file not found: {path}\n"
        "Please ensure the file exists and the path is correct.\n"
        "You can create a default config with: mlpstorage init --config"
    ),

    # Benchmark Execution Errors
    'BENCHMARK_COMMAND_FAILED': (
        "Benchmark command failed with exit code {exit_code}.\n"
        "Command: {command}\n"
        "This may indicate:\n"
        "  - Missing dependencies (check that DLIO/benchmark tool is installed)\n"
        "  - Insufficient permissions\n"
        "  - Invalid benchmark parameters\n"
        "Check the error output above for specific details."
    ),

    'BENCHMARK_TIMEOUT': (
        "Benchmark timed out after {timeout} seconds.\n"
        "The benchmark may still be running in the background.\n"
        "Consider:\n"
        "  - Increasing the timeout with --timeout\n"
        "  - Reducing the workload size\n"
        "  - Checking system resources"
    ),

    # Validation Errors
    'VALIDATION_NOT_CLOSED': (
        "This run does not qualify for CLOSED submission.\n"
        "Category: {category}\n"
        "To submit in the CLOSED division, the following must be addressed:\n"
        "{issues}\n"
        "See the MLPerf Storage rules for CLOSED submission requirements."
    ),

    'VALIDATION_INVALID': (
        "This run is INVALID and cannot be submitted.\n"
        "The following critical issues were found:\n"
        "{issues}\n"
        "These issues must be resolved before submission."
    ),

    # File System Errors
    'RESULTS_DIR_NOT_FOUND': (
        "Results directory not found: {path}\n"
        "Please ensure:\n"
        "  - The directory exists\n"
        "  - You have read permissions\n"
        "  - The path is correct"
    ),

    'RESULTS_DIR_EMPTY': (
        "No benchmark results found in: {path}\n"
        "This directory should contain completed benchmark runs.\n"
        "Expected structure:\n"
        "  {path}/\n"
        "    training/\n"
        "      <model>/\n"
        "        run/\n"
        "          <datetime>/\n"
        "            summary.json\n"
        "            *_metadata.json"
    ),

    'METADATA_FILE_MISSING': (
        "Metadata file not found in: {path}\n"
        "Each run directory should contain a *_metadata.json file.\n"
        "This file is created automatically when running benchmarks.\n"
        "If missing, the run may have failed or been interrupted."
    ),

    # MPI/Cluster Errors
    'MPI_NOT_AVAILABLE': (
        "MPI is not available on this system.\n"
        "MPI is required for distributed benchmarks.\n"
        "Install MPI:\n"
        "  - Ubuntu/Debian: apt-get install openmpi-bin libopenmpi-dev\n"
        "  - RHEL/CentOS: yum install openmpi openmpi-devel\n"
        "Or use --skip-cluster-info to skip cluster information collection."
    ),

    'HOST_UNREACHABLE': (
        "Cannot reach host: {host}\n"
        "Please verify:\n"
        "  - The hostname is correct\n"
        "  - The host is online and reachable\n"
        "  - SSH access is configured (passwordless SSH recommended)"
    ),
}

def format_error(error_key: str, **kwargs) -> str:
    """Format an error message with the given parameters."""
    template = ERROR_MESSAGES.get(error_key, f"Unknown error: {error_key}")
    try:
        return template.format(**kwargs)
    except KeyError as e:
        return f"{template}\n(Missing format parameter: {e})"
```

### Step 8.3: Add Error Handling to CLI

Update `mlpstorage/cli.py`:

```python
from mlpstorage.errors import ConfigurationError, BenchmarkExecutionError, ValidationError
from mlpstorage.error_messages import format_error

def main():
    try:
        args = parse_arguments()
        validate_args(args)
        run_benchmark(args)
    except ConfigurationError as e:
        logger.error(str(e))
        sys.exit(EXIT_CODE.CONFIG_ERROR)
    except BenchmarkExecutionError as e:
        logger.error(str(e))
        sys.exit(EXIT_CODE.BENCHMARK_FAILED)
    except ValidationError as e:
        logger.error(str(e))
        sys.exit(EXIT_CODE.VALIDATION_FAILED)
    except FileNotFoundError as e:
        logger.error(format_error('RESULTS_DIR_NOT_FOUND', path=str(e)))
        sys.exit(EXIT_CODE.FILE_NOT_FOUND)
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(EXIT_CODE.USER_INTERRUPTED)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug("Stack trace:", exc_info=True)
        sys.exit(EXIT_CODE.UNKNOWN_ERROR)
```

### Step 8.4: Add User-Friendly Validation Messages

Update benchmark validation to provide clear guidance:

```python
def validate_for_closed_submission(benchmark, logger):
    """
    Validate benchmark for CLOSED submission with clear messaging.
    """
    verifier = BenchmarkVerifier(benchmark, logger=logger)
    result = verifier.verify()

    if result == PARAM_VALIDATION.CLOSED:
        logger.success("Benchmark meets CLOSED submission requirements")
        return True

    elif result == PARAM_VALIDATION.OPEN:
        logger.warning("Benchmark qualifies for OPEN submission only")
        logger.warning("")
        logger.warning("To qualify for CLOSED submission, address these items:")
        for issue in verifier.issues:
            if issue.validation == PARAM_VALIDATION.OPEN:
                logger.warning(f"  - {issue.parameter}: {issue.message}")
                if issue.expected:
                    logger.warning(f"      Required: {issue.expected}")
                if issue.actual:
                    logger.warning(f"      Current:  {issue.actual}")
        logger.warning("")
        logger.warning("See MLPerf Storage rules for CLOSED requirements.")
        return False

    else:  # INVALID
        logger.error("Benchmark is INVALID - cannot be submitted")
        logger.error("")
        logger.error("The following errors must be fixed:")
        for issue in verifier.issues:
            if issue.validation == PARAM_VALIDATION.INVALID:
                logger.error(f"  - {issue.parameter}: {issue.message}")
                if issue.expected:
                    logger.error(f"      Required: {issue.expected}")
                if issue.actual:
                    logger.error(f"      Actual:   {issue.actual}")
        logger.error("")
        logger.error("Fix these issues and re-run the benchmark.")
        return False
```

### Step 8.5: Add Pre-Run Validation with Clear Messages

```python
def validate_pre_run(args, logger):
    """
    Validate configuration before running benchmark.
    Provides clear error messages for common mistakes.
    """
    errors = []

    # Check required parameters
    if not args.model:
        errors.append(format_error('CONFIG_MISSING_REQUIRED', param='model'))

    if not args.results_dir:
        errors.append(format_error('CONFIG_MISSING_REQUIRED', param='results-dir'))

    # Check hosts are reachable (if provided)
    if args.hosts:
        for host in args.hosts:
            if not is_host_reachable(host):
                errors.append(format_error('HOST_UNREACHABLE', host=host))

    # Check data directory exists (for run command)
    if args.command == 'run' and hasattr(args, 'data_dir'):
        if not os.path.exists(args.data_dir):
            errors.append(format_error('RESULTS_DIR_NOT_FOUND', path=args.data_dir))

    if errors:
        logger.error("Pre-run validation failed:")
        for error in errors:
            logger.error(error)
        raise ConfigurationError(
            "Configuration validation failed",
            suggestion="Fix the above errors and try again"
        )

    logger.info("Pre-run validation passed")
```

---

## Implementation Priority

| Phase | Priority | Effort | Impact |
|-------|----------|--------|--------|
| Phase 1: Interfaces | High | Medium | Foundation for all other phases |
| Phase 2: Rules Modularization | High | High | Better organization, testability |
| Phase 4: Test Infrastructure | High | Medium | Enables safe refactoring |
| Phase 6: KV Cache Integration | High | Medium | Immediate user value |
| Phase 7: Reporting Refactoring | High | Medium | Better validation feedback |
| Phase 8: Error Handling | High | Medium | Improved user experience |
| Phase 3: CLI Refactoring | Medium | High | Extensibility |
| Phase 5: Documentation | Medium | Low | Long-term maintainability |

## Recommended Order

1. **Phase 1** (Interfaces) - Create foundation
2. **Phase 4** (Tests) - Ensure safety net for refactoring
3. **Phase 2** (Rules) - Modularize largest file, separate RunRulesChecker from MultiRunRulesChecker
4. **Phase 8** (Error Handling) - Improve user-facing error messages
5. **Phase 7** (Reporting) - Improve validation reporting and directory handling
6. **Phase 6** (KV Cache) - Add new benchmark using new patterns
7. **Phase 3** (CLI) - Improve extensibility
8. **Phase 5** (Docs) - Document everything

---

## Success Metrics

After completing all phases:

- [ ] All benchmarks implement `BenchmarkInterface`
- [ ] `rules.py` split into <300 LOC modules with clear separation between RunRulesChecker and MultiRunRulesChecker hierarchies
- [ ] New benchmarks can be added without modifying `cli.py`
- [ ] Test coverage >80%
- [ ] All public functions have docstrings and type annotations
- [ ] KV Cache benchmark integrated and working
- [ ] Reporting system provides clear feedback for OPEN vs CLOSED validation failures
- [ ] All error messages clearly indicate what failed and how to fix it
- [ ] Malformed directory structures are handled gracefully with actionable error messages

---

## Appendix: File Changes Summary

### New Files Created

- `mlpstorage/interfaces/__init__.py`
- `mlpstorage/interfaces/benchmark.py`
- `mlpstorage/interfaces/validator.py`
- `mlpstorage/interfaces/collector.py`
- `mlpstorage/registry.py`
- `mlpstorage/rules/__init__.py` (package)
- `mlpstorage/rules/base.py` - RulesChecker, RunRulesChecker, MultiRunRulesChecker base classes
- `mlpstorage/rules/issues.py`
- `mlpstorage/rules/models.py`
- `mlpstorage/rules/run_checkers/__init__.py` - Package for single-run rule checkers
- `mlpstorage/rules/run_checkers/training.py` - TrainingRunRulesChecker
- `mlpstorage/rules/run_checkers/checkpointing.py` - CheckpointingRunRulesChecker
- `mlpstorage/rules/run_checkers/vectordb.py` - VectorDBRunRulesChecker
- `mlpstorage/rules/run_checkers/kvcache.py` - KVCacheRunRulesChecker
- `mlpstorage/rules/submission_checkers/__init__.py` - Package for multi-run submission checkers
- `mlpstorage/rules/submission_checkers/training.py` - TrainingSubmissionRulesChecker
- `mlpstorage/rules/submission_checkers/checkpointing.py` - CheckpointSubmissionRulesChecker
- `mlpstorage/rules/verifier.py`
- `mlpstorage/rules/utils.py`
- `mlpstorage/cli/__init__.py` (package)
- `mlpstorage/cli/training_args.py`
- `mlpstorage/cli/checkpointing_args.py`
- `mlpstorage/cli/vectordb_args.py`
- `mlpstorage/cli/kvcache_args.py`
- `mlpstorage/benchmarks/kvcache.py`
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
- `mlpstorage/reporting.py` - Refactored with better error handling and directory validation
- `mlpstorage/rules.py` - Converted to package (deprecated)

### Files Deprecated

- `mlpstorage/rules.py` - Replaced by `mlpstorage/rules/` package
