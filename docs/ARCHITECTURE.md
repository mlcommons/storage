# MLPerf Storage Architecture

This document describes the architecture of the MLPerf Storage benchmark suite,
including component interactions, data flow, and key design decisions.

## Overview

MLPerf Storage (`mlpstorage`) is a benchmarking framework for evaluating storage
system performance in AI/ML workloads. The framework supports multiple benchmark
types including training data loading, checkpointing, and vector database operations.

## Directory Structure

```
mlpstorage/
├── __init__.py           # Package initialization and version
├── main.py               # Main entry point
├── cli_parser.py         # Argument parsing and validation
├── config.py             # Configuration constants and enums
├── registry.py           # Benchmark registration system
│
├── benchmarks/           # Benchmark implementations
│   ├── __init__.py       # Registers benchmarks with registry
│   ├── base.py           # Abstract Benchmark base class
│   ├── dlio.py           # Training and Checkpointing benchmarks
│   ├── vectordbbench.py  # VectorDB benchmark
│   └── kvcache.py        # KV Cache benchmark for LLM inference
│
├── interfaces/           # Abstract interfaces (Phase 1)
│   ├── __init__.py
│   ├── benchmark.py      # BenchmarkInterface protocol
│   ├── validator.py      # ValidatorInterface protocol
│   └── collector.py      # ClusterCollectorInterface protocol
│
├── rules/                # Validation rules engine (Phase 2)
│   ├── __init__.py       # Public API exports
│   ├── base.py           # RulesChecker ABC
│   ├── models.py         # Data classes (BenchmarkRun, etc.)
│   ├── verifier.py       # BenchmarkVerifier orchestrator
│   ├── run_checkers/     # Single-run validation rules
│   └── submission_checkers/  # Multi-run submission rules
│
├── cli/                  # Modular CLI builders (Phase 3)
│   ├── __init__.py
│   ├── common_args.py    # Shared arguments and help text
│   ├── training_args.py  # Training benchmark arguments
│   ├── checkpointing_args.py
│   ├── vectordb_args.py
│   ├── kvcache_args.py   # KV Cache benchmark arguments
│   └── utility_args.py
│
├── utils.py              # Utility functions
├── mlps_logging.py       # Custom logging setup
├── reporting.py          # Result reporting
├── history.py            # Command history tracking
└── cluster_collector.py  # MPI-based cluster info collection
```

## Core Components

### 1. Entry Point and CLI

```
User Command
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   main.py   │────►│ cli_parser   │────►│ BenchmarkRegistry│
└─────────────┘     └──────────────┘     └─────────────────┘
                           │                      │
                           ▼                      ▼
                    ┌─────────────┐        ┌─────────────┐
                    │ validate_args│       │ get_benchmark│
                    └─────────────┘        └─────────────┘
```

**Flow:**
1. `main.py` calls `parse_arguments()` from `cli_parser.py`
2. CLI parser builds argument parser using modular builders from `cli/` package
3. Arguments are validated using `validate_args()`
4. `BenchmarkRegistry` provides the appropriate benchmark class

### 2. Benchmark Execution

```
┌─────────────────────────────────────────────────────────┐
│                    Benchmark.run()                       │
├─────────────────────────────────────────────────────────┤
│  1. Collect cluster information via MPI                 │
│  2. Verify parameters (closed/open validation)          │
│  3. Generate benchmark command                          │
│  4. Execute command via CommandExecutor                 │
│  5. Write metadata and results                          │
│  6. Return exit code                                    │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                  DLIO Benchmark                          │
│  (dlio_benchmark executable with Hydra configuration)    │
└─────────────────────────────────────────────────────────┘
```

### 3. Validation System

The validation system determines if a benchmark run qualifies for CLOSED or OPEN
submission categories.

```
┌──────────────────┐
│ BenchmarkVerifier │  (Orchestrator)
└────────┬─────────┘
         │
         ├──────────────────────────────────────────┐
         │                                          │
         ▼                                          ▼
┌─────────────────────┐               ┌─────────────────────────┐
│  RunRulesChecker    │               │ SubmissionRulesChecker  │
│  (Single-run rules) │               │ (Multi-run rules)       │
└─────────────────────┘               └─────────────────────────┘
         │                                          │
         ▼                                          ▼
┌─────────────────────┐               ┌─────────────────────────┐
│ • TrainingRunChecker│               │ • TrainingSubmission    │
│ • CheckpointingRun  │               │ • CheckpointSubmission  │
│ • KVCacheRunChecker │               │                         │
└─────────────────────┘               └─────────────────────────┘
```

**Validation States:**
- `CLOSED`: Meets all requirements for closed submission
- `OPEN`: Meets open requirements (relaxed constraints)
- `INVALID`: Does not meet any submission requirements

### 4. Cluster Information Collection

```
┌─────────────────────────────────────────────────────────┐
│              Cluster Collection Flow                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐    MPI     ┌──────────┐    MPI     ┌────┐ │
│  │  Host 1  │◄──────────►│  Host 2  │◄──────────►│... │ │
│  └──────────┘            └──────────┘            └────┘ │
│       │                       │                    │     │
│       └───────────────────────┼────────────────────┘     │
│                               ▼                          │
│                    ┌──────────────────┐                  │
│                    │ClusterInformation│                  │
│                    └──────────────────┘                  │
│                               │                          │
│                               ▼                          │
│                    ┌──────────────────┐                  │
│                    │    HostInfo[]    │                  │
│                    │  - hostname      │                  │
│                    │  - memory        │                  │
│                    │  - cpu           │                  │
│                    └──────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

## Key Design Patterns

### Interface Segregation

The `interfaces/` package defines abstract protocols that allow dependency
injection for testing:

```python
class BenchmarkInterface(ABC):
    """Interface all benchmarks must implement."""

    @property
    @abstractmethod
    def config(self) -> BenchmarkConfig: ...

    @abstractmethod
    def validate_args(self, args) -> List[str]: ...

    @abstractmethod
    def generate_command(self, command: str) -> str: ...
```

### Registry Pattern

`BenchmarkRegistry` enables dynamic benchmark registration:

```python
# In benchmarks/__init__.py
BenchmarkRegistry.register(
    name='training',
    benchmark_class=TrainingBenchmark,
    cli_builder=add_training_arguments,
    description='Run the MLPerf Storage training benchmark',
)

# In cli_parser.py
benchmark_class = BenchmarkRegistry.get_benchmark_class('training')
```

### Modular Rules Engine

Rules are organized by scope (run vs submission) and benchmark type:

```
rules/
├── run_checkers/           # Check individual runs
│   ├── training.py         # Training-specific run rules
│   ├── checkpointing.py    # Checkpointing-specific run rules
│   └── kvcache.py          # KV Cache run rules (preview)
└── submission_checkers/    # Check submission completeness
    ├── training.py         # Training submission rules
    └── checkpointing.py    # Checkpointing submission rules
```

## Data Flow

### Benchmark Run Flow

```
1. Parse CLI arguments
   │
2. Load YAML configuration
   │
3. Collect cluster info (MPI)
   │
4. Validate parameters
   │
   ├─► CLOSED: Proceed
   ├─► OPEN:   Proceed with warning
   └─► INVALID: Exit or warn (--allow-invalid-params)
   │
5. Generate DLIO command
   │
6. Execute via CommandExecutor
   │
7. Write metadata and results
   │
8. Return exit code
```

### Result File Structure

```
results/
└── training/
    └── unet3d/
        └── run/
            └── 20250115_143022/
                ├── summary.json           # DLIO output
                ├── training_*_metadata.json  # Benchmark metadata
                ├── training_cluster_info.json
                └── .hydra/
                    ├── config.yaml
                    └── overrides.yaml
```

## Configuration System

### Configuration Hierarchy

```
1. Default YAML configs (workloads/*.yaml)
   │
   ▼
2. CLI arguments
   │
   ▼
3. --params overrides (highest priority)
```

### Key Configuration Files

- `configs/workloads/`: Workload-specific configurations
- `configs/accelerators/`: Accelerator performance profiles
- `mlpstorage/config.py`: Runtime constants and enums

## Testing Architecture

```
tests/
├── conftest.py           # Shared pytest fixtures
├── fixtures/             # Test mock classes (Phase 4)
│   ├── mock_logger.py
│   ├── mock_executor.py
│   ├── mock_collector.py
│   └── sample_data.py
├── unit/                 # Unit tests
│   ├── test_cli.py
│   ├── test_rules_*.py
│   └── test_*.py
└── integration/          # Integration tests
    ├── test_full_submission.py
    └── test_benchmark_flow.py
```

### Mock Infrastructure

- `MockCommandExecutor`: Test without subprocess execution
- `MockClusterCollector`: Test without MPI
- `MockLogger`: Capture log messages for verification

## Extension Points

### Adding New Benchmarks

See [ADDING_BENCHMARKS.md](ADDING_BENCHMARKS.md) for detailed instructions.

### Adding Validation Rules

1. Create checker class in `rules/run_checkers/` or `rules/submission_checkers/`
2. Inherit from `RulesChecker` base class
3. Implement `check()` method
4. Register in verifier

### Adding CLI Arguments

1. Create argument builder function in `cli/` package
2. Export from `cli/__init__.py`
3. Call from `cli_parser.py`

## Dependencies

### Core Dependencies

- `mpi4py`: MPI communication for cluster collection
- `pyarrow`: Data handling
- `pyyaml`: Configuration file parsing
- `hydra-core`: DLIO configuration management

### Runtime Dependencies

- `dlio_benchmark`: Actual benchmark execution
- OpenMPI or equivalent MPI implementation

## Performance Considerations

- Cluster collection is done once per benchmark run
- Results are written incrementally
- MPI binding optimized for I/O workloads (not compute)
