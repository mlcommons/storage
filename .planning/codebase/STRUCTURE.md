# Codebase Structure

**Analysis Date:** 2026-01-23

## Directory Layout

```
mlperf-storage/
├── mlpstorage/              # Main Python package
│   ├── benchmarks/          # Benchmark implementations
│   ├── cli/                 # CLI argument builders
│   ├── interfaces/          # Abstract interfaces
│   ├── reporting/           # Report formatters and validators
│   ├── rules/               # Validation rules engine
│   │   ├── run_checkers/    # Single-run validation
│   │   └── submission_checkers/  # Multi-run submission validation
│   ├── main.py              # Entry point
│   ├── config.py            # Constants and configuration
│   ├── utils.py             # Utility functions
│   └── ...                  # Other core modules
├── configs/                 # YAML configuration files
│   ├── dlio/                # DLIO benchmark configs
│   │   └── workload/        # Workload-specific configs
│   └── vectordbbench/       # VectorDB configs
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── fixtures/            # Test fixtures and mocks
├── kv_cache_benchmark/      # Standalone KV cache benchmark script
├── docs/                    # Documentation
├── ansible/                 # Deployment automation
└── pyproject.toml           # Package configuration
```

## Directory Purposes

**mlpstorage/**
- Purpose: Main Python package containing all benchmark code
- Contains: Python modules, subpackages for benchmarks/cli/rules/interfaces
- Key files: `main.py`, `config.py`, `utils.py`, `cli_parser.py`

**mlpstorage/benchmarks/**
- Purpose: Benchmark implementations that can be executed
- Contains: Abstract base class (`base.py`), concrete implementations
- Key files:
  - `base.py`: `Benchmark` abstract base class with common functionality
  - `dlio.py`: `DLIOBenchmark`, `TrainingBenchmark`, `CheckpointingBenchmark`
  - `vectordbbench.py`: `VectorDBBenchmark`
  - `kvcache.py`: `KVCacheBenchmark`
  - `__init__.py`: Registry registration at import time

**mlpstorage/cli/**
- Purpose: Modular CLI argument builders for each benchmark type
- Contains: Argument builder functions, help messages, descriptions
- Key files:
  - `common_args.py`: Shared args (MPI, host, DLIO, universal)
  - `training_args.py`: `add_training_arguments()`
  - `checkpointing_args.py`: `add_checkpointing_arguments()`
  - `vectordb_args.py`: `add_vectordb_arguments()`
  - `kvcache_args.py`: `add_kvcache_arguments()`
  - `utility_args.py`: Reports and history args

**mlpstorage/interfaces/**
- Purpose: Abstract interfaces defining contracts for implementations
- Contains: ABC classes for benchmarks, validators, collectors
- Key files:
  - `benchmark.py`: `BenchmarkInterface`, `BenchmarkConfig`, `BenchmarkCommand`
  - `validator.py`: `ValidatorInterface`, `ValidationResult`
  - `collector.py`: `ClusterCollectorInterface`, `CollectionResult`

**mlpstorage/rules/**
- Purpose: Validation rules for OPEN/CLOSED submission compliance
- Contains: Base classes, data models, checkers, verifier
- Key files:
  - `base.py`: `RulesChecker` ABC, `RuleState` enum
  - `models.py`: Data classes (`BenchmarkRun`, `ClusterInformation`, `HostInfo`)
  - `issues.py`: `Issue` dataclass for validation findings
  - `verifier.py`: `BenchmarkVerifier` orchestrator
  - `utils.py`: Utility functions for rules calculations

**mlpstorage/rules/run_checkers/**
- Purpose: Single-run parameter validation
- Contains: Checkers for each benchmark type
- Key files:
  - `base.py`: `RunRulesChecker` base class
  - `training.py`: `TrainingRunRulesChecker`
  - `checkpointing.py`: `CheckpointingRunRulesChecker`
  - `kvcache.py`: `KVCacheRunRulesChecker`

**mlpstorage/rules/submission_checkers/**
- Purpose: Multi-run submission validation (5-run requirements, etc.)
- Contains: Submission-level checkers
- Key files:
  - `base.py`: `MultiRunRulesChecker` base class
  - `training.py`: `TrainingSubmissionRulesChecker`
  - `checkpointing.py`: `CheckpointSubmissionRulesChecker`

**mlpstorage/reporting/**
- Purpose: Report generation utilities and formatters
- Contains: Directory validators, output formatters
- Key files:
  - `directory_validator.py`: `ResultsDirectoryValidator`
  - `formatters.py`: `ValidationMessageFormatter`, `ReportSummaryFormatter`

**configs/dlio/workload/**
- Purpose: YAML configuration files for DLIO benchmark workloads
- Contains: Model+accelerator specific configs
- Key files:
  - `unet3d_h100.yaml`, `unet3d_a100.yaml`, `unet3d_datagen.yaml`
  - `resnet50_h100.yaml`, `resnet50_a100.yaml`, `resnet50_datagen.yaml`
  - `cosmoflow_h100.yaml`, `cosmoflow_a100.yaml`, `cosmoflow_datagen.yaml`
  - `llama3_8b.yaml`, `llama3_70b.yaml`, `llama3_405b.yaml`, `llama3_1t.yaml`

**tests/**
- Purpose: Test suite for the package
- Contains: Unit tests, integration tests, fixtures
- Key files:
  - `conftest.py`: Pytest configuration and shared fixtures
  - `unit/test_*.py`: Unit test modules
  - `fixtures/mock_*.py`: Mock objects for testing

## Key File Locations

**Entry Points:**
- `mlpstorage/main.py`: CLI entry point (`main()` function)
- `pyproject.toml`: Package configuration, defines `mlpstorage` script entry

**Configuration:**
- `mlpstorage/config.py`: Runtime constants, enums, model definitions
- `configs/dlio/workload/*.yaml`: DLIO workload configurations
- `configs/vectordbbench/`: VectorDB configurations

**Core Logic:**
- `mlpstorage/benchmarks/base.py`: Abstract benchmark base class
- `mlpstorage/benchmarks/dlio.py`: DLIO-based training/checkpointing
- `mlpstorage/rules/verifier.py`: Validation orchestrator
- `mlpstorage/report_generator.py`: Report generation

**Testing:**
- `tests/conftest.py`: Shared pytest fixtures
- `tests/fixtures/`: Mock objects and sample data
- `tests/unit/`: Unit test modules

## Naming Conventions

**Files:**
- Modules: lowercase_with_underscores (`cli_parser.py`, `cluster_collector.py`)
- Test files: `test_<module_name>.py`
- Config files: `<model>_<accelerator>.yaml` or `<model>_datagen.yaml`

**Directories:**
- Packages: lowercase_with_underscores (`run_checkers`, `submission_checkers`)
- Special prefixes: None used

**Classes:**
- PascalCase (`TrainingBenchmark`, `BenchmarkVerifier`)
- Suffixes indicate type: `*Benchmark`, `*Checker`, `*Interface`, `*Info`

**Functions:**
- lowercase_with_underscores
- Private methods: `_method_name()`
- Abstract methods: `_run()` (single underscore)

## Where to Add New Code

**New Benchmark Type:**
1. Create implementation: `mlpstorage/benchmarks/<name>.py`
   - Inherit from `Benchmark` base class
   - Set `BENCHMARK_TYPE` class attribute
   - Implement `_run()` method
2. Create CLI args: `mlpstorage/cli/<name>_args.py`
   - Define `add_<name>_arguments(parser)` function
3. Create run checker: `mlpstorage/rules/run_checkers/<name>.py`
   - Inherit from `RunRulesChecker`
4. Register in `mlpstorage/benchmarks/__init__.py`:
   ```python
   BenchmarkRegistry.register(
       name='<name>',
       benchmark_class=<Name>Benchmark,
       cli_builder=add_<name>_arguments,
       description=PROGRAM_DESCRIPTIONS['<name>'],
   )
   ```

**New Validation Rule:**
1. Locate appropriate checker in `mlpstorage/rules/run_checkers/` or `submission_checkers/`
2. Add check method following pattern: `check_<rule_name>()`
3. Return `Issue` objects with appropriate `PARAM_VALIDATION` category

**New CLI Argument:**
1. Find appropriate args file in `mlpstorage/cli/`
2. Add argument to the builder function
3. If shared across benchmarks, add to `common_args.py`

**Utilities:**
- Shared helpers: `mlpstorage/utils.py`
- Cluster/MPI utilities: `mlpstorage/cluster_collector.py`
- Logging utilities: `mlpstorage/mlps_logging.py`

**New Test:**
1. Unit tests: `tests/unit/test_<module>.py`
2. Integration tests: `tests/integration/`
3. New fixtures: `tests/fixtures/`

## Special Directories

**configs/**
- Purpose: YAML configuration templates for benchmarks
- Generated: No
- Committed: Yes

**.planning/**
- Purpose: Claude GSD planning documents
- Generated: By GSD commands
- Committed: Typically yes (project documentation)

**kv_cache_benchmark/**
- Purpose: Standalone KV cache benchmark script (legacy location)
- Generated: No
- Committed: Yes
- Note: Being integrated into main package via `KVCacheBenchmark` wrapper

**checkpoints/**
- Purpose: Sample checkpoint data for testing
- Generated: By benchmark runs
- Committed: Sample only

**mlpstorage.egg-info/**
- Purpose: Build metadata (pip editable install)
- Generated: Yes
- Committed: No (in .gitignore)

**.venv/**
- Purpose: Virtual environment
- Generated: Yes
- Committed: No (in .gitignore)

---

*Structure analysis: 2026-01-23*
