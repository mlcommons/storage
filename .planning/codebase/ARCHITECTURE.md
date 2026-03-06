# Architecture

**Analysis Date:** 2026-01-23

## Pattern Overview

**Overall:** Layered Architecture with Registry Pattern for extensibility

**Key Characteristics:**
- Abstract base classes define contracts (BenchmarkInterface, RulesChecker)
- Registry pattern enables dynamic benchmark registration without core code changes
- Dependency injection supports testing (cluster collectors, validators)
- Configuration-driven behavior via YAML files and CLI arguments
- Separation of validation rules into single-run and multi-run submission checkers

## Layers

**CLI Layer:**
- Purpose: Parse arguments, route to appropriate benchmark/command
- Location: `mlpstorage/cli/`, `mlpstorage/cli_parser.py`
- Contains: Argument builders per benchmark type, help messages, validation
- Depends on: Config constants
- Used by: Entry point (`main.py`)

**Benchmark Layer:**
- Purpose: Execute benchmark workloads, manage results, metadata
- Location: `mlpstorage/benchmarks/`
- Contains: Base class + concrete implementations (Training, Checkpointing, VectorDB, KVCache)
- Depends on: CLI args, Rules layer, Utils, Cluster collector
- Used by: Main entry point

**Rules/Validation Layer:**
- Purpose: Validate benchmark parameters and results against MLPerf submission rules
- Location: `mlpstorage/rules/`
- Contains: Run checkers (single run), Submission checkers (multi-run), Verifier orchestrator
- Depends on: Config constants, Data models
- Used by: Benchmark layer (pre-execution validation), Report generator (post-execution)

**Reporting Layer:**
- Purpose: Generate reports from benchmark results
- Location: `mlpstorage/report_generator.py`, `mlpstorage/reporting/`
- Contains: Directory validation, message formatting, CSV/JSON output
- Depends on: Rules layer, Utils
- Used by: CLI `reports` command

**Infrastructure Layer:**
- Purpose: Common utilities, command execution, cluster information collection
- Location: `mlpstorage/utils.py`, `mlpstorage/cluster_collector.py`
- Contains: CommandExecutor, MPI command generation, config loading, JSON encoding
- Depends on: Config
- Used by: All other layers

**Interfaces Layer:**
- Purpose: Define contracts for testability and extensibility
- Location: `mlpstorage/interfaces/`
- Contains: BenchmarkInterface, ValidatorInterface, ClusterCollectorInterface
- Depends on: None
- Used by: Benchmark implementations, validators

## Data Flow

**Benchmark Execution Flow:**

1. CLI parses arguments (`cli_parser.py` -> `parse_arguments()`)
2. Main routes to appropriate program handler (`main.py` -> `run_benchmark()`)
3. Benchmark class instantiated with args, datetime, logger
4. Cluster information collected via MPI if available (`cluster_collector.py`)
5. Pre-execution validation via `BenchmarkVerifier.verify()`
6. DLIO command generated with parameters from YAML + CLI overrides
7. Command executed via `CommandExecutor.execute()` with signal handling
8. Metadata written to result directory as JSON
9. Exit code returned

**Validation Flow:**

1. `BenchmarkVerifier` receives Benchmark instance or result directory path
2. Creates appropriate `RunRulesChecker` based on benchmark type
3. Checker runs validation rules, produces list of `Issue` objects
4. Each issue has `validation` field: CLOSED, OPEN, or INVALID
5. Verifier aggregates issues to determine overall category
6. Results stored on benchmark run object for metadata

**State Management:**
- Benchmark state held in Benchmark instance attributes
- Cluster information stored in `ClusterInformation` dataclass
- Validation results stored as list of `Issue` dataclasses
- Results persisted as JSON metadata files in result directories

## Key Abstractions

**Benchmark:**
- Purpose: Represents a benchmark workload that can be executed
- Examples: `mlpstorage/benchmarks/dlio.py`, `mlpstorage/benchmarks/kvcache.py`
- Pattern: Abstract base class with Template Method pattern
- Interface: `BenchmarkInterface` in `mlpstorage/interfaces/benchmark.py`
- Key methods: `_run()` (abstract), `run()` (template), `verify_benchmark()`, `_execute_command()`

**BenchmarkRun:**
- Purpose: Represents a completed or in-progress benchmark run with all data needed for validation
- Examples: `mlpstorage/rules/models.py` -> `BenchmarkRun`, `BenchmarkRunData`
- Pattern: Data Transfer Object / Value Object
- Created from: Live Benchmark instance or result directory on disk

**RulesChecker:**
- Purpose: Validates benchmark parameters/results against submission rules
- Examples: `mlpstorage/rules/run_checkers/training.py`, `mlpstorage/rules/submission_checkers/checkpointing.py`
- Pattern: Strategy pattern - different checkers for different benchmark types
- Base class: `RulesChecker` in `mlpstorage/rules/base.py`

**ClusterInformation:**
- Purpose: Aggregated system information from all cluster hosts
- Location: `mlpstorage/rules/models.py`
- Pattern: Aggregate root containing list of `HostInfo` objects
- Created via: MPI-based collection or CLI argument fallback

## Entry Points

**Main CLI Entry Point:**
- Location: `mlpstorage/main.py` -> `main()`
- Triggers: `mlpstorage` command (defined in pyproject.toml `[project.scripts]`)
- Responsibilities: Signal handling, error handling, routing to benchmark/report handlers

**Benchmark Registration:**
- Location: `mlpstorage/benchmarks/__init__.py` -> `register_benchmarks()`
- Triggers: Import of `mlpstorage.benchmarks` module
- Responsibilities: Register all benchmarks with BenchmarkRegistry for CLI construction

**Report Generation:**
- Location: `mlpstorage/report_generator.py` -> `ReportGenerator`
- Triggers: `mlpstorage reports reportgen` command
- Responsibilities: Validate results directory, run verifiers, output reports

## Error Handling

**Strategy:** Hierarchical exception classes with structured error information

**Patterns:**
- Custom exceptions in `mlpstorage/errors.py` inherit from `MLPStorageException`
- Each exception carries `ErrorCode`, message, details, and suggestion
- Main entry point has comprehensive try/except for all exception types
- Error formatter provides colored terminal output with actionable suggestions

**Exception Hierarchy:**
```
MLPStorageException (base)
├── ConfigurationError   - Invalid config values, missing params
├── BenchmarkExecutionError - Command failures, timeouts
├── ValidationError      - Rule validation failures
├── FileSystemError      - Path not found, permissions
├── MPIError            - MPI communication failures
└── DependencyError     - Missing DLIO, MPI
```

## Cross-Cutting Concerns

**Logging:**
- Custom logger setup in `mlpstorage/mlps_logging.py`
- Extended log levels: status, verbose, verboser, ridiculous
- Log level controlled via `--stream-log-level` and `--verbose`/`--debug` flags

**Validation:**
- Pre-execution: `Benchmark.verify_benchmark()` -> `BenchmarkVerifier`
- Post-execution: `ReportGenerator` -> `BenchmarkVerifier`
- Categories: CLOSED (strict rules), OPEN (relaxed), INVALID (fails submission)

**Configuration:**
- YAML configs in `configs/dlio/workload/` define benchmark parameters
- CLI arguments override YAML values
- Environment variables provide fallback defaults (`check_env()` in config.py)
- Parameter precedence: CLI > YAML config > environment > defaults

**MPI Integration:**
- Cluster collector uses MPI to gather system info from all hosts
- DLIO benchmarks generate MPI command prefix via `generate_mpi_prefix_cmd()`
- Host distribution and slot allocation handled automatically

---

*Architecture analysis: 2026-01-23*
