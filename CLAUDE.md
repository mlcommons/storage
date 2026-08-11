# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLPerf Storage Benchmark Suite (v2.0.0b1) - a Python framework for benchmarking storage systems supporting ML workloads. The suite uses DLIO (Deep Learning I/O) benchmark as its execution engine.

## Common Commands

```bash
# Install for development
pip install -e .

# Install with test dependencies
pip install -e ".[test]"

# Install with full DLIO support for running benchmarks
pip install -e ".[full]"

# Run all unit tests
pytest tests/unit -v

# Run a single test file
pytest tests/unit/test_cli.py -v

# Run tests with coverage
pytest tests/unit -v --cov=mlpstorage --cov-report=xml

# Run integration tests
pytest tests/integration -v
```

## CLI Usage

The main entry point is `mlpstorage`. Benchmark commands sit under a required
submission-mode positional (`closed`, `open`, `whatif`); commands that touch
storage take a trailing `file|object` selector. Only training has a model
positional — the other benchmarks select models via flags. `mlpstorage
--help_all` prints the complete hand-curated command reference (kept in
lockstep with the parser by `tests/unit/test_help_all_parity.py`).

```bash
# Training (model positional; closed/open models: unet3d, retinanet)
mlpstorage closed training unet3d datasize ...          # Calculate required dataset size
mlpstorage closed training unet3d datagen file ...      # Generate synthetic data
mlpstorage closed training unet3d run file ...          # Execute benchmark
mlpstorage closed training unet3d configview file ...   # View final configuration

# Checkpointing (--model/-m required: llama3-8b, llama3-70b, llama3-405b, llama3-1t)
mlpstorage closed checkpointing datasize --model llama3-8b ...
mlpstorage closed checkpointing run file --model llama3-8b ...
mlpstorage closed checkpointing configview file --model llama3-8b ...

# Other benchmarks
mlpstorage closed vectordb run file ...    # Vector database (index via --vdb-index)
mlpstorage closed kvcache run ...          # KV cache (no file|object; model fixed in closed)

# Utilities (top-level, no mode positional)
mlpstorage reports reportgen ...           # Generate submission reports
mlpstorage history show/rerun ...          # Command history
mlpstorage init <orgname> <results-dir>    # Pin orgname to a results-dir
mlpstorage validate <submission-dir>       # Rules.md submission checker
```

## Architecture

### Benchmark System

All benchmarks inherit from `Benchmark` base class (`mlpstorage/benchmarks/base.py`):
- Subclasses implement `_run()` method and set `BENCHMARK_TYPE` class attribute
- Base class handles cluster info collection, result directories, metadata, and signal handling
- Supports dependency injection for cluster collectors and validators (for testing)

Concrete implementations in `mlpstorage/benchmarks/`:
- `TrainingBenchmark`, `CheckpointingBenchmark` - DLIO-based benchmarks
- `VectorDBBenchmark` - Vector database operations
- `KVCacheBenchmark` - LLM KV cache management

### Registry Pattern

`BenchmarkRegistry` (`mlpstorage/registry.py`) dynamically registers benchmarks at import time. Each benchmark registration includes its CLI argument builder, enabling automatic CLI construction.

### Configuration Flow

1. CLI arguments parsed via `cli_parser.py`
2. YAML config templates loaded from `configs/dlio/workload/`
3. Parameters merged with precedence: CLI args > YAML config > environment variables
4. Dotted-key parameters (e.g., `dataset.num_files_train`) flattened/unflattened for DLIO

### Validation System

Located in `mlpstorage/rules/`:
- **Run Checkers** (`run_checkers/`) - Real-time validation during execution
- **Submission Checkers** (`submission_checkers/`) - Post-run compliance validation
- **BenchmarkVerifier** (`verifier.py`) - Orchestrates all validation
- Validation states: `CLOSED`, `OPEN`, `INVALID` (defined in `config.py` as `PARAM_VALIDATION` enum)

### MPI Integration

- `cluster_collector.py` - MPI-based system information collection
- Commands executed via `CommandExecutor` in `utils.py` with live output streaming
- Supports both `mpirun` and `mpiexec` via `--mpi-bin` flag

## Key Files

| File | Purpose |
|------|---------|
| `mlpstorage/main.py` | Entry point with signal/error handling |
| `mlpstorage/benchmarks/base.py` | Abstract benchmark base class |
| `mlpstorage/benchmarks/__init__.py` | Benchmark registry initialization |
| `mlpstorage/config.py` | Constants, enums, model configurations |
| `mlpstorage/rules/models.py` | Data classes for validation pipeline |
| `mlpstorage/utils.py` | Command execution, JSON encoding, config loading |

## Adding a New Benchmark

1. Create benchmark class inheriting from `Benchmark`
2. Set `BENCHMARK_TYPE` class attribute
3. Implement `_run()` method
4. Create CLI argument builder in `mlpstorage/cli/`
5. Register in `mlpstorage/benchmarks/__init__.py` via `BenchmarkRegistry.register()`

## Testing

Tests use pytest with fixtures in `tests/fixtures/`:
- `mock_collector.py` - Mock cluster collector
- `mock_executor.py` - Mock command executor
- `mock_logger.py` - Mock logger
- `sample_data.py` - Sample test data

### Test Environment

When running the `mlpstorage` CLI for manual testing or integration tests, use:
- **Data directory**: `/databases/mlps-v3.0/data/`
- **Results directory**: `/databases/mlps-v3.0/results/`

#### Example Commands

```bash
# Generate dataset for unet3d with 4 processes
mlpstorage closed training unet3d datagen file \
    --num-processes 4 \
    --data-dir /databases/mlps-v3.0/data/ \
    --results-dir /databases/mlps-v3.0/results \
    --systemname dev-system

# Run training benchmark for unet3d with 2 b200 accelerators
mlpstorage closed training unet3d run file \
    --num-accelerators 2 \
    --accelerator-type b200 \
    --client-host-memory-in-gb 64 \
    --data-dir /databases/mlps-v3.0/data/ \
    --results-dir /databases/mlps-v3.0/results \
    --systemname dev-system
```

**Note**: These benchmarks require MPI (OpenMPI) to be installed. Install with:
```bash
# Ubuntu/Debian
sudo apt-get install openmpi-bin

# RHEL/CentOS
sudo yum install openmpi
```

## Key Constants

From `mlpstorage/config.py`:
- Training models: `unet3d`, `retinanet` (closed/open); whatif adds `cosmoflow`, `resnet50`, `dlrm`, `flux`
- LLM models (checkpointing): `llama3-8b`, `llama3-70b`, `llama3-405b`, `llama3-1t`
- Accelerators: `b200`, `mi355` (closed/open); whatif adds `h100`, `a100`
- Submission modes: `closed`, `open`, `whatif`

## GSD Workflow

This project uses Get Shit Done (GSD) for structured development. Planning artifacts live in `.planning/`.

### Phase Commands

```bash
/gsd-discuss-phase N    # Discuss approach and clarify before planning
/gsd-plan-phase N       # Generate execution plan for phase N
/gsd-execute-phase N    # Execute the plan for phase N
/gsd-transition         # Complete phase, update PROJECT.md and STATE.md
/gsd-progress           # Check current progress
/gsd-explore            # Open-ended Socratic ideation session
