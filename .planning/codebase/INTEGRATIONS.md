# External Integrations

**Analysis Date:** 2025-01-23

## APIs & External Services

**DLIO Benchmark:**
- Purpose: Core benchmark execution engine for training and checkpointing workloads
- SDK/Client: `dlio-benchmark` package (Git dependency)
- Integration: CLI wrapper via subprocess execution
- Files: `mlpstorage/benchmarks/dlio.py`

**No External SaaS APIs** - This is a self-contained benchmarking tool that runs locally or on private clusters.

## Data Storage

**Local/Distributed Filesystem:**
- Primary storage target for benchmarks
- No database dependencies
- Connection: Direct filesystem access via Python `os` and `pathlib`
- Client: Native Python file I/O

**Result Storage:**
- JSON files for metadata and results
- YAML files for configuration
- Configurable via `--results-dir` and `--data-dir` CLI arguments
- Default: `/tmp/mlperf_storage_results`

**Configuration Files:**
- `mlpstorage.yaml` - Optional global configuration
- Location: Project root or specified path
- Format: YAML with hosts, data_dir, results_dir

## MPI Integration

**MPI Execution Layer:**
- Purpose: Distributed benchmark execution across cluster nodes
- Binaries: `mpirun` or `mpiexec`
- Environment: `MPI_RUN_BIN`, `MPI_EXEC_BIN` environment variables
- Files: `mlpstorage/utils.py` (generate_mpi_prefix_cmd), `mlpstorage/cluster_collector.py`

**mpi4py:**
- Purpose: Python MPI bindings for cluster information collection
- Usage: MPI_COLLECTOR_SCRIPT in `mlpstorage/cluster_collector.py`
- Fallback: Local-only collection if MPI unavailable

## System Information Collection

**Linux /proc Filesystem:**
- `/proc/meminfo` - Memory statistics
- `/proc/cpuinfo` - CPU information
- `/proc/diskstats` - Disk I/O statistics
- `/proc/net/dev` - Network interface statistics
- `/proc/loadavg` - System load
- `/proc/uptime` - System uptime
- `/etc/os-release` - OS distribution info
- Files: `mlpstorage/cluster_collector.py`

**psutil Library:**
- Cross-platform system information
- CPU, memory, process monitoring
- Files: `mlpstorage/utils.py`

## Benchmark Tools

**DLIO (Deep Learning I/O):**
- External benchmark framework
- Invoked via subprocess
- Configuration: Hydra-based YAML files
- Files: `mlpstorage/benchmarks/dlio.py`, `configs/dlio/workload/*.yaml`

**VectorDB Benchmark:**
- External tool: `vdbbench` CLI
- Purpose: Vector database performance testing
- Configuration: `configs/vectordbbench/*.yaml`
- Files: `mlpstorage/benchmarks/vectordbbench.py`

**KV Cache Benchmark:**
- Internal Python script: `kv_cache_benchmark/kv-cache.py`
- Purpose: LLM KV cache offloading simulation
- Optional dependencies: PyTorch for GPU tier
- Files: `mlpstorage/benchmarks/kvcache.py`, `kv_cache_benchmark/`

## CI/CD & Deployment

**GitHub Actions:**
- Test workflow: `.github/workflows/test.yml`
- CLA workflow: `.github/workflows/cla.yml`
- Matrix testing: Python 3.10, 3.11, 3.12

**Codecov:**
- Coverage reporting integration
- Token-based authentication via `CODECOV_TOKEN` secret
- File: `.github/workflows/test.yml` line 45-51

**Ansible Deployment:**
- Automated cluster setup
- UV package manager installation
- MPI environment configuration
- Files: `ansible/setup.yml`, `ansible/inventory`

## Authentication & Identity

**No Authentication Layer** - Local execution model, no user authentication required.

**SSH/MPI Authentication:**
- SSH key-based access between cluster nodes for MPI execution
- Configured externally (not managed by mlpstorage)

## Monitoring & Observability

**Logging:**
- Custom logging framework: `mlpstorage/mlps_logging.py`
- Multiple verbosity levels: debug, verbose, verboser, status, result
- Output: Console (colored) and file-based logs

**Error Tracking:**
- Custom error hierarchy: `mlpstorage/errors.py`
- Error messages: `mlpstorage/error_messages.py`
- No external error tracking service

## Environment Configuration

**Required Environment Variables (optional):**
- `MLPS_DEBUG` - Enable debug mode
- `MPI_RUN_BIN` - Path to mpirun binary
- `MPI_EXEC_BIN` - Path to mpiexec binary

**CLI Configuration Override:**
All settings can be provided via CLI arguments, overriding:
1. Default values
2. YAML config files
3. Environment variables

**Secrets:**
- No secrets required for core functionality
- MPI relies on SSH key authentication (external configuration)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## File Format Integrations

**Input Formats:**
- YAML - Configuration files
- JSON - Existing result files for report generation

**Output Formats:**
- JSON - Metadata, results, cluster information
- YAML - DLIO/Hydra configuration snapshots
- Plain text - stdout/stderr logs
- CSV/XLSX - Optional report exports (KV cache benchmark)

---

*Integration audit: 2025-01-23*
