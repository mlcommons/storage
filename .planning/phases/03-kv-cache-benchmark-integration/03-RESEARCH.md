# Phase 3: KV Cache Benchmark Integration - Research

**Researched:** 2026-01-24
**Domain:** Python benchmark integration, CLI design, MPI execution
**Confidence:** HIGH

## Summary

This research covers what's needed to fully integrate the KV Cache benchmark into the mlpstorage framework. The good news: a substantial foundation already exists. The `KVCacheBenchmark` class is already implemented in `mlpstorage/benchmarks/kvcache.py`, inheriting from the `Benchmark` base class, registered with `BenchmarkRegistry`, and has a CLI argument builder in `mlpstorage/cli/kvcache_args.py`. The standalone `kv-cache.py` script in `kv_cache_benchmark/` is a mature, 2700+ line Python implementation with extensive functionality.

The primary integration gap is **MPI execution support**. The existing `KVCacheBenchmark` class executes the underlying `kv-cache.py` script directly via Python subprocess, but does not support MPI-based multi-host execution. The training and checkpointing benchmarks show the pattern: use `generate_mpi_prefix_cmd()` to wrap commands with MPI invocation. For KV cache, this needs adaptation since the benchmark is not DLIO-based.

**Primary recommendation:** Complete the KVCacheBenchmark integration by adding `--hosts` argument support, MPI command wrapping, and ensuring metadata output is consistent with other benchmarks. Most code already exists; this is incremental completion work.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 1.x/2.x | KV cache data generation | Required by kv-cache.py, core dependency |
| mpi4py | 4.x | MPI execution for multi-host | Existing pattern in training/checkpointing |
| argparse | stdlib | CLI parsing | Already in use in kv-cache.py and mlpstorage |

### Supporting (Optional in kv-cache.py)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch | 2.x | GPU tensor operations | When GPU tier enabled |
| cupy | 13.x | CUDA operations | Alternative to torch for GPU |
| tiktoken | 0.x | Token counting for ShareGPT | When using dataset-path option |
| pandas | 2.x | Excel/CSV output | When using --xlsx-output |
| openpyxl | 3.x | Excel file writing | When using --xlsx-output with .xlsx |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| subprocess for kv-cache.py | Import as module | Would require refactoring kv-cache.py internals; subprocess wrapper is cleaner for external script |
| MPI for multi-host | SSH + multiprocessing | MPI is the established pattern in this codebase; consistency matters |

**Installation:**
```bash
# Core KV cache dependencies
pip install numpy

# Optional dependencies
pip install torch  # or: pip install cupy-cuda12x
pip install tiktoken pandas openpyxl
```

## Architecture Patterns

### Recommended Project Structure
```
mlpstorage/
├── benchmarks/
│   └── kvcache.py           # KVCacheBenchmark class (EXISTS)
├── cli/
│   └── kvcache_args.py      # CLI argument builder (EXISTS)
kv_cache_benchmark/
└── kv-cache.py              # Standalone benchmark script (EXISTS)
```

### Pattern 1: Benchmark Subclass Pattern
**What:** All benchmarks inherit from `Benchmark` base class in `mlpstorage/benchmarks/base.py`
**When to use:** Always for new benchmark types
**Example:**
```python
# Source: mlpstorage/benchmarks/kvcache.py (existing)
class KVCacheBenchmark(Benchmark):
    BENCHMARK_TYPE = BENCHMARK_TYPES.kv_cache

    def __init__(self, args, logger=None, run_datetime=None, run_number=0,
                 cluster_collector=None, validator=None):
        super().__init__(args, logger, run_datetime, run_number,
                         cluster_collector, validator)
        # ... benchmark-specific init

    def _run(self) -> int:
        """Execute the benchmark based on the command."""
        command = getattr(self.args, 'command', 'run')
        handler = self.command_method_map.get(command)
        if handler:
            return handler()
        return 1
```

### Pattern 2: MPI Command Wrapping (from DLIOBenchmark)
**What:** Generate MPI prefix command for multi-host execution
**When to use:** When `--hosts` is provided and benchmark needs distributed execution
**Example:**
```python
# Source: mlpstorage/benchmarks/dlio.py (existing pattern)
if self.args.exec_type == EXEC_TYPE.MPI:
    mpi_prefix = generate_mpi_prefix_cmd(
        self.args.mpi_bin,
        self.args.hosts,
        self.args.num_processes,
        self.args.oversubscribe,
        self.args.allow_run_as_root,
        self.args.mpi_params,
        self.logger
    )
    cmd = f"{mpi_prefix} {cmd}"
```

### Pattern 3: Registry Pattern for CLI
**What:** Register benchmarks with their CLI builders at import time
**When to use:** Every benchmark needs registration
**Example:**
```python
# Source: mlpstorage/benchmarks/__init__.py (existing)
BenchmarkRegistry.register(
    name='kvcache',
    benchmark_class=KVCacheBenchmark,
    cli_builder=add_kvcache_arguments,
    description=PROGRAM_DESCRIPTIONS['kvcache'],
    help_text="KV Cache benchmark options"
)
```

### Pattern 4: Command Handler Map
**What:** Map CLI subcommands to handler methods
**When to use:** When benchmark has multiple subcommands (run, datasize, etc.)
**Example:**
```python
# Source: mlpstorage/benchmarks/kvcache.py (existing)
self.command_method_map = {
    "run": self._execute_run,
    "datasize": self._execute_datasize,
}
```

### Anti-Patterns to Avoid
- **Direct kv-cache.py modification:** Don't modify the standalone script; wrap it via KVCacheBenchmark
- **Bypassing base class:** Don't duplicate metadata, logging, or result directory logic; use base class methods
- **Hardcoded paths:** Use `_find_kvcache_script()` pattern for locating external scripts

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Output directory structure | Custom path logic | `generate_output_location()` from base class | Consistency with all benchmarks |
| Metadata JSON format | Custom JSON writer | Base class `metadata` property + `write_metadata()` | Includes cluster info, timing, verification |
| MPI command generation | String concatenation | `generate_mpi_prefix_cmd()` from utils | Handles slots, root permissions, extra params |
| Cluster info collection | Custom SSH commands | `collect_cluster_info()` from cluster_collector | Existing MPI-based collection |
| Command execution | Raw subprocess | `_execute_command()` from base class | Signal handling, output capture, logging |

**Key insight:** The base class does the heavy lifting. The integration task is mapping kv-cache.py options to the mlpstorage CLI and ensuring MPI wrapping works correctly.

## Common Pitfalls

### Pitfall 1: Missing --hosts Argument Support
**What goes wrong:** User tries `mlpstorage kvcache run --hosts host1 host2` but hosts aren't passed to kv-cache.py or MPI
**Why it happens:** KVCacheBenchmark currently doesn't add host arguments to the CLI or use them in command generation
**How to avoid:** Add `add_host_arguments()` and `add_mpi_arguments()` to kvcache CLI, modify `_build_kvcache_command()` to include MPI wrapper
**Warning signs:** Command runs on localhost only when hosts are specified

### Pitfall 2: Inconsistent Metadata Format
**What goes wrong:** KV cache benchmark results don't appear in `mlpstorage history list` or reports
**Why it happens:** Metadata structure differs from training/checkpointing patterns
**How to avoid:** Ensure metadata property includes required fields: `benchmark_type`, `model`, `command`, `run_datetime`, `result_dir`
**Warning signs:** History command shows empty or missing entries for kvcache runs

### Pitfall 3: Python Interpreter Mismatch in MPI
**What goes wrong:** MPI runs use different Python than mlpstorage was invoked with
**Why it happens:** kv-cache.py invoked via `sys.executable` locally but MPI workers use different Python
**How to avoid:** Pass Python path explicitly in MPI invocation, or use `--mpi-params` to set Python
**Warning signs:** ImportError for numpy or other dependencies on worker nodes

### Pitfall 4: Missing exec-type Argument
**What goes wrong:** User expects MPI execution but gets local-only run
**Why it happens:** KV cache CLI doesn't have `--exec-type` argument that training/checkpointing have
**How to avoid:** Add exec-type argument following training_args.py pattern
**Warning signs:** `--hosts` ignored silently

### Pitfall 5: Cache Directory Permissions on Remote Hosts
**What goes wrong:** Benchmark fails with permission denied on NVMe cache path
**Why it happens:** `--cache-dir` path doesn't exist or isn't writable on all MPI hosts
**How to avoid:** Document that cache-dir must be accessible from all hosts (shared storage or local to each)
**Warning signs:** First host succeeds, workers fail with file errors

## Code Examples

Verified patterns from the existing codebase:

### Adding Host/MPI Arguments (from training_args.py)
```python
# Source: mlpstorage/cli/training_args.py (pattern to follow)
def _add_training_run_arguments(parser):
    # ... other args ...
    add_host_arguments(parser)  # Adds --hosts
    add_mpi_arguments(parser)   # Adds --mpi-bin, --oversubscribe, --allow-run-as-root

    exec_group.add_argument(
        '--exec-type', '-et',
        choices=[EXEC_TYPE.MPI.value, EXEC_TYPE.DOCKER.value],
        default=EXEC_TYPE.MPI.value,
        help=HELP_MESSAGES['exec_type']
    )
```

### MPI Command Generation (from utils.py)
```python
# Source: mlpstorage/utils.py (existing function signature)
def generate_mpi_prefix_cmd(
    mpi_bin: str,
    hosts: List[str],
    num_processes: int,
    oversubscribe: bool = False,
    allow_run_as_root: bool = False,
    mpi_params: Optional[List[List[str]]] = None,
    logger: Optional[logging.Logger] = None
) -> str:
    """Generate MPI prefix command for distributed execution."""
```

### Building Command with MPI Wrapper (pattern to implement)
```python
# Pattern for KVCacheBenchmark._build_kvcache_command()
def _build_kvcache_command(self) -> str:
    cmd_parts = [
        sys.executable,
        self.kvcache_bin_path,
        f"--model {self.model}",
        f"--num-users {self.num_users}",
        # ... other args ...
    ]
    cmd = " ".join(cmd_parts)

    # Add MPI wrapper if distributed execution requested
    if getattr(self.args, 'exec_type', None) == EXEC_TYPE.MPI:
        if hasattr(self.args, 'hosts') and self.args.hosts:
            mpi_prefix = generate_mpi_prefix_cmd(
                self.args.mpi_bin,
                self.args.hosts,
                getattr(self.args, 'num_processes', len(self.args.hosts)),
                getattr(self.args, 'oversubscribe', False),
                getattr(self.args, 'allow_run_as_root', False),
                getattr(self.args, 'mpi_params', None),
                self.logger
            )
            cmd = f"{mpi_prefix} {cmd}"

    return cmd
```

### Metadata Extension (existing in kvcache.py)
```python
# Source: mlpstorage/benchmarks/kvcache.py (existing pattern)
@property
def metadata(self) -> Dict[str, Any]:
    """Generate metadata for the KV cache benchmark run."""
    base_metadata = super().metadata  # Gets standard fields from base

    base_metadata.update({
        'kvcache_model': self.model,
        'num_users': self.num_users,
        'duration': self.duration,
        # ... KV cache specific fields ...
    })

    return base_metadata
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Standalone kv-cache.py only | Integrated KVCacheBenchmark class | Already done | CLI integration, registry, base class benefits |
| Local execution only | MPI-based distributed | To be implemented | Multi-host benchmarking |
| No validation | Fail-fast environment validation | Phase 2 | Early error detection |

**Deprecated/outdated:**
- Direct invocation of `python kv-cache.py`: Use `mlpstorage kvcache run` instead
- Manual result collection: Base class handles metadata and result directories

## Existing Implementation Status

### Already Implemented (HIGH confidence)
- `KVCacheBenchmark` class structure (benchmarks/kvcache.py)
- CLI argument builder (cli/kvcache_args.py)
- Registry entry (benchmarks/__init__.py)
- Config constants: KVCACHE_MODELS, KVCACHE_PERFORMANCE_PROFILES, KVCACHE_GENERATION_MODES
- Script location logic (`_find_kvcache_script()`)
- Command building (`_build_kvcache_command()`)
- Results processing (`_process_results()`)
- Metadata extension
- Commands: `run`, `datasize`

### Not Yet Implemented (needs work)
- `--hosts` argument in CLI
- `--exec-type` argument (MPI/local)
- MPI arguments (`--mpi-bin`, `--oversubscribe`, `--allow-run-as-root`, `--mpi-params`)
- MPI command wrapping in `_build_kvcache_command()`
- Cluster info collection for distributed runs
- `--num-processes` argument for MPI ranks

### kv-cache.py Capabilities (from README)
The standalone script supports:
- Models: tiny-1b, mistral-7b, llama2-7b, llama3.1-8b, llama3.1-70b-instruct
- Multi-tier cache: GPU VRAM -> CPU RAM -> NVMe
- Generation modes: none, fast, realistic
- Performance profiles: latency, throughput
- Autoscaling: qos mode, capacity mode
- ShareGPT dataset replay
- RAG workload simulation
- Excel/CSV output

## Open Questions

Things that couldn't be fully resolved:

1. **MPI with kv-cache.py**
   - What we know: kv-cache.py is designed as single-process; uses threading internally
   - What's unclear: Whether running multiple MPI ranks each executing kv-cache.py is the intended pattern, or if the script needs modification
   - Recommendation: Treat each MPI rank as an independent KV cache instance (simulating different nodes); aggregate results afterward

2. **Cluster info collection for non-DLIO benchmarks**
   - What we know: DLIOBenchmark uses `accumulate_host_info()` with MPI collection
   - What's unclear: Whether KVCacheBenchmark should collect cluster info the same way
   - Recommendation: Add cluster collection when hosts are specified, following DLIOBenchmark pattern

3. **Shared vs local cache-dir in MPI**
   - What we know: `--cache-dir` specifies NVMe storage location
   - What's unclear: Should all MPI ranks use same path (shared storage) or local paths?
   - Recommendation: Document that users should specify appropriate path; don't auto-detect

## Sources

### Primary (HIGH confidence)
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/benchmarks/kvcache.py` - Existing implementation
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/benchmarks/base.py` - Base class pattern
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/benchmarks/dlio.py` - MPI pattern reference
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/cli/kvcache_args.py` - Existing CLI
- `/home/wvaske/Projects/mlperf-storage/kv_cache_benchmark/kv-cache.py` - Standalone script
- `/home/wvaske/Projects/mlperf-storage/kv_cache_benchmark/README.md` - Script documentation

### Secondary (MEDIUM confidence)
- `/home/wvaske/Projects/mlperf-storage/.planning/STATE.md` - Prior decisions and patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All dependencies documented in requirements.txt and kv-cache.py
- Architecture: HIGH - Patterns established in existing codebase
- Pitfalls: HIGH - Derived from comparing existing implementations

**Research date:** 2026-01-24
**Valid until:** 2026-02-24 (stable codebase, 30-day validity)

---

## Quick Implementation Checklist

For the planner to create tasks:

1. [ ] Add `--hosts`, `--exec-type`, and MPI arguments to kvcache CLI
2. [ ] Modify `_build_kvcache_command()` to include MPI wrapper
3. [ ] Add `--num-processes` argument for MPI rank count
4. [ ] Test MPI execution: `mlpstorage kvcache run --hosts host1 host2 --model llama3.1-8b`
5. [ ] Verify metadata appears in `mlpstorage history list`
6. [ ] Document cache-dir behavior for distributed runs
