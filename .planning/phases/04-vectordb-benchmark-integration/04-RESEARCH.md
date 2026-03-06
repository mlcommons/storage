# Phase 4: VectorDB Benchmark Integration - Research

**Researched:** 2026-01-24
**Domain:** Python benchmark integration, CLI design, Vector database benchmarking
**Confidence:** HIGH

## Summary

This research covers what's needed to fully integrate the VectorDB benchmark into the mlpstorage framework for the unified CLI experience. The good news: a substantial foundation already exists. The `VectorDBBenchmark` class is already implemented in `mlpstorage/benchmarks/vectordbbench.py`, inheriting from the `Benchmark` base class, registered with `BenchmarkRegistry`, and has a CLI argument builder in `mlpstorage/cli/vectordb_args.py`. Configuration files exist in `configs/vectordbbench/`.

The primary integration gaps are:
1. **CLI command naming inconsistency**: Current CLI uses `run-search` but should be `run` for consistency with other benchmarks
2. **Missing `--hosts` and MPI support**: VectorDB currently executes locally only; no distributed execution pattern
3. **Missing result directory in output_location()**: The `rules/utils.py:generate_output_location()` handles `vector_database` type but the output structure differs from other benchmarks (no model in path)
4. **Missing validation integration**: No fail-fast validation for VectorDB (e.g., checking vdbbench scripts exist)
5. **External dependency**: Relies on `load-vdb` and `vdbbench` CLI tools which are not included in the repository

**Primary recommendation:** Complete the VectorDBBenchmark integration by: (1) aligning CLI command naming with other benchmarks, (2) adding environment validation for external tools, (3) ensuring metadata output matches the format expected by history module, and (4) adding tests following the KV Cache test patterns.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pymilvus | 2.x | Milvus client SDK | Standard client for Milvus VectorDB (assumed by config) |
| numpy | 1.x/2.x | Vector data generation | Required for synthetic vector generation |
| pyyaml | 6.x | Configuration parsing | Already used by mlpstorage |

### External Tools (Required)
| Tool | Purpose | How Used |
|------|---------|----------|
| `load-vdb` | Data generation script | Called by VectorDBBenchmark.execute_datagen() |
| `vdbbench` | Benchmark execution script | Called by VectorDBBenchmark.execute_run() |

### Supporting (from config files)
| Library | Purpose | When to Use |
|---------|---------|-------------|
| DISKANN | Index type | Default index type in configs |
| COSINE | Metric type | Default similarity metric |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| External vdbbench | Built-in benchmark | Would require significant rewrite; external tools maintain separation |
| Milvus-only | Multiple VectorDB backends | Milvus is the current target; extensibility for future |

**Installation:**
```bash
# Core dependencies (not yet in pyproject.toml)
pip install pymilvus numpy

# External tools (must be on PATH)
# load-vdb and vdbbench commands must be available
```

## Architecture Patterns

### Recommended Project Structure
```
mlpstorage/
├── benchmarks/
│   └── vectordbbench.py     # VectorDBBenchmark class (EXISTS)
├── cli/
│   └── vectordb_args.py     # CLI argument builder (EXISTS)
configs/
└── vectordbbench/
    ├── default.yaml         # Default 1M vectors (EXISTS)
    └── 10m.yaml             # 10M vectors config (EXISTS)
```

### Pattern 1: Benchmark Subclass Pattern
**What:** All benchmarks inherit from `Benchmark` base class in `mlpstorage/benchmarks/base.py`
**When to use:** Always for new benchmark types
**Example:**
```python
# Source: mlpstorage/benchmarks/vectordbbench.py (existing)
class VectorDBBenchmark(Benchmark):
    BENCHMARK_TYPE = BENCHMARK_TYPES.vector_database
    VECTORDB_CONFIG_PATH = "vectordbbench"
    VDBBENCH_BIN = "vdbbench"

    def __init__(self, args):
        super().__init__(args)
        self.command_method_map = {
            "datagen": self.execute_datagen,
            "run-search": self.execute_run  # Note: should be "run"
        }
```

### Pattern 2: Command Handler Map
**What:** Map CLI subcommands to handler methods
**When to use:** When benchmark has multiple subcommands (run, datagen, etc.)
**Example:**
```python
# Source: mlpstorage/benchmarks/vectordbbench.py (existing)
self.command_method_map = {
    "datagen": self.execute_datagen,
    "run-search": self.execute_run,  # Current
    # Should be:
    # "run": self.execute_run,  # For consistency
}
```

### Pattern 3: Registry Pattern for CLI
**What:** Register benchmarks with their CLI builders at import time
**When to use:** Every benchmark needs registration
**Example:**
```python
# Source: mlpstorage/benchmarks/__init__.py (existing)
BenchmarkRegistry.register(
    name='vectordb',
    benchmark_class=VectorDBBenchmark,
    cli_builder=add_vectordb_arguments,
    description=PROGRAM_DESCRIPTIONS['vectordb'],
    help_text="VectorDB benchmark options"
)
```

### Pattern 4: External Script Wrapping (from KVCacheBenchmark)
**What:** Wrap external scripts with command building and execution
**When to use:** When benchmark calls external tools
**Example:**
```python
# Source: mlpstorage/benchmarks/kvcache.py (pattern to follow)
def _find_kvcache_script(self) -> str:
    """Locate the kv-cache.py script with fallback paths."""
    # Check custom path first
    custom_path = getattr(self.args, 'kvcache_bin_path', None)
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)
    # Fallback to known locations...
```

### Anti-Patterns to Avoid
- **Inconsistent CLI naming:** `run-search` vs `run` - use `run` for consistency
- **Hardcoded external tool paths:** Use `_find_script()` pattern or PATH lookup
- **Bypassing base class:** Don't duplicate metadata, logging, or result directory logic
- **Missing validation:** Always validate external dependencies exist before execution

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Output directory structure | Custom path logic | `generate_output_location()` from rules/utils.py | Consistency with all benchmarks |
| Metadata JSON format | Custom JSON writer | Base class `metadata` property + `write_metadata()` | Includes cluster info, timing, verification |
| Command execution | Raw subprocess | `_execute_command()` from base class | Signal handling, output capture, logging |
| CLI argument groups | Inline definitions | `add_universal_arguments()` from common_args | Shared validation, results-dir, loops |
| External tool lookup | `which` command | `shutil.which()` + validation | Cross-platform, proper error messages |

**Key insight:** The base class does the heavy lifting. The integration task is ensuring VectorDBBenchmark follows established patterns for CLI commands, metadata, and external tool validation.

## Common Pitfalls

### Pitfall 1: CLI Command Naming Inconsistency
**What goes wrong:** User expects `mlpstorage vectordb run` but actual command is `mlpstorage vectordb run-search`
**Why it happens:** Original implementation used different naming convention
**How to avoid:** Rename `run-search` to `run` in CLI args and update command_method_map
**Warning signs:** Documentation/help text doesn't match actual commands

### Pitfall 2: External Tools Not Found
**What goes wrong:** Benchmark fails with cryptic error when `load-vdb` or `vdbbench` not on PATH
**Why it happens:** No validation that required tools exist before benchmark execution
**How to avoid:** Add fail-fast validation in `_validate_environment()` or main.py dispatch
**Warning signs:** "Command not found" errors during benchmark execution

### Pitfall 3: Missing Metadata Fields for History
**What goes wrong:** VectorDB benchmark results don't appear in `mlpstorage history list` or reports
**Why it happens:** Metadata structure differs from training/checkpointing patterns
**How to avoid:** Ensure metadata property includes: `benchmark_type`, `model` (or equivalent), `command`, `run_datetime`, `result_dir`
**Warning signs:** History command shows empty or missing entries for vectordb runs

### Pitfall 4: Config File Not Found
**What goes wrong:** Benchmark fails when custom config specified but doesn't exist
**Why it happens:** `read_config_from_file()` throws exception; no pre-validation
**How to avoid:** Validate config file exists in `__init__` before proceeding
**Warning signs:** Stack trace mentioning yaml parsing or file not found

### Pitfall 5: No Database Connection Validation
**What goes wrong:** Benchmark runs for a while before failing on database connection
**Why it happens:** Milvus connection not validated until actual operation
**How to avoid:** Add optional connection check in validation phase (with timeout)
**Warning signs:** Long delay before connection timeout error

## Code Examples

Verified patterns from the existing codebase:

### Current VectorDB CLI Structure (existing)
```python
# Source: mlpstorage/cli/vectordb_args.py (existing)
def add_vectordb_arguments(parser):
    vectordb_subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="sub_commands"
    )

    datagen = vectordb_subparsers.add_parser(
        'datagen',
        help=HELP_MESSAGES['vdb_datagen']
    )
    run_search = vectordb_subparsers.add_parser(
        'run-search',  # Should be 'run' for consistency
        help=HELP_MESSAGES['vdb_run_search']
    )
```

### Pattern for Renaming run-search to run
```python
# Pattern to implement in vectordb_args.py
def add_vectordb_arguments(parser):
    vectordb_subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="sub_commands"
    )

    datagen = vectordb_subparsers.add_parser(
        'datagen',
        help=HELP_MESSAGES['vdb_datagen']
    )
    run_benchmark = vectordb_subparsers.add_parser(  # Changed from run_search
        'run',  # Changed from 'run-search'
        help=HELP_MESSAGES['vdb_run']  # Update help message
    )
```

### External Tool Validation Pattern (from dependency_check.py pattern)
```python
# Pattern to implement for VectorDB
import shutil

def validate_vectordb_dependencies(logger=None) -> tuple:
    """Validate VectorDB external tools are available.

    Returns:
        Tuple of (load_vdb_path, vdbbench_path) or raises DependencyError
    """
    load_vdb_path = shutil.which('load-vdb')
    vdbbench_path = shutil.which('vdbbench')

    errors = []
    if not load_vdb_path:
        errors.append("'load-vdb' command not found. Install vdbbench tools.")
    if not vdbbench_path:
        errors.append("'vdbbench' command not found. Install vdbbench tools.")

    if errors:
        from mlpstorage.errors import DependencyError
        raise DependencyError(
            "Missing VectorDB dependencies",
            suggestion="Install vdbbench tools and ensure they are on PATH",
            missing_packages=['load-vdb', 'vdbbench']
        )

    return load_vdb_path, vdbbench_path
```

### Command Handler Map Update
```python
# Source: mlpstorage/benchmarks/vectordbbench.py (to be updated)
def __init__(self, args):
    super().__init__(args)
    self.command_method_map = {
        "datagen": self.execute_datagen,
        "run": self.execute_run,  # Changed from "run-search"
    }
```

### Metadata Extension (pattern from KVCacheBenchmark)
```python
# Pattern to implement for VectorDBBenchmark
@property
def metadata(self) -> Dict[str, Any]:
    """Generate metadata for the VectorDB benchmark run."""
    base_metadata = super().metadata

    # Add VectorDB specific metadata
    base_metadata.update({
        'vectordb_config': self.config_name,
        'host': getattr(self.args, 'host', '127.0.0.1'),
        'port': getattr(self.args, 'port', 19530),
        'collection': getattr(self.args, 'collection', None),
    })

    # Add command-specific parameters
    if self.command == 'datagen':
        base_metadata.update({
            'dimension': getattr(self.args, 'dimension', None),
            'num_vectors': getattr(self.args, 'num_vectors', None),
            'num_shards': getattr(self.args, 'num_shards', None),
        })
    elif self.command == 'run':
        base_metadata.update({
            'num_query_processes': getattr(self.args, 'num_query_processes', None),
            'batch_size': getattr(self.args, 'batch_size', None),
            'runtime': getattr(self.args, 'runtime', None),
        })

    return base_metadata
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Standalone vdbbench scripts | Integrated VectorDBBenchmark class | Already done | CLI integration, registry, base class benefits |
| `run-search` command | `run` command | To be implemented | Consistent CLI across all benchmarks |
| No external tool validation | Fail-fast validation | To be implemented | Early error detection |

**Deprecated/outdated:**
- `run-search` subcommand name: Will be renamed to `run`
- Direct invocation of external scripts: Use `mlpstorage vectordb run` instead

## Existing Implementation Status

### Already Implemented (HIGH confidence)
- `VectorDBBenchmark` class structure (benchmarks/vectordbbench.py)
- CLI argument builder (cli/vectordb_args.py)
- Registry entry (benchmarks/__init__.py)
- Config constants: VECTOR_DTYPES, DISTRIBUTIONS, SEARCH_METRICS, INDEX_TYPES
- Configuration files (configs/vectordbbench/default.yaml, 10m.yaml)
- Output location handler in rules/utils.py (vector_database type)
- Commands: `datagen`, `run-search` (datagen and run functionality)

### Not Yet Implemented (needs work)
- Rename `run-search` to `run` for CLI consistency
- External tool validation (load-vdb, vdbbench existence check)
- Metadata enhancement for history module compatibility
- Unit tests (test_benchmarks_vectordb.py, test_cli_vectordb.py)
- `--hosts` and MPI support (if distributed execution desired)
- `model` field in metadata (VectorDB doesn't have traditional "model" but needs equivalent)

### VectorDBBenchmark Commands (existing)
The benchmark supports:
- `datagen`: Generate vectors and load into Milvus collection via `load-vdb`
- `run-search` (to be `run`): Execute search benchmark via `vdbbench`

### Configuration Parameters (from configs/vectordbbench/)
```yaml
database:
  host: 127.0.0.1
  port: 19530
  database: milvus

dataset:
  collection_name: mlps_1m_1shards_1536dim_uniform
  num_vectors: 1_000_000
  dimension: 1536
  distribution: uniform
  num_shards: 1
  vector_dtype: FLOAT_VECTOR

index:
  index_type: DISKANN
  metric_type: COSINE
```

## Open Questions

Things that couldn't be fully resolved:

1. **External Tool Distribution**
   - What we know: VectorDBBenchmark calls `load-vdb` and `vdbbench` external commands
   - What's unclear: How are these tools distributed? Are they in a separate package?
   - Recommendation: Document installation requirements; add clear error message when tools missing

2. **Model Equivalent for History**
   - What we know: History module expects `model` field in metadata
   - What's unclear: VectorDB doesn't have a "model" in the ML sense
   - Recommendation: Use `config_name` or `collection_name` as model equivalent for metadata

3. **Distributed VectorDB Benchmarking**
   - What we know: Current implementation is single-host only
   - What's unclear: Is multi-host benchmarking needed for VectorDB?
   - Recommendation: Document as single-host for now; MPI support can be added if needed

4. **Database Connection Validation**
   - What we know: Connection errors surface during benchmark execution
   - What's unclear: Should we validate Milvus connection in fail-fast phase?
   - Recommendation: Add optional connection check with short timeout in validation

## Sources

### Primary (HIGH confidence)
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/benchmarks/vectordbbench.py` - Existing implementation
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/benchmarks/base.py` - Base class pattern
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/benchmarks/kvcache.py` - Parallel track reference
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/cli/vectordb_args.py` - Existing CLI
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/cli/kvcache_args.py` - CLI pattern reference
- `/home/wvaske/Projects/mlperf-storage/configs/vectordbbench/default.yaml` - Configuration structure
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/rules/utils.py` - Output location handling
- `/home/wvaske/Projects/mlperf-storage/tests/unit/test_benchmarks_kvcache.py` - Test pattern reference
- `/home/wvaske/Projects/mlperf-storage/tests/unit/test_cli_kvcache.py` - CLI test pattern reference

### Secondary (MEDIUM confidence)
- `/home/wvaske/Projects/mlperf-storage/.planning/phases/03-kv-cache-benchmark-integration/03-RESEARCH.md` - Parallel phase patterns

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - External tools (load-vdb, vdbbench) not verified in codebase
- Architecture: HIGH - Patterns established in existing codebase
- Pitfalls: HIGH - Derived from comparing existing implementations

**Research date:** 2026-01-24
**Valid until:** 2026-02-24 (stable codebase, 30-day validity)

---

## Quick Implementation Checklist

For the planner to create tasks:

1. [ ] Rename `run-search` to `run` in vectordb_args.py
2. [ ] Update command_method_map in VectorDBBenchmark
3. [ ] Add external tool validation (load-vdb, vdbbench)
4. [ ] Enhance metadata property for history compatibility
5. [ ] Add `model` equivalent field (config_name or collection_name)
6. [ ] Create test_benchmarks_vectordb.py following KVCache pattern
7. [ ] Create test_cli_vectordb.py following KVCache pattern
8. [ ] Update HELP_MESSAGES for new command names
9. [ ] Verify `mlpstorage vectordb run` works end-to-end
10. [ ] Verify `mlpstorage vectordb datagen` works end-to-end
11. [ ] Verify benchmark appears in `mlpstorage history list`
