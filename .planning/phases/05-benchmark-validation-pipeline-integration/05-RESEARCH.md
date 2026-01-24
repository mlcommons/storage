# Phase 5: Benchmark Validation Pipeline Integration - Research

**Researched:** 2026-01-24
**Domain:** Python validation framework, rules engine, report generation
**Confidence:** HIGH

## Summary

This research covers what's needed to integrate KV Cache and VectorDB benchmarks into the existing validation and reporting pipeline. The good news: the validation framework is well-structured and extensible. The `BenchmarkVerifier` orchestrates validation using `RunRulesChecker` subclasses for single-run validation and `MultiRunRulesChecker` subclasses for submission-level validation. KV Cache already has a `KVCacheRunRulesChecker` implemented (Phase 3 work).

The primary integration gaps are:
1. **BenchmarkVerifier routing**: Currently only routes to training/checkpointing checkers; needs KV Cache and VectorDB routing
2. **VectorDB run checker**: `VectorDBRunRulesChecker` does not exist; needs implementation following KVCacheRunRulesChecker pattern
3. **Submission checkers**: Neither KV Cache nor VectorDB have submission-level checkers (multi-run validation)
4. **Result file parsing**: `ResultFilesExtractor` and `DLIOResultParser` are DLIO-specific; need generic fallback for non-DLIO benchmarks
5. **Report formatters**: `ClosedRequirementsFormatter` has KV Cache requirements but VectorDB is missing

**Primary recommendation:** Extend the validation pipeline by: (1) adding VectorDBRunRulesChecker, (2) updating BenchmarkVerifier to route kv_cache and vector_database types, (3) creating submission checkers if submission rules exist for these benchmark types, and (4) ensuring ResultFilesExtractor handles non-DLIO metadata gracefully.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| dataclasses | stdlib | Data models (Issue, BenchmarkRunData) | Already used throughout rules module |
| enum | stdlib | Validation states (PARAM_VALIDATION) | Already defined in config.py |
| json | stdlib | Metadata parsing | Standard format for result files |
| yaml | 6.x (pyyaml) | Hydra config parsing (DLIO runs) | Already used for DLIO results |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib | stdlib | Path manipulation | Directory validation |
| typing | stdlib | Type hints | All validation classes |
| re | stdlib | Pattern matching | Directory name validation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Class-based checkers | Function-based checks | Class-based allows better state management and inheritance |
| Auto-discovery of check_* methods | Explicit method registration | Auto-discovery reduces boilerplate, is the established pattern |

**Installation:**
```bash
# No additional dependencies required - all stdlib or already in project
```

## Architecture Patterns

### Recommended Project Structure
```
mlpstorage/rules/
├── __init__.py              # Exports all checkers (update needed)
├── base.py                  # RulesChecker base class (EXISTS)
├── issues.py                # Issue dataclass (EXISTS)
├── models.py                # BenchmarkRunData, BenchmarkRun, etc (EXISTS)
├── verifier.py              # BenchmarkVerifier orchestrator (UPDATE needed)
├── utils.py                 # Utility functions (EXISTS)
├── run_checkers/
│   ├── __init__.py          # (UPDATE to export VectorDB)
│   ├── base.py              # RunRulesChecker (EXISTS)
│   ├── training.py          # TrainingRunRulesChecker (EXISTS)
│   ├── checkpointing.py     # CheckpointingRunRulesChecker (EXISTS)
│   ├── kvcache.py           # KVCacheRunRulesChecker (EXISTS)
│   └── vectordb.py          # VectorDBRunRulesChecker (NEW)
└── submission_checkers/
    ├── __init__.py          # (UPDATE to export new checkers)
    ├── base.py              # MultiRunRulesChecker (EXISTS)
    ├── training.py          # TrainingSubmissionRulesChecker (EXISTS)
    ├── checkpointing.py     # CheckpointSubmissionRulesChecker (EXISTS)
    ├── kvcache.py           # KVCacheSubmissionRulesChecker (NEW if needed)
    └── vectordb.py          # VectorDBSubmissionRulesChecker (NEW if needed)
```

### Pattern 1: RunRulesChecker Subclass Pattern
**What:** All single-run validators inherit from `RunRulesChecker` and implement `check_*` methods
**When to use:** For validating individual benchmark runs
**Example:**
```python
# Source: mlpstorage/rules/run_checkers/kvcache.py (existing)
class KVCacheRunRulesChecker(RunRulesChecker):
    """Rules checker for KV Cache benchmarks."""

    MIN_DURATION_SECONDS = 30
    MIN_NUM_USERS = 1

    def check_benchmark_type(self) -> Optional[Issue]:
        """Verify this is a KV Cache benchmark."""
        if self.benchmark_run.benchmark_type != BENCHMARK_TYPES.kv_cache:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid benchmark type: {self.benchmark_run.benchmark_type}",
                parameter="benchmark_type",
                expected=BENCHMARK_TYPES.kv_cache,
                actual=self.benchmark_run.benchmark_type
            )
        return None

    def check_duration(self) -> Optional[Issue]:
        """Verify benchmark duration is valid."""
        duration = self.benchmark_run.parameters.get('duration', 60)
        if duration < self.MIN_DURATION_SECONDS:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Duration must be at least {self.MIN_DURATION_SECONDS} seconds",
                parameter="duration",
                expected=f">= {self.MIN_DURATION_SECONDS}",
                actual=duration
            )
        return None
```

### Pattern 2: BenchmarkVerifier Routing
**What:** BenchmarkVerifier creates appropriate checker based on benchmark_type
**When to use:** Centralized validation orchestration
**Example:**
```python
# Source: mlpstorage/rules/verifier.py (existing, needs update)
def _create_rules_checker(self):
    """Create the appropriate rules checker based on mode and benchmark type."""
    if self.mode == "single":
        benchmark_run = self.benchmark_runs[0]
        if benchmark_run.benchmark_type == BENCHMARK_TYPES.training:
            self.rules_checker = TrainingRunRulesChecker(benchmark_run, logger=self.logger)
        elif benchmark_run.benchmark_type == BENCHMARK_TYPES.checkpointing:
            self.rules_checker = CheckpointingRunRulesChecker(benchmark_run, logger=self.logger)
        # ADD: KV Cache and VectorDB routing
        elif benchmark_run.benchmark_type == BENCHMARK_TYPES.kv_cache:
            self.rules_checker = KVCacheRunRulesChecker(benchmark_run, logger=self.logger)
        elif benchmark_run.benchmark_type == BENCHMARK_TYPES.vector_database:
            self.rules_checker = VectorDBRunRulesChecker(benchmark_run, logger=self.logger)
        else:
            raise ValueError(f"Unsupported benchmark type: {benchmark_run.benchmark_type}")
```

### Pattern 3: Issue Creation
**What:** Create Issue objects with appropriate validation level
**When to use:** Every check_* method that finds something
**Example:**
```python
# Source: mlpstorage/rules/issues.py (existing)
@dataclass
class Issue:
    validation: PARAM_VALIDATION  # CLOSED, OPEN, or INVALID
    message: str
    parameter: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    severity: str = "error"

# Usage:
Issue(
    validation=PARAM_VALIDATION.OPEN,
    message="KV Cache benchmark is in preview status",
    parameter="benchmark_status"
)
```

### Pattern 4: MultiRunRulesChecker for Submissions
**What:** Validates groups of runs for submission requirements
**When to use:** When benchmark has specific submission rules (e.g., N runs required)
**Example:**
```python
# Source: mlpstorage/rules/submission_checkers/training.py (existing)
class TrainingSubmissionRulesChecker(MultiRunRulesChecker):
    REQUIRED_RUNS = 5

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

### Anti-Patterns to Avoid
- **Hardcoding benchmark types in multiple places:** Use BENCHMARK_TYPES enum consistently
- **Skipping preview status for KV Cache/VectorDB:** Both should return OPEN or explicit preview status
- **Assuming DLIO result format:** Non-DLIO benchmarks need alternative result parsing
- **Missing check_benchmark_type:** Every checker should verify it's handling the correct type

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Discovering check methods | Manual registration | RulesChecker auto-discovers `check_*` methods | Established pattern in base.py |
| Issue formatting | Custom string building | Issue.\_\_str\_\_() and Issue.to_dict() | Consistent format everywhere |
| Category determination | Custom logic | RulesChecker.get_category() | Already handles INVALID > OPEN > CLOSED precedence |
| BenchmarkRun creation | Manual instantiation | BenchmarkRun.from_result_dir() or from_benchmark() | Factory methods handle parsing |
| Result directory scanning | Custom walk logic | get_runs_files() from utils.py | Already handles metadata and summary detection |

**Key insight:** The framework handles the hard parts. Adding new benchmark types means implementing checkers and wiring them into the verifier.

## Common Pitfalls

### Pitfall 1: BenchmarkVerifier Raises ValueError for New Types
**What goes wrong:** Running `mlpstorage reports reportgen` on KV Cache/VectorDB results raises "Unsupported benchmark type"
**Why it happens:** BenchmarkVerifier._create_rules_checker() only handles training and checkpointing
**How to avoid:** Add routing for BENCHMARK_TYPES.kv_cache and BENCHMARK_TYPES.vector_database
**Warning signs:** "ValueError: Unsupported benchmark type: kv_cache"

### Pitfall 2: ResultFilesExtractor Fails on Non-DLIO Results
**What goes wrong:** BenchmarkRun.from_result_dir() fails when there's no summary.json (DLIO output)
**Why it happens:** DLIOResultParser expects summary.json; KV Cache and VectorDB don't produce it
**How to avoid:** Enhance ResultFilesExtractor._is_complete_metadata() to work with metadata-only results
**Warning signs:** "No summary.json found in ..." error

### Pitfall 3: Missing benchmark_type in Metadata
**What goes wrong:** BenchmarkRun created with None benchmark_type, causing routing failures
**Why it happens:** Metadata doesn't include benchmark_type field or uses wrong value
**How to avoid:** Verify KV Cache and VectorDB benchmarks write correct benchmark_type to metadata
**Warning signs:** benchmark_run.benchmark_type is None in debugger

### Pitfall 4: Multi-run Verification Fails for Mixed Types
**What goes wrong:** User accidentally mixes KV Cache and Training results, gets unhelpful error
**Why it happens:** Multi-run verifier checks all runs are same type but error message is generic
**How to avoid:** Clear error message listing the different types found
**Warning signs:** "Multi-run verification requires all runs are from the same benchmark type"

### Pitfall 5: Preview Status Not Communicated
**What goes wrong:** User submits KV Cache benchmark thinking it's CLOSED, but it's preview-only
**Why it happens:** KVCacheRunRulesChecker returns OPEN for preview status, but user doesn't see clear message
**How to avoid:** Check_preview_status() returns informative Issue; reporting shows this prominently
**Warning signs:** User surprised their KV Cache submission was rejected

## Code Examples

Verified patterns from the existing codebase:

### VectorDBRunRulesChecker (pattern to implement)
```python
# Pattern following mlpstorage/rules/run_checkers/kvcache.py
from typing import Optional, List
from mlpstorage.config import BENCHMARK_TYPES, PARAM_VALIDATION
from mlpstorage.rules.issues import Issue
from mlpstorage.rules.run_checkers.base import RunRulesChecker


class VectorDBRunRulesChecker(RunRulesChecker):
    """Rules checker for VectorDB benchmarks.

    VectorDB benchmark is in preview mode - rules are informational.
    """

    MIN_RUNTIME_SECONDS = 30

    def check_benchmark_type(self) -> Optional[Issue]:
        """Verify this is a VectorDB benchmark."""
        if self.benchmark_run.benchmark_type != BENCHMARK_TYPES.vector_database:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid benchmark type: {self.benchmark_run.benchmark_type}",
                parameter="benchmark_type",
                expected=BENCHMARK_TYPES.vector_database,
                actual=self.benchmark_run.benchmark_type
            )
        return None

    def check_runtime(self) -> Optional[Issue]:
        """Verify benchmark runtime is valid."""
        runtime = self.benchmark_run.parameters.get('runtime', 60)
        if runtime and runtime < self.MIN_RUNTIME_SECONDS:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Runtime must be at least {self.MIN_RUNTIME_SECONDS} seconds",
                parameter="runtime",
                expected=f">= {self.MIN_RUNTIME_SECONDS}",
                actual=runtime
            )
        return None

    def check_preview_status(self) -> Optional[Issue]:
        """Return informational issue that VectorDB is in preview."""
        return Issue(
            validation=PARAM_VALIDATION.OPEN,
            message="VectorDB benchmark is in preview status - not accepted for closed submissions",
            parameter="benchmark_status"
        )
```

### BenchmarkVerifier Update (pattern to implement)
```python
# Source: mlpstorage/rules/verifier.py (to be updated)
from mlpstorage.rules.run_checkers import (
    TrainingRunRulesChecker,
    CheckpointingRunRulesChecker,
    KVCacheRunRulesChecker,
    # NEW:
    VectorDBRunRulesChecker,
)

def _create_rules_checker(self):
    """Create the appropriate rules checker based on mode and benchmark type."""
    if self.mode == "single":
        benchmark_run = self.benchmark_runs[0]
        if benchmark_run.benchmark_type == BENCHMARK_TYPES.training:
            self.rules_checker = TrainingRunRulesChecker(benchmark_run, logger=self.logger)
        elif benchmark_run.benchmark_type == BENCHMARK_TYPES.checkpointing:
            self.rules_checker = CheckpointingRunRulesChecker(benchmark_run, logger=self.logger)
        elif benchmark_run.benchmark_type == BENCHMARK_TYPES.kv_cache:
            self.rules_checker = KVCacheRunRulesChecker(benchmark_run, logger=self.logger)
        elif benchmark_run.benchmark_type == BENCHMARK_TYPES.vector_database:
            self.rules_checker = VectorDBRunRulesChecker(benchmark_run, logger=self.logger)
        else:
            raise ValueError(f"Unsupported benchmark type: {benchmark_run.benchmark_type}")

    elif self.mode == "multi":
        # Multi-run validation for submissions
        benchmark_types = {br.benchmark_type for br in self.benchmark_runs}
        if len(benchmark_types) > 1:
            raise ValueError(
                f"Multi-run verification requires all runs are from the same "
                f"benchmark type. Got types: {benchmark_types}"
            )
        benchmark_type = benchmark_types.pop()

        if benchmark_type == BENCHMARK_TYPES.training:
            self.rules_checker = TrainingSubmissionRulesChecker(self.benchmark_runs, logger=self.logger)
        elif benchmark_type == BENCHMARK_TYPES.checkpointing:
            self.rules_checker = CheckpointSubmissionRulesChecker(self.benchmark_runs, logger=self.logger)
        elif benchmark_type == BENCHMARK_TYPES.kv_cache:
            # KV Cache preview - use basic multi-run checker or dedicated one
            self.rules_checker = MultiRunRulesChecker(self.benchmark_runs, logger=self.logger)
        elif benchmark_type == BENCHMARK_TYPES.vector_database:
            # VectorDB preview - use basic multi-run checker or dedicated one
            self.rules_checker = MultiRunRulesChecker(self.benchmark_runs, logger=self.logger)
        else:
            raise ValueError(f"Unsupported benchmark type for multi-run: {benchmark_type}")
```

### ResultFilesExtractor Enhancement (pattern to implement)
```python
# Source: mlpstorage/rules/models.py (to be updated in _from_metadata)
def _from_metadata(self, metadata: Dict, result_dir: str) -> BenchmarkRunData:
    """Create BenchmarkRunData from a complete metadata dict."""
    benchmark_type_str = metadata.get('benchmark_type', '')
    benchmark_type = None
    for bt in BENCHMARK_TYPES:
        if bt.name == benchmark_type_str or bt.value == benchmark_type_str:
            benchmark_type = bt
            break

    # For non-DLIO benchmarks, parameters may be at top level
    parameters = metadata.get('parameters', {})
    if not parameters:
        # Fallback: use whole metadata as parameters for KV Cache / VectorDB
        parameters = {k: v for k, v in metadata.items()
                      if k not in ['benchmark_type', 'run_datetime', 'result_dir', 'model', 'command']}

    return BenchmarkRunData(
        benchmark_type=benchmark_type,
        model=metadata.get('model'),
        command=metadata.get('command'),
        run_datetime=metadata.get('run_datetime', ''),
        num_processes=metadata.get('num_processes', 1),
        parameters=parameters,
        override_parameters=metadata.get('override_parameters', {}),
        system_info=None,
        metrics=metadata.get('metrics') or metadata.get('kvcache_metrics'),
        result_dir=result_dir,
        accelerator=metadata.get('accelerator'),
    )
```

### ClosedRequirementsFormatter Update (pattern to implement)
```python
# Source: mlpstorage/reporting/formatters.py (to be updated)
class ClosedRequirementsFormatter:
    """Format CLOSED submission requirements as checklists."""

    # ... existing TRAINING_REQUIREMENTS, CHECKPOINTING_REQUIREMENTS, KVCACHE_REQUIREMENTS ...

    VECTORDB_REQUIREMENTS = {
        'title': 'VectorDB Benchmark Requirements (Preview)',
        'requirements': [
            'Minimum runtime of 30 seconds',
            'Valid collection configuration',
            'Database host and port accessible',
            'Note: VectorDB is in preview and not yet accepted for CLOSED submissions',
        ],
        'allowed_params': [],
    }

    @classmethod
    def get_requirements(cls, benchmark_type: str) -> Optional[Dict]:
        """Get requirements for a benchmark type."""
        requirements_map = {
            'training': cls.TRAINING_REQUIREMENTS,
            'checkpointing': cls.CHECKPOINTING_REQUIREMENTS,
            'kv_cache': cls.KVCACHE_REQUIREMENTS,
            'vector_database': cls.VECTORDB_REQUIREMENTS,  # NEW
        }
        return requirements_map.get(benchmark_type)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Only training/checkpointing validation | Four benchmark types | Phase 5 | Full validation coverage |
| DLIO-only result parsing | Generic metadata + DLIO fallback | Phase 5 | Non-DLIO benchmark support |
| No preview status | Explicit OPEN for preview benchmarks | Phase 3-5 | Clear submission guidance |

**Deprecated/outdated:**
- Direct instantiation of RunRulesChecker: Use BenchmarkVerifier which handles routing
- Assuming all benchmarks have summary.json: Check metadata first, fall back to DLIO parsing

## Existing Implementation Status

### Already Implemented (HIGH confidence)
- `RulesChecker` base class with auto-discovery of check_* methods
- `RunRulesChecker` and `MultiRunRulesChecker` base classes
- `TrainingRunRulesChecker` and `CheckpointingRunRulesChecker`
- `TrainingSubmissionRulesChecker` and `CheckpointSubmissionRulesChecker`
- `KVCacheRunRulesChecker` (from Phase 3)
- `BenchmarkVerifier` orchestrator (routes training/checkpointing only)
- `Issue` dataclass with validation levels
- `BenchmarkRun`, `BenchmarkRunData`, `ClusterInformation` models
- `ResultFilesExtractor` and `DLIOResultParser` (DLIO-focused)
- `ReportGenerator` with validation and output
- `ClosedRequirementsFormatter` (has KV Cache, missing VectorDB)
- `ResultsDirectoryValidator` (supports all four benchmark types)

### Not Yet Implemented (needs work)
- `VectorDBRunRulesChecker` - new file needed
- BenchmarkVerifier routing for kv_cache and vector_database types
- Submission checkers for KV Cache and VectorDB (if submission rules exist)
- ResultFilesExtractor enhancement for non-DLIO metadata
- ClosedRequirementsFormatter VectorDB entry
- Export updates in `rules/__init__.py` and `run_checkers/__init__.py`
- Unit tests for VectorDB checker

### Rules Export Structure (from rules/__init__.py)
The rules module exports:
- All run checkers: TrainingRunRulesChecker, CheckpointingRunRulesChecker, KVCacheRunRulesChecker
- All submission checkers: TrainingSubmissionRulesChecker, CheckpointSubmissionRulesChecker
- BenchmarkVerifier
- Utility functions: get_runs_files, generate_output_location

## Open Questions

Things that couldn't be fully resolved:

1. **Submission Rules for KV Cache and VectorDB**
   - What we know: Both are "preview" status and return OPEN
   - What's unclear: Do they have specific submission requirements (number of runs, etc.)?
   - Recommendation: Use base MultiRunRulesChecker for now; add specific checkers when rules are defined

2. **VectorDB Parameters for Validation**
   - What we know: VectorDB has host, port, runtime, config_name parameters
   - What's unclear: What parameters should be validated? What are valid ranges?
   - Recommendation: Start with basic validation (runtime >= 30s, benchmark_type correct, preview status); expand based on user feedback

3. **Metrics Location in Non-DLIO Results**
   - What we know: DLIO puts metrics in summary.json under "metric" key
   - What's unclear: Where do KV Cache and VectorDB store metrics?
   - Recommendation: KV Cache uses `kvcache_metrics` in metadata (Phase 3); VectorDB needs equivalent

4. **Combined Reports with All Benchmark Types**
   - What we know: ReportGenerator groups by (model, accelerator)
   - What's unclear: How should KV Cache and VectorDB be grouped in combined reports?
   - Recommendation: Group by (model, None) for now; model is config_name for VectorDB

## Sources

### Primary (HIGH confidence)
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/rules/verifier.py` - BenchmarkVerifier implementation
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/rules/base.py` - RulesChecker base class
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/rules/run_checkers/kvcache.py` - KVCacheRunRulesChecker pattern
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/rules/run_checkers/training.py` - TrainingRunRulesChecker pattern
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/rules/models.py` - BenchmarkRun and related models
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/rules/issues.py` - Issue dataclass
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/report_generator.py` - ReportGenerator integration
- `/home/wvaske/Projects/mlperf-storage/mlpstorage/reporting/formatters.py` - ClosedRequirementsFormatter

### Secondary (MEDIUM confidence)
- `/home/wvaske/Projects/mlperf-storage/tests/unit/test_rules_checkers.py` - Test patterns
- `/home/wvaske/Projects/mlperf-storage/.planning/phases/03-kv-cache-benchmark-integration/03-VERIFICATION.md` - Phase 3 outcomes
- `/home/wvaske/Projects/mlperf-storage/.planning/phases/04-vectordb-benchmark-integration/04-VERIFICATION.md` - Phase 4 outcomes

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All components are stdlib or already in project
- Architecture: HIGH - Patterns well-established in existing codebase
- Pitfalls: HIGH - Derived from code inspection of existing implementations

**Research date:** 2026-01-24
**Valid until:** 2026-02-24 (stable codebase, 30-day validity)

---

## Quick Implementation Checklist

For the planner to create tasks:

1. [ ] Create `mlpstorage/rules/run_checkers/vectordb.py` with VectorDBRunRulesChecker
2. [ ] Update `mlpstorage/rules/run_checkers/__init__.py` to export VectorDBRunRulesChecker
3. [ ] Update `mlpstorage/rules/__init__.py` to export VectorDBRunRulesChecker
4. [ ] Update `mlpstorage/rules/verifier.py` to route kv_cache and vector_database types
5. [ ] Add import for KVCacheRunRulesChecker in verifier.py (already exists in run_checkers)
6. [ ] Update `mlpstorage/reporting/formatters.py` to add VECTORDB_REQUIREMENTS
7. [ ] Enhance `ResultFilesExtractor._is_complete_metadata()` to handle non-DLIO metadata
8. [ ] Create `tests/unit/test_rules_vectordb.py` following test_rules_checkers.py pattern
9. [ ] Verify `mlpstorage reports reportgen` works on KV Cache results
10. [ ] Verify `mlpstorage reports reportgen` works on VectorDB results
11. [ ] Verify combined reports include all benchmark types
12. [ ] Update directory validator expected structure documentation if needed
