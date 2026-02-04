# Phase 10: Progress Indication - Research

**Researched:** 2026-01-25
**Domain:** CLI Progress Indication, User Experience
**Confidence:** HIGH

## Summary

This research investigates how to add clear progress feedback during long-running operations in the MLPerf Storage benchmark suite. The codebase currently uses a custom logging system (`mlpstorage/mlps_logging.py`) with custom log levels (STATUS, VERBOSE, VERBOSER, etc.) and ANSI color support. There is no progress bar or spinner functionality implemented.

The research identified **Rich** as the recommended library for progress indication. Rich is already in the dependency tree (via transitive dependencies as seen in `uv.lock`), provides excellent TTY detection, handles non-interactive output gracefully, and integrates well with existing logging patterns. The codebase has three main categories of long-running operations: cluster information collection, benchmark execution (DLIO-based), and data generation - each requiring different progress indication strategies.

**Primary recommendation:** Use Rich library for progress indication with automatic TTY detection, falling back to simple log messages for non-interactive terminals. Add progress to mlpstorage-owned operations while respecting that DLIO has its own progress output.

## Current State Analysis

### Existing Output Patterns

The codebase uses a custom logging system (`mlpstorage/mlps_logging.py`):

```python
# Custom log levels
RESULT = 35      # Green, for results
STATUS = 25      # Blue, for status messages
INFO = 20        # Normal
VERBOSE = 19     # Detailed info
DEBUG = 10       # Debug info
```

Key logger methods used throughout:
- `logger.status()` - Blue colored status messages (25 files use this)
- `logger.info()` - Standard information
- `logger.verbose()` - Detailed information (lower priority)
- `logger.warning()` / `logger.error()` - Warnings and errors

### Long-Running Operations Identified

**1. Cluster Information Collection** (`mlpstorage/cluster_collector.py`)
- MPI-based collection: 60-second default timeout, collects from multiple hosts
- SSH-based collection: Parallel SSH to multiple hosts with ThreadPoolExecutor
- Time-series collection: Background thread collecting every 10 seconds
- Location: `base.py::_collect_cluster_start()`, `_collect_cluster_end()`

**2. Benchmark Execution** (`mlpstorage/benchmarks/dlio.py`)
- Training benchmarks: `TrainingBenchmark._run()` calls `execute_command()`
- Checkpointing benchmarks: `CheckpointingBenchmark._run()`
- Data generation: Part of the benchmark execution via DLIO
- Note: DLIO already has its own progress output (lines 302-314 in dlio_benchmark/utils/utility.py)

**3. VectorDB Operations** (`mlpstorage/benchmarks/vectordbbench.py`)
- Data generation: `execute_datagen()` via `load-vdb` command
- Benchmark run: `execute_run()` via `vdbbench` command

**4. KV Cache Operations** (`mlpstorage/benchmarks/kvcache.py`)
- Benchmark run with duration-based execution (default 60s)
- Already has status messages: "Running KV Cache benchmark for {duration}s..."

### Command Execution Pattern

All benchmarks use `CommandExecutor` (`mlpstorage/utils.py`) which:
- Streams stdout/stderr in real-time via `select`
- Uses subprocess with `PIPE` for output capture
- Supports signal handling for graceful termination
- Runs MPI commands that have their own output

## Standard Stack

### Core Library

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Rich | 14.x | Progress bars, spinners, terminal detection | Already in dependency tree, excellent TTY handling, beautiful output |

### Alternative Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Rich | tqdm | tqdm is faster (~60ns vs ~good), but Rich has better TTY detection and integrates with console output |
| Rich | progressbar2 | progressbar2 has logging integration but higher overhead (~800ns), less modern API |
| Rich | Click progressbar | Click is CLI-focused but typer/click docs recommend Rich for advanced progress |

**Decision:** Use Rich. It's already a transitive dependency, has superior terminal detection, and provides spinners for indeterminate progress.

### Installation

```bash
# Rich is likely already installed as a transitive dependency
# Add to explicit dependencies if not:
pip install rich>=13.0
```

Add to `pyproject.toml` dependencies:
```toml
dependencies = [
    # ... existing deps ...
    "rich>=13.0",  # Progress indication (UX-04)
]
```

## Architecture Patterns

### Recommended Project Structure

```
mlpstorage/
├── progress.py              # NEW: Progress indication utilities
├── mlps_logging.py          # Existing: Logging (minimal changes)
├── utils.py                 # Existing: CommandExecutor (add progress support)
└── benchmarks/
    └── base.py              # Add progress to cluster collection phases
```

### Pattern 1: Progress Context Manager with TTY Detection

**What:** Wrapper around Rich Progress that auto-disables for non-TTY output
**When to use:** All progress indication points

```python
# Source: Rich official docs + custom adaptation
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console
import sys

class ProgressManager:
    """Context manager for progress indication with TTY detection."""

    def __init__(self, logger=None, force_interactive=None):
        self.logger = logger
        self.console = Console(force_terminal=force_interactive)
        self.is_interactive = self.console.is_terminal if force_interactive is None else force_interactive
        self._progress = None

    def __enter__(self):
        if self.is_interactive:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,  # Clear progress when done
            )
            self._progress.start()
        return self

    def __exit__(self, *args):
        if self._progress:
            self._progress.stop()

    def add_task(self, description, total=None):
        """Add a task. Returns task ID or None if non-interactive."""
        if self._progress:
            return self._progress.add_task(description, total=total or 0)
        elif self.logger:
            self.logger.status(description)
        return None

    def update(self, task_id, advance=None, completed=None, description=None):
        """Update task progress."""
        if self._progress and task_id is not None:
            kwargs = {}
            if advance is not None:
                kwargs['advance'] = advance
            if completed is not None:
                kwargs['completed'] = completed
            if description is not None:
                kwargs['description'] = description
            self._progress.update(task_id, **kwargs)
```

### Pattern 2: Spinner for Indeterminate Operations

**What:** Animated spinner for operations without known duration
**When to use:** Cluster collection, environment validation, initialization

```python
# Source: Rich official docs
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

def create_spinner_progress(console):
    """Create spinner for indeterminate progress."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )

# Usage
with create_spinner_progress(console) as progress:
    task = progress.add_task("Collecting cluster information...", total=None)
    # Do work...
    progress.update(task, description="Collecting from host1...")
```

### Pattern 3: Stage Indicators

**What:** Clear stage transitions during multi-phase operations
**When to use:** Benchmark lifecycle (validation -> collection -> run -> results)

```python
class BenchmarkStages:
    """Stage indicator for benchmark execution."""

    STAGES = [
        ("validating", "Validating environment..."),
        ("collecting", "Collecting cluster information..."),
        ("running", "Running benchmark..."),
        ("processing", "Processing results..."),
        ("writing", "Writing metadata..."),
    ]

    def __init__(self, progress_manager):
        self.pm = progress_manager
        self.current_stage = 0
        self.task = None

    def start(self):
        """Initialize stage tracking."""
        if self.pm.is_interactive:
            self.task = self.pm.add_task(
                self.STAGES[0][1],
                total=len(self.STAGES)
            )

    def advance(self, stage_name=None):
        """Move to next stage."""
        self.current_stage += 1
        if self.current_stage < len(self.STAGES):
            _, desc = self.STAGES[self.current_stage]
            self.pm.update(self.task, advance=1, description=desc)
        elif self.pm.logger:
            self.pm.logger.status(f"Stage: {stage_name or 'complete'}")
```

### Pattern 4: Non-Interactive Fallback

**What:** Graceful degradation for CI/logs/redirected output
**When to use:** Everywhere progress is shown

```python
# Source: Rich docs + tqdm best practices
def show_progress(logger, description, is_interactive):
    """Show progress appropriately based on terminal type."""
    if is_interactive:
        # Rich spinner/progress bar
        pass
    else:
        # Simple log message
        logger.status(description)
```

### Anti-Patterns to Avoid

- **Mixing Rich and raw print:** Use `progress.console.print()` instead of `print()` when progress is active to avoid visual corruption
- **Not checking TTY before animations:** Always use `console.is_terminal` or `force_interactive` parameter
- **Progress for DLIO output:** Don't wrap DLIO command execution in progress - DLIO has its own progress output
- **Blocking progress updates:** Progress updates in long inner loops can slow execution; batch updates

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Terminal detection | `os.isatty()` checks scattered | `Console(force_interactive=...)` | Rich handles edge cases (Windows, proxies, CI) |
| Spinner animation | Custom print-based spinners | `SpinnerColumn()` | Thread-safe, proper terminal handling |
| Time estimates | Manual elapsed/remaining calc | `TimeElapsedColumn`, `TimeRemainingColumn` | Accurate, formatted output |
| Progress bar rendering | Custom bar strings | `BarColumn()` | Terminal width aware, Unicode/ASCII fallback |
| Output redirection | Manual stdout/stderr handling | `Progress(redirect_stdout=True)` | Proper interleaving with progress display |

**Key insight:** Terminal handling is surprisingly complex (CI environments, Windows, proxies, SSH sessions). Rich handles all these edge cases; custom solutions invariably miss some.

## Common Pitfalls

### Pitfall 1: Progress Spam in Non-Interactive Mode
**What goes wrong:** Progress bars print many lines when output is redirected
**Why it happens:** Not checking `is_terminal` before showing animated progress
**How to avoid:** Always use `Console().is_terminal` check; fall back to single log lines
**Warning signs:** CI logs full of progress line spam

### Pitfall 2: Progress Interfering with Subprocess Output
**What goes wrong:** DLIO's own progress output gets mangled with Rich progress
**Why it happens:** Both writing to same terminal simultaneously
**How to avoid:** Don't show Rich progress during DLIO command execution; let DLIO's output flow through
**Warning signs:** Garbled terminal output, overlapping progress bars

### Pitfall 3: Progress Not Clearing on Exit/Exception
**What goes wrong:** Progress bar remains on screen after error
**Why it happens:** Not using context manager or not calling `stop()` in finally
**How to avoid:** Always use `with Progress() as progress:` pattern
**Warning signs:** Terminal in corrupted state after errors

### Pitfall 4: Blocking Main Thread with Progress Updates
**What goes wrong:** Benchmark slows down due to progress update overhead
**Why it happens:** Updating progress on every small operation
**How to avoid:** Batch updates, update every N iterations or every second
**Warning signs:** Significant performance difference with/without progress

### Pitfall 5: Hardcoded Terminal Assumptions
**What goes wrong:** Progress works locally but fails in CI/Docker
**Why it happens:** Assuming terminal capabilities not checking environment
**How to avoid:** Respect `NO_COLOR`, `TERM=dumb`, `CI` environment variables
**Warning signs:** CI pipeline failures, blank output in containers

## Code Examples

### Example 1: Progress Manager Module

```python
# Source: Rich documentation + custom patterns
# mlpstorage/progress.py

from contextlib import contextmanager
from typing import Optional, Callable, Any
import sys

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskID,
)


def is_interactive_terminal() -> bool:
    """Check if output is to an interactive terminal."""
    console = Console()
    return console.is_terminal


@contextmanager
def progress_context(
    description: str = "Processing...",
    total: Optional[int] = None,
    logger=None,
    transient: bool = True,
):
    """Context manager for progress indication.

    Args:
        description: Initial description text
        total: Total steps (None for indeterminate/spinner)
        logger: Logger for non-interactive fallback
        transient: Clear progress when done

    Yields:
        Tuple of (update_func, set_description_func) or (None, None) if non-interactive
    """
    console = Console()

    if not console.is_terminal:
        # Non-interactive: just log and yield no-ops
        if logger:
            logger.status(description)
        yield lambda *args, **kwargs: None, lambda *args, **kwargs: None
        return

    # Interactive: create Rich progress
    columns = [SpinnerColumn()]
    columns.append(TextColumn("[progress.description]{task.description}"))

    if total is not None:
        columns.extend([
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ])

    columns.append(TimeElapsedColumn())

    with Progress(*columns, console=console, transient=transient) as progress:
        task_id = progress.add_task(description, total=total)

        def update(advance: int = 1, completed: Optional[int] = None):
            if completed is not None:
                progress.update(task_id, completed=completed)
            else:
                progress.update(task_id, advance=advance)

        def set_description(desc: str):
            progress.update(task_id, description=desc)

        yield update, set_description
```

### Example 2: Integration in Cluster Collection

```python
# Source: Pattern for base.py integration
def _collect_cluster_start(self) -> None:
    """Collect cluster information with progress indication."""
    from mlpstorage.progress import progress_context

    if not self._should_collect_cluster_info():
        return

    hosts = self.args.hosts if hasattr(self.args, 'hosts') else []
    desc = f"Collecting cluster info ({len(hosts)} hosts)..."

    with progress_context(desc, total=None, logger=self.logger) as (update, set_desc):
        if self._should_use_ssh_collection():
            set_desc("Collecting via SSH...")
            self._cluster_info_start = self._collect_via_ssh()
        else:
            set_desc("Collecting via MPI...")
            self._cluster_info_start = self._collect_cluster_information()
```

### Example 3: Benchmark Lifecycle Stages

```python
# Source: Pattern for run() method enhancement
def run(self) -> int:
    """Execute benchmark with stage indicators."""
    from mlpstorage.progress import progress_context

    stages = [
        "Validating environment...",
        "Collecting cluster info...",
        "Running benchmark...",
        "Processing results...",
    ]

    with progress_context("Benchmark execution", total=len(stages), logger=self.logger) as (update, set_desc):
        # Stage 1: Validation
        set_desc(stages[0])
        self._validate_environment()
        update()

        # Stage 2: Cluster collection
        set_desc(stages[1])
        self._collect_cluster_start()
        self._start_timeseries_collection()
        update()

        # Stage 3: Benchmark (no progress here - DLIO has its own)
        set_desc(stages[2])
        result = self._run()
        update()

        # Stage 4: Cleanup
        set_desc(stages[3])
        self._stop_timeseries_collection()
        self._collect_cluster_end()
        self.write_timeseries_data()
        update()

    return result
```

## Integration Points

### Key Files to Modify

1. **`mlpstorage/progress.py`** (NEW)
   - Progress context managers
   - TTY detection utilities
   - Spinner and progress bar factories

2. **`mlpstorage/benchmarks/base.py`**
   - `run()` method: Add stage indicators
   - `_collect_cluster_start/end()`: Add spinner
   - Keep benchmark execution without wrapping (let DLIO output flow)

3. **`mlpstorage/utils.py`**
   - `CommandExecutor.execute()`: Optional status callback for long commands

4. **`mlpstorage/main.py`**
   - `run_benchmark()`: Could add overall progress wrapper
   - Environment validation: Add spinner

5. **`pyproject.toml`**
   - Add `rich>=13.0` to dependencies

### What NOT to Modify

- DLIO execution output - DLIO has its own progress bars
- Log formatting in `mlps_logging.py` - Keep separate concerns
- Core benchmark timing logic - Progress is presentation layer only

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No progress indication | Rich Progress bars | This phase | User experience improvement |
| Print-based spinners | Rich SpinnerColumn | 2023+ | Thread-safe, proper terminal handling |
| Manual TTY checks | Console.is_terminal | Rich 10+ | Handles edge cases automatically |
| Separate progress libs | Rich (unified) | 2022+ | Single dependency for console output |

**Current state:**
- Rich is the de-facto standard for Python CLI progress indication (2024-2026)
- tqdm remains popular for Jupyter/notebook use cases
- progressbar2 is legacy but stable

## Environment Variables

Rich respects these environment variables (document for users):

| Variable | Effect |
|----------|--------|
| `NO_COLOR` | Disables all color output |
| `FORCE_COLOR` | Forces color even for non-TTY |
| `TERM=dumb` | Disables colors and progress animations |
| `TTY_COMPATIBLE=1` | Indicates TTY support in CI |
| `TTY_INTERACTIVE=0` | Disables animations in CI |

For CI environments like GitHub Actions:
```bash
export TTY_COMPATIBLE=1
export TTY_INTERACTIVE=0
```

## Open Questions

1. **DLIO Progress Integration**
   - What we know: DLIO has its own progress output for data generation
   - What's unclear: Whether DLIO progress can be suppressed or redirected
   - Recommendation: Leave DLIO progress alone; only add progress to mlpstorage-owned operations

2. **MPI Rank 0 Only**
   - What we know: DLIO only shows progress on rank 0
   - What's unclear: Whether mlpstorage progress should follow same pattern
   - Recommendation: Yes, only show progress on primary process when running MPI

3. **Progress During SSH Collection**
   - What we know: SSH collection uses ThreadPoolExecutor
   - What's unclear: Best way to show per-host progress
   - Recommendation: Use single spinner with description updates, not per-host progress bars

## Sources

### Primary (HIGH confidence)
- Rich official documentation: https://rich.readthedocs.io/en/latest/progress.html - Progress bar API, TTY handling
- Rich Console API: https://rich.readthedocs.io/en/latest/console.html - Terminal detection, force_interactive

### Secondary (MEDIUM confidence)
- DataCamp Python Progress Bar Guide: https://www.datacamp.com/tutorial/progress-bars-in-python - Library comparison
- Medium: From Tqdm to Rich: https://medium.com/pythoneers/from-tqdm-to-rich-my-quest-for-better-progress-bars-afff39985ffc - Migration patterns
- Timothy Gebhard: Richer progress bars: https://timothygebhard.de/posts/richer-progress-bars-for-rich/ - Custom columns

### Tertiary (LOW confidence)
- GitHub discussions on Rich/logging integration
- Stack Overflow patterns for progress in CLI tools

## Metadata

**Confidence breakdown:**
- Standard stack (Rich): HIGH - Official docs, already in dependency tree
- Architecture patterns: HIGH - Rich docs provide clear patterns
- Integration points: MEDIUM - Requires code analysis of edge cases
- DLIO interaction: MEDIUM - DLIO has own progress, need to avoid conflicts

**Research date:** 2026-01-25
**Valid until:** 2026-07-25 (Rich is stable, patterns unlikely to change)
