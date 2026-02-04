"""Progress indication utilities using Rich library.

This module provides progress indication utilities that automatically detect
interactive vs non-interactive terminals and adjust behavior accordingly.

In interactive terminals, Rich progress bars and spinners are displayed.
In non-interactive terminals (CI, logs), status messages are logged instead.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from logging import Logger

# Type aliases for yielded functions
UpdateFunc = Callable[..., None]
SetDescriptionFunc = Callable[[str], None]
AdvanceStageFunc = Callable[..., None]


def is_interactive_terminal() -> bool:
    """Detect if running in an interactive terminal.

    Returns:
        True if output is to an interactive terminal, False otherwise.
    """
    console = Console()
    return console.is_terminal


@contextmanager
def progress_context(
    description: str,
    total: Optional[int] = None,
    logger: Optional["Logger"] = None,
    transient: bool = True,
) -> Iterator[Tuple[UpdateFunc, SetDescriptionFunc]]:
    """Context manager for progress indication with automatic TTY detection.

    In interactive terminals, displays a Rich progress bar/spinner.
    In non-interactive mode, logs status via the provided logger.

    Args:
        description: Initial description text for the progress indicator.
        total: Total count for determinate progress (e.g., 100 for percentage).
            If None, shows indeterminate spinner.
        logger: Logger instance for non-interactive mode status messages.
            If None in non-interactive mode, no output is produced.
        transient: If True, progress is cleared when complete (default True).

    Yields:
        Tuple of (update_func, set_description_func):
            - update_func(advance=1, completed=None): Advances progress
            - set_description_func(desc): Updates description text

    Example:
        >>> with progress_context("Processing files", total=100) as (update, set_desc):
        ...     for i in range(100):
        ...         process_file(i)
        ...         update()  # Advances by 1
        ...     set_desc("Complete!")
    """
    if not is_interactive_terminal():
        # Non-interactive: log status and provide no-op functions
        if logger is not None:
            logger.status(f"{description}...")

        def noop_update(advance: int = 1, completed: Optional[int] = None) -> None:
            """No-op update function for non-interactive mode."""
            pass

        def noop_set_description(desc: str) -> None:
            """No-op set_description function for non-interactive mode."""
            pass

        yield (noop_update, noop_set_description)
        return

    # Interactive: create Rich Progress
    if total is None:
        # Indeterminate: spinner only
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
        ]
    else:
        # Determinate: full progress bar with percentage
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]

    progress = Progress(*columns, transient=transient)
    task_id: TaskID = TaskID(0)  # Will be set after add_task

    try:
        progress.start()
        task_id = progress.add_task(description, total=total)

        def update_func(advance: int = 1, completed: Optional[int] = None) -> None:
            """Update progress by advancing or setting completed value."""
            if completed is not None:
                progress.update(task_id, completed=completed)
            else:
                progress.update(task_id, advance=advance)

        def set_description_func(desc: str) -> None:
            """Update the progress description text."""
            progress.update(task_id, description=desc)

        yield (update_func, set_description_func)
    finally:
        progress.stop()


@contextmanager
def create_stage_progress(
    stages: List[str],
    logger: Optional["Logger"] = None,
    transient: bool = True,
) -> Iterator[AdvanceStageFunc]:
    """Context manager for multi-stage operations with progress indication.

    Displays progress through a sequence of named stages (e.g., "Validating",
    "Collecting", "Running", "Processing").

    In interactive terminals, shows a progress bar advancing through stages.
    In non-interactive mode, logs each stage via the provided logger.

    Args:
        stages: List of stage names to progress through.
        logger: Logger instance for non-interactive mode status messages.
        transient: If True, progress is cleared when complete (default True).

    Yields:
        advance_stage(stage_name=None) function:
            - If stage_name is None, advances to the next stage in sequence
            - If stage_name is provided, updates description and advances

    Example:
        >>> stages = ["Validating", "Collecting", "Running", "Processing"]
        >>> with create_stage_progress(stages) as advance_stage:
        ...     validate()
        ...     advance_stage()  # Now at "Collecting"
        ...     collect()
        ...     advance_stage("Running tests")  # Custom description
        ...     run_tests()
    """
    if not stages:
        # Empty stages: provide no-op and return
        def noop_advance(stage_name: Optional[str] = None) -> None:
            pass

        yield noop_advance
        return

    if not is_interactive_terminal():
        # Non-interactive: log each stage
        current_stage_idx = 0

        if logger is not None:
            logger.status(f"Stage 1/{len(stages)}: {stages[0]}...")

        def advance_stage_noninteractive(stage_name: Optional[str] = None) -> None:
            nonlocal current_stage_idx
            current_stage_idx += 1
            if current_stage_idx < len(stages):
                if logger is not None:
                    desc = stage_name if stage_name else stages[current_stage_idx]
                    logger.status(
                        f"Stage {current_stage_idx + 1}/{len(stages)}: {desc}..."
                    )

        yield advance_stage_noninteractive
        return

    # Interactive: show progress bar through stages
    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ]

    progress = Progress(*columns, transient=transient)
    task_id: TaskID = TaskID(0)
    current_stage_idx = 0

    try:
        progress.start()
        task_id = progress.add_task(stages[0], total=len(stages), completed=1)

        def advance_stage_interactive(stage_name: Optional[str] = None) -> None:
            nonlocal current_stage_idx
            current_stage_idx += 1
            if current_stage_idx < len(stages):
                desc = stage_name if stage_name else stages[current_stage_idx]
                progress.update(task_id, advance=1, description=desc)

        yield advance_stage_interactive
    finally:
        progress.stop()


__all__ = [
    "is_interactive_terminal",
    "progress_context",
    "create_stage_progress",
]
