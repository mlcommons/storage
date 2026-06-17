"""
Atomic reservation of per-run result directories.

Run directories are stamped with second-resolution timestamps. Two runs
started within the same wall-clock second would otherwise collide and
clobber each other's metadata. The helpers here use exclusive create
(O_EXCL via os.mkdir) and on collision bump the timestamp one second
forward, retrying up to a bounded budget.

Kept dependency-free so tests can exercise the logic without pulling in
the rest of the Benchmark class (which imports pyarrow and MPI).
"""

from __future__ import annotations

import datetime as _datetime
import os
from typing import Callable

RUN_DATETIME_FORMAT = "%Y%m%d_%H%M%S"
DEFAULT_COLLISION_BUMP_BUDGET = 10


def bump_datetime_one_second(run_datetime: str) -> str:
    """Return ``run_datetime`` advanced by one second, preserving format."""
    parsed = _datetime.datetime.strptime(run_datetime, RUN_DATETIME_FORMAT)
    return (parsed + _datetime.timedelta(seconds=1)).strftime(RUN_DATETIME_FORMAT)


def reserve_run_directory(
    initial_run_datetime: str,
    path_for_datetime: Callable[[str], str],
    budget: int = DEFAULT_COLLISION_BUMP_BUDGET,
) -> tuple[str, str]:
    """Atomically reserve a unique run directory.

    Args:
        initial_run_datetime: Starting timestamp string (YYYYMMDD_HHMMSS).
        path_for_datetime: Callable that returns the full result-dir path for
            a given timestamp string. Lets the caller plug in their own layout
            (training/<model>/<command>/<datetime>/, etc.) without this helper
            having to know it.
        budget: Max number of one-second forward bumps before giving up.

    Returns:
        (reserved_path, final_run_datetime) — the caller should store
        ``final_run_datetime`` because metadata/timeseries filenames derive
        from it and must match the reserved directory.

    Raises:
        RuntimeError: If the budget is exhausted without finding a free slot.
    """
    run_datetime = initial_run_datetime
    last_parent = ""
    for _ in range(budget):
        candidate = path_for_datetime(run_datetime)
        last_parent = os.path.dirname(candidate)
        if last_parent:
            os.makedirs(last_parent, exist_ok=True)
        try:
            os.mkdir(candidate)
        except FileExistsError:
            run_datetime = bump_datetime_one_second(run_datetime)
            continue
        return candidate, run_datetime

    # Use the parent captured during the loop instead of calling
    # path_for_datetime again — callers may use a side-effecting closure
    # (e.g. Benchmark._reserve_run_directory mutates self.run_datetime),
    # and re-invoking would clobber that state.
    raise RuntimeError(
        f"Could not reserve a unique run directory after {budget} attempts "
        f"starting from {initial_run_datetime}; clean stale entries under "
        f"{last_parent} and retry."
    )
