"""Wall-clock bookends of the mlpstorage process.

``INVOCATION_START`` is captured at first import of this module. ``main.py``
imports this before any heavier ``mlpstorage_py`` submodule so the timestamp
is recorded before framework startup (Python imports, MPI spawn, CAP env-
validation, etc.). ``benchmarks/base.py`` reads the value when populating
``invocation_start_time`` in the run's metadata; the submission checker uses
that field as the read-side origin of the §4.7.1 gap check so per-invocation
framework overhead is not charged against the 30-second failover-callout
budget.

``INVOCATION_END`` is the symmetric write-side bookend. ``benchmarks/base.py``
calls :func:`mark_invocation_end` at the top of ``write_metadata()`` — the
latest moment where the timestamp can still land in the persisted file, and
the closest we can get to rank 0's actual exit. The value is emitted as
``invocation_end_time`` in the write-phase metadata; the submission checker
uses it as the write-side origin of §4.7.1. Symmetric with the read-side
capture-at-first-import: everything before ``INVOCATION_START`` on the read
side (framework startup — Python imports, MPI spawn, CAP env-validation) is
excluded from the callout gap, and everything after ``INVOCATION_END`` on
the write side (post-benchmark cluster collection, rank-0 teardown
processing, JSON serialization, interpreter shutdown) is excluded too. See
mlcommons/storage#782.
"""
import time

INVOCATION_START: float = time.time()
INVOCATION_END: float | None = None


def mark_invocation_end() -> float:
    """Record the current wall-clock instant as the invocation-end bookend.

    Idempotent: subsequent calls overwrite the recorded value so the last
    call before ``write_metadata()`` wins. Returns the recorded value.
    """
    global INVOCATION_END
    INVOCATION_END = time.time()
    return INVOCATION_END
