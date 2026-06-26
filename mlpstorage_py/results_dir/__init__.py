"""
``mlpstorage_py.results_dir`` — sentinel infrastructure for the canonical
results-directory layout.

This package owns the ``<results-dir>/mlperf-results.yaml`` sentinel file
and the helpers every downstream consumer uses to read or write it.

Public surface (consumed by Slices 2-5 in this phase, and by every command
that takes ``--results-dir`` once the gate lands in Slice 4):

- Schema:
    * ``MlperfResultsSentinel`` — Pydantic v2 model
    * ``validate_dict`` / ``validate_file`` helpers
- Persistence:
    * ``write_sentinel(results_dir, orgname) -> str``  — atomic create
    * ``read_sentinel(results_dir) -> MlperfResultsSentinel`` — validated read
    * ``resolve_orgname(results_dir) -> str``  — thin wrapper
- Errors:
    * ``ResultsDirNotInitializedError``
    * ``DoubleInitError``
    * ``NonEmptyDirError``
- Constants:
    * ``MLPERF_RESULTS_FILENAME``  = ``"mlperf-results.yaml"``
    * ``MLPERF_RESULTS_VERSION``  = ``1``

The constants live in this module to keep the import graph acyclic: both
``schema.py`` and ``sentinel.py`` import-free of each other can pull them
from the package root. ``sentinel.py`` (Task 2) imports the constants from
here and registers ``write_sentinel`` / ``read_sentinel`` / ``resolve_orgname``
back through this module's re-exports.

Imports are kept light at the package level (no MPI, pyarrow, or DLIO) so
this module is cheap to import from CLI entrypoints and tests.

Refs: 01-canonical-layout-and-init / 01-01-PLAN.md
"""

# --- Constants -------------------------------------------------------------- #
# Single source of truth for the sentinel filename and current schema version.
# These are exposed at the package root so callers can ``from
# mlpstorage_py.results_dir import MLPERF_RESULTS_FILENAME`` without pulling in
# ``sentinel.py`` (which depends on ``mlpstorage_py.VERSION`` resolution).

MLPERF_RESULTS_FILENAME: str = "mlperf-results.yaml"
MLPERF_RESULTS_VERSION: int = 1

# --- Errors ----------------------------------------------------------------- #
from mlpstorage_py.results_dir.errors import (  # noqa: E402
    DoubleInitError,
    NonEmptyDirError,
    ResultsDirNotInitializedError,
)

# --- Schema ----------------------------------------------------------------- #
from mlpstorage_py.results_dir.schema import (  # noqa: E402
    MlperfResultsSentinel,
    validate_dict,
    validate_file,
)

# --- Persistence ------------------------------------------------------------ #
# ``sentinel.py`` is loaded last so it can ``from . import
# MLPERF_RESULTS_FILENAME, MLPERF_RESULTS_VERSION`` without circular-import
# pain. The re-exports below give callers a single import target.
#
# Import is wrapped in a try/except to keep the package importable during
# bootstrap phases where ``sentinel.py`` does not yet exist (Task 1 stand-up
# in 01-01-PLAN.md adds schema + errors first; Task 2 adds sentinel). In
# steady state the import succeeds and the three symbols are re-exported.
try:  # pragma: no cover — bootstrap-only fallback path
    from mlpstorage_py.results_dir.sentinel import (  # noqa: E402
        read_sentinel,
        resolve_orgname,
        write_sentinel,
    )
except ImportError:  # pragma: no cover
    pass

# --- Init dispatcher -------------------------------------------------------- #
# Re-exported so downstream callers (and tests) can `from
# mlpstorage_py.results_dir import run_init` without knowing the module layout.
try:  # pragma: no cover — bootstrap-only fallback path
    from mlpstorage_py.results_dir.init import run_init  # noqa: E402
except ImportError:  # pragma: no cover
    pass

# --- Code-image capture (LAY-06) ------------------------------------------- #
# Re-exported so ``Benchmark.__init__`` (and tests) can
# ``from mlpstorage_py.results_dir import capture_code_image``.
try:  # pragma: no cover — bootstrap-only fallback path
    from mlpstorage_py.results_dir.code_image import capture_code_image  # noqa: E402
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "MLPERF_RESULTS_FILENAME",
    "MLPERF_RESULTS_VERSION",
    "MlperfResultsSentinel",
    "validate_dict",
    "validate_file",
    "write_sentinel",
    "read_sentinel",
    "resolve_orgname",
    "run_init",
    "capture_code_image",
    "ResultsDirNotInitializedError",
    "DoubleInitError",
    "NonEmptyDirError",
]
