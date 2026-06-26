"""
Atomic ``mlperf-results.yaml`` sentinel I/O.

This module owns the create-once / read-many lifecycle of the sentinel file
that pins an organisation name to a results-dir. Every later phase consumes
this API:

- Slice 2 (``mlpstorage init`` CLI) calls ``write_sentinel``.
- Slice 4 (orgname-resolution gate) calls ``read_sentinel`` / ``resolve_orgname``
  on every command that takes ``--results-dir``.

Security
--------

T-1-01 (Tampering / Race) — TOCTOU on sentinel write:
    The classic pattern of "stat; if absent → write" leaves a race window in
    which two concurrent processes both observe absence and both succeed at
    writing, with the second silently clobbering the first. We close that
    window with a single ``os.open(path, O_CREAT | O_EXCL | O_WRONLY, 0o644)``
    syscall — the kernel guarantees that exactly one caller succeeds and any
    other receives ``FileExistsError`` (which we surface as ``DoubleInitError``).

T-1-S1 — YAML deserialisation RCE:
    All YAML I/O uses ``yaml.safe_dump`` / ``yaml.safe_load`` only. The unsafe
    loader / dumper variants are never imported in this module.

T-1-S2 — Information disclosure (file mode):
    The sentinel is written ``0o644`` — owner-writable, world-readable. This
    is intentional: LAY-03 says every command must read the orgname, including
    on shared multi-user boxes (see CONTEXT.md + RESEARCH.md Security Domain
    "Information disclosure" row).

Refs: 01-canonical-layout-and-init / 01-01-PLAN.md Task 2;
01-RESEARCH.md Pattern 4 + "Code Examples → sentinel write";
01-PATTERNS.md row ``results_dir/sentinel.py``.
"""

from __future__ import annotations

import datetime
import os
from typing import Final

import yaml
from pydantic import ValidationError

from mlpstorage_py import VERSION
from mlpstorage_py.results_dir import (
    MLPERF_RESULTS_FILENAME,
    MLPERF_RESULTS_VERSION,
)
from mlpstorage_py.results_dir.errors import (
    DoubleInitError,
    ResultsDirNotInitializedError,
)
from mlpstorage_py.results_dir.schema import (
    MlperfResultsSentinel,
    validate_file,
)

# Permission mode for the sentinel file. World-readable on purpose (LAY-03).
_SENTINEL_MODE: Final[int] = 0o644


def _sentinel_path(results_dir: str) -> str:
    return os.path.join(results_dir, MLPERF_RESULTS_FILENAME)


def write_sentinel(results_dir: str, orgname: str) -> str:
    """Atomically create ``<results_dir>/mlperf-results.yaml``.

    Args:
        results_dir: Existing directory in which to write the sentinel.
            Caller is responsible for ensuring the directory exists — this
            function does not ``mkdir`` (the ``init`` command in Slice 2
            performs the D-09 parent-existence check before calling here).
        orgname: Organisation name to pin to this results-dir. Must pass
            ``MlperfResultsSentinel.orgname`` validation (alphanumeric +
            ``._-``, ``min_length=1``).

    Returns:
        The absolute / relative path of the freshly-written sentinel file.

    Raises:
        DoubleInitError: A file already exists at the sentinel path. The
            existing file is NOT modified — exclusive-create (``O_EXCL``)
            guarantees that the failed write never opened the target.
        pydantic.ValidationError: The supplied ``orgname`` (or one of the
            stamped fields) fails schema validation — caught early so we
            do not even attempt the filesystem write. This is a programmer
            error path; user-supplied orgname is also validated upstream
            in the ``init`` command.
    """
    payload = {
        "mlperf_results_version": MLPERF_RESULTS_VERSION,
        "orgname": orgname,
        # ``datetime.timezone.utc`` keeps the timestamp portable and
        # explicit ("…+00:00"); ``isoformat`` returns an ISO-8601 string.
        "initialized_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "initialized_by": f"mlpstorage {VERSION}",
    }

    # Validate before opening the FD — this catches a malformed orgname
    # without creating an empty file we'd have to clean up. The model
    # itself is discarded; we serialise the original dict so the on-disk
    # field order matches the canonical sentinel layout.
    MlperfResultsSentinel.model_validate(payload)

    sentinel_path = _sentinel_path(results_dir)

    # T-1-01 — atomic exclusive create. Kernel-level race-free.
    try:
        fd = os.open(
            sentinel_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            _SENTINEL_MODE,
        )
    except FileExistsError as exc:
        raise DoubleInitError(
            f"results-dir {results_dir!r} is already initialized "
            f"({sentinel_path} exists)",
            suggestion=(
                "Use a fresh path, or remove the existing "
                "mlperf-results.yaml if you intend to re-initialize."
            ),
        ) from exc

    # ``os.fdopen`` adopts the fd; closing the file object closes the fd.
    # ``sort_keys=False`` + ``default_flow_style=False`` give us the
    # canonical, block-style YAML in the same order as ``payload`` above.
    # WR-04: pin ``encoding="utf-8"`` so the write side is locale-independent —
    # mirrors the explicit encoding on the read side in ``schema.validate_path``.
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, default_flow_style=False, sort_keys=False)

    return sentinel_path


def read_sentinel(results_dir: str) -> MlperfResultsSentinel:
    """Validate and return the sentinel under ``results_dir``.

    Wraps both "file missing" and "file malformed" into the same
    ``ResultsDirNotInitializedError`` so the user always sees the same
    actionable hint regardless of which way the sentinel is broken.

    Raises:
        ResultsDirNotInitializedError: The sentinel is missing, unparseable,
            or fails schema validation. When the underlying cause is a
            ``ValidationError`` or ``yaml.YAMLError`` it is chained via
            ``__cause__`` for debug-level inspection.
    """
    sentinel_path = _sentinel_path(results_dir)

    if not os.path.isfile(sentinel_path):
        raise ResultsDirNotInitializedError(
            f"results-dir {results_dir!r} has not been initialized.",
            suggestion=(
                f"Run `mlpstorage init <orgname> {results_dir}` first."
            ),
        )

    try:
        return validate_file(sentinel_path)
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        # Chain the underlying parser/validator error for debug context.
        raise ResultsDirNotInitializedError(
            f"results-dir {results_dir!r} sentinel "
            f"({MLPERF_RESULTS_FILENAME}) is malformed or incomplete.",
            suggestion=(
                f"Re-initialize with `mlpstorage init <orgname> "
                f"{results_dir}` after removing the broken sentinel."
            ),
        ) from exc


def resolve_orgname(results_dir: str) -> str:
    """Return the orgname pinned to ``results_dir``.

    Thin wrapper around ``read_sentinel`` — kept as a distinct symbol so
    Slice 4 (orgname-resolution gate in ``main._main_impl``) imports a
    single targeted name. Same error contract as ``read_sentinel``.
    """
    return read_sentinel(results_dir).orgname
