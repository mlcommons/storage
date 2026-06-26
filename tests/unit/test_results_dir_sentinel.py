"""
Unit tests for ``mlpstorage_py.results_dir.sentinel``.

Plan: 01-canonical-layout-and-init / 01-01 — Task 2 (atomic sentinel I/O).

Covers:
- ``write_sentinel(results_dir, orgname)``:
    * writes ``<results-dir>/mlperf-results.yaml`` containing the four
      required keys in the canonical order with correct types,
    * uses atomic exclusive-create (O_EXCL) — a second call raises
      ``DoubleInitError`` and does NOT modify the pre-existing file
      (T-1-01 — TOCTOU mitigation),
    * preserves orgname casing (Rules.md treats orgnames as case-sensitive),
    * writes the file with mode ``0o644`` (V4 ASVS — file permissions).
- ``read_sentinel(results_dir)``:
    * round-trips a freshly-written sentinel back to a
      ``MlperfResultsSentinel`` model,
    * raises ``ResultsDirNotInitializedError`` when the sentinel is absent,
    * raises ``ResultsDirNotInitializedError`` when the sentinel content
      is malformed (missing required field, etc.) — the original
      ``ValidationError`` is chained via ``__cause__``.
- ``resolve_orgname(results_dir)`` returns the orgname as a string —
  stub-level test; full gate integration lands in Slice 4.

Refs: 01-RESEARCH.md Pattern 4 ("Atomic exclusive-create") and
"Code Examples → sentinel write"; 01-PATTERNS.md row
``results_dir/sentinel.py``.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mlpstorage_py.results_dir import (
    MLPERF_RESULTS_FILENAME,
    MLPERF_RESULTS_VERSION,
    DoubleInitError,
    MlperfResultsSentinel,
    ResultsDirNotInitializedError,
    read_sentinel,
    resolve_orgname,
    write_sentinel,
)


def _sentinel_path(results_dir: Path) -> Path:
    return results_dir / MLPERF_RESULTS_FILENAME


# --------------------------------------------------------------------------- #
# write_sentinel — happy path & field shape
# --------------------------------------------------------------------------- #


def test_write_sentinel_returns_path(tmp_path: Path):
    returned = write_sentinel(str(tmp_path), "Acme")
    assert returned == str(_sentinel_path(tmp_path))
    assert _sentinel_path(tmp_path).is_file()


def test_write_sentinel_fields(tmp_path: Path):
    write_sentinel(str(tmp_path), "Acme")
    with open(_sentinel_path(tmp_path)) as fh:
        loaded = yaml.safe_load(fh)

    # Exactly the four documented keys — no extras.
    assert set(loaded.keys()) == {
        "mlperf_results_version",
        "orgname",
        "initialized_at",
        "initialized_by",
    }
    assert loaded["mlperf_results_version"] == MLPERF_RESULTS_VERSION
    assert loaded["orgname"] == "Acme"
    # initialized_at is an ISO-8601-shaped string. Validate roughly: starts
    # with YYYY-MM-DDThh:mm:ss and includes a timezone offset / Z suffix.
    iso = loaded["initialized_at"]
    assert isinstance(iso, str)
    assert re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$",
        iso,
    ), f"initialized_at not ISO-8601 shaped: {iso!r}"
    # initialized_by is "mlpstorage <version>".
    assert re.match(r"^mlpstorage \S+$", loaded["initialized_by"])


def test_write_sentinel_uses_safe_yaml(tmp_path: Path):
    """The serialised sentinel must be parseable by yaml.safe_load — i.e.
    contains no Python-tagged objects (no ``!!python/...``). Guards against
    accidentally using ``yaml.dump`` (which can emit tags) instead of
    ``yaml.safe_dump``.
    """
    write_sentinel(str(tmp_path), "Acme")
    raw = _sentinel_path(tmp_path).read_text()
    assert "!!python" not in raw, f"sentinel contains python tag: {raw!r}"


# --------------------------------------------------------------------------- #
# write_sentinel — atomicity (T-1-01)
# --------------------------------------------------------------------------- #


def test_write_is_atomic_double_init_raises(tmp_path: Path):
    """T-1-01 mitigation — a second ``write_sentinel`` against the same
    results-dir raises ``DoubleInitError`` and does NOT touch the existing
    file (O_EXCL single-syscall create).
    """
    write_sentinel(str(tmp_path), "Acme")
    pre_existing_bytes = _sentinel_path(tmp_path).read_bytes()
    pre_existing_mtime = _sentinel_path(tmp_path).stat().st_mtime_ns

    with pytest.raises(DoubleInitError):
        write_sentinel(str(tmp_path), "OtherOrg")

    # Pre-existing content must be intact: the failed write should not
    # have opened the file at all.
    assert _sentinel_path(tmp_path).read_bytes() == pre_existing_bytes
    assert _sentinel_path(tmp_path).stat().st_mtime_ns == pre_existing_mtime


def test_write_is_atomic_when_sentinel_pre_seeded(tmp_path: Path):
    """A hostile / accidental pre-seeded sentinel file is preserved — the
    O_EXCL flag refuses to overwrite, even though the pre-seeded content is
    not valid YAML at all.
    """
    seeded = b"this is not a valid sentinel\n"
    _sentinel_path(tmp_path).write_bytes(seeded)

    with pytest.raises(DoubleInitError):
        write_sentinel(str(tmp_path), "Acme")

    assert _sentinel_path(tmp_path).read_bytes() == seeded


# --------------------------------------------------------------------------- #
# write_sentinel — orgname casing & file mode
# --------------------------------------------------------------------------- #


def test_write_preserves_orgname_casing(tmp_path: Path):
    write_sentinel(str(tmp_path), "AcMe")
    with open(_sentinel_path(tmp_path)) as fh:
        loaded = yaml.safe_load(fh)
    assert loaded["orgname"] == "AcMe"


def test_sentinel_file_mode_0o644(tmp_path: Path):
    """V4 ASVS — file permissions. The sentinel is world-readable
    (LAY-03 — every command reads orgname from it on multi-user boxes),
    owner-writable, and not executable.
    """
    write_sentinel(str(tmp_path), "Acme")
    mode = os.stat(_sentinel_path(tmp_path)).st_mode
    # Strip the file-type bits, compare the permission bits only.
    assert stat.S_IMODE(mode) == 0o644


# --------------------------------------------------------------------------- #
# read_sentinel — round-trip + error paths
# --------------------------------------------------------------------------- #


def test_write_then_read_round_trip(tmp_path: Path):
    write_sentinel(str(tmp_path), "Acme")
    model = read_sentinel(str(tmp_path))
    assert isinstance(model, MlperfResultsSentinel)
    assert model.orgname == "Acme"
    assert model.mlperf_results_version == MLPERF_RESULTS_VERSION


def test_read_missing_raises_not_initialized(tmp_path: Path):
    with pytest.raises(ResultsDirNotInitializedError) as exc_info:
        read_sentinel(str(tmp_path))
    # The actionable suggestion should reference ``mlpstorage init``.
    assert "init" in (exc_info.value.suggestion or "").lower()


def test_read_malformed_raises_not_initialized(tmp_path: Path):
    """A sentinel that loads as YAML but fails schema validation (e.g.
    missing required field) surfaces as ``ResultsDirNotInitializedError``
    with the underlying ``ValidationError`` chained via ``__cause__``.
    """
    bad = {"mlperf_results_version": 1, "orgname": "Acme"}  # missing keys
    with open(_sentinel_path(tmp_path), "w") as fh:
        yaml.safe_dump(bad, fh)

    with pytest.raises(ResultsDirNotInitializedError) as exc_info:
        read_sentinel(str(tmp_path))
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_read_unparseable_yaml_raises_not_initialized(tmp_path: Path):
    """A file that exists but is unparseable YAML also surfaces as
    ``ResultsDirNotInitializedError`` so the user sees the same actionable
    "run `mlpstorage init`" hint regardless of which way the sentinel is
    broken.
    """
    _sentinel_path(tmp_path).write_text("not: valid: : yaml: : :\n")

    with pytest.raises(ResultsDirNotInitializedError):
        read_sentinel(str(tmp_path))


# --------------------------------------------------------------------------- #
# resolve_orgname — Slice 4 stub
# --------------------------------------------------------------------------- #


def test_resolve_orgname_returns_string(tmp_path: Path):
    write_sentinel(str(tmp_path), "Acme")
    org = resolve_orgname(str(tmp_path))
    assert isinstance(org, str)
    assert org == "Acme"


def test_resolve_orgname_missing_raises_not_initialized(tmp_path: Path):
    with pytest.raises(ResultsDirNotInitializedError):
        resolve_orgname(str(tmp_path))
