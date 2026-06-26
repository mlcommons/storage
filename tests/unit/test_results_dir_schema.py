"""
Unit tests for ``mlpstorage_py.results_dir.schema`` and
``mlpstorage_py.results_dir.errors``.

Plan: 01-canonical-layout-and-init / 01-01 — sentinel infrastructure foundation.

Covers:
- Public-API import surface of the ``mlpstorage_py.results_dir`` package.
- Pydantic v2 ``MlperfResultsSentinel`` schema:
    * accepts a well-formed payload,
    * forbids unknown keys (``extra='forbid'``),
    * rejects empty ``orgname`` and orgnames containing path separators / control
      chars / NUL bytes (T-1-03 hardening),
    * rejects ``mlperf_results_version < 1`` (``ge=1``),
    * rejects payloads missing a required field.
- ``validate_dict`` returns a model on valid input and re-raises ``ValidationError``
  on malformed input (mirrors ``schema_validator.validate_dict`` shape).
- ``validate_file`` reads YAML via ``yaml.safe_load`` (V12 ASVS — never
  ``yaml.load``) and returns a validated model; nonexistent paths raise a clear
  ``OSError``.
- Domain errors ``ResultsDirNotInitializedError``, ``DoubleInitError``,
  ``NonEmptyDirError`` subclass ``ConfigurationError`` (and therefore
  ``MLPStorageException``), and each accepts ``suggestion=`` keyword.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mlpstorage_py.errors import ConfigurationError, ErrorCode, MLPStorageException
from mlpstorage_py.results_dir import (
    MLPERF_RESULTS_FILENAME,
    MLPERF_RESULTS_VERSION,
    DoubleInitError,
    MlperfResultsSentinel,
    NonEmptyDirError,
    ResultsDirNotInitializedError,
)
from mlpstorage_py.results_dir.schema import validate_dict, validate_file


GOOD_PAYLOAD = {
    "mlperf_results_version": 1,
    "orgname": "Acme",
    "initialized_at": "2026-06-18T12:00:00+00:00",
    "initialized_by": "mlpstorage 2.0.0b1",
}


# --------------------------------------------------------------------------- #
# Package-level constants & API surface
# --------------------------------------------------------------------------- #


def test_filename_constant():
    assert MLPERF_RESULTS_FILENAME == "mlperf-results.yaml"


def test_version_constant():
    assert MLPERF_RESULTS_VERSION == 1


def test_package_exports_smoke():
    # Round-trip the documented public surface (drift check vs PLAN.md
    # <artifacts_this_phase_produces>).
    from mlpstorage_py import results_dir as pkg

    for symbol in (
        "MLPERF_RESULTS_FILENAME",
        "MLPERF_RESULTS_VERSION",
        "MlperfResultsSentinel",
        "ResultsDirNotInitializedError",
        "DoubleInitError",
        "NonEmptyDirError",
        "resolve_orgname",
        "write_sentinel",
        "read_sentinel",
    ):
        assert hasattr(pkg, symbol), f"package missing public export: {symbol}"


# --------------------------------------------------------------------------- #
# MlperfResultsSentinel — happy path
# --------------------------------------------------------------------------- #


def test_model_accepts_good_payload():
    model = MlperfResultsSentinel.model_validate(GOOD_PAYLOAD)
    assert model.mlperf_results_version == 1
    assert model.orgname == "Acme"
    assert model.initialized_at == "2026-06-18T12:00:00+00:00"
    assert model.initialized_by == "mlpstorage 2.0.0b1"


def test_model_preserves_orgname_casing():
    # Rules.md treats orgnames as case-sensitive; no normalization.
    model = MlperfResultsSentinel.model_validate({**GOOD_PAYLOAD, "orgname": "AcMe"})
    assert model.orgname == "AcMe"


# --------------------------------------------------------------------------- #
# MlperfResultsSentinel — validation rejections
# --------------------------------------------------------------------------- #


def test_model_rejects_unknown_keys():
    # extra='forbid' — guards against typo'd field names.
    bad = {**GOOD_PAYLOAD, "extra_key": "x"}
    with pytest.raises(ValidationError) as exc_info:
        MlperfResultsSentinel.model_validate(bad)
    assert any(err["type"] == "extra_forbidden" for err in exc_info.value.errors())


def test_model_rejects_empty_orgname():
    bad = {**GOOD_PAYLOAD, "orgname": ""}
    with pytest.raises(ValidationError):
        MlperfResultsSentinel.model_validate(bad)


@pytest.mark.parametrize(
    "bad_orgname",
    [
        "../etc",           # path traversal
        "foo/bar",          # path separator
        "a\x00b",           # NUL byte
        "with space",       # space — not in pattern
        "control\x01char",  # control char
        "back\\slash",      # backslash separator
    ],
)
def test_model_rejects_orgname_with_dangerous_chars(bad_orgname):
    """T-1-03 mitigation — Pydantic Field pattern rejects path separators,
    NUL bytes, control chars, etc. so a crafted orgname cannot escape the
    results-dir at write time (RESEARCH.md Security Domain V5).
    """
    bad = {**GOOD_PAYLOAD, "orgname": bad_orgname}
    with pytest.raises(ValidationError):
        MlperfResultsSentinel.model_validate(bad)


def test_model_rejects_version_below_one():
    bad = {**GOOD_PAYLOAD, "mlperf_results_version": 0}
    with pytest.raises(ValidationError):
        MlperfResultsSentinel.model_validate(bad)


def test_model_rejects_missing_required_field():
    bad = {k: v for k, v in GOOD_PAYLOAD.items() if k != "initialized_at"}
    with pytest.raises(ValidationError):
        MlperfResultsSentinel.model_validate(bad)


def test_model_rejects_empty_initialized_at():
    bad = {**GOOD_PAYLOAD, "initialized_at": ""}
    with pytest.raises(ValidationError):
        MlperfResultsSentinel.model_validate(bad)


def test_model_rejects_empty_initialized_by():
    bad = {**GOOD_PAYLOAD, "initialized_by": ""}
    with pytest.raises(ValidationError):
        MlperfResultsSentinel.model_validate(bad)


# --------------------------------------------------------------------------- #
# validate_dict / validate_file helpers
# --------------------------------------------------------------------------- #


def test_validate_dict_returns_model_on_good_input():
    model = validate_dict(GOOD_PAYLOAD)
    assert isinstance(model, MlperfResultsSentinel)
    assert model.orgname == "Acme"


def test_validate_dict_raises_on_bad_input():
    with pytest.raises(ValidationError):
        validate_dict({**GOOD_PAYLOAD, "orgname": ""})


def test_validate_file_reads_yaml_and_returns_model(tmp_path: Path):
    path = tmp_path / "mlperf-results.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(GOOD_PAYLOAD, fh, default_flow_style=False, sort_keys=False)
    model = validate_file(str(path))
    assert isinstance(model, MlperfResultsSentinel)
    assert model.orgname == "Acme"


def test_validate_file_raises_on_nonexistent_path(tmp_path: Path):
    path = tmp_path / "does-not-exist.yaml"
    with pytest.raises((OSError, FileNotFoundError)):
        validate_file(str(path))


def test_validate_file_raises_validation_error_on_malformed_content(tmp_path: Path):
    path = tmp_path / "mlperf-results.yaml"
    bad = {**GOOD_PAYLOAD, "orgname": ""}
    with open(path, "w") as fh:
        yaml.safe_dump(bad, fh)
    with pytest.raises(ValidationError):
        validate_file(str(path))


# --------------------------------------------------------------------------- #
# Domain error classes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "exc_cls",
    [ResultsDirNotInitializedError, DoubleInitError, NonEmptyDirError],
)
def test_errors_subclass_configuration_error(exc_cls):
    exc = exc_cls("a message")
    assert isinstance(exc, ConfigurationError)
    assert isinstance(exc, MLPStorageException)


@pytest.mark.parametrize(
    "exc_cls",
    [ResultsDirNotInitializedError, DoubleInitError, NonEmptyDirError],
)
def test_errors_accept_suggestion_kwarg(exc_cls):
    exc = exc_cls("a message", suggestion="do the right thing")
    assert exc.suggestion == "do the right thing"


def test_results_dir_not_initialized_default_code():
    exc = ResultsDirNotInitializedError("missing sentinel")
    assert exc.code is ErrorCode.CONFIG_MISSING_REQUIRED


def test_double_init_default_code():
    exc = DoubleInitError("already initialized")
    assert exc.code is ErrorCode.CONFIG_INVALID_VALUE


def test_non_empty_dir_default_code():
    exc = NonEmptyDirError("dir not empty")
    assert exc.code is ErrorCode.CONFIG_INVALID_VALUE
