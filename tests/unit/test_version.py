"""Regression tests for version resolution."""
import importlib.metadata
import pathlib
import tomllib

import mlpstorage_py


def test_version_matches_pyproject():
    """VERSION constant must equal the version declared in pyproject.toml."""
    pyproject = pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert mlpstorage_py.VERSION == declared


def test_dist_name_matches_pyproject():
    """The dist name used by _resolve_version must equal pyproject's
    project.name. Catches the regression where the in-code lookup string
    drifts from the declared package name — independent of whether the
    package is currently pip-installed in the test environment."""
    from mlpstorage_py import _DIST_NAME

    pyproject = pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        declared_name = tomllib.load(f)["project"]["name"]
    assert _DIST_NAME == declared_name


def test_version_fallback_reads_pyproject(monkeypatch):
    """When installed metadata is absent, version is read from pyproject.toml."""
    from importlib.metadata import PackageNotFoundError

    def _raise(_name):
        raise PackageNotFoundError(_name)

    monkeypatch.setattr(importlib.metadata, "version", _raise)

    # Re-run the resolver function directly
    from mlpstorage_py import _resolve_version
    result = _resolve_version()

    pyproject = pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert result == declared
