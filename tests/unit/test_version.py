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


def test_version_lookup_uses_correct_distribution_name():
    """importlib.metadata lookup must succeed under the 'mlpstorage' dist name."""
    # Will raise PackageNotFoundError (not caught) if wrong name is used
    pkg_version = importlib.metadata.version("mlpstorage")
    assert pkg_version == mlpstorage_py.VERSION


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
