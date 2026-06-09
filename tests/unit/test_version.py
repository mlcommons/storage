from pathlib import Path
import sys
import tomllib

import pytest


def _pyproject_version() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    with (repo_root / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_package_version_matches_pyproject():
    import mlpstorage_py

    expected = _pyproject_version()
    assert mlpstorage_py.VERSION == expected
    assert mlpstorage_py.__version__ == expected
    assert mlpstorage_py.VERSION != "unknown"


def test_version_from_pyproject_matches_project_version():
    import mlpstorage_py

    assert mlpstorage_py._version_from_pyproject() == _pyproject_version()


def test_resolve_version_uses_distribution_name_when_pyproject_missing(monkeypatch):
    import mlpstorage_py

    seen = {}

    def fake_pkg_version(name: str) -> str:
        seen["name"] = name
        return "9.8.7"

    monkeypatch.setattr(mlpstorage_py, "_version_from_pyproject", lambda: None)
    monkeypatch.setattr(mlpstorage_py, "_pkg_version", fake_pkg_version)

    assert mlpstorage_py._resolve_version() == "9.8.7"
    assert seen["name"] == "mlpstorage"


def test_resolve_version_returns_unknown_when_no_metadata(monkeypatch):
    import mlpstorage_py

    def missing_distribution(name: str) -> str:
        raise mlpstorage_py._PkgNF(name)

    monkeypatch.setattr(mlpstorage_py, "_version_from_pyproject", lambda: None)
    monkeypatch.setattr(mlpstorage_py, "_pkg_version", missing_distribution)

    assert mlpstorage_py._resolve_version() == "unknown"


def test_cli_version_prints_project_version(monkeypatch, capsys):
    from mlpstorage_py.cli_parser import parse_arguments

    monkeypatch.setattr(sys, "argv", ["mlpstorage", "--version"])

    with pytest.raises(SystemExit) as exc:
        parse_arguments()

    assert exc.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out == f"mlpstorage {_pyproject_version()}"
    assert "unknown" not in out
