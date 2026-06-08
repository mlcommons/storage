# Plan: `version` Command and Version Resolution Fix

## Goal

Add `mlpstorage version` as a top-level subcommand (sibling of `reports`, `history`,
`lockfile`) and simultaneously fix the version resolution bug in `__init__.py` — the
current code looks up the wrong distribution name (`mlpstorage_py`) instead of the
name declared in `pyproject.toml` (`mlpstorage`).

---

## Bug: wrong distribution name

`mlpstorage_py/__init__.py` line 3:

```python
VERSION = _pkg_version("mlpstorage_py")   # WRONG
```

`pyproject.toml` declares `name = "mlpstorage"`.  When the package is installed via
`pip install -e .`, importlib.metadata registers the distribution as `mlpstorage`.
A lookup for `mlpstorage_py` raises `PackageNotFoundError`, so `VERSION` silently
falls back to `"unknown"` on any installed copy.

---

## Changes required

### 1. `mlpstorage_py/__init__.py` — fix distribution name + add pyproject.toml fallback

```python
from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNF
import pathlib
import tomllib   # stdlib since Python 3.11; project requires >=3.12

def _resolve_version() -> str:
    # Primary: installed distribution metadata (correct dist name is "mlpstorage")
    try:
        return _pkg_version("mlpstorage")
    except _PkgNF:
        pass
    # Fallback: parse pyproject.toml for source-checkout usage
    _pyproject = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    try:
        with open(_pyproject, "rb") as _f:
            return tomllib.load(_f)["project"]["version"]
    except Exception:
        return "unknown"

VERSION = _resolve_version()
__version__ = VERSION
```

Key points:
- `tomllib` is stdlib in Python 3.11+; no extra dependency needed (project requires >=3.12).
- `Path(__file__).parent.parent` resolves to the repo root from `mlpstorage_py/`.
- Fallback chain: installed metadata → pyproject.toml → "unknown".
- `VERSION` and `__version__` stay in sync (both assigned from the same call).

### 2. `mlpstorage_py/cli/utility_args.py` — add `add_version_arguments()`

```python
def add_version_arguments(parser, is_closed):
    """Add version command arguments to the parser.

    No subcommands or flags — the command prints the version string and exits.
    """
    pass   # argparse parser with no arguments; action handled in main.py
```

The actual version printing is handled in `main.py` (or `cli_parser.py`) by checking
`args.program == "version"` and calling `print(VERSION); sys.exit(0)`.

### 3. `mlpstorage_py/cli/__init__.py` — export `add_version_arguments`

Add to imports:
```python
from mlpstorage_py.cli.utility_args import (
    add_reports_arguments,
    add_history_arguments,
    add_version_arguments,      # new
)
```

Add to `__all__`:
```python
'add_version_arguments',
```

### 4. `mlpstorage_py/cli_parser.py` — wire in version subparser

In `parse_arguments()`, alongside `reports_parsers` and `history_parsers`:

```python
version_parsers = sub_programs.add_parser(
    "version",
    description="Print the mlpstorage package version",
    help="Show installed package version"
)
```

Add to `sub_programs_map`:
```python
'version': version_parsers,
```

Call:
```python
add_version_arguments(version_parsers, is_closed)
```

In `validate_args()`, add:
```python
if args.program == 'version':
    return   # nothing to validate
```

In `main.py` (or wherever programs dispatch), add early-exit handling:
```python
if args.program == "version":
    from mlpstorage_py import VERSION
    print(VERSION)
    sys.exit(0)
```

### 5. `tests/unit/test_version.py` — three regression tests

```python
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
```

---

## Acceptance criteria

| # | Criterion |
|---|-----------|
| 1 | `mlpstorage version` prints `3.0.2` (or current pyproject.toml version) and exits 0 |
| 2 | `mlpstorage --help` lists `version` as a valid subcommand |
| 3 | `from mlpstorage_py import VERSION` returns the pyproject.toml version, not `"unknown"` |
| 4 | `importlib.metadata.version("mlpstorage")` returns the same string as `VERSION` |
| 5 | `test_version_matches_pyproject` passes |
| 6 | `test_version_lookup_uses_correct_distribution_name` passes |
| 7 | `test_version_fallback_reads_pyproject` passes |
| 8 | No new runtime dependencies introduced (`tomllib` is stdlib) |
