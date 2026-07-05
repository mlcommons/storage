"""Shared fixtures for Phase 6 Plan 06-04 pool-layout integration tests.

Consolidates the setup helpers reused across the six `test_pool_*.py` files:

* `MockLogger` — captures `.status/error/warning/info/debug` messages for
  assertion (mirrors the shape in `mlpstorage_py/tests/test_capture_or_verify_pool.py`).
* `_make_capture_args` — constructs a `SimpleNamespace` matching what
  `capture_or_verify_code_image` reads (mode, command, results_dir, orgname,
  systemname, benchmark, model). This helper is exposed as a fixture-returning
  callable so tests can build args on demand.
* `fake_source_root` — stages an isolated source tree with `pyproject.toml` +
  a minimal `mlpstorage_py/` package inside `tmp_path`, and monkeypatches
  `find_source_root` so the capture path hashes THIS tree (not the real
  running-project checkout, which would introduce test noise).
* `init_results_dir` — invokes `run_init` to populate `mlperf-results.yaml`
  under `tmp_path/results` so the LAY-03 sentinel is present.

Scope: `tests/integration/` only. Plan 06-04 introduces this file as SC#12
permits ("extract into a shared `tests/integration/conftest.py` fixture if
one is warranted"). Existing integration tests
(`test_canonical_layout_end_to_end.py`, etc.) continue to use their own
module-local helpers — they are not migrated here in this plan.

Refs: Phase 6 06-04-PLAN.md SC#7, SC#12.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# MockLogger — captures messages for assertion
# ---------------------------------------------------------------------------


class MockLogger:
    """Test double that captures every message call site into a list per level.

    Shape matches the logger contract that `capture_or_verify_code_image`
    consumes: `.status`, `.error`, `.warning`, `.info`, `.debug`, plus the
    Phase-1 verbose levels (unused but required for compatibility with the
    real logger).
    """

    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.debugs: list[str] = []

    def status(self, msg, *args):
        self.statuses.append(msg % args if args else msg)

    def error(self, msg, *args):
        self.errors.append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)

    def debug(self, msg, *args):
        self.debugs.append(msg % args if args else msg)

    # Phase 1 verbose levels — kept for logger-contract parity
    def verbose(self, msg, *args): pass
    def verboser(self, msg, *args): pass
    def ridiculous(self, msg, *args): pass


@pytest.fixture
def log() -> MockLogger:
    """Fresh `MockLogger` per test."""
    return MockLogger()


# ---------------------------------------------------------------------------
# Args builder — SimpleNamespace shape matching main._main_impl:224
# ---------------------------------------------------------------------------


def make_capture_args(
    *,
    results_dir: Path | str,
    mode: str,
    orgname: str,
    command: str = "run",
    benchmark: str = "training",
    model: str = "unet3d",
    systemname: str | None = None,
    skip_validation: bool = False,
) -> Namespace:
    """Build an `argparse.Namespace` matching `capture_or_verify_code_image`'s contract.

    Mirrors the shape `main._main_impl` assembles at call-time (main.py:224).
    `systemname` is only required for OPEN mode; caller populates it
    explicitly for OPEN tests.

    HARDEN-03: `orgname` is populated as an args attribute (LAY-03 hook),
    NOT via env — closes the trust-contract regression.
    """
    ns = Namespace(
        mode=mode,
        command=command,
        results_dir=str(results_dir),
        benchmark=benchmark,
        model=model,
        orgname=orgname,
        systemname=systemname,
        skip_validation=skip_validation,
    )
    return ns


@pytest.fixture
def capture_args_factory():
    """Return the `make_capture_args` builder for tests that need multiple args.

    Usage:
        def test_x(capture_args_factory, tmp_path):
            args = capture_args_factory(results_dir=tmp_path, mode="closed", orgname="Acme")
    """
    return make_capture_args


# ---------------------------------------------------------------------------
# Fake source tree — deterministic hash across capture and verify
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_source_root(tmp_path, monkeypatch):
    """Stage a minimal source tree with `pyproject.toml` and monkeypatch
    `find_source_root` to return it.

    The staged tree contains:
        <tmp_path>/src_root/pyproject.toml
        <tmp_path>/src_root/mlpstorage_py/__init__.py   (with __version__)
        <tmp_path>/src_root/mlpstorage_py/stub.py       (initial content)

    This isolates the test from the running-project checkout: the
    md5-tree-v2 hash depends only on `src_root` contents, so mutating this
    tree changes the hash deterministically across test invocations.

    Returns the source-root `Path` so tests can mutate the tree (add files,
    etc.) to trigger source-change scenarios.

    Ref: PATTERNS `### fake_source_root` (mirrors
    `mlpstorage_py/tests/test_capture_or_verify_pool.py:87-99`).
    """
    src = tmp_path / "src_root"
    src.mkdir()
    (src / "pyproject.toml").write_text("[project]\nname = 'x'\nversion='0.0.1'\n")
    (src / "mlpstorage_py").mkdir()
    (src / "mlpstorage_py" / "__init__.py").write_text("__version__ = '0.0.1'\n")
    (src / "mlpstorage_py" / "stub.py").write_text("X = 1\n")
    monkeypatch.setattr(
        "mlpstorage_py.submission_checker.tools.code_image.find_source_root",
        lambda: src,
    )
    return src


# ---------------------------------------------------------------------------
# Initialized results dir — writes mlperf-results.yaml sentinel
# ---------------------------------------------------------------------------


@pytest.fixture
def init_results_dir(tmp_path):
    """Populate `<tmp_path>/results/mlperf-results.yaml` via `run_init`.

    Returns a callable `init(orgname="Acme") -> Path` so tests can pick an
    orgname (some tests need distinct orgs per call for per-org isolation
    coverage). Directory `<tmp_path>/results/` is created by `run_init`.

    Ref: `tests/integration/test_canonical_layout_end_to_end.py::_init_results_dir`.
    """
    from mlpstorage_py.results_dir import run_init

    def _init(orgname: str = "Acme") -> Path:
        rd = tmp_path / "results"
        args = Namespace(mode="init", orgname=orgname, path=str(rd))
        run_init(args)
        sentinel = rd / "mlperf-results.yaml"
        assert sentinel.is_file(), (
            f"run_init must write sentinel at {sentinel}"
        )
        return rd

    return _init


# ---------------------------------------------------------------------------
# Pool-directory helper (used across every pool test file)
# ---------------------------------------------------------------------------


def pool_dirs(org_root: Path) -> list[Path]:
    """Return sorted list of `code-*` pool dirs under `org_root`.

    Empty list when `org_root` does not exist. Mirrors the helper in
    `mlpstorage_py/tests/test_capture_or_verify_pool.py::_pool_dirs`.
    """
    if not org_root.is_dir():
        return []
    return sorted(org_root.glob("code-*"))
