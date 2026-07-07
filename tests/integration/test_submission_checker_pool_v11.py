"""Integration tests for Phase 8 success criteria (SC-1..SC-4).

Exercises the complete submission checker run() function against v1.1 pool
layout trees, covering:

  SC-1: valid v1.1 tree → run() returns 0
  SC-2: missing/dangling .mlps-code-image → run() returns 1 with CHECK-01 message
  SC-3: modified pool image / renamed pool dir → run() returns 1 with CHECK-02 message
  SC-4: orphan pool image / legacy code/ dir → run() returns 1 with CHECK-03/04 message

Strategy: The run() function also invokes SubmissionStructureCheck,
SystemYamlSchemaCheck, DirectoryCheck, and a full TrainingCheck suite (many
of which require 6 run timestamps, results.json, dlio_config/, etc.). These
checks are orthogonal to Phase 8's pool structure. To keep the integration
tests focused on pool behavior and maintainable:

  - SubmissionStructureCheck and SystemYamlSchemaCheck are monkeypatched to
    no-op (return True) — they already have their own test coverage.
  - MODE_TO_CHECKERS is monkeypatched to an empty checker list so the full
    TrainingCheck battery does not run against the minimal v1.1 fixtures.
  - PoolStructureCheck is never monkeypatched — it runs for real in all tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
import pytest

from mlpstorage_py.submission_checker.main import run
from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5
from mlpstorage_py.submission_checker.tools.code_image import (
    _pool_dir_name,
    _read_hash_file,
    _write_pointer_atomic,
)


# ---------------------------------------------------------------------------
# Logger helpers
# ---------------------------------------------------------------------------

class _MockLog:
    """Minimal logger routing through Python's logging so caplog captures it."""

    def __init__(self, name: str = "test_pool_v11"):
        self._log = logging.getLogger(name)

    def error(self, msg, *args): self._log.error(msg, *args)
    def warning(self, msg, *args): self._log.warning(msg, *args)
    def info(self, msg, *args): self._log.info(msg, *args)
    def debug(self, msg, *args): self._log.debug(msg, *args)
    def status(self, msg, *args): self._log.info(msg, *args)


# ---------------------------------------------------------------------------
# Pool image builder
# ---------------------------------------------------------------------------

def _build_pool_image(
    org_root: Path,
    content: str = "[project]\nname='x'\nversion='0.0.1'\n",
    mlpstorage_version: str = "1.0.0",
    log=None,
) -> tuple[Path, str]:
    """Build a valid pool image dir under org_root.

    Returns (pool_dir, full_hash).

    Steps: (a) create staging dir with content, (b) hash it, (c) create pool
    dir with _pool_dir_name, (d) write content into pool dir, (e) stamp
    .code-hash.json with all required fields.
    """
    if log is None:
        log = _MockLog()

    stage = org_root / "_stage_tmp"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "pyproject.toml").write_text(content)

    full_hash = compute_code_tree_md5(str(stage), log)

    pool_dir = org_root / _pool_dir_name(full_hash)
    pool_dir.mkdir(parents=True, exist_ok=True)
    (pool_dir / "pyproject.toml").write_text(content)

    hash_data = {
        "hash": full_hash,
        "algorithm": "md5-tree-v2",
        "captured_at": "2026-01-01T00:00:00Z",
        "mlpstorage_version": mlpstorage_version,
        "git_sha": None,
    }
    (pool_dir / ".code-hash.json").write_text(json.dumps(hash_data, indent=2) + "\n")

    shutil.rmtree(str(stage), ignore_errors=True)

    return pool_dir, full_hash


# ---------------------------------------------------------------------------
# v1.1 tree factory (returns a callable)
# ---------------------------------------------------------------------------

class _V11TreeFactory:
    """Callable factory that builds valid v1.1 submission trees under tmp_path."""

    def __init__(self, tmp_path: Path):
        self._tmp_path = tmp_path

    def __call__(
        self,
        orgname: str = "Acme",
        n_run_leaves: int = 1,
        mode: str = "closed",
        pool_content: str = "[project]\nname='x'\nversion='0.0.1'\n",
        mlpstorage_version: str = "1.0.0",
    ) -> dict:
        """Build a minimal v1.1 submission tree.

        Returns dict with keys:
          root: submission root Path
          pool_dir: Path to code-<hash8>/ pool image
          run_leaves: list of run leaf Paths (datetime dirs)
          sentinel: Path to .mlps-image-pool
          hash_data: dict from .code-hash.json
          full_hash: str pool image's full MD5 hash
          org_root: Path to <root>/<orgname>/
        """
        log = _MockLog()
        root = self._tmp_path / "submission"
        root.mkdir(parents=True, exist_ok=True)

        # Pool root with sentinel
        org_root = root / orgname
        org_root.mkdir(parents=True, exist_ok=True)
        sentinel = org_root / ".mlps-image-pool"
        sentinel.write_text("2026-01-01T00:00:00Z\n")

        # Build pool image
        pool_dir, full_hash = _build_pool_image(
            org_root, content=pool_content, mlpstorage_version=mlpstorage_version, log=log
        )
        hash_data = json.loads((pool_dir / ".code-hash.json").read_text())

        # Run leaf dirs under closed/<org>/results/sys1/training/unet3d/run/<ts>/
        run_leaves = []
        for i in range(n_run_leaves):
            ts = f"20260101_12000{i}"
            leaf = (
                root / mode / orgname / "results" / "sys1"
                / "training" / "unet3d" / "run" / ts
            )
            leaf.mkdir(parents=True, exist_ok=True)
            (leaf / "output.txt").write_text(f"run {i}\n")
            _write_pointer_atomic(leaf, full_hash, log)
            run_leaves.append(leaf)

        # Minimal systems/ structure: sys1.yaml + sys1.pdf
        systems_dir = root / mode / orgname / "systems"
        systems_dir.mkdir(parents=True, exist_ok=True)
        (systems_dir / "sys1.yaml").write_text(
            "system_under_test:\n"
            "  solution:\n"
            "    submission_name: TestSys\n"
            "    friendly_description: Test\n"
            "    architecture:\n"
            "      storage_location: remote\n"
            "      benchmark_API: file\n"
            "      product_API: file\n"
            "      client_footprint: open_source\n"
            "      client_installation: in_box\n"
            "    capabilities:\n"
            "      multi_host: true\n"
            "      simultaneous_write: true\n"
            "      simultaneous_read: true\n"
            "      remap_time_in_seconds: 0\n"
            "  deployment: onprem\n"
            "  product_nodes: []\n"
        )
        (systems_dir / "sys1.pdf").write_bytes(b"%PDF-1.0 minimal")

        return {
            "root": root,
            "pool_dir": pool_dir,
            "run_leaves": run_leaves,
            "sentinel": sentinel,
            "hash_data": hash_data,
            "full_hash": full_hash,
            "org_root": org_root,
        }


@pytest.fixture
def v11_tree_factory(tmp_path):
    """Return a callable that builds valid v1.1 submission trees."""
    return _V11TreeFactory(tmp_path)


# ---------------------------------------------------------------------------
# Shared run() invocation helper
# ---------------------------------------------------------------------------

def _run_args(root: Path, tmp_path: Path, version: str = "v3.0") -> argparse.Namespace:
    """Build the argparse.Namespace that run() expects."""
    return argparse.Namespace(
        input=str(root),
        version=version,
        submitters=None,
        csv=str(tmp_path / "out.csv"),
        skip_output_file=True,
    )


# ===========================================================================
# TestPhase8SuccessCriteria
# ===========================================================================

class TestPhase8SuccessCriteria:
    """ROADMAP Phase 8 success criteria SC-1..SC-4 integration tests.

    Each test method exercises run() against a v1.1 tree. Non-pool checks
    (SubmissionStructureCheck, SystemYamlSchemaCheck) are monkeypatched to
    no-op so the test focuses on pool behavior. The training checker list
    in MODE_TO_CHECKERS is emptied so the full TrainingCheck battery does
    not run against the minimal v1.1 fixtures.
    """

    def _patch_non_pool_checks(self, monkeypatch):
        """Monkeypatch SubmissionStructureCheck and SystemYamlSchemaCheck to no-op."""
        from mlpstorage_py.submission_checker.checks import submission_structure_checks
        from mlpstorage_py.submission_checker.checks import system_yaml_schema_checks

        # Make SubmissionStructureCheck.__call__ always return True
        original_ssc_call = submission_structure_checks.SubmissionStructureCheck.__call__

        def _noop_ssc(self):
            return True

        monkeypatch.setattr(
            submission_structure_checks.SubmissionStructureCheck,
            "__call__",
            _noop_ssc,
        )
        # Make SystemYamlSchemaCheck.__call__ always return True
        monkeypatch.setattr(
            system_yaml_schema_checks.SystemYamlSchemaCheck,
            "__call__",
            lambda self: True,
        )

    def _patch_mode_checkers(self, monkeypatch):
        """Monkeypatch MODE_TO_CHECKERS to an empty training checker list."""
        import mlpstorage_py.submission_checker.main as main_mod
        monkeypatch.setattr(
            main_mod,
            "MODE_TO_CHECKERS",
            {"training": []},
        )

    # -------------------------------------------------------------------
    # SC-1: valid v1.1 tree → run() returns 0
    # -------------------------------------------------------------------

    def test_sc1_valid_v11_tree_passes(
        self, tmp_path, v11_tree_factory, monkeypatch, caplog
    ):
        """SC-1: well-formed v1.1 tree returns exit code 0."""
        self._patch_non_pool_checks(monkeypatch)
        self._patch_mode_checkers(monkeypatch)

        tree = v11_tree_factory()
        args = _run_args(tree["root"], tmp_path)

        with caplog.at_level(logging.DEBUG):
            rc = run(args)

        # Check no code-image related errors in the log
        code_image_errors = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR
            and any(kw in r.getMessage() for kw in [
                "CHECK-01", "CHECK-02", "CHECK-03", "CHECK-04",
                "mlps-code-image", "pool image", "orphan", "Legacy code/",
            ])
        ]
        assert rc == 0, (
            f"SC-1: expected exit code 0 for valid tree, got {rc}. "
            f"Pool errors: {[r.getMessage() for r in code_image_errors]}"
        )

    # -------------------------------------------------------------------
    # SC-2: missing .mlps-code-image pointer → run() returns 1
    # -------------------------------------------------------------------

    def test_sc2_missing_pointer_returns_1(
        self, tmp_path, v11_tree_factory, monkeypatch, caplog
    ):
        """SC-2a: run leaf with no .mlps-code-image → exit code 1, error names run leaf."""
        self._patch_non_pool_checks(monkeypatch)
        self._patch_mode_checkers(monkeypatch)

        tree = v11_tree_factory()
        # Delete the pointer from the run leaf
        leaf = tree["run_leaves"][0]
        (leaf / ".mlps-code-image").unlink()

        args = _run_args(tree["root"], tmp_path)
        with caplog.at_level(logging.ERROR):
            rc = run(args)

        assert rc == 1
        all_messages = " ".join(caplog.messages)
        assert "mlps-code-image" in all_messages.lower() or "mlps-code-image" in all_messages, (
            f"SC-2a: expected message containing 'mlps-code-image', got: {all_messages}"
        )

    def test_sc2_dangling_pointer_returns_1(
        self, tmp_path, v11_tree_factory, monkeypatch, caplog
    ):
        """SC-2b: dangling pointer (non-existent pool hash) → exit code 1, references hash."""
        self._patch_non_pool_checks(monkeypatch)
        self._patch_mode_checkers(monkeypatch)

        tree = v11_tree_factory()
        leaf = tree["run_leaves"][0]

        # Overwrite pointer with a non-existent hash
        fake_hash = "b" * 32
        (leaf / ".mlps-code-image").write_text(f"md5-tree-v2:{fake_hash}")

        args = _run_args(tree["root"], tmp_path)
        with caplog.at_level(logging.ERROR):
            rc = run(args)

        assert rc == 1
        all_messages = " ".join(caplog.messages)
        assert "not found in pool" in all_messages, (
            f"SC-2b: expected 'not found in pool' in messages: {all_messages}"
        )

    # -------------------------------------------------------------------
    # SC-3: modified/renamed pool image → run() returns 1 (CHECK-02)
    # -------------------------------------------------------------------

    def test_sc3_modified_pool_content_returns_1(
        self, tmp_path, v11_tree_factory, monkeypatch, caplog
    ):
        """SC-3a: pool image content modified post-capture → exit code 1, CHECK-02 message."""
        self._patch_non_pool_checks(monkeypatch)
        self._patch_mode_checkers(monkeypatch)

        tree = v11_tree_factory()
        pool_dir = tree["pool_dir"]

        # Modify content AFTER hash was stamped
        (pool_dir / "pyproject.toml").write_text("[project]\nname='TAMPERED'\n")

        args = _run_args(tree["root"], tmp_path)
        with caplog.at_level(logging.ERROR):
            rc = run(args)

        assert rc == 1
        all_messages = " ".join(caplog.messages)
        assert "CHECK-02" in all_messages or "poolImageSelfConsistency" in all_messages or "self-consistency" in all_messages, (
            f"SC-3a: expected CHECK-02 violation, messages: {all_messages}"
        )

    def test_sc3_renamed_pool_dir_returns_1(
        self, tmp_path, v11_tree_factory, monkeypatch, caplog
    ):
        """SC-3b: pool dir renamed to wrong hash8 suffix → exit code 1, CHECK-02 message."""
        self._patch_non_pool_checks(monkeypatch)
        self._patch_mode_checkers(monkeypatch)

        tree = v11_tree_factory()
        pool_dir = tree["pool_dir"]
        org_root = tree["org_root"]

        # Rename pool dir to wrong hash8 suffix
        wrong_dir = org_root / "code-12345678"
        pool_dir.rename(wrong_dir)

        args = _run_args(tree["root"], tmp_path)
        with caplog.at_level(logging.ERROR):
            rc = run(args)

        assert rc == 1
        all_messages = " ".join(caplog.messages)
        assert "CHECK-02" in all_messages or "poolImageSelfConsistency" in all_messages, (
            f"SC-3b: expected CHECK-02 violation, messages: {all_messages}"
        )

    # -------------------------------------------------------------------
    # SC-4: orphan pool image / legacy code/ dir → run() returns 1
    # -------------------------------------------------------------------

    def test_sc4_orphan_pool_image_returns_1(
        self, tmp_path, v11_tree_factory, monkeypatch, caplog
    ):
        """SC-4a: unreferenced pool image (orphan) → exit code 1, CHECK-03 message."""
        self._patch_non_pool_checks(monkeypatch)
        self._patch_mode_checkers(monkeypatch)

        tree = v11_tree_factory()
        org_root = tree["org_root"]

        # Add a second unreferenced pool image
        _build_pool_image(
            org_root,
            content="[project]\nname='orphan'\nversion='99.0'\n",
            log=_MockLog(),
        )

        args = _run_args(tree["root"], tmp_path)
        with caplog.at_level(logging.ERROR):
            rc = run(args)

        assert rc == 1
        all_messages = " ".join(caplog.messages)
        assert "orphan" in all_messages.lower() or "not referenced" in all_messages, (
            f"SC-4a: expected orphan message, got: {all_messages}"
        )

    def test_sc4_legacy_code_dir_returns_1(
        self, tmp_path, v11_tree_factory, monkeypatch, caplog
    ):
        """SC-4b: literal code/ directory (unmigrated legacy) → exit code 1, CHECK-04 message."""
        self._patch_non_pool_checks(monkeypatch)
        self._patch_mode_checkers(monkeypatch)

        tree = v11_tree_factory()
        root = tree["root"]

        # Add a legacy code/ directory under closed/<org>/
        legacy = root / "closed" / "Acme" / "code"
        legacy.mkdir(parents=True, exist_ok=True)
        (legacy / "pyproject.toml").write_text("[project]\nname='legacy'\n")

        args = _run_args(root, tmp_path)
        with caplog.at_level(logging.ERROR):
            rc = run(args)

        assert rc == 1
        all_messages = " ".join(caplog.messages)
        assert "Legacy code/ layout detected" in all_messages, (
            f"SC-4b: expected 'Legacy code/ layout detected', got: {all_messages}"
        )

