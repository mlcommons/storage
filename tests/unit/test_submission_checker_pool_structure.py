"""Unit tests for PoolStructureCheck — CHECK-01..04 direct method tests.

Phase 08-03 Plan 03 Task 1.

Covers all behavior points required by the plan:
- TestPoolPointerResolutionCheck: CHECK-01 missing pointer, dangling pointer, valid tree
- TestPoolImageSelfConsistencyCheck: CHECK-02 pass, modified content, renamed pool dir
- TestPoolOrphanCheck: CHECK-03 orphan image, no orphan
- TestPoolLegacyCheck: CHECK-04 legacy code/, partial migration (D-91), empty pool (D-90)
- TestTopLevelSubdirPoolRoot: top_level_subdirectories_check with pool root
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from mlpstorage_py.submission_checker.checks.pool_structure_checks import PoolStructureCheck
from mlpstorage_py.submission_checker.checks.submission_structure_checks import SubmissionStructureCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
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
    """Minimal logger that routes to Python's logging so caplog captures messages."""

    def __init__(self, name: str = "test_pool_structure"):
        self._log = logging.getLogger(name)

    def error(self, msg, *args):
        self._log.error(msg, *args)

    def warning(self, msg, *args):
        self._log.warning(msg, *args)

    def info(self, msg, *args):
        self._log.info(msg, *args)

    def debug(self, msg, *args):
        self._log.debug(msg, *args)

    def status(self, msg, *args):
        self._log.info(msg, *args)


def _make_log():
    return _MockLog()


def _make_config():
    return Config(version="1.0", submitters=None)


# ---------------------------------------------------------------------------
# Tree-building helpers
# ---------------------------------------------------------------------------

def _build_pool_image(
    org_root: Path,
    content: str = "[project]\nname='x'\nversion='0.0.1'\n",
    log=None,
) -> tuple[Path, str]:
    """Build a valid pool image dir under org_root.

    Returns (pool_dir, full_hash).
    """
    if log is None:
        log = _MockLog()

    # (a) Create a staging directory with content files
    stage = org_root / "_stage"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "pyproject.toml").write_text(content)

    # (b) Compute the real hash
    full_hash = compute_code_tree_md5(str(stage), log)

    # (c) Create the pool dir using _pool_dir_name
    pool_dir = org_root / _pool_dir_name(full_hash)
    pool_dir.mkdir(parents=True, exist_ok=True)

    # Copy content files into the final pool dir
    (pool_dir / "pyproject.toml").write_text(content)

    # (d) Write .code-hash.json with all required fields
    hash_data = {
        "hash": full_hash,
        "algorithm": "md5-tree-v2",
        "captured_at": "2026-01-01T00:00:00Z",
        "mlpstorage_version": "1.0.0",
        "git_sha": None,
    }
    (pool_dir / ".code-hash.json").write_text(json.dumps(hash_data, indent=2) + "\n")

    # Clean up staging directory
    import shutil
    shutil.rmtree(str(stage), ignore_errors=True)

    return pool_dir, full_hash


def _build_v11_tree(
    root: Path,
    orgname: str = "Acme",
    n_run_leaves: int = 1,
    log=None,
    pool_content: str = "[project]\nname='x'\nversion='0.0.1'\n",
) -> dict:
    """Build a minimal valid v1.1 submission tree rooted at `root`.

    Returns dict with keys:
      - pool_dir: Path to code-<hash8>/ pool image
      - run_leaves: list of run leaf Paths (datetime dirs)
      - sentinel: Path to .mlps-image-pool
      - org_root: Path to <root>/<orgname>/
      - full_hash: the pool image's full MD5 hash
    """
    if log is None:
        log = _MockLog()

    # Create pool root sentinel
    org_root = root / orgname
    org_root.mkdir(parents=True, exist_ok=True)
    sentinel = org_root / ".mlps-image-pool"
    sentinel.write_text("2026-01-01T00:00:00Z\n")

    # Build the pool image
    pool_dir, full_hash = _build_pool_image(org_root, content=pool_content, log=log)

    # Create run leaf directories and write pointers
    run_leaves = []
    for i in range(n_run_leaves):
        ts = f"20260101_12000{i}"
        leaf = (
            root / "closed" / orgname / "results" / "sys1"
            / "training" / "unet3d" / "run" / ts
        )
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / "output.txt").write_text(f"run {i}\n")
        # (e) Write pointer file atomically
        _write_pointer_atomic(leaf, full_hash, log)
        run_leaves.append(leaf)

    # Add systems dir to avoid spurious required_subdirectories_check failures
    systems_dir = root / "closed" / orgname / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "sys1.yaml").write_text("system: sys1\n")

    return {
        "pool_dir": pool_dir,
        "run_leaves": run_leaves,
        "sentinel": sentinel,
        "org_root": org_root,
        "full_hash": full_hash,
    }


def _make_pool_check(root: Path) -> PoolStructureCheck:
    return PoolStructureCheck(
        log=_make_log(),
        config=_make_config(),
        root_path=str(root),
    )


def _make_struct_check(root: Path) -> SubmissionStructureCheck:
    return SubmissionStructureCheck(
        log=_make_log(),
        config=_make_config(),
        root_path=str(root),
    )


# ===========================================================================
# TestPoolPointerResolutionCheck — CHECK-01
# ===========================================================================

class TestPoolPointerResolutionCheck:
    """CHECK-01: pointer resolution — missing, dangling, and valid cases."""

    def test_missing_pointer_returns_false(self, tmp_path, caplog):
        """CHECK-01: run leaf with no .mlps-code-image fails with descriptive message."""
        root = tmp_path / "root"
        tree = _build_v11_tree(root)
        leaf = tree["run_leaves"][0]

        # Delete the pointer file
        (leaf / ".mlps-code-image").unlink()

        check = _make_pool_check(root)
        with caplog.at_level(logging.ERROR):
            result = check.pool_pointer_resolution_check()

        assert result is False
        all_messages = " ".join(caplog.messages)
        assert "has no .mlps-code-image pointer" in all_messages

    def test_dangling_pointer_returns_false(self, tmp_path, caplog):
        """CHECK-01: run leaf with pointer referencing non-existent pool dir fails."""
        root = tmp_path / "root"
        tree = _build_v11_tree(root)
        leaf = tree["run_leaves"][0]

        # Overwrite pointer with a valid-format hash pointing to non-existent dir
        fake_hash = "a" * 32
        (leaf / ".mlps-code-image").write_text(f"md5-tree-v2:{fake_hash}")

        check = _make_pool_check(root)
        with caplog.at_level(logging.ERROR):
            result = check.pool_pointer_resolution_check()

        assert result is False
        all_messages = " ".join(caplog.messages)
        assert "not found in pool" in all_messages

    def test_valid_tree_returns_true(self, tmp_path, caplog):
        """CHECK-01: valid v1.1 tree with pointer and matching pool dir passes."""
        root = tmp_path / "root"
        _build_v11_tree(root)

        check = _make_pool_check(root)
        with caplog.at_level(logging.ERROR):
            result = check.pool_pointer_resolution_check()

        assert result is True
        # No error messages
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors == [], f"Unexpected errors: {[r.message for r in errors]}"


# ===========================================================================
# TestPoolImageSelfConsistencyCheck — CHECK-02
# ===========================================================================

class TestPoolImageSelfConsistencyCheck:
    """CHECK-02: pool image self-consistency."""

    def test_valid_pool_image_passes(self, tmp_path, caplog):
        """CHECK-02: valid pool image with matching hash passes."""
        root = tmp_path / "root"
        _build_v11_tree(root)

        check = _make_pool_check(root)
        with caplog.at_level(logging.ERROR):
            result = check.pool_image_self_consistency_check()

        assert result is True

    def test_modified_content_returns_false(self, tmp_path, caplog):
        """CHECK-02: pool image with modified content fails self-consistency."""
        root = tmp_path / "root"
        tree = _build_v11_tree(root)
        pool_dir = tree["pool_dir"]

        # Overwrite content AFTER the hash was stamped
        (pool_dir / "pyproject.toml").write_text("[project]\nname='MODIFIED'\n")

        check = _make_pool_check(root)
        with caplog.at_level(logging.ERROR):
            result = check.pool_image_self_consistency_check()

        assert result is False
        # Should mention CHECK-02 or self-consistency in the log
        all_messages = " ".join(caplog.messages)
        assert "CHECK-02" in all_messages or "self-consistency" in all_messages or "poolImageSelfConsistency" in all_messages

    def test_renamed_pool_dir_returns_false(self, tmp_path, caplog):
        """CHECK-02: pool dir renamed to different hash8 suffix fails."""
        root = tmp_path / "root"
        tree = _build_v11_tree(root)
        pool_dir = tree["pool_dir"]
        org_root = tree["org_root"]

        # Rename pool dir to a different hash8 suffix
        wrong_dir = org_root / "code-12345678"
        pool_dir.rename(wrong_dir)

        check = _make_pool_check(root)
        with caplog.at_level(logging.ERROR):
            result = check.pool_image_self_consistency_check()

        # The renamed dir either fails to read .code-hash.json (renamed dir has
        # the old hash in the JSON but directory name disagrees) → fails
        assert result is False


# ===========================================================================
# TestPoolOrphanCheck — CHECK-03
# ===========================================================================

class TestPoolOrphanCheck:
    """CHECK-03: orphan pool image detection."""

    def test_orphan_image_returns_false(self, tmp_path, caplog):
        """CHECK-03: unreferenced pool image is an orphan → returns False."""
        root = tmp_path / "root"
        tree = _build_v11_tree(root)
        org_root = tree["org_root"]
        log = _MockLog()

        # Add an extra pool image not referenced by any run leaf
        orphan_content = "[project]\nname='orphan'\nversion='99.0'\n"
        _build_pool_image(org_root, content=orphan_content, log=log)

        check = _make_pool_check(root)
        with caplog.at_level(logging.ERROR):
            result = check.pool_orphan_check()

        assert result is False
        all_messages = " ".join(caplog.messages)
        assert "not referenced" in all_messages or "orphan" in all_messages.lower()

    def test_all_referenced_returns_true(self, tmp_path, caplog):
        """CHECK-03: all pool images referenced by run leaves → returns True."""
        root = tmp_path / "root"
        _build_v11_tree(root)

        check = _make_pool_check(root)
        with caplog.at_level(logging.ERROR):
            result = check.pool_orphan_check()

        assert result is True


# ===========================================================================
# TestPoolLegacyCheck — CHECK-04
# ===========================================================================

class TestPoolLegacyCheck:
    """CHECK-04: legacy code/ detection, partial migration, empty pool."""

    def test_legacy_code_dir_returns_false(self, tmp_path, caplog):
        """CHECK-04 D-81: legacy code/ directory triggers failure."""
        root = tmp_path / "root"
        # Build a v1.1 tree first (provides the org submitter in closed/)
        _build_v11_tree(root)

        # Add a literal code/ directory under closed/<orgname>/
        legacy = root / "closed" / "Acme" / "code"
        legacy.mkdir(parents=True, exist_ok=True)
        (legacy / "pyproject.toml").write_text("[project]\nname='x'\n")

        check = _make_pool_check(root)
        with caplog.at_level(logging.ERROR):
            result = check.pool_legacy_check()

        assert result is False
        all_messages = " ".join(caplog.messages)
        assert "Legacy code/ layout detected" in all_messages

    def test_partial_migration_d91_returns_false(self, tmp_path, caplog):
        """CHECK-04 D-91: pool images present but .mlps-image-pool absent → partial migration."""
        root = tmp_path / "root"
        orgname = "Acme"

        # Create closed/<orgname>/results/... so the org appears via _iter_submitter_dirs
        leaf = (
            root / "closed" / orgname / "results" / "sys1"
            / "training" / "unet3d" / "run" / "20260101_120000"
        )
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / "output.txt").write_text("run 0\n")

        # Create pool images WITHOUT the sentinel
        org_root = root / orgname
        org_root.mkdir(parents=True, exist_ok=True)
        # sentinel NOT created
        log = _MockLog()
        _build_pool_image(org_root, log=log)

        check = _make_pool_check(root)
        with caplog.at_level(logging.ERROR):
            result = check.pool_legacy_check()

        assert result is False
        all_messages = " ".join(caplog.messages)
        assert "Partial migration detected" in all_messages

    def test_d90_empty_pool_returns_true_with_warning(self, tmp_path, caplog):
        """CHECK-04 D-90: sentinel present but no pool images → warning only, not failure."""
        root = tmp_path / "root"
        orgname = "Acme"

        # Create closed/<orgname>/results/... so org appears
        leaf = (
            root / "closed" / orgname / "results" / "sys1"
            / "training" / "unet3d" / "run" / "20260101_120000"
        )
        leaf.mkdir(parents=True, exist_ok=True)

        # Create sentinel WITHOUT any pool images
        org_root = root / orgname
        org_root.mkdir(parents=True, exist_ok=True)
        (org_root / ".mlps-image-pool").write_text("2026-01-01T00:00:00Z\n")
        # No code-<hash8>/ dirs

        check = _make_pool_check(root)
        with caplog.at_level(logging.WARNING):
            result = check.pool_legacy_check()

        # D-90: must return True (warning, not failure)
        assert result is True
        # Should emit a warning about no pool images
        all_messages = " ".join(caplog.messages)
        assert "no pool images found" in all_messages


# ===========================================================================
# TestTopLevelSubdirPoolRoot
# ===========================================================================

class TestTopLevelSubdirPoolRoot:
    """top_level_subdirectories_check: pool root sentinel recognition (D-83)."""

    def test_pool_root_with_sentinel_is_permitted(self, tmp_path, caplog):
        """D-83: top-level dir containing .mlps-image-pool is a recognized pool root."""
        root = tmp_path / "root"
        _build_v11_tree(root, orgname="Acme")

        check = _make_struct_check(root)
        with caplog.at_level(logging.ERROR):
            result = check.top_level_subdirectories_check()

        assert result is True
        # Ensure no violation about unexpected top-level directory
        errors = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR and "unexpected top-level" in r.getMessage()
        ]
        assert errors == [], f"Unexpected errors: {[r.getMessage() for r in errors]}"

    def test_unexpected_dir_without_sentinel_returns_false(self, tmp_path, caplog):
        """D-85: top-level dir without .mlps-image-pool (and not closed/open) is a structural error."""
        root = tmp_path / "root"

        # Create closed/ division so the check doesn't fail on "no closed or open"
        closed_dir = root / "closed" / "Acme"
        closed_dir.mkdir(parents=True, exist_ok=True)

        # Create a top-level dir WITHOUT sentinel — unexpected
        unexpected = root / "SomeRandomDir"
        unexpected.mkdir(parents=True, exist_ok=True)

        check = _make_struct_check(root)
        with caplog.at_level(logging.ERROR):
            result = check.top_level_subdirectories_check()

        assert result is False
        all_messages = " ".join(caplog.messages)
        assert "unexpected top-level" in all_messages or "SomeRandomDir" in all_messages
