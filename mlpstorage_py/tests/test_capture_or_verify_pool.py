#!/usr/bin/env python3
"""Phase 6 Plan 06-02 Task 2 (RED) — Content-addressed pool rewrite tests.

Locks the CAPVER-01/02/03 + POOL-01/02/03/04 + PTR-01 + D-64/D-65/D-66/D-67
contract for the rewritten `capture_or_verify_code_image`. Every test in this
file drives the SURVIVING module `mlpstorage_py.submission_checker.tools.code_image`;
the rewrite lands in Plan 06-02 Task 4 (GREEN).

Test fixtures duplicate MockLogger and _make_args from
`mlpstorage_py/tests/test_capture_or_verify_code_image.py:33-75` per
PATTERNS `### New unit test files under mlpstorage_py/tests/` guidance.

Run with:
    pytest mlpstorage_py/tests/test_capture_or_verify_pool.py -v
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlpstorage_py.submission_checker.tools.code_image import (
    CodeImageError,
    LegacyLayoutDetected,
    PoolCorruption,
    capture_or_verify_code_image,
)


# ---------------------------------------------------------------------------
# MockLogger — same shape as the analog
# ---------------------------------------------------------------------------

class MockLogger:
    def __init__(self):
        self.statuses = []
        self.errors = []
        self.warnings = []
        self.infos = []
        self.debugs = []

    def status(self, msg, *args):  self.statuses.append(msg % args if args else msg)
    def error(self, msg, *args):   self.errors.append(msg % args if args else msg)
    def warning(self, msg, *args): self.warnings.append(msg % args if args else msg)
    def info(self, msg, *args):    self.infos.append(msg % args if args else msg)
    def debug(self, msg, *args):   self.debugs.append(msg % args if args else msg)
    def verbose(self, msg, *args): pass
    def verboser(self, msg, *args): pass
    def ridiculous(self, msg, *args): pass


@pytest.fixture
def log():
    return MockLogger()


def _make_args(
    *,
    mode,
    command,
    results_dir,
    benchmark="training",
    model="unet3d",
    orgname=None,
    systemname=None,
    skip_validation=False,
):
    return SimpleNamespace(
        mode=mode,
        command=command,
        results_dir=str(results_dir),
        benchmark=benchmark,
        model=model,
        orgname=orgname,
        systemname=systemname,
        skip_validation=skip_validation,
    )


# ---------------------------------------------------------------------------
# Isolated source-tree fixture — deterministic hash across capture and verify.
# Mirrors the pattern in `mlpstorage_py/tests/test_cli_code_image.py:73-84`.
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_source_root(tmp_path, monkeypatch):
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


def _pool_dirs(org_root: Path) -> list[Path]:
    """List of `code-*` pool dirs under org_root (glob, sorted)."""
    if not org_root.is_dir():
        return []
    return sorted(org_root.glob("code-*"))


# ---------------------------------------------------------------------------
# Class-scoped tests: content-addressed pool + pointer semantics
# ---------------------------------------------------------------------------

class TestCaptureOrVerifyPool:
    """CAPVER-01/02/03 + POOL-01/02/03/04 + PTR-01 + D-63/D-64/D-65/D-66/D-67."""

    # ---- Match branch: SC#2, CAPVER-01 ----

    def test_second_call_with_matching_hash_returns_existing_pool_dir_no_new_capture(
        self, tmp_path, fake_source_root, log
    ):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = _make_args(
            mode="closed", command="datagen", results_dir=results_dir, orgname="Acme",
        )
        # First call captures.
        capture_or_verify_code_image(args, {}, log)
        pool_dirs_after_first = _pool_dirs(results_dir / "Acme")
        assert len(pool_dirs_after_first) == 1, pool_dirs_after_first

        # Second call must not create a new pool dir.
        capture_or_verify_code_image(args, {}, log)
        pool_dirs_after_second = _pool_dirs(results_dir / "Acme")
        assert len(pool_dirs_after_second) == 1, pool_dirs_after_second
        assert pool_dirs_after_first == pool_dirs_after_second

    def test_second_call_writes_pointer_in_new_run_leaf(
        self, tmp_path, fake_source_root, log, monkeypatch
    ):
        """PTR-01: two calls at two datetimes each get their own pointer file."""
        import mlpstorage_py.config as cfg

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        args = _make_args(
            mode="closed", command="datagen", results_dir=results_dir, orgname="Acme",
        )

        # First run at ts1.
        monkeypatch.setattr(cfg, "DATETIME_STR", "2026-06-20_10-00-00")
        # generate_output_location reads DATETIME_STR at call time via
        # `from mlpstorage_py.config import DATETIME_STR` so patch the imported
        # binding in rules.utils as well.
        import mlpstorage_py.rules.utils as ru
        monkeypatch.setattr(ru, "DATETIME_STR", "2026-06-20_10-00-00")

        capture_or_verify_code_image(args, {}, log)
        leaf1 = (
            results_dir / "closed" / "Acme" / "results" / "sys-A" / "training"
            / "unet3d" / "datagen" / "2026-06-20_10-00-00"
        )
        # In CLOSED mode systemname is unused for leaf; use the actual leaf shape
        # generate_output_location produces. Compute below via read of pointer.

        # Second run at ts2.
        monkeypatch.setattr(cfg, "DATETIME_STR", "2026-06-20_11-00-00")
        monkeypatch.setattr(ru, "DATETIME_STR", "2026-06-20_11-00-00")
        capture_or_verify_code_image(args, {}, log)

        # Find every .mlps-code-image pointer under results_dir.
        pointers = list(results_dir.rglob(".mlps-code-image"))
        assert len(pointers) >= 2, pointers
        # Each pointer holds `md5-tree-v2:<32-hex>` with the SAME hash.
        contents = {p.read_text().strip() for p in pointers}
        assert len(contents) == 1, contents
        assert next(iter(contents)).startswith("md5-tree-v2:")

    # ---- No-match / capture branch: SC#1, SC#3, CAPVER-02 ----

    def test_fresh_tree_creates_pool_and_pointer(
        self, tmp_path, fake_source_root, log
    ):
        """SC#1: fresh tree writes pool image + pointer."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
        )
        pool_dir = capture_or_verify_code_image(args, {}, log)

        # Pool directory exists under <rd>/Acme/
        assert pool_dir is not None
        assert Path(pool_dir).is_dir()
        pool_dirs = _pool_dirs(results_dir / "Acme")
        assert len(pool_dirs) == 1

        # POOL-02: .code-hash.json.hash[:8] == dir suffix
        hash_json = Path(pool_dir) / ".code-hash.json"
        assert hash_json.is_file()
        data = json.loads(hash_json.read_text())
        assert Path(pool_dir).name == f"code-{data['hash'][:8]}"

        # Pointer exists, points to full hash
        pointers = list(results_dir.rglob(".mlps-code-image"))
        assert len(pointers) == 1
        assert pointers[0].read_text().strip() == f"md5-tree-v2:{data['hash']}"

    def test_source_change_creates_second_pool_dir_alongside_first(
        self, tmp_path, fake_source_root, log
    ):
        """SC#3: source change captures a second pool dir alongside first.
        Does NOT raise."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
        )
        # First capture
        capture_or_verify_code_image(args, {}, log)

        # Modify source
        (fake_source_root / "mlpstorage_py" / "new_file.py").write_text("Y = 2\n")

        # Second capture
        capture_or_verify_code_image(args, {}, log)

        pool_dirs = _pool_dirs(results_dir / "Acme")
        assert len(pool_dirs) == 2, pool_dirs
        # Distinct .code-hash.json hashes
        hashes = {
            json.loads((p / ".code-hash.json").read_text())["hash"]
            for p in pool_dirs
        }
        assert len(hashes) == 2

    def test_source_change_does_NOT_raise_CodeImageError(
        self, tmp_path, fake_source_root, log
    ):
        """CAPVER-03: hash mismatch NO LONGER raises CodeImageError."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
        )
        capture_or_verify_code_image(args, {}, log)
        (fake_source_root / "mlpstorage_py" / "another.py").write_text("Z = 3\n")

        # Must not raise — CAPVER-03 at unit scope.
        try:
            capture_or_verify_code_image(args, {}, log)
        except (LegacyLayoutDetected, PoolCorruption):
            raise
        except CodeImageError as e:
            pytest.fail(f"CAPVER-03 violated: {e!r}")

    def test_source_change_stderr_does_NOT_contain_retired_reject_string(
        self, tmp_path, fake_source_root, log
    ):
        """UX-01: retired string not emitted on hash mismatch."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
        )
        capture_or_verify_code_image(args, {}, log)
        (fake_source_root / "mlpstorage_py" / "modme.py").write_text("K = 4\n")
        capture_or_verify_code_image(args, {}, log)

        joined = "\n".join(log.errors + log.statuses + log.warnings + log.infos)
        assert "changes to the codebase are not allowed" not in joined
        assert "all runs of this type must use the same codebase" not in joined

    # ---- CAPVER-03 direct guard ----

    def test_capve_no_longer_raises_CodeImageError_on_mismatch(
        self, tmp_path, fake_source_root, log
    ):
        """Pre-seed a pool image with a hash that does NOT match live. Live
        source's REAL hash is different → scan misses → new capture. The
        function completes successfully; no CodeImageError raised."""
        results_dir = tmp_path / "results"
        org_root = results_dir / "Acme"
        org_root.mkdir(parents=True)
        fake_pool = org_root / "code-deadbeef"
        fake_pool.mkdir()
        # Legitimate-shape .code-hash.json with fake hash mismatching live.
        (fake_pool / ".code-hash.json").write_text(json.dumps({
            "hash": "deadbeef" + "0" * 24,
            "algorithm": "md5-tree-v2",
            "captured_at": "2026-01-01T00:00:00Z",
            "mlpstorage_version": "0.0.1",
            "git_sha": None,
        }))
        args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
        )
        capture_or_verify_code_image(args, {}, log)
        pool_dirs = _pool_dirs(org_root)
        # Pre-seeded + newly captured = 2 pool dirs.
        assert len(pool_dirs) == 2, pool_dirs

    # ---- POOL-04 mode-agnostic dedup (SC#4) ----

    def test_closed_then_open_same_source_reuses_pool(
        self, tmp_path, fake_source_root, log
    ):
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        closed_args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
        )
        capture_or_verify_code_image(closed_args, {}, log)

        open_args = _make_args(
            mode="open", command="run", results_dir=results_dir,
            benchmark="training", model="unet3d", orgname="Acme", systemname="rig01",
        )
        capture_or_verify_code_image(open_args, {}, log)

        pool_dirs = _pool_dirs(results_dir / "Acme")
        assert len(pool_dirs) == 1, pool_dirs

    def test_open_then_closed_same_source_reuses_pool(
        self, tmp_path, fake_source_root, log
    ):
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        open_args = _make_args(
            mode="open", command="run", results_dir=results_dir,
            benchmark="training", model="unet3d", orgname="Acme", systemname="rig01",
        )
        capture_or_verify_code_image(open_args, {}, log)

        closed_args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
        )
        capture_or_verify_code_image(closed_args, {}, log)

        pool_dirs = _pool_dirs(results_dir / "Acme")
        assert len(pool_dirs) == 1, pool_dirs

    # ---- POOL-03 per-org isolation ----

    def test_two_orgs_maintain_separate_pool_dirs(
        self, tmp_path, fake_source_root, log
    ):
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        for org in ("Acme", "Beta"):
            args = _make_args(
                mode="closed", command="run", results_dir=results_dir, orgname=org,
            )
            capture_or_verify_code_image(args, {}, log)

        acme_pools = _pool_dirs(results_dir / "Acme")
        beta_pools = _pool_dirs(results_dir / "Beta")
        assert len(acme_pools) == 1, acme_pools
        assert len(beta_pools) == 1, beta_pools
        # Names should not cross-contaminate (each org owns its own subtree).
        for p in acme_pools:
            assert "Beta" not in str(p)
        for p in beta_pools:
            assert "Acme" not in str(p)

    # ---- PTR-01 + D-65 atomicity ordering ----

    def test_pointer_written_after_run_leaf_created(
        self, tmp_path, fake_source_root, log
    ):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
        )
        capture_or_verify_code_image(args, {}, log)
        pointers = list(results_dir.rglob(".mlps-code-image"))
        assert len(pointers) == 1
        pointer = pointers[0]
        # Pointer's parent (run_leaf) must exist and be a directory.
        assert pointer.parent.is_dir()
        # No stale tmp sibling.
        tmp_siblings = list(pointer.parent.glob(".mlps-code-image.tmp.*"))
        assert tmp_siblings == [], tmp_siblings

    def test_pointer_content_matches_full_32_hex_of_live_source(
        self, tmp_path, fake_source_root, log
    ):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
        )
        pool_dir = capture_or_verify_code_image(args, {}, log)
        hash_json = Path(pool_dir) / ".code-hash.json"
        data = json.loads(hash_json.read_text())
        pointers = list(results_dir.rglob(".mlps-code-image"))
        assert pointers[0].read_text().strip() == f"md5-tree-v2:{data['hash']}"

    # ---- D-66 loser branch (SC#10 unit) ----

    def test_loser_branch_when_target_pool_already_exists_with_matching_hash(
        self, tmp_path, fake_source_root, log, monkeypatch
    ):
        """Simulate a race where our scan returned None because the winner
        committed AFTER our scan. Force `_find_matching_pool_image` to return
        None on the ONLY call; pre-seed the pool_dir with a matching hash so
        the rename fails (target non-empty) and the loser branch's
        hash-equality check succeeds silently."""
        # Determine live hash by real capture, then reset.
        import mlpstorage_py.submission_checker.tools.code_image as mod

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
        )
        first_pool = capture_or_verify_code_image(args, {}, log)
        live_hash = json.loads(
            (Path(first_pool) / ".code-hash.json").read_text()
        )["hash"]
        hash8 = live_hash[:8]

        # New tree (throw away results_dir, start fresh)
        import shutil
        shutil.rmtree(results_dir)
        results_dir.mkdir()

        # Pre-seed the pool dir with matching live hash — the "winner"
        org_root = results_dir / "Acme"
        org_root.mkdir()
        winner_pool = org_root / f"code-{hash8}"
        winner_pool.mkdir()
        (winner_pool / ".code-hash.json").write_text(json.dumps({
            "hash": live_hash,
            "algorithm": "md5-tree-v2",
            "captured_at": "2026-01-01T00:00:00Z",
            "mlpstorage_version": "0.0.1",
            "git_sha": None,
        }, indent=2))
        # Populate with at least one file so it looks like a real capture and
        # the rename target is a genuine winner.
        (winner_pool / "sentinel.py").write_text("# winner\n")

        # Force scan to return None → triggers capture path
        monkeypatch.setattr(mod, "_find_matching_pool_image", lambda *a, **k: None)

        # Must complete successfully — loser branch verifies winner's hash
        # equals live_hash and proceeds silently.
        capture_or_verify_code_image(args, {}, log)

        pool_dirs = _pool_dirs(org_root)
        assert len(pool_dirs) == 1, pool_dirs
        # No leaked tmp sibling
        tmp_leaks = list(org_root.glob(".code-*.tmp.*"))
        assert tmp_leaks == [], tmp_leaks

    def test_loser_branch_raises_PoolCorruption_when_winner_hash_differs(
        self, tmp_path, fake_source_root, log, monkeypatch
    ):
        """Simulate a race where our scan missed the winner, AND the winner's
        pool dir contains a hash that does NOT match live_hash. Loser branch
        must raise PoolCorruption."""
        import mlpstorage_py.submission_checker.tools.code_image as mod

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
        )
        # Compute live hash via real capture first
        first_pool = capture_or_verify_code_image(args, {}, log)
        live_hash = json.loads(
            (Path(first_pool) / ".code-hash.json").read_text()
        )["hash"]
        hash8 = live_hash[:8]

        # Reset and pre-seed with MISMATCHING content at the expected path.
        import shutil
        shutil.rmtree(results_dir)
        results_dir.mkdir()
        org_root = results_dir / "Acme"
        org_root.mkdir()
        winner_pool = org_root / f"code-{hash8}"
        winner_pool.mkdir()
        # Different hash — simulates filesystem corruption.
        (winner_pool / ".code-hash.json").write_text(json.dumps({
            "hash": "f" * 32,
            "algorithm": "md5-tree-v2",
            "captured_at": "2026-01-01T00:00:00Z",
            "mlpstorage_version": "0.0.1",
            "git_sha": None,
        }, indent=2))
        (winner_pool / "sentinel.py").write_text("# fake winner\n")

        # Force scan miss → hit capture path
        monkeypatch.setattr(mod, "_find_matching_pool_image", lambda *a, **k: None)

        with pytest.raises(PoolCorruption):
            capture_or_verify_code_image(args, {}, log)

    # ---- D-67 (--skip-validation does not bypass) ----

    def test_skip_validation_arg_does_not_bypass_capture(
        self, tmp_path, fake_source_root, log
    ):
        """D-67: --skip-validation gates validate_benchmark_environment at
        main.py:210, NOT capture_or_verify_code_image at main.py:224. When
        called with skip_validation=True, capture still writes pool + pointer.
        """
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
            skip_validation=True,
        )
        pool_dir = capture_or_verify_code_image(args, {}, log)
        assert pool_dir is not None
        pool_dirs = _pool_dirs(results_dir / "Acme")
        assert len(pool_dirs) == 1
        pointers = list(results_dir.rglob(".mlps-code-image"))
        assert len(pointers) == 1

    # ---- Gating preserved ----

    def test_whatif_mode_returns_None_no_side_effects(self, tmp_path, log):
        args = _make_args(
            mode="whatif", command="run", results_dir=tmp_path, orgname="Acme",
        )
        assert capture_or_verify_code_image(args, {}, log) is None
        assert log.errors == []

    def test_configview_command_under_closed_returns_None(self, tmp_path, log):
        args = _make_args(
            mode="closed", command="configview", results_dir=tmp_path,
            orgname="Acme",
        )
        assert capture_or_verify_code_image(args, {}, log) is None

    # ---- Path safety (regression) ----

    def test_orgname_with_dot_dot_raises_ConfigurationError(self, tmp_path, log):
        from mlpstorage_py.errors import ConfigurationError

        args = _make_args(
            mode="closed", command="run", results_dir=tmp_path, orgname="..",
        )
        with pytest.raises(ConfigurationError):
            capture_or_verify_code_image(args, {}, log)


class TestSentinelSelfHeal:
    """Issue #716: capture_or_verify_code_image must leave a .mlps-image-pool
    sentinel whenever it leaves a pool image. Otherwise CHECK-04 D-91 flags
    the tree as a partial migration forever.
    """

    def test_first_run_on_fresh_tree_writes_sentinel(
        self, tmp_path, fake_source_root, log
    ):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
        )
        capture_or_verify_code_image(args, {}, log)

        sentinel = results_dir / "Acme" / ".mlps-image-pool"
        assert sentinel.is_file(), (
            "capture on a fresh tree must write the sentinel; "
            "otherwise CHECK-04 D-91 flags the tree as partial-migration"
        )
        # Sentinel content shape (D-72): two key=value lines.
        text = sentinel.read_text()
        assert "mlpstorage_version=" in text
        assert "migration_completed_at=" in text

    def test_match_branch_writes_sentinel_when_absent(
        self, tmp_path, fake_source_root, log
    ):
        """Sentinel-heal fires on the reuse path too — covers legacy trees
        that had pool images before the fix landed."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
        )
        # First capture writes both pool and sentinel.
        capture_or_verify_code_image(args, {}, log)
        sentinel = results_dir / "Acme" / ".mlps-image-pool"
        assert sentinel.is_file()

        # Simulate a pre-fix tree: pool dir on disk, sentinel deleted.
        sentinel.unlink()
        assert not sentinel.exists()

        # Second call goes through the match branch and must re-heal the sentinel.
        capture_or_verify_code_image(args, {}, log)
        assert sentinel.is_file(), (
            "reuse path must also self-heal the sentinel — this covers "
            "trees created before the #716 fix landed"
        )

    def test_existing_sentinel_is_not_rewritten(
        self, tmp_path, fake_source_root, log
    ):
        """If the sentinel already exists, don't clobber it — preserves
        migration_completed_at timestamps from prior legitimate migrations."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = _make_args(
            mode="closed", command="run", results_dir=results_dir, orgname="Acme",
        )
        capture_or_verify_code_image(args, {}, log)
        sentinel = results_dir / "Acme" / ".mlps-image-pool"
        first_content = sentinel.read_text()
        first_mtime = sentinel.stat().st_mtime_ns

        # Second call must not rewrite the sentinel.
        capture_or_verify_code_image(args, {}, log)
        assert sentinel.read_text() == first_content
        assert sentinel.stat().st_mtime_ns == first_mtime
