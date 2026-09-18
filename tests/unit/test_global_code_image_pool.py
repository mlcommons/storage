"""Global code-image pool: ``<results-dir>/code-images/`` (PR4 of the
results-dir hygiene effort).

One content-addressed pool per tree, shared by every organization and both
divisions, replaces the per-organization pools at ``<results-dir>/<org>/``.
Per-organization pools remain readable forever (the frozen v3.0 tree carries
19 of them); the CLI relocates the current org's pool into ``code-images/``
the next time it captures into the tree.

Covered here:
- capture writes new images + the ``.mlps-image-pool`` sentinel under
  ``code-images/`` and never creates ``<results-dir>/<org>/``;
- ``_check_and_migrate_legacy_layout`` moves a per-org pool (with or without
  its sentinel) into the global pool, drops duplicates after verifying the
  kept copy, removes the empty org dir, and is a no-op afterwards;
- legacy ``code/`` trees now materialize straight into ``code-images/``;
- CHECK-01 resolves pointers against the global pool first, then the org
  pool; a missing image names both places;
- CHECK-03 counts references from every org against the global pool;
- CHECK-04 flags an unsentinelled global pool and stays silent about a
  per-org pool (supported layout, not legacy);
- 2.1.2 accepts ``code-images`` as a reserved top-level name;
- reportgen's Code/Logs href points at ``code-images/code-<hash8>/`` when
  the image lives there;
- ``mlpstorage runs gc`` trashes orphans from the global pool.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mlpstorage_py.config import EXIT_CODE
from mlpstorage_py.submission_checker.checks.pool_structure_checks import PoolStructureCheck
from mlpstorage_py.submission_checker.checks.submission_structure_checks import (
    SubmissionStructureCheck,
)
from mlpstorage_py.submission_checker.tools import code_image
from mlpstorage_py.submission_checker.tools.code_image import (
    GLOBAL_POOL_DIRNAME,
    _pool_dir_name,
    _read_pointer,
    _write_pointer_atomic,
    capture_or_verify_code_image,
    global_pool_root,
    resolve_pool_image,
)
from mlpstorage_py.submission_checker.tools.legacy_migration import (
    _SENTINEL_FILENAME,
    _check_and_migrate_legacy_layout,
    migrate_org_pool,
)

from tests.unit.test_submission_checker_pool_structure import (
    _MockLog,
    _build_pool_image,
    _make_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leaf(root: Path, org: str, ts: str, division: str = "closed") -> Path:
    leaf = root / division / org / "results" / "sys1" / "training" / "unet3d" / "run" / ts
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "output.txt").write_text("run\n")
    systems = root / division / org / "systems"
    systems.mkdir(parents=True, exist_ok=True)
    (systems / "sys1.yaml").write_text("system: sys1\n")
    return leaf


def _sentinel(pool_root: Path) -> Path:
    pool_root.mkdir(parents=True, exist_ok=True)
    s = pool_root / _SENTINEL_FILENAME
    s.write_text("mlpstorage_version=3.0.0\nmigration_completed_at=2026-01-01T00:00:00Z\n")
    return s


def _pool_check(root: Path) -> PoolStructureCheck:
    return PoolStructureCheck(log=_MockLog(), config=_make_config(), root_path=str(root))


def _struct_check(root: Path) -> SubmissionStructureCheck:
    return SubmissionStructureCheck(log=_MockLog(), config=_make_config(), root_path=str(root))


def _capture_args(results_dir: Path, orgname: str = "Acme") -> SimpleNamespace:
    return SimpleNamespace(
        mode="closed", command="run", results_dir=str(results_dir),
        benchmark="training", model="unet3d", orgname=orgname,
        systemname="sys1", skip_validation=False,
    )


@pytest.fixture
def fake_source_root(tmp_path, monkeypatch):
    src = tmp_path / "src_root"
    src.mkdir()
    (src / "pyproject.toml").write_text("[project]\nname = 'x'\nversion='0.0.1'\n")
    (src / "mlpstorage_py").mkdir()
    (src / "mlpstorage_py" / "__init__.py").write_text("__version__ = '0.0.1'\n")
    monkeypatch.setattr(code_image, "find_source_root", lambda: src)
    return src


@pytest.fixture
def rd(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Pool root helpers
# ---------------------------------------------------------------------------

class TestPoolRootHelpers:
    def test_global_pool_dirname_and_root(self, rd):
        assert GLOBAL_POOL_DIRNAME == "code-images"
        assert global_pool_root(rd) == rd / "code-images"

    def test_resolve_prefers_global_then_org(self, rd):
        g_img, h = _build_pool_image(rd / "code-images")
        assert resolve_pool_image(rd, h, "Acme") == g_img
        # Same hash also present in the org pool: global still wins.
        o_img, _ = _build_pool_image(rd / "Acme")
        assert resolve_pool_image(rd, h, "Acme") == g_img
        shutil.rmtree(g_img)
        assert resolve_pool_image(rd, h, "Acme") == o_img
        shutil.rmtree(o_img)
        assert resolve_pool_image(rd, h, "Acme") is None

    def test_resolve_without_orgname_scans_sentinelled_org_pools(self, rd):
        o_img, h = _build_pool_image(rd / "Beta")
        _sentinel(rd / "Beta")
        assert resolve_pool_image(rd, h, None) == o_img


# ---------------------------------------------------------------------------
# Capture writes to the global pool
# ---------------------------------------------------------------------------

class TestCaptureIntoGlobalPool:
    def test_new_image_and_sentinel_land_in_code_images(self, rd, fake_source_root):
        log = _MockLog()
        pool_dir = capture_or_verify_code_image(_capture_args(rd), {}, log)
        assert pool_dir.parent == rd / "code-images"
        assert (rd / "code-images" / _SENTINEL_FILENAME).is_file()
        assert not (rd / "Acme").exists(), "capture must not create a per-org dir"
        # Pointer resolves back to the same image.
        leaves = list((rd / "closed" / "Acme" / "results").rglob(".mlps-code-image"))
        assert len(leaves) == 1
        _alg, full_hash = _read_pointer(leaves[0].parent, log)
        assert _pool_dir_name(full_hash) == pool_dir.name

    def test_second_capture_reuses_global_image(self, rd, fake_source_root):
        log = _MockLog()
        first = capture_or_verify_code_image(_capture_args(rd), {}, log)
        second = capture_or_verify_code_image(_capture_args(rd), {}, log)
        assert first == second
        assert sorted((rd / "code-images").glob("code-*")) == [first]

    def test_two_orgs_share_one_image(self, rd, fake_source_root):
        log = _MockLog()
        a = capture_or_verify_code_image(_capture_args(rd, "Acme"), {}, log)
        b = capture_or_verify_code_image(_capture_args(rd, "Beta"), {}, log)
        assert a == b
        assert not (rd / "Acme").exists() and not (rd / "Beta").exists()


# ---------------------------------------------------------------------------
# Per-org pool → global pool relocation
# ---------------------------------------------------------------------------

class TestOrgPoolRelocation:
    def _org_pool(self, rd, org="Acme", n=2, sentinel=True):
        images = []
        for i in range(n):
            img, h = _build_pool_image(rd / org, content=f"[project]\nname='x{i}'\n")
            leaf = _leaf(rd, org, f"2026010{i + 1}_120000")
            _write_pointer_atomic(leaf, h, _MockLog())
            images.append((img, h, leaf))
        if sentinel:
            _sentinel(rd / org)
        return images

    def test_org_pool_moves_into_global_pool(self, rd):
        images = self._org_pool(rd)
        log = _MockLog()
        _check_and_migrate_legacy_layout(_capture_args(rd), {}, log)
        assert not (rd / "Acme").exists(), "empty org pool dir must be removed"
        assert (rd / "code-images" / _SENTINEL_FILENAME).is_file()
        for img, h, leaf in images:
            moved = rd / "code-images" / img.name
            assert moved.is_dir()
            assert json.loads((moved / ".code-hash.json").read_text())["hash"] == h
            # Leaves are never rewritten: the pointer still names the hash.
            assert _read_pointer(leaf, log)[1] == h
        assert resolve_pool_image(rd, images[0][1], "Acme") == rd / "code-images" / images[0][0].name

    def test_relocation_is_idempotent(self, rd):
        self._org_pool(rd)
        log = _MockLog()
        _check_and_migrate_legacy_layout(_capture_args(rd), {}, log)
        before = sorted(p.name for p in (rd / "code-images").iterdir())
        statuses_before = len(log.statuses)
        _check_and_migrate_legacy_layout(_capture_args(rd), {}, log)
        assert sorted(p.name for p in (rd / "code-images").iterdir()) == before
        assert len(log.statuses) == statuses_before, "second pass must be silent"

    def test_unsentinelled_org_pool_is_relocated_too(self, rd):
        self._org_pool(rd, sentinel=False)
        _check_and_migrate_legacy_layout(_capture_args(rd), {}, _MockLog())
        assert not (rd / "Acme").exists()
        assert len(list((rd / "code-images").glob("code-*"))) == 2

    def test_duplicate_image_in_both_pools_keeps_global_copy(self, rd):
        images = self._org_pool(rd, n=1)
        img, h, _leaf_ = images[0]
        # The same content already sits in the global pool.
        g_img, g_h = _build_pool_image(rd / "code-images")
        assert g_h == h and g_img.name == img.name
        _sentinel(rd / "code-images")
        (g_img / "marker").write_text("")  # would break self-consistency if re-hashed
        os.remove(g_img / "marker")
        _check_and_migrate_legacy_layout(_capture_args(rd), {}, _MockLog())
        assert not (rd / "Acme").exists()
        assert (rd / "code-images" / img.name).is_dir()
        assert len(list((rd / "code-images").glob("code-*"))) == 1

    def test_org_dir_with_foreign_files_is_kept(self, rd):
        self._org_pool(rd, n=1)
        (rd / "Acme" / "NOTES.txt").write_text("keep me\n")
        _check_and_migrate_legacy_layout(_capture_args(rd), {}, _MockLog())
        assert (rd / "Acme" / "NOTES.txt").is_file()
        assert not (rd / "Acme" / _SENTINEL_FILENAME).exists()
        assert not list((rd / "Acme").glob("code-*"))

    def test_other_orgs_pools_are_left_alone(self, rd):
        self._org_pool(rd, org="Acme", n=1)
        self._org_pool(rd, org="Beta", n=1)
        _check_and_migrate_legacy_layout(_capture_args(rd, "Acme"), {}, _MockLog())
        assert not (rd / "Acme").exists()
        assert (rd / "Beta" / _SENTINEL_FILENAME).is_file()
        assert len(list((rd / "Beta").glob("code-*"))) == 1

    def test_migrate_org_pool_returns_count_and_logs_two_status_lines(self, rd):
        self._org_pool(rd, n=2)
        log = _MockLog()
        assert migrate_org_pool(rd, "Acme", log) == 2
        assert len(log.statuses) == 2
        assert migrate_org_pool(rd, "Acme", log) == 0

    def test_legacy_code_dir_materializes_into_global_pool(self, rd):
        from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5

        legacy = rd / "closed" / "Acme" / "code"
        legacy.mkdir(parents=True)
        (legacy / "pyproject.toml").write_text("[project]\nname='legacy'\n")
        h = compute_code_tree_md5(str(legacy), _MockLog())
        (legacy / ".code-hash.json").write_text(json.dumps({
            "hash": h, "algorithm": "md5-tree-v2",
            "captured_at": "2026-01-01T00:00:00Z",
            "mlpstorage_version": "1.0.0", "git_sha": None,
        }))
        leaf = _leaf(rd, "Acme", "20260101_120000")
        _check_and_migrate_legacy_layout(_capture_args(rd), {}, _MockLog())
        assert not legacy.exists()
        assert (rd / "code-images" / _pool_dir_name(h)).is_dir()
        assert (rd / "code-images" / _SENTINEL_FILENAME).is_file()
        assert not (rd / "Acme").exists()
        assert _read_pointer(leaf, _MockLog())[1] == h


# ---------------------------------------------------------------------------
# Validator: CHECK-01 / CHECK-03 / CHECK-04 / 2.1.2
# ---------------------------------------------------------------------------

class TestCheckerWithGlobalPool:
    def _global_tree(self, rd, orgs=("Acme",)):
        img, h = _build_pool_image(rd / "code-images")
        _sentinel(rd / "code-images")
        leaves = {}
        for org in orgs:
            leaf = _leaf(rd, org, "20260101_120000")
            _write_pointer_atomic(leaf, h, _MockLog())
            leaves[org] = leaf
        return img, h, leaves

    def test_check01_resolves_via_global_pool_without_org_dir(self, rd, caplog):
        self._global_tree(rd, orgs=("Acme", "Beta"))
        with caplog.at_level("ERROR"):
            assert _pool_check(rd).pool_pointer_resolution_check() is True
        assert "CHECK-01" not in caplog.text

    def test_check01_still_resolves_via_org_pool(self, rd, caplog):
        img, h = _build_pool_image(rd / "Acme")
        _sentinel(rd / "Acme")
        _write_pointer_atomic(_leaf(rd, "Acme", "20260101_120000"), h, _MockLog())
        with caplog.at_level("ERROR"):
            assert _pool_check(rd).pool_pointer_resolution_check() is True
        assert "CHECK-01" not in caplog.text

    def test_check01_missing_image_names_both_pools(self, rd, caplog):
        img, h, _ = self._global_tree(rd)
        shutil.rmtree(img)
        with caplog.at_level("ERROR"):
            assert _pool_check(rd).pool_pointer_resolution_check() is False
        assert "code-images/" in caplog.text and "Acme/" in caplog.text

    def test_check01_no_pool_anywhere_is_one_error_per_org(self, rd, caplog):
        _write_pointer_atomic(_leaf(rd, "Acme", "20260101_120000"), "a" * 32, _MockLog())
        with caplog.at_level("ERROR"):
            assert _pool_check(rd).pool_pointer_resolution_check() is False
        assert caplog.text.count("CHECK-01") == 1
        assert "code-images" in caplog.text

    def test_check03_reference_from_any_org_keeps_global_image(self, rd, caplog):
        img, h, leaves = self._global_tree(rd, orgs=("Acme", "Beta"))
        # Only Beta still points at the image.
        os.remove(leaves["Acme"] / ".mlps-code-image")
        with caplog.at_level("ERROR"):
            assert _pool_check(rd).pool_orphan_check() is True
        assert "CHECK-03" not in caplog.text

    def test_check03_flags_unreferenced_global_image(self, rd, caplog):
        self._global_tree(rd)
        orphan, _ = _build_pool_image(rd / "code-images", content="[project]\nname='orphan'\n")
        with caplog.at_level("ERROR"):
            assert _pool_check(rd).pool_orphan_check() is False
        assert orphan.name in caplog.text

    def test_check02_covers_global_pool(self, rd, caplog):
        img, h, _ = self._global_tree(rd)
        (img / "pyproject.toml").write_text("tampered\n")
        with caplog.at_level("ERROR"):
            assert _pool_check(rd).pool_image_self_consistency_check() is False
        assert "CHECK-02" in caplog.text and img.name in caplog.text

    def test_check04_unsentinelled_global_pool_is_partial_migration(self, rd, caplog):
        img, h, _ = self._global_tree(rd)
        os.remove(rd / "code-images" / _SENTINEL_FILENAME)
        with caplog.at_level("ERROR"):
            assert _pool_check(rd).pool_legacy_check() is False
        assert "CHECK-04" in caplog.text and "code-images" in caplog.text

    def test_check04_is_silent_about_a_per_org_pool(self, rd, caplog):
        img, h = _build_pool_image(rd / "Acme")
        _sentinel(rd / "Acme")
        _write_pointer_atomic(_leaf(rd, "Acme", "20260101_120000"), h, _MockLog())
        with caplog.at_level("INFO"):
            assert _pool_check(rd).pool_legacy_check() is True
        assert "CHECK-04" not in caplog.text

    def test_check04_global_and_org_pools_coexist_without_noise(self, rd, caplog):
        self._global_tree(rd, orgs=("Acme",))
        img, h = _build_pool_image(rd / "Beta", content="[project]\nname='beta'\n")
        _sentinel(rd / "Beta")
        _write_pointer_atomic(_leaf(rd, "Beta", "20260101_120000"), h, _MockLog())
        with caplog.at_level("INFO"):
            check = _pool_check(rd)
            assert check.pool_pointer_resolution_check() is True
            assert check.pool_orphan_check() is True
            assert check.pool_legacy_check() is True
        assert "CHECK-0" not in caplog.text

    def test_212_accepts_code_images_even_without_sentinel(self, rd, caplog):
        (rd / "closed").mkdir()
        (rd / "code-images").mkdir()
        with caplog.at_level("ERROR"):
            assert _struct_check(rd).top_level_subdirectories_check() is True
        assert "2.1.2" not in caplog.text

    def test_212_message_names_code_images(self, rd, caplog):
        (rd / "closed").mkdir()
        (rd / "stray").mkdir()
        with caplog.at_level("ERROR"):
            assert _struct_check(rd).top_level_subdirectories_check() is False
        assert "code-images" in caplog.text


# ---------------------------------------------------------------------------
# reportgen Code/Logs href
# ---------------------------------------------------------------------------

class TestReportgenHref:
    def test_href_points_at_global_pool(self, tmp_path):
        from tests.unit.test_reportgen_sut_block import (
            _CODE_DIR, _CODE_HASH, _acme_per_model_dir, _inject_code_pointer,
            _prepare_tree, _run_reportgen,
        )

        root = _prepare_tree(tmp_path)
        _inject_code_pointer(root)
        pool = root / "code-images" / _CODE_DIR
        pool.mkdir(parents=True)
        (pool / ".code-hash.json").write_text(
            json.dumps({"hash": _CODE_HASH, "algorithm": "md5-tree-v2"}))
        _run_reportgen(root)
        rows = json.loads((_acme_per_model_dir(root) / "results.json").read_text())
        assert rows[0]["Training - Code"]["href"] == f"code-images/{_CODE_DIR}/"


# ---------------------------------------------------------------------------
# runs gc
# ---------------------------------------------------------------------------

class TestRunsGcGlobalPool:
    def test_gc_trashes_orphan_from_global_pool(self, tmp_path, monkeypatch, capsys):
        from mlpstorage_py.results_dir import write_sentinel
        from mlpstorage_py.results_dir.user_config import record_results_dir
        from tests.unit.test_runs_management import (
            FULL_HASH_A, FULL_HASH_B, LEAF_TRAIN_RUN, _main, _make_leaf,
        )

        cfg = tmp_path / "xdg"
        cfg.mkdir()
        monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg))
        monkeypatch.delenv("MLPSTORAGE_RESULTS_DIR", raising=False)
        rd = tmp_path / "results"
        rd.mkdir()
        write_sentinel(str(rd), "Acme")
        record_results_dir(str(rd))
        _make_leaf(str(rd), LEAF_TRAIN_RUN, exit_status=0, pointer=FULL_HASH_A)
        _sentinel(rd / "code-images")
        for h in (FULL_HASH_A, FULL_HASH_B):
            img = rd / "code-images" / f"code-{h[:8]}"
            img.mkdir()
            (img / ".code-hash.json").write_text(json.dumps({"hash": h}))
        assert _main(["runs", "gc", "--yes"]) == EXIT_CODE.SUCCESS
        assert (rd / "code-images" / "code-aaaaaaaa").is_dir()
        assert not (rd / "code-images" / "code-bbbbbbbb").exists()
        batch = os.listdir(rd / ".mlps" / "trash")[0]
        assert (rd / ".mlps" / "trash" / batch / "code-images" / "code-bbbbbbbb").is_dir()
        assert "code-bbbbbbbb" in capsys.readouterr().out
