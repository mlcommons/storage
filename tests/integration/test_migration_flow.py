"""MIG-01 end-to-end migration flow integration tests.

Covers Phase 7 decisions D-70/D-71/D-73/D-74 and requirement MIG-01:
  - MIG-01: one-shot, automatic, idempotent legacy-tree migration
  - SC-1 (ROADMAP): run leaf receives pool pointer after migration
  - D-74: exactly two log.status lines emitted during migration
  - All three benchmark shapes (training, checkpointing, vector_database) each
    produce correctly wired run-leaf pointers after migration.

Wave 0 xfail stubs replaced with real assertions by Plan 07-03 Task 3.

Refs: 07-01-PLAN.md Task 2, 07-CONTEXT.md D-70/D-71/D-73/D-74, MIG-01,
RESEARCH §7 three-shape enumeration, PATTERNS §test_migration_flow.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from mlpstorage_py.submission_checker.tools.code_image import HandEditedCodeImage
from mlpstorage_py.submission_checker.tools.legacy_migration import (
    _check_and_migrate_legacy_layout,
    migrate_legacy_layout,
)

# Import the pool_dirs helper and MockLogger from conftest (accessible via fixture
# and direct import for inline use).
from tests.integration.conftest import MockLogger, pool_dirs


class TestMigrateEndToEnd:
    """MIG-01 canonical happy-path scenarios."""

    def test_fresh_v1_tree_migrates_to_v11_pool_pointers_sentinel(
        self, tmp_path, legacy_tree_factory, log
    ):
        """MIG-01 canonical happy path: training shape, 3 run leaves.

        A fresh v1.0-layout tree (legacy code/ with valid .code-hash.json,
        3 run-leaf datetime dirs) must produce:
        - One pool image at <rd>/Acme/code-<hash8>/
        - A .mlps-code-image pointer in every run leaf
        - A .mlps-image-pool sentinel at <rd>/Acme/.mlps-image-pool
        - Original legacy code/ dir deleted
        """
        rd = legacy_tree_factory(orgname="Acme", n_run_leaves=3)
        migrate_legacy_layout(rd, "Acme", log)

        # (a) Exactly one pool image under rd/Acme/
        pools = pool_dirs(rd / "Acme")
        assert len(pools) == 1, f"expected 1 pool dir, got {pools}"

        # (b) 3 pointer files in run leaves
        pointers = list(rd.rglob(".mlps-code-image"))
        assert len(pointers) == 3, f"expected 3 pointer files, got {pointers}"

        # (c) Legacy code/ dir is gone
        assert not (rd / "closed" / "Acme" / "code").exists(), (
            "legacy code/ dir must be deleted after migration"
        )

        # (d) Sentinel written
        assert (rd / "Acme" / ".mlps-image-pool").exists(), "sentinel must be written"

        # (e) Exactly two status log lines (D-74)
        assert len(log.statuses) == 2, f"expected 2 status lines, got {log.statuses}"

        # Additional: pointer content starts with md5-tree-v2: and matches pool hash8
        pool_dir_name = pools[0].name  # e.g. code-ab12cd34
        hash8 = pool_dir_name[len("code-"):]
        for ptr in pointers:
            content = ptr.read_text().strip()
            assert content.startswith("md5-tree-v2:"), f"pointer content malformed: {content!r}"
            assert content.split(":")[1].startswith(hash8), (
                f"pointer hash {content.split(':')[1][:8]} does not match pool dir {hash8}"
            )

    def test_two_legacy_dirs_same_hash_dedup_to_one_pool_image(
        self, tmp_path, legacy_tree_factory, log
    ):
        """MIG-01 dedup path: closed + open legacy dirs with equal hash -> one pool image.

        When two legacy code/ dirs (e.g. closed/Acme/code/ and open/Acme/code/)
        hash identically, migration materializes only ONE pool image (M=1 unique
        from N=2 legacy dirs). Both run-leaf subtrees receive pointers pointing
        at the same pool image. Sentinel summary line reports
        "Migrated 2 legacy code images into pool (1 unique)."
        """
        # Build closed-mode tree with 2 run leaves.
        rd = legacy_tree_factory(orgname="Acme", mode="closed", n_run_leaves=2)

        # Plant identical open/Acme/code/ dir (byte-equal content -> same hash).
        closed_legacy = rd / "closed" / "Acme" / "code"
        open_legacy = rd / "open" / "Acme" / "code"
        open_legacy.mkdir(parents=True)

        # Copy the same files so content is byte-equal -> same hash.
        import shutil
        for f in closed_legacy.iterdir():
            shutil.copy2(f, open_legacy / f.name)

        # Read the same hash from closed's .code-hash.json
        payload = json.loads((closed_legacy / ".code-hash.json").read_text())
        (open_legacy / ".code-hash.json").write_text(json.dumps(payload))

        # Plant a run leaf under open/Acme so _enumerate_run_leaves finds it.
        open_leaf = (
            rd / "open" / "Acme" / "results" / "sys1"
            / "training" / "unet3d" / "run" / "20260202_120000"
        )
        open_leaf.mkdir(parents=True)
        (open_leaf / "output.txt").write_text("open run\n")

        migrate_legacy_layout(rd, "Acme", log)

        # Exactly one pool image (dedup M=1 from N=2).
        pools = pool_dirs(rd / "Acme")
        assert len(pools) == 1, f"dedup: expected 1 pool image, got {pools}"

        # Summary line contains "(1 unique)".
        assert len(log.statuses) == 2
        assert "(1 unique)" in log.statuses[1], (
            f"summary line should report '(1 unique)': {log.statuses[1]!r}"
        )

        # Both legacy code/ dirs are gone.
        assert not closed_legacy.exists(), "closed legacy dir must be deleted"
        assert not open_legacy.exists(), "open legacy dir must be deleted"

    def test_migration_emits_exactly_two_log_status_lines_and_no_more(
        self, tmp_path, legacy_tree_factory, log
    ):
        """D-74: migration emits exactly two log.status() calls — header + summary.

        After invoking migrate_legacy_layout on a valid v1.0 tree, assert
        len(log.statuses) == 2.
        """
        rd = legacy_tree_factory(orgname="Acme", n_run_leaves=1)
        migrate_legacy_layout(rd, "Acme", log)

        assert len(log.statuses) == 2, f"D-74: expected 2 status lines, got {log.statuses}"
        assert log.statuses[0].startswith("Migrating legacy code-image layout under Acme"), (
            f"header line malformed: {log.statuses[0]!r}"
        )
        assert log.statuses[1].startswith("Migrated 1 legacy code images into pool"), (
            f"summary line malformed: {log.statuses[1]!r}"
        )


class TestMigrateBenchmarkShapes:
    """MIG-01 coverage for all three benchmark run-leaf shapes.

    Ensures the run-leaf enumerator (_enumerate_run_leaves) discovers datetime
    dirs regardless of the 4-level / 5-level / 6-level shape variant
    (RESEARCH §7 three-shape enumeration / Pitfall 2).
    """

    def test_training_shape_receives_pointers_in_every_run_leaf(
        self, tmp_path, legacy_tree_factory, log
    ):
        """training shape (5-level): results/<sys>/<bench>/<model>/<cmd>/<datetime>/.

        Every datetime leaf under the training run-leaf base receives a
        .mlps-code-image pointer after migration.
        """
        rd = legacy_tree_factory(orgname="Acme", benchmark_shape="training", n_run_leaves=2)
        migrate_legacy_layout(rd, "Acme", log)

        base = rd / "closed" / "Acme" / "results" / "sys1" / "training" / "unet3d" / "run"
        leaves = sorted(base.iterdir())
        assert len(leaves) == 2, f"expected 2 training run leaves, got {leaves}"
        for leaf in leaves:
            ptr = leaf / ".mlps-code-image"
            assert ptr.exists(), f"pointer missing in training leaf {leaf}"

    def test_checkpointing_shape_receives_pointers_in_every_run_leaf(
        self, tmp_path, legacy_tree_factory, log
    ):
        """checkpointing shape (4-level): results/<sys>/<bench>/<model>/<datetime>/.

        No <command> level — the datetime dir lives one level higher than training.
        Every datetime leaf receives a pointer.
        """
        rd = legacy_tree_factory(orgname="Acme", benchmark_shape="checkpointing", n_run_leaves=2)
        migrate_legacy_layout(rd, "Acme", log)

        base = rd / "closed" / "Acme" / "results" / "sys1" / "checkpointing" / "llama3-8b"
        leaves = sorted(base.iterdir())
        assert len(leaves) == 2, f"expected 2 checkpointing run leaves, got {leaves}"
        for leaf in leaves:
            ptr = leaf / ".mlps-code-image"
            assert ptr.exists(), f"pointer missing in checkpointing leaf {leaf}"

    def test_vector_database_shape_receives_pointers_in_every_run_leaf(
        self, tmp_path, legacy_tree_factory, log
    ):
        """vector_database shape (6-level): results/<sys>/<bench>/<engine>/<index>/<cmd>/<datetime>/.

        The deepest shape — two extra levels (engine, index) before command.
        Every datetime leaf receives a pointer. Locks RESEARCH §Pitfall 2.

        Note: _enumerate_run_leaves returns both the 5-level 'run/' dir (via
        */*/*/*/*) and the 6-level datetime dirs (via */*/*/*/*/*). Pointers are
        written to all of them. We assert the 2 datetime leaves each have a pointer.
        """
        rd = legacy_tree_factory(orgname="Acme", benchmark_shape="vector_database", n_run_leaves=2)
        migrate_legacy_layout(rd, "Acme", log)

        base = (
            rd / "closed" / "Acme" / "results" / "sys1"
            / "vector_database" / "diskann" / "HNSW" / "run"
        )
        # The datetime leaves are the subdirectories of run/ (excluding hidden pointer files).
        datetime_leaves = sorted(p for p in base.iterdir() if p.is_dir())
        assert len(datetime_leaves) == 2, (
            f"expected 2 vector_database datetime leaves, got {datetime_leaves}"
        )
        for leaf in datetime_leaves:
            ptr = leaf / ".mlps-code-image"
            assert ptr.exists(), f"pointer missing in vector_database datetime leaf {leaf}"


class TestMigrateLeafSubdirs:
    """#725 Bug 1: migration must not drop pointers into leaf subdirectories.

    Real DLIO run leaves contain ``dlio_config/``, ``collector-staging/``,
    ``.chk_iterations/`` and similar subdirs. The migration's fixed-depth
    globs overlap between benchmark shapes (training's 5-level glob also
    matches checkpointing's ``<dt>/dlio_config``), and only a leaf-name
    filter keeps pointer writes off those subdirectories. Without it,
    2.1.15 / 2.1.20 / 2.1.26 fail on previously-valid submissions.
    """

    def test_training_leaf_subdirs_do_not_receive_pointer(
        self, tmp_path, legacy_tree_factory, log
    ):
        """A training leaf's ``dlio_config/`` and ``collector-staging/`` must
        NOT receive a ``.mlps-code-image`` pointer.
        """
        rd = legacy_tree_factory(
            orgname="Acme", benchmark_shape="training", n_run_leaves=1
        )
        base = (
            rd / "closed" / "Acme" / "results" / "sys1"
            / "training" / "unet3d" / "run"
        )
        leaf = next(base.iterdir())
        (leaf / "dlio_config").mkdir()
        (leaf / "dlio_config" / "hydra.yaml").write_text("")
        (leaf / "dlio_config" / "overrides.yaml").write_text("")
        (leaf / "dlio_config" / "config.yaml").write_text("")
        (leaf / "collector-staging").mkdir()

        migrate_legacy_layout(rd, "Acme", log)

        assert (leaf / ".mlps-code-image").exists(), (
            "training leaf root must still receive a pointer"
        )
        assert not (leaf / "dlio_config" / ".mlps-code-image").exists(), (
            "Bug 1: dlio_config/ must NOT receive a pointer (2.1.20 breaker)"
        )
        assert not (leaf / "collector-staging" / ".mlps-code-image").exists(), (
            "Bug 1: collector-staging/ must NOT receive a pointer"
        )

    def test_checkpointing_leaf_subdirs_do_not_receive_pointer(
        self, tmp_path, legacy_tree_factory, log
    ):
        """A checkpointing leaf's ``dlio_config/`` and ``collector-staging/``
        must NOT receive a pointer (2.1.26 breaker in the reported repro).
        """
        rd = legacy_tree_factory(
            orgname="Acme", benchmark_shape="checkpointing", n_run_leaves=1
        )
        base = (
            rd / "closed" / "Acme" / "results" / "sys1"
            / "checkpointing" / "llama3-8b"
        )
        leaf = next(base.iterdir())
        (leaf / "dlio_config").mkdir()
        (leaf / "collector-staging").mkdir()

        migrate_legacy_layout(rd, "Acme", log)

        assert (leaf / ".mlps-code-image").exists(), (
            "checkpointing leaf root must still receive a pointer"
        )
        assert not (leaf / "dlio_config" / ".mlps-code-image").exists(), (
            "Bug 1: dlio_config/ must NOT receive a pointer (2.1.26 breaker)"
        )
        assert not (leaf / "collector-staging" / ".mlps-code-image").exists(), (
            "Bug 1: collector-staging/ must NOT receive a pointer"
        )


class TestMigrateEmptyRunLeaves:
    """Edge cases for trees with zero or absent run-leaf dirs."""

    def test_legacy_code_dir_without_run_leaves_still_writes_sentinel(
        self, tmp_path, legacy_tree_factory, log
    ):
        """n_run_leaves=0: migration still writes the sentinel, no pointer writes.

        A legacy tree with a valid code/ dir but no datetime run-leaf dirs
        must still materialize the pool image and write the sentinel. No pointer
        write occurs (nothing to point at). Sentinel content is valid.
        """
        rd = legacy_tree_factory(orgname="Acme", n_run_leaves=0)
        migrate_legacy_layout(rd, "Acme", log)

        # Sentinel written
        assert (rd / "Acme" / ".mlps-image-pool").exists(), (
            "sentinel must be written even with n_run_leaves=0"
        )

        # No pointer files (nothing to point at)
        pointers = list(rd.rglob(".mlps-code-image"))
        assert pointers == [], f"expected zero pointer files with n_run_leaves=0, got {pointers}"

        # Pool image still materialized
        pools = pool_dirs(rd / "Acme")
        assert len(pools) >= 1, "pool image must be materialized even with n_run_leaves=0"

    def test_fresh_tree_no_legacy_no_sentinel_written(self, tmp_path, log):
        """Locks Assumption A3 / Pitfall 6 recommendation (b).

        A truly-fresh tree with no legacy code/ directory — the pre-check
        _check_and_migrate_legacy_layout sees no offenders and returns without
        writing a sentinel. Sentinel must NOT be written on fresh trees.
        """
        rd = tmp_path / "results"
        rd.mkdir()
        args = Namespace(
            mode="closed",
            command="run",
            results_dir=str(rd),
            orgname="Acme",
            systemname=None,
        )
        _check_and_migrate_legacy_layout(args, {}, log)

        # No sentinel — fresh tree has no migration event.
        acme_root = rd / "Acme"
        sentinel_written = acme_root.exists() and (acme_root / ".mlps-image-pool").exists()
        assert not sentinel_written, (
            "fresh-tree (no legacy code/) must NOT receive a sentinel (A3b)"
        )
