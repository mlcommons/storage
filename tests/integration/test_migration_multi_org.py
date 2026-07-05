"""D-70 per-org migration isolation integration tests.

Covers Phase 7 decision D-70: migration is scoped to the invoking org.
A --results-dir shared by two orgs migrates each independently the first
time each org invokes mlpstorage. Migrating Acme must not touch Bravo's
legacy code/ dirs, run leaves, or pool sentinel.

Wave 0 xfail stubs replaced with real assertions by Plan 07-03 Task 4.

Refs: 07-01-PLAN.md Task 2, 07-CONTEXT.md D-70, RESEARCH §5 per-org scope.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlpstorage_py.submission_checker.tools.code_checksum import compute_code_tree_md5
from mlpstorage_py.submission_checker.tools.legacy_migration import migrate_legacy_layout
from tests.integration.conftest import MockLogger, pool_dirs


def _plant_bravo_legacy(rd: Path) -> Path:
    """Plant a valid Bravo legacy code/ tree under rd/closed/Bravo/code/.

    Returns the Bravo legacy code dir path.  Plants slightly different content
    from Acme (so the hash differs) and writes a valid .code-hash.json.
    Also plants one run leaf under rd/closed/Bravo/results/... for pointer tests.
    """
    bravo_legacy = rd / "closed" / "Bravo" / "code"
    bravo_legacy.mkdir(parents=True)
    (bravo_legacy / "pyproject.toml").write_text("[project]\nname='bravo'\n")
    (bravo_legacy / "file1.py").write_text("X = 100\n")  # different from Acme's X = 1

    # Stamp .code-hash.json with the real hash.
    log = MockLogger()
    h = compute_code_tree_md5(str(bravo_legacy), log)
    (bravo_legacy / ".code-hash.json").write_text(
        json.dumps(
            {
                "hash": h,
                "algorithm": "md5-tree-v2",
                "captured_at": "2026-01-01T00:00:00Z",
                "mlpstorage_version": "1.0.0",
                "git_sha": None,
            }
        )
    )

    # Plant a run leaf for the second multi-org test (pointer write path).
    bravo_leaf = (
        rd / "closed" / "Bravo" / "results" / "sys1"
        / "training" / "unet3d" / "run" / "20260101_120000"
    )
    bravo_leaf.mkdir(parents=True)
    (bravo_leaf / "output.txt").write_text("bravo run\n")

    return bravo_legacy


class TestPerOrgMigrationIsolation:
    """D-70: migration scoped to the invoking org leaves other orgs untouched."""

    def test_migrating_as_acme_leaves_bravos_legacy_untouched(
        self, tmp_path, legacy_tree_factory, log
    ):
        """Migrating Acme does not touch Bravo's legacy tree.

        Plant legacy trees for both Acme and Bravo under the same results_dir.
        Invoke migrate_legacy_layout scoped to Acme. Assert:
        - Bravo's legacy code/ directory is unchanged (still present, same content).
        - <rd>/Bravo/.mlps-image-pool does not exist (no sentinel for Bravo).
        - Acme migration completes normally (sentinel written, legacy deleted).
        """
        rd = legacy_tree_factory(orgname="Acme", n_run_leaves=1)
        bravo_legacy = _plant_bravo_legacy(rd)

        migrate_legacy_layout(rd, "Acme", log)

        # (a) Acme sentinel written
        assert (rd / "Acme" / ".mlps-image-pool").exists(), "Acme sentinel must be written"

        # (b) Acme legacy dir gone
        assert not (rd / "closed" / "Acme" / "code").exists(), (
            "Acme legacy code/ must be deleted after migration"
        )

        # (c) Bravo's legacy code/ dir is still there (untouched)
        assert bravo_legacy.is_dir(), (
            "Bravo's legacy code/ must remain intact when migrating Acme"
        )

        # (d) Bravo has no sentinel (migration not invoked for Bravo)
        bravo_sentinel = rd / "Bravo" / ".mlps-image-pool"
        assert not bravo_sentinel.exists(), (
            "Bravo must NOT have a sentinel when only Acme was migrated (D-70 per-org scoping)"
        )

    def test_running_second_org_migrates_independently(
        self, tmp_path, legacy_tree_factory, log
    ):
        """Migrating Acme then Bravo produces independent pool sentinels for each.

        After migrating Acme, migrate Bravo separately. Assert:
        - <rd>/Acme/.mlps-image-pool exists (Acme sentinel written).
        - <rd>/Bravo/.mlps-image-pool exists (Bravo sentinel written).
        - The two sentinel paths are distinct (no cross-org pool dir sharing).
        - Each org's pool lives under its own org root (D-70 per-org scoping).
        """
        rd = legacy_tree_factory(orgname="Acme", n_run_leaves=1)
        _plant_bravo_legacy(rd)

        # Migrate Acme first.
        migrate_legacy_layout(rd, "Acme", log)

        # Migrate Bravo independently with a fresh logger.
        log_bravo = MockLogger()
        migrate_legacy_layout(rd, "Bravo", log_bravo)

        # Both sentinels present
        assert (rd / "Acme" / ".mlps-image-pool").exists(), "Acme sentinel must be written"
        assert (rd / "Bravo" / ".mlps-image-pool").exists(), "Bravo sentinel must be written"

        # Each org has at least one pool image in its own root
        acme_pools = pool_dirs(rd / "Acme")
        bravo_pools = pool_dirs(rd / "Bravo")
        assert len(acme_pools) >= 1, f"Acme must have at least 1 pool image, got {acme_pools}"
        assert len(bravo_pools) >= 1, f"Bravo must have at least 1 pool image, got {bravo_pools}"

        # Pools are in separate org roots (no cross-org sharing)
        acme_pool_parents = {p.parent for p in acme_pools}
        bravo_pool_parents = {p.parent for p in bravo_pools}
        assert acme_pool_parents == {rd / "Acme"}, (
            f"Acme pools must live under rd/Acme, got {acme_pool_parents}"
        )
        assert bravo_pool_parents == {rd / "Bravo"}, (
            f"Bravo pools must live under rd/Bravo, got {bravo_pool_parents}"
        )
