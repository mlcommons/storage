"""Wave-0 xfail scaffolding for D-70 per-org migration isolation tests.

Covers Phase 7 decision D-70: migration is scoped to the invoking org.
A --results-dir shared by two orgs migrates each independently the first
time each org invokes mlpstorage. Migrating Acme must not touch Bravo's
legacy code/ dirs, run leaves, or pool sentinel.

Wave 0 note: every test stub raises NotImplementedError and is marked
xfail(strict=True). Wave-2 (Plan 07-03) removes xfail decorators and
populates test bodies by importing the production ``migrate_legacy_layout``
function from ``mlpstorage_py.submission_checker.tools.legacy_migration``
(module does not exist until Plan 07-02).

Refs: 07-01-PLAN.md Task 2, 07-CONTEXT.md D-70, RESEARCH §5 per-org scope.
"""

from __future__ import annotations

import pytest
from pathlib import Path


class TestPerOrgMigrationIsolation:
    """D-70: migration scoped to the invoking org leaves other orgs untouched."""

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-03/07-04", raises=NotImplementedError)
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
        raise NotImplementedError(
            "Wave 0 stub — implementation lands with production module"
        )

    @pytest.mark.xfail(strict=True, reason="Wave 0 scaffold — implementation in Plan 07-03/07-04", raises=NotImplementedError)
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
        raise NotImplementedError(
            "Wave 0 stub — implementation lands with production module"
        )
