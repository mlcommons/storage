"""Integration coverage for SC-4 (cross-mode pool dedup) — Phase 6 Plan 06-04 Task 4.

Exercises the ROADMAP SC-4 assertion end-to-end against the already-landed
`capture_or_verify_code_image` rewrite from Plan 06-02:

* D-64 mode-agnostic pool: a CLOSED-mode capture followed by an OPEN-mode
  capture (same org, same source) reuses the single pool image — the org root
  contains exactly one `code-<hash8>/` regardless of run mode.
* Symmetric ordering: OPEN-first then CLOSED-second also produces exactly one
  pool image. Both pointer files carry identical hash content.

The mode-agnostic pool layout (POOL-01, D-64) is what makes cross-mode dedup
possible — pool images live at `<results_dir>/<orgname>/code-*/`, NOT at
`<results_dir>/{closed,open}/<orgname>/code-*/`.

Scope note (per SC#7): tests call `capture_or_verify_code_image` DIRECTLY
(integration-scope: real files, real hashing).

Refs: 06-04-PLAN.md Task 4, 06-CONTEXT.md D-64, POOL-01, CAPVER-01.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mlpstorage_py.submission_checker.tools.code_image import (
    capture_or_verify_code_image,
)

from tests.integration.conftest import pool_dirs


class TestCrossModeDedup:
    """SC-4: pool images dedup across CLOSED and OPEN modes."""

    def test_closed_then_open_reuses_pool(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """SC-4 primary: CLOSED capture first, then OPEN capture — the OPEN
        call reuses the CLOSED-created pool (D-64 mode-agnostic)."""
        rd = tmp_path / "results"
        rd.mkdir()

        args_closed = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        with patch("mlpstorage_py.rules.utils.DATETIME_STR", "20260704_140000"):
            pool_closed = capture_or_verify_code_image(args_closed, {}, log)

        args_open = capture_args_factory(
            results_dir=rd, mode="open", orgname="Acme", systemname="testsys",
            benchmark="training", command="run", model="unet3d",
        )
        with patch("mlpstorage_py.rules.utils.DATETIME_STR", "20260704_140010"):
            pool_open = capture_or_verify_code_image(args_open, {}, log)

        # D-64: mode-agnostic pool. Only ONE pool dir under <rd>/Acme/.
        org_root = rd / "Acme"
        pools = pool_dirs(org_root)
        assert len(pools) == 1, (
            f"SC-4 violated: expected 1 pool dir after closed→open, got {pools}"
        )
        assert Path(pool_closed) == Path(pool_open) == pools[0]

        # Both pointer files exist and carry identical content.
        pointers = sorted(rd.rglob(".mlps-code-image"))
        assert len(pointers) == 2, pointers
        contents = {p.read_text().strip() for p in pointers}
        assert len(contents) == 1, (
            f"pointer contents diverged across modes: {contents}"
        )

    def test_open_then_closed_reuses_pool(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """SC-4 symmetric: OPEN capture first, then CLOSED — symmetry
        preserved (D-64 is order-independent)."""
        rd = tmp_path / "results"
        rd.mkdir()

        args_open = capture_args_factory(
            results_dir=rd, mode="open", orgname="Acme", systemname="testsys",
            benchmark="training", command="run", model="unet3d",
        )
        with patch("mlpstorage_py.rules.utils.DATETIME_STR", "20260704_150000"):
            pool_open = capture_or_verify_code_image(args_open, {}, log)

        args_closed = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        with patch("mlpstorage_py.rules.utils.DATETIME_STR", "20260704_150010"):
            pool_closed = capture_or_verify_code_image(args_closed, {}, log)

        org_root = rd / "Acme"
        pools = pool_dirs(org_root)
        assert len(pools) == 1, (
            f"SC-4 symmetric violated: expected 1 pool dir after open→closed, got {pools}"
        )
        assert Path(pool_open) == Path(pool_closed) == pools[0]

        pointers = sorted(rd.rglob(".mlps-code-image"))
        assert len(pointers) == 2, pointers
        contents = {p.read_text().strip() for p in pointers}
        assert len(contents) == 1, contents
