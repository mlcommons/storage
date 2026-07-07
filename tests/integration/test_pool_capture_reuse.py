"""Integration coverage for SC-2 (pool image reuse across runs) — Phase 6 Plan 06-04 Task 2.

Exercises the ROADMAP SC-2 assertion end-to-end against the already-landed
`capture_or_verify_code_image` rewrite from Plan 06-02:

* A second invocation with an unchanged source tree reuses the existing pool
  image (CAPVER-01) — the org root still contains exactly one `code-<hash8>/`
  directory (POOL-01, POOL-04 idempotent).
* The second invocation still writes a pointer file in ITS OWN run leaf; both
  pointer files carry identical content (PTR-01) — reuse never desynchronizes
  the pointer contract.

Because `DATETIME_STR` is captured once at module load time in
`mlpstorage_py.config`, two capture calls inside the same test process
naturally collide on the same run leaf. This test patches
`mlpstorage_py.rules.utils.DATETIME_STR` between calls so the two invocations
land in distinct leaves (matching real-world "user runs the benchmark twice
on separate days" semantics).

Scope note (per SC#7): tests call `capture_or_verify_code_image` DIRECTLY
(integration-scope: real files, real hashing) rather than driving through
`main._main_impl`.

Refs: 06-04-PLAN.md Task 2, 06-CONTEXT.md CAPVER-01, POOL-01, POOL-04, PTR-01.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mlpstorage_py.submission_checker.tools.code_image import (
    capture_or_verify_code_image,
)

from tests.integration.conftest import pool_dirs


class TestPoolImageReuse:
    """SC-2: second invocation with unchanged source reuses the existing pool."""

    def test_second_call_with_unchanged_source_produces_zero_new_pool_images(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """SC-2 primary: two calls, no source change → 1 pool dir; both run
        leaves contain identical `.mlps-code-image` pointer content."""
        rd = tmp_path / "results"
        rd.mkdir()

        args1 = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        # First call at a fixed timestamp.
        with patch("mlpstorage_py.rules.utils.DATETIME_STR", "20260704_120000"):
            pool_dir_1 = capture_or_verify_code_image(args1, {}, log)

        # Second call at a different timestamp — distinct run leaf.
        args2 = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        with patch("mlpstorage_py.rules.utils.DATETIME_STR", "20260704_120005"):
            pool_dir_2 = capture_or_verify_code_image(args2, {}, log)

        # SC-2: only ONE pool dir under <rd>/Acme/ — the second call reused.
        org_root = rd / "Acme"
        pools = pool_dirs(org_root)
        assert len(pools) == 1, (
            f"SC-2 violated: expected 1 pool dir after two same-source calls, got {pools}"
        )
        assert Path(pool_dir_1) == Path(pool_dir_2) == pools[0]

        # Both run leaves have a .mlps-code-image pointer with identical content.
        pointers = sorted(rd.rglob(".mlps-code-image"))
        assert len(pointers) == 2, pointers
        contents = {p.read_text().strip() for p in pointers}
        assert len(contents) == 1, (
            f"pointer contents diverged between reused calls: {contents}"
        )

    def test_second_call_writes_pointer_in_new_run_leaf(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """SC-2 companion: two calls produce two DIFFERENT pointer files (one
        per run leaf), even though the pool image is reused — the pointer is
        run-scoped, not pool-scoped (D-61, PTR-01)."""
        rd = tmp_path / "results"
        rd.mkdir()

        args1 = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        with patch("mlpstorage_py.rules.utils.DATETIME_STR", "20260704_130000"):
            capture_or_verify_code_image(args1, {}, log)

        args2 = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        with patch("mlpstorage_py.rules.utils.DATETIME_STR", "20260704_130010"):
            capture_or_verify_code_image(args2, {}, log)

        pointers = sorted(rd.rglob(".mlps-code-image"))
        # Two distinct pointer FILES (different paths).
        assert len(pointers) == 2, pointers
        assert pointers[0] != pointers[1]
        # But same content (same live hash).
        assert pointers[0].read_text() == pointers[1].read_text()
