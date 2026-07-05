"""Integration coverage for D-66 (first-writer-wins concurrent capture) —
Phase 6 Plan 06-04 Task 6.

This test exercises D-66 on tmpfs which matches ext4 rename semantics (per
06-RESEARCH.md Pattern 3 empirical verification). Real-NFS coverage is
deferred to hardware verification per 06-VALIDATION.md Manual-Only
Verifications.

Two concurrent captures targeting the same content-addressed pool dir MUST
produce exactly one `code-<hash8>/` (D-66 first-writer-wins via
`os.rename` + non-empty tmp target). The race loser observes ENOTEMPTY,
reads the winner's `.code-hash.json`, verifies its `hash` matches the live
hash (byte-equal content), and silently returns the winner's path. No
`.code-<hash8>.tmp.<pid>/` sibling is leaked into `org_root` on either
branch (Pitfall 4 cleanup contract).

Also covers the D-66 loser branch directly: pre-seeding a pool dir with a
matching `.code-hash.json` before the capture call exercises the reuse path
that shares its verify semantics with the loser branch (`_find_matching_pool_image`
matches, `_capture_new_pool_image` is not called).

Fork strategy: each subprocess re-imports the code_image module and
monkeypatches `find_source_root` INSIDE the subprocess to the shared src
root path passed as a Process arg. The `fake_source_root` fixture in
`conftest.py` uses `monkeypatch.setattr` which does NOT survive `fork()`,
hence the per-process re-patch.

Refs: 06-04-PLAN.md Task 6, 06-CONTEXT.md D-66, POOL-01, POOL-04.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from pathlib import Path

from mlpstorage_py.submission_checker.tools.code_image import (
    capture_or_verify_code_image,
    compute_code_tree_md5,
)

from tests.integration.conftest import MockLogger, make_capture_args, pool_dirs


def _capture_worker(
    src_root_str: str,
    results_dir_str: str,
    stagger_seconds: float,
) -> None:
    """Subprocess entry point. Re-patches `find_source_root` to the shared
    `src_root_str` (fixture monkeypatch does not survive fork) and calls
    `capture_or_verify_code_image` once.

    stagger_seconds gives a small per-process delay so the two processes do
    not clash on identical `DATETIME_STR` when writing their run leaves
    (the pool image D-66 race is the invariant under test; the run-leaf
    write is best-effort and orthogonal to D-66).

    Exits with returncode 0 on success, non-zero on any raised exception.
    """
    import mlpstorage_py.submission_checker.tools.code_image as m
    from pathlib import Path as _P

    time.sleep(stagger_seconds)

    m.find_source_root = lambda: _P(src_root_str)

    log = MockLogger()
    args = make_capture_args(
        results_dir=results_dir_str,
        mode="closed",
        orgname="Acme",
        benchmark="training",
        command="run",
        model="unet3d",
    )
    # Any exception here → non-zero exit via SystemExit propagation.
    m.capture_or_verify_code_image(args, {}, log)


class TestConcurrentCapture:
    """D-66: two concurrent captures produce exactly one pool image."""

    def test_two_concurrent_captures_produce_one_pool_image(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """D-66 primary: fork two processes that both target the same
        content-addressed pool dir. Exactly one `code-<hash8>/` survives;
        no `.code-*.tmp.*` sibling leaks; the surviving pool has a valid
        `.code-hash.json`. Retried up to 5 iterations for stability against
        scheduler timing (the race is real; the invariant must hold every
        time)."""
        rd = tmp_path / "results"
        rd.mkdir()

        ctx = multiprocessing.get_context("fork")

        # 5 iterations — the D-66 invariant must hold every time.
        for iteration in range(5):
            iter_rd = rd / f"iter-{iteration}"
            iter_rd.mkdir()

            p1 = ctx.Process(
                target=_capture_worker,
                args=(str(fake_source_root), str(iter_rd), 0.0),
            )
            p2 = ctx.Process(
                target=_capture_worker,
                args=(str(fake_source_root), str(iter_rd), 0.01),
            )
            p1.start()
            p2.start()
            p1.join(timeout=30)
            p2.join(timeout=30)

            assert p1.exitcode == 0, (
                f"iter {iteration}: worker 1 crashed with exitcode {p1.exitcode}"
            )
            assert p2.exitcode == 0, (
                f"iter {iteration}: worker 2 crashed with exitcode {p2.exitcode}"
            )

            org_root = iter_rd / "Acme"
            pools = pool_dirs(org_root)
            assert len(pools) == 1, (
                f"iter {iteration}: D-66 violated — expected 1 pool dir, got {pools}"
            )

            # No leaked .tmp sibling (Pitfall 4 cleanup contract).
            tmp_leftovers = list(org_root.glob(".code-*.tmp.*"))
            assert tmp_leftovers == [], (
                f"iter {iteration}: leaked tmp siblings — {tmp_leftovers}"
            )

            # Surviving pool has a valid .code-hash.json.
            sidecar = pools[0] / ".code-hash.json"
            assert sidecar.is_file(), f"iter {iteration}: missing sidecar in {pools[0]}"
            data = json.loads(sidecar.read_text())
            assert len(data["hash"]) == 32
            assert data["algorithm"] == "md5-tree-v2"

            # At least one run leaf received the pointer file. The two
            # workers may collide on DATETIME_STR (same-second launch) and
            # share a run leaf, so we allow 1 or 2 pointer files.
            pointers = list(iter_rd.rglob(".mlps-code-image"))
            assert 1 <= len(pointers) <= 2, (
                f"iter {iteration}: expected 1-2 pointer files, got {pointers}"
            )
            for pointer in pointers:
                assert pointer.read_text().strip() == f"md5-tree-v2:{data['hash']}"

    def test_pre_seeded_pool_matches_live_hash_loser_branch_returns_winner(
        self, tmp_path, fake_source_root, capture_args_factory, log
    ):
        """D-66 loser-branch shape: pre-seed a matching pool dir before the
        capture call. `_find_matching_pool_image` finds it, capture reuses
        without writing a new pool. This exercises the same
        content-verification contract the D-66 race loser branch runs
        (`.code-hash.json.hash` == live_hash → silent success)."""
        rd = tmp_path / "results"
        rd.mkdir()

        # Compute the live hash of the fake source tree first so we know
        # what pool dir name to pre-seed.
        full_live_hash = compute_code_tree_md5(str(fake_source_root), log)
        assert full_live_hash is not None
        hash8 = full_live_hash[:8]

        # Pre-seed <rd>/Acme/code-<hash8>/ with a matching .code-hash.json.
        org_root = rd / "Acme"
        org_root.mkdir()
        pre_pool = org_root / f"code-{hash8}"
        pre_pool.mkdir()
        (pre_pool / ".code-hash.json").write_text(json.dumps({
            "hash": full_live_hash,
            "algorithm": "md5-tree-v2",
            "captured_at": "2026-07-04T00:00:00Z",
            "mlpstorage_version": "pre-seeded",
            "git_sha": None,
        }))

        args = capture_args_factory(
            results_dir=rd, mode="closed", orgname="Acme",
            benchmark="training", command="run", model="unet3d",
        )
        returned = capture_or_verify_code_image(args, {}, log)

        # Reuse: returned pool is the pre-seeded one — no new dir written.
        assert Path(returned) == pre_pool

        # Still exactly one pool dir under <rd>/Acme/.
        pools = pool_dirs(org_root)
        assert len(pools) == 1
        assert pools[0] == pre_pool

        # Pointer written in the run leaf with the live hash.
        pointers = list(rd.rglob(".mlps-code-image"))
        assert len(pointers) == 1, pointers
        assert pointers[0].read_text().strip() == f"md5-tree-v2:{full_live_hash}"
