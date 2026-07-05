"""MIG-02 idempotency and crash-safety tests.

Covers Phase 7 decisions D-70/D-71 and requirement MIG-02:
  - MIG-02a: sentinel short-circuit — second invocation skips migration scan
    when <rd>/<org>/.mlps-image-pool sentinel is already present (O(2) syscalls)
  - MIG-02b: crash-resumability — re-invoking migration after a simulated SIGKILL
    at any of the four D-71 step boundaries converges to the same final state as
    an uninterrupted migration

D-71 checkpoint matrix (four SIGKILL moments):
  1. After step 1 (pool images materialized), before step 2 (pointer writes)
  2. After step 2 (pointer writes), before step 3 (legacy dir deletion)
  3. After step 3 (legacy dirs deleted), before step 4 (sentinel write)
  4. Mid-step 2: partial pointer write (Nth of M leaves written, then crash)

Refs: 07-01-PLAN.md Task 2, 07-CONTEXT.md D-70/D-71, MIG-02,
RESEARCH §10 crash-safety checkpoint matrix.
"""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock

import pytest

import mlpstorage_py.submission_checker.tools.legacy_migration as lm
from mlpstorage_py.submission_checker.tools.legacy_migration import (
    _check_and_migrate_legacy_layout,
    migrate_legacy_layout,
)
from tests.integration.conftest import pool_dirs


class _AbortAtStep(Exception):
    """Sentinel exception injected by monkeypatch to simulate SIGKILL at a D-71 step boundary."""


class TestSentinelShortCircuit:
    """MIG-02a: sentinel presence causes immediate skip on second invocation."""

    def test_second_invocation_skips_scan_when_sentinel_present(
        self, tmp_path, legacy_tree_factory, log, monkeypatch
    ):
        """MIG-02a: second call with sentinel present skips _scan_legacy_layout.

        After a successful first migration (sentinel written), monkeypatch spy
        on ``_scan_legacy_layout`` and invoke _check_and_migrate_legacy_layout a
        second time. Assert spy.call_count == 0 — the sentinel short-circuits
        before the scan reaches the filesystem (D-70).
        """
        rd = legacy_tree_factory(orgname="Acme", n_run_leaves=2)

        # First call: migrate normally so the sentinel is written.
        migrate_legacy_layout(rd, "Acme", log)
        assert (rd / "Acme" / ".mlps-image-pool").exists()

        # Spy on _scan_legacy_layout to confirm it is NOT called.
        spy = MagicMock(wraps=lm._scan_legacy_layout)
        monkeypatch.setattr(lm, "_scan_legacy_layout", spy)

        # Clear logger accumulators before second call.
        log.statuses.clear()
        log.debugs.clear()
        log.infos.clear()
        log.warnings.clear()
        log.errors.clear()

        # Build args namespace for the pre-check helper.
        args = Namespace(
            mode="closed",
            command="run",
            results_dir=str(rd),
            orgname="Acme",
            systemname=None,
        )

        # Second call — sentinel present; scan must be skipped.
        _check_and_migrate_legacy_layout(args, {}, log)

        assert spy.call_count == 0, (
            "D-70: sentinel present → _scan_legacy_layout MUST NOT be called "
            f"(call_count={spy.call_count})"
        )

    def test_second_invocation_emits_zero_log_lines(
        self, tmp_path, legacy_tree_factory, log
    ):
        """MIG-02a: second call emits zero status and debug log lines.

        After a successful first migration, clear all logger lists and invoke
        _check_and_migrate_legacy_layout again. Assert every accumulator
        remains empty — the sentinel short-circuit path is completely silent.
        """
        rd = legacy_tree_factory(orgname="Acme", n_run_leaves=2)

        # First call: migrate normally so the sentinel is written.
        migrate_legacy_layout(rd, "Acme", log)
        assert (rd / "Acme" / ".mlps-image-pool").exists()

        # Clear ALL logger accumulators.
        log.statuses.clear()
        log.debugs.clear()
        log.infos.clear()
        log.warnings.clear()
        log.errors.clear()

        # Build args namespace for the pre-check helper.
        args = Namespace(
            mode="closed",
            command="run",
            results_dir=str(rd),
            orgname="Acme",
            systemname=None,
        )

        # Second call — sentinel present; no log output expected.
        _check_and_migrate_legacy_layout(args, {}, log)

        assert log.statuses == [], f"Expected no status entries, got: {log.statuses}"
        assert log.debugs == [], f"Expected no debug entries, got: {log.debugs}"
        assert log.infos == [], f"Expected no info entries, got: {log.infos}"
        assert log.warnings == [], f"Expected no warning entries, got: {log.warnings}"
        assert log.errors == [], f"Expected no error entries, got: {log.errors}"


class TestCrashResume:
    """MIG-02b: crash-resumability via simulated SIGKILL at D-71 checkpoints.

    Each test patches a step function to raise _AbortAtStep at a specific
    boundary, catches the resulting exception (simulating SIGKILL), then
    re-invokes migrate_legacy_layout and asserts the final state is identical
    to an uninterrupted run (RESEARCH §10 crash-safety checkpoint matrix).
    """

    def test_crash_after_step_1_materialize_before_step_2_pointers(
        self, tmp_path, legacy_tree_factory, monkeypatch, log
    ):
        """D-71 checkpoint 1: crash after pool images materialized, before pointer writes.

        Patch ``_write_pointers_for_migrated_leaves`` to raise _AbortAtStep on
        first call (simulating SIGKILL between step 1 and step 2). Catch the
        error, then re-invoke migrate_legacy_layout. Assert convergence: sentinel
        present, all run leaves have pointers, legacy code/ deleted.
        """
        rd = legacy_tree_factory(orgname="Acme", n_run_leaves=3)
        org_root = rd / "Acme"

        # Patch step 2 to raise before writing any pointers.
        def _fail(*a, **kw):
            raise _AbortAtStep("simulated SIGKILL after step 1")

        monkeypatch.setattr(lm, "_write_pointers_for_migrated_leaves", _fail)

        # First invocation: crashes mid-migration.
        with pytest.raises(_AbortAtStep):
            migrate_legacy_layout(rd, "Acme", log)

        # Post-crash partial state: step 1 succeeded, step 2 onward did not.
        assert len(pool_dirs(org_root)) >= 1, "step 1 must have materialized ≥1 pool image"
        assert (rd / "closed" / "Acme" / "code").is_dir(), "step 3 must NOT have deleted legacy code/"
        assert not (org_root / ".mlps-image-pool").exists(), "step 4 must NOT have written sentinel"
        pointers = list(rd.rglob(".mlps-code-image"))
        assert pointers == [], f"step 2 must NOT have written any pointers, got: {pointers}"

        # Restore the real implementation and re-invoke.
        monkeypatch.undo()
        migrate_legacy_layout(rd, "Acme", log)

        # Final converged state: sentinel present, legacy gone, all 3 leaves have pointers.
        assert (org_root / ".mlps-image-pool").exists(), "sentinel must be present after re-invoke"
        assert not (rd / "closed" / "Acme" / "code").exists(), "legacy code/ must be deleted after re-invoke"
        pointers = list(rd.rglob(".mlps-code-image"))
        assert len(pointers) == 3, f"all 3 run leaves must have pointer files, got {len(pointers)}"

    def test_crash_after_step_2_pointers_before_step_3_delete(
        self, tmp_path, legacy_tree_factory, monkeypatch, log
    ):
        """D-71 checkpoint 2: crash after pointer writes, before legacy dir deletion.

        Patch ``_delete_legacy_dirs`` to raise _AbortAtStep on first call.
        Catch, then re-invoke. Assert convergence: sentinel present, all run
        leaves have pointers, legacy code/ dirs deleted on the resume run.
        """
        rd = legacy_tree_factory(orgname="Acme", n_run_leaves=3)
        org_root = rd / "Acme"

        # Patch step 3 to raise before deleting legacy dirs.
        def _fail(*a, **kw):
            raise _AbortAtStep("simulated SIGKILL after step 2")

        monkeypatch.setattr(lm, "_delete_legacy_dirs", _fail)

        # First invocation: crashes mid-migration.
        with pytest.raises(_AbortAtStep):
            migrate_legacy_layout(rd, "Acme", log)

        # Post-crash partial state: steps 1+2 succeeded, steps 3+4 did not.
        assert len(pool_dirs(org_root)) >= 1, "step 1 must have materialized ≥1 pool image"
        assert (rd / "closed" / "Acme" / "code").is_dir(), "step 3 must NOT have deleted legacy code/"
        assert not (org_root / ".mlps-image-pool").exists(), "step 4 must NOT have written sentinel"
        pointers = list(rd.rglob(".mlps-code-image"))
        assert len(pointers) == 3, f"step 2 must have written all 3 pointers, got {len(pointers)}"

        # Restore the real implementation and re-invoke.
        monkeypatch.undo()
        migrate_legacy_layout(rd, "Acme", log)

        # Final converged state.
        assert (org_root / ".mlps-image-pool").exists(), "sentinel must be present after re-invoke"
        assert not (rd / "closed" / "Acme" / "code").exists(), "legacy code/ must be deleted after re-invoke"
        pointers = list(rd.rglob(".mlps-code-image"))
        assert len(pointers) == 3, f"all 3 run leaves must have pointer files, got {len(pointers)}"

    def test_crash_after_step_3_delete_before_step_4_sentinel(
        self, tmp_path, legacy_tree_factory, monkeypatch, log
    ):
        """D-71 checkpoint 3: crash after legacy dirs deleted, before sentinel write.

        Patch ``_write_sentinel_atomic`` to raise _AbortAtStep on first call.
        Catch, then re-invoke. Assert convergence: sentinel present after resume,
        all run leaves still have pointers.

        Note: on re-invoke, _verify_all_legacy_dirs returns [] (legacy code/ is
        gone). migrate_legacy_layout takes the empty-verified branch, which calls
        _write_sentinel_atomic directly and returns. The sentinel is written
        silently (no log.status lines for N=0 per D-74).
        """
        rd = legacy_tree_factory(orgname="Acme", n_run_leaves=3)
        org_root = rd / "Acme"

        # Patch step 4 to raise before writing the sentinel.
        def _fail(*a, **kw):
            raise _AbortAtStep("simulated SIGKILL after step 3")

        monkeypatch.setattr(lm, "_write_sentinel_atomic", _fail)

        # First invocation: crashes mid-migration.
        with pytest.raises(_AbortAtStep):
            migrate_legacy_layout(rd, "Acme", log)

        # Post-crash partial state: steps 1+2+3 succeeded, step 4 did not.
        assert len(pool_dirs(org_root)) >= 1, "step 1 must have materialized ≥1 pool image"
        assert not (rd / "closed" / "Acme" / "code").exists(), "step 3 must have deleted legacy code/"
        assert not (org_root / ".mlps-image-pool").exists(), "step 4 must NOT have written sentinel"
        pointers = list(rd.rglob(".mlps-code-image"))
        assert len(pointers) == 3, f"step 2 must have written all 3 pointers, got {len(pointers)}"

        # Restore the real implementation and re-invoke.
        monkeypatch.undo()
        migrate_legacy_layout(rd, "Acme", log)

        # Final converged state: sentinel written via the empty-verified branch.
        assert (org_root / ".mlps-image-pool").exists(), "sentinel must be present after re-invoke"
        # Legacy code/ is already gone (step 3 ran on first invocation).
        assert not (rd / "closed" / "Acme" / "code").exists(), "legacy code/ must still be absent"
        pointers = list(rd.rglob(".mlps-code-image"))
        assert len(pointers) == 3, f"all 3 pointer files must still be present, got {len(pointers)}"

    def test_crash_mid_step_2_partial_pointer_writes(
        self, tmp_path, legacy_tree_factory, monkeypatch, log
    ):
        """D-71 mid-step-2 variant: crash after Nth pointer write of M total.

        Patch ``_write_pointer_atomic`` to succeed on call 1, then raise
        _AbortAtStep on call 2+ — simulating a partial pointer-write crash per
        RESEARCH §Additional crash-safety observation. Re-invoke and assert
        all 3 leaves end up with pointers (the partial writes are idempotent
        via atomic os.rename, so re-running overwrites cleanly).
        """
        rd = legacy_tree_factory(orgname="Acme", n_run_leaves=3)
        org_root = rd / "Acme"

        # Stateful mock: let first call succeed, raise on 2nd and later calls.
        orig = lm._write_pointer_atomic
        call_count = {"n": 0}

        def _stateful(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise _AbortAtStep("simulated SIGKILL mid pointer-write")
            return orig(*a, **kw)

        monkeypatch.setattr(lm, "_write_pointer_atomic", _stateful)

        # First invocation: crashes after writing exactly 1 pointer.
        with pytest.raises(_AbortAtStep):
            migrate_legacy_layout(rd, "Acme", log)

        # Post-crash partial state: step 1 succeeded; step 2 partially succeeded
        # (exactly 1 pointer written before the crash).
        assert len(pool_dirs(org_root)) >= 1, "step 1 must have materialized ≥1 pool image"
        assert (rd / "closed" / "Acme" / "code").is_dir(), "step 3 must NOT have deleted legacy code/"
        assert not (org_root / ".mlps-image-pool").exists(), "step 4 must NOT have written sentinel"
        pointers = list(rd.rglob(".mlps-code-image"))
        assert len(pointers) == 1, (
            f"exactly 1 pointer must exist after partial step-2 crash, got {len(pointers)}"
        )

        # Restore the real _write_pointer_atomic and re-invoke.
        monkeypatch.undo()
        migrate_legacy_layout(rd, "Acme", log)

        # Final converged state: all 3 leaves now have pointers (atomic idempotent writes).
        assert (org_root / ".mlps-image-pool").exists(), "sentinel must be present after re-invoke"
        assert not (rd / "closed" / "Acme" / "code").exists(), "legacy code/ must be deleted after re-invoke"
        pointers = list(rd.rglob(".mlps-code-image"))
        assert len(pointers) == 3, (
            f"all 3 run leaves must have pointer files after re-invoke, got {len(pointers)}"
        )
