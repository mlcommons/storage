"""Tests for issue #644 — CAP-01 capacity-gate math in checkpointing subset mode.

``CheckpointingBenchmark.required_bytes_for_capacity_gate`` (and the
duplicated ``datasize`` printout) currently uses ``self.args.num_processes``
as the divisor in the per-rank checkpoint-size math. That formula was
written assuming ``num_processes == ClosedGPUs`` (full CLOSED run); in
subset mode (``--num-processes < ClosedGPUs``), the per-rank share
should still be computed against the full-run denominator ``ClosedGPUs``
(zero_level=3) or ``ClosedGPUs`` for the optimizer term / ``GPUpDP`` for
the model term (zero_level=1). The bug over-counts by a factor of
``ClosedGPUs / num_processes`` and blocks legitimate subset runs at
CAP-01.

Reporter's specific case (llama3-70b, 8 procs, 10 checkpoints on a
6.5 TB drive): current math reports ~9.1 TB required (blocks), correct
math reports ~1.14 TB (would pass).

Tests hit the arithmetic through the public
``required_bytes_for_capacity_gate`` API and construct the bench via
``__new__`` to bypass ``__init__``'s DLIO config loading (irrelevant to
the byte-count math).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_bench(model: str, num_processes: int, num_ckpts: int):
    """Construct a bare CheckpointingBenchmark whose only live state is
    the args the size math consults. ``__new__`` bypasses ``__init__``
    (which loads YAML DLIO configs, applies object-storage params, etc.)
    since none of that matters to the per-rank GiB arithmetic."""
    from mlpstorage_py.benchmarks.dlio import CheckpointingBenchmark
    bench = CheckpointingBenchmark.__new__(CheckpointingBenchmark)
    bench.args = SimpleNamespace(
        model=model,
        num_processes=num_processes,
        num_checkpoints_write=num_ckpts,
    )
    return bench


def _expected_bytes(model: str, num_processes: int, num_ckpts: int) -> int:
    """Reference implementation of the CORRECT subset-aware math.
    Reproduces the formula the fix should install; tests compare against
    this rather than a hardcoded byte count so future model additions
    don't require test edits."""
    from mlpstorage_py.config import LLM_ALLOWED_VALUES, LLM_SIZE_BY_RANK
    min_procs, zero_level, GPUpDP, ClosedGPUs = LLM_ALLOWED_VALUES[model]
    model_gb, opt_gb = LLM_SIZE_BY_RANK[model]
    rank_gb = []
    for rank in range(num_processes):
        if zero_level == 3:
            rank_gb.append((model_gb + opt_gb) / ClosedGPUs)
        elif zero_level == 1:
            share = opt_gb / ClosedGPUs
            if rank < GPUpDP:
                share += model_gb / GPUpDP
            rank_gb.append(share)
        else:
            raise ValueError(zero_level)
    return int(sum(rank_gb) * 1024**3 * num_ckpts)


class TestSubsetModeCapacityGate:
    """Subset runs must scale required_bytes to ``num_processes /
    ClosedGPUs`` of the full-mode total. RED on main because the current
    formula uses ``/ num_processes`` in the denominator and yields the
    full-mode total regardless of subset size."""

    def test_llama3_70b_subset_reporter_case(self):
        """Reporter's exact reproducer: 8 procs, 10 checkpoints. Correct
        answer is ~1.14 TB; buggy answer is ~9.1 TB. This test locks the
        byte count exactly so the regression is grep-able."""
        bench = _make_bench("llama3-70b", num_processes=8, num_ckpts=10)
        got = bench.required_bytes_for_capacity_gate()
        want = _expected_bytes("llama3-70b", 8, 10)
        assert got == want, (
            f"llama3-70b subset (8 procs, 10 ckpts): got {got:,} "
            f"({got / 1024**4:.2f} TiB), want {want:,} "
            f"({want / 1024**4:.2f} TiB)"
        )
        # Sanity: the correct value must comfortably fit on the
        # reporter's 6.5 TiB drive; a regression that reintroduces
        # full-mode math would clear the 6.5 TiB threshold.
        assert got < 2 * 1024**4, (
            f"llama3-70b subset must not exceed 2 TiB; got {got:,}"
        )

    def test_llama3_405b_subset_zero_level_1(self):
        """zero_level=1 subset (llama3-405b, 8 procs). Exercises the
        second buggy branch — optimizer term uses ``/ num_processes``
        instead of ``/ ClosedGPUs``. All 8 subset ranks are < GPUpDP=256
        so all write the model term."""
        bench = _make_bench("llama3-405b", num_processes=8, num_ckpts=10)
        got = bench.required_bytes_for_capacity_gate()
        want = _expected_bytes("llama3-405b", 8, 10)
        assert got == want, (
            f"llama3-405b subset (8 procs, 10 ckpts): got {got:,}, "
            f"want {want:,}"
        )

    def test_llama3_1t_subset_zero_level_1(self):
        """zero_level=1 subset on the largest model (llama3-1t, 8 procs,
        ClosedGPUs=1024)."""
        bench = _make_bench("llama3-1t", num_processes=8, num_ckpts=10)
        got = bench.required_bytes_for_capacity_gate()
        want = _expected_bytes("llama3-1t", 8, 10)
        assert got == want


class TestFullModeCapacityGateUnchanged:
    """Full CLOSED runs (num_processes == ClosedGPUs) must be
    unchanged. The fix should be a no-op in this regime because
    ``1/ClosedGPUs == 1/num_processes`` when the two are equal. These
    are GREEN on main and must stay GREEN after the fix — proves no
    full-mode regression."""

    @pytest.mark.parametrize(
        "model,full_procs",
        [
            ("llama3-8b", 8),
            ("llama3-70b", 64),
            ("llama3-405b", 512),
            ("llama3-1t", 1024),
        ],
    )
    def test_full_mode_bytes_unchanged(self, model, full_procs):
        bench = _make_bench(model, num_processes=full_procs, num_ckpts=10)
        got = bench.required_bytes_for_capacity_gate()
        want = _expected_bytes(model, full_procs, 10)
        assert got == want, (
            f"{model} full-mode ({full_procs} procs): got {got:,}, "
            f"want {want:,}"
        )


class TestSubsetScalingInvariant:
    """Cross-check the scaling relationship directly: subset bytes must
    equal full-mode bytes × (num_processes / ClosedGPUs) for zero_level=3.
    RED on main because subset returns == full."""

    def test_llama3_70b_subset_is_one_eighth_of_full(self):
        subset = _make_bench("llama3-70b", num_processes=8, num_ckpts=10)
        full = _make_bench("llama3-70b", num_processes=64, num_ckpts=10)
        subset_bytes = subset.required_bytes_for_capacity_gate()
        full_bytes = full.required_bytes_for_capacity_gate()
        # 8 / 64 = 1/8; subset should be one-eighth of full.
        assert subset_bytes * 8 == full_bytes, (
            f"llama3-70b subset (8) × 8 must equal full (64): "
            f"subset={subset_bytes:,}, full={full_bytes:,}, "
            f"ratio={full_bytes / subset_bytes if subset_bytes else 'inf'}"
        )
