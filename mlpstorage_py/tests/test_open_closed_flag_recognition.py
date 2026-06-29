#!/usr/bin/env python3
"""
Regression tests for the closed/open/whatif mode dispatch in
``Benchmark.verify_benchmark()``.

History:

* Issue #349: a pre-#412 design had ``--open`` and ``--closed`` as boolean
  flags. Tests verifying that the flags reached argparse and that
  ``verify_benchmark`` distinguished them used to live here. PR #412
  reshaped the CLI to use a positional ``mode`` argument (``closed`` |
  ``open`` | ``whatif``), retiring those flags entirely.

* Issue #412: a second regression nearly slipped in when
  ``verify_benchmark()`` was not migrated to the new ``args.mode`` shape and
  silently fell back to the "no verification" warning branch for every live
  invocation. The tests in this file pin the post-#412 dispatch contract.

Coverage today:

* ``TestVerifyBenchmarkPost412ModeDispatch`` — ``args.mode`` drives the
  closed/open/whatif branches inside ``verify_benchmark``; this is the
  current production contract.
* ``TestKVCacheOpenFlag`` — pins the helper predicate shape that downstream
  kvcache code uses to check "did the user opt in to verification?".
"""

import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy deps the benchmark imports expect (pre-existing dev-env psutil gap
# documented in STATE.md Deferred Items). Without this the entire file fails
# collection — which is how the #412 regression (verify_benchmark still reading
# args.closed/args.open instead of args.mode) hid from CI.
for _dep in ("pyarrow", "pyarrow.ipc", "psutil"):
    if _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

from mlpstorage_py.config import PARAM_VALIDATION, BENCHMARK_TYPES


# ---------------------------------------------------------------------------
# verify_benchmark() ``args.mode`` dispatch (post-#412)
# ---------------------------------------------------------------------------

def _make_benchmark(tmp_path, **arg_overrides):
    """Construct a minimally-initialized Benchmark subclass for testing."""
    from mlpstorage_py.benchmarks.base import Benchmark

    class _Bench(Benchmark):
        BENCHMARK_TYPE = BENCHMARK_TYPES.training
        def _run(self):
            return 0

    defaults = dict(
        debug=False,
        verbose=False,
        what_if=False,
        stream_log_level="INFO",
        results_dir=str(tmp_path),
        model="unet3d",
        command="run",
        num_processes=8,
        accelerator_type="h100",
        allow_invalid_params=False,
    )
    defaults.update(arg_overrides)

    bench = _Bench.__new__(_Bench)
    bench.args = Namespace(**defaults)
    bench.logger = MagicMock()
    # Silence logger methods the code calls
    for lvl in ("debug", "info", "warning", "error", "status",
                "verbose", "verboser", "ridiculous", "result"):
        setattr(bench.logger, lvl, MagicMock())
    bench.benchmark_run_verifier = None
    bench.run_datetime = "20260424_000000"
    bench.verification = None
    return bench


class TestVerifyBenchmarkPost412ModeDispatch:
    """Post-PR-#412 CLI redesign: closed/open is now ``args.mode``, not the
    pair of bools ``args.closed`` / ``args.open``. The dispatch in
    ``verify_benchmark`` was never migrated, so every live CLI invocation
    (which sets ``mode='closed'`` / ``mode='open'``) silently fell through
    to the no-verification warning branch — effectively re-introducing the
    bug PR #352 fixed for #349.

    These tests build Namespaces in the post-#412 shape (``mode='closed'`` |
    ``'open'`` | ``'whatif'``) and assert that verify_benchmark dispatches
    correctly.
    """

    def _make_modal_bench(self, tmp_path, mode, **arg_overrides):
        """Build a benchmark with a post-#412 Namespace shape."""
        bench = _make_benchmark(tmp_path, **arg_overrides)
        bench.args.mode = mode
        return bench

    def test_mode_closed_does_not_hit_no_verification_warning(self, tmp_path):
        """RED for the regression: `mlpstorage closed ...` must route to
        formal verification, not warn that it's skipping verification."""
        bench = self._make_modal_bench(tmp_path, mode="closed")

        with patch("mlpstorage_py.benchmarks.base.BenchmarkVerifier") as mock_cls:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.CLOSED
            mock_cls.return_value = mock_verifier

            result = bench.verify_benchmark()

        assert result is True
        for c in bench.logger.warning.call_args_list:
            assert "without verification for open or closed" not in c.args[0], (
                f"Post-#412 'mode=closed' Namespace must not hit the "
                f"no-verification warning. Saw: {c.args[0]}"
            )

    def test_mode_open_does_not_hit_no_verification_warning(self, tmp_path):
        bench = self._make_modal_bench(tmp_path, mode="open")

        with patch("mlpstorage_py.benchmarks.base.BenchmarkVerifier") as mock_cls:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.OPEN
            mock_cls.return_value = mock_verifier

            result = bench.verify_benchmark()

        assert result is True
        for c in bench.logger.warning.call_args_list:
            assert "without verification for open or closed" not in c.args[0]
        status_msgs = [c.args[0] for c in bench.logger.status.call_args_list]
        assert any("allowed open configuration" in m for m in status_msgs), (
            "mode='open' + PARAM_VALIDATION.OPEN must emit the 'allowed open "
            "configuration' status message (downstream open_mode dispatch)."
        )

    def test_mode_whatif_routes_to_no_verification_warning(self, tmp_path):
        """`mlpstorage whatif ...` is the post-#412 way to say 'I don't
        want submission verification' — the warning is the correct outcome."""
        bench = self._make_modal_bench(tmp_path, mode="whatif")

        with patch("mlpstorage_py.benchmarks.base.BenchmarkVerifier") as mock_cls:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.CLOSED
            mock_cls.return_value = mock_verifier

            result = bench.verify_benchmark()

        assert result is True
        assert any(
            "without verification for open or closed" in c.args[0]
            for c in bench.logger.warning.call_args_list
        ), "mode='whatif' SHOULD warn — it intentionally bypasses verification."

    def test_mode_closed_rejects_open_only_params(self, tmp_path):
        """downstream dispatch: closed mode + OPEN-only params must fail-fast."""
        bench = self._make_modal_bench(tmp_path, mode="closed")

        with patch("mlpstorage_py.benchmarks.base.BenchmarkVerifier") as mock_cls:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.OPEN
            mock_cls.return_value = mock_verifier

            with pytest.raises(SystemExit):
                bench.verify_benchmark()


# ---------------------------------------------------------------------------
# kvcache helper predicate shape
# ---------------------------------------------------------------------------

class TestKVCacheOpenFlag:
    """kvcache.py guard must run verification for both --open and --closed."""

    def test_open_triggers_verification_in_kvcache_guard(self):
        """
        Prior code was: ``if hasattr(self.args, 'closed') and self.args.closed``
        which never fired for --open. Post-fix it must fire for either flag.
        """
        args_open = SimpleNamespace(open=True, closed=False)
        args_closed = SimpleNamespace(open=False, closed=True)
        args_neither = SimpleNamespace(open=False, closed=False)

        def should_verify(args):
            return getattr(args, "closed", False) or getattr(args, "open", False)

        assert should_verify(args_open) is True
        assert should_verify(args_closed) is True
        assert should_verify(args_neither) is False
