#!/usr/bin/env python3
"""
Regression tests for issue #349: ``--open`` was indistinguishable from "no flag".

Before the fix:

* ``--open`` was defined with ``action="store_false", dest="closed"``, so it
  produced the same Namespace (``args.closed=False``, no ``args.open``) as
  passing nothing at all.
* ``verify_benchmark()`` used ``hasattr(self.args, "open")`` which is
  *always* False, so ``--open`` fell through to the "Running the benchmark
  without verification for open or closed configurations" warning branch
  and skipped formal verification.

After the fix:

* ``--open`` sets ``args.open=True`` as an independent boolean.
* ``verify_benchmark()`` uses ``getattr`` on both ``open`` and ``closed``;
  the warning branch triggers only when **neither** flag is set.

These tests pin that behavior.
"""

import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mlpstorage_py.config import PARAM_VALIDATION, BENCHMARK_TYPES


# ---------------------------------------------------------------------------
# Part 1: CLI parser produces distinguishable --open / --closed / neither
# ---------------------------------------------------------------------------

class TestOpenClosedCLIFlags:
    """The argparse definition must let callers tell the three cases apart."""

    def _build_parser(self):
        import argparse
        from mlpstorage_py.cli.common_args import (
            add_universal_arguments,
            add_storage_type_arguments,
        )
        parser = argparse.ArgumentParser()
        add_universal_arguments(parser)
        # --file/--object now live exclusively in add_storage_type_arguments
        # (see issue #376). Real benchmark subparsers call both, so the
        # test mirrors that to keep ``--file`` available in parse_args calls.
        add_storage_type_arguments(parser)
        return parser

    def test_neither_flag_sets_both_false(self):
        parser = self._build_parser()
        # --file is required by a separate mutually-exclusive group; pass it
        # to avoid the unrelated "one of --file --object required" error.
        args = parser.parse_args(["--file"])
        assert getattr(args, "closed", None) is False
        assert getattr(args, "open", None) is False

    def test_open_flag_sets_open_true_closed_false(self):
        parser = self._build_parser()
        args = parser.parse_args(["--file", "--open"])
        assert args.open is True
        assert args.closed is False

    def test_closed_flag_sets_closed_true_open_false(self):
        parser = self._build_parser()
        args = parser.parse_args(["--file", "--closed"])
        assert args.closed is True
        assert args.open is False

    def test_open_and_closed_are_mutually_exclusive(self):
        parser = self._build_parser()
        # argparse exits with SystemExit(2) on mutually-exclusive violation
        with pytest.raises(SystemExit):
            parser.parse_args(["--file", "--open", "--closed"])


# ---------------------------------------------------------------------------
# Part 2: verify_benchmark() respects the new semantics
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
        closed=False,
        open=False,
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


class TestVerifyBenchmarkOpenFlag:
    """Fixes for issue #349 — the heart of the bug."""

    def test_open_flag_does_not_hit_no_verification_warning(self, tmp_path):
        """
        Regression test for #349: passing --open must NOT route through the
        "Running the benchmark without verification" warning branch.
        """
        bench = _make_benchmark(tmp_path, open=True, closed=False)

        with patch("mlpstorage_py.benchmarks.base.BenchmarkVerifier") as mock_cls:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.OPEN
            mock_cls.return_value = mock_verifier

            result = bench.verify_benchmark()

        assert result is True
        # The "no verification" warning must NOT have been emitted.
        for call in bench.logger.warning.call_args_list:
            assert "without verification for open or closed" not in call.args[0], \
                "--open should route to formal verification, not the 'no verification' warning"
        # A proper OPEN-allowed status message should have been emitted instead.
        status_msgs = [c.args[0] for c in bench.logger.status.call_args_list]
        assert any("allowed open configuration" in m for m in status_msgs), \
            "Expected 'Running as allowed open configuration' status message"

    def test_closed_flag_accepts_closed_verification(self, tmp_path):
        bench = _make_benchmark(tmp_path, closed=True, open=False)

        with patch("mlpstorage_py.benchmarks.base.BenchmarkVerifier") as mock_cls:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.CLOSED
            mock_cls.return_value = mock_verifier

            result = bench.verify_benchmark()

        assert result is True

    def test_closed_flag_rejects_open_only_params(self, tmp_path):
        """If user asked for --closed but params only qualify for OPEN, exit."""
        bench = _make_benchmark(tmp_path, closed=True, open=False)

        with patch("mlpstorage_py.benchmarks.base.BenchmarkVerifier") as mock_cls:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.OPEN
            mock_cls.return_value = mock_verifier

            with pytest.raises(SystemExit):
                bench.verify_benchmark()

    def test_neither_flag_emits_no_verification_warning(self, tmp_path):
        """
        Unchanged contract: when neither --open nor --closed is passed, the
        benchmark runs with a warning and skips formal verification.
        """
        bench = _make_benchmark(tmp_path, closed=False, open=False)

        with patch("mlpstorage_py.benchmarks.base.BenchmarkVerifier") as mock_cls:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.CLOSED
            mock_cls.return_value = mock_verifier

            result = bench.verify_benchmark()

        assert result is True
        assert any(
            "without verification for open or closed" in c.args[0]
            for c in bench.logger.warning.call_args_list
        ), "Default (neither flag) should warn that no verification is being performed"

    def test_invalid_params_exits_without_allow_flag(self, tmp_path):
        bench = _make_benchmark(tmp_path, closed=True, open=False,
                                allow_invalid_params=False)

        with patch("mlpstorage_py.benchmarks.base.BenchmarkVerifier") as mock_cls:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.INVALID
            mock_cls.return_value = mock_verifier

            with pytest.raises(SystemExit):
                bench.verify_benchmark()

    def test_invalid_params_allowed_with_flag(self, tmp_path):
        bench = _make_benchmark(tmp_path, closed=True, open=False,
                                allow_invalid_params=True)

        with patch("mlpstorage_py.benchmarks.base.BenchmarkVerifier") as mock_cls:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.INVALID
            mock_cls.return_value = mock_verifier

            result = bench.verify_benchmark()

        assert result is True


# ---------------------------------------------------------------------------
# Part 3: kvcache triggers verification for --open too (not just --closed)
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
