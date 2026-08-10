"""
Regression tests for mlcommons/storage#841 — runtime-prevention half.

Rules.md §4.3.5 is missing the word "not": the published text errors a subset
run when "the model is '8B'", where the intent is "the model is *not* '8B'".
Subset mode exists for exactly one case — the 8B workload on a single 8-GPU
node, marking a local-NVMe architecture's claim of linear scale-out. No subset
form is defined for the larger models; they exist to measure shared central
storage at full scale.

The inversion propagated into the tool: ``check_num_processes`` (the #792
fail-fast gate) accepts 8 processes as a valid CLOSED form for *every* model,
and no explicit subset CLI parameter exists at all (§4.3.5's first sentence),
so subset-ness is only ever auto-inferred.

This module pins the corrected runtime behavior:

* ``TestCheckpointSubsetCliFlag`` — ``--checkpoint-subset`` exists on the
  checkpointing subcommands and lands in ``args.checkpoint_subset``.
* ``TestSubsetFlagRunGate`` — new
  ``CheckpointingRunRulesChecker.check_subset_mode``: the explicit flag with
  any model but 8B is INVALID (illegal combination, aborts before DLIO), as
  is the flag with a process count other than 8.
* ``TestNumProcessesSubsetNoLongerClosed`` — ``check_num_processes`` no
  longer treats 8 processes as a CLOSED form for the large models: 70B@8 is
  OPEN-eligible (multiple of TP*PP=8), 405B@8 and 1T@8 are INVALID, and the
  full respective counts remain the only CLOSED forms.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock

import pytest

from mlpstorage_py.cli import add_checkpointing_arguments
from mlpstorage_py.config import (
    BENCHMARK_TYPES,
    LLAMA3_1T,
    LLAMA3_8B,
    LLAMA3_70B,
    LLAMA3_405B,
    LLM_SUBSET_PROCS,
    PARAM_VALIDATION,
)
from mlpstorage_py.rules import (
    BenchmarkRun,
    BenchmarkRunData,
    CheckpointingRunRulesChecker,
)


def _make_chkpt_run(model: str, num_processes: int, logger,
                    run_args: dict | None = None) -> BenchmarkRun:
    data = BenchmarkRunData(
        benchmark_type=BENCHMARK_TYPES.checkpointing,
        model=model,
        command="run",
        run_datetime="20260810_120000",
        num_processes=num_processes,
        parameters={},
        override_parameters={},
        run_args=run_args or {},
    )
    return BenchmarkRun.from_data(data, logger)


@pytest.fixture
def mock_logger():
    return MagicMock()


class TestCheckpointSubsetCliFlag:
    """§4.3.5 sentence 1: the tool must accept an explicit subset parameter.

    Before #841 the ``checkpoint_subset`` help text sat orphaned in
    ``cli/common_args.py`` — no parser ever registered the argument.
    """

    @pytest.fixture
    def parser(self):
        parser = argparse.ArgumentParser()
        add_checkpointing_arguments(parser, 'open')
        return parser

    _BASE = [
        'run',
        '--model', 'llama3-8b',
        '--num-processes', '8',
        '--client-host-memory-in-gb', '512',
        '--checkpoint-folder', '/ckpt',
        '--results-dir', '/tmp',
        '--systemname', 'sys-v1',
    ]

    def test_flag_parses_true(self, parser):
        args = parser.parse_args(self._BASE + ['--checkpoint-subset', 'file'])
        assert args.checkpoint_subset is True

    def test_flag_defaults_false(self, parser):
        args = parser.parse_args(self._BASE + ['file'])
        assert args.checkpoint_subset is False


class TestSubsetFlagRunGate:
    """The explicit subset flag is an 8B-only, 8-process claim marker."""

    @pytest.mark.parametrize("model", [LLAMA3_70B, LLAMA3_405B, LLAMA3_1T])
    def test_subset_flag_with_large_model_is_invalid(self, mock_logger, model):
        run = _make_chkpt_run(model, LLM_SUBSET_PROCS, mock_logger,
                              run_args={'checkpoint_subset': True})
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_subset_mode()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
        assert "4.3.5" in issue.message
        # The remedy must be named: subset is defined only for the 8B model.
        assert "8b" in issue.message.lower()

    def test_subset_flag_with_8b_8_procs_passes(self, mock_logger):
        run = _make_chkpt_run(LLAMA3_8B, LLM_SUBSET_PROCS, mock_logger,
                              run_args={'checkpoint_subset': True})
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        assert checker.check_subset_mode() is None

    def test_subset_flag_with_wrong_proc_count_is_invalid(self, mock_logger):
        run = _make_chkpt_run(LLAMA3_8B, 16, mock_logger,
                              run_args={'checkpoint_subset': True})
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_subset_mode()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
        assert issue.parameter == "num_processes"
        assert issue.actual == 16

    @pytest.mark.parametrize("model", [LLAMA3_8B, LLAMA3_70B, LLAMA3_405B, LLAMA3_1T])
    def test_no_flag_silent_passes(self, mock_logger, model):
        """Without the explicit flag this check has no surface —
        downscaled-run classification belongs to ``check_num_processes``."""
        run = _make_chkpt_run(model, LLM_SUBSET_PROCS, mock_logger)
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        assert checker.check_subset_mode() is None

    def test_check_is_discovered_by_run_checks(self, mock_logger):
        run = _make_chkpt_run(LLAMA3_8B, 8, mock_logger)
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        method_names = [m.__name__ for m in checker.check_methods]
        assert "check_subset_mode" in method_names


class TestNumProcessesSubsetNoLongerClosed:
    """8 processes is no longer a CLOSED form for the large models.

    Rules.md §4.6.1 required CLOSED counts (Table 2, respective):
        llama3-8b → 8, llama3-70b → 64, llama3-405b → 512, llama3-1t → 1024.
    The pre-#841 gate also accepted 8 for any model as "the subset form",
    which the missing "not" in §4.3.5 appeared to endorse.
    """

    @pytest.mark.parametrize("model,closed_gpus", [
        (LLAMA3_8B, 8),
        (LLAMA3_70B, 64),
        (LLAMA3_405B, 512),
        (LLAMA3_1T, 1024),
    ])
    def test_full_closed_run_still_passes(self, mock_logger, model, closed_gpus):
        run = _make_chkpt_run(model, closed_gpus, mock_logger)
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        assert checker.check_num_processes() is None

    def test_8_procs_70b_is_open_only(self, mock_logger):
        """70B@8 is a multiple of TP*PP (8), so it can re-run as OPEN."""
        run = _make_chkpt_run(LLAMA3_70B, 8, mock_logger)
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_num_processes()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.OPEN
        assert "open" in issue.message.lower()

    @pytest.mark.parametrize("model,closed_gpus", [
        (LLAMA3_405B, 512),
        (LLAMA3_1T, 1024),
    ])
    def test_8_procs_large_model_is_invalid(self, mock_logger, model, closed_gpus):
        """405B@8 / 1T@8: not a TP*PP multiple, not the full count — INVALID."""
        run = _make_chkpt_run(model, 8, mock_logger)
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_num_processes()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
        assert str(closed_gpus) in issue.message
        assert issue.actual == 8

    def test_invalid_message_does_not_offer_subset_as_closed_form(self, mock_logger):
        """The message must stop advertising "8 (subset run)" as a CLOSED
        form for large models — that is the #841 inversion in prose."""
        run = _make_chkpt_run(LLAMA3_405B, 8, mock_logger)
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_num_processes()

        assert issue is not None
        assert "subset run per" not in issue.message
        assert "4.3.5" in issue.message
