"""
Regression tests for mlcommons/storage#792.

The reporter ran ``mlpstorage closed checkpointing run file`` with
``--num-processes 408`` for llama3-405b (51 hosts × 8 ranks). Rules.md §4.6.1
requires *exactly* 512 (full run) or 8 (subset run per §4.3.5) processes for
that model in CLOSED. 408 is neither.

Two things went wrong before the fix:

1. **No fail-fast gate.** ``CheckpointingRunRulesChecker`` did not validate
   ``num_processes``, so DLIO ran the full write+read pair (many minutes to
   hours of I/O time) before ``mlpstorage validate`` eventually rejected it.

2. **Misleading post-hoc message.** ``add_checkpoint_params`` in
   ``benchmarks/dlio.py`` auto-labels *any* downscaled run
   (``num_processes < ClosedGPUs``) with ``checkpoint.mode="subset"`` so DLIO
   applies the ``model.parallelism.data`` override. At validation time
   §4.3.5 saw the ``subset`` label and reported "subset run requires exactly
   8 accelerators, got 408" — implying the user had opted into subset mode
   when in fact the tool had auto-labeled it.

This module covers both fixes:

* ``TestCheckNumProcessesFailFast`` — new
  ``CheckpointingRunRulesChecker.check_num_processes`` returns INVALID/OPEN
  for the misconfigured process counts and None for the two valid CLOSED
  forms per model.
* ``TestSubsetRunValidationClarifiedMessage`` — §4.3.5 error now names
  both valid CLOSED submission forms (8 subset OR ClosedGPUs full) when the
  auto-set ``subset`` label surfaces on a run that isn't actually a
  8-accelerator subset run.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
from mlpstorage_py.submission_checker.checks.checkpointing_checks import (
    CheckpointingCheck,
)
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import Loader
from mlpstorage_py.tests.conftest import build_submission


# ---------------------------------------------------------------------------
# Fixture: sample checkpointing BenchmarkRun for the pre-run checker
# ---------------------------------------------------------------------------

def _make_chkpt_run(model: str, num_processes: int, logger) -> BenchmarkRun:
    data = BenchmarkRunData(
        benchmark_type=BENCHMARK_TYPES.checkpointing,
        model=model,
        command="run",
        run_datetime="20260715_071751",
        num_processes=num_processes,
        parameters={},
        override_parameters={},
    )
    return BenchmarkRun.from_data(data, logger)


class TestCheckNumProcessesFailFast:
    """Pre-run gate that stops misconfigured runs before DLIO starts.

    Rules.md §4.6.1 (Table 2) — required CLOSED process counts:
        llama3-8b   → 8
        llama3-70b  → 64
        llama3-405b → 512
        llama3-1t   → 1024
    §4.3.5 subset form → 8 processes (any model except 8b — that constraint
    is enforced by ``subset_run_validation``, not here).
    """

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    # ------------------ CLOSED-valid cases return None -------------------

    @pytest.mark.parametrize("model,closed_gpus", [
        (LLAMA3_8B, 8),
        (LLAMA3_70B, 64),
        (LLAMA3_405B, 512),
        (LLAMA3_1T, 1024),
    ])
    def test_full_closed_run_passes(self, mock_logger, model, closed_gpus):
        run = _make_chkpt_run(model, closed_gpus, mock_logger)
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        assert checker.check_num_processes() is None

    @pytest.mark.parametrize("model,verdict", [
        (LLAMA3_70B, PARAM_VALIDATION.OPEN),
        (LLAMA3_405B, PARAM_VALIDATION.INVALID),
        (LLAMA3_1T, PARAM_VALIDATION.INVALID),
    ])
    def test_8_procs_large_model_no_longer_closed(self, mock_logger, model, verdict):
        """Superseded by mlcommons/storage#841: the "8 (subset run)" CLOSED
        allowance this test originally pinned traced to a missing "not" in
        Rules.md 4.3.5 — subset mode is 8B-only. 70B@8 is a TP*PP multiple
        (OPEN-eligible); 405B@8 / 1T@8 are INVALID. See
        test_issue841_subset_mode_gate.py for the full surface."""
        run = _make_chkpt_run(model, LLM_SUBSET_PROCS, mock_logger)
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_num_processes()
        assert issue is not None
        assert issue.validation == verdict

    # ------------------ The reporter's exact scenario --------------------

    def test_issue792_405b_408procs_is_invalid(self, mock_logger):
        """51 hosts × 8 ranks = 408 for llama3-405b — the exact reporter case."""
        run = _make_chkpt_run(LLAMA3_405B, 408, mock_logger)
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_num_processes()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
        # The error must name both valid CLOSED forms so the user isn't left
        # guessing which knob to turn.
        assert "8" in issue.message
        assert "512" in issue.message
        assert issue.parameter == "num_processes"
        assert issue.actual == 408

    # -------------- OPEN-only path (multiple of TP*PP) -------------------

    def test_open_only_multiple_of_tp_pp(self, mock_logger):
        """For 405b, TP*PP=256. num_processes=256 is a multiple → OPEN-only.

        This case matters because reporting INVALID here would push a
        submitter into the ``--allow-invalid-params`` / re-run cycle when
        their run *is* a valid OPEN submission. The check should tell them
        to re-run under the 'open' positional instead.
        """
        run = _make_chkpt_run(LLAMA3_405B, 256, mock_logger)
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_num_processes()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.OPEN
        assert "open" in issue.message.lower()

    # -------------- Non-LLM model silent-passes --------------------------

    def test_unrecognized_model_silent_passes(self, mock_logger):
        """``check_model`` owns the unknown-model surface — don't double-report."""
        run = _make_chkpt_run("unet3d", 408, mock_logger)
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        assert checker.check_num_processes() is None

    # -------------- Method discovered by run_checks ----------------------

    def test_check_is_discovered_by_run_checks(self, mock_logger):
        run = _make_chkpt_run(LLAMA3_405B, 512, mock_logger)
        checker = CheckpointingRunRulesChecker(run, logger=mock_logger)
        method_names = [m.__name__ for m in checker.check_methods]
        assert "check_num_processes" in method_names


# ---------------------------------------------------------------------------
# Post-hoc submission-validator message (§4.3.5)
# ---------------------------------------------------------------------------

def _inject_override_params_into_chkpt_metadata(root: Path, params: dict) -> None:
    """Post-fixture: set override_parameters on every checkpoint metadata.json.

    Mirrors ``_inject_yaml_params_into_chkpt_metadata`` from
    ``test_checkpointing_check_retrofit.py`` — the shared conftest fixture
    doesn't expose a knob for override_parameters on checkpointing runs, so
    tests that need to trip auto-subset labeling do it out-of-band.
    """
    for meta_path in Path(root).rglob("checkpointing/*/*/metadata.json"):
        meta = json.loads(meta_path.read_text())
        meta["override_parameters"] = params
        meta_path.write_text(json.dumps(meta))


def _inject_summary_num_accelerators(root: Path, num_accelerators: int) -> None:
    """Post-fixture: set ``num_accelerators`` in every checkpoint summary.json."""
    for summ_path in Path(root).rglob("checkpointing/*/*/summary.json"):
        summ = json.loads(summ_path.read_text())
        summ["num_accelerators"] = num_accelerators
        summ_path.write_text(json.dumps(summ))


def _run_checkpointing_check(root: Path, mock_logger):
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    loader = Loader(config=config, root=str(root), version="v2.0")
    for logs in loader.load():
        if logs.loader_metadata.mode == "checkpointing":
            return CheckpointingCheck(
                log=mock_logger, config=config, submissions_logs=logs
            )
    raise AssertionError("no checkpointing SubmissionLogs from fixture")


class TestSubsetRunValidationClarifiedMessage:
    """§4.3.5 error message clarity for auto-labeled ``subset`` runs.

    The reporter's run landed here with num_accelerators=408 and the
    auto-set ``checkpoint.mode="subset"`` override. Post-fix the emitted
    violation must:

      * still fail the check (subset ≠ 8 is a genuine rule violation), and
      * name both valid CLOSED submission forms (8 or ClosedGPUs) so the
        user can see which knob to turn.
    """

    def test_405b_408_accelerators_message_names_the_remedy(
        self, tmp_path, mock_logger
    ):
        # Build a submission with llama3-405b and inject the exact failure state.
        root = build_submission(
            tmp_path,
            chkpt_model="llama3-405b",
            chkpt_closed_num_processes=408,
        )
        _inject_override_params_into_chkpt_metadata(
            root, {"checkpoint.mode": "subset"}
        )
        _inject_summary_num_accelerators(root, 408)

        check = _run_checkpointing_check(root, mock_logger)
        ok = check.subset_run_validation()

        assert ok is False, "auto-subset labeling with !=8 accelerators must fail"

        errors = [c for c in mock_logger.method_calls if c[0] == "error"]
        assert errors, f"expected an error to be logged; got {mock_logger.method_calls}"
        joined = " ".join(str(c) for c in errors)
        # Post-#841 the message names the actual remedy: no subset form for
        # 405b — a CLOSED submission must be the full 512-process run …
        assert "512" in joined, joined
        # … and points at the corrected rule reading.
        assert "storage#841" in joined or "8B" in joined, joined

    def test_405b_subset_run_now_fails(self, tmp_path, mock_logger):
        """Superseded by mlcommons/storage#841: this test originally pinned an
        8-accelerator 405b subset run as the happy path. The published 4.3.5
        text was missing the word "not" — subset mode is defined only for the
        8B model, so this exact shape (ten of which published in v3.0) must
        now flag."""
        root = build_submission(
            tmp_path,
            chkpt_model="llama3-405b",
            chkpt_closed_num_processes=LLM_SUBSET_PROCS,  # 8
        )
        _inject_override_params_into_chkpt_metadata(
            root, {"checkpoint.mode": "subset"}
        )
        _inject_summary_num_accelerators(root, LLM_SUBSET_PROCS)

        check = _run_checkpointing_check(root, mock_logger)
        ok = check.subset_run_validation()

        assert ok is False, "405b subset run must fail 4.3.5 post-#841"
        errors = [c for c in mock_logger.method_calls if c[0] == "error"]
        joined = " ".join(str(c) for c in errors)
        assert "not defined" in joined or "only for the 8B" in joined, joined


# Fixture at module scope so the class methods can share it.
@pytest.fixture
def mock_logger():
    return MagicMock()
