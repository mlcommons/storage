"""Issue #865: the workload-level verifier pass must run submission checkers
even when a workload group holds a single invocation.

``BenchmarkVerifier`` picks its checker family from the number of sources:
one source → the per-run ``*RunRulesChecker``, two or more → the
``*SubmissionRulesChecker``. reportgen's ``_process_workload_groups`` calls
it once per workload group as its *submission* pass, so a single-invocation
group (the common CLOSED checkpointing shape: one run with 10 writes and 10
reads) silently got the per-run checks a second time and never saw
``CheckpointSubmissionRulesChecker.check_num_runs`` /
``check_invocation_structure``. The retired D-24 metric-list gate was the
only thing that looked like op-count enforcement for those rows, and it
could never fire on real DLIO output.

These tests pin the explicit ``mode="multi"`` request and the run-command
scoping the training submission checker needs so that a lone training
``datagen`` group does not become INVALID for "requires 5 runs" (#717 shape).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from mlpstorage_py.config import BENCHMARK_TYPES, PARAM_VALIDATION
from mlpstorage_py.rules.models import BenchmarkRun, BenchmarkRunData
from mlpstorage_py.rules.run_checkers.checkpointing import CheckpointingRunRulesChecker
from mlpstorage_py.rules.submission_checkers.checkpointing import CheckpointSubmissionRulesChecker
from mlpstorage_py.rules.submission_checkers.training import TrainingSubmissionRulesChecker
from mlpstorage_py.rules.verifier import BenchmarkVerifier


def _run(*, benchmark_type, model, command="run", parameters=None, ts="20260703_120000"):
    data = BenchmarkRunData(
        benchmark_type=benchmark_type,
        model=model,
        command=command,
        run_datetime=ts,
        num_processes=8,
        parameters=parameters or {},
        override_parameters={},
        system_info=None,
        metrics={},
        result_dir=f"/nonexistent/{ts}",
        accelerator=None,
        run_args={},
    )
    return BenchmarkRun.from_data(data)


def _ckpt(write, read, ts="20260703_120000"):
    return _run(
        benchmark_type=BENCHMARK_TYPES.checkpointing,
        model="llama3-8b",
        parameters={"checkpoint": {
            "num_checkpoints_write": write, "num_checkpoints_read": read,
        }},
        ts=ts,
    )


class TestExplicitSubmissionMode:

    def test_default_single_source_keeps_run_checker(self):
        """No ``mode`` → one source still means the per-run checker (unchanged)."""
        v = BenchmarkVerifier(_ckpt(10, 10), logger=MagicMock())
        assert v.mode == "single"
        assert isinstance(v.rules_checker, CheckpointingRunRulesChecker)

    def test_multi_mode_on_single_checkpointing_run_uses_submission_checker(self):
        v = BenchmarkVerifier(_ckpt(10, 10), logger=MagicMock(), mode="multi")
        assert v.mode == "multi"
        assert isinstance(v.rules_checker, CheckpointSubmissionRulesChecker)

    def test_multi_mode_single_run_three_writes_is_invalid(self):
        """The whole point: a lone 3-write invocation must fail the 10/10 rule."""
        v = BenchmarkVerifier(_ckpt(3, 10), logger=MagicMock(), mode="multi")
        assert v.verify() == PARAM_VALIDATION.INVALID
        messages = [i.message for i in v.issues if i.validation == PARAM_VALIDATION.INVALID]
        assert any("Expected 10 total write operations, but found 3" in m for m in messages), messages

    def test_multi_mode_single_run_ten_ten_is_not_invalid(self):
        v = BenchmarkVerifier(_ckpt(10, 10), logger=MagicMock(), mode="multi")
        assert v.verify() != PARAM_VALIDATION.INVALID

    def test_multi_mode_split_pair_still_valid(self):
        """Two-source behaviour is untouched by the explicit request."""
        v = BenchmarkVerifier(
            _ckpt(10, 0, ts="20260703_120000"),
            _ckpt(0, 10, ts="20260703_120500"),
            logger=MagicMock(), mode="multi",
        )
        assert isinstance(v.rules_checker, CheckpointSubmissionRulesChecker)
        assert v.verify() != PARAM_VALIDATION.INVALID

    def test_single_mode_cannot_be_forced_on_multiple_sources(self):
        with pytest.raises(ValueError):
            BenchmarkVerifier(_ckpt(10, 0), _ckpt(0, 10), logger=MagicMock(), mode="single")

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            BenchmarkVerifier(_ckpt(10, 10), logger=MagicMock(), mode="bogus")


class TestTrainingSubmissionCheckerScopesToRunCommand:
    """#717 / #791 shape: auxiliary commands are not submission invocations."""

    def test_lone_datagen_group_in_multi_mode_is_not_invalid(self):
        datagen = _run(
            benchmark_type=BENCHMARK_TYPES.training, model="unet3d", command="datagen",
        )
        v = BenchmarkVerifier(datagen, logger=MagicMock(), mode="multi")
        assert isinstance(v.rules_checker, TrainingSubmissionRulesChecker)
        assert v.verify() != PARAM_VALIDATION.INVALID, [i.message for i in v.issues]

    def test_check_num_runs_counts_only_run_invocations(self):
        runs = [
            _run(benchmark_type=BENCHMARK_TYPES.training, model="unet3d",
                 command="datagen", ts="20260703_110000"),
        ] + [
            _run(benchmark_type=BENCHMARK_TYPES.training, model="unet3d",
                 ts=f"20260703_12000{i}")
            for i in range(5)
        ]
        checker = TrainingSubmissionRulesChecker(runs, logger=MagicMock())
        issue = checker.check_num_runs()
        assert issue.validation == PARAM_VALIDATION.CLOSED
        assert issue.actual == 5

    def test_check_num_runs_no_run_invocations_emits_nothing(self):
        checker = TrainingSubmissionRulesChecker(
            [_run(benchmark_type=BENCHMARK_TYPES.training, model="unet3d", command="datagen")],
            logger=MagicMock(),
        )
        assert checker.check_num_runs() is None
