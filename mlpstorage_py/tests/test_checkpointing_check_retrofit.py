"""
Tests for Plan 03-02 Task 4: CheckpointingCheck @rule retrofit (D-R3).

Covers:
  - Parametrized assertion that each of the 8 retrofitted (rule_id, method)
    pairs on CheckpointingCheck has __rule_id__ == expected.
  - 4.3.1 routing pin (xfail under W-03 lock — the conftest
    build_submission factory does NOT expose a kwarg to inject
    summary["metric"]["checkpoint_size_GB"], so the warning path inside
    checkpoint_data_size_ratio is unreachable through the fixture API).
    The xfail decoration documents the W-03 follow-up; removing it once
    conftest is extended promotes the test to a passing assertion.
  - 4.4.1 (checkpoint_folder == results_dir) emits prefixed violation.
  - 4.6.3 (closed CLOSED metadata with disallowed yaml_params) emits
    prefixed violation — uses an in-test metadata mutation helper rather
    than extending conftest (out of scope for Phase 3).

References:
  - Plan 03-02 `<interfaces>` CheckpointingCheck binding table
  - Plan 03-02 `<interfaces>` W-03 lock (xfail rationale)
  - Phase 1 / Phase 2 conftest.build_submission + MockLogger fixture

Run with:
    pytest mlpstorage_py/tests/test_checkpointing_check_retrofit.py -v
"""

import json
from pathlib import Path

import pytest

from mlpstorage_py.submission_checker.checks.checkpointing_checks import CheckpointingCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import Loader


# ---------------------------------------------------------------------------
# Locked binding table — Plan 03-02 <interfaces>
# ---------------------------------------------------------------------------

RETROFITTED_CHECKPOINTING_METHODS = [
    ("4.3.1", "checkpoint_data_size_ratio"),
    ("4.3.2", "fsync_verification"),
    ("4.3.3", "model_configuration_req"),
    ("4.3.4", "aggregate_accelerator_memory"),
    ("4.3.5", "subset_run_validation"),
    ("4.4.1", "checkpoint_path_args"),
    ("4.6.2", "closed_accelerators_per_host"),
    ("4.6.3", "closed_checkpoint_parameters"),
]


# ---------------------------------------------------------------------------
# Helper — mirror test_training_check_phase2.py shape, for checkpointing mode
# ---------------------------------------------------------------------------

def _run_checkpointing_check(root_path, mock_logger):
    """Load the submission tree and return the first CheckpointingCheck found.

    Args:
        root_path: Path to the submission root (as returned by build_submission).
        mock_logger: MockLogger instance to use for violation capture.

    Returns:
        CheckpointingCheck instance for the first checkpointing SubmissionLogs.

    Raises:
        AssertionError: if no checkpointing mode was yielded by Loader.load().
    """
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    loader = Loader(config=config, root=str(root_path), version="v2.0")
    for logs in loader.load():
        if logs.loader_metadata.mode == "checkpointing":
            return CheckpointingCheck(log=mock_logger, config=config, submissions_logs=logs)
    raise AssertionError("no checkpointing SubmissionLogs yielded from the fixture")


def _inject_yaml_params_into_chkpt_metadata(root, yaml_params):
    """Mutate every checkpoint metadata.json under ``root`` in-place to include
    ``yaml_params``. Used to trigger 4.6.3 closed_checkpoint_parameters without
    extending conftest (out of scope for Phase 3 per the W-03 lock pattern).

    Walks the standard build_submission tree:
        Acme/closed/Acme/results/<sysname>/checkpointing/<model>/<ts>/metadata.json
    """
    for meta_path in Path(root).rglob("checkpointing/*/*/metadata.json"):
        meta = json.loads(meta_path.read_text())
        meta["yaml_params"] = yaml_params
        meta_path.write_text(json.dumps(meta))


# ---------------------------------------------------------------------------
# Parametrized rule-id binding assertion (D-R3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule_id,method_name", RETROFITTED_CHECKPOINTING_METHODS)
def test_every_retrofitted_checkpointing_method_has_rule_id_attribute(rule_id, method_name):
    """Every retrofitted method has __rule_id__ == expected rule_id (D-R3)."""
    method = getattr(CheckpointingCheck, method_name, None)
    assert method is not None, f"CheckpointingCheck has no method named {method_name!r}"
    assert getattr(method, "__rule_id__", None) == rule_id, (
        f"{method_name}.__rule_id__ = "
        f"{getattr(method, '__rule_id__', None)!r}, expected {rule_id!r}"
    )


# ---------------------------------------------------------------------------
# 4.3.1 routing pin — W-03 lock: xfail until conftest gains
# summary.metric.checkpoint_size_GB injection kwarg.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "conftest build_submission has no kwarg to inject "
        "summary.metric.checkpoint_size_GB; the 4.3.1 advisory trigger requires "
        "checkpoint_size_GB > 0. Follow-up: extend conftest in a later phase to "
        "enable this assertion. Tracking: TODO-W03."
    ),
    strict=False,
)
def test_4_3_1_undersized_checkpoint_emits_warn_prefix_not_error(tmp_path, mock_logger):
    """4.3.1 routing pin: when checkpoint data per node < 3x host memory,
    checkpoint_data_size_ratio MUST emit a [4.3.1 checkpointDataSizeRatio]
    warning (via warn_violation) and NOT an error (via log_violation).

    Currently xfails because conftest's _DEFAULT_SUMMARY has no
    metric.checkpoint_size_GB field and build_submission does not expose a
    kwarg to inject one, so the early `continue` in
    checkpoint_data_size_ratio (when checkpoint_size_gb == 0) prevents the
    warn-violation branch from ever executing.

    Once conftest gains a `chkpt_summary_checkpoint_size_GB` kwarg (or
    equivalent), remove the xfail decoration to promote this to a passing
    assertion. The structural pin (Task 2 grep that 4.3.1 routes through
    warn_violation) covers the routing decision at source-code level today.
    """
    from mlpstorage_py.tests.conftest import build_submission
    # The default fixture passes through the early-continue guard.
    # We invoke the method anyway so the test body documents the expected
    # post-extension behavior.
    root = build_submission(tmp_path)
    check = _run_checkpointing_check(root, mock_logger)
    check.checkpoint_data_size_ratio()
    # Expected behavior (post conftest extension):
    assert any(
        m.startswith("[4.3.1 checkpointDataSizeRatio]")
        for m in mock_logger.warnings
    ), (
        f"expected warning starting with [4.3.1 checkpointDataSizeRatio]; "
        f"got warnings={mock_logger.warnings}"
    )
    # Expected behavior (post conftest extension): no error-level 4.3.1 record
    assert not any(
        m.startswith("[4.3.1 ")
        for m in mock_logger.errors
    ), (
        f"4.3.1 must route through warn_violation (warnings), not "
        f"log_violation (errors); got errors={mock_logger.errors}"
    )


# ---------------------------------------------------------------------------
# Behavior-preservation: 4.4.1 checkpointPathArgs
# ---------------------------------------------------------------------------

def test_4_4_1_checkpoint_folder_equals_results_dir_emits_prefixed_violation(tmp_path, mock_logger):
    """4.4.1: when checkpoint_folder == results_dir, checkpoint_path_args
    emits a [4.4.1 checkpointPathArgs] violation and returns False.
    """
    from mlpstorage_py.tests.conftest import build_submission
    root = build_submission(
        tmp_path,
        chkpt_checkpoint_folder="/shared/same",
        chkpt_results_dir="/shared/same",
    )
    check = _run_checkpointing_check(root, mock_logger)
    result = check.checkpoint_path_args()
    assert result is False, (
        f"expected False (checkpoint_folder == results_dir); got {result!r}. "
        f"errors: {mock_logger.errors}"
    )
    assert any(
        m.startswith("[4.4.1 checkpointPathArgs]")
        for m in mock_logger.errors
    ), (
        f"expected error starting with [4.4.1 checkpointPathArgs]; "
        f"got {mock_logger.errors}"
    )
    assert any(
        "must be different" in m
        for m in mock_logger.errors
    ), f"expected 'must be different' in errors; got {mock_logger.errors}"


# ---------------------------------------------------------------------------
# Behavior-preservation: 4.6.3 checkpointClosedCheckpointParameters
# ---------------------------------------------------------------------------

def test_4_6_3_closed_param_modified_emits_prefixed_violation(tmp_path, mock_logger):
    """4.6.3: when a CLOSED checkpoint metadata's yaml_params has a key not
    present in the reference config, closed_checkpoint_parameters emits a
    [4.6.3 checkpointClosedCheckpointParameters] violation and returns False.

    The conftest factory does not expose a yaml_params injection kwarg, so
    this test mutates each generated metadata.json in-place after
    build_submission. Out-of-scope extensions to conftest are reserved for a
    later phase (W-03 lock pattern).
    """
    from mlpstorage_py.tests.conftest import build_submission
    root = build_submission(
        tmp_path,
        chkpt_closed_num_processes=8,  # forces verification="closed" in metadata
    )
    # Inject a yaml_params block with a key that is guaranteed not to exist
    # in the reference llama3_8b.yaml.
    _inject_yaml_params_into_chkpt_metadata(
        root,
        {"not_a_real_dlio_param": "some_value"},
    )
    check = _run_checkpointing_check(root, mock_logger)
    result = check.closed_checkpoint_parameters()
    assert result is False, (
        f"expected False (disallowed yaml_params for CLOSED); got {result!r}. "
        f"errors: {mock_logger.errors}"
    )
    assert any(
        m.startswith("[4.6.3 checkpointClosedCheckpointParameters]")
        for m in mock_logger.errors
    ), (
        f"expected error starting with "
        f"[4.6.3 checkpointClosedCheckpointParameters]; "
        f"got {mock_logger.errors}"
    )
