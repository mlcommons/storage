"""
Tests for Plan 03-02 Task 3: TrainingCheck @rule retrofit (D-R3).

Covers:
  - Parametrized assertion that each of the 13 retrofitted (rule_id, method)
    pairs on TrainingCheck has __rule_id__ == expected.
  - Behavior-preservation for 3.1.1 (missing dataset params emits prefixed
    violation) and 3.4.1 (data_dir == results_dir emits prefixed violation).
  - Deferred-stub assertions for 3.3.5 distributed_data_accessibility_check
    and 3.3.7 node_capability_consistency_check (info-only, returns True).

References:
  - Plan 03-02 `<interfaces>` TrainingCheck binding table (locked 13 rows)
  - Plan 03-01 SUMMARY.md retrofit-test pattern (RETROFITTED_DIRECTORY_METHODS)
  - Phase 1 / Phase 2 conftest.build_submission + MockLogger fixture

Run with:
    pytest mlpstorage_py/tests/test_training_check_retrofit.py -v
"""

import pytest

from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import Loader


# ---------------------------------------------------------------------------
# Locked binding table — Plan 03-02 <interfaces>
# ---------------------------------------------------------------------------

RETROFITTED_TRAINING_METHODS = [
    ("3.1.1", "verify_datasize_usage"),
    ("3.1.2", "recalculate_dataset_size"),
    ("3.2.1", "datagen_minimum_size"),
    ("3.3.1", "run_data_matches_datasize"),
    ("3.3.2", "accelerator_utilization_check"),
    ("3.3.3", "single_host_simulated_accelerators"),
    ("3.3.5", "distributed_data_accessibility_check"),
    ("3.3.6", "identical_accelerators_per_node"),
    ("3.3.7", "node_capability_consistency_check"),
    ("3.4.1", "mlpstorage_path_args"),
    ("3.6.1", "closed_submission_checksum"),
    ("3.6.2", "closed_submission_parameters"),
    ("3.6.3", "open_submission_parameters"),
]


# ---------------------------------------------------------------------------
# Helper — mirror test_training_check_phase2.py::_run_training_check
# ---------------------------------------------------------------------------

def _run_training_check(root_path, mock_logger):
    """Load the submission tree and return the first TrainingCheck instance found.

    Args:
        root_path: Path to the submission root (as returned by build_submission).
        mock_logger: MockLogger instance to use for violation capture.

    Returns:
        TrainingCheck instance for the first training SubmissionLogs found.

    Raises:
        AssertionError: if no training mode was yielded by Loader.load().
    """
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    loader = Loader(config=config, root=str(root_path), version="v2.0")
    for logs in loader.load():
        if logs.loader_metadata.mode == "training":
            return TrainingCheck(log=mock_logger, config=config, submissions_logs=logs)
    raise AssertionError("no training SubmissionLogs yielded from the fixture")


# ---------------------------------------------------------------------------
# Parametrized rule-id binding assertion (D-R3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule_id,method_name", RETROFITTED_TRAINING_METHODS)
def test_every_retrofitted_training_method_has_rule_id_attribute(rule_id, method_name):
    """Every retrofitted method has __rule_id__ == expected rule_id (D-R3)."""
    method = getattr(TrainingCheck, method_name, None)
    assert method is not None, f"TrainingCheck has no method named {method_name!r}"
    assert getattr(method, "__rule_id__", None) == rule_id, (
        f"{method_name}.__rule_id__ = "
        f"{getattr(method, '__rule_id__', None)!r}, expected {rule_id!r}"
    )


# ---------------------------------------------------------------------------
# Behavior-preservation: 3.1.1 trainingVerifyDatasizeUsage
# ---------------------------------------------------------------------------

def test_3_1_1_missing_dataset_params_emits_prefixed_violation(tmp_path, mock_logger):
    """3.1.1: when run metadata has no combined_params (no dataset block),
    verify_datasize_usage emits a [3.1.1 trainingVerifyDatasizeUsage] violation
    and returns False.

    The default build_submission writes metadata.json with empty combined_params
    (default _DEFAULT_METADATA has `"combined_params": {}`) and a non-empty
    args dict. The first branch (no params and no combined_params) is skipped
    because args is non-empty; the second branch ("dataset parameters not
    found in metadata") fires because combined_params.get('dataset', {}) is
    empty.
    """
    from mlpstorage_py.tests.conftest import build_submission
    root = build_submission(tmp_path)
    check = _run_training_check(root, mock_logger)
    result = check.verify_datasize_usage()
    assert result is False, (
        f"expected False (dataset params missing); got {result!r}. "
        f"errors: {mock_logger.errors}"
    )
    assert any(
        m.startswith("[3.1.1 trainingVerifyDatasizeUsage]")
        for m in mock_logger.errors
    ), (
        f"expected error starting with [3.1.1 trainingVerifyDatasizeUsage]; "
        f"got {mock_logger.errors}"
    )


# ---------------------------------------------------------------------------
# Behavior-preservation: 3.4.1 trainingMlpstoragePathArgs
# ---------------------------------------------------------------------------

def test_3_4_1_data_dir_equals_results_dir_emits_prefixed_violation(tmp_path, mock_logger):
    """3.4.1: when data_dir == results_dir, mlpstorage_path_args emits a
    [3.4.1 trainingMlpstoragePathArgs] violation and returns False.
    """
    from mlpstorage_py.tests.conftest import build_submission
    root = build_submission(
        tmp_path,
        run_data_dir="/shared/same",
        run_results_dir="/shared/same",
    )
    check = _run_training_check(root, mock_logger)
    result = check.mlpstorage_path_args()
    assert result is False, (
        f"expected False (data_dir == results_dir); got {result!r}. "
        f"errors: {mock_logger.errors}"
    )
    assert any(
        m.startswith("[3.4.1 trainingMlpstoragePathArgs]")
        for m in mock_logger.errors
    ), (
        f"expected error starting with [3.4.1 trainingMlpstoragePathArgs]; "
        f"got {mock_logger.errors}"
    )
    # The 'must be different' branch should specifically fire
    assert any(
        "must be different" in m
        for m in mock_logger.errors
    ), f"expected 'must be different' in errors; got {mock_logger.errors}"


# ---------------------------------------------------------------------------
# Deferred-stub: 3.3.5 distributed_data_accessibility_check (info-only)
# ---------------------------------------------------------------------------

def test_3_3_5_distributed_data_accessibility_logs_info_returns_true(tmp_path, mock_logger):
    """3.3.5 deferred stub: emits a single info-level record with the
    [3.3.5 trainingDistributedDataAccessibility] prefix, returns True, and
    does NOT emit any error-level record (no violation contribution).
    """
    from mlpstorage_py.tests.conftest import build_submission
    root = build_submission(tmp_path)
    check = _run_training_check(root, mock_logger)
    result = check.distributed_data_accessibility_check()
    assert result is True, f"expected True; got {result!r}"
    assert any(
        m.startswith("[3.3.5 trainingDistributedDataAccessibility]")
        for m in mock_logger.infos
    ), (
        f"expected info starting with [3.3.5 trainingDistributedDataAccessibility]; "
        f"got infos={mock_logger.infos}"
    )
    # Stub must not emit any 3.3.5-tagged error
    assert not any(
        m.startswith("[3.3.5 ")
        for m in mock_logger.errors
    ), f"deferred stub emitted error: {mock_logger.errors}"


# ---------------------------------------------------------------------------
# Deferred-stub: 3.3.7 node_capability_consistency_check (info-only)
# ---------------------------------------------------------------------------

def test_3_3_7_node_capability_consistency_logs_info_returns_true(tmp_path, mock_logger):
    """3.3.7 deferred stub: emits a single info-level record with the
    [3.3.7 trainingNodeCapabilityConsistency] prefix, returns True, and
    does NOT emit any error-level record.
    """
    from mlpstorage_py.tests.conftest import build_submission
    root = build_submission(tmp_path)
    check = _run_training_check(root, mock_logger)
    result = check.node_capability_consistency_check()
    assert result is True, f"expected True; got {result!r}"
    assert any(
        m.startswith("[3.3.7 trainingNodeCapabilityConsistency]")
        for m in mock_logger.infos
    ), (
        f"expected info starting with [3.3.7 trainingNodeCapabilityConsistency]; "
        f"got infos={mock_logger.infos}"
    )
    assert not any(
        m.startswith("[3.3.7 ")
        for m in mock_logger.errors
    ), f"deferred stub emitted error: {mock_logger.errors}"
