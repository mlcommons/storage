"""Regression tests for storage#669 — host_memory_GB aggregation.

DLIO populates ``summary.json["host_memory_GB"]`` via an MPI SUM-Reduce
across ranks: local-rank-0 on each host writes its per-host memory into
a slot indexed by ``self.MPI.node()``, and the array is then summed
across ranks. On multi-rank-per-host clusters where ``self.MPI.node()``
does not monotonically map to distinct 0..nnodes-1 slots, the resulting
array is malformed:

* Some slots carry doubled values (two hosts' local-rank-0 wrote to the
  same slot; the SUM combined them).
* Some slots carry 0 (no host's local-rank-0 wrote there).

The array's *sum* is always correct (SUM-reduce is total-preserving);
*positional* access is not.

Every mlpstorage_py caller that computed cluster memory as
``num_hosts * host_memory_GB[0]`` — or that iterated positionally — was
subject to a spurious 2x inflation or 0x collapse depending on which
slot happened to land at index [0]. The primary caller is rule 3.1.2
(``recalculate_dataset_size``), and the reporter's real submission
(15 hosts, 30 accelerators, 2 ranks/host) was falsely rejected because
``host_memory_GB[0]`` was a doubled entry, doubling the minimum-samples
requirement.

Reporter's exact evidence array (`storage#669`):

    [376.18, 376.18, 376.18, 376.18, 376.18, 376.18, 376.18,
     187.90, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    sum(array)                         = 2821 GiB  (correct total)
    15 * array[0] = 15 * 376.18        = 5642.7 GiB (2x inflation)
    per-host truth = 2821 / 15         = 188.09 GiB
"""

from unittest.mock import MagicMock

import pytest

from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs


# Reporter's exact evidence array (storage#669). Sum = 2821 GiB.
REPORTER_MALFORMED_ARRAY = [
    376.18, 376.18, 376.18, 376.18, 376.18, 376.18, 376.18,
    187.90, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]
REPORTER_NUM_HOSTS = 15
REPORTER_NUM_ACCELERATORS = 30              # 2 ranks per host
REPORTER_RECORD_LENGTH_BYTES = 146_600_628  # unet3d record size
REPORTER_NUM_FILES_TRAIN = 105_000          # what the submitter generated
REPORTER_BATCH_SIZE = 7

# The correct minimum from Rules.md 3.1.2 given real 2821 GiB cluster
# memory: 2821 * 5 * 1024**3 / 146_600_628 ≈ 103_313 files. Submitter
# generated 105_000 → PASS. Buggy code computes 15 * 376.18 = 5642 GiB
# → ≈206_641 files → FAIL.
REPORTER_TRUE_MIN_FILES = 103_313


def _make_training_check(tmp_path, run_files):
    log = MagicMock()
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    submissions_logs = SubmissionLogs(
        datagen_files=[],
        run_files=run_files,
        system_file=None,
        loader_metadata=LoaderMetadata(
            division="closed",
            submitter="Acme",
            system="sys-v1",
            mode="training",
            benchmark="unet3d",
            folder=str(tmp_path),
        ),
    )
    return TrainingCheck(log=log, config=config, submissions_logs=submissions_logs), log


def _reporter_summary_and_metadata():
    """Build the exact summary + metadata pair the reporter submitted."""
    summary = {
        "num_hosts":        REPORTER_NUM_HOSTS,
        "num_accelerators": REPORTER_NUM_ACCELERATORS,
        "host_memory_GB":   list(REPORTER_MALFORMED_ARRAY),
        "num_files_train":  REPORTER_NUM_FILES_TRAIN,
        "num_files_eval":   0,
        "metric": {
            "train_au_meet_expectation":  "success",
            "train_au_mean_percentage":   99.0,
        },
    }
    metadata = {
        "parameters": {
            "dataset": {
                "num_files_train":       REPORTER_NUM_FILES_TRAIN,
                "num_samples_per_file":  1,
                "record_length_bytes":   REPORTER_RECORD_LENGTH_BYTES,
            },
            "reader": {"batch_size": REPORTER_BATCH_SIZE},
        },
        "args": {"hosts": [f"h{i}" for i in range(REPORTER_NUM_HOSTS)],
                 "data_dir": "/data", "results_dir": "/results"},
        "verification": "closed",
        "params_dict": {},
    }
    return summary, metadata


class TestRule312DoesNotDoubleCountMalformedHostMemoryArray:
    """storage#669 primary: rule 3.1.2 must aggregate the whole
    host_memory_GB array (sum), not index [0] * num_hosts.

    The reporter's real submission has 105_000 files, above the true
    minimum of ~103_313 files given the real 2821 GiB cluster memory.
    The buggy code inflates that minimum to ~206_641 files and falsely
    logs a violation. This test class pins the correct behavior.
    """

    def test_reporter_scenario_passes_3_1_2(self, tmp_path):
        """105k files against 2821 GiB real cluster memory must pass
        Rules.md 3.1.2 (min ≈ 103_313 files), not falsely fail with the
        2x-inflated 206_641-files minimum."""
        summary, metadata = _reporter_summary_and_metadata()
        run_files = [(summary, metadata, "20260703_120000")]
        check, log = _make_training_check(tmp_path, run_files)

        result = check.recalculate_dataset_size()

        assert result is True, (
            "storage#669: rule 3.1.2 must PASS for the reporter's real "
            "submission (105_000 files vs true min ~103_313). The 2x "
            "inflation from num_hosts*array[0] must not fire. "
            f"log calls: {log.mock_calls}"
        )

    def test_true_undersized_submission_still_fails_3_1_2(self, tmp_path):
        """Guardrail: an actually-undersized submission (fewer files
        than the sum-based minimum) must still fail. This is the flip
        side of the fix — we're moving the wall to the right place,
        not tearing it down."""
        summary, metadata = _reporter_summary_and_metadata()
        # Cut num_files_train in half: 52_500 < 103_313 → must fail.
        summary["num_files_train"] = 52_500
        metadata["parameters"]["dataset"]["num_files_train"] = 52_500
        run_files = [(summary, metadata, "20260703_120000")]
        check, log = _make_training_check(tmp_path, run_files)

        result = check.recalculate_dataset_size()

        assert result is False, (
            "Rule 3.1.2 must still fail for a genuinely undersized "
            "dataset (52_500 files < true min ~103_313)."
        )

    def test_homogeneous_array_still_passes_3_1_2(self, tmp_path):
        """Guardrail: the well-formed case must keep working. When the
        DLIO array is clean (one entry per host, no zeros, no doubling),
        sum(array) == num_hosts * array[0], so the fix is a no-op on
        homogeneous input."""
        summary, metadata = _reporter_summary_and_metadata()
        # Replace with a clean 15-entry homogeneous array. Same total.
        summary["host_memory_GB"] = [188.09] * REPORTER_NUM_HOSTS
        run_files = [(summary, metadata, "20260703_120000")]
        check, log = _make_training_check(tmp_path, run_files)

        result = check.recalculate_dataset_size()

        assert result is True, (
            "Rule 3.1.2 must PASS for the well-formed-array equivalent "
            "of the reporter's scenario (regression guard)."
        )
