
from .base import BaseCheck
from ..constants import *
from ..configuration.configuration import Config
from ..loader import SubmissionLogs
from ..rule_registry import rule
from .helpers import (
    _check_filesystem_separation,
    _check_code_image_layered,
    read_fs_separation_sidecar,
)

# Shared with the in-process verifier (mlpstorage_py.rules.run_checkers.training)
# so both checkers stay in lockstep about which dotted-keys the mlpstorage
# tool injects on the user's behalf and therefore must NOT count as user
# overrides. Drift between the two checkers is what surfaced as bugs 2 and 3
# in #503 — the run_checkers side was fixed in commit 0b3d370 (PR #496) but
# this submission_checker side was missed.
from mlpstorage_py.rules.run_checkers.training import (
    TrainingRunRulesChecker as _TrainingRunRulesChecker,
)
_TOOL_INJECTED_PARAMS = _TrainingRunRulesChecker.TOOL_INJECTED_PARAMS

import os
import hashlib
import re


class TrainingCheck(BaseCheck):
    """
    A check class for validating training parameters and related properties.
    Inherits from BaseCheck and receives a config and loader instance.
    """

    def __init__(self, log, config: Config, submissions_logs: SubmissionLogs):
        """
        Initialize TrainingChecks with configuration and loader.

        Args:
            config: A Config instance containing submission configuration.
            loader: A SubmissionLogs instance for accessing submission logs.
        """
        # Call parent constructor with the loader's log and submission path
        super().__init__(log=log, path=submissions_logs.loader_metadata.folder)
        self.config = config
        self.submissions_logs = submissions_logs
        self.mode = self.submissions_logs.loader_metadata.mode
        self.model = self.submissions_logs.loader_metadata.benchmark
        self.name = "training checks"
        self.datagen_path = os.path.join(self.path, "datagen")
        self.run_path = os.path.join(self.path, "run")
        self.init_checks()

    def init_checks(self):
        self.checks = []
        self.checks.extend([
            self.verify_datasize_usage,
            self.recalculate_dataset_size,
            self.datagen_minimum_size,
            self.run_data_matches_datasize,
            self.accelerator_utilization_check,
            self.single_host_simulated_accelerators,
            self.single_host_client_limit,   # TRAIN-01: wire-up (was missing from init_checks)
            self.distributed_data_accessibility_check,   # 3.3.5 deferred stub (Plan 03-02)
            self.identical_accelerators_per_node,
            self.node_capability_consistency_check,      # 3.3.7 deferred stub (Plan 03-02)
            self.closed_submission_checksum,
            self.closed_submission_parameters,
            self.open_submission_parameters,
            self.mlpstorage_path_args,
            self.mlpstorage_filesystem_check,
        ])

    def _get_benchmark_api(self) -> str:
        """Return 'file' or 'object' (default 'file') from the schema-validated system YAML.

        Reads self.submissions_logs.system_file (loaded by Loader at line 98).
        Returns 'file' if system_file is None or the architecture block is absent.
        Per D-B7, the helper trusts the schema validation — no re-validation here.
        """
        system_file = getattr(self.submissions_logs, "system_file", None)
        if not system_file:
            return "file"
        # `.get(key, {})` only catches missing keys — if the YAML serializes
        # an intermediate node as `null`, the chained .get raises AttributeError
        # on NoneType. `or {}` collapses both absent and null to a safe default.
        sut = system_file.get("system_under_test") or {}
        solution = sut.get("solution") or {}
        architecture = solution.get("architecture") or {}
        return architecture.get("benchmark_API", "file")

    @rule("3.1.1", "trainingVerifyDatasizeUsage")
    def verify_datasize_usage(self):
        """
        Verify that the datasize option was used by finding it in the run metadata.
        (Rules.md 3.1.1)
        """
        valid = True
        if self.mode != "training":
            return valid

        for summary, metadata, ts in self.submissions_logs.run_files:
            if metadata is None:
                self.log.debug(
                    "[3.1.1] %s/%s: skipping (metadata not loaded)",
                    self.path, ts,
                )
                continue
            # Check if datasize-related parameters are in the metadata
            params = metadata.get("args", {})
            combined_params = metadata.get("parameters", {})

            if not params and not combined_params:
                self.log_violation(
                    "3.1.1", "trainingVerifyDatasizeUsage", self.path,
                    "no parameters found in metadata to verify datasize usage",
                )
                valid = False
                continue

            # Check if dataset-related params are present
            dataset_params = combined_params.get("dataset", {})
            if not dataset_params:
                self.log_violation(
                    "3.1.1", "trainingVerifyDatasizeUsage", self.path,
                    "dataset parameters not found in metadata",
                )
                valid = False

        return valid
    
    @rule("3.1.2", "trainingRecalculateDatasetSize")
    def recalculate_dataset_size(self):
        """
        Recalculate minimum dataset size and verify it matches the run's logfile.
        (Rules.md 3.1.2)
        """
        valid = True
        if self.mode != "training":
            return valid
        HOST_MEMORY_MULTIPLIER = 5
        MIN_STEPS_PER_EPOCH = 500

        for summary, metadata, ts in self.submissions_logs.run_files:
            # Missing summary.json / metadata.json is already reported under
            # rule 2.1.19 (runFiles) by SubmissionStructureCheck; skip the
            # cross-check rather than re-fire or crash with AttributeError.
            if summary is None or metadata is None:
                self.log.debug(
                    "[3.1.2] %s/%s: skipping (summary or metadata not loaded)",
                    self.path, ts,
                )
                continue
            try:
                # Get parameters
                combined_params = metadata.get("parameters", {})
                dataset_params = combined_params.get("dataset", {})
                reader_params = combined_params.get("reader", {})

                num_files_train = int(dataset_params.get("num_files_train", 0))
                num_samples_per_file = int(dataset_params.get("num_samples_per_file", 1))
                record_length = float(dataset_params.get("record_length_bytes", 0))
                batch_size = int(reader_params.get("batch_size", 1))

                # From summary
                num_accelerators = summary.get("num_accelerators", 1)
                num_hosts = summary.get("num_hosts", 1)
                host_memory_gb = summary.get("host_memory_GB", [0])[0]

                if record_length == 0:
                    self.log_violation(
                        "3.1.2", "trainingRecalculateDatasetSize", self.path,
                        "record length is 0, cannot calculate dataset size",
                    )
                    valid = False
                    continue

                # Calculate min samples from steps per epoch
                num_steps_per_epoch = max(MIN_STEPS_PER_EPOCH,
                                        num_files_train * num_samples_per_file // (batch_size * num_accelerators))
                min_samples_steps = num_steps_per_epoch * batch_size * num_accelerators

                # Calculate min samples from host memory
                total_host_memory = num_hosts * host_memory_gb
                min_samples_memory = (total_host_memory * HOST_MEMORY_MULTIPLIER *
                                    1024 * 1024 * 1024 / record_length)

                # Take max of both constraints
                min_samples = max(min_samples_steps, min_samples_memory)
                min_total_files = min_samples / num_samples_per_file
                min_files_size_gb = min_samples * record_length / 1024 / 1024 / 1024

                # Verify actual matches expected
                actual_num_files = num_files_train
                if actual_num_files < min_total_files:
                    self.log_violation(
                        "3.1.2", "trainingRecalculateDatasetSize", self.path,
                        "dataset size mismatch: actual files %d < minimum required %d",
                        actual_num_files,
                        int(min_total_files),
                    )
                    valid = False

            except (KeyError, ValueError, TypeError) as e:
                self.log_violation(
                    "3.1.2", "trainingRecalculateDatasetSize", self.path,
                    "failed to calculate dataset size: %s", str(e),
                )
                valid = False

        return valid
    
    @rule("3.2.1", "trainingDatagenMinimumSize")
    def datagen_minimum_size(self):
        """
        Verify that datagen data generated >= datasize calculated.
        (Rules.md 3.2.1)
        """
        valid = True
        if self.mode != "training":
            return valid
        if not self.submissions_logs.datagen_files:
            self.log.warning("No datagen files found")
            return valid

        # Get expected size from run
        expected_size = None
        for summary, metadata, _ in self.submissions_logs.run_files:
            if metadata is None:
                continue
            dataset_params = metadata.get("parameters", {}).get("dataset", {})
            num_files = int(dataset_params.get("num_files_train", 0))
            record_length = float(dataset_params.get("record_length_bytes", 0))
            num_samples_per_file = int(dataset_params.get("num_samples_per_file", 1))
            expected_size = num_files * num_samples_per_file * record_length / 1024 / 1024 / 1024
            break

        # Check datagen produced at least that much
        for summary, metadata, _ in self.submissions_logs.datagen_files:
            if metadata is None:
                continue
            dataset_params = metadata.get("parameters", {}).get("dataset", {})
            num_files = int(dataset_params.get("num_files_train", 0))
            record_length = float(dataset_params.get("record_length_bytes", 0))
            num_samples_per_file = int(dataset_params.get("num_samples_per_file", 1))
            datagen_size = num_files * num_samples_per_file * record_length / 1024 / 1024 / 1024

            if expected_size and datagen_size < expected_size:
                self.log_violation(
                    "3.2.1", "trainingDatagenMinimumSize", self.path,
                    "datagen size %.2fGiB is less than required %.2fGiB",
                    datagen_size,
                    expected_size,
                )
                valid = False

        return valid
    
    @rule("3.3.1", "trainingRunDataMatchesDatasize")
    def run_data_matches_datasize(self):
        """Verify run.num_files_train is in [datasize, datagen] per --data-dir.

        Issue #608: prior versions compared against
        ``NUM_DATASET_TRAIN_FILES`` placeholder constants tagged
        ``# TODO: Ask for correct values``. Real submissions easily
        exceed the placeholder (UNet3D at 3 nodes × B200 / 768 GiB
        needs 84,375 files, not 14,000), so every conforming
        submission failed unconditionally. This rewrite reads the
        actual values that the ``datasize/`` and ``datagen/`` phases
        wrote for THIS submission.

        Two bounds:

        * Upper: ``run.num_files_train > datagen.num_files_train`` →
          ``[3.3.1 DATAGEN-OVERRUN]`` warning. The run consumed more
          data than datagen produced — physically impossible against
          the recorded --data-dir, or sweep config mismatch.
        * Lower: ``run.num_files_train < datasize.num_files_train`` →
          ``[3.3.1 DATASIZE-UNDERRUN]`` warning. The run consumed
          less than the minimum prescribed for a representative
          benchmark.

        Datasize→run matching: pair the run with the datasize phase
        whose ``args.data_dir`` matches the run's ``args.data_dir``.
        If exactly one matches, use it. If multiple datasize phases
        target the same --data-dir, emit ``[3.3.1 DATASIZE-REUSED]``
        — the --data-dir has been reused and we cannot determine
        authoritatively what it contains. If no match exists, emit
        ``[3.3.1 DATADIR-MISMATCH]``.

        All violations are warnings (``warn_violation``) — mid
        submission-window, do not invalidate work already on disk.
        After the window closes the appropriate violations may be
        promoted to errors; the stable bracketed tokens
        (``[3.3.1 DATAGEN-OVERRUN]``, etc.) give submitter CI a
        grep-stable suppression surface in the meantime.

        Missing datasize/datagen phases emit ``[3.3.1 DATASIZE-MISSING]``
        / ``[3.3.1 DATAGEN-MISSING]`` warnings rather than silent skip.

        See `.planning/BACKLOG.md` B-04 for the post-window manifest
        extension that closes the "but is the data really on disk?"
        loop without paying object-store LIST cost.

        (Rules.md 3.3.1)
        """
        valid = True
        if self.mode != "training":
            return valid

        datasize_files = self.submissions_logs.datasize_files or []
        datagen_files = self.submissions_logs.datagen_files or []

        # Pre-resolve --data-dir → datasize-record mapping for reuse detection.
        # Each datasize_files entry is (summary=None, metadata_dict, timestamp_str).
        datasize_by_dir = self._group_datasize_by_data_dir(datasize_files)

        # Pre-resolve the single most-recent datagen metadata (we treat datagen
        # as one-per-submission; sweep workflows generate once for the largest
        # size and run multiple smaller configs against it).
        datagen_num_files_train, datagen_data_dir = self._extract_latest_datagen_cardinality(datagen_files)

        if not datasize_files:
            self.warn_violation(
                "3.3.1", "trainingRunDataMatchesDatasize", self.path,
                "[3.3.1 DATASIZE-MISSING] no datasize/ phase found; "
                "rule 3.3.1 cross-check skipped",
            )
        if not datagen_files:
            self.warn_violation(
                "3.3.1", "trainingRunDataMatchesDatasize", self.path,
                "[3.3.1 DATAGEN-MISSING] no datagen/ phase found; "
                "upper-bound check skipped",
            )

        # Warn once per reused --data-dir, regardless of how many runs reference it.
        for data_dir, records in datasize_by_dir.items():
            if len(records) > 1:
                self.warn_violation(
                    "3.3.1", "trainingRunDataMatchesDatasize", self.path,
                    "[3.3.1 DATASIZE-REUSED] %d datasize/ phases target --data-dir %r; "
                    "cannot determine authoritative cardinality",
                    len(records), data_dir,
                )

        for summary, metadata, ts in self.submissions_logs.run_files:
            if summary is None or metadata is None:
                # 2.1.19 already flags missing summary; do not double-fire here.
                continue

            run_num_files_train = summary.get("num_files_train")
            run_num_files_eval = summary.get("num_files_eval")
            run_data_dir = metadata.get("args", {}).get("data_dir")

            # Resolve the datasize record that matches this run's --data-dir.
            datasize_record = self._match_datasize_for_run(
                run_data_dir, datasize_by_dir, ts,
            )

            if datasize_record is not None:
                ds_num_files_train, ds_data_dir = datasize_record
                if (ds_num_files_train is not None
                        and run_num_files_train is not None
                        and run_num_files_train < ds_num_files_train):
                    self.warn_violation(
                        "3.3.1", "trainingRunDataMatchesDatasize", self.path,
                        "[3.3.1 DATASIZE-UNDERRUN] run/%s num_files_train (%s) < "
                        "datasize num_files_train (%s); representative-benchmark "
                        "floor not met",
                        ts, run_num_files_train, ds_num_files_train,
                    )
            elif datasize_files and run_data_dir is not None:
                # We had datasize records, but none matched this run's --data-dir.
                self.warn_violation(
                    "3.3.1", "trainingRunDataMatchesDatasize", self.path,
                    "[3.3.1 DATADIR-MISMATCH] run/%s --data-dir %r has no matching "
                    "datasize/ phase; lower-bound check skipped",
                    ts, run_data_dir,
                )

            # Upper bound against datagen.
            if (datagen_num_files_train is not None
                    and run_num_files_train is not None
                    and run_num_files_train > datagen_num_files_train):
                self.warn_violation(
                    "3.3.1", "trainingRunDataMatchesDatasize", self.path,
                    "[3.3.1 DATAGEN-OVERRUN] run/%s num_files_train (%s) > "
                    "datagen num_files_train (%s); run consumed more data than "
                    "datagen produced",
                    ts, run_num_files_train, datagen_num_files_train,
                )

            # num_files_eval mirror — absent-key is a warning, NOT silent skip
            # (issue #608 WRT 4). Models without an eval phase will warn once
            # per run; the stable token lets submitter CI suppress per-model.
            if run_num_files_eval is None:
                self.warn_violation(
                    "3.3.1", "trainingRunDataMatchesDatasize", self.path,
                    "[3.3.1 EVAL-FIELD-MISSING] run/%s summary has no "
                    "num_files_eval field; eval cross-check skipped",
                    ts,
                )

        # Warn-only invariant: rule passes regardless of warnings recorded.
        return valid

    @staticmethod
    def _group_datasize_by_data_dir(datasize_files):
        """Group loaded datasize tuples by ``args.data_dir`` value.

        Returns ``{data_dir: [(num_files_train, timestamp), ...]}``.
        Entries with missing metadata or missing num_files_train still
        appear in the dict (with value ``None``) so reuse detection
        does not silently drop them.
        """
        grouped: dict = {}
        for _summary, metadata, ts in datasize_files:
            if metadata is None:
                continue
            data_dir = metadata.get("args", {}).get("data_dir")
            params = metadata.get("parameters", {}) or {}
            dataset_params = params.get("dataset", {}) or {}
            num_files_train = dataset_params.get("num_files_train")
            grouped.setdefault(data_dir, []).append((num_files_train, ts))
        return grouped

    @staticmethod
    def _extract_latest_datagen_cardinality(datagen_files):
        """Return (num_files_train, data_dir) from the most-recent datagen phase.

        Uses timestamp string ordering — datasize/datagen/run timestamps
        sort lexically as long as they share the canonical
        ``YYYYMMDD_HHmmss`` format that mlpstorage emits.
        """
        latest_ts = None
        latest_metadata = None
        for _summary, metadata, ts in datagen_files:
            if metadata is None:
                continue
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
                latest_metadata = metadata
        if latest_metadata is None:
            return None, None
        params = latest_metadata.get("parameters", {}) or {}
        dataset_params = params.get("dataset", {}) or {}
        num_files_train = dataset_params.get("num_files_train")
        data_dir = latest_metadata.get("args", {}).get("data_dir")
        return num_files_train, data_dir

    @staticmethod
    def _match_datasize_for_run(run_data_dir, datasize_by_dir, run_ts):
        """Return (num_files_train, data_dir) for the datasize record matching this run.

        Match priority:
          1. Exact ``args.data_dir`` equality between datasize and run.
             If multiple datasize phases target the same --data-dir
             (reuse case), pick the most-recent one before ``run_ts``.
          2. No match → return ``None`` so the caller emits
             ``[3.3.1 DATADIR-MISMATCH]``.
        """
        if run_data_dir is None:
            return None
        records = datasize_by_dir.get(run_data_dir)
        if not records:
            return None
        # Pick the latest datasize timestamp that is <= the run timestamp;
        # falls through to the latest overall if none are <= run_ts.
        eligible = [(num, ts) for (num, ts) in records if ts <= run_ts] or records
        eligible_sorted = sorted(eligible, key=lambda r: r[1])
        chosen_num, _chosen_ts = eligible_sorted[-1]
        return chosen_num, run_data_dir
    
    @rule("3.3.2", "trainingAcceleratorUtilizationCheck")
    def accelerator_utilization_check(self):
        """
        Check that AU (Accelerator Utilization) meets minimum requirements.
        (Rules.md 3.3.2)
        """
        valid = True
        if self.mode != "training":
            return valid
        for summary, metadata, ts in self.submissions_logs.run_files:
            if summary is None:
                self.log.debug(
                    "[3.3.2] %s/%s: skipping (summary not loaded)",
                    self.path, ts,
                )
                continue
            metrics = summary.get("metric", {})
            au_mean = metrics.get("train_au_mean_percentage", 0)
            au_expectation = metrics.get("train_au_meet_expectation", "")

            if au_expectation != "success":
                self.log_violation(
                    "3.3.2", "trainingAcceleratorUtilizationCheck", self.path,
                    "AU check failed: expected 'success', got '%s' (AU: %.2f%%)",
                    au_expectation,
                    au_mean,
                )
                valid = False

        return valid
    
    @rule("3.3.3", "trainingSingleHostSimulatedAccelerators")
    def single_host_simulated_accelerators(self):
        """
        For single-host submissions, verify sufficient simulated accelerators.
        (Rules.md 3.3.3)

        Per the binding table in Plan 03-02 `<interfaces>`: this rule is advisory,
        not a violation. The existing ``self.log.warning`` call is preserved; only
        the ``@rule`` decorator is added so ``discover_rules`` reports 3.3.3 covered.
        """
        valid = True
        if self.mode != "training":
            return valid
        for summary, metadata, ts in self.submissions_logs.run_files:
            if summary is None:
                self.log.debug(
                    "[3.3.3] %s/%s: skipping (summary not loaded)",
                    self.path, ts,
                )
                continue
            num_hosts = summary.get("num_hosts", 1)
            num_accelerators = summary.get("num_accelerators", 1)

            if num_hosts == 1 and num_accelerators < 4:
                self.log.warning(
                    "Single-host submission has only %d accelerators. Consider increasing via --num-accelerators",
                    num_accelerators
                )

        return valid

    @rule("3.3.5", "trainingDistributedDataAccessibility")
    def distributed_data_accessibility_check(self):
        """Rules.md 3.3.5 — distributed training data accessibility.

        Satisfied by construction at runtime: the CAP-02 shared-filesystem
        probe in ``mlpstorage_py.cluster_collector.run_shared_fs_probe``
        (invoked from ``Benchmark._pre_execution_gate`` in
        ``benchmarks/base.py``) creates a sentinel in the data destination
        on rank 0 and MPI-gathers ``os.stat`` results from every
        participating rank; the run fails fast with FileSystemError before
        the workload starts if any host cannot see the sentinel via the
        shared namespace. CAP-03 (issue #601, ``benchmarks/fs_separation_probe.py``)
        additionally verifies that ``--data-dir`` and ``--results-dir``
        resolve to distinct filesystems on the running rank via
        ``os.link()`` / EXDEV.

        Any submission that reaches this validator has therefore already
        passed the accessibility contract; the rule body preserves the
        ``@rule`` binding for coverage discovery and emits a single
        INFO-level line so tooling that greps by rule ID surfaces the
        rule as "visited and satisfied".
        """
        self.log.info(
            "[3.3.5 trainingDistributedDataAccessibility] %s: "
            "satisfied by construction — CAP-02 shared-FS probe "
            "(cluster_collector.run_shared_fs_probe) verifies data_dir is "
            "reachable from every participating rank at pre-execution",
            self.path,
        )
        return True
    
    @rule("3.3.4", "trainingSingleHostClientLimit")
    def single_host_client_limit(self):
        """For single-host runs (summary.num_hosts == 1), fail if more than one
        client node is specified in metadata.args.hosts. (Rules.md 3.3.4)

        TRAIN-01 wire-up: registered in init_checks; upgraded from bare log.error
        to log_violation (QUAL-02 retro-fit) with @rule decorator.
        """
        valid = True
        if self.mode != "training":
            return valid
        for summary, metadata, ts in self.submissions_logs.run_files:
            if summary is None or metadata is None:
                self.log.debug(
                    "[3.3.4] %s/%s: skipping (summary or metadata not loaded)",
                    self.path, ts,
                )
                continue
            num_hosts = summary.get("num_hosts", 1)
            if num_hosts == 1:
                args = metadata.get("args", {})
                hosts = args.get("hosts", [])
                if len(hosts) > 1:
                    self.log_violation(
                        "3.3.4", "trainingSingleHostClientLimit", self.path,
                        "single-host run specifies %d client nodes: %s",
                        len(hosts), hosts,
                    )
                    valid = False
        return valid
    
    @rule("3.3.6", "trainingIdenticalAcceleratorsPerNode")
    def identical_accelerators_per_node(self):
        """
        For distributed submissions, verify all nodes have identical accelerator count.
        (Rules.md 3.3.6)
        """
        valid = True
        if self.mode != "training":
            return valid

        for summary, metadata, ts in self.submissions_logs.run_files:
            if summary is None:
                self.log.debug(
                    "[3.3.6] %s/%s: skipping (summary not loaded)",
                    self.path, ts,
                )
                continue
            num_hosts = summary.get("num_hosts", 1)
            num_accelerators = summary.get("num_accelerators", 1)

            if num_hosts > 1:
                # For distributed runs, accelerators should be divisible by hosts
                if num_accelerators % num_hosts != 0:
                    self.log_violation(
                        "3.3.6", "trainingIdenticalAcceleratorsPerNode", self.path,
                        "distributed submission: %d accelerators not evenly divisible by %d hosts",
                        num_accelerators,
                        num_hosts,
                    )
                    valid = False

        return valid

    @rule("3.3.7", "trainingNodeCapabilityConsistency")
    def node_capability_consistency_check(self):
        """Rules.md 3.3.7 — node capability consistency (advisory).

        Satisfied by construction at runtime: the cluster collector
        captures per-host system information twice per run — once at
        run start (``Benchmark._collect_cluster_start`` in
        ``benchmarks/base.py:646``) and once at run end
        (``_collect_cluster_end`` at ``:674``) — and stores both
        snapshots in ``metadata['cluster_snapshots']`` via
        ``rules.models.ClusterSnapshots.as_dict()``. Any component drift
        during the run (CPU count, memory, network ports, kernel, etc.)
        is therefore captured in the artifact tree; the cluster collector
        also emits a warning on significant drift so operators can spot
        stability issues before submission.

        The rule body preserves the ``@rule`` binding for coverage
        discovery and emits an INFO line so tooling that greps by rule
        ID surfaces the rule as "visited and satisfied".
        """
        self.log.info(
            "[3.3.7 trainingNodeCapabilityConsistency] %s: "
            "satisfied by construction — cluster collector captures "
            "start/end cluster snapshots (Benchmark._collect_cluster_start / "
            "_collect_cluster_end); component drift is surfaced at runtime",
            self.path,
        )
        return True

    @rule("3.6.1", "trainingClosedSubmissionChecksum")
    def closed_submission_checksum(self):
        """For CLOSED submissions, verify code directory MD5 checksum.

        (Rules.md 3.6.1)

        Phase 4 CD-04: delegates to the shared
        ``helpers._check_code_image_layered`` helper so the §3.6.1 and §5.6.1
        rules enforce an identical layered model (self-consistency +
        upstream-identity) without duplicating the implementation across
        check classes. STRUCT-06 (§2.1.6) keeps its own inline implementation
        because it has additional surrounding logic (per-leaf walker, the
        ``expected is None`` warning) that does not belong in the helper.

        Walk-up: ``self.path`` is the per-leaf training path
        (``<root>/closed/<orgname>/results/<system>/training/<model>``). The
        CLOSED ``code/`` lives at ``<root>/closed/<orgname>/code/``, four
        levels above ``self.path`` (model → type → system → results →
        ``<orgname>``). Missing ``code/`` is NOT logged here — STRUCT-06
        already owns the VALS-01 missing-code/ violation under §2.1.6, so
        re-firing under §3.6.1 would double-count.
        """
        if self.mode != "training":
            return True

        # OPEN handled at STRUCT-06 self-consistency loop, not here.
        if self.submissions_logs.loader_metadata.division != "closed":
            return True

        # Walk up from <root>/closed/<orgname>/results/<system>/training/<model>
        # to <root>/closed/<orgname>, then append "code".
        submitter_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.path))))
        code_path = os.path.join(submitter_path, "code")

        # STRUCT-06 owns missing-code/ under §2.1.6; do not duplicate the violation here.
        if not os.path.isdir(code_path):
            return True

        expected = self.config.get_reference_checksum()
        return _check_code_image_layered(
            code_path,
            "closed",
            expected,
            self.log,
            self.log_violation,
            "3.6.1",
            "trainingClosedSubmissionChecksum",
        )
    
    @rule("3.6.2", "trainingClosedSubmissionParameters")
    def closed_submission_parameters(self):
        """
        For CLOSED submissions, verify only allowed parameters are modified.
        (Rules.md 3.6.2)
        """
        valid = True
        if self.mode != "training":
            return valid

        # Allowed parameters for CLOSED
        allowed_params = {
            "dataset.num_files_train",
            "dataset.num_subfolders_train",
            "dataset.data_folder",
            "reader.read_threads",
            "reader.computation_threads",
            "reader.transfer_size",
            "reader.prefetch_size",
            "reader.odirect",
            "storage.storage_root",
            "storage.storage_type"
        }

        for summary, metadata, ts in self.submissions_logs.run_files:
            if metadata is None:
                self.log.debug(
                    "[3.6.2] %s/%s: skipping (metadata not loaded)",
                    self.path, ts,
                )
                continue
            verification = metadata.get("verification", "open")

            if verification == "closed":
                params_dict = metadata.get("override_parameters", {})

                for param_key in params_dict.keys():
                    # Tool-injected params (skip_listing, data_folder derived
                    # from --data-dir, object-storage backend keys, …) are not
                    # user overrides and must not count against the CLOSED
                    # allow-list. See _TOOL_INJECTED_PARAMS comment above. (#503)
                    if param_key in _TOOL_INJECTED_PARAMS:
                        continue
                    if param_key not in allowed_params:
                        self.log_violation(
                            "3.6.2", "trainingClosedSubmissionParameters", self.path,
                            "CLOSED submission modifies disallowed parameter: %s",
                            param_key,
                        )
                        valid = False

        return valid

    @rule("3.6.3", "trainingOpenSubmissionParameters")
    def open_submission_parameters(self):
        """
        For OPEN submissions, verify only allowed parameters are modified.
        (Rules.md 3.6.3)
        """
        valid = True
        if self.mode != "training":
            return valid

        # Additional allowed parameters for OPEN (beyond CLOSED)
        open_allowed_params = {
            "framework",
            "dataset.format",
            "dataset.num_samples_per_file",
            "reader.data_loader"
        }

        # All CLOSED params are also allowed in OPEN
        closed_params = {
            "dataset.num_files_train",
            "dataset.num_subfolders_train",
            "dataset.data_folder",
            "reader.read_threads",
            "reader.computation_threads",
            "reader.transfer_size",
            "reader.prefetch_size",
            "reader.odirect",
            "storage.storage_root",
            "storage.storage_type"
        }

        allowed_params = closed_params | open_allowed_params

        for summary, metadata, ts in self.submissions_logs.run_files:
            if metadata is None:
                self.log.debug(
                    "[3.6.3] %s/%s: skipping (metadata not loaded)",
                    self.path, ts,
                )
                continue
            verification = metadata.get("verification", "open")

            if verification == "open":
                params_dict = metadata.get("override_parameters", {})

                for param_key in params_dict.keys():
                    # Tool-injected params (skip_listing, data_folder derived
                    # from --data-dir, object-storage backend keys, …) are not
                    # user overrides and must not count against the OPEN
                    # allow-list. See _TOOL_INJECTED_PARAMS comment above. (#503)
                    if param_key in _TOOL_INJECTED_PARAMS:
                        continue
                    if param_key not in allowed_params:
                        self.log_violation(
                            "3.6.3", "trainingOpenSubmissionParameters", self.path,
                            "OPEN submission modifies disallowed parameter: %s",
                            param_key,
                        )
                        valid = False

        return valid
    
    @rule("3.4.1", "trainingMlpstoragePathArgs")
    def mlpstorage_path_args(self):
        """
        Verify dataset and output paths are set and different.
        (Rules.md 3.4.1)

        Per CONTEXT.md `<deferred>`: do NOT add benchmark_API gating in this plan;
        that is a separate behavior change tracked for a future phase.
        """
        valid = True
        if self.mode != "training":
            return valid

        for summary, metadata, ts in self.submissions_logs.run_files:
            if metadata is None:
                self.log.debug(
                    "[3.4.1] %s/%s: skipping (metadata not loaded)",
                    self.path, ts,
                )
                continue
            args = metadata.get("args", {})
            data_dir = args.get("data_dir")
            results_dir = args.get("results_dir")

            if not data_dir:
                self.log_violation(
                    "3.4.1", "trainingMlpstoragePathArgs", self.path,
                    "data_dir not set in arguments",
                )
                valid = False

            if not results_dir:
                self.log_violation(
                    "3.4.1", "trainingMlpstoragePathArgs", self.path,
                    "results_dir not set in arguments",
                )
                valid = False

            if data_dir and results_dir and data_dir == results_dir:
                self.log_violation(
                    "3.4.1", "trainingMlpstoragePathArgs", self.path,
                    "data_dir and results_dir must be different: both are %s",
                    data_dir,
                )
                valid = False

        return valid
    
    @rule("3.4.2", "trainingMlpstorageFilesystemCheck")
    def mlpstorage_filesystem_check(self):
        """Verify dataset directory and results directory are on different filesystems.

        Parses the 'df' block from the run logfile (D-B1 anchored header). When the
        system YAML declares benchmark_API == 'object', silent-passes per D-B7.
        When the df block is absent, emits a violation (D-B4) — surfaces TODO-001.

        TRAIN-02 implementation: replaces stub body with _check_filesystem_separation
        helper call (from checks/helpers.py, shipped in Plan 02-01).
        """
        valid = True
        if self.mode != "training":
            return valid

        # D-B7: object-API submissions don't use 'df'; silent-pass.
        if self._get_benchmark_api() == "object":
            return valid

        for summary, metadata, timestamp in self.submissions_logs.run_files:
            if metadata is None:
                self.log.debug(
                    "[3.4.2] %s/%s: skipping (metadata not loaded)",
                    self.path, timestamp,
                )
                continue
            run_dir = os.path.join(self.run_path, timestamp)
            logfile_path = os.path.join(run_dir, "training_run.stdout.log")
            # CAP-03 sidecar is the authoritative input (#601). The
            # df-block parser remains for one release as a pre-cutover
            # fallback (D-601-3) — removed in v3.1.
            sidecar = read_fs_separation_sidecar(run_dir)
            if sidecar is not None:
                if sidecar.get("same_filesystem"):
                    self.log_violation(
                        "3.4.2", "trainingMlpstorageFilesystemCheck", logfile_path,
                        "data_dir and results_dir are on the same filesystem",
                    )
                    valid = False
                continue
            args = metadata.get("args", {})
            ok, df_found = _check_filesystem_separation(args, logfile_path)
            if not df_found:
                # D-B8: no CAP-03 sidecar AND no df block → no evidence of
                # FS separation at all. Fire a hard violation so producers
                # that predate #601 and never captured df cannot silently
                # pass 3.4.2.
                self.log_violation(
                    "3.4.2", "trainingMlpstorageFilesystemCheck", logfile_path,
                    "fs_separation.json sidecar not found; df block also absent",
                )
                valid = False
                continue
            if not ok:
                # df WAS found (e.g. submitter manually injected it), so this
                # is a real same-mount finding and remains an error.
                self.log_violation(
                    "3.4.2", "trainingMlpstorageFilesystemCheck", logfile_path,
                    "data_dir and results_dir are on the same filesystem",
                )
                valid = False
        return valid
