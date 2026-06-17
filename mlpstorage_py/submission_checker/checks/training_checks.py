
from .base import BaseCheck
from ..constants import *
from ..configuration.configuration import Config
from ..loader import SubmissionLogs
from ..rule_registry import rule
from .helpers import _check_filesystem_separation

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
        return (
            system_file.get("system_under_test", {})
                       .get("solution", {})
                       .get("architecture", {})
                       .get("benchmark_API", "file")
        )

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
            combined_params = metadata.get("combined_params", {})

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
                combined_params = metadata.get("combined_params", {})
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
            dataset_params = metadata.get("combined_params", {}).get("dataset", {})
            num_files = int(dataset_params.get("num_files_train", 0))
            record_length = float(dataset_params.get("record_length_bytes", 0))
            num_samples_per_file = int(dataset_params.get("num_samples_per_file", 1))
            expected_size = num_files * num_samples_per_file * record_length / 1024 / 1024 / 1024
            break

        # Check datagen produced at least that much
        for summary, metadata, _ in self.submissions_logs.datagen_files:
            if metadata is None:
                continue
            dataset_params = metadata.get("combined_params", {}).get("dataset", {})
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
        """
        Verify that run data matches the calculated datasize exactly.
        (Rules.md 3.3.1)
        """
        # Question: Subfolders?
        # What are the true values of the dataset
        valid = True
        if self.mode != "training":
            return valid

        # Resolve expected dataset cardinalities up front. Returns None if
        # the workload directory name does not match a known model
        # ({unet3d, retinanet}); 2.1.11 trainingWorkloads already
        # flags the structural complaint for the non-conforming name, so
        # skip the cardinality cross-check rather than crash on the dict
        # lookup or on a None comparison below.
        expected_train = self.config.get_num_train_files(self.model)
        expected_eval = self.config.get_num_eval_files(self.model)
        if expected_train is None or expected_eval is None:
            self.log.info(
                "[3.3.1 trainingRunDataMatchesDatasize] %s: skipping "
                "cardinality cross-check — workload %r is not in known "
                "models {unet3d, retinanet}; see 2.1.11 violation "
                "for the structural complaint",
                self.path, self.model,
            )
            return valid

        for summary, metadata, ts in self.submissions_logs.run_files:
            if summary is None:
                self.log.debug(
                    "[3.3.1] %s/%s: skipping (summary not loaded; "
                    "missing summary.json reported under 2.1.19)",
                    self.path, ts,
                )
                continue
            num_files_train = summary.get("num_files_train", None)
            num_files_eval = summary.get("num_files_eval", None)

            if num_files_train is None:
                self.log_violation(
                    "3.3.1", "trainingRunDataMatchesDatasize", self.path,
                    "num_files_train not set",
                )
                valid = False
            elif num_files_train > expected_train:
                self.log_violation(
                    "3.3.1", "trainingRunDataMatchesDatasize", self.path,
                    "num_files_train should be lower than in dataset",
                )
                valid = False

            if num_files_eval is None:
                self.log_violation(
                    "3.3.1", "trainingRunDataMatchesDatasize", self.path,
                    "num_files_eval not set",
                )
                valid = False
            elif num_files_eval > expected_eval:
                self.log_violation(
                    "3.3.1", "trainingRunDataMatchesDatasize", self.path,
                    "num_files_eval should be lower than in dataset",
                )
                valid = False

        return valid
    
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
        """Rules.md 3.3.5 — distributed training data accessibility cross-check.

        Deferred stub: Rules.md 3.3.5 requires verifying that all data is accessible
        to all host nodes for distributed Training submissions. The current
        summary.json / metadata.json schemas do not surface a per-host accessibility
        signal, so a runtime cross-check is not yet implementable. The structural
        anchor is the schema-validated systems/<name>.yaml (deployment + clients
        block). This stub follows the same pattern as
        ``CheckpointingCheck.simultaneous_rw_support`` (TODO-002 analog) — emit
        info-level note, return True, contribute no violation. When richer
        per-host run data is captured upstream, replace the info call with the
        real check.
        """
        self.log.info(
            "[3.3.5 trainingDistributedDataAccessibility] %s: "
            "runtime accessibility cross-check deferred — schema-validated "
            "systems/<name>.yaml is the structural anchor; see TODO-002 pattern",
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
        """Rules.md 3.3.7 — node capability consistency cross-check (advisory).

        Rules.md 3.3.7 mandates that, for distributed Training submissions, the
        validator "should emit a warning (not fail the validation) if the
        physical nodes that run the benchmark code are widely enough different
        in their capability". Per-host capability data is not surfaced in the
        current summary.json schema, so this runtime check is deferred (analog
        of ``CheckpointingCheck.simultaneous_rw_support`` / TODO-002 pattern).
        Emits an info-level deferral note and returns True. When per-host
        capability data is captured upstream, replace the info call with the
        real ``self.log.warning`` advisory.
        """
        self.log.info(
            "[3.3.7 trainingNodeCapabilityConsistency] %s: "
            "runtime per-host capability cross-check deferred — current "
            "summary.json schema does not surface per-host capability data; "
            "see TODO-002 pattern",
            self.path,
        )
        return True

    @rule("3.6.1", "trainingClosedSubmissionChecksum")
    def closed_submission_checksum(self):
        """
        For CLOSED submissions, verify code directory MD5 checksum.
        (Rules.md 3.6.1)

        Stub: body currently returns True (decorator-only retrofit per Plan 03-02
        Task 1 step 1 — "decorator only, no body change"). The real implementation
        will leverage the QUAL-05 MD5 predicate landed in Phase 1.
        """
        # TODO
        return True
    
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
                params_dict = metadata.get("params_dict", {})

                for param_key in params_dict.keys():
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
                params_dict = metadata.get("params_dict", {})

                for param_key in params_dict.keys():
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
            logfile_path = os.path.join(self.run_path, timestamp, "training_run.stdout.log")
            args = metadata.get("args", {})
            ok, df_found = _check_filesystem_separation(args, logfile_path)
            if not df_found:
                self.log_violation(
                    "3.4.2", "trainingMlpstorageFilesystemCheck", logfile_path,
                    "df output not found",
                )
                valid = False
                continue
            if not ok:
                self.log_violation(
                    "3.4.2", "trainingMlpstorageFilesystemCheck", logfile_path,
                    "data_dir and results_dir are on the same filesystem",
                )
                valid = False
        return valid
