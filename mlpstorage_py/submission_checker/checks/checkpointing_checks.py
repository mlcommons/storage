
from .base import BaseCheck
from ..constants import *
from ..configuration.configuration import Config
from ..loader import SubmissionLogs
from ..rule_registry import rule
from .helpers import _check_filesystem_separation, _pair_checkpoint_runs, _parse_iso_gap

import os
import re
import yaml


class CheckpointingCheck(BaseCheck):
    """
    A check class for validating checkpointing parameters and related properties.
    Inherits from BaseCheck and receives a config and loader instance.
    """

    def __init__(self, log, config: Config, submissions_logs: SubmissionLogs):
        """
        Initialize CheckpointingChecks with configuration and loader.

        Args:
            config: A Config instance containing submission configuration.
            loader: A SubmissionLogs instance for accessing submission logs.
        """
        # Call parent constructor with the loader's log and submission path
        super().__init__(log=log, path=submissions_logs.loader_metadata.folder)
        self.config = config
        self.submissions_logs = submissions_logs.checkpoint_files
        self.system_file = submissions_logs.system_file  # D-D3: needed by CHKPT-04/05/06 for benchmark_API + capabilities reads
        self.name = "checkpointing checks"
        self.mode = submissions_logs.loader_metadata.mode
        self.benchmark = submissions_logs.loader_metadata.benchmark
        self.checks = []
        self.checkpointing_path = self.path
        self.init_checks()
    
    def init_checks(self):
        """Initialize the list of checks to run."""
        self.checks = [
            self.checkpoint_data_size_ratio,
            self.fsync_verification,
            self.model_configuration_req,
            self.closed_mpi_processes,
            self.closed_accelerators_per_host,
            self.aggregate_accelerator_memory,
            self.closed_checkpoint_parameters,
            self.checkpoint_path_args,
            self.subset_run_validation,
            self.open_mpi_processes,           # CHKPT-01 (Task 2a)
            self.cache_flush_validation,       # CHKPT-02 (Task 2a)
            self.checkpoint_invocation_structure,  # 4.7.1 strict structural enforcement
            self.total_test_duration,          # CHKPT-03 (Task 2a)
            self.remapping_time_reporting,     # CHKPT-04 (Task 2b)
            self.simultaneous_rw_support,      # CHKPT-05 (Task 2b)
            self.checkpoint_filesystem_check,  # CHKPT-06 (Task 2b)
        ]

    def _get_benchmark_api(self) -> str:
        """Return 'file' or 'object' (default 'file') from the schema-validated system YAML.

        Reads self.system_file (D-D3 assignment already done by Plan 02-01 Task 2)
        with safe .get chain. Per D-B7, trusts the schema-validation pass and does
        not re-validate.
        """
        if not self.system_file:
            return "file"
        return (
            self.system_file.get("system_under_test", {})
                            .get("solution", {})
                            .get("architecture", {})
                            .get("benchmark_API", "file")
        )

    def _get_capability(self, field: str):
        """Return capabilities.<field> from the system YAML, or None if absent.

        SystemYamlSchemaCheck owns the 'field missing' violation per D-A3;
        CHKPT-04/05 (Task 2b) silent-skip when this returns None to avoid double-emit.
        """
        if not self.system_file:
            return None
        return (
            self.system_file.get("system_under_test", {})
                            .get("solution", {})
                            .get("capabilities", {})
                            .get(field)
        )

    def _iter_valid_files(self):
        """Yield (summary, metadata, timestamp) tuples, skipping any where
        summary or metadata is None.

        Missing summary.json / metadata.json is reported by
        SubmissionStructureCheck under 2.1.22 (checkpointingResultsJson) and
        2.1.25 (checkpointingFiles). This generator silently skips None
        entries so per-rule checks don't crash with AttributeError on
        ``None.get(...)`` and don't double-report the same structural cause.
        """
        for summary, metadata, timestamp in self.submissions_logs:
            if summary is None or metadata is None:
                self.log.debug(
                    "[checkpointing] skipping %s (summary or metadata not loaded)",
                    timestamp,
                )
                continue
            yield summary, metadata, timestamp
    
    @rule("4.3.1", "checkpointDataSizeRatio")
    def checkpoint_data_size_ratio(self):
        """
        Verify that checkpoint data written per node > 3x node memory.
        (Rules.md 4.3.1)

        Per Plan 03-02 PATTERNS.md: Rules.md 4.3.1 is advisory ("filesystem cache
        needs to be cleared"), not a hard violation. The advisory uses
        ``self.warn_violation`` (warning level) rather than ``self.log_violation``
        (error level) to preserve the original ``self.log.warning`` semantics.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid

        for summary, metadata, _ in self._iter_valid_files():
            checkpoint_size_gb = summary.get("metric", {}).get("checkpoint_size_GB", 0)
            host_memory_gb = summary.get("host_memory_GB", [0])[0]
            num_hosts = summary.get("num_hosts", 1)

            if checkpoint_size_gb == 0 or host_memory_gb == 0:
                continue

            # Data written per node
            data_per_node = checkpoint_size_gb / num_hosts
            min_required = 3 * host_memory_gb

            if data_per_node < min_required:
                self.warn_violation(
                    "4.3.1", "checkpointDataSizeRatio", self.path,
                    "Checkpoint data per node %.2fGiB < 3x memory %.2fGiB. "
                    "Cache flush may be needed.",
                    data_per_node,
                    min_required,
                )

        return valid
    
    @rule("4.3.2", "checkpointFsyncVerification")
    def fsync_verification(self):
        """
        Verify that fsync is enabled in checkpoint configuration.
        (Rules.md 4.3.2)
        """
        valid = True
        if self.mode != "checkpointing":
            return valid

        for summary, metadata, _ in self._iter_valid_files():
            combined_params = metadata.get("combined_params", {})
            checkpoint_params = combined_params.get("checkpoint", {})
            fsync_enabled = checkpoint_params.get("fsync", False)

            if not fsync_enabled:
                self.log_violation(
                    "4.3.2", "checkpointFsyncVerification", self.path,
                    "checkpoint fsync is not enabled in configuration",
                )
                valid = False

        return valid
    
    @rule("4.3.3", "checkpointModelConfigurationReq")
    def model_configuration_req(self):
        """
        Verify benchmark uses one of the four supported models.
        (Rules.md 4.3.3)
        """
        valid = True
        if self.mode != "checkpointing":
            return valid

        allowed_models = {"8b", "70b", "405b", "1t"}

        for summary, metadata, _ in self._iter_valid_files():
            model_name = metadata.get("args", {}).get("model", "").lower()

            # Extract just the size part (8b, 70b, etc.)
            model_size = re.search(r"(8b|70b|405b|1t)", model_name)

            if not model_size or model_size.group(1) not in allowed_models:
                self.log_violation(
                    "4.3.3", "checkpointModelConfigurationReq", self.path,
                    "invalid model '%s'. Must be one of: %s",
                    model_name,
                    allowed_models,
                )
                valid = False

        return valid
    
    @rule("4.6.1", "checkpointClosedMpiProcesses")
    def closed_mpi_processes(self):
        """For CLOSED submissions, verify MPI processes match requirements per model.

        BUG-03 fix (D-C4): derives model_key BEFORE the subset-branch check so
        it is always defined when used. Replaces the inline model_process_requirements
        dict with self.config.get_closed_mpi_processes(model_key) indirection.
        Uses self.log_violation (QUAL-02 retro-fit) instead of bare self.log.error.

        subset mode: requires exactly 8 processes for any model.
        non-subset (combined/full) mode: requires CLOSED_MPI_PROCESSES[model_key]
        total processes (Rules.md Table 2 — TP*PP*DP).
        """
        valid = True
        if self.mode != "checkpointing":
            return valid

        for summary, metadata, _ in self._iter_valid_files():
            verification = metadata.get("verification", "closed")

            if verification == "closed":
                checkpoint_mode = metadata.get("params_dict", {}).get("checkpoint.mode", "").lower()
                model_name = metadata.get("args", {}).get("model", "").lower()
                num_processes = metadata.get("args", {}).get("num_processes", 0)

                # BUG-03 fix: derive model_key BEFORE any branch that uses it
                model_size = re.search(r"(8b|70b|405b|1t)", model_name)
                model_key = model_size.group(1) if model_size else None

                if checkpoint_mode == "subset":
                    if num_processes != 8:
                        self.log_violation(
                            "4.6.1", "checkpointClosedMpiProcesses", self.path,
                            "CLOSED subset mode requires 8 processes, got %d (model: %s)",
                            num_processes, model_key or model_name,
                        )
                        valid = False
                else:
                    if model_key:
                        required = self.config.get_closed_mpi_processes(model_key)
                        if num_processes != required:
                            self.log_violation(
                                "4.6.1", "checkpointClosedMpiProcesses", self.path,
                                "CLOSED submission model %s requires %d processes, got %d",
                                model_key, required, num_processes,
                            )
                            valid = False

        return valid
    
    @rule("4.6.2", "checkpointClosedAcceleratorsPerHost")
    def closed_accelerators_per_host(self):
        """
        For CLOSED submissions, verify accelerators per host > 4 and total matches requirement.
        (Rules.md 4.6.2)
        """
        valid = True
        if self.mode != "checkpointing":
            return valid

        for summary, metadata, _ in self._iter_valid_files():
            verification = metadata.get("verification", "open")

            if verification == "closed":
                num_accelerators = summary.get("num_accelerators", 0)
                num_hosts = summary.get("num_hosts", 1)

                accelerators_per_host = num_accelerators / num_hosts if num_hosts > 0 else 0

                if accelerators_per_host <= 4:
                    self.log_violation(
                        "4.6.2", "checkpointClosedAcceleratorsPerHost", self.path,
                        "CLOSED submission: accelerators per host %.2f must be > 4",
                        accelerators_per_host,
                    )
                    valid = False

        return valid
    
    @rule("4.3.4", "checkpointAggregateAcceleratorMemory")
    def aggregate_accelerator_memory(self):
        """
        Verify total accelerator memory >= checkpoint size.
        H100 has 80GB per accelerator.
        (Rules.md 4.3.4)
        """
        valid = True
        if self.mode != "checkpointing":
            return valid

        ACCELERATOR_MEMORY_GB = 80  # H100

        for summary, metadata, _ in self._iter_valid_files():
            checkpoint_size_gb = summary.get("metric", {}).get("checkpoint_size_GB", 0)
            num_accelerators = summary.get("num_accelerators", 0)

            total_accelerator_memory = num_accelerators * ACCELERATOR_MEMORY_GB

            if total_accelerator_memory < checkpoint_size_gb:
                self.log_violation(
                    "4.3.4", "checkpointAggregateAcceleratorMemory", self.path,
                    "aggregate accelerator memory %.2fGiB < checkpoint size %.2fGiB",
                    total_accelerator_memory,
                    checkpoint_size_gb,
                )
                valid = False

        return valid
    
    def _get_nested_value(self, config_dict, key_path):
        """
        Get a value from nested dictionary using dot notation.
        Example: "checkpoint.fsync" -> config_dict["checkpoint"]["fsync"]
        
        Args:
            config_dict: The dictionary to search
            key_path: Dot-separated key path
            
        Returns:
            The value if found, None otherwise
        """
        keys = key_path.split(".")
        current = config_dict
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _get_nested_items(self, d, prefix = ""):
        for key, value in d.items():
            if isinstance(value, dict):
                p = prefix + "." if prefix != "" else ""
                yield from self._get_nested_items(value, prefix = p + key)
            else:
                p = prefix + "." if prefix != "" else prefix
                yield (p + key, value)
    
    @rule("4.6.3", "checkpointClosedCheckpointParameters")
    def closed_checkpoint_parameters(self):
        """
        For CLOSED submissions, verify yaml parameters match reference file.
        Only checkpoint_folder is allowed to differ.
        (Rules.md 4.6.3)

        Per Plan 03-02 Task 2 step 3: both early-return error branches
        (reference-config-not-found and reference-config-load-failure) are
        routed through ``log_violation`` under rule 4.6.3 with the more-specific
        ``config_ref_full_path`` as the path argument.
        """

        config_ref_path = os.path.join(os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            os.pardir,
            "configs",
            "dlio",
            "workload"
        )
        config_ref_file = self.config.get_checkpoint_file(self.benchmark)
        valid = True
        if self.mode != "checkpointing":
            return valid
        # 2.1.21 checkpointingWorkloads flags non-conforming workload dir
        # names structurally (e.g. "llama3-8b_a100" instead of "llama3-8b").
        # When that happens, get_checkpoint_file returns None and the
        # reference-config join + read below crash with TypeError. Skip the
        # cross-check rather than crash; the structural rule owns the
        # complaint.
        if config_ref_file is None:
            self.log.info(
                "[4.6.3 checkpointClosedCheckpointParameters] %s: "
                "skipping reference-config cross-check — workload %r is "
                "not in known LLM models; see 2.1.21 violation for the "
                "structural complaint",
                self.path, self.benchmark,
            )
            return valid
        # Load reference YAML file.
        #
        # Per WR-09 (review 2026-06-10): both branches below break the per-loop
        # ``valid &= False ; continue`` accumulation pattern used elsewhere in
        # this file and return False directly. That is intentional: without a
        # loaded reference_config there is nothing to compare any per-summary
        # yaml_params against, so every iteration of the loop below would emit
        # a duplicate "reference not loaded" violation. Short-circuiting at the
        # infrastructure-error boundary keeps the diagnostic count to one per
        # missing/broken reference file. CLAUDE.md's
        # "log-and-continue across checks" rule still holds at the outer
        # ``run_checks`` boundary — this method returns False once, and the
        # next check method in DirectoryCheck/CheckpointingCheck still runs.
        config_ref_full_path = os.path.join(config_ref_path, config_ref_file)
        if not os.path.exists(config_ref_full_path):
            self.log_violation(
                "4.6.3", "checkpointClosedCheckpointParameters", config_ref_full_path,
                "reference config file not found",
            )
            return False

        try:
            with open(config_ref_full_path, 'r') as f:
                reference_config = yaml.safe_load(f)
        except Exception as e:
            self.log_violation(
                "4.6.3", "checkpointClosedCheckpointParameters", config_ref_full_path,
                "failed to load reference config file: %s",
                str(e),
            )
            return False

        allowed_diff_params = {
            "checkpoint.checkpoint_folder"
        }
        for summary, metadata, _ in self._iter_valid_files():
            verification = metadata.get("verification", "open")
            if verification == "closed":
                yaml_params = metadata.get("yaml_params", {})

                # Compare yaml parameters with reference config
                for key, value in self._get_nested_items(yaml_params):
                    # Skip allowed differing parameters
                    if key in allowed_diff_params:
                        continue

                    # Navigate reference config to find the parameter
                    ref_value = self._get_nested_value(reference_config, key)

                    if ref_value is None:
                        self.log_violation(
                            "4.6.3", "checkpointClosedCheckpointParameters", self.path,
                            "parameter %s not found in reference config",
                            key,
                        )
                        valid = False
                    elif value != ref_value:
                        self.log_violation(
                            "4.6.3", "checkpointClosedCheckpointParameters", self.path,
                            "CLOSED submission parameter %s differs from reference. "
                            "Expected: %s, Got: %s",
                            key,
                            ref_value,
                            value,
                        )
                        valid = False
        return valid
    
    @rule("4.4.1", "checkpointPathArgs")
    def checkpoint_path_args(self):
        """
        Verify checkpoint folder and output paths are set and different.
        (Rules.md 4.4.1)

        Per CONTEXT.md `<deferred>`: do NOT add benchmark_API gating in this plan;
        that is a separate behavior change tracked for a future phase.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid

        for summary, metadata, _ in self._iter_valid_files():
            args = metadata.get("args", {})
            checkpoint_folder = args.get("checkpoint_folder")
            results_dir = args.get("results_dir")

            if not checkpoint_folder:
                self.log_violation(
                    "4.4.1", "checkpointPathArgs", self.path,
                    "checkpoint_folder not set in arguments",
                )
                valid = False

            if not results_dir:
                self.log_violation(
                    "4.4.1", "checkpointPathArgs", self.path,
                    "results_dir not set in arguments",
                )
                valid = False

            if checkpoint_folder and results_dir and checkpoint_folder == results_dir:
                self.log_violation(
                    "4.4.1", "checkpointPathArgs", self.path,
                    "checkpoint_folder and results_dir must be different: both are %s",
                    checkpoint_folder,
                )
                valid = False

        return valid
    
    @rule("4.3.5", "checkpointSubsetRunValidation")
    def subset_run_validation(self):
        """
        For subset runs, verify exactly 8 accelerators and not 8B model.
        (Rules.md 4.3.5)
        """
        valid = True
        if self.mode != "checkpointing":
            return valid

        for summary, metadata, _ in self._iter_valid_files():
            params_dict = metadata.get("params_dict", {})
            checkpoint_mode = params_dict.get("checkpoint.mode", "")

            if checkpoint_mode == "subset":
                num_accelerators = summary.get("num_accelerators", 0)
                model_name = metadata.get("args", {}).get("model", "").lower()

                if num_accelerators != 8:
                    self.log_violation(
                        "4.3.5", "checkpointSubsetRunValidation", self.path,
                        "subset run requires exactly 8 accelerators, got %d",
                        num_accelerators,
                    )
                    valid = False

                if "8b" in model_name:
                    self.log_violation(
                        "4.3.5", "checkpointSubsetRunValidation", self.path,
                        "subset run cannot use 8B model: %s",
                        model_name,
                    )
                    valid = False

        return valid

    # -------------------------------------------------------------------------
    # CHKPT-01..03 — Phase 2 Plan 02-03 Task 2a
    # -------------------------------------------------------------------------

    @rule("4.6.4", "checkpointOpenSubmissionScaling")
    def open_mpi_processes(self):
        """For OPEN submissions, verify num_processes is a positive multiple of TP*PP. (Rules.md 4.6.4)

        TP and PP come from configs/dlio/workload/llama3_{key}.yaml via
        Config.get_model_parallelism. Silent-skips when model regex doesn't match
        one of {8b, 70b, 405b, 1t} (different rule's surface).
        """
        valid = True
        if self.mode != "checkpointing":
            return valid
        for summary, metadata, _ in self._iter_valid_files():
            verification = metadata.get("verification", "closed")
            if verification != "open":
                continue
            model_name = metadata.get("args", {}).get("model", "").lower()
            num_processes = metadata.get("args", {}).get("num_processes", 0)
            model_size = re.search(r"(8b|70b|405b|1t)", model_name)
            if not model_size:
                continue
            model_key = model_size.group(1)
            tp, pp = self.config.get_model_parallelism(model_key)
            tp_pp = tp * pp
            if tp_pp == 0:
                continue
            if num_processes <= 0 or num_processes % tp_pp != 0:
                self.log_violation(
                    "4.6.4", "checkpointOpenSubmissionScaling", self.path,
                    "num_processes (%d) is not a positive multiple of TP*PP (%d) for model %s",
                    num_processes, tp_pp, model_key,
                )
                valid = False
        return valid

    @rule("4.7.1", "checkpointCacheFlushValidation")
    def cache_flush_validation(self):
        """Detect cache-flush pattern in split-mode runs (write then read with <=30s gap).
        (Rules.md 4.7.1)

        Combined-mode submissions (write and read in the same run) are valid per
        Pitfall 15 — return True with no violation when no split-mode pairs exist.
        Reads timestamps from summary.json (tries 'end_time'/'start_time' first,
        then 'end'/'start' fallback per RESEARCH.md Gray Area 3).

        30-second threshold is strict per Rules.md 4.7.1; no slop applied (Gemini
        review concern #3 noted; revisit with CACHE_FLUSH_SLOP_SECONDS constant
        in constants.py if filesystem-metadata latency becomes empirically
        problematic).
        """
        valid = True
        if self.mode != "checkpointing":
            return valid
        pairs = _pair_checkpoint_runs(self.submissions_logs)
        if not pairs:
            return valid
        for write_entry, read_entry in pairs:
            write_summary, _, write_ts = write_entry
            read_summary, _, read_ts = read_entry
            write_end = (write_summary or {}).get("end_time") or (write_summary or {}).get("end")
            read_start = (read_summary or {}).get("start_time") or (read_summary or {}).get("start")
            if write_end is None or read_start is None:
                self.log_violation(
                    "4.7.1", "checkpointCacheFlushValidation", self.path,
                    "cannot compute cache-flush gap: missing end_time/start_time in summary.json "
                    "(write_ts=%s, read_ts=%s)", write_ts, read_ts,
                )
                valid = False
                continue
            try:
                gap_seconds = _parse_iso_gap(write_end, read_start)
            except (ValueError, TypeError) as e:
                self.log_violation(
                    "4.7.1", "checkpointCacheFlushValidation", self.path,
                    "cannot parse timestamps for gap computation: %s (write_ts=%s, read_ts=%s)",
                    str(e), write_ts, read_ts,
                )
                valid = False
                continue
            if gap_seconds > 30:
                self.log_violation(
                    "4.7.1", "checkpointCacheFlushValidation", self.path,
                    "cache-flush gap %.1f seconds exceeds 30-second limit "
                    "(write end=%s, read start=%s)",
                    gap_seconds, write_end, read_start,
                )
                valid = False
        return valid

    @rule("4.7.1", "checkpointCacheFlushValidation")
    def checkpoint_invocation_structure(self):
        """Enforce the strict invocation structure for CLOSED checkpointing (Rules.md 4.7.1).

        A CLOSED submission must consist of either:
          - a single invocation with --num-checkpoints-write=10 AND
            --num-checkpoints-read=10; or
          - exactly two invocations: a write phase (10/0) followed by a read
            phase (0/10), with the read phase starting AFTER the write phase
            has ended (no overlap).

        The 30-second upper bound on the inter-phase gap is enforced separately
        by ``cache_flush_validation``. OPEN submissions may use any non-negative
        integer for --num-checkpoints-* per Rules.md Table 3, so this check is
        a no-op for them.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid

        closed_runs = [
            (summary, metadata, ts)
            for summary, metadata, ts in self._iter_valid_files()
            if (metadata or {}).get("verification") == "closed"
        ]
        if not closed_runs:
            return valid

        REQUIRED = 10

        def _wr(metadata):
            args = (metadata or {}).get("args", {}) or {}
            return (
                int(args.get("num_checkpoints_write", 0) or 0),
                int(args.get("num_checkpoints_read", 0) or 0),
            )

        if len(closed_runs) == 1:
            _summary, metadata, ts = closed_runs[0]
            writes, reads = _wr(metadata)
            if (writes, reads) != (REQUIRED, REQUIRED):
                self.log_violation(
                    "4.7.1", "checkpointCacheFlushValidation", self.path,
                    "single-invocation CLOSED submission must use "
                    "--num-checkpoints-write=10 AND --num-checkpoints-read=10 "
                    "(got writes=%d, reads=%d in %s)",
                    writes, reads, ts,
                )
                valid = False
            return valid

        if len(closed_runs) == 2:
            # Sort by ts (YYYYMMDD_HHmmss is lexicographically chronological).
            ordered = sorted(closed_runs, key=lambda e: e[2])
            first_summary, first_md, first_ts = ordered[0]
            second_summary, second_md, second_ts = ordered[1]

            first_w, first_r = _wr(first_md)
            second_w, second_r = _wr(second_md)

            if (first_w, first_r) != (REQUIRED, 0) or (second_w, second_r) != (0, REQUIRED):
                self.log_violation(
                    "4.7.1", "checkpointCacheFlushValidation", self.path,
                    "two-invocation CLOSED submission must be write-phase (10/0) "
                    "followed by read-phase (0/10); got first=(writes=%d,reads=%d) "
                    "in %s, second=(writes=%d,reads=%d) in %s",
                    first_w, first_r, first_ts, second_w, second_r, second_ts,
                )
                valid = False
                return valid

            # Overlap check: read phase must start after write phase ends.
            write_end = (first_summary or {}).get("end_time") or (first_summary or {}).get("end")
            read_start = (second_summary or {}).get("start_time") or (second_summary or {}).get("start")
            if write_end and read_start:
                try:
                    gap_seconds = _parse_iso_gap(write_end, read_start)
                except (ValueError, TypeError):
                    # Parse errors are already reported by cache_flush_validation;
                    # don't double-log.
                    return valid
                if gap_seconds < 0:
                    self.log_violation(
                        "4.7.1", "checkpointCacheFlushValidation", self.path,
                        "read phase started %.1fs before write phase ended; "
                        "the two phases must not overlap (write_end=%s, read_start=%s)",
                        -gap_seconds, write_end, read_start,
                    )
                    valid = False
            return valid

        # 3+ invocations — disallowed.
        self.log_violation(
            "4.7.1", "checkpointCacheFlushValidation", self.path,
            "CLOSED checkpointing submission must consist of 1 or 2 invocations "
            "(got %d). See Rules.md 4.7.1.",
            len(closed_runs),
        )
        return False

    @rule("4.7.2", "checkpointTotalTestDuration")
    def total_test_duration(self):
        """Compute and log total checkpoint test duration. (Rules.md 4.7.2)

        Per D-D1: happy path emits via self.log.info (informational, not a violation).
        ResultExporter / RPT-01 (v2 milestone) will surface this value in CSV later.
        Failure path (missing/malformed timestamps) emits via log_violation.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid
        pairs = _pair_checkpoint_runs(self.submissions_logs)
        if not pairs:
            return valid
        first_write_summary, _, first_write_ts = pairs[0][0]
        last_read_summary, _, last_read_ts = pairs[-1][1]
        first_write_start = (first_write_summary or {}).get("start_time") or (first_write_summary or {}).get("start")
        last_read_end = (last_read_summary or {}).get("end_time") or (last_read_summary or {}).get("end")
        if first_write_start is None or last_read_end is None:
            self.log_violation(
                "4.7.2", "checkpointTotalTestDuration", self.path,
                "cannot compute total duration: missing start_time/end_time in summary.json",
            )
            return False
        try:
            total_duration = _parse_iso_gap(first_write_start, last_read_end)
        except (ValueError, TypeError) as e:
            self.log_violation(
                "4.7.2", "checkpointTotalTestDuration", self.path,
                "cannot compute total duration: %s", str(e),
            )
            return False
        # remap_interval is best-effort — failure does not invalidate the duration log.
        first_read_summary, _, _ = pairs[0][1]
        last_write_summary, _, _ = pairs[-1][0]
        first_read_start = (first_read_summary or {}).get("start_time") or (first_read_summary or {}).get("start")
        last_write_end = (last_write_summary or {}).get("end_time") or (last_write_summary or {}).get("end")
        remap_interval = None
        if first_read_start and last_write_end:
            try:
                remap_interval = _parse_iso_gap(last_write_end, first_read_start)
            except (ValueError, TypeError):
                remap_interval = None
        # D-D1: happy path uses self.log.info, NOT log_violation
        self.log.info(
            "[4.7.2 checkpointTotalTestDuration] %s: "
            "total_test_duration_seconds=%.1f "
            "(write_start=%s, read_end=%s, remap_interval=%s)",
            self.path, total_duration, first_write_start, last_read_end,
            f"{remap_interval:.1f}" if remap_interval is not None else "N/A",
        )
        return valid

    # -------------------------------------------------------------------------
    # CHKPT-04..06 — Phase 2 Plan 02-03 Task 2b
    # -------------------------------------------------------------------------

    @rule("4.7.3", "checkpointRemappingTimeReporting")
    def remapping_time_reporting(self):
        """Cross-check declared remap_time_in_seconds against observed remap interval.
        (Rules.md 4.7.3)

        Schema validation (SystemYamlSchemaCheck, Plan 02-02) covers presence + type +
        Rule-13 consistency per D-A3. This runtime cross-check uses a 0.5x tolerance
        band: observed interval should be at least declared * 0.5 (else likely
        under-reported).

        Returns True silently if the capability field is absent — SystemYamlSchemaCheck
        owns the missing-field violation.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid
        declared = self._get_capability("remap_time_in_seconds")
        if declared is None:
            return valid
        if declared == 0:
            return valid
        pairs = _pair_checkpoint_runs(self.submissions_logs)
        if not pairs:
            return valid
        first_read_summary, _, _ = pairs[0][1]
        last_write_summary, _, _ = pairs[-1][0]
        first_read_start = (first_read_summary or {}).get("start_time") or (first_read_summary or {}).get("start")
        last_write_end = (last_write_summary or {}).get("end_time") or (last_write_summary or {}).get("end")
        if first_read_start is None or last_write_end is None:
            self.log_violation(
                "4.7.3", "checkpointRemappingTimeReporting", self.path,
                "cannot verify remap interval: missing timestamps in summary.json",
            )
            return False
        try:
            observed_remap = _parse_iso_gap(last_write_end, first_read_start)
        except (ValueError, TypeError):
            return valid   # CHKPT-03 will emit the parse violation
        if observed_remap < declared * 0.5:
            self.log_violation(
                "4.7.3", "checkpointRemappingTimeReporting", self.path,
                "observed remap interval (%.1fs) is much less than declared remap_time_in_seconds (%ds)",
                observed_remap, declared,
            )
            valid = False
        return valid

    @rule("4.7.4", "checkpointSimultaneousRwSupport")
    def simultaneous_rw_support(self):
        """Verify declared simultaneous R/W capabilities are consistent with run data.
        (Rules.md 4.7.4)

        Schema validation (SystemYamlSchemaCheck, Plan 02-02) covers field presence +
        type + Rule-13 cross-field consistency per D-A3. This runtime cross-check is
        DEFERRED (TODO-002) — current summary.json does not expose per-host timing,
        which is required to determine whether write/read overlapped on the same host.
        Method emits an informational log.info noting that schema-validation owns
        the rule for Phase 2 and returns True. The deferred state is pinned by
        TestChkpt05DeferredFollowUp in Plan 02-04.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid
        sim_write = self._get_capability("simultaneous_write")
        sim_read = self._get_capability("simultaneous_read")
        if sim_write is None or sim_read is None:
            return valid
        self.log.info(
            "[4.7.4 checkpointSimultaneousRwSupport] %s: "
            "schema validation (SystemYamlSchemaCheck) covers Rules.md 4.7.4 "
            "structural requirements; runtime per-host cross-check awaits "
            "richer summary.json data — see TODO-002 "
            "(simultaneous_write=%s, simultaneous_read=%s)",
            self.path, sim_write, sim_read,
        )
        return valid

    @rule("4.4.2", "checkpointFilesystemCheck")
    def checkpoint_filesystem_check(self):
        """Verify checkpoint_folder and results_dir are on different filesystems.
        (Rules.md 4.4.2 — Phase 2 NEW requirement CHKPT-06)

        Analog of TRAIN-02 for checkpointing. Per D-B5, shares
        _check_filesystem_separation helper. Per D-B7, silent-passes when
        benchmark_API == 'object'. Per D-B4, 'df output not found' is itself a
        violation tagged with this rule ID.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid
        if self._get_benchmark_api() == "object":
            return valid
        for summary, metadata, timestamp in self._iter_valid_files():
            logfile_path = os.path.join(self.checkpointing_path, timestamp, "checkpointing_run.stdout.log")
            args = metadata.get("args", {})
            # For checkpointing, checkpoint_folder is the "data path" analog (RESEARCH.md).
            chkpt_args = {
                "data_dir": args.get("checkpoint_folder"),
                "results_dir": args.get("results_dir"),
            }
            ok, df_found = _check_filesystem_separation(chkpt_args, logfile_path)
            if not df_found:
                self.log_violation(
                    "4.4.2", "checkpointFilesystemCheck", logfile_path,
                    "df output not found",
                )
                valid = False
                continue
            if not ok:
                self.log_violation(
                    "4.4.2", "checkpointFilesystemCheck", logfile_path,
                    "checkpoint_folder and results_dir are on the same filesystem",
                )
                valid = False
        return valid
