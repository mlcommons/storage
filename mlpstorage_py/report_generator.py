"""
Report generation for MLPerf Storage benchmark results.

This module provides the ReportGenerator class for validating and
reporting on benchmark results with clear OPEN vs CLOSED messaging.
"""

from __future__ import annotations
from typing import Union, Literal

import csv
import json
import os.path
import pprint
import statistics
import sys

from dataclasses import dataclass
from statistics import fmean, StatisticsError
from typing import List, Dict, Any, Optional, Set

from mlpstorage_py.mlps_logging import setup_logging, apply_logging_options
from mlpstorage_py.config import MLPS_DEBUG, BENCHMARK_TYPES, EXIT_CODE, PARAM_VALIDATION, LLM_MODELS, LLM_ALLOWED_VALUES, MODELS, ACCELERATORS
from mlpstorage_py.rules import get_runs_files, BenchmarkVerifier, BenchmarkRun, Issue, RunID
from mlpstorage_py.rules.datagen_hierarchy import (
    validate_datagen_leaf,
    validate_supported_model,
)
from mlpstorage_py.errors import ConfigurationError
from mlpstorage_py.utils import MLPSJsonEncoder
from mlpstorage_py.reporting import (
    ResultsDirectoryValidator,
    ValidationMessageFormatter,
    ClosedRequirementsFormatter,
    ReportSummaryFormatter,
    discover_scan_roots,
)


# D-24 verbatim INVALID message templates. DO NOT paraphrase — tests/unit/test_aggregation.py
# asserts these substrings verbatim (Phase 5 D-02 precedent). Formatted at call sites via
# ``.format(n=..., key=..., basename=...)``; the ``§`` (U+00A7) character is intentional and
# grep-tested.
_INVALID_MSG_TRAINING_COUNT = (
    "expected 6 training invocations per Rules.md §2.1.17 (1 warmup + 5 real); found {n}"
)
_INVALID_MSG_CHECKPOINT_COUNT = (
    "expected 10 checkpoint operations per Rules.md §2.1.23; found {n}"
)
_INVALID_MSG_EMPTY_METRIC = (
    "metric {key} is empty in invocation {basename}; cannot aggregate"
)


class Hyperlink:
    """A results-table cell that is a (visible-text, href) pair.

    Renders differently per output format (Task F, v3.0 results table):

    - CSV: ``str(link)`` yields an HTML anchor ``<a href="...">text</a>``
      (``csv.DictWriter`` str-coerces non-string cell values, and
      ``flatten_nested_dict`` treats this non-dict object as a leaf).
    - JSON: ``MLPSJsonEncoder`` serializes it via its ``__dict__`` branch
      to ``{"text": ..., "href": ...}``.

    ``href`` is a repo-root-relative URL (the aggregated-submissions repo
    root == ``<results-dir>``); a future change prepends the full repo URL.
    """

    def __init__(self, text: str, href: str) -> None:
        self.text = text
        self.href = href

    def __str__(self) -> str:
        return f'<a href="{self.href}">{self.text}</a>'

    def __repr__(self) -> str:
        return f"Hyperlink(text={self.text!r}, href={self.href!r})"

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Hyperlink)
                and self.text == other.text and self.href == other.href)


# --------------------------------------------------------------------------- #
# Column-parity fixed output schema (Results Table Structure.xlsx, v3.0).      #
#                                                                              #
# ``results.{csv,json}`` is a FIXED contract: exactly the reference webpage    #
# columns for all 8 tables (2 training + 4 checkpointing + 1 kvcache + 1 vdb)  #
# sharing a System-Under-Test block, plus 3 agreed discriminator columns       #
# (Division / Benchmark Type / Model) so a single flat file can carry every    #
# table's rows. A MLCommons staff member opens results.csv in Excel and         #
# reduces it to any one table by deleting the workload blocks + discriminators  #
# that don't apply.                                                            #
#                                                                              #
# The schema is FIXED — never data-driven. reportgen cherry-picks each column  #
# from the in-memory row (``_final_row``); a run that lacks a metric leaves    #
# that cell blank rather than dropping/adding a column. Metric-block names are  #
# unique per column (a row is a dict keyed by column name), so the shared      #
# reference names (# Client Nodes, Code, Read B/W, …) are workload-qualified.  #
# --------------------------------------------------------------------------- #

# Model column display labels (reference table titles). Blank for the
# single-table workloads (kvcache, vdb) — the Benchmark Type column
# identifies those.
_MODEL_DISPLAY_LABELS = {
    "unet3d": "Unet3D",
    "retinanet": "RetinaNet",
    "llama3-8b": "8B",
    "llama3-70b": "70B",
    "llama3-405b": "405B",
    "llama3-1t": "1250B",
}

# Left edge + shared System-Under-Test block. Each entry is
# (final_column_name, internal_row_key_or_None). ``None`` => manual /
# present-but-blank placeholder (Public ID + the 5 submitter-filled cells).
# Division / Benchmark Type / Model are computed in ``_final_row``.
_SUT_FINAL_COLUMNS = [
    ("Public ID", None),
    ("Organization", "orgname"),
    ("Division", "__division__"),
    ("Benchmark Type", "__benchmark_type__"),
    ("Model", "__model__"),
    ("Name", "sut_name"),
    ("Description", "sut_description"),
    ("Type", None),
    ("Access Protocol", None),
    ("Availability", None),
    ("RU's", "sut_rus"),
    ("Integrated Client Storage", None),
    ("Usable Capacity (TiB)", None),
]

# Per-workload metric/param blocks. ``__code__`` / ``__logs__`` are the
# per-row code-image hyperlinks (routed from the internal ``sut_code`` /
# ``sut_logs`` values into THIS row's own workload block; the reference
# places Code/Logs in each workload's Test-Parameters group because an
# OPEN row may ship its own code). Every other value is an internal
# ``_aggregate_*`` output key.
_TRAINING_BLOCK = [
    ("Accelerator Type", "accelerator"),
    ("# Client Nodes", "train_num_client_nodes"),
    ("Code", "__code__"),
    ("Logs", "__logs__"),
    ("# Simulated Accelerators", "train_num_simulated_accelerators"),
    ("Read B/W (GiB/s)", "train_read_bw_gibps"),
]
_CHECKPOINTING_BLOCK = [
    ("Checkpoint Mode", "checkpoint_mode"),
    ("# Client Nodes", "checkpoint_num_client_nodes"),
    ("DP Instances", "checkpoint_dp_instances"),
    ("Code", "__code__"),
    ("Logs", "__logs__"),
    ("Write B/W (GiB/s)", "checkpoint_write_bw_gibps"),
    ("Write Duration (secs)", "checkpoint_write_duration_secs"),
    ("Read B/W (GiB/s)", "checkpoint_read_bw_gibps"),
    ("Read Duration (secs)", "checkpoint_read_duration_secs"),
]
_VDB_BLOCK = [
    ("# Client Nodes", "vdb_num_client_nodes"),
    ("Code", "__code__"),
    ("Logs", "__logs__"),
    ("Vector Count", "vdb_num_vectors"),
    ("Vector Dimension", "vdb_dimension"),
    ("Index Type", "vdb_index_type"),
    ("Queries per Sec", "vdb_throughput_qps"),
    ("Query Latency (ms)", "vdb_p99_latency_ms"),
    ("Recall Percentage", "vdb_recall"),
    ("Storage IOPs", "vdb_storage_iops"),
    ("Read B/W (GiB/s)", "vdb_read_bw_gibps"),
]
# KVCache: 3 shared columns + 3 option-groups x 4 metrics. Groups map to
# the fixed kv-cache.py WORKLOAD_PARAMS option numbers (1/2/3). summary.json
# deserializes those keys as strings, so the internal column key uses the
# string option number.
_KVCACHE_SHARED_BLOCK = [
    ("# Client Nodes", "kvcache_num_client_nodes"),
    ("Code", "__code__"),
    ("Logs", "__logs__"),
]
_KVCACHE_GROUPS = [
    ("1", "llama3.1-8b Storage Only"),
    ("2", "llama3.1-8b Storage + Mem"),
    ("3", "llama3.1-70b Storage Only"),
]
_KVCACHE_GROUP_METRICS = [
    ("Throughput (tok/s)", "aggregated_avg_throughput_tokens_per_sec"),
    ("Read B/W (GiB/s)", "aggregated_read_bandwidth_gbps"),
    ("Write B/W (GiB/s)", "aggregated_write_bandwidth_gbps"),
    ("P95 Read Latency (ms)", "aggregated_p95_latency_ms"),
]

# Which workload block a row fills, keyed by the internal benchmark_type
# value. A row fills exactly one block; every other block stays blank.
_WORKLOAD_BLOCK_BY_TYPE = {
    "training": ("Training", _TRAINING_BLOCK),
    "checkpointing": ("Checkpointing", _CHECKPOINTING_BLOCK),
    "vector_database": ("VDB", _VDB_BLOCK),
    "kv_cache": ("KVCache", _KVCACHE_SHARED_BLOCK),  # + groups, see _final_row
}


def _build_final_schema():
    """Assemble the ordered fixed column list from the block definitions.

    Building the header from the SAME structures the projection reads keeps
    the two from drifting.
    """
    cols = [name for name, _ in _SUT_FINAL_COLUMNS]
    cols += [f"Training - {name}" for name, _ in _TRAINING_BLOCK]
    cols += [f"Checkpointing - {name}" for name, _ in _CHECKPOINTING_BLOCK]
    cols += [f"VDB - {name}" for name, _ in _VDB_BLOCK]
    cols += [f"KVCache - {name}" for name, _ in _KVCACHE_SHARED_BLOCK]
    for _opt, label in _KVCACHE_GROUPS:
        cols += [f"KVCache {label} - {name}" for name, _ in _KVCACHE_GROUP_METRICS]
    return cols


_FINAL_SCHEMA = _build_final_schema()


def _blank_if_nan(value: Any) -> Any:
    """Coerce a NaN float to ``""`` (JSON has no NaN literal; CSV wants blank).

    NaN is the only value not equal to itself, so this needs no ``math``
    import. Keeps every fixed column present-but-blank rather than dropping
    it (which ``remove_nan_values`` would do, breaking schema completeness).
    """
    if isinstance(value, float) and value != value:
        return ""
    return value


@dataclass
class Result:
    """Container for a single benchmark run result."""
    multi: bool
    benchmark_type: BENCHMARK_TYPES
    benchmark_command: str
    benchmark_model: Union[LLM_MODELS, MODELS, str]
    benchmark_run: Union[BenchmarkRun, List[BenchmarkRun]]
    issues: List[Issue]
    category: PARAM_VALIDATION
    metrics: Dict[str, Any]


class ReportGenerator:
    """
    Generate validation reports for benchmark results.

    This class provides:
    - Directory structure validation before processing
    - Clear OPEN vs CLOSED submission messaging
    - Error isolation for individual runs
    - Summary reports by category

    Args:
        results_dir: Path to the results directory.
        args: Optional argparse namespace with configuration.
        logger: Optional logger instance.
        validate_structure: Whether to validate directory structure (default True).
        use_colors: Whether to use terminal colors in output (default True).
    """

    def __init__(self, results_dir: str, args=None, logger=None,
                 validate_structure: bool = True, use_colors: bool = True):
        self.args = args
        if self.args is not None:
            self.debug = self.args.debug or MLPS_DEBUG
        else:
            self.debug = MLPS_DEBUG

        if logger:
            self.logger = logger
        else:
            # Ensure there is always a logger available
            self.logger = setup_logging(name="mlpstorage_py")
            apply_logging_options(self.logger, args)

        self.results_dir = results_dir
        # Per-org rollup targets for a multi-org tree (worklist A11
        # decision (a)): populated by _resolve_effective_results_dir when
        # the canonical probe finds >= 2 distinct orgs. Keyed by
        # (division, orgname) -> "<root>/<division>/<orgname>/results"
        # so generate_reports can drop each org's own results.{csv,json}
        # next to its per-system subdirectories.
        self._per_org_rollup_dirs: Dict[tuple, str] = {}
        # Detect the canonical mlpstorage submission tree (sentinel-bearing root
        # with <division>/<orgname>/results/<system>/<benchmark-type>/...) and
        # resolve --results-dir down to the per-system subtree that contains
        # the benchmark-type directories the rest of ReportGenerator expects.
        # No-op when --results-dir already points at a flat benchmark-type root.
        self.results_dir = self._resolve_effective_results_dir(self.results_dir)

        # Directory where the GLOBAL results.{json,csv} summary is written.
        # When we rebind results_dir to a per-system slice (canonical
        # submission tree), the global summary should NOT land inside that
        # system's folder — it should land one level up, in the
        # `<sentinel>/<div>/<org>/results/` directory that houses every
        # system's slice. That way a submitter reviewing the tree sees
        # one aggregate at the org's results/ level, next to each
        # per-system subdirectory.
        #
        # If results_dir was not rebound (flat layout / no canonical
        # match), the parent lookup falls back to results_dir itself.
        self.global_summary_dir = self._global_summary_dir_for(
            self.results_dir
        )

        # Issue #599: resolve the effective scan roots up-front.
        #
        # When --results-dir is a sentinel-bearing submission root (the
        # canonical layout that `mlpstorage init` / `<bench> run` / `validate`
        # all produce), the runs live at
        # `<results-dir>/<closed|open>/<orgname>/results/<systemname>/...`
        # — five levels below the directory the user passed in. Pre-fix,
        # ResultsDirectoryValidator looked for benchmark-type dirs at the
        # top level only and `get_runs_files` would walk every system's
        # subtree, mashing them into one report tagged with the requested
        # systemname.
        #
        # discover_scan_roots probes both modes against args.orgname (pinned
        # by the LAY-03 sentinel gate in main.py) and args.systemname (the
        # required CLI flag), returning per-mode slices when found and
        # falling back to [results_dir] for legacy flat layouts.
        orgname = getattr(self.args, 'orgname', None) if self.args else None
        systemname = (
            getattr(self.args, 'systemname', None) if self.args else None
        )
        self.scan_roots: List[str] = discover_scan_roots(
            results_dir, orgname=orgname, systemname=systemname,
            logger=self.logger,
        )

        # Initialize formatters
        self.msg_formatter = ValidationMessageFormatter(use_colors=use_colors)
        self.summary_formatter = ReportSummaryFormatter(use_colors=use_colors)

        # Validate directory structure first if requested
        if validate_structure:
            if not self._validate_directory_structure():
                sys.exit(EXIT_CODE.FILE_NOT_FOUND)

        # Keyed by (system_scope, run_id) so two systems producing the
        # same RunID (RunID is program+command+model+run_datetime — no
        # system field) do not collide and get mislabelled as warmup/real
        # pairs. See _process_single_run for the collision-detection logic
        # and _system_scope_key for how the scope is derived from the
        # canonical result-tree layout.
        self.run_results: Dict[tuple, Result] = {}
        self.workload_results: Dict[tuple, Result] = {}
        # Absolute paths of result_dir directories detected as warmup runs.
        # Absolute (not basenames) so two systems with the same <ts>
        # basename cannot cross-mark each other's real runs as warmups.
        self.warmup_result_dirs: set = set()
        self.processing_errors: List[str] = []

        self.accumulate_results()
        self.print_results()

    def _resolve_effective_results_dir(self, results_dir: str) -> str:
        """Resolve --results-dir to the directory that holds benchmark-type subdirs.

        Accepts both shapes:

        * Flat layout (legacy / what ResultsDirectoryValidator expected):
          ``<results-dir>/<benchmark-type>/<model>/...`` — returned unchanged.
        * Canonical mlpstorage submission tree (what ``mlpstorage init`` /
          ``<bench> run`` / ``validate`` produce):
          ``<sentinel-root>/<division>/<orgname>/results/<system>/<benchmark-type>/...``
          — resolved down to the per-system subtree.

        When ``--systemname`` is supplied, the canonical-tree resolution
        scopes to ``results/<systemname>/`` so reportgen aggregates only
        that system's runs (fixes the prior behavior of walking every
        system's runs regardless of --systemname).

        When ``--systemname`` is NOT supplied (optional for reportgen), the
        canonical-tree resolution rebinds to the org's ``results/`` folder
        itself so the global summary lands there (next to each system's
        subdirectory). The per-system walk is still handled correctly by
        ``discover_scan_roots``, which enumerates every system slice.
        """
        from pathlib import Path  # local import to avoid hoisting Path globally
        root = Path(results_dir)
        if not root.is_dir():
            return results_dir
        # Already flat?
        expected = ResultsDirectoryValidator.EXPECTED_BENCHMARK_TYPES
        if any((root / b).is_dir() for b in expected):
            return results_dir
        # Canonical tree probe. Collect EVERY (division, org) match first
        # — returning on the first org found is the worklist A11 bug: on
        # an assembled multi-submitter tree the alphabetically-first
        # org's results/ received the global rollup for everyone.
        systemname = None
        if self.args is not None:
            systemname = getattr(self.args, 'systemname', None)
        # (division, org_dir, rebind_target) per canonical match.
        matches: List[tuple] = []
        for division in ('closed', 'open'):
            division_dir = root / division
            if not division_dir.is_dir():
                continue
            for org_dir in sorted(p for p in division_dir.iterdir() if p.is_dir()):
                results_root = org_dir / 'results'
                if not results_root.is_dir():
                    continue
                if systemname:
                    system_dir = results_root / systemname
                    if system_dir.is_dir():
                        matches.append((division, org_dir, system_dir))
                else:
                    matches.append((division, org_dir, results_root))
        if not matches:
            return results_dir

        distinct_orgs = {org_dir.name for _div, org_dir, _t in matches}
        if len(distinct_orgs) > 1:
            # Multi-org tree (A11 decision (a), 2026-07-24): keep the
            # tree root as the effective results dir so the GLOBAL
            # rollup lands there (covering every org), and register each
            # org's results/ folder for its own per-org rollup.
            self._per_org_rollup_dirs = {
                (division, org_dir.name): str(org_dir / 'results')
                for division, org_dir, _target in matches
            }
            self.logger.info(
                "Detected multi-org submission tree (%d orgs) under %s; "
                "global rollup lands at the tree root, per-org rollups "
                "in each <division>/<org>/results/",
                len(distinct_orgs), root,
            )
            return results_dir

        # Single org — preserve the established single-submitter
        # behavior: rebind to the first matching slice.
        division, org_dir, target = matches[0]
        if systemname:
            self.logger.info(
                "Detected canonical submission tree; scoping to "
                "%s/results/%s for --systemname=%s",
                org_dir, systemname, systemname,
            )
        else:
            # No --systemname: rebind to the org's results/ folder
            # itself. The global summary lands here (see
            # _global_summary_dir_for), and discover_scan_roots
            # walks each system slice under it.
            self.logger.info(
                "Detected canonical submission tree without "
                "--systemname; aggregating across every system "
                "under %s",
                target,
            )
        return str(target)

    def _global_summary_dir_for(self, effective_results_dir: str) -> str:
        """Return the directory where the GLOBAL rollup should be written.

        Two canonical-tree shapes trigger a redirect to the org's
        ``results/`` folder:

        * ``<sentinel>/<div>/<org>/results/<system>/`` — a per-system slice
          (parent basename == ``results``): return that parent.
        * ``<sentinel>/<div>/<org>/results/`` — the org's ``results/``
          folder itself (used in the no-``--systemname`` aggregation
          mode): return it directly.

        In either case the global summary sits next to each per-system
        subdirectory rather than inside one of them, so a submitter sees
        a single aggregate at the org's ``results/`` level.

        Otherwise (flat layout, or the resolver did not rebind), return
        ``effective_results_dir`` unchanged.
        """
        abs_dir = os.path.abspath(effective_results_dir)
        # Case: results_dir IS the org's results/ folder (no --systemname
        # canonical-tree rebind).
        if os.path.basename(abs_dir) == "results" and os.path.isdir(abs_dir):
            self.logger.debug(
                "Global summary directory resolved to canonical results/ "
                "folder itself: %s",
                abs_dir,
            )
            return abs_dir
        # Case: results_dir is a per-system slice under a results/ parent.
        parent = os.path.dirname(abs_dir)
        if os.path.basename(parent) == "results" and os.path.isdir(parent):
            self.logger.debug(
                "Global summary directory resolved to canonical results/ "
                "parent: %s",
                parent,
            )
            return parent
        return effective_results_dir

    def _validate_directory_structure(self) -> bool:
        """
        Validate the results directory structure before processing.

        When the canonical submission layout is detected (one or both of
        ``<results-dir>/{closed,open}/<orgname>/results/<systemname>/``),
        the validator runs against each canonical slice independently —
        each slice is itself a flat-layout root structurally
        (``<benchmark>/<model>/<command>/<datetime>/`` immediately below).

        Returns:
            True if structure is valid, False otherwise.
        """
        total_runs = 0
        total_benchmark_types: set = set()
        all_warnings: List[str] = []
        any_failed = False
        last_validator: Optional[ResultsDirectoryValidator] = None

        for scan_root in self.scan_roots:
            validator = ResultsDirectoryValidator(scan_root, logger=self.logger)
            last_validator = validator
            result = validator.validate()

            if not result.is_valid:
                self.logger.error(
                    f"Results directory structure validation failed for "
                    f"{scan_root}:"
                )
                self.logger.error(validator.get_error_report())
                any_failed = True
                continue

            total_runs += result.found_runs
            total_benchmark_types.update(result.found_benchmark_types)
            all_warnings.extend(result.warnings)

        if any_failed:
            self.logger.error("")
            self.logger.error("Expected structure:")
            if last_validator is not None:
                self.logger.error(last_validator.get_expected_structure_help())
            return False

        for warning in all_warnings:
            self.logger.warning(warning)

        self.logger.info(
            f"Directory validation passed: found {total_runs} runs "
            f"in {len(total_benchmark_types)} benchmark types "
            f"across {len(self.scan_roots)} scan root(s)"
        )
        return True

    def generate_reports(self):
        # Verify the results directory exists:
        self.logger.info(f'Generating reports for {self.results_dir}')

        # Bottom-up build per D-02 / D-03 / D-08 / D-09.
        #
        # Data-flow inversion vs post-PR #620:
        #
        #   (a) Iterate `self.workload_results` — the SINGLE source of
        #       truth. One aggregated row per workload. Category /
        #       orgname / systemname / benchmark_type / model /
        #       accelerator make up the D-10 6-column prefix; the
        #       aggregation dict from `_aggregate_workload_metrics` fills
        #       the D-11 grouped body; the `; `-joined `Result.issues`
        #       fills the trailing D-12 `issues` column.
        #   (b) Group the row dicts per-model via
        #       `_model_group_folder_for_workload` (workload-aware
        #       wrapper around the existing `_model_group_folder`).
        #   (c) Emit per-model `results.{csv,json}` FIRST — the source
        #       of truth for each model.
        #   (d) Emit empty per-model `results.{csv,json}` (header-only
        #       CSV, `[]` JSON) for any on-disk `<...>/<model>/`
        #       directory that produced zero workload rows. Closes the
        #       D-03 empty-model-dir corner in Phase 6 itself.
        #   (e) Assemble the top-level file BOTTOM-UP by concatenating
        #       every per-model row list into `all_rows`, sorted by the
        #       6-column prefix for deterministic order across runs.
        #   (f) Emit top-level `results.{csv,json}` at
        #       `self.global_summary_dir`.
        #
        # SC-6 grep gate: both per-model and top-level writer call
        # sites use `target_dir=<...>` explicitly, so
        # `grep -c 'target_dir=' mlpstorage_py/report_generator.py`
        # returns >= 2 (in practice: 4 — per-model json + per-model
        # csv + top-level json + top-level csv + Task 4's empty-emission
        # sites).
        #
        # D-04 preservation: only paths matching
        # `<...>/<model>/results.{csv,json}` and
        # `<global_summary_dir>/results.{csv,json}` are ever written.
        # Unrelated files under `<results-dir>` are not touched.

        # (a) + (b): iterate workload_results, build rows, group per model.
        rows_by_model: Dict[str, List[dict]] = {}
        skipped_no_model_folder = 0
        for workload_key, workload_result in self.workload_results.items():
            row = self._workload_result_to_row(workload_key, workload_result)
            model_folder = self._model_group_folder_for_workload(workload_result)
            if model_folder is None:
                skipped_no_model_folder += 1
                continue
            rows_by_model.setdefault(model_folder, []).append(row)

        # D-08 "no runs" preservation — early-return SUCCESS when there
        # is genuinely nothing to write. Empty-model-dir emission (Task
        # 4) still runs so on-disk model dirs get a header-only CSV /
        # `[]` JSON.
        if not self.workload_results:
            self.logger.warning(
                "No workload results to write rollups for "
                "(skipped %d workloads without result_dir).",
                skipped_no_model_folder,
            )
            # Still walk on-disk model dirs so D-03 empty-model-dir
            # emission fires.
            self._emit_empty_model_dirs(rows_by_model)
            return EXIT_CODE.SUCCESS

        # (c) Per-model file emission — the D-02 "per-model file IS the
        # source" step. Sorted for deterministic order.
        for model_folder, rows in sorted(rows_by_model.items()):
            self.write_json_file(rows, target_dir=model_folder)
            self.write_csv_file(rows, target_dir=model_folder)

        # (d) Empty-model-dir emission (D-03 corner) — MUST run BEFORE
        # top-level assembly so it does not contribute rows to
        # `all_rows`. Implemented in Task 4 as an explicit walker.
        self._emit_empty_model_dirs(rows_by_model)

        # (e) Bottom-up top-level assembly — concatenate every per-model
        # list, sort by the 6-column prefix for deterministic order.
        all_rows: List[dict] = []
        for _folder, rows in sorted(rows_by_model.items()):
            all_rows.extend(rows)
        all_rows.sort(
            key=lambda r: (
                r.get('category', '') or '',
                r.get('orgname', '') or '',
                r.get('systemname', '') or '',
                r.get('benchmark_type', '') or '',
                r.get('model', '') or '',
                r.get('accelerator', '') or '',
            )
        )

        # (f) Top-level file emission — the D-02 "collection, not
        # aggregation" step. Also emitted when `rows_by_model` was
        # empty but `workload_results` was not (e.g., every workload
        # lacked a result_dir); in that case `all_rows` is `[]` and the
        # top-level file becomes a header-only CSV / `[]` JSON.
        self.write_json_file(all_rows, target_dir=self.global_summary_dir)
        self.write_csv_file(all_rows, target_dir=self.global_summary_dir)

        # (g) Multi-org tree only (worklist A11 decision (a)): each org
        # additionally gets its own rollup at <division>/<org>/results/,
        # holding only that org's rows — the same placement a single-org
        # canonical tree gets for its global. `_per_org_rollup_dirs` is
        # empty outside multi-org mode, so this is a no-op elsewhere.
        for (division, org), target_dir in sorted(
            self._per_org_rollup_dirs.items()
        ):
            org_rows = [
                r for r in all_rows
                if (r.get('category', '') or '').lower() == division
                and r.get('orgname', '') == org
            ]
            self.write_json_file(org_rows, target_dir=target_dir)
            self.write_csv_file(org_rows, target_dir=target_dir)

        return EXIT_CODE.SUCCESS

    def _workload_result_to_row(
        self,
        workload_key: tuple,
        workload_result: 'Result',
    ) -> Dict[str, Any]:
        """Convert one aggregated workload ``Result`` into a flat row dict.

        Row shape per D-10/D-11/D-12/D-14:

        - Fixed 6-column prefix populated from the workload key
          (path-derived when possible) and from ``benchmark_run[0]``.
        - Aggregated metric columns from ``Result.metrics`` (already
          prefixed by ``_aggregate_workload_metrics``).
        - Trailing ``issues`` column: verbatim ``Result.issues`` joined
          by ``'; '`` per D-25.
        """
        # Unpack the 5-tuple key. Shape is always
        # (category, orgname, systemname, id1, id2) per D-05; id1/id2
        # meaning varies by benchmark type but the prefix positions do
        # not, so this unpack is stable across types. Issue #771 adds an
        # optional 6th ``command`` discriminator for auxiliary training
        # commands — slice off the fixed 5-prefix so both shapes unpack.
        try:
            category_key, orgname_key, systemname_key, _id1, _id2 = workload_key[:5]
        except ValueError:
            category_key = orgname_key = systemname_key = ""

        # Resolve category value. Prefer the workload_key derivation
        # (path-based, D-07). Fall back to Result.category (enum or
        # string) when the key was empty.
        if category_key:
            category_val = category_key
        else:
            cat = workload_result.category
            try:
                category_val = cat.value  # PARAM_VALIDATION enum
            except AttributeError:
                category_val = str(cat) if cat is not None else ""

        first_run = (
            workload_result.benchmark_run[0]
            if isinstance(workload_result.benchmark_run, list)
            else workload_result.benchmark_run
        )
        bt = workload_result.benchmark_type
        bt_val = bt.value if bt is not None else ""

        # Model / accelerator identity (D-10). vdb rows leave model +
        # accelerator empty and carry engine / index_type in vdb_*
        # columns (D-14). kvcache rows have model populated and
        # accelerator empty.
        if bt == BENCHMARK_TYPES.vector_database:
            model_val = ""
            accelerator_val = ""
        elif bt == BENCHMARK_TYPES.kv_cache:
            model_val = str(getattr(first_run, 'model', "") or "")
            accelerator_val = ""
        else:
            model_val = str(getattr(first_run, 'model', "") or "")
            accelerator_val = str(getattr(first_run, 'accelerator', "") or "")

        row: Dict[str, Any] = {
            'category': category_val,
            'orgname': orgname_key,
            'systemname': systemname_key,
            'benchmark_type': bt_val,
            'model': model_val,
            'accelerator': accelerator_val,
        }
        # Task F: shared System-Under-Test block (from system-description.yaml),
        # inserted after the D-10 prefix and before metric groups so the
        # prefix stays first / issues stays last.
        row.update(self._sut_columns(
            category_val, orgname_key, systemname_key, first_run))
        # Aggregated metric columns (D-11 grouped body).
        for metric_key, metric_val in (workload_result.metrics or {}).items():
            row[metric_key] = metric_val

        # D-25 trailing `issues` column: verbatim Result.issues joined
        # by `'; '` (semicolon + single space). Grep gate:
        # `grep -c "'; '.join" ...` >= 1 lives here.
        issue_texts: List[str] = []
        for issue in (workload_result.issues or []):
            msg = getattr(issue, 'message', None)
            issue_texts.append(str(msg) if msg is not None else str(issue))
        row['issues'] = '; '.join(issue_texts)

        return row

    # Task F: shared System-Under-Test column order (v3.0 results table).
    # Group prefix `sut_`; placed after the D-10 prefix by _ordered_fieldnames.
    _SUT_COLUMNS = [
        'sut_public_id',
        'sut_organization',
        'sut_name',
        'sut_description',
        'sut_type',
        'sut_access_protocol',
        'sut_availability',
        'sut_rus',
        'sut_integrated_client_storage',
        'sut_usable_capacity_tib',
        'sut_code',
        'sut_logs',
    ]

    def _sut_columns(
        self,
        category: str,
        orgname: str,
        systemname: str,
        first_run: Any,
    ) -> Dict[str, Any]:
        """Build the shared SUT block for a workload row (Task F).

        AUTO-filled: ``sut_organization`` (path), ``sut_name`` /
        ``sut_description`` (Hyperlinks to the system-description
        ``.yaml`` / ``.pdf``), ``sut_rus`` (``total_rack_units``). The
        remaining six cells are present-but-blank placeholders the
        submitter fills manually. Hrefs are repo-root-relative (Option A).
        """
        cols: Dict[str, Any] = {key: "" for key in self._SUT_COLUMNS}
        cols['sut_organization'] = orgname or ""

        if category and orgname and systemname:
            base = f"{category}/{orgname}/systems/{systemname}"
            _submission_name, rack_units = self._read_system_description(
                first_run, systemname)
            # Visible text (user-confirmed 2026-07-21): Name shows the
            # system name and links to the system-description.yaml;
            # Description shows literal "PDF" and links to the .pdf.
            cols['sut_name'] = Hyperlink(systemname, f"{base}.yaml")
            cols['sut_description'] = Hyperlink("PDF", f"{base}.pdf")
            if rack_units is not None:
                cols['sut_rus'] = rack_units

            # Task F-b: Code/Logs anchors -> the run's content-addressed
            # code-image pool dir (repo-root-relative). Logs is a
            # placeholder pointing at the same dir for now (future change
            # will retarget it). Blank when the pointer is absent/malformed.
            code_href = self._code_image_href(category, orgname, first_run)
            if code_href:
                cols['sut_code'] = Hyperlink("code", code_href)
                cols['sut_logs'] = Hyperlink("logs", code_href)
        return cols

    def _code_image_href(
        self, category: str, orgname: str, first_run: Any,
    ) -> Optional[str]:
        """Repo-root-relative URL of the run's ``code-<hash8>`` pool dir.

        Resolves the run leaf's ``.mlps-code-image`` pointer via the
        canonical code_image helpers (D-61 pointer parse, D-62 pool-dir
        name). Returns ``None`` when the pointer is absent or unreadable —
        the caller then leaves Code/Logs blank.
        """
        result_dir = getattr(first_run, 'result_dir', None)
        if not result_dir:
            return None
        from pathlib import Path
        from mlpstorage_py.submission_checker.tools.code_image import (
            _read_pointer, _pool_dir_name, CodeImageError,
        )
        try:
            _alg, full_hash = _read_pointer(Path(result_dir), self.logger)
        except (FileNotFoundError, CodeImageError, OSError) as e:
            self.logger.warning(
                "reportgen: no/invalid code-image pointer at %s: %s; "
                "Code/Logs will be blank.", result_dir, e)
            return None
        return f"{category}/{orgname}/{_pool_dir_name(full_hash)}/"

    def _read_system_description(self, first_run: Any, systemname: str):
        """Locate + parse ``systems/<systemname>.yaml`` for a workload.

        Walks up from the run's ``result_dir`` to the first ancestor that
        contains ``systems/<systemname>.yaml`` (the ``<org>`` level of the
        submission tree). Returns ``(submission_name, total_rack_units)``,
        each ``None`` when the file is absent/unreadable — the caller then
        falls back to the systemname text and a blank RU cell.
        """
        result_dir = getattr(first_run, 'result_dir', None)
        if not result_dir:
            return None, None
        import yaml  # local import: pyyaml is a runtime dep, keep top clean
        from pathlib import Path
        start = Path(result_dir)
        for ancestor in [start, *start.parents]:
            candidate = ancestor / "systems" / f"{systemname}.yaml"
            if candidate.is_file():
                try:
                    with open(candidate, "r") as fh:
                        data = yaml.safe_load(fh) or {}
                    sut = data.get('system_under_test') or {}
                    solution = sut.get('solution') or {}
                    return (solution.get('submission_name'),
                            sut.get('total_rack_units'))
                except (OSError, yaml.YAMLError) as e:
                    self.logger.warning(
                        "reportgen: could not read system description %s: %s",
                        candidate, e)
                    return None, None
        self.logger.warning(
            "reportgen: no systems/%s.yaml found above %s; SUT block will "
            "fall back to systemname + blank RUs.", systemname, result_dir)
        return None, None

    def _model_group_folder_for_workload(
        self, workload_result: 'Result'
    ) -> Optional[str]:
        """Workload-aware wrapper around ``_model_group_folder``.

        Workload-level ``Result`` objects hold ``benchmark_run`` as a
        list; ``_model_group_folder`` expects a single BenchmarkRun.
        This wrapper synthesizes a single-run view (uses
        ``benchmark_run[0]``) so the existing folder-resolution logic
        can be reused as-is (D-02 "reuse `_model_group_folder`, do not
        replicate").
        """
        first_run = (
            workload_result.benchmark_run[0]
            if isinstance(workload_result.benchmark_run, list)
            else workload_result.benchmark_run
        )
        proxy = Result(
            multi=False,
            benchmark_type=workload_result.benchmark_type,
            benchmark_command=workload_result.benchmark_command,
            benchmark_model=workload_result.benchmark_model,
            benchmark_run=first_run,
            issues=[],
            category=workload_result.category,
            metrics={},
        )
        return self._model_group_folder(proxy)

    def _emit_empty_model_dirs(
        self, rows_by_model: Dict[str, List[dict]]
    ) -> None:
        """Emit header-only CSV + `[]` JSON for empty per-model dirs (D-03).

        Task 4 implementation site. Walks the on-disk per-model
        directories under ``<results-dir>`` (using the same shape map
        as ``_model_group_folder``) and, for every directory NOT
        already present in ``rows_by_model``, emits an empty
        ``results.csv`` (header row only) and an empty ``results.json``
        (`[]`).

        Preserves D-04: only touches paths matching the per-model
        directory shape, never deletes anything.
        """
        for model_dir in self._enumerate_on_disk_model_dirs():
            if model_dir in rows_by_model:
                continue
            # Empty rows list -> header-only CSV, [] JSON via
            # write_csv_file / write_json_file (which handle an empty
            # `flattened_results` list gracefully).
            self.write_json_file([], target_dir=model_dir)
            self.write_csv_file([], target_dir=model_dir)

    def _enumerate_on_disk_model_dirs(self) -> List[str]:
        """Return the absolute per-model directory paths on disk (D-03).

        Walks each `self.scan_roots` entry looking for the canonical
        per-model shape:

        - training:      ``<scan_root>/[...]/training/<model>/``
        - checkpointing: ``<scan_root>/[...]/checkpointing/<model>/``
        - kv_cache:      ``<scan_root>/[...]/kv_cache/<model>/``
        - vdb:           ``<scan_root>/[...]/vector_database/<engine>/<index_type>/``

        Uses `os.walk` bounded to a shallow depth relative to each scan
        root. Returns absolute paths. Any I/O errors during the walk
        are logged and skipped (D-04: never abort the whole pass).
        """
        from pathlib import Path

        found: Set[str] = set()
        benchmark_type_names = {
            'training', 'checkpointing', 'kv_cache', 'vector_database',
        }
        for scan_root in getattr(self, 'scan_roots', None) or []:
            root_path = Path(scan_root)
            if not root_path.is_dir():
                continue
            try:
                for bt_dir in root_path.rglob('*'):
                    if not bt_dir.is_dir():
                        continue
                    if bt_dir.name not in benchmark_type_names:
                        continue
                    # bt_dir is <...>/<benchmark_type>/. Its immediate
                    # children are per-model dirs (or engine dirs for
                    # vdb).
                    for model_dir in bt_dir.iterdir():
                        if not model_dir.is_dir():
                            continue
                        if bt_dir.name == 'vector_database':
                            # Two levels deeper: <engine>/<index_type>/
                            for index_dir in model_dir.iterdir():
                                if index_dir.is_dir():
                                    found.add(str(index_dir.resolve()))
                        elif bt_dir.name == 'training':
                            # training: per-model rollup lives at <model>/run/
                            # (Rules.md 2.1.16). Only add run/ if it exists.
                            run_dir = model_dir / 'run'
                            if run_dir.is_dir():
                                found.add(str(run_dir.resolve()))
                        else:
                            found.add(str(model_dir.resolve()))
            except (OSError, PermissionError) as e:
                self.logger.warning(
                    f"Could not enumerate on-disk model dirs under "
                    f"{scan_root}: {e}"
                )
                continue
        return sorted(found)

    def _model_group_folder(self, result: 'Result') -> Optional[str]:
        """Return the on-disk folder that groups runs at the model level.

        Uses the canonical layouts from generate_output_location():

        * training        — leaf is `.../<model>/run/<ts>/`
                          → walk up one level to land at `<model>/run/`.
                          Rules.md 2.1.16 (runResultsJson) mandates
                          results.json inside the "run" phase directory.
        * kv_cache        — leaf is `.../<model>/<command>/<ts>/`
                          → walk up two levels to land at `<model>/`.
        * checkpointing   — leaf is `.../<model>/<ts>/` (no <command>)
                          → walk up one level to land at `<model>/`.
        * vector_database — leaf is `.../<engine>/<index>/<command>/<ts>/`
                          → walk up two levels to land at `<engine>/<index>/`
                             (the "model-like" grouping key for vdb).

        Returns None (and logs a warning) if the run has no result_dir.
        """
        leaf = getattr(result.benchmark_run, 'result_dir', None)
        if not leaf:
            self.logger.warning(
                "Run %s has no result_dir on its BenchmarkRun; "
                "cannot place a model-level rollup for it. Skipping.",
                result.benchmark_run.run_id,
            )
            return None
        leaf_abs = os.path.abspath(leaf)
        bt = result.benchmark_type
        if bt in (BENCHMARK_TYPES.checkpointing, BENCHMARK_TYPES.training):
            # checkpointing: <ts>/ → <model>/
            # training:      <ts>/ → run/   (Rules.md 2.1.16 — results.json
            #                                lives inside the run/ phase dir)
            return os.path.dirname(leaf_abs)
        # kv_cache, vector_database: <ts>/ → <command>/ → group folder
        return os.path.dirname(os.path.dirname(leaf_abs))

    # Canonical result-tree depths: number of parent hops from the run's
    # <ts>/ leaf up to the <system>/ folder. Derived from
    # rules/utils.generate_output_location:
    #   training     : <system>/training/<model>/<command>/<ts>/       → 4
    #   kv_cache     : <system>/kv_cache/<model>/<command>/<ts>/       → 4
    #   checkpointing: <system>/checkpointing/<model>/<ts>/            → 3
    #   vector_db    : <system>/vector_database/<engine>/<index>/<command>/<ts>/ → 5
    _SYSTEM_SCOPE_LEVELS_UP = {
        BENCHMARK_TYPES.training: 4,
        BENCHMARK_TYPES.kv_cache: 4,
        BENCHMARK_TYPES.checkpointing: 3,
        BENCHMARK_TYPES.vector_database: 5,
    }

    def _system_scope_key(self, benchmark_run: 'BenchmarkRun') -> str:
        """Return the ``<system>/`` folder that owns this run.

        Used as the collision-detection namespace for ``run_results`` so
        that identical RunIDs originating from different systems are
        kept as distinct real runs (they only collide with warmups
        *within* one system's <model>/ subtree). Falls back to the
        absolute leaf path (which is inherently unique per system) if
        the layout cannot be resolved — never returns "" so it can
        always serve as a dict key.
        """
        leaf = getattr(benchmark_run, 'result_dir', None) or ''
        if not leaf:
            # Distinct sentinel per run so unknowns never share a scope.
            return f"<no-result-dir:{benchmark_run.run_id}>"
        leaf_abs = os.path.abspath(leaf)
        levels = self._SYSTEM_SCOPE_LEVELS_UP.get(
            benchmark_run.benchmark_type, 4
        )
        parent = leaf_abs
        for _ in range(levels):
            new_parent = os.path.dirname(parent)
            if new_parent == parent:  # hit filesystem root; stop climbing
                break
            parent = new_parent
        return parent

    def accumulate_results(self) -> None:
        """
        Accumulate and validate results from all benchmark runs.

        This method:
        1. Scans the results directory for benchmark runs
        2. Validates each run individually (with error isolation)
        3. Groups runs by workload for submission validation
        4. Runs multi-run verifiers on workload groups

        Errors in individual runs are logged but do not stop processing.
        """
        # Walk each effective scan root and accumulate runs. In canonical
        # mode, this naturally narrows to the requested system's subtree
        # (issue #599 bug 3); in flat mode, it's a single pass over the
        # original results_dir.
        benchmark_runs: List = []
        for scan_root in self.scan_roots:
            try:
                benchmark_runs.extend(
                    get_runs_files(scan_root, logger=self.logger)
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to scan results directory {scan_root}: {e}"
                )
                self.processing_errors.append(
                    f"Directory scan failed for {scan_root}: {e}"
                )

        if not benchmark_runs:
            scan_paths = ', '.join(self.scan_roots)
            self.logger.warning(
                f"No valid benchmark runs found in {scan_paths}. "
                "Ensure runs have completed and contain metadata files."
            )
            return

        self.logger.info(f'Accumulating results from {len(benchmark_runs)} runs')

        # Process individual runs with error isolation
        for benchmark_run in benchmark_runs:
            try:
                self._process_single_run(benchmark_run)
            except Exception as e:
                error_msg = f"Failed to process run {benchmark_run.run_id}: {e}"
                self.logger.error(f"{error_msg}. Skipping.")
                self.processing_errors.append(error_msg)
                continue

        # Group runs for workload-level validation
        self._process_workload_groups(benchmark_runs)

    def _process_single_run(self, benchmark_run: BenchmarkRun) -> None:
        """
        Process and validate a single benchmark run.

        Training workloads have 6 disk directories per (model, accelerator):
        1 throwaway warmup run + 5 submission runs. Only the 5 real runs
        should be aggregated into results.{csv,json}. Checkpointing has
        1 disk dir = 1 run (write-then-read self-warms), no warmup.

        Warmup detection is two-tier:

        Tier 1 (collision) — retained for backward compatibility with the
        retired ``--loops`` orchestrator. DLIO writes the warmup's
        ``summary.start`` value to match the FIRST real run's start time
        (not the warmup's own directory timestamp), so the two runs produce
        equal ``run_id`` values and collide in ``self.run_results``.
        Detection here: on collision, the run whose ``result_dir`` basename
        is lex-earlier is the warmup — its basename is recorded in
        ``self.warmup_result_dirs`` and the later run wins the dict slot
        (matching prior dict-overwrite semantics, which are preserved so
        aggregate counts are unchanged).

        Tier 2 (earliest-timestamp fallback per Rules.md §2.1.17) — applied
        in ``_process_workload_groups`` for 6-invocation training groups
        when Tier 1 finds nothing. The v3.0 flow prescribed by #502 uses
        6 independent ``run`` invocations, each with its own
        ``summary.start`` → no Tier-1 collision. Fallback picks the
        lex-earliest ``basename(result_dir)`` in the group as the warmup.

        The workload printer looks up ``warmup_result_dirs`` to render the
        warmup with a ``[WARMUP, not aggregated]`` label instead of a
        category badge.

        Args:
            benchmark_run: The benchmark run to process.

        Raises:
            Exception: If processing fails critically.
        """
        self.logger.ridiculous(f'Processing run: \n{pprint.pformat(benchmark_run)}')

        verifier = BenchmarkVerifier(benchmark_run, logger=self.logger)
        category = verifier.verify()
        issues = verifier.issues

        result = Result(
            multi=False,
            benchmark_run=benchmark_run,
            benchmark_type=benchmark_run.benchmark_type,
            benchmark_command=benchmark_run.command,
            benchmark_model=benchmark_run.model,
            issues=issues,
            category=category,
            metrics=benchmark_run.metrics or {}
        )

        # Collision-check scope: (<system>, run_id). Two runs with the
        # same RunID under different systems are distinct real runs and
        # must both be kept — the warmup collision only exists within a
        # single system's <model>/ subtree, where DLIO stamps the warmup's
        # summary.start to equal the first real run's start time.
        scope_key = self._system_scope_key(benchmark_run)
        result_key = (scope_key, benchmark_run.run_id)

        existing = self.run_results.get(result_key)
        if existing is not None:
            incoming_dir = os.path.abspath(benchmark_run.result_dir or "")
            existing_dir = os.path.abspath(existing.benchmark_run.result_dir or "")
            incoming_base = os.path.basename(incoming_dir)
            existing_base = os.path.basename(existing_dir)
            if incoming_base < existing_base:
                self.warmup_result_dirs.add(incoming_dir)
                # Keep the existing (later-basename, real) run in run_results.
                self.logger.debug(
                    f"Detected warmup run (collision on {benchmark_run.run_id} "
                    f"within system {scope_key!r}): {incoming_base} "
                    "(excluded from aggregate)"
                )
                return
            else:
                self.warmup_result_dirs.add(existing_dir)
                self.logger.debug(
                    f"Detected warmup run (collision on {benchmark_run.run_id} "
                    f"within system {scope_key!r}): {existing_base} "
                    "(excluded from aggregate)"
                )
                # Fall through to overwrite the existing (warmup) entry.

        self.run_results[result_key] = result

        # Log category for the run
        self.logger.debug(
            f"Run {benchmark_run.run_id} validated as {category.value.upper()}"
        )

    # ------------------------------------------------------------------
    # Per-workload aggregation helpers (D-19..D-22)
    # ------------------------------------------------------------------
    #
    # ``_aggregate_workload_metrics`` is the single point where per-workload
    # aggregation math lives. It is wired into the reporting pass at the
    # ``aggregated_metrics = self._aggregate_workload_metrics(...)`` call in
    # ``accumulate_results`` (the ``metrics={}`` in ``_model_group_folder``'s
    # proxy Result is a throwaway for folder derivation, not this call site).
    #
    # Dispatch on ``runs[0].benchmark_type``:
    #   - training       -> ``_aggregate_training``  (D-19, 5-run mean)
    #   - checkpointing  -> ``_aggregate_checkpointing``  (D-20/D-28)
    #   - vector_database-> ``_aggregate_vdb``  (D-21, pass-through)
    #   - kv_cache       -> ``_aggregate_kvcache``  (D-22/D-16, pass-through)
    #
    # Empty-metric handling: raises ``statistics.StatisticsError`` — the
    # helper deliberately does NOT swallow it (D-23). The caller in 06-04
    # is responsible for the ``except StatisticsError`` split from the
    # broad ``except Exception:`` and for downgrading ``category`` to
    # ``PARAM_VALIDATION.INVALID`` with the D-24 verbatim message.

    def _aggregate_workload_metrics(
        self,
        runs: List[BenchmarkRun],
        warmup_set: Set[str],
    ) -> Dict[str, Any]:
        """
        Compute aggregated metrics for one workload (per D-19..D-22).

        Dispatches on ``runs[0].benchmark_type`` and returns the aggregated
        metric dict that populates the ``metrics=`` slot on the workload
        ``Result`` (wired in ``accumulate_results``).

        Args:
            runs: The workload's ``BenchmarkRun`` invocations. For training,
                a 6-invocation set (1 warmup + 5 real) after
                ``_process_single_run`` collision detection. For
                checkpointing, 1–2 invocations per Rules.md §2.1.23. For
                vdb / kvcache, a single-invocation list.
            warmup_set: Set of ABSOLUTE paths (as populated in
                ``self.warmup_result_dirs``). Used ONLY by the training
                branch to filter warmup runs out of the 5-run mean
                (D-19). The checkpointing / vdb / kvcache branches ignore
                this argument (D-20/D-28: no warmup for those types).

        Returns:
            A dict of aggregated metric names to values. Key convention:
              - training      : ``train_mean_of_<basename>`` (D-13)
              - checkpointing : ``checkpoint_mean_of_<basename>`` (D-13)
              - vdb           : ``vdb_<source-name>`` pass-through (D-15)
              - kvcache       : ``kvcache_aggregated_<...>`` +
                ``kvcache_option_<opt>_aggregated_<...>`` (D-15..D-17)

        Raises:
            statistics.StatisticsError: When a metric list is empty (training
                or checkpointing branch). The caller in 06-04 catches this
                and downgrades the workload's ``category`` to
                ``PARAM_VALIDATION.INVALID`` per D-23. Do NOT swallow this
                here — the split from the broad ``except Exception:`` is
                the loud-failure contract (D-23; PITFALLS #3).
        """
        if not runs:
            return {}
        bt = runs[0].benchmark_type
        if bt == BENCHMARK_TYPES.training:
            return self._aggregate_training(runs, warmup_set)
        elif bt == BENCHMARK_TYPES.checkpointing:
            return self._aggregate_checkpointing(runs)
        elif bt == BENCHMARK_TYPES.vector_database:
            return self._aggregate_vdb(runs)
        elif bt == BENCHMARK_TYPES.kv_cache:
            return self._aggregate_kvcache(runs)
        else:
            return {}

    def _aggregate_training(
        self,
        runs: List[BenchmarkRun],
        warmup_set: Set[str],
    ) -> Dict[str, float]:
        """
        Training-branch aggregation (D-19, Rules.md §2.1.17).

        Filters ``runs`` down to the non-warmup invocations
        (``abspath(run.result_dir) not in warmup_set``), then for each
        metric key present in EVERY non-warmup invocation, computes:

            per_run = [fmean(inv.metrics[key]) for inv in non_warmup]
            outer   = fmean(per_run)

        emitting under ``train_mean_of_<basename>`` where basename is the
        source key with any redundant ``train_`` prefix stripped (D-13).

        Empty inner metric list -> ``StatisticsError`` propagates (D-23).
        Missing key in a given invocation is skipped via key intersection
        (a partial-coverage key is not aggregated — the caller sees it
        absent from the output).

        Count strictness (D-27) and warmup detection (D-26) are enforced
        at the CALLER site in 06-04, not here — the helper focuses on
        math. If ``non_warmup`` is empty after filtering, this raises
        ``StatisticsError`` with a helpful ``args[0]`` so the caller can
        surface INVALID.
        """
        non_warmup = [
            r for r in runs
            if os.path.abspath(r.result_dir or "") not in warmup_set
        ]
        if not non_warmup:
            raise StatisticsError(
                "no non-warmup invocations to aggregate for training workload"
            )

        # Intersection of metric keys across all non-warmup invocations.
        # A key present in some invocations but not all is skipped — the
        # caller sees it missing from the output and can decide.
        metric_dicts = [(inv.metrics or {}) for inv in non_warmup]
        common_keys = set(metric_dicts[0].keys())
        for md in metric_dicts[1:]:
            common_keys &= set(md.keys())

        out: Dict[str, float] = {}
        for key in sorted(common_keys):
            per_run_means: List[float] = []
            for inv in non_warmup:
                metric_list = (inv.metrics or {}).get(key)
                if not metric_list:
                    # Loud failure per D-23: empty metric list is NOT
                    # silently coerced to 0.0. Caller wraps this in a
                    # try/except StatisticsError and emits the D-24
                    # ``_INVALID_MSG_EMPTY_METRIC`` template.
                    raise StatisticsError(
                        f"metric {key} is empty in invocation "
                        f"{os.path.basename(inv.result_dir or '')}"
                    )
                per_run_means.append(fmean(metric_list))
            outer = fmean(per_run_means)
            # D-13: strip redundant ``train_`` prefix from the source key
            # so the emitted column looks like ``train_mean_of_<basename>``.
            basename = key
            if basename.startswith("train_"):
                basename = basename[len("train_"):]
            out[f"train_mean_of_{basename}"] = outer

        # Slice-Training: v3.0 final-table columns the list-valued metric
        # aggregation above does not carry. num_hosts / num_accelerators
        # live at summary.json top level; Read B/W is DLIO's scalar
        # ``train_io_mean_MB_per_second`` (MiB/s despite the ``MB`` label —
        # DLIO's statscounter.py computes ``samples/s * record_size / 1024
        # / 1024`` and logs it as "MiB/second"). All three are dropped by
        # the metadata-complete parse path (system_info is never
        # reconstructed; scalar metrics are filtered out), so read them
        # straight from each run's summary.json — consistent with the
        # vdb / kvcache branches.
        summaries = [self._load_workload_summary(inv) for inv in non_warmup]

        # Identity columns — run configuration, identical across the
        # measured invocations; take the first invocation that reports each.
        def _first_present(field: str) -> Any:
            for s in summaries:
                value = s.get(field)
                if value is not None:
                    return value
            return None

        out["train_num_client_nodes"] = _first_present("num_hosts")
        out["train_num_simulated_accelerators"] = _first_present(
            "num_accelerators"
        )

        # Read B/W (GiB/s): mean of DLIO's per-run MiB/s scalar across all
        # non-warmup invocations, MiB -> GiB (binary, /1024). Present-but-
        # blank when any invocation omits the scalar (blank only when
        # truly absent — no silent partial-mean).
        io_mibps = [
            (s.get("metric") or {}).get("train_io_mean_MB_per_second")
            for s in summaries
        ]
        if io_mibps and all(isinstance(v, (int, float)) for v in io_mibps):
            out["train_read_bw_gibps"] = fmean(io_mibps) / 1024.0
        else:
            out["train_read_bw_gibps"] = None
        return out

    def _aggregate_checkpointing(
        self,
        runs: List[BenchmarkRun],
    ) -> Dict[str, float]:
        """
        Checkpointing-branch aggregation (D-20/D-28, Rules.md §2.1.23).

        ``warmup_set`` is not accepted here — checkpointing has no warmup
        per Rules.md §2.1.23; the dispatcher in
        ``_aggregate_workload_metrics`` ignores ``warmup_set`` for this
        branch and calls this method without it.

        For each metric key present in every invocation's ``metric`` dict,
        computes ``fmean(run.metrics[key])`` intra-list over the 10-op
        list per invocation. If ``len(runs) > 1`` (rare per Rules.md
        §2.1.23 which permits 1–2 timestamp directories), takes the
        inter-invocation ``fmean`` for shape consistency with the
        training branch. Op-count strictness (D-24 template c) is a
        caller-side gate in 06-04, not here.

        Empty metric list -> ``StatisticsError`` propagates (D-23). Do
        NOT copy the "fmean(x) if x, coerce to zero otherwise" idiom
        from ``benchmarks/kvcache.py:793`` — that is the PITFALLS #3
        anti-pattern and the loud-failure principle forbids it here.
        """
        if not runs:
            raise StatisticsError(
                "no checkpointing invocations to aggregate"
            )

        metric_dicts = [(r.metrics or {}) for r in runs]
        common_keys = set(metric_dicts[0].keys())
        for md in metric_dicts[1:]:
            common_keys &= set(md.keys())

        out: Dict[str, float] = {}
        for key in sorted(common_keys):
            per_run_means: List[float] = []
            for r in runs:
                metric_list = (r.metrics or {}).get(key)
                if not metric_list:
                    raise StatisticsError(
                        f"metric {key} is empty in invocation "
                        f"{os.path.basename(r.result_dir or '')}"
                    )
                per_run_means.append(fmean(metric_list))
            outer = fmean(per_run_means) if len(per_run_means) > 1 else per_run_means[0]
            basename = key
            if basename.startswith("checkpoint_"):
                basename = basename[len("checkpoint_"):]
            out[f"checkpoint_mean_of_{basename}"] = outer

        # Slice-Checkpointing: v3.0 final-table columns. Write/Read B/W and
        # durations come from DLIO's per-run SCALAR keys
        # (save_/load_checkpoint_io_mean_GB_per_second, _duration_mean_
        # seconds) — already GiB/s BINARY (checkpoint_size is in GiB and
        # throughput = size / seconds; DLIO logs it "GiB/second"), so NO
        # conversion. These scalars are dropped by the metadata-complete
        # parse path (list-only metric filter) and are absent from the
        # fabricated fixtures, so read them straight from each run's
        # summary.json — consistent with the training / vdb / kvcache
        # branches. num_hosts is summary top-level; Checkpoint Mode is a
        # param; DP Instances is a per-model constant.
        summaries = [self._load_workload_summary(r) for r in runs]

        def _first_present(field: str) -> Any:
            for s in summaries:
                value = s.get(field)
                if value is not None:
                    return value
            return None

        out["checkpoint_num_client_nodes"] = _first_present("num_hosts")

        # Checkpoint Mode (Full / Subset). The benchmark writes
        # ``checkpoint.mode = "subset"`` only for subset runs
        # (benchmarks/dlio.py:add_checkpoint_params); a "default" / absent
        # mode is a full-model run. Handle both the nested params dict and
        # a flattened ``checkpoint.mode`` key defensively.
        params = runs[0].parameters or {}
        mode_raw = None
        ckpt_params = params.get("checkpoint")
        if isinstance(ckpt_params, dict):
            mode_raw = ckpt_params.get("mode")
        if mode_raw is None:
            mode_raw = params.get("checkpoint.mode")
        out["checkpoint_mode"] = "Subset" if mode_raw == "subset" else "Full"

        # DP Instances — configured data parallelism int(ClosedGPUs /
        # GPUpDP) from LLM_ALLOWED_VALUES (the same value
        # benchmarks/dlio.py:add_checkpoint_params computes). Per-model
        # constant; blank when the model is unknown.
        allowed = LLM_ALLOWED_VALUES.get(runs[0].model)
        if allowed:
            _min_procs, _zero_level, gpu_per_dp, closed_gpus = allowed
            out["checkpoint_dp_instances"] = (
                int(closed_gpus / gpu_per_dp) if gpu_per_dp else None
            )
        else:
            out["checkpoint_dp_instances"] = None

        # B/W (verbatim GiB/s) + durations — mean of DLIO's per-run scalar,
        # taken PER DIRECTION over the invocations that were configured to
        # produce it (Rules.md §2.1.23 permits 1-2 timestamp dirs; §4.7.1
        # MANDATES a write→read split when checkpoint-per-node < 3x client
        # RAM). In a split, the write dir carries only save_* and the read
        # dir only load_*, so a field's None in the OPPOSITE phase is
        # expected — not data loss — and must not blank the column.
        #
        # Each invocation self-declares its phase via
        # ``checkpoint.num_checkpoints_{write,read}`` (CLOSED forces each to
        # 10 or 0, never both 0 — cli/checkpointing_args.py). We keep the
        # loud-failure guard (D-23 / PITFALLS #3: no silent partial-mean)
        # but scope it to the producing phase: a save_* missing from an
        # invocation that WAS configured to write is real data loss and
        # still blanks. When the phase signal is absent (legacy / OPEN
        # packages that don't persist the counts) we cannot classify, so we
        # fall back to a present-only mean rather than regress those rows.
        def _phase_count(run: BenchmarkRun, which: str) -> Any:
            params = run.parameters or {}
            key = f"num_checkpoints_{which}"
            ckpt = params.get("checkpoint")
            if isinstance(ckpt, dict) and key in ckpt:
                return ckpt.get(key)
            return params.get(f"checkpoint.{key}")

        def _is_number(v: Any) -> bool:
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        def _directional_mean(field: str, which: str) -> Optional[float]:
            # Classify each invocation as a producer of `which` direction
            # from its configured count. If ANY invocation lacks the signal
            # the set is unclassifiable -> present-only fallback.
            counts = [_phase_count(r, which) for r in runs]
            if all(_is_number(c) for c in counts):
                producers = [
                    s for s, c in zip(summaries, counts) if c > 0
                ]
                if not producers:
                    return None
                values = [(s.get("metric") or {}).get(field) for s in producers]
                # Strict within the producing phase: a missing scalar here
                # is data loss -> blank (loud), never a partial mean.
                if values and all(_is_number(v) for v in values):
                    return fmean(values)
                return None
            # Unclassifiable: mean over whatever's present, blank if none.
            values = [(s.get("metric") or {}).get(field) for s in summaries]
            present = [v for v in values if _is_number(v)]
            return fmean(present) if present else None

        out["checkpoint_write_bw_gibps"] = _directional_mean(
            "save_checkpoint_io_mean_GB_per_second", "write"
        )
        out["checkpoint_write_duration_secs"] = _directional_mean(
            "save_checkpoint_duration_mean_seconds", "write"
        )
        out["checkpoint_read_bw_gibps"] = _directional_mean(
            "load_checkpoint_io_mean_GB_per_second", "read"
        )
        out["checkpoint_read_duration_secs"] = _directional_mean(
            "load_checkpoint_duration_mean_seconds", "read"
        )
        return out

    def _aggregate_vdb(
        self,
        runs: List[BenchmarkRun],
    ) -> Dict[str, Any]:
        """
        VDB-branch aggregation (D-21) — pass-through, NOT math.

        vdb's internal ``vdb-aggregate`` tool (see
        ``VectorDBBenchmark._run_aggregate`` in ``benchmarks/vectordbbench.py``)
        owns the math contract per the D-22 boundary; this helper copies
        pre-computed values from the workload's ``summary.json`` into
        ``vdb_*`` columns.

        Read fields (per ``submission_checker/checks/vdb_checks.py:44-51``
        ``_REQUIRED_METRIC_FIELDS``): ``throughput_qps``,
        ``mean_latency_ms``, ``p95_latency_ms``, ``p99_latency_ms``,
        ``p999_latency_ms``. Emit as ``vdb_<source-name>`` (D-15). Missing
        fields emit ``None`` (D-22 boundary: vdb owns validity — INVALID
        rules-strict is training + checkpointing only).

        Recall fallback: if ``recall`` is absent from ``summary.json``,
        fall back to ``<workload_dir>/recall_stats.json`` per the pattern
        at ``submission_checker/checks/vdb_checks.py:462-469``.

        Identity columns (D-15): ``vdb_engine`` and ``vdb_index_type``
        read from the ``run`` invocation's ``parameters`` (the
        ``BenchmarkRun`` accessor for CLI/YAML args on disk), falling
        back to the other grouped leaves.

        Run-leaf selection: the VDB workload key (D-06) is
        ``(category, orgname, systemname, engine, index_type)`` — it does
        NOT include ``command``, so a workload's ``datasize`` /
        ``datagen`` / ``run`` leaves all group under one key and arrive
        here as a single ``runs`` list in discovery order. Only the
        ``run`` leaf carries the native query-phase metrics
        (``statistics.json`` / ``summary.json``); the datasize/datagen
        leaves have none. Reading ``runs[0]`` blindly left QPS / latency
        / recall blank whenever a non-``run`` leaf sorted first, so
        select the ``run`` invocation for metrics and recall. Fall back
        to ``runs[0]`` only when no ``run`` leaf is present (e.g. a
        datasize-only tree) so identity columns still populate.
        """
        # Metrics + recall come from the ``run`` leaf (see docstring).
        run = next((r for r in runs if r.command == 'run'), runs[0])
        summary = self._load_vdb_summary(run)

        _VDB_METRIC_FIELDS = (
            "throughput_qps",
            "mean_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "p999_latency_ms",
        )
        out: Dict[str, Any] = {}
        for field_name in _VDB_METRIC_FIELDS:
            out[f"vdb_{field_name}"] = summary.get(field_name)

        # Recall (D-21) — first summary.json, then recall_stats.json fallback
        # (per submission_checker/checks/vdb_checks.py:462-469). A dict recall
        # block is reduced to its scalar ``mean_recall``; the value is emitted
        # as a PERCENTAGE (0-100) to match the "Recall Percentage" column.
        recall = summary.get("recall")
        if isinstance(recall, dict):
            recall = recall.get("mean_recall")
        if recall is None and run.result_dir:
            recall_stats_path = os.path.join(run.result_dir, "recall_stats.json")
            if os.path.isfile(recall_stats_path):
                try:
                    with open(recall_stats_path, "r") as f:
                        recall_stats = json.load(f)
                    recall = recall_stats.get("recall")
                    if isinstance(recall, dict):
                        recall = recall.get("mean_recall")
                except (OSError, ValueError) as e:
                    self.logger.warning(
                        f"vdb: could not read recall_stats.json at "
                        f"{recall_stats_path}: {e}"
                    )
        out["vdb_recall"] = (recall * 100) if isinstance(recall, (int, float)) else None

        # Read B/W + # Client Nodes derive from the ``disk_io`` sub-block.
        # Read B/W is converted bytes/sec -> GiB/s (binary, /1024**3).
        disk_io = summary.get("disk_io")
        read_bps = disk_io.get("total_bytes_read_per_sec") if isinstance(disk_io, dict) else None
        out["vdb_read_bw_gibps"] = (
            read_bps / (1024 ** 3) if isinstance(read_bps, (int, float)) else None
        )
        out["vdb_num_client_nodes"] = (
            disk_io.get("host_count") if isinstance(disk_io, dict) else None
        )
        # Storage IOPs is not measured (disk_io carries bytes, not op counts) —
        # emit the column present-but-blank so the table has it (submitter or a
        # future instrumentation task fills it).
        out["vdb_storage_iops"] = None

        # Identity columns (D-15). Prefer the structured ``parameters`` block
        # (the combined_params fix); fall back to the persisted per-run
        # ``metadata['args']`` snapshot for legacy packages whose parameters
        # block is empty. Resolve across ALL grouped leaves (the ``run`` leaf
        # first, then the datasize/datagen leaves): an identity arg such as
        # ``num_vectors`` / ``dimension`` may be recorded only on the
        # datasize leaf while the query metrics live on the run leaf. Blank
        # only when truly absent from every leaf.
        identity_runs = [run] + [r for r in runs if r is not run]

        def _identity(param_key: str, arg_keys: tuple) -> Any:
            for candidate_run in identity_runs:
                value = self._vdb_identity(candidate_run, param_key, arg_keys)
                if value is not None:
                    return value
            return None

        out["vdb_num_vectors"] = _identity("num_vectors", ("num_vectors",))
        out["vdb_dimension"] = _identity("dimension", ("dimension",))
        out["vdb_engine"] = _identity("engine", ("vdb_engine", "engine"))
        out["vdb_index_type"] = _identity("index_type", ("index_type", "vdb_index"))
        return out

    @staticmethod
    def _vdb_identity(run: BenchmarkRun, param_key: str, arg_keys: tuple) -> Any:
        """Resolve a VDB identity value: parameters first, then persisted args.

        ``param_key`` is looked up in ``run.parameters`` (the combined_params
        block). If absent/None, ``arg_keys`` are tried in order against
        ``run.run_args`` (the persisted ``metadata['args']`` snapshot).
        Returns ``None`` when the value is in neither.
        """
        params = run.parameters or {}
        value = params.get(param_key)
        if value is not None:
            return value
        args = getattr(run, "run_args", None) or {}
        for ak in arg_keys:
            candidate = args.get(ak)
            if candidate is not None:
                return candidate
        return None

    def _load_vdb_summary(self, run: BenchmarkRun) -> Dict[str, Any]:
        """Load a VDB run's ``summary.json``, backfilling legacy packages.

        Fresh submissions ship ``summary.json`` (written by the producer).
        Legacy packages predate that write — they carry only the native
        ``statistics.json``. For those, derive the summary IN MEMORY via
        ``vdb_summary.build_vdb_summary``; NEVER write into the submission
        package (it may be read-only or owned by another submitter).
        """
        summary = self._load_workload_summary(run)
        if summary:
            return summary
        if run.result_dir:
            try:
                from mlpstorage_py.benchmarks.vdb_summary import build_vdb_summary

                built = build_vdb_summary(run.result_dir)
            except Exception as e:  # noqa: BLE001 — backfill is best-effort
                self.logger.warning(
                    f"vdb: could not derive summary from native stats at "
                    f"{run.result_dir!r}: {e}"
                )
                built = None
            if built:
                self.logger.verbose(
                    f"vdb: derived summary in-memory from native stats at "
                    f"{run.result_dir!r} (legacy package, no summary.json)."
                )
                return built
        return {}

    def _aggregate_kvcache(
        self,
        runs: List[BenchmarkRun],
    ) -> Dict[str, Any]:
        """
        KVCache-branch aggregation (D-22/D-16/D-17) — pass-through, NOT math.

        kvcache's internal ``_aggregate_option_results`` at
        ``benchmarks/kvcache.py:791-803`` owns the math contract per the
        D-22 boundary. Trust the source verbatim — including any ``0.0``
        sentinel values from the "fmean-when-nonempty, coerce-to-zero
        otherwise" idiom at ``benchmarks/kvcache.py:793``. That
        silent-failure amplifier
        belongs in kvcache, not in reportgen (PITFALLS #3).

        Emits:
          - Top-level identity (D-15): ``kvcache_performance_profile``
            from ``runs[0].parameters['performance_profile']``.
          - Top-level per-run aggregates (D-14) copied verbatim from
            ``summary.json`` — ``kvcache_aggregated_read_bandwidth_gbps``,
            ``kvcache_aggregated_write_bandwidth_gbps``,
            ``kvcache_aggregated_avg_throughput_tokens_per_sec``,
            ``kvcache_aggregated_storage_throughput_tokens_per_sec``,
            ``kvcache_aggregated_p95_latency_ms``.
          - Per-option flattening (D-16): for each
            ``(option_name, option_dict)`` in ``summary['options']``,
            emits ``kvcache_option_<option_name>_<metric_name>`` for each
            ``(metric_name, value)`` pair. Since kvcache's internal keys
            are ``aggregated_*`` (per ``benchmarks/kvcache.py:791-803``),
            the resulting output columns look like
            ``kvcache_option_<opt>_aggregated_read_bandwidth_gbps`` —
            preserving the grep-chain from source to output (D-17).
        """
        run = runs[0]
        summary = self._load_workload_summary(run)

        out: Dict[str, Any] = {}
        # Identity column (D-15) — read from BenchmarkRun.parameters.
        params = run.parameters or {}
        out["kvcache_performance_profile"] = params.get("performance_profile")

        # Slice-KVCache: shared ``# Client Nodes`` column for the v3.0
        # final table. Source is the top-level ``host_count`` field that
        # kvcache's ``_write_run_summary`` persists (benchmarks/kvcache.py).
        # This is a run-wide count, not per-option, so it is emitted once.
        out["kvcache_num_client_nodes"] = summary.get("host_count")

        # Top-level per-run aggregates (D-14). Copy verbatim from source;
        # Zero sentinels from kvcache's "fmean-or-zero" idiom are
        # preserved (D-22 boundary — do NOT re-interpret).
        _KVCACHE_TOPLEVEL_FIELDS = (
            "aggregated_read_bandwidth_gbps",
            "aggregated_write_bandwidth_gbps",
            "aggregated_avg_throughput_tokens_per_sec",
            "aggregated_storage_throughput_tokens_per_sec",
            "aggregated_p95_latency_ms",
        )
        for field_name in _KVCACHE_TOPLEVEL_FIELDS:
            out[f"kvcache_{field_name}"] = summary.get(field_name)

        # Per-option flattening (D-16).
        options = summary.get("options") or {}
        if not options:
            self.logger.warning(
                f"kvcache: summary.json at {run.result_dir!r} has no "
                "'options' dict — per-option columns will be absent."
            )
        else:
            for option_name, option_dict in options.items():
                if not isinstance(option_dict, dict):
                    continue
                for metric_name, value in option_dict.items():
                    # D-17: keep the source ``aggregated_`` prefix
                    # verbatim to preserve the grep-chain from source
                    # summary.json field to output column name.
                    out[f"kvcache_option_{option_name}_{metric_name}"] = value
        return out

    def _load_workload_summary(self, run: BenchmarkRun) -> Dict[str, Any]:
        """
        Load the workload's ``summary.json`` for vdb / kvcache pass-through
        branches.

        ``BenchmarkRun`` does not expose a ``summary_data`` accessor on
        this branch of the code base, so the helper reads
        ``<run.result_dir>/summary.json`` directly. Missing file or
        malformed JSON logs a warning and returns ``{}`` — the caller
        emits empty pass-through columns rather than aborting the whole
        reportgen pass (threat T-06-06 mitigation).
        """
        if not run.result_dir:
            return {}
        summary_path = os.path.join(run.result_dir, "summary.json")
        if not os.path.isfile(summary_path):
            # datagen/datasize invocations can never have a summary.json
            # (DLIO doesn't write one for datagen; datasize is a pure
            # calculation) — absence is only an artifact gap for run phases.
            if getattr(run, 'command', None) in ('datagen', 'datasize'):
                self.logger.debug(
                    f"summary.json not applicable for {run.command} "
                    f"invocation at {summary_path}; pass-through columns "
                    "will be empty."
                )
            else:
                self.logger.warning(
                    f"summary.json not found at {summary_path}; "
                    "pass-through columns will be empty."
                )
            return {}
        try:
            with open(summary_path, "r") as f:
                return json.load(f)
        except (OSError, ValueError) as e:
            self.logger.warning(
                f"summary.json at {summary_path} could not be read: {e}; "
                "pass-through columns will be empty."
            )
            return {}

    # ------------------------------------------------------------------
    # Workload identity derivation (D-05/D-07)
    # ------------------------------------------------------------------
    _CATEGORY_PATH_TOKENS = ("closed", "open", "whatif")

    def _derive_category_from_path(self, benchmark_run: BenchmarkRun) -> Optional[str]:
        """D-07 category derivation from the workload's on-disk path.

        Returns one of ``'closed'``, ``'open'``, ``'whatif'`` when the
        workload's absolute result_dir contains that segment. Returns
        ``None`` when no segment matches — the caller then falls back
        to the upstream :class:`BenchmarkVerifier` category value.
        """
        leaf = getattr(benchmark_run, 'result_dir', None) or ""
        if not leaf:
            return None
        # Normalize to path segments; scan case-insensitively but return
        # the lowercased canonical token so downstream string comparisons
        # (D-29 ``category != 'whatif'``) are unambiguous.
        parts = [p.lower() for p in os.path.abspath(leaf).split(os.sep) if p]
        for token in self._CATEGORY_PATH_TOKENS:
            if token in parts:
                return token
        return None

    def _derive_orgname_from_path(self, benchmark_run: BenchmarkRun) -> str:
        """D-07 orgname derivation.

        Prefers ``benchmark_run.parameters['orgname']`` (when set at run
        time); otherwise walks the workload's absolute result_dir looking
        for the ``<division>/<orgname>/results/`` triplet that the
        canonical submission tree produces (see
        ``_resolve_effective_results_dir``). Returns an empty string when
        the path shape does not match — never raises.
        """
        params = getattr(benchmark_run, 'parameters', None)
        if isinstance(params, dict):
            candidate = params.get('orgname')
            if isinstance(candidate, str) and candidate:
                return candidate
        leaf = getattr(benchmark_run, 'result_dir', None) or ""
        if not leaf:
            return ""
        parts = os.path.abspath(leaf).split(os.sep)
        # Look for `<division>/<orgname>/results/` where division is
        # `closed`, `open`, or `whatif`.
        for idx in range(len(parts) - 2):
            if (
                parts[idx].lower() in self._CATEGORY_PATH_TOKENS
                and parts[idx + 2] == "results"
            ):
                return parts[idx + 1]
        return ""

    def _derive_systemname_from_path(self, benchmark_run: BenchmarkRun) -> str:
        """D-07 systemname derivation.

        When ``self.args.systemname`` was supplied on the CLI, uses it.
        Otherwise walks the workload's absolute path looking for
        ``results/<systemname>/`` — the canonical submission tree shape.
        Returns an empty string when the path shape does not match.
        """
        systemname = None
        if self.args is not None:
            systemname = getattr(self.args, 'systemname', None)
        if systemname:
            return str(systemname)
        leaf = getattr(benchmark_run, 'result_dir', None) or ""
        if not leaf:
            return ""
        parts = os.path.abspath(leaf).split(os.sep)
        for idx in range(len(parts) - 1):
            if parts[idx] == "results":
                return parts[idx + 1]
        return ""

    def _derive_workload_key(
        self,
        benchmark_run: BenchmarkRun,
        fallback_category: Optional[str] = None,
    ) -> tuple:
        """Return the per-benchmark-type grouping tuple (D-05/D-06).

        - training / checkpointing: ``(category, orgname, systemname, model, accelerator)``
        - vector_database:          ``(category, orgname, systemname, engine, index_type)``
        - kv_cache:                 ``(category, orgname, systemname, model, performance_profile)``

        ``category`` is derived from the workload's on-disk path segment
        when possible (D-07); when the path does not match, falls back to
        the caller-supplied ``fallback_category`` (typically the upstream
        :class:`BenchmarkVerifier` category converted to string).
        """
        category = self._derive_category_from_path(benchmark_run)
        if category is None:
            category = fallback_category or ""
        orgname = self._derive_orgname_from_path(benchmark_run)
        systemname = self._derive_systemname_from_path(benchmark_run)
        bt = benchmark_run.benchmark_type
        raw_params = getattr(benchmark_run, 'parameters', None)
        params = raw_params if isinstance(raw_params, dict) else {}

        def _p(key: str) -> str:
            """Return params[key] as a str only when it's a real string."""
            v = params.get(key)
            return v if isinstance(v, str) else ""

        if bt == BENCHMARK_TYPES.vector_database:
            return (category, orgname, systemname, _p('engine'), _p('index_type'))
        if bt == BENCHMARK_TYPES.kv_cache:
            model = str(benchmark_run.model or "")
            return (category, orgname, systemname, model, _p('performance_profile'))
        # training / checkpointing (and any other type) share the
        # (category, orgname, systemname, model, accelerator) shape.
        model = str(benchmark_run.model or "")
        accelerator = str(benchmark_run.accelerator or "")
        # Issue #771: auxiliary training commands (``datasize``, ``datagen``)
        # run once per (model, accelerator) and are NOT part of the
        # 1-warmup + 5-real invocation set that Rules.md §2.1.17 governs.
        # ``datasize`` in particular carries a real accelerator (e.g.
        # ``b200``) because it needs one to compute the required data
        # size — so under the plain 5-tuple key it collides with the 6
        # ``run`` invocations, inflates the count to 7, and trips the
        # D-27 count gate. ``datagen`` avoids the collision only by the
        # accident of carrying an empty accelerator. Adding ``command``
        # as an extra discriminator for non-``run`` training commands
        # makes the isolation explicit and independent of that accident,
        # while preserving accelerator-scoped grouping for multi-
        # accelerator submissions (b200 + mi355 datasize runs stay in
        # separate groups per accelerator).
        #
        # Issue #791: checkpointing ``datasize`` has the same collision
        # shape — it carries the same (model, accelerator) as the
        # ``run`` invocations, so under the plain 5-tuple key it lands
        # in the same workload group and inflates the invocation count
        # past the 1-or-2 Rules.md §4.7.1 bound. Extend the
        # discriminator to cover checkpointing for the same reason.
        if (
            bt in (BENCHMARK_TYPES.training, BENCHMARK_TYPES.checkpointing)
            and benchmark_run.command != 'run'
        ):
            return (
                category, orgname, systemname, model, accelerator,
                str(benchmark_run.command),
            )
        return (category, orgname, systemname, model, accelerator)

    def _process_workload_groups(self, benchmark_runs: List[BenchmarkRun]) -> None:
        """
        Group runs by workload and run submission-level validation.

        Grouping key is per-benchmark-type per D-05/D-06:

        - training / checkpointing: ``(category, orgname, systemname, model, accelerator)``
        - vector_database:          ``(category, orgname, systemname, engine, index_type)``
        - kv_cache:                 ``(category, orgname, systemname, model, performance_profile)``

        The aggregation dispatch (``_aggregate_workload_metrics``) sits
        BELOW rules-strict INVALID gates (D-23/D-26/D-27) so that empty
        metric lists never silently produce ``0.0`` and count/warmup
        violations produce the D-24 verbatim message rather than a
        computed-but-wrong value. ``whatif`` rows skip the rules-strict
        gates entirely per D-29 (whatif is simulation, not submission).
        """
        # Group by per-benchmark-type key (D-05). Category comes from
        # path derivation when possible; the upstream verifier still runs
        # for each group to produce issues + a fallback category.
        workload_runs: Dict[tuple, List[BenchmarkRun]] = {}
        for benchmark_run in benchmark_runs:
            workload_key = self._derive_workload_key(benchmark_run)
            workload_runs.setdefault(workload_key, []).append(benchmark_run)

        for workload_key, runs in workload_runs.items():
            if not runs:
                continue

            try:
                # workload_key = (category, orgname, systemname, id1, id2)
                # or, for auxiliary training commands (Issue #771),
                # (category, orgname, systemname, id1, id2, command).
                # Only the fixed 5-prefix is unpacked here — the optional
                # 6th element is a discriminator, not something the log
                # line needs to render.
                category_str, orgname_str, systemname_str, ident1, ident2 = workload_key[:5]
                self.logger.info(
                    f'Running submission verifiers for '
                    f'{runs[0].benchmark_type.value if runs[0].benchmark_type else "?"} '
                    f'({ident1}, {ident2}) — {len(runs)} runs'
                )
                verifier = BenchmarkVerifier(*runs, logger=self.logger)
                verifier_category = verifier.verify()
                issues = list(verifier.issues) if verifier.issues else []

                # If path-based category derivation returned "", fall back
                # to the upstream verifier's category value.
                if not category_str and verifier_category is not None:
                    try:
                        category_str = verifier_category.value
                    except AttributeError:
                        category_str = str(verifier_category)

                # ----------------------------------------------------------
                # Rules-strict INVALID gates (D-23/D-26/D-27)
                # ----------------------------------------------------------
                # whatif rows SKIP these gates entirely per D-29 — whatif
                # is a simulation, not a submission; INVALID semantics
                # don't apply.
                invalid_messages: List[str] = []
                # Issue #717: the D-26/D-27 (training) and D-20/D-24
                # (checkpointing) rules-strict gates encode Rules.md
                # invariants that apply to the ``run`` command only —
                # ``datagen`` legitimately produces a single invocation
                # and carries no metric lists. The workload grouping
                # key does not include ``command`` (D-05), and datagen
                # runs typically land in their own group anyway because
                # they carry ``accelerator=None`` while ``run`` groups
                # carry e.g. ``h100``. Skip these gates for any group
                # whose command is not ``run`` — otherwise a datagen
                # group gets a spurious "found 1" INVALID.
                if category_str != 'whatif' and runs[0].command == 'run':
                    bt = runs[0].benchmark_type
                    if bt == BENCHMARK_TYPES.training:
                        # D-27: training must be exactly 6 invocations
                        # (1 warmup + 5 real per Rules.md §2.1.17).
                        if len(runs) != 6:
                            invalid_messages.append(
                                _INVALID_MSG_TRAINING_COUNT.format(n=len(runs))
                            )
                        else:
                            # Issue #719: when the ``summary.start`` collision
                            # path (retired ``--loops`` orchestrator) does not
                            # fire — the v3.0 flow of 6 independent ``run``
                            # invocations produces no collision — fall back
                            # to the Rules.md §2.1.17 positional rule: "the
                            # 1st of those 6 is the warm up run". Pick the
                            # lex-earliest ``basename(result_dir)`` (dir
                            # names are ISO-8601-ish so lex == chronological;
                            # tiebreak on abspath is defensive, atomic
                            # datetime allocation prevents same-second
                            # collisions in practice). Retain the collision
                            # path for backward compat with ``--loops``-era
                            # trees.
                            warmup_present = any(
                                os.path.abspath(r.result_dir or "") in self.warmup_result_dirs
                                for r in runs
                            )
                            if not warmup_present:
                                earliest = min(
                                    runs,
                                    key=lambda r: (
                                        os.path.basename(r.result_dir or ""),
                                        os.path.abspath(r.result_dir or ""),
                                    ),
                                )
                                earliest_abs = os.path.abspath(earliest.result_dir or "")
                                self.warmup_result_dirs.add(earliest_abs)
                                self.logger.debug(
                                    f"Detected warmup run (earliest-timestamp "
                                    f"fallback per Rules.md §2.1.17): "
                                    f"{os.path.basename(earliest_abs)} "
                                    "(excluded from aggregate)"
                                )
                    elif bt == BENCHMARK_TYPES.checkpointing:
                        # D-20/D-24: each invocation's metric lists MUST
                        # have exactly 10 entries (Rules.md §2.1.23).
                        for run in runs:
                            violated = False
                            for _mkey, val in (run.metrics or {}).items():
                                if isinstance(val, list) and len(val) != 10:
                                    invalid_messages.append(
                                        _INVALID_MSG_CHECKPOINT_COUNT.format(n=len(val))
                                    )
                                    violated = True
                                    break  # one violation per run is enough
                            if violated:
                                break
                    # vdb + kvcache SKIP rules-strict gates entirely per
                    # the D-22 pass-through boundary — their INVALID
                    # category (if any) is set by the upstream
                    # BenchmarkVerifier, not by Phase 6.

                # ----------------------------------------------------------
                # Datagen-side reportgen checks (training datagen only):
                #   - Supported-model gate: reject models not in the
                #     current mode's allowlist. INVALID.
                #   - Leaf presence: WARN for any missing required file
                #     under the datagen leaf. Does NOT invalidate — a
                #     malformed datagen contribution is worth surfacing
                #     but does not by itself void the submission.
                # Whatif skips both, matching the D-29 policy the #717
                # fix already established.
                # ----------------------------------------------------------
                if (
                    category_str != 'whatif'
                    and runs[0].command == 'datagen'
                    and runs[0].benchmark_type == BENCHMARK_TYPES.training
                ):
                    try:
                        validate_supported_model(
                            runs[0].model or "", category_str or ""
                        )
                    except ConfigurationError as e:
                        invalid_messages.append(str(e))
                    for run in runs:
                        result_dir = run.result_dir or ""
                        missing = validate_datagen_leaf(result_dir)
                        for item in missing:
                            leaf_name = os.path.basename(result_dir) or "?"
                            issues.append(
                                Issue(
                                    PARAM_VALIDATION.CLOSED,
                                    f"[WARN] datagen leaf incomplete "
                                    f"({leaf_name}): {item}",
                                    severity="warning",
                                )
                            )

                # ----------------------------------------------------------
                # Aggregation dispatch — inner try/except for
                # StatisticsError (D-23 loud-failure path).
                # ----------------------------------------------------------
                aggregated_metrics: Dict[str, Any] = {}
                if invalid_messages:
                    # Rules-strict already failed — skip aggregation
                    # entirely; the emitted row has no computed values.
                    category_str = 'INVALID'
                    aggregated_metrics = {}
                else:
                    try:
                        aggregated_metrics = self._aggregate_workload_metrics(runs, self.warmup_result_dirs)  # noqa: E501
                    except statistics.StatisticsError as se:
                        # D-24 empty-metric template. Best-effort key /
                        # basename extraction from the exception message —
                        # the helper's raise-sites format the message as
                        # "metric <key> is empty in invocation <basename>".
                        category_str = 'INVALID'
                        msg_text = str(se)
                        parsed_key = "(unknown)"
                        parsed_basename = "(unknown)"
                        if msg_text.startswith("metric ") and " is empty in invocation " in msg_text:
                            try:
                                after_metric = msg_text[len("metric "):]
                                parsed_key, rest = after_metric.split(
                                    " is empty in invocation ", 1
                                )
                                parsed_basename = rest.strip()
                            except ValueError:
                                pass
                        invalid_messages.append(
                            _INVALID_MSG_EMPTY_METRIC.format(
                                key=parsed_key, basename=parsed_basename
                            )
                        )
                        aggregated_metrics = {}

                # Merge rules-strict issues into the workload's Issue list.
                # Preserve verbatim string (D-25 join happens at write
                # time; Result.issues stays as a typed Issue list until
                # then).
                if invalid_messages:
                    for msg in invalid_messages:
                        issues.append(Issue(PARAM_VALIDATION.INVALID, msg))

                # Resolve the Result.category. Path-derived 'whatif' and
                # rules-strict 'INVALID' are strings; other paths use the
                # verifier's PARAM_VALIDATION enum. Downstream writers
                # coerce this to a string via ``.value`` when needed.
                if category_str == 'INVALID':
                    result_category: Union[PARAM_VALIDATION, str] = PARAM_VALIDATION.INVALID
                elif category_str == 'whatif':
                    result_category = 'whatif'
                else:
                    result_category = verifier_category

                result = Result(
                    multi=True,
                    benchmark_run=runs,
                    benchmark_type=runs[0].benchmark_type,
                    benchmark_command=runs[0].command,
                    benchmark_model=runs[0].model,
                    issues=issues,
                    category=result_category,
                    metrics=aggregated_metrics,
                )
                self.workload_results[workload_key] = result

            except Exception as e:
                error_msg = f"Failed to validate workload {workload_key}: {e}"
                self.logger.error(f"{error_msg}. Skipping workload.")
                self.processing_errors.append(error_msg)

    def print_results(self) -> None:
        """
        Print results with clear OPEN/CLOSED distinction.

        Results are organized by category with INVALID runs first (most critical),
        followed by OPEN runs, then CLOSED runs.
        """
        if not self.run_results and not self.workload_results:
            print("\nNo results to display.")
            if self.processing_errors:
                print("\nProcessing errors occurred:")
                for error in self.processing_errors:
                    print(f"  - {error}")
            return

        # Calculate summary counts
        closed_count = sum(1 for r in self.run_results.values()
                          if r.category == PARAM_VALIDATION.CLOSED)
        open_count = sum(1 for r in self.run_results.values()
                        if r.category == PARAM_VALIDATION.OPEN)
        invalid_count = sum(1 for r in self.run_results.values()
                           if r.category == PARAM_VALIDATION.INVALID)

        # Print summary header
        print(self.summary_formatter.format_summary_header(
            len(self.run_results), closed_count, open_count, invalid_count
        ))

        # Print INVALID runs first (most important to address)
        if invalid_count > 0:
            print(self.summary_formatter.format_section_header(
                PARAM_VALIDATION.INVALID, invalid_count
            ))
            for result in self.run_results.values():
                if result.category == PARAM_VALIDATION.INVALID:
                    self._print_run_details(result)

        # Print OPEN runs
        if open_count > 0:
            print(self.summary_formatter.format_section_header(
                PARAM_VALIDATION.OPEN, open_count
            ))
            for result in self.run_results.values():
                if result.category == PARAM_VALIDATION.OPEN:
                    self._print_run_details(result)

        # Print CLOSED runs
        if closed_count > 0:
            print(self.summary_formatter.format_section_header(
                PARAM_VALIDATION.CLOSED, closed_count
            ))
            for result in self.run_results.values():
                if result.category == PARAM_VALIDATION.CLOSED:
                    self._print_run_details(result)

        # Print submission-level results
        self._print_submission_results()

        # Print any processing errors at the end
        if self.processing_errors:
            print("\n" + "-" * 70)
            print("PROCESSING ERRORS")
            print("-" * 70)
            for error in self.processing_errors:
                print(f"  - {error}")

    def _print_run_details(self, result: Result) -> None:
        """
        Print details for a single run result.

        Args:
            result: The Result object to print.
        """
        # Print header with badge
        run_id = result.benchmark_run.run_id
        print(self.msg_formatter.format_run_header(
            run_id=run_id,
            category=result.category,
            benchmark_type=result.benchmark_type.value if result.benchmark_type else "unknown",
            model=str(result.benchmark_model),
            command=result.benchmark_command
        ))

        # Print issues (only non-CLOSED for brevity)
        print(self.msg_formatter.format_issues_list(result.issues, show_all=False))

        # Print metrics
        print(self.msg_formatter.format_metrics(result.metrics))
        print()

    def _print_submission_results(self) -> None:
        """Print submission-level (workload group) results."""
        if not self.workload_results:
            return

        print("\n" + "=" * 70)
        print("SUBMISSION VALIDATION REPORT")
        print("=" * 70)

        # Group by category
        for category in [PARAM_VALIDATION.INVALID, PARAM_VALIDATION.OPEN, PARAM_VALIDATION.CLOSED]:
            category_results = [
                (k, v) for k, v in self.workload_results.items()
                if v.category == category
            ]

            if not category_results:
                continue

            badge = self.msg_formatter.format_category_badge(category)
            print(f"\n{badge} Submissions ({len(category_results)})")
            print("-" * 40)

            for workload_key, workload_result in category_results:
                self._print_workload_details(workload_key, workload_result)

    def _print_workload_details(self, workload_key: tuple, workload_result: Result) -> None:
        """
        Print details for a workload submission.

        Args:
            workload_key: Per-benchmark-type workload identity tuple (D-05).
                Shape depends on ``benchmark_type``:
                  - training/checkpointing:
                    ``(category, orgname, systemname, model, accelerator)``
                  - vector_database:
                    ``(category, orgname, systemname, engine, index_type)``
                  - kv_cache:
                    ``(category, orgname, systemname, model, performance_profile)``
                Model / accelerator identity is read from
                ``workload_result.benchmark_run[0]`` (D-05 recommendation)
                so this method is decoupled from the key shape and works
                unchanged if the tuple layout evolves.
            workload_result: The Result object for the workload.
        """
        # Read identity from the first BenchmarkRun rather than unpacking
        # the workload_key tuple — the tuple shape now varies by
        # benchmark type (D-05) and unpacking a fixed shape would break
        # for vdb/kvcache. This keeps the printer decoupled from key
        # layout evolution.
        first_run = (
            workload_result.benchmark_run[0]
            if isinstance(workload_result.benchmark_run, list)
            else workload_result.benchmark_run
        )
        model = getattr(first_run, 'model', workload_result.benchmark_model)
        accelerator = getattr(first_run, 'accelerator', '')

        # Determine workload type
        if workload_result.benchmark_model in LLM_MODELS:
            workload_id = f"Checkpointing - {workload_result.benchmark_model}"
        elif workload_result.benchmark_model in MODELS:
            workload_id = f"Training - {workload_result.benchmark_model}, Accelerator: {accelerator}"
        else:
            workload_id = f"{workload_result.benchmark_type.value} - {workload_result.benchmark_model}"

        badge = self.msg_formatter.format_category_badge(workload_result.category)
        print(f"\n{badge} {workload_id}")
        print(f"    Benchmark Type: {workload_result.benchmark_type.value}")

        if workload_result.benchmark_command:
            print(f"    Command: {workload_result.benchmark_command}")

        # Print run summary — sort by disk basename so warmup (always
        # lex-earliest by design of the DLIO stamp mismatch) renders first.
        print("    Runs:")
        sorted_runs = sorted(
            workload_result.benchmark_run,
            key=lambda r: os.path.basename(r.result_dir or "")
        )
        for run in sorted_runs:
            abs_dir = os.path.abspath(run.result_dir or "")
            base = os.path.basename(abs_dir)
            if abs_dir in self.warmup_result_dirs:
                # Warmup runs are excluded from the aggregate — render with
                # a WARMUP label + disk basename (which is unique, unlike
                # the mis-stamped run_id shared with the first real run).
                print(f"      - {run.run_id} [WARMUP, not aggregated — dir: {base}]")
            else:
                result_key = (self._system_scope_key(run), run.run_id)
                run_category = self.run_results[result_key].category
                run_badge = self.msg_formatter.format_category_badge(run_category)
                print(f"      - {run.run_id} {run_badge}")

        # Print submission-level issues
        print(self.msg_formatter.format_issues_list(workload_result.issues, show_all=False))

        # Print requirements checklist for non-CLOSED
        if workload_result.category != PARAM_VALIDATION.CLOSED:
            benchmark_type = workload_result.benchmark_type.value
            checklist = ClosedRequirementsFormatter.format_checklist(benchmark_type)
            if checklist:
                print(f"\n    {checklist}")


    def _final_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Project one internal (machine-key) row onto the fixed
        ``_FINAL_SCHEMA`` (column-parity contract).

        The emitted column set is decided HERE and nowhere else — it is the
        fixed reference schema, never data-driven. Each fixed column is
        cherry-picked from the in-memory row; a column that does not apply
        to this row's workload (or whose metric is absent) stays blank.
        Code/Logs are routed into the row's OWN workload block. The internal
        aggregation (``*_mean_of_*``, the D-24 INVALID gate, etc.) is left
        untouched upstream — those values are simply not emitted here.
        """
        bt = row.get("benchmark_type") or ""
        out: Dict[str, Any] = {col: "" for col in _FINAL_SCHEMA}

        # Left edge + shared SUT block.
        for final_name, key in _SUT_FINAL_COLUMNS:
            if key is None:
                continue  # manual placeholder — present-but-blank
            if key == "__division__":
                out[final_name] = (row.get("category") or "").upper()
            elif key == "__benchmark_type__":
                out[final_name] = bt
            elif key == "__model__":
                # Reference display label; blank for the single-table
                # workloads (kvcache, vdb) — Benchmark Type identifies those.
                if bt in ("kv_cache", "vector_database"):
                    out[final_name] = ""
                else:
                    model = row.get("model") or ""
                    out[final_name] = _MODEL_DISPLAY_LABELS.get(model, model)
            else:
                out[final_name] = row.get(key, "")

        # The row's own workload block (exactly one is populated).
        block = _WORKLOAD_BLOCK_BY_TYPE.get(bt)
        if block is not None:
            prefix, columns = block
            self._fill_block(out, row, prefix, columns)
            if bt == "kv_cache":
                # Fixed option-groups: option N -> its reference group.
                for opt, label in _KVCACHE_GROUPS:
                    for metric_name, metric_suffix in _KVCACHE_GROUP_METRICS:
                        final_name = f"KVCache {label} - {metric_name}"
                        src = f"kvcache_option_{opt}_{metric_suffix}"
                        out[final_name] = row.get(src, "")

        return {k: _blank_if_nan(v) for k, v in out.items()}

    @staticmethod
    def _fill_block(out: Dict[str, Any], row: Dict[str, Any],
                    prefix: str, columns) -> None:
        """Fill one workload block. ``__code__`` / ``__logs__`` route the
        row's per-run code-image hyperlinks (``sut_code`` / ``sut_logs``)
        into this block's Code/Logs columns."""
        for name, key in columns:
            final_name = f"{prefix} - {name}"
            if key == "__code__":
                out[final_name] = row.get("sut_code", "")
            elif key == "__logs__":
                out[final_name] = row.get("sut_logs", "")
            else:
                out[final_name] = row.get(key, "")

    def write_json_file(self, results, target_dir: Optional[str] = None):
        out_dir = target_dir if target_dir is not None else self.results_dir
        json_file = os.path.join(out_dir, 'results.json')
        self.logger.info(f'Writing results to {json_file}')
        projected = [self._final_row(r) for r in results]
        with open(json_file, 'w') as f:
            json.dump(projected, f, indent=2, cls=MLPSJsonEncoder)

    def write_csv_file(self, results, target_dir: Optional[str] = None):
        out_dir = target_dir if target_dir is not None else self.results_dir
        csv_file = os.path.join(out_dir, 'results.csv')
        self.logger.info(f'Writing results to {csv_file}')
        # Project each internal row onto the fixed column-parity schema.
        # ``_final_row`` returns EXACTLY the _FINAL_SCHEMA keys (NaN already
        # blanked), so no flatten/field-discovery is needed — the header is
        # the fixed schema regardless of input (header-only when empty).
        projected = [self._final_row(r) for r in results]
        with open(csv_file, 'w+', newline='') as file_object:
            csv_writer = csv.DictWriter(
                f=file_object, fieldnames=_FINAL_SCHEMA, lineterminator='\n')
            csv_writer.writeheader()
            csv_writer.writerows(projected)
