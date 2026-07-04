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
import sys

from dataclasses import dataclass
from statistics import fmean, StatisticsError
from typing import List, Dict, Any, Optional, Set

from mlpstorage_py.mlps_logging import setup_logging, apply_logging_options
from mlpstorage_py.config import MLPS_DEBUG, BENCHMARK_TYPES, EXIT_CODE, PARAM_VALIDATION, LLM_MODELS, MODELS, ACCELERATORS
from mlpstorage_py.rules import get_runs_files, BenchmarkVerifier, BenchmarkRun, Issue, RunID
from mlpstorage_py.utils import flatten_nested_dict, remove_nan_values
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
_INVALID_MSG_WARMUP_UNDETECTED = (
    "expected exactly 1 warmup invocation to be detected; found 0 in a 6-invocation set. "
    "Cannot compute Rules.md §2.1.17 5-run mean."
)
_INVALID_MSG_CHECKPOINT_COUNT = (
    "expected 10 checkpoint operations per Rules.md §2.1.23; found {n}"
)
_INVALID_MSG_EMPTY_METRIC = (
    "metric {key} is empty in invocation {basename}; cannot aggregate"
)


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
        # Canonical tree probe.
        systemname = None
        if self.args is not None:
            systemname = getattr(self.args, 'systemname', None)
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
                        self.logger.info(
                            "Detected canonical submission tree; scoping to "
                            "%s/results/%s for --systemname=%s",
                            org_dir, systemname, systemname,
                        )
                        return str(system_dir)
                else:
                    # No --systemname: rebind to the org's results/ folder
                    # itself. The global summary lands here (see
                    # _global_summary_dir_for), and discover_scan_roots
                    # walks each system slice under it.
                    self.logger.info(
                        "Detected canonical submission tree without "
                        "--systemname; aggregating across every system "
                        "under %s",
                        results_root,
                    )
                    return str(results_root)
        return results_dir

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

        # Two rollups are written on every reportgen invocation:
        #
        # 1. GLOBAL summary at `<results-dir>/results.{json,csv}` — every
        #    successfully-processed run in one file. Preserves the
        #    pre-model-rollup contract that downstream tooling and the
        #    submission-review flow rely on.
        # 2. PER-MODEL rollups at `<model>/results.{json,csv}` (or the
        #    equivalent grouping folder per benchmark type — see
        #    _model_group_folder for the layout map). Same runs, but
        #    partitioned so a submitter can inspect one model in isolation.
        #
        # Both derive from the same `self.run_results` collection; the
        # per-model loop groups by `_model_group_folder(result)`.

        all_run_dicts: List[dict] = []
        groups: Dict[str, List[dict]] = {}
        skipped = 0
        for result in self.run_results.values():
            run_dict = result.benchmark_run.as_dict()
            all_run_dicts.append(run_dict)
            model_folder = self._model_group_folder(result)
            if model_folder is None:
                skipped += 1
                continue
            groups.setdefault(model_folder, []).append(run_dict)

        if not all_run_dicts:
            self.logger.warning(
                "No runs to write rollups for (skipped %d runs without result_dir).",
                skipped,
            )
            return EXIT_CODE.SUCCESS

        # (1) Global summary — written to self.global_summary_dir. For a
        # canonical submission tree that is the org's `results/` folder
        # (one level above each per-system slice), so the aggregate sits
        # alongside every system's subdirectory instead of inside one of
        # them. For a flat layout it equals self.results_dir.
        self.write_json_file(all_run_dicts, target_dir=self.global_summary_dir)
        self.write_csv_file(all_run_dicts, target_dir=self.global_summary_dir)

        # (2) Per-model rollups.
        if not groups:
            self.logger.warning(
                "No model folders to write per-model rollups for "
                "(skipped %d runs without result_dir).",
                skipped,
            )
        for model_folder, run_dicts in sorted(groups.items()):
            self.write_json_file(run_dicts, target_dir=model_folder)
            self.write_csv_file(run_dicts, target_dir=model_folder)

        return EXIT_CODE.SUCCESS

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

        DLIO writes the warmup's ``summary.start`` value to match the FIRST
        real run's start time (not the warmup's own directory timestamp), so
        the two runs produce equal ``run_id`` values and collide in
        ``self.run_results``. Detection here: on collision, the run whose
        ``result_dir`` basename is lex-earlier is the warmup — its basename
        is recorded in ``self.warmup_result_dirs`` and the later run wins
        the dict slot (matching prior dict-overwrite semantics, which are
        preserved so aggregate counts are unchanged). The workload printer
        looks up ``warmup_result_dirs`` to render the warmup with a
        ``[WARMUP, not aggregated]`` label instead of a category badge.

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
    # aggregation math lives. It is added here (Plan 06-02) but not yet
    # wired into ``_process_workload_groups``; wiring happens in Plan 06-04
    # (the TODO at ``metrics={}`` in this file's ``_process_workload_groups``
    # is the eventual call site).
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
        metric dict that eventually populates the ``metrics=`` slot on the
        workload ``Result`` (see the TODO at ``_process_workload_groups``).

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
        return out

    def _aggregate_vdb(
        self,
        runs: List[BenchmarkRun],
    ) -> Dict[str, Any]:
        """
        VDB-branch aggregation (D-21) — pass-through, NOT math.

        vdb's internal ``vdb-aggregate`` tool (see
        ``benchmarks/vectordbbench.py:508``) owns the math contract per
        the D-22 boundary; this helper copies pre-computed values from
        the workload's ``summary.json`` into ``vdb_*`` columns.

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
        read from ``runs[0].parameters`` (the ``BenchmarkRun`` accessor
        for CLI/YAML args on disk).
        """
        run = runs[0]
        summary = self._load_workload_summary(run)

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
        # (per submission_checker/checks/vdb_checks.py:462-469).
        recall = summary.get("recall")
        if recall is None and run.result_dir:
            recall_stats_path = os.path.join(run.result_dir, "recall_stats.json")
            if os.path.isfile(recall_stats_path):
                try:
                    with open(recall_stats_path, "r") as f:
                        recall_stats = json.load(f)
                    recall = recall_stats.get("recall")
                except (OSError, ValueError) as e:
                    self.logger.warning(
                        f"vdb: could not read recall_stats.json at "
                        f"{recall_stats_path}: {e}"
                    )
        out["vdb_recall"] = recall

        # Identity columns (D-15). ``BenchmarkRun`` exposes CLI/YAML args
        # via ``.parameters``; use ``.get`` so missing keys emit ``None``.
        params = run.parameters or {}
        out["vdb_engine"] = params.get("engine")
        out["vdb_index_type"] = params.get("index_type")
        return out

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

    def _process_workload_groups(self, benchmark_runs: List[BenchmarkRun]) -> None:
        """
        Group runs by workload and run submission-level validation.

        Args:
            benchmark_runs: List of all benchmark runs.
        """
        # Group by (model, accelerator) for training, (model,) for others
        workload_runs: Dict[tuple, List[BenchmarkRun]] = {}

        for benchmark_run in benchmark_runs:
            workload_key = (benchmark_run.model, benchmark_run.accelerator)
            if workload_key not in workload_runs:
                workload_runs[workload_key] = []
            workload_runs[workload_key].append(benchmark_run)

        # Run workload-level verifiers
        for workload_key, runs in workload_runs.items():
            model, accelerator = workload_key
            if not runs:
                continue

            try:
                self.logger.info(
                    f'Running submission verifiers for model: {model}, '
                    f'accelerator: {accelerator} ({len(runs)} runs)'
                )
                verifier = BenchmarkVerifier(*runs, logger=self.logger)
                category = verifier.verify()
                issues = verifier.issues

                result = Result(
                    multi=True,
                    benchmark_run=runs,
                    benchmark_type=runs[0].benchmark_type,
                    benchmark_command=runs[0].command,
                    benchmark_model=runs[0].model,
                    issues=issues,
                    category=category,
                    metrics={}  # TODO: Add function to aggregate metrics
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
            workload_key: Tuple of (model, accelerator).
            workload_result: The Result object for the workload.
        """
        model, accelerator = workload_key

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


    def write_json_file(self, results, target_dir: Optional[str] = None):
        out_dir = target_dir if target_dir is not None else self.results_dir
        json_file = os.path.join(out_dir, 'results.json')
        self.logger.info(f'Writing results to {json_file}')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

    def write_csv_file(self, results, target_dir: Optional[str] = None):
        out_dir = target_dir if target_dir is not None else self.results_dir
        csv_file = os.path.join(out_dir, 'results.csv')
        self.logger.info(f'Writing results to {csv_file}')
        flattened_results = [flatten_nested_dict(r) for r in results]
        flattened_results = [remove_nan_values(r) for r in flattened_results]
        fieldnames = set()
        for l in flattened_results:
            fieldnames.update(l.keys())

        with open(csv_file, 'w+', newline='') as file_object:
            csv_writer = csv.DictWriter(f=file_object, fieldnames=sorted(fieldnames), lineterminator='\n')
            csv_writer.writeheader()
            csv_writer.writerows(flattened_results)
