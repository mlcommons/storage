"""VdbCheck — Rules.md §5 (Vector Database) implementation.

Implements all 16 rules from Rules.md §5 (5.1.1–5.6.5) as
``@rule``-decorated methods on a single ``BaseCheck`` subclass. Every
rule body guards on ``self.mode != "vector_database"`` so the check is a
no-op on non-vdb subtrees — the on-disk type-segment is ``vector_database``
(Phase 4 D-02), so the loader at ``loader.py:99-103`` yields
``loader_metadata.mode == "vector_database"`` on those leaves.

§5.6.1 (``vdbClosedSubmissionChecksum``) delegates to the shared
``helpers._check_code_image_layered`` (Phase 4 CD-04 + D-06) — the same
helper TrainingCheck.3.6.1 uses — so the layered self-consistency +
upstream-identity model is enforced once and attributed under the
caller's rule ID.

Index-type rules (5.3.1, 5.6.3) compare the on-disk directory name
(UPPERCASE — e.g. ``"DISKANN"``) directly against the
``summary.json.index_type`` token (also UPPERCASE).

Loader caveat: at Phase 4 land time, ``loader.py`` has only two branches
(``training`` and an ``else`` for checkpointing) and therefore does NOT
populate ``submissions_logs.run_files`` / ``datagen_files`` for
``vector_database`` mode. Rule bodies that depend on those fields detect the
absence and emit ``warn_violation`` so the gap is grep-visible — see
the Phase-4 invariant: "must NEVER be a ``return True`` stub." When the
loader gains a vdb branch, the warn paths drop out automatically and the
real checks fire.
"""

import os

from .base import BaseCheck
from ..configuration.configuration import Config
from ..loader import SubmissionLogs
from ..rule_registry import rule
from .helpers import _check_code_image_layered, _check_filesystem_separation
from mlpstorage_py.config import VDB_INDEX_TYPES_CLOSED


# Required latency / throughput fields each run's summary.json must report (§5.3.4).
_REQUIRED_METRIC_FIELDS = (
    "throughput_qps",
    "mean_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "p999_latency_ms",
)


# Allowed CLOSED tunable parameters per Rules.md §5.6.4 table.
_CLOSED_ALLOWED_PARAMS = frozenset({
    # Database
    "database.database",
    # Index selection
    "index.index_type",
    "index.metric_type",
    # DISKANN / HNSW / AISAQ build + search params (combined; submitter chooses one family)
    "index.max_degree",
    "index.search_list_size",
    "index.M",
    "index.ef_construction",
    "index.inline_pq",
    "search.search_ef",
    # Run-time
    "run.mode",
    "run.num_query_processes",
    "run.batch_size",
    "run.report_count",
    # Dataset / load
    "dataset.collection_name",
    "dataset.num_shards",
    "dataset.chunk_size",
    "dataset.batch_size",
    "dataset.vector_dtype",
    # Storage
    "storage.storage_root",
    "storage.storage_type",
})


# Additional OPEN params beyond the CLOSED set (Rules.md §5.6.5 table).
# Backend-specific params (pgvector lists/probes, Elasticsearch m / ef_construction
# / num_candidates, etc.) are NOT enumerable up-front; non-Milvus backends are
# handled via a warn-and-skip path below.
_OPEN_EXTRA_ALLOWED_PARAMS = frozenset({
    "database.host",
    "database.port",
})


class VdbCheck(BaseCheck):
    """Check class for Rules.md §5 (Vector Database) rules.

    Mirrors the ``TrainingCheck`` / ``CheckpointingCheck`` constructor shape
    (``__init__(self, log, config, submissions_logs)``) — Phase 3 D-S4
    invariant preserved so ``main.py`` instantiates every checker generically.
    """

    def __init__(self, log, config: Config, submissions_logs: SubmissionLogs):
        """Initialize VdbCheck.

        Args:
            log: Logger instance (passed through to ``BaseCheck``).
            config: A ``Config`` instance for submission configuration.
            submissions_logs: A ``SubmissionLogs`` instance for accessing
                submission logs.
        """
        super().__init__(log=log, path=submissions_logs.loader_metadata.folder)
        self.config = config
        self.submissions_logs = submissions_logs
        self.mode = self.submissions_logs.loader_metadata.mode
        self.division = self.submissions_logs.loader_metadata.division
        self.name = "vdb checks"
        self.run_path = os.path.join(self.path, "run")
        self.datagen_path = os.path.join(self.path, "datagen")
        self.init_checks()

    def init_checks(self):
        """Register all 16 §5 rule methods (Phase 4 D-01 full implementation)."""
        self.checks = [
            self.vdb_dataset_scale,                # 5.1.1
            self.vdb_dimension_consistency,        # 5.1.2
            self.vdb_collection_populated,         # 5.2.1
            self.vdb_index_build_completed,        # 5.2.2
            self.vdb_run_count,                    # 5.3.1
            self.vdb_recall_reported,              # 5.3.2
            self.vdb_query_count_minimum,          # 5.3.3
            self.vdb_metrics_reported,             # 5.3.4
            self.vdb_path_args,                    # 5.4.1
            self.vdb_filesystem_check,             # 5.4.2
            self.vdb_object_storage_backend,       # 5.5.1
            self.vdb_closed_submission_checksum,   # 5.6.1
            self.vdb_closed_database_backend,      # 5.6.2
            self.vdb_closed_index_types,           # 5.6.3
            self.vdb_closed_submission_parameters, # 5.6.4
            self.vdb_open_submission_parameters,   # 5.6.5
        ]

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _iter_run_files(self):
        """Yield run-summary tuples or empty iterable when the loader did not populate them.

        Phase 4 land time: ``Loader.load()`` only fills ``run_files`` /
        ``datagen_files`` for ``mode == "training"``; the ``else`` branch
        fills ``checkpoint_files`` for everything else. For ``vector_database``
        leaves this means ``run_files`` is ``None`` (the dataclass default).
        Rule methods consume this iterator instead of touching ``run_files``
        directly so they degrade to an empty walk without crashing.
        """
        run_files = self.submissions_logs.run_files
        if not run_files:
            return iter(())
        return iter(run_files)

    def _iter_datagen_files(self):
        """Counterpart to ``_iter_run_files`` for the datagen list."""
        datagen_files = self.submissions_logs.datagen_files
        if not datagen_files:
            return iter(())
        return iter(datagen_files)

    def _get_benchmark_api(self) -> str:
        """Return 'file' or 'object' (default 'file') from the system YAML.

        Mirrors ``TrainingCheck._get_benchmark_api`` so 5.4.2 and 5.5.1 honor
        the same per-API gating as the training filesystem check.
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

    def _vdb_loader_gap_warning(self, rule_id: str, rule_name: str) -> None:
        """Emit a single warn_violation that the loader does not yet surface vector_database logs.

        This is the grep-visible signal required by the Phase-4 invariant
        "must NEVER be a ``return True`` stub." When the loader gains a
        vdb branch (loader.py 99-143), the run_files / datagen_files iters
        become non-empty and these warnings drop out.
        """
        self.warn_violation(
            rule_id, rule_name, self.path,
            "vector_database summary/metadata not surfaced by Loader at this revision; "
            "rule structure is in place but cannot fire — gap tracked for the "
            "loader vector_database branch follow-up",
        )

    # -----------------------------------------------------------------------
    # 5.1 Sizing
    # -----------------------------------------------------------------------

    @rule("5.1.1", "vdbDatasetScale")
    def vdb_dataset_scale(self):
        """Read num_vectors / dimension from each run's summary.json and compare
        against a defined-scale table. (Rules.md 5.1.1)

        The scale-table constant is not yet defined in ``constants.py`` /
        ``config.py`` — when it lands, replace the warn_violation with a
        real lookup. Until then the rule is implemented and grep-visible
        but does not fire.
        """
        valid = True
        if self.mode != "vector_database":
            return valid

        # The defined-scale table is not yet in config.py; surface the gap.
        self.warn_violation(
            "5.1.1", "vdbDatasetScale", self.path,
            "vdb scale table (num_vectors, dimension) not yet defined in "
            "config.py; per-run scale check deferred",
        )

        any_run = False
        for summary, metadata, ts in self._iter_run_files():
            any_run = True
            if summary is None:
                self.log.debug(
                    "[5.1.1] %s/%s: skipping (summary not loaded)",
                    self.path, ts,
                )
                continue
            num_vectors = summary.get("num_vectors")
            dimension = summary.get("dimension")
            if num_vectors is None:
                self.log_violation(
                    "5.1.1", "vdbDatasetScale", self.path,
                    "summary.json at %s/%s is missing num_vectors",
                    self.path, ts,
                )
                valid = False
            if dimension is None:
                self.log_violation(
                    "5.1.1", "vdbDatasetScale", self.path,
                    "summary.json at %s/%s is missing dimension",
                    self.path, ts,
                )
                valid = False

        if not any_run:
            self._vdb_loader_gap_warning("5.1.1", "vdbDatasetScale")

        return valid

    @rule("5.1.2", "vdbDimensionConsistency")
    def vdb_dimension_consistency(self):
        """Compare the load-time dimension against each run's dimension; mismatch fails.
        (Rules.md 5.1.2)
        """
        valid = True
        if self.mode != "vector_database":
            return valid

        load_dimensions = []
        for summary, metadata, ts in self._iter_datagen_files():
            if summary is None:
                continue
            dim = summary.get("dimension")
            if dim is not None:
                load_dimensions.append((dim, ts))

        if not load_dimensions:
            self.log.debug(
                "[5.1.2] %s: no datagen summary surfaced; dimension cross-check skipped "
                "(STRUCT-12/STRUCT-13 cover missing-datagen)",
                self.path,
            )

        any_run = False
        for summary, metadata, ts in self._iter_run_files():
            any_run = True
            if summary is None:
                continue
            run_dim = summary.get("dimension")
            if run_dim is None:
                continue
            for load_dim, load_ts in load_dimensions:
                if load_dim != run_dim:
                    self.log_violation(
                        "5.1.2", "vdbDimensionConsistency", self.path,
                        "vdb dimension mismatch: datagen %s reports %s but run %s reports %s",
                        load_ts, load_dim, ts, run_dim,
                    )
                    valid = False

        if not any_run and not load_dimensions:
            self._vdb_loader_gap_warning("5.1.2", "vdbDimensionConsistency")

        return valid

    # -----------------------------------------------------------------------
    # 5.2 Generation
    # -----------------------------------------------------------------------

    @rule("5.2.1", "vdbCollectionPopulated")
    def vdb_collection_populated(self):
        """Confirm inserted_vectors >= num_vectors at load. (Rules.md 5.2.1)"""
        valid = True
        if self.mode != "vector_database":
            return valid

        any_load = False
        for summary, metadata, ts in self._iter_datagen_files():
            any_load = True
            if summary is None:
                continue
            inserted = summary.get("inserted_vectors")
            declared = summary.get("num_vectors")
            if inserted is None or declared is None:
                self.log_violation(
                    "5.2.1", "vdbCollectionPopulated", self.path,
                    "datagen summary at %s/%s missing inserted_vectors or num_vectors",
                    self.path, ts,
                )
                valid = False
                continue
            try:
                if int(inserted) < int(declared):
                    self.log_violation(
                        "5.2.1", "vdbCollectionPopulated", self.path,
                        "vdb collection underpopulated at %s/%s: "
                        "inserted %s of %s vectors at load time",
                        self.path, ts, inserted, declared,
                    )
                    valid = False
            except (TypeError, ValueError) as e:
                self.log_violation(
                    "5.2.1", "vdbCollectionPopulated", self.path,
                    "datagen summary at %s/%s has non-numeric inserted/declared counts: %s",
                    self.path, ts, str(e),
                )
                valid = False

        if not any_load:
            self._vdb_loader_gap_warning("5.2.1", "vdbCollectionPopulated")

        return valid

    @rule("5.2.2", "vdbIndexBuildCompleted")
    def vdb_index_build_completed(self):
        """Confirm an index-build record is present in the load summary and that
        the load-time index_type matches the run-time index_type. (Rules.md 5.2.2)
        """
        valid = True
        if self.mode != "vector_database":
            return valid

        load_index_types = []
        any_load = False
        for summary, metadata, ts in self._iter_datagen_files():
            any_load = True
            if summary is None:
                continue
            idx_type = summary.get("index_type")
            if idx_type is None:
                self.log_violation(
                    "5.2.2", "vdbIndexBuildCompleted", self.path,
                    "datagen summary at %s/%s missing index_type "
                    "(no index-build record)",
                    self.path, ts,
                )
                valid = False
                continue
            load_index_types.append((idx_type, ts))

        any_run = False
        for summary, metadata, ts in self._iter_run_files():
            any_run = True
            if summary is None:
                continue
            run_idx = summary.get("index_type")
            if run_idx is None:
                continue
            for load_idx, load_ts in load_index_types:
                if load_idx != run_idx:
                    self.log_violation(
                        "5.2.2", "vdbIndexBuildCompleted", self.path,
                        "vdb index_type changed between datagen %s (%s) and run %s (%s)",
                        load_ts, load_idx, ts, run_idx,
                    )
                    valid = False

        if not any_load and not any_run:
            self._vdb_loader_gap_warning("5.2.2", "vdbIndexBuildCompleted")

        return valid

    # -----------------------------------------------------------------------
    # 5.3 Run
    # -----------------------------------------------------------------------

    @rule("5.3.1", "vdbRunCount")
    def vdb_run_count(self):
        """Verify exactly five timestamp directories under <leaf>/run/.
        (Rules.md 5.3.1; Phase 4 D-04: count applies to run/, not datagen/.)
        """
        valid = True
        if self.mode != "vector_database":
            return valid

        # STRUCT layer owns missing-run/ structural violation.
        if not os.path.isdir(self.run_path):
            return valid

        timestamps = [
            d for d in os.listdir(self.run_path)
            if os.path.isdir(os.path.join(self.run_path, d)) and not d.startswith(".")
        ]
        if len(timestamps) != 5:
            self.log_violation(
                "5.3.1", "vdbRunCount", self.run_path,
                "vdbRunCount: expected exactly 5 run timestamp directories under %s, found %d",
                self.run_path, len(timestamps),
            )
            valid = False

        return valid

    @rule("5.3.2", "vdbRecallReported")
    def vdb_recall_reported(self):
        """Verify a recall value is present in summary.json or recall_stats.json
        for each run and that it meets the minimum target for the scale.
        (Rules.md 5.3.2)

        The minimum-target table per scale/metric is not yet in config.py;
        the presence check still runs, the threshold check is deferred via
        warn_violation.
        """
        valid = True
        if self.mode != "vector_database":
            return valid

        # The minimum-recall target table is not yet in config.py.
        self.warn_violation(
            "5.3.2", "vdbRecallReported", self.path,
            "vdb minimum-recall target table (per scale/metric) not yet "
            "defined in config.py; threshold check deferred — presence "
            "check still runs",
        )

        any_run = False
        for summary, metadata, ts in self._iter_run_files():
            any_run = True
            if summary is None:
                self.log.debug(
                    "[5.3.2] %s/%s: skipping (summary not loaded)",
                    self.path, ts,
                )
                continue
            recall = summary.get("recall")
            if recall is None:
                # Fall back to rank-local recall_stats.json adjacent to summary.json
                recall_stats_path = os.path.join(self.run_path, ts, "recall_stats.json")
                if not os.path.isfile(recall_stats_path):
                    self.log_violation(
                        "5.3.2", "vdbRecallReported", self.path,
                        "vdbRecallReported: no recall value present in "
                        "summary.json or recall_stats.json at %s/%s",
                        self.path, ts,
                    )
                    valid = False

        if not any_run:
            self._vdb_loader_gap_warning("5.3.2", "vdbRecallReported")

        return valid

    @rule("5.3.3", "vdbQueryCountMinimum")
    def vdb_query_count_minimum(self):
        """Verify each run issued at least the minimum number of queries.
        (Rules.md 5.3.3)

        The minimum-query table per scale is not yet in config.py; structure
        is in place, threshold check deferred via warn_violation.
        """
        valid = True
        if self.mode != "vector_database":
            return valid

        self.warn_violation(
            "5.3.3", "vdbQueryCountMinimum", self.path,
            "vdb minimum-query target table (per scale) not yet defined in "
            "config.py; threshold check deferred — presence check still runs",
        )

        any_run = False
        for summary, metadata, ts in self._iter_run_files():
            any_run = True
            if summary is None:
                continue
            qps = summary.get("throughput_qps")
            total_time = summary.get("total_time_seconds")
            query_count = summary.get("query_count")
            if qps is None and query_count is None:
                self.log_violation(
                    "5.3.3", "vdbQueryCountMinimum", self.path,
                    "vdbQueryCountMinimum: summary.json at %s/%s has neither "
                    "throughput_qps nor query_count — cannot compute issued queries",
                    self.path, ts,
                )
                valid = False
            elif query_count is None and total_time is None:
                self.log_violation(
                    "5.3.3", "vdbQueryCountMinimum", self.path,
                    "vdbQueryCountMinimum: summary.json at %s/%s missing total_time_seconds "
                    "for QPS-based issued-query computation",
                    self.path, ts,
                )
                valid = False

        if not any_run:
            self._vdb_loader_gap_warning("5.3.3", "vdbQueryCountMinimum")

        return valid

    @rule("5.3.4", "vdbMetricsReported")
    def vdb_metrics_reported(self):
        """Verify each run's summary.json reports the required metric fields.
        (Rules.md 5.3.4)
        """
        valid = True
        if self.mode != "vector_database":
            return valid

        any_run = False
        for summary, metadata, ts in self._iter_run_files():
            any_run = True
            if summary is None:
                continue
            for field in _REQUIRED_METRIC_FIELDS:
                if field not in summary:
                    self.log_violation(
                        "5.3.4", "vdbMetricsReported", self.path,
                        "vdbMetricsReported: required field %r missing from summary.json at %s/%s",
                        field, self.path, ts,
                    )
                    valid = False

        if not any_run:
            self._vdb_loader_gap_warning("5.3.4", "vdbMetricsReported")

        return valid

    # -----------------------------------------------------------------------
    # 5.4 POSIX-API options
    # -----------------------------------------------------------------------

    @rule("5.4.1", "vdbPathArgs")
    def vdb_path_args(self):
        """Verify vdb data path and results dir args are both set and differ.
        (Rules.md 5.4.1)
        """
        valid = True
        if self.mode != "vector_database":
            return valid

        any_run = False
        for summary, metadata, ts in self._iter_run_files():
            any_run = True
            if metadata is None:
                self.log.debug(
                    "[5.4.1] %s/%s: skipping (metadata not loaded)",
                    self.path, ts,
                )
                continue
            args = metadata.get("args", {})
            # The vdb runner uses storage-root for the data path; data_dir is the
            # generic mlpstorage name. Honor either to keep the rule resilient
            # to the args-shape refactor that lands alongside Phase 4.
            data_path = (
                args.get("storage_root")
                or args.get("data_dir")
                or args.get("vdb_data_path")
            )
            results_dir = args.get("results_dir")

            if not data_path:
                self.log_violation(
                    "5.4.1", "vdbPathArgs", self.path,
                    "vdbPathArgs: vdb data path arg not set in metadata at %s/%s",
                    self.path, ts,
                )
                valid = False
            if not results_dir:
                self.log_violation(
                    "5.4.1", "vdbPathArgs", self.path,
                    "vdbPathArgs: results_dir not set in metadata at %s/%s",
                    self.path, ts,
                )
                valid = False
            if data_path and results_dir and data_path == results_dir:
                self.log_violation(
                    "5.4.1", "vdbPathArgs", self.path,
                    "vdbPathArgs: vdb data path %s and results_dir %s must differ",
                    data_path, results_dir,
                )
                valid = False

        if not any_run:
            self._vdb_loader_gap_warning("5.4.1", "vdbPathArgs")

        return valid

    @rule("5.4.2", "vdbFilesystemCheck")
    def vdb_filesystem_check(self):
        """Verify vdb data dir and results dir are on different filesystems.
        (Rules.md 5.4.2)

        Reuses the canonical ``_check_filesystem_separation`` helper that
        TrainingCheck.3.4.2 / CheckpointingCheck.4.4.2 use. Object-API
        submissions silent-pass (D-B7).
        """
        valid = True
        if self.mode != "vector_database":
            return valid

        if self._get_benchmark_api() == "object":
            return valid

        any_run = False
        for summary, metadata, ts in self._iter_run_files():
            any_run = True
            if metadata is None:
                self.log.debug(
                    "[5.4.2] %s/%s: skipping (metadata not loaded)",
                    self.path, ts,
                )
                continue
            args = metadata.get("args", {}) or {}
            # _check_filesystem_separation looks up "data_dir" or
            # "checkpoint_folder"; for vdb the analog is storage_root. Synthesize
            # a flat dict so the helper sees data_dir + results_dir.
            shim_args = dict(args)
            if "data_dir" not in shim_args:
                storage_root = args.get("storage_root") or args.get("vdb_data_path")
                if storage_root:
                    shim_args["data_dir"] = storage_root
            logfile_path = os.path.join(self.run_path, ts, "vdb_run.stdout.log")
            ok, df_found = _check_filesystem_separation(shim_args, logfile_path)
            if not df_found:
                self.log_violation(
                    "5.4.2", "vdbFilesystemCheck", logfile_path,
                    "df output not found",
                )
                valid = False
                continue
            if not ok:
                self.log_violation(
                    "5.4.2", "vdbFilesystemCheck", logfile_path,
                    "vdbFilesystemCheck: vdb data path and results_dir are on the "
                    "same filesystem",
                )
                valid = False

        if not any_run:
            self._vdb_loader_gap_warning("5.4.2", "vdbFilesystemCheck")

        return valid

    # -----------------------------------------------------------------------
    # 5.5 Object-API options
    # -----------------------------------------------------------------------

    @rule("5.5.1", "vdbObjectStorageBackend")
    def vdb_object_storage_backend(self):
        """For object-API submissions, verify the storage backend is S3-compatible
        and consistent with the declared API. (Rules.md 5.5.1)
        """
        valid = True
        if self.mode != "vector_database":
            return valid

        # Only applies under object API.
        if self._get_benchmark_api() != "object":
            return valid

        any_run = False
        for summary, metadata, ts in self._iter_run_files():
            any_run = True
            if summary is None:
                continue
            backend = (
                summary.get("database", {}).get("storage_backend")
                if isinstance(summary.get("database"), dict)
                else None
            )
            if not backend:
                self.log_violation(
                    "5.5.1", "vdbObjectStorageBackend", self.path,
                    "vdbObjectStorageBackend: object-API submission missing "
                    "database.storage_backend in summary.json at %s/%s",
                    self.path, ts,
                )
                valid = False
                continue
            # S3-compatible backends: accept exact names or `s3-` prefix
            # (e.g. "s3-compatible", "s3-express"). Substring match is too
            # loose — "non-s3-storage" / "s3-incompatible-fork" should fail.
            backend_lc = str(backend).lower()
            _S3_COMPATIBLE_NAMES = frozenset({"s3", "s3-compatible", "minio", "ceph"})
            _S3_COMPATIBLE_PREFIXES = ("s3-",)
            if (
                backend_lc not in _S3_COMPATIBLE_NAMES
                and not backend_lc.startswith(_S3_COMPATIBLE_PREFIXES)
            ):
                self.log_violation(
                    "5.5.1", "vdbObjectStorageBackend", self.path,
                    "vdbObjectStorageBackend: object-API submission must record an "
                    "S3-compatible backend; found %r",
                    backend,
                )
                valid = False

        if not any_run:
            self._vdb_loader_gap_warning("5.5.1", "vdbObjectStorageBackend")

        return valid

    # -----------------------------------------------------------------------
    # 5.6 OPEN vs CLOSED
    # -----------------------------------------------------------------------

    @rule("5.6.1", "vdbClosedSubmissionChecksum")
    def vdb_closed_submission_checksum(self):
        """For CLOSED submissions, verify the code-image self-consistency +
        upstream-identity via the shared layered helper. (Rules.md 5.6.1)

        Phase 4 D-06 / CD-04: delegates to
        ``helpers._check_code_image_layered`` — the SAME helper
        ``TrainingCheck.3.6.1`` calls — so the layered model is implemented
        once and attributed under the caller's rule ID/name.

        Walk-up: ``self.path`` is the per-leaf vdb path
        (``<root>/closed/<orgname>/results/<system>/vector_database/<DisplayIndex>``).
        The CLOSED ``code/`` lives at ``<root>/closed/<orgname>/code/``,
        four levels above ``self.path`` (DisplayIndex → vector_database → system
        → results → ``<orgname>``).

        Missing ``code/`` is NOT logged here — STRUCT-06 (§2.1.6) owns the
        VALS-01 missing-code/ violation; re-firing here would double-count.
        """
        if self.mode != "vector_database":
            return True
        if self.division != "closed":
            return True

        # <root>/closed/<orgname>/results/<system>/vector_database/<DisplayIndex>
        # walk up four levels: DisplayIndex → vector_database → system → results → <orgname>
        submitter_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.path))))
        code_path = os.path.join(submitter_path, "code")

        if not os.path.isdir(code_path):
            return True  # STRUCT-06 owns missing-code/

        expected = self.config.get_reference_checksum()
        return _check_code_image_layered(
            code_path,
            "closed",
            expected,
            self.log,
            self.log_violation,
            "5.6.1",
            "vdbClosedSubmissionChecksum",
        )

    @rule("5.6.2", "vdbClosedDatabaseBackend")
    def vdb_closed_database_backend(self):
        """For CLOSED, verify database.database == 'milvus'. (Rules.md 5.6.2)"""
        valid = True
        if self.mode != "vector_database":
            return valid
        if self.division != "closed":
            return valid

        any_run = False
        for summary, metadata, ts in self._iter_run_files():
            any_run = True
            if summary is None:
                continue
            db_block = summary.get("database")
            backend = db_block.get("database") if isinstance(db_block, dict) else None
            if backend != "milvus":
                self.log_violation(
                    "5.6.2", "vdbClosedDatabaseBackend", self.path,
                    "vdbClosedDatabaseBackend: CLOSED requires milvus backend, "
                    "found %r at %s/%s",
                    backend, self.path, ts,
                )
                valid = False

        if not any_run:
            self._vdb_loader_gap_warning("5.6.2", "vdbClosedDatabaseBackend")

        return valid

    @rule("5.6.3", "vdbClosedIndexTypes")
    def vdb_closed_index_types(self):
        """For CLOSED, verify index type is DISKANN / HNSW / AISAQ and that
        the on-disk directory name matches the summary.json index_type.
        (Rules.md 5.6.3.)
        """
        valid = True
        if self.mode != "vector_database":
            return valid
        if self.division != "closed":
            return valid

        # On-disk directory name is the UPPERCASE token; compare directly.
        dir_name = os.path.basename(self.path.rstrip(os.sep))
        if dir_name not in VDB_INDEX_TYPES_CLOSED:
            self.log_violation(
                "5.6.3", "vdbClosedIndexTypes", self.path,
                "vdbClosedIndexTypes: directory name %r is not a CLOSED index "
                "type (allowed: %s)",
                dir_name, list(VDB_INDEX_TYPES_CLOSED),
            )
            valid = False
            return valid
        token = dir_name

        any_run = False
        for summary, metadata, ts in self._iter_run_files():
            any_run = True
            if summary is None:
                continue
            run_idx = summary.get("index_type")
            if run_idx is None:
                self.log_violation(
                    "5.6.3", "vdbClosedIndexTypes", self.path,
                    "vdbClosedIndexTypes: summary.json at %s/%s missing index_type",
                    self.path, ts,
                )
                valid = False
                continue
            if run_idx != token:
                self.log_violation(
                    "5.6.3", "vdbClosedIndexTypes", self.path,
                    "vdbClosedIndexTypes: directory %r expects index_type %r "
                    "but summary.json reports %r",
                    dir_name, token, run_idx,
                )
                valid = False

        if not any_run:
            # On-disk check has already run; loader gap only affects the per-run
            # comparison. Surface the gap so the rule's grep-visible signal is
            # consistent with the rest.
            self._vdb_loader_gap_warning("5.6.3", "vdbClosedIndexTypes")

        return valid

    @rule("5.6.4", "vdbClosedSubmissionParameters")
    def vdb_closed_submission_parameters(self):
        """For CLOSED, verify only allowed parameters are modified.
        (Rules.md 5.6.4)
        """
        valid = True
        if self.mode != "vector_database":
            return valid
        if self.division != "closed":
            return valid

        any_run = False
        for summary, metadata, ts in self._iter_run_files():
            any_run = True
            if metadata is None:
                self.log.debug(
                    "[5.6.4] %s/%s: skipping (metadata not loaded)",
                    self.path, ts,
                )
                continue
            params_dict = metadata.get("params_dict", {}) or {}
            for param_key in params_dict.keys():
                if param_key not in _CLOSED_ALLOWED_PARAMS:
                    self.log_violation(
                        "5.6.4", "vdbClosedSubmissionParameters", self.path,
                        "CLOSED vdb submission modifies disallowed parameter: %s",
                        param_key,
                    )
                    valid = False

        if not any_run:
            self._vdb_loader_gap_warning("5.6.4", "vdbClosedSubmissionParameters")

        return valid

    @rule("5.6.5", "vdbOpenSubmissionParameters")
    def vdb_open_submission_parameters(self):
        """For OPEN, verify only allowed parameters are modified.
        (Rules.md 5.6.5)

        OPEN extends the CLOSED allowlist with database.host / database.port.
        Backend-specific parameters for non-Milvus backends are NOT
        enumerable up-front; for those backends the strict allowlist is
        relaxed and a single warn_violation is emitted per leaf so the
        relaxation is grep-visible.
        """
        valid = True
        if self.mode != "vector_database":
            return valid
        if self.division != "open":
            return valid

        allowed_params = _CLOSED_ALLOWED_PARAMS | _OPEN_EXTRA_ALLOWED_PARAMS

        any_run = False
        for summary, metadata, ts in self._iter_run_files():
            any_run = True
            if metadata is None:
                self.log.debug(
                    "[5.6.5] %s/%s: skipping (metadata not loaded)",
                    self.path, ts,
                )
                continue
            # Determine backend from this run's summary so non-Milvus
            # backends are exempted from the strict allowlist (Rules.md
            # §5.6.5: "any index types, metrics, and parameters native to
            # a non-Milvus backend").
            backend = None
            if summary is not None:
                db_block = summary.get("database")
                if isinstance(db_block, dict):
                    backend = db_block.get("database")
            if backend not in (None, "milvus"):
                self.warn_violation(
                    "5.6.5", "vdbOpenSubmissionParameters", self.path,
                    "OPEN vdb submission uses non-Milvus backend %r at %s/%s; "
                    "backend-specific parameter validation is permitted but "
                    "not enforced — strict allowlist relaxed for this leaf",
                    backend, self.path, ts,
                )
                continue
            params_dict = metadata.get("params_dict", {}) or {}
            for param_key in params_dict.keys():
                if param_key not in allowed_params:
                    self.log_violation(
                        "5.6.5", "vdbOpenSubmissionParameters", self.path,
                        "OPEN vdb submission modifies disallowed parameter: %s",
                        param_key,
                    )
                    valid = False

        if not any_run:
            self._vdb_loader_gap_warning("5.6.5", "vdbOpenSubmissionParameters")

        return valid
