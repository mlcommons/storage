"""
Pytest fixture factory for SubmissionStructureCheck tests.

Provides:
  - ``MockLogger`` — capture-mode logger that stores ``error()``, ``warning()``,
    and ``info()`` calls as formatted strings.
  - ``mock_logger`` fixture — yields a fresh ``MockLogger`` per test.
  - ``build_submission(tmp_path, **overrides)`` — factory that builds a
    minimal but valid closed-only submission tree under ``tmp_path``.

Run with:
    pytest mlpstorage_py/tests/ -v
"""

import copy
import datetime
import json
import os
import shutil
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# MockLogger — capture-mode logger
# ---------------------------------------------------------------------------

class MockLogger:
    """Capture-mode mock logger for STRUCT-* and Phase-2 rule-check tests.

    Stores ``error()``, ``warning()``, and ``info()`` calls as fully-formatted
    strings (``msg % args`` applied) so tests can assert on the locked
    ``[<rule_id> <rule_name>] ...`` prefix without pytest caplog plumbing.

    Phase 2 Plan 02-04 extension: ``info()`` is no longer a no-op — it appends
    to ``self.infos``. This enables CHKPT-03 happy-path assertions and the
    CHKPT-05 deferred-followup pinning test.

    All other logging methods (``debug``, ``verbose``, ``verboser``,
    ``ridiculous``) remain no-ops.
    """

    def __init__(self):
        self.errors = []    # list[str] — formatted error messages
        self.warnings = []  # list[str] — formatted warning messages
        self.infos = []     # list[str] — formatted info messages (NEW Plan 02-04)

    def error(self, msg, *args, **kwargs):
        # **kwargs swallows exc_info=True etc. so the mock stays compatible
        # with logging callers that ask for traceback rendering.
        self.errors.append(msg % args if args else msg)

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg % args if args else msg)

    def debug(self, msg, *args, **kwargs):
        pass

    def verbose(self, msg, *args, **kwargs):
        pass

    def verboser(self, msg, *args):
        pass

    def ridiculous(self, msg, *args):
        pass


@pytest.fixture
def mock_logger():
    """Return a fresh MockLogger per test."""
    return MockLogger()


# ---------------------------------------------------------------------------
# Default submission tree constants
# ---------------------------------------------------------------------------

_SUBMITTER = "Acme"
_SYSNAME = "acme-storage-v1"

# Six run timestamps (1 warm-up + 5 measured per Rules.md 2.1.17)
_RUN_TIMESTAMPS = [
    "20250111_140001",
    "20250111_150001",
    "20250111_160001",
    "20250111_170001",
    "20250111_180001",
    "20250111_190001",
]
# One datagen timestamp
_DATAGEN_TIMESTAMPS = ["20250111_130000"]

# Two checkpointing timestamps — Rules.md 2.1.23 in conjunction with 4.7.1
# allows 1 or 2 invocations (one timestamp dir per invocation). The fixture
# uses the two-invocation shape (write phase + read phase) so split-mode
# kwargs like ``chkpt_split_mode`` can exercise the pairing helpers.
_CHKPT_TIMESTAMPS = [
    "20250112_100001",
    "20250112_110001",
]

# Stable 3-file code/ tree content (binary-stable, deterministic)
_CODE_FILES = {
    "mod.py": b"# mod\ndef hello():\n    return 'hello'\n",
    "helper.py": b"# helper\ndef util():\n    return 42\n",
    "README.md": b"# Submission Code\n\nThis is the reference implementation.\n",
}

# Default summary.json content per run timestamp (feeds STRUCT-09)
_DEFAULT_SUMMARY = {
    "num_hosts": 2,
    "host_memory_GB": [256, 256],
}

# ---------------------------------------------------------------------------
# Phase 2 Plan 02-04: Per-timestamp metadata dict template
# ---------------------------------------------------------------------------

_DEFAULT_METADATA = {
    "benchmark_type": "training",       # overridden to "checkpointing" for chkpt timestamps
    "model": "unet3d",                  # overridden to llama3-8b/etc. for chkpt
    "verification": "closed",
    "args": {
        "num_processes": 8,
        "data_dir": "/data",
        "results_dir": "/results",
        "hosts": ["host1"],
        "checkpoint_folder": "/chkpts",  # used for chkpt only
        "model": "unet3d",               # overridden for chkpt
        "num_checkpoints_write": 10,
        "num_checkpoints_read": 10,
    },
    "combined_params": {},
    "params_dict": {},
}

# ---------------------------------------------------------------------------
# Phase 2 Plan 02-04: Example df block snippets for TRAIN-02 / CHKPT-06 tests
# ---------------------------------------------------------------------------

_MOCK_DF_OUTPUT_DIFFERENT_MOUNTS = """\
Some preamble line.
Filesystem     1K-blocks  Used  Available  Use%  Mounted on
/dev/sda1      1000       500   500        50%   /data
/dev/sda2      1000       500   500        50%   /results
/dev/sda3      1000       500   500        50%   /

Other text after.
"""

_MOCK_DF_OUTPUT_SAME_MOUNT = """\
Filesystem     1K-blocks  Used  Available  Use%  Mounted on
/dev/sda1      1000       500   500        50%   /shared

"""


def _build_system_yaml(
    submission_name: str,
    multi_host: bool = True,
    benchmark_api: str = "file",
    remap_time_seconds: int | None = None,
    simultaneous_flags: dict | None = None,
) -> dict:
    """Build a schema-valid system YAML dict for the given submission_name.

    Uses ``deployment: cloud`` + ``storage_location: local`` to minimise
    required fields (cloud drops rack_units/power requirements; local drops
    the networking requirement on nodes).  Both are valid ``DeploymentMode``
    and ``StorageLocation`` enum values in schema_validator.py.

    Phase 2 Plan 02-04 extensions:
    - ``benchmark_api``: sets ``architecture.benchmark_API`` (default 'file').
    - ``remap_time_seconds``: sets ``capabilities.remap_time_in_seconds``.
    - ``simultaneous_flags``: dict of capability field overrides
      (simultaneous_write, simultaneous_read, multi_host).

    Bug fix (Phase 2 Plan 02-02): the original dict used ``power_capacity_watts``
    (not a valid ``PowerSupply`` field) and omitted the required ``inlet_voltage``
    and ``nameplate_power_watts`` fields.  Also, ``deployment: onprem`` requires
    ``total_rack_units`` and ``rack_units`` / ``power`` on all nodes; switching to
    ``cloud`` removes those requirements and keeps the fixture minimal.
    """
    caps = {
        "multi_host": multi_host,
        "simultaneous_write": True,
        "simultaneous_read": True,
        "remap_time_in_seconds": 0,
    }
    # Apply remap_time override
    if remap_time_seconds is not None:
        caps["remap_time_in_seconds"] = remap_time_seconds
    # Apply simultaneous_flags overrides
    if simultaneous_flags:
        for k, v in simultaneous_flags.items():
            caps[k] = v

    return {
        "system_under_test": {
            "solution": {
                "submission_name": submission_name,
                "friendly_description": "Test NAS system",
                "architecture": {
                    "storage_location": "local",
                    "benchmark_API": benchmark_api,
                    "product_API": "file",
                    "client_footprint": "open_source",
                    "client_installation": "in_box",
                },
                "capabilities": caps,
            },
            "deployment": "cloud",
            "clients": [
                {
                    "friendly_description": "Benchmark client",
                    "quantity": 2,
                    "chassis": {
                        "model_name": "TestServer-A",
                        "cpu_model": "Xeon Gold 6338",
                        "cpu_qty": 2,
                        "cpu_cores": 64,
                        "memory_capacity": 256,
                    },
                    "operating_system": {
                        "name": "RHEL",
                        "version": "9.2",
                    },
                }
            ],
        }
    }


# ---------------------------------------------------------------------------
# build_submission — fixture factory
# ---------------------------------------------------------------------------

def build_submission(tmp_path, **overrides) -> Path:
    """Build a minimal valid MLPerf Storage submission tree under *tmp_path*.

    Returns the root path (``tmp_path`` itself).  Instantiate
    ``SubmissionStructureCheck(log, config, str(tmp_path))`` to walk from
    there.

    Default tree (closed-only, one system, one training workload, one
    checkpointing workload):

    .. code-block:: text

        tmp_path/
          closed/
            Acme/
              code/
                mod.py
                helper.py
                README.md
              systems/
                acme-storage-v1.yaml   (schema-valid)
                acme-storage-v1.pdf    (1-byte placeholder)
              results/
                acme-storage-v1/
                  training/
                    unet3d/
                      datagen/
                        20250111_130000/
                          metadata.json     (NEW Plan 02-04)
                      run/
                        20250111_140001/   (+ summary.json + metadata.json)
                        ... (6 total)
                  checkpointing/
                    llama3-8b/
                      20250112_100001/   (+ summary.json + metadata.json)
                      20250112_110001/   (+ summary.json + metadata.json)

    Mutation kwargs (sealed — unknown kwargs raise ``TypeError``):

    Phase 1 kwargs (unchanged):
    * ``submitter_name_with_space`` (bool)  — name becomes "Acme Storage"
    * ``top_level_capitalcase`` (bool)      — "closed" → "CLOSED"
    * ``extra_top_level`` (str)             — adds an extra top-level dir
    * ``no_top_level_dirs`` (bool)          — removes closed/ entirely
    * ``open_mismatches_closed`` (bool)     — adds open/ missing one subdir
    * ``wrong_submitter_in_closed`` (bool)  — closed/OtherAcme/ instead
    * ``multiple_submitters_in_closed`` (bool) — two submitter dirs under closed/
    * ``missing_required_subdir`` (str)     — removes code/results/systems
    * ``extra_submitter_subdir`` (str)      — adds a stray dir under submitter
    * ``mutate_code`` (bool)                — adds extra file → hash differs
    * ``set_reference_checksum`` (str)      — unused in tree; caller passes to Config
    * ``code_with_symlink`` (bool)          — adds a symlink in code/
    * ``code_with_pycache`` (bool)          — adds code/pkg/__pycache__/mod.pyc
    * ``unpaired_yaml`` (bool)              — systems/ yaml without pdf
    * ``extra_systems_file`` (str)          — adds a stray file in systems/
    * ``unpaired_results_system`` (bool)    — adds results/no-yaml-for-this/
    * ``missing_systems_pdf`` (bool)        — drops systems/<sysname>.pdf
    * ``submission_name_mismatch`` (bool)   — YAML submission_name ≠ <name>
    * ``num_hosts_mismatch`` (bool)         — summary.json num_hosts mismatch
    * ``memory_mismatch`` (bool)            — summary.json host_memory_GB mismatch
    * ``multi_host_capability_inconsistent`` (bool) — multi_host=False + num_hosts>1
    * ``missing_summary_field`` (str)       — drops a field from summary.json
    * ``extra_workload_category`` (str)     — adds a stray workload category dir
    * ``wrong_training_workload`` (str)     — adds invalid training workload dir
    * ``wrong_training_phase`` (str)        — adds invalid training phase dir
    * ``datagen_timestamps`` (int)          — overrides datagen timestamp count
    * ``bad_datagen_timestamp_format`` (bool) — uses non-timestamp datagen dir name
    * ``wrong_checkpointing_workload`` (str) — adds invalid checkpointing workload

    Phase 2 Plan 02-02 kwargs:
    * ``system_yaml_bad_capabilities`` (dict | None) — Phase 2: perturb capabilities
      block.  Dict keys overwrite matching capability fields; special key
      ``"remove"`` is a list of field names to drop (→ missing-required-field
      schema error); ``"add"`` is a dict of extra key/values to inject.
    * ``system_yaml_rule13_violation`` (bool) — Phase 2: set capabilities to
      ``simultaneous_write=True, simultaneous_read=True, remap_time_in_seconds=5``
      (Rule-13 cross-field violation, triggers CHKPT-04 schema error tagged
      ``[4.7.3 checkpointRemappingTimeReporting]``).
    * ``system_yaml_bad_deployment`` (int | str | None) — Phase 2 Resolution A:
      set ``system_under_test.deployment`` to a non-``DeploymentMode`` value (e.g.
      integer ``12345``).  Pydantic v2 emits a ValidationError at loc
      ``"system_under_test -> deployment"`` which is NOT in ``SCHEMA_ERROR_RULE_MAP``,
      so the violation falls through to the ``("2.1.7", "systemsDirectoryFiles")``
      default (D-A2 fallback test driver).

    Phase 2 Plan 02-04 kwargs (supports TRAIN-01, TRAIN-02, CHKPT-01..06):
    * ``benchmark_api`` (str, default "file") — D-B7: sets
      ``architecture.benchmark_API`` in system YAML.
    * ``run_logfile_df_block`` (str | None) — TRAIN-02: full content written to
      ``training_run.stdout.log`` in every run timestamp directory.
    * ``chkpt_logfile_df_block`` (str | None) — CHKPT-06: full content written to
      ``checkpointing_run.stdout.log`` in every chkpt timestamp directory.
    * ``run_metadata_hosts`` (list | None) — TRAIN-01: sets ``args.hosts`` in
      each run timestamp metadata.json.
    * ``chkpt_summary_timestamps`` (bool) — CHKPT-02/03: when True, each
      checkpointing summary.json gets ``start``, ``end``, ``start_time``,
      ``end_time`` fields (both key spellings written for defensive-read compat).
    * ``chkpt_split_mode`` (bool) — CHKPT-02/03/04: when True, first half of
      checkpoint timestamps get ``num_checkpoints_write=10, num_checkpoints_read=0``
      (write-only) and second half get ``num_checkpoints_write=0,
      num_checkpoints_read=10`` (read-only) — pairs them per
      ``_pair_checkpoint_runs`` semantics.
    * ``chkpt_cache_flush_gap_seconds`` (int, default 25) — CHKPT-02/04: when
      ``chkpt_split_mode=True`` AND ``chkpt_summary_timestamps=True``, sets the
      gap between a write run's ``end`` and the following read run's ``start`` to
      this many seconds.
    * ``chkpt_model`` (str, default "llama3-8b") — CHKPT-01: model directory name
      under ``checkpointing/``.
    * ``chkpt_open_num_processes`` (int | None) — CHKPT-01: when non-None, sets
      ``verification="open"`` and ``args.num_processes=<value>`` in each
      checkpoint metadata.json.
    * ``chkpt_closed_num_processes`` (int | None) — CHKPT-01/BUG-03: when
      non-None, sets ``verification="closed"`` and ``args.num_processes=<value>``
      (for CHKPT-01 closed-branch / BUG-03 regression coverage).
    * ``chkpt_remap_time_seconds`` (int | None) — CHKPT-04: sets
      ``capabilities.remap_time_in_seconds`` in system YAML (None → no field
      written, SystemYamlSchemaCheck owns it; 0 → present but no remap).
    * ``chkpt_simultaneous_flags`` (dict | None) — CHKPT-05: overrides any of
      ``simultaneous_write``, ``simultaneous_read``, ``multi_host`` in system YAML
      capabilities block.
    * ``chkpt_checkpoint_folder`` (str | None) — CHKPT-06: sets
      ``args.checkpoint_folder`` in each checkpoint metadata.json.
    * ``chkpt_results_dir`` (str | None) — CHKPT-06: sets ``args.results_dir``
      in each checkpoint metadata.json.
    * ``run_data_dir`` (str | None) — TRAIN-02: sets ``args.data_dir`` in each
      run metadata.json.
    * ``run_results_dir`` (str | None) — TRAIN-02: sets ``args.results_dir`` in
      each run metadata.json.
    """
    # -----------------------------------------------------------------------
    # Pop all known overrides before the sealed-enum guard runs
    # -----------------------------------------------------------------------
    submitter_name_with_space = overrides.pop("submitter_name_with_space", False)
    top_level_capitalcase = overrides.pop("top_level_capitalcase", False)
    extra_top_level = overrides.pop("extra_top_level", None)
    no_top_level_dirs = overrides.pop("no_top_level_dirs", False)
    open_mismatches_closed = overrides.pop("open_mismatches_closed", False)
    wrong_submitter_in_closed = overrides.pop("wrong_submitter_in_closed", False)
    multiple_submitters_in_closed = overrides.pop("multiple_submitters_in_closed", False)
    missing_required_subdir = overrides.pop("missing_required_subdir", None)
    extra_submitter_subdir = overrides.pop("extra_submitter_subdir", None)
    mutate_code = overrides.pop("mutate_code", False)
    set_reference_checksum = overrides.pop("set_reference_checksum", None)  # caller uses this
    code_with_symlink = overrides.pop("code_with_symlink", False)
    code_with_pycache = overrides.pop("code_with_pycache", False)
    unpaired_yaml = overrides.pop("unpaired_yaml", False)
    extra_systems_file = overrides.pop("extra_systems_file", None)
    unpaired_results_system = overrides.pop("unpaired_results_system", False)
    missing_systems_pdf = overrides.pop("missing_systems_pdf", False)
    submission_name_mismatch = overrides.pop("submission_name_mismatch", False)
    num_hosts_mismatch = overrides.pop("num_hosts_mismatch", False)
    memory_mismatch = overrides.pop("memory_mismatch", False)
    multi_host_capability_inconsistent = overrides.pop("multi_host_capability_inconsistent", False)
    missing_summary_field = overrides.pop("missing_summary_field", None)
    extra_workload_category = overrides.pop("extra_workload_category", None)
    wrong_training_workload = overrides.pop("wrong_training_workload", None)
    wrong_training_phase = overrides.pop("wrong_training_phase", None)
    datagen_timestamps_count = overrides.pop("datagen_timestamps", None)
    bad_datagen_timestamp_format = overrides.pop("bad_datagen_timestamp_format", False)
    wrong_checkpointing_workload = overrides.pop("wrong_checkpointing_workload", None)
    # Phase 2 Plan 02-02: system YAML mutation kwargs for SystemYamlSchemaCheck tests.
    system_yaml_bad_capabilities = overrides.pop("system_yaml_bad_capabilities", None)
    system_yaml_rule13_violation = overrides.pop("system_yaml_rule13_violation", False)
    system_yaml_bad_deployment = overrides.pop("system_yaml_bad_deployment", None)
    # Phase 2 Plan 02-04: new kwargs for TRAIN-01, TRAIN-02, CHKPT-01..06 tests.
    benchmark_api = overrides.pop("benchmark_api", "file")
    run_logfile_df_block = overrides.pop("run_logfile_df_block", None)
    chkpt_logfile_df_block = overrides.pop("chkpt_logfile_df_block", None)
    run_metadata_hosts = overrides.pop("run_metadata_hosts", None)
    chkpt_summary_timestamps = overrides.pop("chkpt_summary_timestamps", False)
    chkpt_split_mode = overrides.pop("chkpt_split_mode", False)
    chkpt_cache_flush_gap_seconds = overrides.pop("chkpt_cache_flush_gap_seconds", 25)
    chkpt_model = overrides.pop("chkpt_model", "llama3-8b")
    chkpt_open_num_processes = overrides.pop("chkpt_open_num_processes", None)
    chkpt_closed_num_processes = overrides.pop("chkpt_closed_num_processes", None)
    chkpt_remap_time_seconds = overrides.pop("chkpt_remap_time_seconds", None)
    chkpt_simultaneous_flags = overrides.pop("chkpt_simultaneous_flags", None)
    chkpt_checkpoint_folder = overrides.pop("chkpt_checkpoint_folder", None)
    chkpt_results_dir = overrides.pop("chkpt_results_dir", None)
    run_data_dir = overrides.pop("run_data_dir", None)
    run_results_dir = overrides.pop("run_results_dir", None)

    # Sealed-enum guard — any leftover key is unknown
    if overrides:
        raise TypeError(f"unknown override: {sorted(overrides)}")

    # -----------------------------------------------------------------------
    # Determine submitter name
    # -----------------------------------------------------------------------
    submitter = _SUBMITTER
    if submitter_name_with_space:
        submitter = "Acme Storage"

    # -----------------------------------------------------------------------
    # Determine top-level division name
    # -----------------------------------------------------------------------
    division = "closed"
    if top_level_capitalcase:
        division = "CLOSED"

    # -----------------------------------------------------------------------
    # Build the tree
    # -----------------------------------------------------------------------
    # STRUCT-04 (D-277) requires the input root basename to equal the submitter
    # name. Nest the build inside tmp_path/Acme/ so that
    # os.path.basename(root) == _SUBMITTER for the default tree.
    root = Path(tmp_path) / _SUBMITTER
    root.mkdir(parents=True, exist_ok=True)

    if not no_top_level_dirs:
        # Division directory
        div_path = root / division
        div_path.mkdir(parents=True)

        # Submitter directory inside division
        if wrong_submitter_in_closed:
            # Use a different submitter name inside closed/
            sub_path = div_path / "OtherAcme"
        else:
            sub_path = div_path / submitter
        sub_path.mkdir()

        if multiple_submitters_in_closed:
            (div_path / "AlsoAcme").mkdir()

        # Required subdirectories
        required_subdirs = {"code", "results", "systems"}
        if missing_required_subdir:
            required_subdirs.discard(missing_required_subdir)

        for sd in required_subdirs:
            (sub_path / sd).mkdir()

        if extra_submitter_subdir:
            (sub_path / extra_submitter_subdir).mkdir()

        code_path = sub_path / "code"
        results_path = sub_path / "results"
        systems_path = sub_path / "systems"

        # ---------------------------------------------------------------
        # code/ tree
        # ---------------------------------------------------------------
        if code_path.exists():
            for fname, content in _CODE_FILES.items():
                (code_path / fname).write_bytes(content)

            if mutate_code:
                (code_path / "extra_unexpected.py").write_bytes(b"# extra\n")

            if code_with_symlink:
                target = code_path / "mod.py"
                link = code_path / "link_to_mod.py"
                os.symlink(str(target), str(link))

            if code_with_pycache:
                pkg_dir = code_path / "pkg"
                pkg_dir.mkdir()
                pycache_dir = pkg_dir / "__pycache__"
                pycache_dir.mkdir()
                (pycache_dir / "mod.pyc").write_bytes(b"\x00\x00\x00\x00")

        # ---------------------------------------------------------------
        # systems/ directory
        # ---------------------------------------------------------------
        if systems_path.exists():
            # Determine system YAML content
            yaml_submission_name = _SYSNAME
            if submission_name_mismatch:
                yaml_submission_name = "wrong-name-here"

            multi_host_val = True
            if multi_host_capability_inconsistent:
                multi_host_val = False

            # Phase 2 Plan 02-04: resolve effective remap_time for system YAML.
            # chkpt_remap_time_seconds=None → keep default 0 in _build_system_yaml.
            # chkpt_remap_time_seconds=N → set to N.
            effective_remap_time = chkpt_remap_time_seconds  # may be None → use default

            sys_yaml_dict = _build_system_yaml(
                yaml_submission_name,
                multi_host=multi_host_val,
                benchmark_api=benchmark_api,
                remap_time_seconds=effective_remap_time,
                simultaneous_flags=chkpt_simultaneous_flags,
            )

            # Phase 2 Plan 02-02: apply system_yaml_* mutation kwargs.
            if system_yaml_rule13_violation:
                # Trigger Capabilities.check_remap_time (Rule 13):
                # both simultaneous flags True + non-zero remap_time.
                caps = sys_yaml_dict["system_under_test"]["solution"]["capabilities"]
                caps["simultaneous_write"] = True
                caps["simultaneous_read"] = True
                caps["remap_time_in_seconds"] = 5

            if system_yaml_bad_capabilities is not None:
                caps = sys_yaml_dict["system_under_test"]["solution"]["capabilities"]
                # "remove" → list of capability keys to drop (missing-required-field)
                for key in system_yaml_bad_capabilities.get("remove", []):
                    caps.pop(key, None)
                # "add" → dict of extra keys to inject
                for key, val in system_yaml_bad_capabilities.get("add", {}).items():
                    caps[key] = val
                # All other keys → overwrite the matching capability field value
                for key, val in system_yaml_bad_capabilities.items():
                    if key not in ("remove", "add"):
                        caps[key] = val

            if system_yaml_bad_deployment is not None:
                # Resolution A: set deployment to a non-DeploymentMode value so Pydantic
                # emits a ValidationError at loc "system_under_test -> deployment"
                # (which is NOT in SCHEMA_ERROR_RULE_MAP → fallback 2.1.7).
                sys_yaml_dict["system_under_test"]["deployment"] = system_yaml_bad_deployment

            yaml_content = yaml.dump(sys_yaml_dict, default_flow_style=False)

            if not unpaired_yaml:
                (systems_path / f"{_SYSNAME}.yaml").write_text(yaml_content, encoding="utf-8")
            else:
                # No .pdf — yaml without matching pdf
                (systems_path / f"{_SYSNAME}.yaml").write_text(yaml_content, encoding="utf-8")
                # deliberately do NOT write the pdf

            if not missing_systems_pdf and not unpaired_yaml:
                # Write PDF placeholder
                (systems_path / f"{_SYSNAME}.pdf").write_bytes(b"%PDF")

            if extra_systems_file:
                (systems_path / extra_systems_file).write_text("stray content\n")

        # ---------------------------------------------------------------
        # results/ subtree
        # ---------------------------------------------------------------
        if results_path.exists():
            sys_results = results_path / _SYSNAME
            sys_results.mkdir()

            # Determine summary.json content
            base_summary = dict(_DEFAULT_SUMMARY)
            if num_hosts_mismatch:
                base_summary["num_hosts"] = 3  # system YAML has quantity:2 → mismatch
            if memory_mismatch:
                base_summary["host_memory_GB"] = [128, 128]  # system YAML has 256
            if missing_summary_field:
                base_summary.pop(missing_summary_field, None)
            if multi_host_capability_inconsistent:
                base_summary["num_hosts"] = 2  # capabilities.multi_host=False but num_hosts=2

            # training/unet3d
            training_path = sys_results / "training"
            training_path.mkdir()

            unet3d_path = training_path / "unet3d"
            unet3d_path.mkdir()

            # datagen timestamps
            datagen_dir = unet3d_path / "datagen"
            datagen_dir.mkdir()
            if bad_datagen_timestamp_format:
                dg_timestamps = ["not-a-timestamp"]
            elif datagen_timestamps_count is not None:
                dg_timestamps = [
                    f"2025011{i}_130000" for i in range(1, datagen_timestamps_count + 1)
                ]
            else:
                dg_timestamps = _DATAGEN_TIMESTAMPS
            for ts in dg_timestamps:
                ts_dir = datagen_dir / ts
                ts_dir.mkdir()
                # Phase 2 Plan 02-04: write per-timestamp metadata.json for datagen dirs
                datagen_meta = copy.deepcopy(_DEFAULT_METADATA)
                datagen_meta["benchmark_type"] = "training"
                datagen_meta["model"] = "unet3d"
                datagen_meta["args"]["model"] = "unet3d"
                (ts_dir / "metadata.json").write_text(
                    json.dumps(datagen_meta), encoding="utf-8"
                )

            # run timestamps
            run_dir = unet3d_path / "run"
            run_dir.mkdir()
            for ts in _RUN_TIMESTAMPS:
                ts_dir = run_dir / ts
                ts_dir.mkdir()
                (ts_dir / "summary.json").write_text(
                    json.dumps(base_summary), encoding="utf-8"
                )
                # Phase 2 Plan 02-04: write per-timestamp metadata.json for run dirs
                run_meta = copy.deepcopy(_DEFAULT_METADATA)
                run_meta["benchmark_type"] = "training"
                run_meta["model"] = "unet3d"
                run_meta["args"]["model"] = "unet3d"
                # Apply TRAIN-01 host override
                if run_metadata_hosts is not None:
                    run_meta["args"]["hosts"] = run_metadata_hosts
                # Apply TRAIN-02 path overrides
                if run_data_dir is not None:
                    run_meta["args"]["data_dir"] = run_data_dir
                if run_results_dir is not None:
                    run_meta["args"]["results_dir"] = run_results_dir
                (ts_dir / "metadata.json").write_text(
                    json.dumps(run_meta), encoding="utf-8"
                )
                # Phase 2 Plan 02-04: write training_run.stdout.log when df block provided
                if run_logfile_df_block is not None:
                    (ts_dir / "training_run.stdout.log").write_text(
                        run_logfile_df_block, encoding="utf-8"
                    )

            if wrong_training_workload:
                (training_path / wrong_training_workload).mkdir()

            if wrong_training_phase:
                (unet3d_path / wrong_training_phase).mkdir()

            if extra_workload_category:
                (sys_results / extra_workload_category).mkdir()

            # ---------------------------------------------------------------
            # checkpointing/<model> timestamps
            # ---------------------------------------------------------------
            chkpt_path = sys_results / "checkpointing"
            chkpt_path.mkdir()

            # Use chkpt_model for the directory name (default llama3-8b)
            llama_path = chkpt_path / chkpt_model
            llama_path.mkdir()

            # Determine verification and num_processes for checkpoint metadata
            if chkpt_open_num_processes is not None:
                chkpt_verification = "open"
                chkpt_num_processes = chkpt_open_num_processes
            elif chkpt_closed_num_processes is not None:
                chkpt_verification = "closed"
                chkpt_num_processes = chkpt_closed_num_processes
            else:
                chkpt_verification = "closed"
                chkpt_num_processes = 8

            # Derive model name string for metadata (strip "llama3-" prefix if present)
            # For the model metadata arg, use chkpt_model as-is (e.g., "llama3-8b")
            chkpt_meta_model = chkpt_model

            # Split mode: first half are write-only, second half are read-only
            half = len(_CHKPT_TIMESTAMPS) // 2
            write_timestamps = _CHKPT_TIMESTAMPS[:half] if chkpt_split_mode else []
            read_timestamps = _CHKPT_TIMESTAMPS[half:] if chkpt_split_mode else []

            # Base timestamps for summary.json timestamps (2025-01-11T14:30:22)
            # Each write run starts at base + 10*i minutes; runs for 5 minutes.
            # Read run starts gap_seconds after write run ends.
            _BASE_DT = datetime.datetime(2025, 1, 11, 14, 30, 22)

            for i, ts in enumerate(_CHKPT_TIMESTAMPS):
                ts_dir = llama_path / ts
                ts_dir.mkdir()

                # Build summary.json
                chkpt_summary = dict(_DEFAULT_SUMMARY)
                if chkpt_summary_timestamps:
                    if chkpt_split_mode:
                        if ts in write_timestamps:
                            wi = write_timestamps.index(ts)
                            write_start = _BASE_DT + datetime.timedelta(minutes=10 * wi)
                            write_end = write_start + datetime.timedelta(minutes=5)
                            # The corresponding read run starts gap_seconds after write_end
                            read_start = write_end + datetime.timedelta(seconds=chkpt_cache_flush_gap_seconds)
                            read_end = read_start + datetime.timedelta(minutes=5)
                            chkpt_summary["start"] = write_start.isoformat()
                            chkpt_summary["end"] = write_end.isoformat()
                            chkpt_summary["start_time"] = write_start.isoformat()
                            chkpt_summary["end_time"] = write_end.isoformat()
                        else:
                            # Read run
                            ri = read_timestamps.index(ts)
                            # Corresponding write run index is ri
                            write_start = _BASE_DT + datetime.timedelta(minutes=10 * ri)
                            write_end = write_start + datetime.timedelta(minutes=5)
                            read_start = write_end + datetime.timedelta(seconds=chkpt_cache_flush_gap_seconds)
                            read_end = read_start + datetime.timedelta(minutes=5)
                            chkpt_summary["start"] = read_start.isoformat()
                            chkpt_summary["end"] = read_end.isoformat()
                            chkpt_summary["start_time"] = read_start.isoformat()
                            chkpt_summary["end_time"] = read_end.isoformat()
                    else:
                        # Non-split mode: simple timestamps per index
                        ts_start = _BASE_DT + datetime.timedelta(minutes=10 * i)
                        ts_end = ts_start + datetime.timedelta(minutes=5)
                        chkpt_summary["start"] = ts_start.isoformat()
                        chkpt_summary["end"] = ts_end.isoformat()
                        chkpt_summary["start_time"] = ts_start.isoformat()
                        chkpt_summary["end_time"] = ts_end.isoformat()

                (ts_dir / "summary.json").write_text(
                    json.dumps(chkpt_summary), encoding="utf-8"
                )

                # Build metadata.json for this checkpoint timestamp
                chkpt_meta = copy.deepcopy(_DEFAULT_METADATA)
                chkpt_meta["benchmark_type"] = "checkpointing"
                chkpt_meta["model"] = chkpt_meta_model
                chkpt_meta["verification"] = chkpt_verification
                chkpt_meta["args"]["model"] = chkpt_meta_model
                chkpt_meta["args"]["num_processes"] = chkpt_num_processes
                # Apply checkpoint_folder override
                if chkpt_checkpoint_folder is not None:
                    chkpt_meta["args"]["checkpoint_folder"] = chkpt_checkpoint_folder
                # Apply chkpt_results_dir override
                if chkpt_results_dir is not None:
                    chkpt_meta["args"]["results_dir"] = chkpt_results_dir

                # Split mode: set num_checkpoints_write / num_checkpoints_read
                if chkpt_split_mode:
                    if ts in write_timestamps:
                        chkpt_meta["args"]["num_checkpoints_write"] = 10
                        chkpt_meta["args"]["num_checkpoints_read"] = 0
                    else:
                        chkpt_meta["args"]["num_checkpoints_write"] = 0
                        chkpt_meta["args"]["num_checkpoints_read"] = 10

                (ts_dir / "metadata.json").write_text(
                    json.dumps(chkpt_meta), encoding="utf-8"
                )

                # Phase 2 Plan 02-04: write checkpointing_run.stdout.log when df block provided
                if chkpt_logfile_df_block is not None:
                    (ts_dir / "checkpointing_run.stdout.log").write_text(
                        chkpt_logfile_df_block, encoding="utf-8"
                    )

            if wrong_checkpointing_workload:
                (chkpt_path / wrong_checkpointing_workload).mkdir()

            if unpaired_results_system:
                (results_path / "no-yaml-for-this").mkdir(parents=True)

    # -----------------------------------------------------------------------
    # open/ mirror (only when open_mismatches_closed=True)
    # -----------------------------------------------------------------------
    if open_mismatches_closed:
        open_div = root / "open"
        open_div.mkdir(parents=True)
        open_sub = open_div / submitter
        open_sub.mkdir()
        # Add some but not all required subdirs to trigger STRUCT-03
        (open_sub / "results").mkdir()
        # Deliberately omit code/ and systems/

    if extra_top_level:
        (root / extra_top_level).mkdir()

    return root
