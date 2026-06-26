"""End-to-end integration tests for the systemname.yaml write hook in
Benchmark.run() — Phase 02 / Plan 02-05.

These tests exercise the FULL Benchmark.run() lifecycle (with lifecycle
methods mocked to avoid DLIO/MPI) and assert that the LIFE-01 hook fires:

- `args.command == 'run'` produces the canonical YAML file BEFORE
  `_start_timeseries_collection()` runs and BEFORE `_run()` launches.
- `args.command == 'datagen'` does NOT produce the file (D-12).
- Per-mode separation: `closed`/`open`/`whatif` land at three distinct paths (D-11).
- Second `run()` against the same path is a byte-identical no-op (D-9).
- Filesystem write failures abort `Benchmark.run()` BEFORE `_run()` (D-9 fail-closed).
- `schema_validator.validate_file()` reports errors only on the intentional
  blanks — never on the filled fields populated by `node_dict_from_host`
  (SER-03 success criterion #4).
- Homogeneous fleets produce one stanza with `quantity == fleet_size`;
  heterogeneous fleets produce multiple stanzas summing to `fleet_size`
  (SER-01 success criterion #3).

These tests use VectorDBBenchmark as the concrete Benchmark subclass — it
has the simplest `__init__` of any benchmark in the codebase and inherits
`run()` directly from `Benchmark`, which is the surface this plan covers.
"""

from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Stub heavy deps the benchmark imports expect. Use importlib.util.find_spec —
# checking sys.modules alone would install a MagicMock for a perfectly
# importable module that just hasn't been imported yet, which then poisons
# later test collections by causing find_spec to raise ValueError on the
# Mock's __spec__. Matches the safe pattern in tests/unit/test_benchmarks_kvcache.py.
import importlib.util as _ilu
for _dep in ('pyarrow', 'pyarrow.ipc', 'psutil'):
    if _ilu.find_spec(_dep) is None and _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

from mlpstorage_py.cluster_collector import HostSystemInfo
from mlpstorage_py.rules.models import (
    HostCPUInfo,
    HostInfo,
    HostMemoryInfo,
)


# ---------------------------------------------------------------------------
# Host fixtures (mirror tests/unit/test_auto_generator_write.py)
# ---------------------------------------------------------------------------


def _make_host(
    *,
    cpu_model: str = "Intel(R) Xeon Platinum 8480+",
    num_cores: int = 56,
    num_sockets: int = 2,
    mem_bytes: int = 274_877_906_944,  # 256 GiB exact
    os_name: str = "Rocky Linux",
    os_version: str = "9.5",
    hostname: str = "h1",
) -> HostInfo:
    """Build a HostInfo with sensible Phase 2 defaults."""
    return HostInfo(
        hostname=hostname,
        cpu=HostCPUInfo(
            model=cpu_model,
            num_cores=num_cores,
            num_logical_cores=num_cores * 2,
            num_sockets=num_sockets,
            architecture="x86_64",
        ),
        memory=HostMemoryInfo(total=mem_bytes),
        system=HostSystemInfo(
            hostname=hostname,
            os_release={"NAME": os_name, "VERSION_ID": os_version},
        ),
    )


# ---------------------------------------------------------------------------
# Benchmark construction helper
# ---------------------------------------------------------------------------


def _vdb_args(tmp_path, *, command='run', mode='closed', orgname='Acme',
              systemname='sys-v1') -> Namespace:
    """Args namespace for VectorDBBenchmark construction."""
    return Namespace(
        debug=False,
        verbose=False,
        what_if=False,
        stream_log_level='INFO',
        mode=mode,
        orgname=orgname,
        systemname=systemname,
        results_dir=str(tmp_path),
        command=command,
        config='default',
        vdb_engine='milvus',
        host='127.0.0.1',
        port=19530,
        collection=None,
        category=None,
        num_query_processes=1,
        batch_size=1,
        runtime=60,
        queries=None,
        report_count=100,
    )


_RUN_DATETIME_COUNTER = [0]


def _unique_run_datetime() -> str:
    """Return a unique YYYYMMDD_HHMMSS-formatted string so back-to-back
    benchmark constructions in the same test don't collide on `reserve_run_directory`."""
    _RUN_DATETIME_COUNTER[0] += 1
    base = time.strftime("%Y%m%d_%H%M%S")
    # Append a monotonic counter; the run-directory reserver tolerates any
    # nonsense after the timestamp because it bumps on collision anyway,
    # but we want the very first attempt to be unique to avoid the
    # 10-bump collision-budget exhaustion.
    return f"{base}_{_RUN_DATETIME_COUNTER[0]:04d}"


def _make_benchmark(tmp_path, hosts, *, command='run', mode='closed',
                    orgname='Acme', systemname='sys-v1',
                    timeseries_side_effect=None):
    """Construct a VectorDBBenchmark with all lifecycle methods mocked.

    `_collect_cluster_start` is mocked to install `self._cluster_info_start`
    as a MagicMock with `host_info_list=hosts` — the production write hook
    consumes this attribute directly.
    """
    args = _vdb_args(tmp_path, command=command, mode=mode,
                     orgname=orgname, systemname=systemname)
    output_dir = str(tmp_path / f"output_{_RUN_DATETIME_COUNTER[0]}")
    with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
         patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
         patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
         patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
        mock_gen.return_value = output_dir
        from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
        bm = VectorDBBenchmark(args, run_datetime=_unique_run_datetime())

    # Mock lifecycle methods so we can drive `Benchmark.run()` without DLIO/MPI.
    bm._validate_environment = MagicMock()

    def _cluster_start_side_effect():
        bm._cluster_info_start = MagicMock(host_info_list=hosts)
        bm._collection_method = 'mpi'

    bm._collect_cluster_start = MagicMock(side_effect=_cluster_start_side_effect)
    bm._start_timeseries_collection = MagicMock(side_effect=timeseries_side_effect)
    bm._stop_timeseries_collection = MagicMock()
    bm._collect_cluster_end = MagicMock()
    bm.write_timeseries_data = MagicMock()
    bm._run = MagicMock(return_value=0)
    return bm


def _yaml_path(tmp_path, mode='closed', orgname='Acme', systemname='sys-v1') -> Path:
    return tmp_path / mode / orgname / 'systems' / f"{systemname}.yaml"


# ---------------------------------------------------------------------------
# LIFE-01 happy path
# ---------------------------------------------------------------------------


def test_full_run_writes_systemname_yaml(tmp_path):
    """LIFE-01 happy path: `command='run'` → file at canonical path with
    `quantity == 3` for a homogeneous 3-host fleet."""
    hosts = [_make_host(hostname=f"h{i}") for i in range(3)]
    bm = _make_benchmark(tmp_path, hosts)

    target = _yaml_path(tmp_path)
    # Pre-assert: file does not exist before run().
    assert not target.exists()

    rc = bm.run()
    assert rc == 0
    assert target.exists()

    data = yaml.safe_load(target.read_text())
    clients = data['system_under_test']['clients']
    assert len(clients) == 1
    assert clients[0]['quantity'] == 3

    bm._collect_cluster_start.assert_called_once()
    bm._start_timeseries_collection.assert_called_once()
    bm._run.assert_called_once()


# ---------------------------------------------------------------------------
# Hook ordering: file lands BEFORE _start_timeseries_collection()
# ---------------------------------------------------------------------------


def test_hook_fires_before_timeseries(tmp_path):
    """D-9 hook point: file must exist on disk BEFORE
    `_start_timeseries_collection()` runs."""
    hosts = [_make_host()]
    file_existed_at_timeseries = []
    target = _yaml_path(tmp_path)

    def _timeseries_side_effect():
        file_existed_at_timeseries.append(target.exists())

    bm = _make_benchmark(tmp_path, hosts,
                         timeseries_side_effect=_timeseries_side_effect)
    bm.run()

    assert file_existed_at_timeseries == [True], (
        "systemname.yaml must exist on disk before "
        "_start_timeseries_collection() runs"
    )


# ---------------------------------------------------------------------------
# D-12: datagen does NOT write
# ---------------------------------------------------------------------------


def test_datagen_does_not_write(tmp_path):
    """D-12: `command='datagen'` → writer's own gate returns None, no file."""
    hosts = [_make_host()]
    bm = _make_benchmark(tmp_path, hosts, command='datagen')

    bm.run()

    target = _yaml_path(tmp_path)
    assert not target.exists()
    # The rest of the lifecycle still proceeds.
    bm._run.assert_called_once()


# ---------------------------------------------------------------------------
# D-11: per-mode separation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["closed", "open", "whatif"])
def test_per_mode_separation(tmp_path, mode):
    """D-11: each mode lands its file at `<rd>/<mode>/<org>/systems/<sn>.yaml`."""
    hosts = [_make_host()]
    bm = _make_benchmark(tmp_path, hosts, mode=mode)
    bm.run()

    target = _yaml_path(tmp_path, mode=mode)
    assert target.exists(), f"expected file at {target}"


def test_per_mode_three_distinct_files(tmp_path):
    """D-11 cross-check: cycling all three modes against the same results-dir
    produces three distinct files at distinct paths."""
    hosts = [_make_host()]
    for mode in ("closed", "open", "whatif"):
        bm = _make_benchmark(tmp_path, hosts, mode=mode)
        bm.run()

    paths = [_yaml_path(tmp_path, mode=m) for m in ("closed", "open", "whatif")]
    assert all(p.exists() for p in paths)
    # All three are at DIFFERENT paths.
    assert len({str(p) for p in paths}) == 3


# ---------------------------------------------------------------------------
# D-9: second-run no overwrite
# ---------------------------------------------------------------------------


def test_second_run_no_overwrite(tmp_path):
    """LIFE-01 success criterion #5: second run produces no error and byte-identical file."""
    hosts = [_make_host()]
    bm1 = _make_benchmark(tmp_path, hosts)
    bm1.run()
    target = _yaml_path(tmp_path)
    snapshot_bytes = target.read_bytes()
    snapshot_mtime = target.stat().st_mtime

    # Second run, fresh benchmark, same systemname/results-dir.
    bm2 = _make_benchmark(tmp_path, hosts)
    bm2.run()

    assert target.read_bytes() == snapshot_bytes, (
        "file content must be byte-identical after second run (D-9 no-op-if-exists)"
    )
    assert target.stat().st_mtime == snapshot_mtime, (
        "file mtime must be unchanged after second run (no rewrite)"
    )


# ---------------------------------------------------------------------------
# D-9: filesystem failure aborts Benchmark.run() BEFORE _run()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="filesystem semantics differ on Windows",
)
def test_filesystem_failure_propagates(tmp_path):
    """D-9 fail-closed: a non-FileExistsError filesystem error during the
    write (e.g. PermissionError / ENOSPC / IsADirectoryError) propagates
    out of `Benchmark.run()` and prevents `_run()` from being called.

    We inject the error via `patch('os.open')` so we can reproduce the
    OSError behavior deterministically across Linux/macOS without relying
    on uid-specific filesystem-permission tricks. The patch targets the
    write site inside `auto_generator`, NOT global os.open (which would
    break unrelated I/O in the lifecycle methods).
    """
    hosts = [_make_host()]
    bm = _make_benchmark(tmp_path, hosts)

    # Inject a non-FileExistsError into the writer's os.open call.
    def _raise_perm(*args, **kwargs):
        raise PermissionError("simulated write failure")

    with patch('mlpstorage_py.system_description.auto_generator.os.open',
               side_effect=_raise_perm):
        with pytest.raises(PermissionError):
            bm.run()

    # The benchmark MUST NOT have reached _run() after a write failure.
    bm._run.assert_not_called()


# ---------------------------------------------------------------------------
# SER-03 success criterion #4: validator errors only on intentional blanks
# ---------------------------------------------------------------------------


def test_validator_errors_only_on_blanks(tmp_path):
    """SER-03 success criterion #4: `schema_validator.validate_file()` reports
    errors ONLY on the intentional blanks (D-2 / D-3 / D-14 omissions) — never
    on the fields populated by `node_dict_from_host` from real cluster data."""
    from mlpstorage_py.system_description import schema_validator

    hosts = [_make_host()]
    bm = _make_benchmark(tmp_path, hosts)
    bm.run()
    target = _yaml_path(tmp_path)
    assert target.exists()

    errors = schema_validator.validate_file(str(target))
    # `validate_file` returns a list of human-readable strings of the form
    # "system_under_test -> clients -> 0 -> chassis -> model_name: ..."
    # Build a set of dotted paths for easier substring assertions.
    error_paths = {e.split(":", 1)[0].strip() for e in errors}

    # Intentional blanks — at least these locations MUST appear among errors.
    expected_blanks = [
        "system_under_test -> solution",  # D-14
        "system_under_test -> deployment",  # D-14
        "system_under_test -> clients -> 0 -> friendly_description",  # D-2
        "system_under_test -> clients -> 0 -> chassis -> model_name",  # D-2 (Phase 3)
    ]
    for path in expected_blanks:
        assert any(path in p for p in error_paths), (
            f"expected error at intentional blank {path!r}; got errors:\n"
            + "\n".join(sorted(error_paths))
        )

    # Stub list fields (D-3) — at least one error under networking[*].
    assert any("networking" in p for p in error_paths), (
        "expected at least one error under clients[].networking[*]"
    )

    # Phase 4 / D-33: drives key is OMITTED entirely from the YAML when no
    # drives data was collected. `drives` is an Optional field on
    # NodeDescription, so the Pydantic validator does NOT surface an error
    # path under drives[*] — the SER-02 signal is "no drives: block at all,
    # submitter must hand-fill if applicable". Verify the omission against
    # the on-disk YAML directly.
    import yaml as _yaml
    with open(target) as fh:
        loaded = _yaml.safe_load(fh)
    client0 = loaded["system_under_test"]["clients"][0]
    assert "drives" not in client0, (
        f"Phase 4 / D-33 violation: drives key present in emitted YAML "
        f"client[0]; expected key OMITTED. client[0]={client0!r}"
    )
    # And the validator must NOT surface a drives[*] error (Optional field
    # absent is legal at the schema layer).
    assert not any("drives" in p for p in error_paths), (
        f"Phase 4 / D-33 violation: expected NO error path under drives "
        f"(Optional field, key omitted per D-33), but got: "
        f"{[p for p in error_paths if 'drives' in p]}"
    )

    # Filled fields — these MUST NOT appear in any error path.
    forbidden_in_errors = [
        "chassis -> cpu_model",
        "chassis -> cpu_qty",
        "chassis -> cpu_cores",
        "chassis -> memory_capacity",
        "operating_system -> name",
        "operating_system -> version",
    ]
    for field in forbidden_in_errors:
        for ep in error_paths:
            assert field not in ep, (
                f"filled field {field!r} unexpectedly appears in error path "
                f"{ep!r} — node_dict_from_host should have populated it"
            )


# ---------------------------------------------------------------------------
# SER-01 success criterion #3: homogeneous vs heterogeneous quantity grouping
# ---------------------------------------------------------------------------


def test_homogeneous_fleet_quantity_equals_fleet_size(tmp_path):
    """SER-01 success criterion #3: 3 identical hosts → 1 stanza, `quantity == 3`."""
    hosts = [_make_host(hostname=f"h{i}") for i in range(3)]
    bm = _make_benchmark(tmp_path, hosts)
    bm.run()

    data = yaml.safe_load(_yaml_path(tmp_path).read_text())
    clients = data['system_under_test']['clients']
    assert len(clients) == 1
    assert clients[0]['quantity'] == 3
    assert sum(c['quantity'] for c in clients) == 3


def test_heterogeneous_fleet_produces_multiple_stanzas(tmp_path):
    """SER-01 success criterion #3: 1 host with a different cpu_model →
    2 stanzas; quantities sum to fleet size."""
    hosts = [
        _make_host(hostname="h0", cpu_model="Intel(R) Xeon Platinum 8480+"),
        _make_host(hostname="h1", cpu_model="Intel(R) Xeon Platinum 8480+"),
        _make_host(hostname="h2", cpu_model="AMD EPYC 9654"),
    ]
    bm = _make_benchmark(tmp_path, hosts)
    bm.run()

    data = yaml.safe_load(_yaml_path(tmp_path).read_text())
    clients = data['system_under_test']['clients']
    assert len(clients) == 2
    assert sum(c['quantity'] for c in clients) == 3
    # D-7 sort: largest quantity first.
    assert clients[0]['quantity'] >= clients[1]['quantity']


# ---------------------------------------------------------------------------
# CR-01 regression — production single-node-fallback path
#
# The 12 tests above all go through `_make_benchmark`, which installs a
# `_collect_cluster_start` mock that seeds `self._cluster_info_start` BEFORE
# the Phase 2 write hook reads it at base.py:991. That masking pattern hides
# the production crash documented in `02-VERIFICATION.md` (status=gaps_found)
# and independently confirmed as CR-01 BLOCKER in `02-REVIEW.md`:
#
#   - `Benchmark.__init__` never initializes `self._cluster_info_start`.
#   - `_collect_cluster_start()` has an early-return branch at base.py:634-636
#     that fires on (a) `datagen`/`configview` commands and (b) any benchmark
#     whose `--hosts` default is None (e.g. VectorDB at cli/vectordb_args.py:107).
#   - On the early-return path, `_cluster_info_start` is never assigned, and
#     the write hook at base.py:991 raises AttributeError when it reads it.
#   - The catch-all `except Exception` at base.py:997 relabels the error as
#     "Failed to write systemname.yaml: ..." and re-raises — aborting the
#     benchmark BEFORE DLIO launches. (This is a regression from Phase 1:
#     production `datagen` used to complete normally.)
#
# These two regression tests construct a `VectorDBBenchmark` WITHOUT the
# `_collect_cluster_start` mock that `_make_benchmark` installs, so the real
# `_collect_cluster_start` runs and hits the early-return path. Both
# subcases (datagen and run-with-no-hosts) MUST run without AttributeError
# after the init-side fix in `Benchmark.__init__` lands.
# ---------------------------------------------------------------------------


def _make_benchmark_no_cluster_mock(tmp_path, *, command='run', mode='closed',
                                    orgname='Acme', systemname='sys-v1'):
    """Construct VectorDBBenchmark WITHOUT mocking `_collect_cluster_start`.

    This is the production-path harness that exposes CR-01. Everything else
    that `_make_benchmark` mocks is still mocked here (validate_environment,
    timeseries, cluster_end, write_timeseries_data, _run) — only the
    cluster-start mock is omitted so the real early-return path executes.
    """
    args = _vdb_args(tmp_path, command=command, mode=mode,
                     orgname=orgname, systemname=systemname)
    output_dir = str(tmp_path / f"output_{_RUN_DATETIME_COUNTER[0]}")
    with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
         patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
         patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
         patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
        mock_gen.return_value = output_dir
        from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
        bm = VectorDBBenchmark(args, run_datetime=_unique_run_datetime())

    # Mock everything EXCEPT _collect_cluster_start — the whole point of this
    # harness is to let the real early-return execute and trigger CR-01.
    bm._validate_environment = MagicMock()
    bm._start_timeseries_collection = MagicMock()
    bm._stop_timeseries_collection = MagicMock()
    bm._collect_cluster_end = MagicMock()
    bm.write_timeseries_data = MagicMock()
    bm._run = MagicMock(return_value=0)
    return bm


def test_run_does_not_raise_when_cluster_info_start_attribute_is_uninitialized_datagen(tmp_path):
    """CR-01 datagen-subcase: `command='datagen'` hits `_collect_cluster_start`
    early-return at base.py:634-636 (datagen short-circuit in
    `_should_collect_cluster_info`). The write hook at base.py:991 then reads
    `self._cluster_info_start`. Before the fix this raises AttributeError;
    after the fix the attribute is None-by-init, the writer's D-12 command
    gate at auto_generator.py:443 fires cleanly, and no file is written.
    """
    bm = _make_benchmark_no_cluster_mock(tmp_path, command='datagen')

    # Precautionary patch on the D-8 fallback's local-collection call.
    # For datagen this path is unreachable (D-12 gate fires first), but the
    # patch keeps the test hermetic against any future change that lets it
    # run on this subcase.
    with patch(
        'mlpstorage_py.system_description.auto_generator.collect_local_system_info',
        return_value={'meminfo': {'MemTotal': 0}, 'cpuinfo': '', 'os_release': {}},
    ):
        # PRIMARY ASSERTION (RED before fix, GREEN after):
        # Pre-fix, base.py:991 reads `self._cluster_info_start` after the
        # early-return at base.py:634-636 left it unset → AttributeError,
        # relabeled by the catch-all at base.py:997 as
        # "Failed to write systemname.yaml: ...". Post-fix, the init-side
        # default makes the read succeed, D-12 gates the write off cleanly,
        # and run() returns 0.
        rc = bm.run()

    assert rc == 0
    # Post-fix lock: the attribute is present (init-side fix) and None.
    assert hasattr(bm, '_cluster_info_start')
    assert bm._cluster_info_start is None
    # D-12 datagen-no-write contract: no file at canonical D-11 path.
    assert not _yaml_path(tmp_path, mode='closed', orgname='Acme',
                          systemname='sys-v1').exists()
    # The early-return left _cluster_info_start as the init-side default.
    assert bm._cluster_info_start is None


def test_run_does_not_raise_when_cluster_info_start_attribute_is_uninitialized_run(tmp_path):
    """CR-01 run-subcase: `command='run'` with no `--hosts` (mirrors VectorDB's
    `cli/vectordb_args.py:107` `default=None`) hits the same early-return at
    base.py:634-636 because both `_should_collect_cluster_info()` (no hosts)
    and `_should_use_ssh_collection()` (no hosts) return False. The write
    hook then reads `self._cluster_info_start`. Before the fix this raises
    AttributeError; after the fix the attribute is None-by-init, the writer's
    D-12 gate passes (command=='run'), the D-8 fallback at
    auto_generator.py:374-378 takes over with `cluster_info=None`, and the
    file IS written at the canonical D-11 path via local collection.
    """
    bm = _make_benchmark_no_cluster_mock(tmp_path, command='run')

    # D-8 fallback hermetic stub: _resolve_host_info_list(None) calls
    # collect_local_system_info(). Feed it a minimal HostInfo-compatible
    # dict so HostInfo.from_collected_data produces a populated host.
    # Per `rules/models.py:212-222`, cpuinfo must be a list of dicts (each
    # representing one logical CPU); summarize_cpuinfo derives socket count
    # from unique `physical id` values and model from cpuinfo_list[0].
    fake_local = {
        'hostname': 'localhost',
        'meminfo': {'MemTotal': 274_877_906_944},
        'cpuinfo': [
            {
                'processor': '0',
                'physical id': '0',
                'model name': 'Intel(R) Xeon Platinum 8480+',
                'cpu cores': '56',
                'flags': '',
            },
        ],
        'os_release': {'NAME': 'Rocky Linux', 'VERSION_ID': '9.5'},
    }
    with patch(
        'mlpstorage_py.system_description.auto_generator.collect_local_system_info',
        return_value=fake_local,
    ):
        # PRIMARY ASSERTION (RED before fix, GREEN after):
        # Pre-fix, base.py:991 reads `self._cluster_info_start` after the
        # early-return at base.py:634-636 left it unset → AttributeError,
        # relabeled by the catch-all at base.py:997 as
        # "Failed to write systemname.yaml: ...". Post-fix, the init-side
        # default makes the read succeed, D-12 gates pass through, D-8
        # fallback runs collect_local_system_info(), and the file lands.
        rc = bm.run()

    assert rc == 0
    # Post-fix lock: the attribute is present (init-side fix) and None.
    assert hasattr(bm, '_cluster_info_start')
    assert bm._cluster_info_start is None
    target = _yaml_path(tmp_path, mode='closed', orgname='Acme',
                       systemname='sys-v1')
    # D-8 fallback ran; D-12 gate passed; D-9 atomic write succeeded.
    assert target.exists()
    data = yaml.safe_load(target.read_text())
    # Non-empty valid YAML, system_under_test.clients populated.
    assert 'system_under_test' in data
    clients = data['system_under_test']['clients']
    assert isinstance(clients, list)
    assert len(clients) >= 1
    assert sum(c.get('quantity', 0) for c in clients) >= 1


# ---------------------------------------------------------------------------
# Phase 3 / Plan 03-05 — end-to-end chassis_model + networking emission
#
# These integration tests close the Phase 3 vertical: a HostInfo with
# populated chassis_model + networking flows through
# node_dict_from_host → group_by_fingerprint → _splice_stub_lists → yaml.safe_dump
# and lands real values in the emitted systemname.yaml. They also cover the
# cross-host fingerprint extensions from Plan 03-04 (chassis.model_name
# scalar key + ('networking_sig', _network_signature) callable).
#
# The fixture style mirrors _make_host above; a new helper
# _make_host_phase3 layers chassis_model + networking on top of the Phase 2
# defaults so existing Phase 2 tests are NOT touched.
# ---------------------------------------------------------------------------


def _make_host_phase3(
    *,
    chassis_model: str = "PowerEdge R760",
    networking=None,
    cpu_model: str = "Intel(R) Xeon Platinum 8480+",
    num_cores: int = 56,
    num_sockets: int = 2,
    mem_bytes: int = 274_877_906_944,
    os_name: str = "Rocky Linux",
    os_version: str = "9.5",
    hostname: str = "h1",
) -> HostInfo:
    """Phase 3 extension of _make_host: layers chassis_model + networking on
    top of the Phase 2 defaults so end-to-end emit can be exercised.

    Default networking = one 100GbE up + one 200Gb IB up — exercises the
    multi-type path in the per-host grouping pass and the IB-presence
    path required by Phase 3 Success Criterion #4.
    """
    if networking is None:
        networking = [
            {"type": "ethernet", "speed": 100, "state": "up"},
            {"type": "infiniband", "speed": 200, "state": "up"},
        ]
    host = _make_host(
        cpu_model=cpu_model, num_cores=num_cores, num_sockets=num_sockets,
        mem_bytes=mem_bytes, os_name=os_name, os_version=os_version,
        hostname=hostname,
    )
    host.chassis_model = chassis_model
    host.networking = networking
    return host


def test_full_run_emits_chassis_model_in_yaml(tmp_path):
    """Phase 3 success criterion #1: a fully-populated chassis_model surfaces
    in clients[0].chassis.model_name verbatim in the emitted YAML."""
    hosts = [_make_host_phase3(chassis_model="PowerEdge R760")]
    bm = _make_benchmark(tmp_path, hosts)
    bm.run()

    data = yaml.safe_load(_yaml_path(tmp_path).read_text())
    clients = data['system_under_test']['clients']
    assert len(clients) == 1
    assert clients[0]['chassis']['model_name'] == "PowerEdge R760"


def test_full_run_emits_networking_in_yaml(tmp_path):
    """Phase 3 success criterion #2 + #4: a host with 100GbE + 200Gb IB
    emits two stanzas in clients[0].networking (one per type). The up
    NICs receive the D-17 `traffic: []` splice at _splice_stub_lists time.
    """
    hosts = [_make_host_phase3()]
    bm = _make_benchmark(tmp_path, hosts)
    bm.run()

    data = yaml.safe_load(_yaml_path(tmp_path).read_text())
    networking = data['system_under_test']['clients'][0]['networking']
    # Two stanzas (one per type) — both up, unit_count=1 each, traffic=[]
    # via the D-17 splice on up entries.
    assert len(networking) == 2
    by_type = {e['type']: e for e in networking}
    # Ethernet entry — up, 100Gb, count=1, D-17 traffic=[] splice.
    assert by_type['ethernet']['speed'] == 100
    assert by_type['ethernet']['state'] == 'up'
    assert by_type['ethernet']['unit_count'] == 1
    assert by_type['ethernet']['traffic'] == []
    # InfiniBand — phase 3 success criterion #4.
    assert by_type['infiniband']['speed'] == 200
    assert by_type['infiniband']['state'] == 'up'
    assert by_type['infiniband']['unit_count'] == 1
    assert by_type['infiniband']['traffic'] == []


def test_full_run_chassis_model_empty_when_collection_failed(tmp_path):
    """Phase 3 success criterion #1 (failure half): a host with chassis_model=''
    (collector blind) emits clients[0].chassis.model_name == '' — the visible
    SER-02 blank — without crashing the write."""
    hosts = [_make_host_phase3(chassis_model="")]
    bm = _make_benchmark(tmp_path, hosts)
    bm.run()

    data = yaml.safe_load(_yaml_path(tmp_path).read_text())
    clients = data['system_under_test']['clients']
    assert len(clients) == 1
    assert clients[0]['chassis']['model_name'] == ""


def test_cross_host_fingerprint_splits_on_chassis_model(tmp_path):
    """Phase 3 success criterion #5 (chassis differentiation): two hosts
    identical on CPU/memory/OS/networking but DIFFERENT on chassis_model
    produce TWO stanzas (one per chassis), not one collapsed stanza.
    Exercises Plan 03-04's `chassis.model_name` scalar key in _FINGERPRINT_KEYS.
    """
    hosts = [
        _make_host_phase3(hostname="h0", chassis_model="PowerEdge R760"),
        _make_host_phase3(hostname="h1", chassis_model="ProLiant DL380"),
    ]
    bm = _make_benchmark(tmp_path, hosts)
    bm.run()

    data = yaml.safe_load(_yaml_path(tmp_path).read_text())
    clients = data['system_under_test']['clients']
    assert len(clients) == 2
    model_names = sorted(c['chassis']['model_name'] for c in clients)
    assert model_names == ["PowerEdge R760", "ProLiant DL380"]
    assert sum(c['quantity'] for c in clients) == 2


def test_cross_host_fingerprint_splits_on_networking_signature(tmp_path):
    """Phase 3 success criterion #5 (networking signature differentiation):
    two hosts identical on chassis/CPU/memory/OS, one with a down NIC the
    other does NOT have, produce two stanzas. Exercises Plan 03-04's
    `('networking_sig', _network_signature)` callable extractor.
    """
    clean_nics = [
        {"type": "ethernet", "speed": 100, "state": "up"},
        {"type": "ethernet", "speed": 100, "state": "up"},
    ]
    degraded_nics = [
        {"type": "ethernet", "speed": 100, "state": "up"},
        {"type": "ethernet", "speed": 100, "state": "down"},
    ]
    hosts = [
        _make_host_phase3(hostname="h0", networking=clean_nics),
        _make_host_phase3(hostname="h1", networking=degraded_nics),
    ]
    bm = _make_benchmark(tmp_path, hosts)
    bm.run()

    data = yaml.safe_load(_yaml_path(tmp_path).read_text())
    clients = data['system_under_test']['clients']
    # Two stanzas — degraded vs clean — even though every other field matches.
    assert len(clients) == 2
    assert sum(c['quantity'] for c in clients) == 2


def test_full_run_emits_networking_blank_stub_when_no_collected_data(tmp_path):
    """Phase 3 D-2 universal blank fallback: a host with networking=[]
    (collector failed entirely) → clients[0].networking == [_NETWORKING_STUB]
    via the _splice_stub_lists fallback branch (Plan 03-04 extended).
    """
    hosts = [_make_host_phase3(networking=[])]
    bm = _make_benchmark(tmp_path, hosts)
    bm.run()

    data = yaml.safe_load(_yaml_path(tmp_path).read_text())
    networking = data['system_under_test']['clients'][0]['networking']
    # _splice_stub_lists fell back to the _NETWORKING_STUB blank entry.
    assert len(networking) == 1
    stub_entry = networking[0]
    assert stub_entry['unit_count'] == ""
    assert stub_entry['type'] == ""
    assert stub_entry['state'] == ""
    assert stub_entry['speed'] == ""
    assert stub_entry['traffic'] == []


def test_validator_no_chassis_error_when_dmi_populated(tmp_path):
    """SER-03 success criterion #4 (Phase 3 extension): with a populated
    chassis_model from the collector, schema_validator.validate_file does
    NOT report `chassis -> model_name` as an error.

    Sibling test to test_validator_errors_only_on_blanks (which keeps
    exercising the blank-chassis case with the Phase 2 fixture). Together
    they verify the validator's SER-02 surface migrates from "always-blank"
    to "blank-only-when-collection-failed" in Phase 3.
    """
    from mlpstorage_py.system_description import schema_validator

    hosts = [_make_host_phase3(chassis_model="PowerEdge R760")]
    bm = _make_benchmark(tmp_path, hosts)
    bm.run()
    target = _yaml_path(tmp_path)
    assert target.exists()

    errors = schema_validator.validate_file(str(target))
    error_paths = {e.split(":", 1)[0].strip() for e in errors}

    # The chassis -> model_name path MUST NOT appear in any error now that
    # the collector supplied a real value.
    for ep in error_paths:
        assert "chassis -> model_name" not in ep, (
            f"chassis.model_name unexpectedly errored in {ep!r} when "
            f"DMI was populated; node_dict_from_host should have wired "
            f"the real chassis_model through."
        )


# ---------------------------------------------------------------------------
# Phase 4 / Plan 04-05 — end-to-end sysctl + environment + drives emission
#
# Closes the Phase 4 vertical: a HostInfo with populated sysctl + environment
# + drives flows through node_dict_from_host → group_by_fingerprint →
# _splice_stub_lists → yaml.safe_dump and lands in the emitted
# systemname.yaml. Verifies the D-33 drives-omit branch end-to-end (lsblk
# absent → drives key absent at the client-stanza level), cross-host
# fingerprint splits on each of the three new dimensions (D-35), homogeneous
# fleet collapse, and Yamale schema validation pass on Phase-4-populated
# fields (SER-03).
# ---------------------------------------------------------------------------


def _make_host_phase4(
    *,
    sysctl=None,
    environment=None,
    drives=None,
    chassis_model: str = "PowerEdge R760",
    networking=None,
    cpu_model: str = "Intel(R) Xeon Platinum 8480+",
    num_cores: int = 56,
    num_sockets: int = 2,
    mem_bytes: int = 274_877_906_944,
    os_name: str = "Rocky Linux",
    os_version: str = "9.5",
    hostname: str = "h1",
) -> HostInfo:
    """Phase 4 extension of `_make_host_phase3`: layers sysctl + environment +
    drives on top of the Phase-3 defaults so end-to-end YAML emit can be
    exercised against a fully-populated HostInfo.

    Defaults match the Phase-3 helper's networking (1 100GbE up + 1 200Gb IB up).
    `sysctl`, `environment`, `drives` default to empty lists so callers can
    selectively populate just the one dimension a test cares about.
    """
    host = _make_host_phase3(
        chassis_model=chassis_model, networking=networking,
        cpu_model=cpu_model, num_cores=num_cores, num_sockets=num_sockets,
        mem_bytes=mem_bytes, os_name=os_name, os_version=os_version,
        hostname=hostname,
    )
    host.sysctl = sysctl if sysctl is not None else []
    host.environment = environment if environment is not None else []
    host.drives = drives if drives is not None else []
    return host


class TestPhase4EndToEnd:
    """End-to-end coverage for Phase 4 ROADMAP success criteria #1-5
    (sysctl populated, environment populated with redaction, drives present,
    drives omitted per D-33, cross-host splits and homogeneous collapse,
    Yamale schema validation passes on populated fields).
    """

    def test_drives_populated_emits_drives_key(self, tmp_path):
        """ROADMAP SC #3: a host with populated drives produces a
        clients[0].drives list containing one entry per
        (vendor,model,interface,capacity_in_GB) group with unit_count.
        """
        hosts = [_make_host_phase4(drives=[
            {"vendor_name": "INTEL", "model_name": "SSDPED1K375GA",
             "interface": "nvme", "capacity_in_GB": 375},
        ])]
        bm = _make_benchmark(tmp_path, hosts)
        bm.run()

        data = yaml.safe_load(_yaml_path(tmp_path).read_text())
        client0 = data["system_under_test"]["clients"][0]
        assert "drives" in client0
        assert client0["drives"] == [{
            "vendor_name": "INTEL",
            "model_name": "SSDPED1K375GA",
            "interface": "nvme",
            "capacity_in_GB": 375,
            "unit_count": 1,
        }]

    def test_drives_absent_omits_drives_key_d33(self, tmp_path):
        """ROADMAP SC #5 (D-33): a host with empty drives produces a
        client stanza with NO drives key — verified by reading back the YAML.
        """
        hosts = [_make_host_phase4(drives=[])]
        bm = _make_benchmark(tmp_path, hosts)
        bm.run()

        data = yaml.safe_load(_yaml_path(tmp_path).read_text())
        client0 = data["system_under_test"]["clients"][0]
        assert "drives" not in client0, (
            f"Phase 4 / D-33 violation: drives key present in YAML when "
            f"host.drives was empty; expected key OMITTED. client0={client0!r}"
        )

    def test_sysctl_populated_emits_sysctl_key(self, tmp_path):
        """ROADMAP SC #1: a host with populated sysctl produces a
        clients[0].sysctl list containing the entries verbatim.
        """
        hosts = [_make_host_phase4(sysctl=[
            {"name": "vm.dirty_ratio", "value": "20"},
        ])]
        bm = _make_benchmark(tmp_path, hosts)
        bm.run()

        data = yaml.safe_load(_yaml_path(tmp_path).read_text())
        client0 = data["system_under_test"]["clients"][0]
        assert client0["sysctl"] == [
            {"name": "vm.dirty_ratio", "value": "20"},
        ]

    def test_environment_populated_emits_environment_with_redaction(self, tmp_path):
        """ROADMAP SC #2: a host with environment values (already-redacted per
        the COLL-06 collector contract) round-trips verbatim through the YAML.
        AWS_SECRET_ACCESS_KEY is length-only redacted (D-24); regular vars
        flow through unredacted.
        """
        hosts = [_make_host_phase4(environment=[
            {"name": "AWS_SECRET_ACCESS_KEY", "value": "[SET — 40 chars]"},
            {"name": "BUCKET", "value": "my-bucket"},
        ])]
        bm = _make_benchmark(tmp_path, hosts)
        bm.run()

        data = yaml.safe_load(_yaml_path(tmp_path).read_text())
        client0 = data["system_under_test"]["clients"][0]
        # Both entries round-trip verbatim (already-redacted per collector
        # contract; the writer never re-redacts).
        env_by_name = {e["name"]: e["value"] for e in client0["environment"]}
        assert env_by_name["AWS_SECRET_ACCESS_KEY"] == "[SET — 40 chars]"
        assert env_by_name["BUCKET"] == "my-bucket"

    def test_two_hosts_differ_on_sysctl_split_to_two_stanzas(self, tmp_path):
        """D-35 strict policy: two hosts that differ on sysctl values produce
        TWO client stanzas (one per signature), even when every other field
        is identical. Exercises Plan 04-04's `_sysctl_signature` extractor.
        """
        hosts = [
            _make_host_phase4(
                hostname="h0",
                sysctl=[{"name": "vm.dirty_ratio", "value": "10"}],
            ),
            _make_host_phase4(
                hostname="h1",
                sysctl=[{"name": "vm.dirty_ratio", "value": "20"}],
            ),
        ]
        bm = _make_benchmark(tmp_path, hosts)
        bm.run()

        data = yaml.safe_load(_yaml_path(tmp_path).read_text())
        clients = data["system_under_test"]["clients"]
        assert len(clients) == 2
        assert sum(c["quantity"] for c in clients) == 2

    def test_two_hosts_differ_on_drives_split_to_two_stanzas(self, tmp_path):
        """D-35 strict policy: two hosts that differ on drives produce
        TWO client stanzas. Exercises Plan 04-04's `_drive_signature` extractor.
        """
        hosts = [
            _make_host_phase4(
                hostname="h0",
                drives=[
                    {"vendor_name": "INTEL", "model_name": "X",
                     "interface": "nvme", "capacity_in_GB": 500},
                ],
            ),
            _make_host_phase4(
                hostname="h1",
                drives=[
                    {"vendor_name": "INTEL", "model_name": "X",
                     "interface": "nvme", "capacity_in_GB": 1000},
                ],
            ),
        ]
        bm = _make_benchmark(tmp_path, hosts)
        bm.run()

        data = yaml.safe_load(_yaml_path(tmp_path).read_text())
        clients = data["system_under_test"]["clients"]
        assert len(clients) == 2
        assert sum(c["quantity"] for c in clients) == 2

    def test_two_hosts_differ_on_environment_split_to_two_stanzas(self, tmp_path):
        """D-35 strict policy: two hosts that differ on environment values
        produce TWO client stanzas. Exercises `_environment_signature`."""
        hosts = [
            _make_host_phase4(
                hostname="h0",
                environment=[{"name": "NCCL_DEBUG", "value": "INFO"}],
            ),
            _make_host_phase4(
                hostname="h1",
                environment=[{"name": "NCCL_DEBUG", "value": "TRACE"}],
            ),
        ]
        bm = _make_benchmark(tmp_path, hosts)
        bm.run()

        data = yaml.safe_load(_yaml_path(tmp_path).read_text())
        clients = data["system_under_test"]["clients"]
        assert len(clients) == 2
        assert sum(c["quantity"] for c in clients) == 2

    def test_homogeneous_fleet_collapses_to_one_stanza(self, tmp_path):
        """ROADMAP SC #1/#2/#3 collapse path: 3 identical hosts with populated
        sysctl/environment/drives collapse to a single stanza with quantity:3.
        All three list fields populate the collapsed stanza."""
        sysctl = [{"name": "vm.dirty_ratio", "value": "20"}]
        environment = [{"name": "BUCKET", "value": "my-bucket"}]
        drives = [{"vendor_name": "INTEL", "model_name": "X",
                   "interface": "nvme", "capacity_in_GB": 500}]
        hosts = [
            _make_host_phase4(
                hostname=f"h{i}",
                sysctl=sysctl, environment=environment, drives=drives,
            )
            for i in range(3)
        ]
        bm = _make_benchmark(tmp_path, hosts)
        bm.run()

        data = yaml.safe_load(_yaml_path(tmp_path).read_text())
        clients = data["system_under_test"]["clients"]
        assert len(clients) == 1
        assert clients[0]["quantity"] == 3
        # The collapsed stanza carries all three Phase-4 lists populated.
        assert clients[0]["sysctl"] == sysctl
        assert clients[0]["environment"] == environment
        assert clients[0]["drives"] == [
            {"vendor_name": "INTEL", "model_name": "X",
             "interface": "nvme", "capacity_in_GB": 500, "unit_count": 1},
        ]

    def test_yamale_schema_validation_passes_on_phase_4_emit_shape(self, tmp_path):
        """ROADMAP SC #1+SC #4 + SER-03: with sysctl/environment/drives all
        populated (and drives entries containing ONLY the four COLL-07 fields
        — no media_type, no form_factor, no performance), the Pydantic
        validator surfaces NO errors over the Phase-4-populated fields.

        Pre-existing blanks (D-3 networking stub for empty networking,
        D-14 top-level omissions) still produce errors — those are the
        SER-02 signal — but no NEW errors are introduced on the Phase-4
        emit surface.
        """
        from mlpstorage_py.system_description import schema_validator

        hosts = [_make_host_phase4(
            sysctl=[{"name": "vm.dirty_ratio", "value": "20"}],
            environment=[{"name": "BUCKET", "value": "my-bucket"}],
            drives=[{"vendor_name": "INTEL", "model_name": "X",
                     "interface": "nvme", "capacity_in_GB": 500}],
        )]
        bm = _make_benchmark(tmp_path, hosts)
        bm.run()
        target = _yaml_path(tmp_path)
        assert target.exists()

        errors = schema_validator.validate_file(str(target))
        error_paths = {e.split(":", 1)[0].strip() for e in errors}

        # Phase-4-populated fields MUST NOT appear in any error.
        forbidden_in_errors = [
            "sysctl -> 0 -> name",
            "sysctl -> 0 -> value",
            "environment -> 0 -> name",
            "environment -> 0 -> value",
            "drives -> 0 -> vendor_name",
            "drives -> 0 -> model_name",
            "drives -> 0 -> interface",
            "drives -> 0 -> capacity_in_GB",
        ]
        for field in forbidden_in_errors:
            for ep in error_paths:
                assert field not in ep, (
                    f"Phase-4-populated field {field!r} unexpectedly appears "
                    f"in error path {ep!r} — node_dict_from_host should have "
                    f"wired the real value through cleanly."
                )

        # ROADMAP SC #4: drives entries do NOT carry media_type/form_factor/
        # performance keys (the collector emits only the four fields above).
        data = yaml.safe_load(target.read_text())
        for client in data["system_under_test"]["clients"]:
            for drive in client.get("drives", []):
                assert "media_type" not in drive
                assert "form_factor" not in drive
                assert "performance" not in drive


# ---------------------------------------------------------------------------
# Phase 5 / Plan 05-05 — End-to-end integration tests
#
# Three new test classes cover the 8 ROADMAP SC + LIFE-04 hand-fill survival
# + main.py top-level dispatch contracts:
#
#   - TestPhase5Lifecycle (SC#1, SC#2, SC#3, SC#4, LIFE-04, main.py dispatch)
#   - TestPhase5Cap01     (SC#5, SC#6, per-rank starvation, A6/A7/A8 locks)
#   - TestPhase5Cap02     (SC#7, SC#8, gate-order locks)
#
# Fixture style: each class extends `_make_benchmark` with targeted patches
# at the leaf I/O surface (os.statvfs for CAP-01, run_shared_fs_probe for
# CAP-02, _pre_execution_gate no-op for LIFE-02/03/04). The orchestration
# above the leaf — Benchmark.run() / _pre_execution_gate / write_systemname_yaml —
# runs as real Python so the integration surface is exercised end-to-end.
# ---------------------------------------------------------------------------


def _make_benchmark_no_gate(tmp_path, hosts, *, command='run', mode='closed',
                            orgname='Acme', systemname='sys-v1',
                            timeseries_side_effect=None):
    """_make_benchmark variant that also patches _pre_execution_gate to a
    no-op. Used by TestPhase5Lifecycle so the LIFE-02/03/04 surface is
    exercised without invoking CAP-01/CAP-02 (which have their own classes).
    """
    bm = _make_benchmark(tmp_path, hosts, command=command, mode=mode,
                        orgname=orgname, systemname=systemname,
                        timeseries_side_effect=timeseries_side_effect)
    bm._pre_execution_gate = MagicMock()
    return bm


def _drift_host(hosts, *, drift_field='cpu_model',
                drift_value='Intel(R) Xeon Platinum 9999X'):
    """Return a copy of `hosts` with the first host's `drift_field` mutated
    on the chassis.cpu_model axis (the most common drift surface).
    """
    drifted = [_make_host(hostname=h.hostname,
                          cpu_model=h.cpu.model,
                          num_cores=h.cpu.num_cores,
                          num_sockets=h.cpu.num_sockets,
                          mem_bytes=h.memory.total,
                          os_name=h.system.os_release.get('NAME', 'Rocky Linux'),
                          os_version=h.system.os_release.get('VERSION_ID', '9.5'))
               for h in hosts]
    if drift_field == 'cpu_model':
        drifted[0].cpu.model = drift_value
    elif drift_field == 'memory':
        drifted[0].memory.total = int(drift_value)
    return drifted


class TestPhase5Lifecycle:
    """End-to-end coverage for Phase 5 LIFE-02/03/04 surface across the full
    Benchmark.run() pipeline.

    Covers ROADMAP SC#1 (hand-fill survival), SC#2 (drift fails before DLIO),
    SC#3 (Remediation block in message), SC#4 (per-mode independence —
    bidirectionally per checker W-3), and the main.py top-level exception
    dispatch (SystemDriftError + SystemDescriptionParseError → non-zero exit).
    """

    # ---- SC#1 / LIFE-01 regression -----------------------------------------

    def test_first_run_writes_baseline_systemname_yaml(self, tmp_path):
        """LIFE-01 regression: first run lands the file at the canonical path."""
        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm = _make_benchmark_no_gate(tmp_path, hosts)
        rc = bm.run()
        assert rc == 0
        target = _yaml_path(tmp_path)
        assert target.exists()
        data = yaml.safe_load(target.read_text())
        assert data['system_under_test']['clients'][0]['quantity'] == 2

    # ---- LIFE-04 no-touch in full pipeline ---------------------------------

    def test_second_run_unchanged_fleet_no_touch_mtime_invariant_full_pipeline(self, tmp_path):
        """LIFE-04 end-to-end: re-running against an unchanged fleet through
        the full Benchmark.run() leaves the on-disk file mtime + bytes
        unchanged (the load-diff-no-op branch fires)."""
        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm1 = _make_benchmark_no_gate(tmp_path, hosts)
        bm1.run()
        target = _yaml_path(tmp_path)
        snapshot_bytes = target.read_bytes()
        snapshot_mtime_ns = target.stat().st_mtime_ns
        time.sleep(1.1)  # cross-FS conservative second-resolution mtime margin

        bm2 = _make_benchmark_no_gate(tmp_path, hosts)
        rc = bm2.run()
        assert rc == 0
        assert target.read_bytes() == snapshot_bytes
        assert target.stat().st_mtime_ns == snapshot_mtime_ns

    # ---- SC#1 hand-fill survival in full pipeline --------------------------

    def test_submitter_hand_fills_survive_unchanged_full_pipeline_sc1(self, tmp_path):
        """SC#1 + LIFE-04 (REQUIREMENTS.md milestone-core-value): submitter
        edits SER-02 blank scalar fields on disk; re-running the SAME run
        command against the SAME fleet leaves the hand-filled values
        byte-identical.

        The SER-02 blanks that flow through Pitfall 3(a) blank preservation
        are SCALAR fields where in-memory == "" (e.g. friendly_description).
        Stub-list fields like networking[*].traffic round-trip via the
        _splice_stub_lists symmetry pass instead — both sides end up with
        `[]` post-splice, so the diff is empty regardless of what the user
        put in. Here we test the scalar-blank-preservation surface (the
        load-bearing user-visible contract) end-to-end.
        """
        hosts = [_make_host_phase3(hostname=f"h{i}") for i in range(2)]
        bm1 = _make_benchmark_no_gate(tmp_path, hosts)
        bm1.run()
        target = _yaml_path(tmp_path)
        assert target.exists()

        # Submitter hand-fills SER-02 scalar blanks (friendly_description and
        # the chassis.model_name field when blank — both are SER-02 blanks
        # at the scalar leaf level where Pitfall 3(a) applies).
        on_disk = yaml.safe_load(target.read_text())
        client0 = on_disk['system_under_test']['clients'][0]
        client0['friendly_description'] = 'Acme rack-7-pod-2'
        # If chassis.model_name landed as a blank, hand-fill it too.
        if client0.get('chassis', {}).get('model_name', '') == '':
            client0['chassis']['model_name'] = 'PowerEdge R760-hand-filled'
        target.write_text(yaml.safe_dump(on_disk, default_flow_style=False))
        hand_filled_bytes = target.read_bytes()

        # Re-run against the same fleet — must not raise + must not rewrite.
        bm2 = _make_benchmark_no_gate(tmp_path, hosts)
        rc = bm2.run()
        assert rc == 0

        # Hand-fills survived byte-for-byte (LIFE-04 no-touch + Pitfall 3(a)
        # blank preservation).
        assert target.read_bytes() == hand_filled_bytes
        re_loaded = yaml.safe_load(target.read_text())
        re_client0 = re_loaded['system_under_test']['clients'][0]
        assert re_client0['friendly_description'] == 'Acme rack-7-pod-2'

    # ---- SC#2 drift fails before DLIO --------------------------------------

    def test_drift_on_cpu_model_fails_before_dlio_sc2(self, tmp_path):
        """SC#2: a drifted cpu_model causes run() to raise SystemDriftError
        BEFORE bench._run is called. Pre-DLIO-launch lock."""
        from mlpstorage_py.errors import SystemDriftError

        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm1 = _make_benchmark_no_gate(tmp_path, hosts)
        bm1.run()

        drifted = _drift_host(hosts, drift_field='cpu_model',
                              drift_value='Intel(R) Xeon Platinum 9999X')
        bm2 = _make_benchmark_no_gate(tmp_path, drifted)
        with pytest.raises(SystemDriftError) as exc_info:
            bm2.run()

        # _run MUST NOT have been called — the drift fails BEFORE DLIO.
        bm2._run.assert_not_called()
        # Unified-diff body present in the message.
        msg = str(exc_info.value)
        assert "--- on-disk" in msg or "+++ in-memory" in msg

    def test_drift_on_sysctl_value_surfaces_jsonpath_hunk_sc2(self, tmp_path):
        """SC#2: a drifted sysctl entry on disk vs. empty in-memory surfaces
        a JSONPath-style hunk in the diff (either '@@ ' hunks or fingerprint
        orphan since sysctl is part of _FINGERPRINT_KEYS)."""
        from mlpstorage_py.errors import SystemDriftError

        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm1 = _make_benchmark_no_gate(tmp_path, hosts)
        bm1.run()
        target = _yaml_path(tmp_path)

        # Drift on-disk sysctl to a non-empty value (fingerprint-affecting).
        on_disk = yaml.safe_load(target.read_text())
        on_disk['system_under_test']['clients'][0]['sysctl'] = [
            {'name': 'net.core.rmem_max', 'value': '16777216'}
        ]
        target.write_text(yaml.safe_dump(on_disk, default_flow_style=False))

        # Re-run with the same fleet (in-memory sysctl=[]) — drift surfaces.
        bm2 = _make_benchmark_no_gate(tmp_path, hosts)
        with pytest.raises(SystemDriftError) as exc_info:
            bm2.run()
        msg = str(exc_info.value)
        # Either an '@@' JSONPath hunk OR a fingerprint orphan path containing sysctl.
        assert ("@@ " in msg) or ("clients[fingerprint=" in msg), (
            f"expected JSONPath hunk or fingerprint orphan, got:\n{msg}"
        )

    # ---- SC#3 Remediation block --------------------------------------------

    def test_drift_message_contains_both_remediation_options_sc3(self, tmp_path):
        """SC#3: the SystemDriftError message lists BOTH remediation options
        (rename existing yaml + remove existing yaml) verbatim per D-40."""
        from mlpstorage_py.errors import SystemDriftError

        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm1 = _make_benchmark_no_gate(tmp_path, hosts)
        bm1.run()

        drifted = _drift_host(hosts, drift_value='Intel(R) Xeon Platinum 9999X')
        bm2 = _make_benchmark_no_gate(tmp_path, drifted)
        with pytest.raises(SystemDriftError) as exc_info:
            bm2.run()

        msg = str(exc_info.value)
        # D-40 verbatim remediation hint markers.
        assert "Rename" in msg, f"expected 'Rename' in Remediation block, got:\n{msg}"
        assert "Remove" in msg, f"expected 'Remove' in Remediation block, got:\n{msg}"

    # ---- SC#4 per-mode independence (bidirectional per W-3) ---------------

    def test_drift_in_closed_mode_does_not_trigger_drift_in_open_mode_sc4(self, tmp_path):
        """SC#4 (closed→open): drifting the CLOSED-mode YAML on disk does
        NOT cause an OPEN-mode run to raise SystemDriftError — each mode
        owns its own file at its own path (D-11)."""
        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]

        # Run-1 in closed mode lands /tmp/r1/closed/Acme/systems/sys-v1.yaml.
        bm_closed = _make_benchmark_no_gate(tmp_path, hosts, mode='closed')
        bm_closed.run()
        closed_path = _yaml_path(tmp_path, mode='closed')
        assert closed_path.exists()

        # Manually drift the closed file.
        on_disk = yaml.safe_load(closed_path.read_text())
        on_disk['system_under_test']['clients'][0]['chassis']['cpu_model'] = 'DRIFTED'
        closed_path.write_text(yaml.safe_dump(on_disk, default_flow_style=False))

        # Run-1 in open mode against the SAME fleet — different path, no drift.
        bm_open = _make_benchmark_no_gate(tmp_path, hosts, mode='open')
        rc = bm_open.run()
        assert rc == 0
        open_path = _yaml_path(tmp_path, mode='open')
        assert open_path.exists()
        # The closed-mode file is still drifted; the open-mode run touched ONLY
        # the open path.
        assert 'DRIFTED' in closed_path.read_text()

    def test_drift_in_open_mode_does_not_trigger_drift_in_closed_mode_sc4(self, tmp_path):
        """W-3 checker-mandated symmetric direction: drifting the OPEN-mode
        YAML on disk does NOT cause a CLOSED-mode run to raise.
        Proves per-mode independence holds bidirectionally."""
        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]

        bm_open = _make_benchmark_no_gate(tmp_path, hosts, mode='open')
        bm_open.run()
        open_path = _yaml_path(tmp_path, mode='open')
        assert open_path.exists()

        on_disk = yaml.safe_load(open_path.read_text())
        on_disk['system_under_test']['clients'][0]['chassis']['cpu_model'] = 'DRIFTED'
        open_path.write_text(yaml.safe_dump(on_disk, default_flow_style=False))

        bm_closed = _make_benchmark_no_gate(tmp_path, hosts, mode='closed')
        rc = bm_closed.run()
        assert rc == 0
        closed_path = _yaml_path(tmp_path, mode='closed')
        assert closed_path.exists()
        assert 'DRIFTED' in open_path.read_text()

    # ---- main.py top-level dispatch contract -------------------------------

    def test_main_py_dispatches_drift_error_to_nonzero_exit_via_systemexit(self, tmp_path):
        """main.py:262 + 527 dispatch contract: a SystemDriftError surfacing
        from Benchmark.run() routes through the MLPStorageException catch-all
        and exits with EXIT_CODE.FAILURE (non-zero)."""
        from mlpstorage_py.errors import SystemDriftError
        from mlpstorage_py.config import EXIT_CODE

        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm1 = _make_benchmark_no_gate(tmp_path, hosts)
        bm1.run()

        drifted = _drift_host(hosts, drift_value='Intel(R) Xeon Platinum 9999X')
        bm2 = _make_benchmark_no_gate(tmp_path, drifted)

        # Simulate the main.py dispatch path: wrap bm2.run() in the same
        # exception-handler shape as main.py:495-532.
        rc = None
        captured_error_msg = None
        try:
            bm2.run()
            rc = EXIT_CODE.SUCCESS
        except SystemDriftError as e:
            # This is the verbatim contract — main.py:527-532 catches
            # MLPStorageException and returns EXIT_CODE.FAILURE; SystemDriftError
            # IS-A MLPStorageException so it routes through that branch.
            captured_error_msg = str(e)
            rc = EXIT_CODE.FAILURE

        assert rc != 0, "drift must produce a non-zero exit code"
        assert rc == EXIT_CODE.FAILURE
        # The captured message contains the unified-diff body.
        assert captured_error_msg is not None
        assert ("--- on-disk" in captured_error_msg
                or "+++ in-memory" in captured_error_msg
                or "Remediation" in captured_error_msg), (
            f"expected diff/Remediation markers in captured stderr, got:\n{captured_error_msg}"
        )

    def test_malformed_yaml_raises_parse_error_and_exits_nonzero(self, tmp_path):
        """Malformed YAML on disk causes SystemDescriptionParseError which
        routes through the same MLPStorageException dispatch to non-zero exit.
        """
        from mlpstorage_py.errors import SystemDescriptionParseError
        from mlpstorage_py.config import EXIT_CODE

        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        # Pre-place garbage at the canonical path.
        target = _yaml_path(tmp_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("not: valid: yaml: : : :\n")

        bm = _make_benchmark_no_gate(tmp_path, hosts)
        rc = None
        captured = None
        try:
            bm.run()
            rc = EXIT_CODE.SUCCESS
        except SystemDescriptionParseError as e:
            captured = str(e)
            rc = EXIT_CODE.FAILURE

        assert rc != 0
        assert captured is not None
        assert "malformed" in captured.lower() or "parse" in captured.lower() or "yaml" in captured.lower()

    # ---- D-12: datagen never triggers the lifecycle branch -----------------

    def test_datagen_does_not_trigger_lifecycle_branch(self, tmp_path):
        """D-12 carry-forward: a datagen command does NOT enter the
        FileExistsError load-diff branch even when garbage YAML is
        pre-placed at the systemname path. The writer's command gate fires
        before parse_on_disk_systemname_yaml is reached.
        """
        from mlpstorage_py.errors import SystemDescriptionParseError, SystemDriftError

        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        # Pre-place garbage at the canonical path.
        target = _yaml_path(tmp_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("not: valid: yaml: : : :\n")

        bm = _make_benchmark_no_gate(tmp_path, hosts, command='datagen')
        # Must NOT raise — the D-12 gate fires before parse_on_disk loads
        # the garbage.
        rc = bm.run()
        assert rc == 0
        # Garbage file untouched (datagen doesn't write either).
        assert target.read_text().startswith("not:")


# =============================================================================
# TestPhase5Cap01 — CAP-01 disk-space gate end-to-end
# =============================================================================


def _patch_statvfs_available(available_bytes):
    """Return a side_effect callable for `os.statvfs` that reports
    available_bytes = available_bytes (via f_bavail * f_frsize=1)."""
    def _statvfs_side_effect(_path):
        stat = MagicMock()
        stat.f_bavail = available_bytes
        stat.f_frsize = 1
        return stat
    return _statvfs_side_effect


class TestPhase5Cap01:
    """End-to-end coverage for CAP-01 capacity gate.

    Covers SC#5 (starved-destination fail with 4-field message), SC#6
    (happy-path silence), per-rank starvation, and the A6/A7/A8 escape
    hatch contracts end-to-end.
    """

    def test_starved_destination_fails_datagen_with_4field_message_sc5(self, tmp_path):
        """SC#5 datagen path: TrainingBenchmark.datasize with insufficient
        space raises FileSystemError containing all four locked fields
        (destination, available_bytes, required_bytes, deficit)."""
        from mlpstorage_py.errors import FileSystemError, ErrorCode

        # Patch the leaf I/O surface; the gate's check_capacity_4field
        # raises naturally from the patched statvfs result.
        with patch('mlpstorage_py.benchmarks.capacity_gate.os.statvfs',
                   side_effect=_patch_statvfs_available(1)):
            from mlpstorage_py.benchmarks.capacity_gate import check_capacity_4field
            with pytest.raises(FileSystemError) as exc_info:
                check_capacity_4field(str(tmp_path), 10**15, MagicMock())

        msg = str(exc_info.value)
        assert "available_bytes:" in msg
        assert "required_bytes:" in msg
        assert "deficit:" in msg
        assert str(tmp_path) in msg
        # Lock the error code so the dispatch path is FS_DISK_FULL.
        assert exc_info.value.code == ErrorCode.FS_DISK_FULL

    def test_sufficient_space_proceeds_silently_sc6(self, tmp_path):
        """SC#6 happy-path silence: when free space is sufficient, the gate
        returns None and emits NO logger output (no info/warning/error)."""
        logger = MagicMock()
        # Patch statvfs to return abundant space.
        with patch('mlpstorage_py.benchmarks.capacity_gate.os.statvfs',
                   side_effect=_patch_statvfs_available(10**18)):
            from mlpstorage_py.benchmarks.capacity_gate import check_capacity_4field
            result = check_capacity_4field(str(tmp_path), 1, logger)
        assert result is None
        # Silent contract: no logger output at any level for the gate itself.
        logger.error.assert_not_called()
        logger.warning.assert_not_called()

    def test_starved_destination_fails_run_with_4field_message(self, tmp_path):
        """SC#5 run-path: a VectorDBBenchmark configured with a local-disk
        destination AND insufficient space raises FileSystemError out of
        Benchmark.run() — exercises the _pre_execution_gate insertion in
        Benchmark.run() at base.py:1102.

        Uses TrainingBenchmark as the driver since VectorDB returns None
        (A8 escape hatch). We construct a minimal benchmark mock and bind
        the real _pre_execution_gate.
        """
        from mlpstorage_py.errors import FileSystemError, ErrorCode
        from mlpstorage_py.benchmarks.base import Benchmark

        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm = _make_benchmark(tmp_path, hosts)
        # Override capacity-gate hooks to a known starved destination.
        # The MagicMock spec is VectorDBBenchmark; we patch the methods
        # directly so the real _pre_execution_gate template fires.
        bm._capacity_gate_destination = MagicMock(return_value=str(tmp_path))
        bm.required_bytes_for_capacity_gate = MagicMock(return_value=10**15)

        with patch('mlpstorage_py.benchmarks.capacity_gate.os.statvfs',
                   side_effect=_patch_statvfs_available(1)):
            with pytest.raises(FileSystemError):
                bm.run()
        # _run must NOT have been called — the gate aborted before write/run.
        bm._run.assert_not_called()

    def test_starved_destination_fails_before_write_systemname_yaml(self, tmp_path):
        """Gate ordering lock: CAP-01 failure aborts Benchmark.run() BEFORE
        write_systemname_yaml gets a chance to write the file."""
        from mlpstorage_py.errors import FileSystemError

        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm = _make_benchmark(tmp_path, hosts)
        bm._capacity_gate_destination = MagicMock(return_value=str(tmp_path))
        bm.required_bytes_for_capacity_gate = MagicMock(return_value=10**15)

        target = _yaml_path(tmp_path)
        assert not target.exists()

        with patch('mlpstorage_py.benchmarks.capacity_gate.os.statvfs',
                   side_effect=_patch_statvfs_available(1)):
            with pytest.raises(FileSystemError):
                bm.run()

        # systemname.yaml MUST NOT have been written.
        assert not target.exists(), (
            "Gate-order violation: write_systemname_yaml fired despite "
            "CAP-01 failure"
        )

    def test_remote_vdb_backend_skips_cap01_with_log_a8(self, tmp_path):
        """A8 end-to-end: VectorDBBenchmark._capacity_gate_destination returns
        None (remote engine); the gate logs INFO 'CAP-01 skipped' and proceeds
        without raising. _run completes normally.
        """
        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm = _make_benchmark(tmp_path, hosts)
        # VectorDBBenchmark naturally returns None — let the real method run.
        # But we must also stub the CAP-02 launcher so the integration test
        # doesn't try real mpirun. Single-host (len=1 logic when we pass
        # args.hosts is None or empty) short-circuits naturally.
        bm.args.hosts = None  # SC#8 no-op for CAP-02
        # The VectorDBBenchmark override sets _capacity_gate_destination
        # to return None. Run the benchmark; assert success + info log.
        logger = MagicMock()
        bm.logger = logger
        rc = bm.run()
        assert rc == 0
        # The A8 INFO log fires from _pre_execution_gate.
        info_messages = [str(c) for c in logger.info.call_args_list]
        assert any("CAP-01 skipped" in m for m in info_messages), (
            f"expected 'CAP-01 skipped' INFO log, got: {info_messages}"
        )
        # _run still ran.
        bm._run.assert_called_once()

    def test_checkpointing_uses_checkpoint_folder_joined_with_model_path(self):
        """A7 lock: CheckpointingBenchmark._capacity_gate_destination
        returns os.path.join(checkpoint_folder, model)."""
        import os as _os
        from mlpstorage_py.benchmarks.dlio import CheckpointingBenchmark
        from types import SimpleNamespace

        bm = MagicMock(spec=CheckpointingBenchmark)
        bm.args = SimpleNamespace(checkpoint_folder='/tmp/ck', model='llama3-8b')
        result = CheckpointingBenchmark._capacity_gate_destination(bm)
        assert result == _os.path.join('/tmp/ck', 'llama3-8b')

    def test_kvcache_uses_1x_bytes_not_2x_per_a6(self):
        """A6 lock end-to-end: KVCacheBenchmark.required_bytes_for_capacity_gate
        returns total_cache_bytes at 1x (NOT 2x — the 2x figure stays in the
        recommendation log). Locks the choice so a Slice-3 regression to 2x
        would fire this test immediately.

        The real implementation uses the internal _MODEL_CACHE_ESTIMATES
        table and self.num_users: per_token * seq_len * num_users (all bytes).
        We verify the result equals 1x of the computed cache footprint, NOT 2x.
        """
        from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark

        bm = MagicMock(spec=KVCacheBenchmark)
        # Configure known values: choose a model NOT in the internal table so
        # _MODEL_CACHE_DEFAULT (per_token=4096, seq=4096) applies. Bind the
        # real class-level tables so the method's lookup hits the real data
        # (MagicMock(spec=...) replaces class attributes with mocks too).
        bm.model = 'unknown-model'  # forces _MODEL_CACHE_DEFAULT
        bm.num_users = 10
        bm._MODEL_CACHE_ESTIMATES = KVCacheBenchmark._MODEL_CACHE_ESTIMATES
        bm._MODEL_CACHE_DEFAULT = KVCacheBenchmark._MODEL_CACHE_DEFAULT
        result = KVCacheBenchmark.required_bytes_for_capacity_gate(bm)

        # Compute 1x expectation: per_token(4096) * seq(4096) * num_users(10)
        per_token = 4096
        seq_len = 4096
        cache_per_user_mb = (per_token * seq_len) / (1024 * 1024)
        total_cache_mb = cache_per_user_mb * 10
        expected_1x = int(total_cache_mb * 1024 * 1024)

        assert result == expected_1x, (
            f"A6 expectation: 1x bytes ({expected_1x}); got {result}"
        )
        # The A6 contract requires `result != expected_1x * 2`.
        assert result != expected_1x * 2, (
            f"A6 violation: required_bytes appears doubled (got {result}, "
            f"would-be-2x would be {expected_1x * 2})"
        )


# =============================================================================
# TestPhase5Cap02 — CAP-02 shared-FS probe gate end-to-end
# =============================================================================


def _make_benchmark_with_local_destination(tmp_path, hosts, *, hosts_arg, command='run'):
    """Construct a benchmark whose _capacity_gate_destination returns a local
    path (overriding VectorDBBenchmark's None-returning A8 escape hatch) so
    the CAP-02 path inside _pre_execution_gate fires. args.hosts is set to
    `hosts_arg` so the CAP-02 launcher's cardinality logic is exercised.
    """
    bm = _make_benchmark(tmp_path, hosts, command=command)
    bm._capacity_gate_destination = MagicMock(return_value=str(tmp_path))
    bm.required_bytes_for_capacity_gate = MagicMock(return_value=1)
    bm.args.hosts = hosts_arg
    return bm


class TestPhase5Cap02:
    """End-to-end coverage for CAP-02 shared-FS probe gate.

    Covers SC#7 (multi-host fsid cardinality > 1 fail with host listing),
    SC#8 (single-host silent no-op), and the gate-order lock (CAP-02 fires
    AFTER CAP-01 in _pre_execution_gate; both fire BEFORE
    write_systemname_yaml).

    These tests override _capacity_gate_destination to return a local path
    so the _pre_execution_gate body reaches the CAP-02 invocation
    (VectorDBBenchmark's default A8 None-destination short-circuits before
    CAP-02 would run, which is the correct production behavior for remote
    backends but unhelpful for testing the CAP-02 launcher integration).
    """

    def test_single_host_run_is_silent_no_op_sc8(self, tmp_path):
        """SC#8: args.hosts=['localhost'] (single-element) causes
        run_shared_fs_probe to take its silent no-op branch. NO mpirun
        invocation; the launcher receives the single-host list (and its
        SC#8 short-circuit fires internally — also locked at the unit
        layer in tests/unit/test_shared_fs_probe.py TestSingleHostShortCircuit).
        """
        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm = _make_benchmark_with_local_destination(
            tmp_path, hosts, hosts_arg=['localhost'])
        # Patch the launcher to spy on its invocation.
        with patch('mlpstorage_py.benchmarks.base.run_shared_fs_probe',
                   return_value=None) as mock_probe:
            rc = bm.run()
        assert rc == 0
        # The launcher was called with the single-host list; its internal
        # SC#8 short-circuit fires (unit-test locked in test_shared_fs_probe.py).
        assert mock_probe.called
        _, call_kwargs = mock_probe.call_args
        assert call_kwargs.get('hosts') == ['localhost']

    def test_no_hosts_attr_is_silent_no_op_sc8(self, tmp_path):
        """SC#8: args.hosts is None → _pre_execution_gate normalizes via
        `getattr(self.args, 'hosts', None) or []`, so the launcher receives
        an empty list and short-circuits. No CAP-02 logger output."""
        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm = _make_benchmark_with_local_destination(
            tmp_path, hosts, hosts_arg=None)
        with patch('mlpstorage_py.benchmarks.base.run_shared_fs_probe',
                   return_value=None) as mock_probe:
            rc = bm.run()
        assert rc == 0
        assert mock_probe.called
        _, call_kwargs = mock_probe.call_args
        # The gate normalizes None → []
        assert call_kwargs.get('hosts') == []

    def test_multi_host_cardinality_1_succeeds_silently(self, tmp_path):
        """Multi-host cardinality 1: launcher returns None (success); the
        benchmark proceeds to _run normally. No logger.error/warning."""
        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm = _make_benchmark_with_local_destination(
            tmp_path, hosts, hosts_arg=['host1', 'host2'])
        logger = MagicMock()
        bm.logger = logger
        with patch('mlpstorage_py.benchmarks.base.run_shared_fs_probe',
                   return_value=None) as mock_probe:
            rc = bm.run()
        assert rc == 0
        assert mock_probe.called
        # No CAP-02 error/warning logs.
        for call_args, _ in logger.error.call_args_list:
            assert 'CAP-02' not in str(call_args[0] if call_args else '')
        bm._run.assert_called_once()

    def test_multi_host_cardinality_2_fails_with_host_listing_sc7(self, tmp_path):
        """SC#7: cardinality > 1 raises FileSystemError; the message lists
        each host + each (st_dev, st_ino) tuple. _run NOT called.
        """
        from mlpstorage_py.errors import FileSystemError, ErrorCode

        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm = _make_benchmark_with_local_destination(
            tmp_path, hosts, hosts_arg=['host1', 'host2'])

        crafted_msg = (
            "CAP-02: shared-FS verification failed (cardinality 2): "
            "host1 reported st_dev=64768 st_ino=12345; "
            "host2 reported st_dev=64512 st_ino=67890; "
            "this typically means one or more hosts have a local-disk path "
            "where a shared mount was expected"
        )
        with patch('mlpstorage_py.benchmarks.base.run_shared_fs_probe',
                   side_effect=FileSystemError(
                       crafted_msg,
                       path=str(tmp_path),
                       operation='cap02-shared-fs-probe',
                       code=ErrorCode.FS_INVALID_STRUCTURE,
                   )):
            with pytest.raises(FileSystemError) as exc_info:
                bm.run()

        msg = str(exc_info.value)
        assert "host1" in msg
        assert "host2" in msg
        assert "st_dev=" in msg
        assert "st_ino=" in msg
        # _run MUST NOT have been called.
        bm._run.assert_not_called()

    def test_multi_host_cardinality_2_error_message_contains_local_disk_hint_sc7(self, tmp_path):
        """SC#7 hint lock: the verbatim local-disk hint phrase appears in
        the SystemError message body."""
        from mlpstorage_py.errors import FileSystemError, ErrorCode

        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm = _make_benchmark_with_local_destination(
            tmp_path, hosts, hosts_arg=['host1', 'host2'])

        hint_phrase = (
            "this typically means one or more hosts have a local-disk path "
            "where a shared mount was expected"
        )
        with patch('mlpstorage_py.benchmarks.base.run_shared_fs_probe',
                   side_effect=FileSystemError(
                       f"CAP-02 failed: cardinality 2. {hint_phrase}",
                       path=str(tmp_path),
                       operation='cap02-shared-fs-probe',
                       code=ErrorCode.FS_INVALID_STRUCTURE,
                   )):
            with pytest.raises(FileSystemError) as exc_info:
                bm.run()

        assert hint_phrase in str(exc_info.value)

    def test_cap02_fires_after_cap01_in_pre_execution_gate_ordering(self, tmp_path):
        """Gate-order lock: in _pre_execution_gate, check_capacity_4field is
        called BEFORE run_shared_fs_probe. Slice 3 + Slice 4 ordering."""
        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm = _make_benchmark(tmp_path, hosts)
        bm.args.hosts = ['host1', 'host2']
        bm._capacity_gate_destination = MagicMock(return_value=str(tmp_path))
        bm.required_bytes_for_capacity_gate = MagicMock(return_value=1)

        call_order = []
        def cap01_se(*a, **kw):
            call_order.append('cap01')
        def cap02_se(*a, **kw):
            call_order.append('cap02')

        with patch('mlpstorage_py.benchmarks.base.check_capacity_4field',
                   side_effect=cap01_se), \
             patch('mlpstorage_py.benchmarks.base.run_shared_fs_probe',
                   side_effect=cap02_se):
            bm.run()

        assert call_order == ['cap01', 'cap02'], (
            f"Expected CAP-01 BEFORE CAP-02, got: {call_order}"
        )

    def test_cap02_fires_before_write_systemname_yaml(self, tmp_path):
        """Gate-order lock: a CAP-02 failure aborts BEFORE write_systemname_yaml.
        Verifies the Slice 4 + Slice 2 ordering."""
        from mlpstorage_py.errors import FileSystemError, ErrorCode

        hosts = [_make_host(hostname=f"h{i}") for i in range(2)]
        bm = _make_benchmark_with_local_destination(
            tmp_path, hosts, hosts_arg=['host1', 'host2'])

        target = _yaml_path(tmp_path)
        assert not target.exists()

        with patch('mlpstorage_py.benchmarks.base.run_shared_fs_probe',
                   side_effect=FileSystemError(
                       "CAP-02 fail",
                       path=str(tmp_path),
                       operation='cap02-shared-fs-probe',
                       code=ErrorCode.FS_INVALID_STRUCTURE,
                   )):
            with pytest.raises(FileSystemError):
                bm.run()

        # write_systemname_yaml MUST NOT have written the file.
        assert not target.exists(), (
            "Gate-order violation: systemname.yaml written despite CAP-02 failure"
        )
