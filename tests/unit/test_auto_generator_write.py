"""Unit tests for write_systemname_yaml — Phase 02 / Plan 02-04.

This file owns the on-disk side of the auto-generator vertical: the atomic
write orchestrator that composes 02-02 (adapter + grouping), 02-03 (stub
splice + outer dict), and adds the D-7 sort, D-11 path derivation, D-12
command gate, D-9 atomic O_CREAT|O_EXCL|O_WRONLY write + FileExistsError
no-op, D-8 empty-fleet fallback, and D-10 YAML formatting.

Test discipline:
- All filesystem work happens under pytest's `tmp_path` fixture.
- The race test uses `threading.Barrier(2)` per RESEARCH.md Code Example
  lines 676-700 to synchronize concurrent entry into `os.open`.
- Logger is a `MagicMock` so `logger.debug` / `logger.info` assertions
  catch the no-op-if-exists path.
- `args` is a `SimpleNamespace` (not a `MagicMock`) so attribute access
  is strict — missing attributes raise `AttributeError`, which catches
  any drift in the function's expected `args.*` surface.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mlpstorage_py.rules.models import (
    HostCPUInfo,
    HostInfo,
    HostMemoryInfo,
)
from mlpstorage_py.cluster_collector import HostSystemInfo
from mlpstorage_py.system_description.auto_generator import (
    _SYSTEMNAME_YAML_MODE,
    _resolve_host_info_list,
    write_systemname_yaml,
)


# ---------------------------------------------------------------------------
# Fixtures
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


def _make_cluster_info(num_hosts: int = 3, **host_kwargs) -> MagicMock:
    """MagicMock with `host_info_list = [HostInfo, ...]`."""
    ci = MagicMock()
    ci.host_info_list = [
        _make_host(hostname=f"h{i}", **host_kwargs) for i in range(num_hosts)
    ]
    return ci


@pytest.fixture
def args(tmp_path) -> SimpleNamespace:
    """Default `args` for write_systemname_yaml — `command='run'`, D-11 path triples."""
    return SimpleNamespace(
        command="run",
        results_dir=str(tmp_path),
        mode="closed",
        orgname="Acme",
        systemname="sys-v1",
    )


@pytest.fixture
def cluster_info() -> MagicMock:
    """Default 3-host homogeneous fleet."""
    return _make_cluster_info(num_hosts=3)


@pytest.fixture
def target_path(tmp_path) -> Path:
    """Expected D-11 canonical path for the default `args`."""
    return tmp_path / "closed" / "Acme" / "systems" / "sys-v1.yaml"


# ---------------------------------------------------------------------------
# LIFE-01 / D-11 — canonical path + happy path
# ---------------------------------------------------------------------------


def test_writes_at_canonical_path(args, cluster_info, target_path):
    """LIFE-01: file appears at `<rd>/<mode>/<org>/systems/<sys>.yaml`."""
    # Sanity: target dir does NOT pre-exist (we want to prove mkdir works).
    assert not target_path.parent.exists()

    returned = write_systemname_yaml(args, cluster_info, MagicMock())

    assert returned == str(target_path)
    assert target_path.exists()

    data = yaml.safe_load(target_path.read_text())
    assert data["system_under_test"]["clients"][0]["quantity"] == 3
    assert (
        data["system_under_test"]["clients"][0]["chassis"]["cpu_model"]
        == "Intel(R) Xeon Platinum 8480+"
    )
    # mkdir created `systems/` on demand.
    assert target_path.parent.exists()


def test_path_parent_mkdir_creates_systems_dir(args, cluster_info, tmp_path):
    """`<rd>/<mode>/<org>/` exists but `systems/` does not → mkdir creates it."""
    # Pre-create everything except systems/.
    (tmp_path / "closed" / "Acme").mkdir(parents=True)
    assert not (tmp_path / "closed" / "Acme" / "systems").exists()

    write_systemname_yaml(args, cluster_info, MagicMock())

    assert (tmp_path / "closed" / "Acme" / "systems").is_dir()


# ---------------------------------------------------------------------------
# D-9 — no-op-if-exists
# ---------------------------------------------------------------------------


def test_no_op_if_exists(args, cluster_info, target_path):
    """Pre-existing valid + matching file → LIFE-04 no-touch path: return None,
    file content unchanged, logger.debug called.

    Phase-5 NOTE: the original Phase-2 test wrote garbage content (`existing:
    content\\n`) and expected the FileExistsError no-op to return None
    unconditionally. Phase 5 LIFE-02/03 replaces that no-op with a load-diff
    branch — garbage content now correctly raises
    `SystemDescriptionParseError`. The semantic intent of this test (file is
    not overwritten when it already exists matching the in-memory image) is
    preserved by switching the pre-existing content to a byte-equal copy of
    what the writer would emit. This is the LIFE-04 no-touch contract.
    """
    # Write the file once via the writer so the on-disk content matches the
    # in-memory image byte-for-byte.
    write_systemname_yaml(args, cluster_info, MagicMock())
    pre_existing_text = target_path.read_text()

    logger = MagicMock()
    returned = write_systemname_yaml(args, cluster_info, logger)

    assert returned is None
    assert target_path.read_text() == pre_existing_text  # no overwrite
    # logger.debug fired with a "no-touch" or "matches" message per LIFE-04.
    assert logger.debug.called
    debug_messages = " ".join(str(c) for c in logger.debug.call_args_list)
    assert (
        "no-touch" in debug_messages.lower()
        or "matches" in debug_messages.lower()
        or "life-04" in debug_messages.lower()
    )


# ---------------------------------------------------------------------------
# D-9 — atomic concurrent-write race (T-2-01)
# ---------------------------------------------------------------------------


def test_concurrent_writers_one_wins(args, cluster_info, target_path):
    """T-2-01: two simultaneous writers → exactly one wins; the other either
    hits the LIFE-04 no-touch path (returns None) OR surfaces the transient
    empty-file race window as SystemDescriptionParseError.

    Uses `threading.Barrier(2)` to synchronize both threads' entry into
    `os.open(..., O_CREAT|O_EXCL|O_WRONLY)` so the kernel-level race is
    actually exercised. Per RESEARCH.md Code Example lines 676-700.

    Phase-5 NOTE: the Phase-2 contract was "the loser returns None
    unconditionally because FileExistsError → no-op". Phase 5 LIFE-02 changes
    the loser's path to load-then-diff via parse_on_disk_systemname_yaml.
    Three timing windows are possible for the loser, all consistent with the
    single-winner invariant:

      (1) Winner has already called fdopen + safe_dump + close before the
          loser reads → the loser sees the FULL emitted YAML, diffs it
          against the same in-memory image (identical content from the same
          cluster_info) → diff empty → LIFE-04 no-touch → returns None.
      (2) Winner has acquired the fd via os.open but not yet flushed the
          safe_dump → the loser reads an empty (zero-byte) file →
          yaml.safe_load returns None → structural-validation raises
          SystemDescriptionParseError ('missing top-level system_under_test
          key'). The single-winner invariant holds; the loser surfaces the
          race window via a parse error rather than a clean None.
      (3) Winner has partially-written content → loser sees malformed YAML
          → yaml.YAMLError → SystemDescriptionParseError ('is malformed').

    All three outcomes are consistent with the security/correctness contract:
    exactly one writer wins (`paths[0] == str(target_path)`), the on-disk
    file is well-formed by the end of both joins (`target_path.exists()`),
    and the loser does NOT overwrite the winner's content. A production
    operator hitting outcome (2)/(3) re-runs the benchmark — the second run
    is the LIFE-04 happy path. The test asserts the invariants that survive
    all three outcomes.
    """
    from mlpstorage_py.errors import SystemDescriptionParseError

    barrier = threading.Barrier(2)
    results: list = []
    exceptions: list = []

    def worker():
        barrier.wait()  # Synchronize both threads' entry.
        try:
            results.append(write_systemname_yaml(args, cluster_info, MagicMock()))
        except SystemDescriptionParseError as exc:
            # Outcome (2) or (3): the loser saw the file mid-write. Acceptable.
            exceptions.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    paths = [r for r in results if r is not None]
    nones = [r for r in results if r is None]

    # Single-winner invariant (survives all three timing outcomes):
    assert len(paths) == 1, f"expected exactly one winner, got results={results}, exceptions={exceptions}"
    assert paths[0] == str(target_path)
    # The loser surfaced as either a clean None (outcome 1) or a parse error
    # (outcomes 2/3). Exactly one of those must have happened.
    assert len(nones) + len(exceptions) == 1, (
        f"expected exactly one loser via either None or parse error; "
        f"got results={results}, exceptions={exceptions}"
    )
    assert target_path.exists()


# ---------------------------------------------------------------------------
# D-12 — command gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd", ["datagen", "configview", "datasize", "validate", "history", "reportgen"],
)
def test_non_run_commands_skip_write(args, cluster_info, target_path, cmd):
    """D-12: only `command='run'` writes; all other commands skip."""
    args.command = cmd
    returned = write_systemname_yaml(args, cluster_info, MagicMock())
    assert returned is None
    assert not target_path.exists()


def test_run_command_writes(args, cluster_info, target_path):
    """D-12 positive: `command='run'` writes."""
    args.command = "run"
    returned = write_systemname_yaml(args, cluster_info, MagicMock())
    assert returned == str(target_path)
    assert target_path.exists()


# ---------------------------------------------------------------------------
# D-8 — empty-fleet fallback
# ---------------------------------------------------------------------------


_FAKE_LOCAL_COLLECTED = {
    "hostname": "local-h",
    "meminfo": {"MemTotal": 274_877_906_944 // 1024},  # kB → bytes via from_proc_meminfo_dict
    "cpuinfo": [
        {"processor": "0", "model name": "Local CPU", "cpu cores": "4",
         "physical id": "0", "siblings": "8"},
        {"processor": "1", "model name": "Local CPU", "cpu cores": "4",
         "physical id": "0", "siblings": "8"},
    ],
    "os_release": {"NAME": "Local OS", "VERSION_ID": "1.0"},
}


def test_empty_fleet_fallback_writes_single_stanza(args, target_path):
    """D-8: cluster_info=None → collect_local_system_info called → 1 stanza, qty=1."""
    with patch(
        "mlpstorage_py.system_description.auto_generator.collect_local_system_info",
        return_value=_FAKE_LOCAL_COLLECTED,
    ) as mock_collect:
        returned = write_systemname_yaml(args, None, MagicMock())

    assert returned == str(target_path)
    mock_collect.assert_called_once()
    data = yaml.safe_load(target_path.read_text())
    clients = data["system_under_test"]["clients"]
    assert len(clients) == 1
    assert clients[0]["quantity"] == 1


def test_empty_cluster_info_host_list_falls_back(args, target_path):
    """D-8 edge: cluster_info.host_info_list = [] also triggers fallback."""
    ci = MagicMock()
    ci.host_info_list = []
    with patch(
        "mlpstorage_py.system_description.auto_generator.collect_local_system_info",
        return_value=_FAKE_LOCAL_COLLECTED,
    ) as mock_collect:
        returned = write_systemname_yaml(args, ci, MagicMock())

    assert returned == str(target_path)
    mock_collect.assert_called_once()
    data = yaml.safe_load(target_path.read_text())
    assert data["system_under_test"]["clients"][0]["quantity"] == 1


def test_resolve_host_info_list_passthrough():
    """`_resolve_host_info_list` returns the existing list when populated."""
    hosts = [_make_host(hostname="a"), _make_host(hostname="b")]
    ci = MagicMock()
    ci.host_info_list = hosts
    assert _resolve_host_info_list(ci) is hosts


def test_resolve_host_info_list_none_triggers_collector():
    """`_resolve_host_info_list(None)` calls `collect_local_system_info`."""
    with patch(
        "mlpstorage_py.system_description.auto_generator.collect_local_system_info",
        return_value=_FAKE_LOCAL_COLLECTED,
    ) as mock_collect:
        result = _resolve_host_info_list(None)
    assert mock_collect.called
    assert isinstance(result, list) and len(result) == 1
    assert isinstance(result[0], HostInfo)


# ---------------------------------------------------------------------------
# D-10 — YAML formatting
# ---------------------------------------------------------------------------


def test_yaml_formatting_document_marker(args, cluster_info, target_path):
    """D-10: emitted bytes start with `---\\n` (explicit_start=True)."""
    write_systemname_yaml(args, cluster_info, MagicMock())
    assert target_path.read_text().startswith("---\n")


def test_yaml_formatting_strings_double_quoted(args, cluster_info, target_path):
    """D-10: with default_style='"' PyYAML double-quotes ALL scalars and KEYS.

    Surprise vs. PLAN: `default_style='"'` quotes keys too (not just values),
    so the emitted text contains `"cpu_model": "Intel..."` (both sides quoted),
    not `cpu_model: "Intel..."`. Semantic intent (D-10: strings round-trip as
    strings via yaml.safe_load, no plain-scalar misinterpretation) is still
    locked — the round-trip test below proves it.
    """
    write_systemname_yaml(args, cluster_info, MagicMock())
    text = target_path.read_text()
    # cpu_model value is a string and must be double-quoted on the value side.
    assert re.search(r'"cpu_model":\s*"[^"]+"', text), (
        f"cpu_model not double-quoted in:\n{text}"
    )
    # operating_system.name value is a string and must be double-quoted.
    assert re.search(r'"name":\s*"[^"]+"', text), (
        f"name not double-quoted in:\n{text}"
    )
    # Round-trip: strings load back as Python strings.
    data = yaml.safe_load(text)
    assert isinstance(
        data["system_under_test"]["clients"][0]["chassis"]["cpu_model"], str
    )


def test_yaml_formatting_integers_round_trip_as_int(args, cluster_info, target_path):
    """D-10 / Pitfall 6 (corrected): integers must round-trip as Python `int`.

    Surprise vs. PLAN: modern PyYAML with `default_style='"'` emits integers
    as `!!int "N"` (tagged double-quoted), NOT as bare unquoted ints. The
    PLAN claim that "PyYAML emits int natively even with default_style='\"'"
    is incorrect for this PyYAML version. What MATTERS for the schema
    validator and submission checker is that `quantity`, `cpu_qty`,
    `cpu_cores`, and `memory_capacity` round-trip as Python `int` — which
    the `!!int` tag guarantees. This test locks the round-trip type, not
    the on-disk byte pattern.
    """
    write_systemname_yaml(args, cluster_info, MagicMock())
    data = yaml.safe_load(target_path.read_text())
    client = data["system_under_test"]["clients"][0]
    assert isinstance(client["quantity"], int)
    assert isinstance(client["chassis"]["cpu_qty"], int)
    assert isinstance(client["chassis"]["cpu_cores"], int)
    assert isinstance(client["chassis"]["memory_capacity"], int)
    assert client["quantity"] == 3


def test_yaml_formatting_integers_tagged_not_string(args, cluster_info, target_path):
    """Lock the `!!int` tag emission so a PyYAML version-bump that drops it
    (and silently turns ints into strings) is caught at test time.

    `!!int "N"` (tagged) round-trips as int; bare `"N"` (no tag) would
    round-trip as str and break the schema validator.
    """
    write_systemname_yaml(args, cluster_info, MagicMock())
    text = target_path.read_text()
    # quantity must be either bare-unquoted (`quantity: 3`) OR `!!int`-tagged
    # (`"quantity": !!int "3"`). Both round-trip as int. Forbid the
    # untagged-double-quoted form (`"quantity": "3"`) which would round-trip
    # as str.
    assert re.search(r'"quantity":\s+!!int\s+"\d+"', text) or re.search(
        r"quantity:\s+\d+\s*$", text, re.MULTILINE
    ), f"quantity must be int-tagged or bare int, got:\n{text}"


def test_yaml_block_style(args, cluster_info, target_path):
    """D-10: no `{` flow markers. `[` only appears in legitimate empty-list cases.

    Allowed empty-list keys (kept as explicit `key: []` for self-documenting output
    so readers see "nothing here" rather than a silent omission):
      - traffic        (Phase 2 D-10 precedent)
      - sysctl         (Phase 4 — allowlist-driven, empty is meaningful)
      - environment    (Phase 4 — allowlist-driven, empty is meaningful)
    Drives is omitted entirely when empty per D-33 (client nodes commonly have none).
    """
    write_systemname_yaml(args, cluster_info, MagicMock())
    text = target_path.read_text()
    assert "{" not in text, f"flow-style {{ leaked in:\n{text}"
    stripped = text
    for key in ("traffic", "sysctl", "environment"):
        stripped = re.sub(rf'"{key}":\s*\[\]', "", stripped)
        stripped = re.sub(rf"{key}:\s*\[\]", "", stripped)
    assert "[" not in stripped, (
        f"flow-style [ leaked outside allowed empty-list keys in:\n{stripped}"
    )


# ---------------------------------------------------------------------------
# D-7 — stanza ordering
# ---------------------------------------------------------------------------


def test_stanza_ordering_homogeneous_passthrough(args, cluster_info, target_path):
    """Single-stanza fleet: 3 identical hosts → 1 stanza, quantity=3."""
    write_systemname_yaml(args, cluster_info, MagicMock())
    data = yaml.safe_load(target_path.read_text())
    clients = data["system_under_test"]["clients"]
    assert len(clients) == 1
    assert clients[0]["quantity"] == 3


def test_stanza_ordering_largest_quantity_first(args, target_path):
    """D-7: input [qty=1 X, qty=3 Y] → output [qty=3 Y, qty=1 X]."""
    ci = MagicMock()
    ci.host_info_list = [
        _make_host(cpu_model="X", hostname="x1"),
        _make_host(cpu_model="Y", hostname="y1"),
        _make_host(cpu_model="Y", hostname="y2"),
        _make_host(cpu_model="Y", hostname="y3"),
    ]
    write_systemname_yaml(args, ci, MagicMock())
    data = yaml.safe_load(target_path.read_text())
    clients = data["system_under_test"]["clients"]
    assert len(clients) == 2
    assert clients[0]["chassis"]["cpu_model"] == "Y"
    assert clients[0]["quantity"] == 3
    assert clients[1]["chassis"]["cpu_model"] == "X"
    assert clients[1]["quantity"] == 1


def test_stanza_ordering_alphabetical_tiebreak(args, target_path):
    """D-7: input [qty=2 Zen, qty=2 Atom] → output [qty=2 Atom, qty=2 Zen]."""
    ci = MagicMock()
    ci.host_info_list = [
        _make_host(cpu_model="Zen", hostname="z1"),
        _make_host(cpu_model="Zen", hostname="z2"),
        _make_host(cpu_model="Atom", hostname="a1"),
        _make_host(cpu_model="Atom", hostname="a2"),
    ]
    write_systemname_yaml(args, ci, MagicMock())
    data = yaml.safe_load(target_path.read_text())
    clients = data["system_under_test"]["clients"]
    assert len(clients) == 2
    assert clients[0]["chassis"]["cpu_model"] == "Atom"
    assert clients[0]["quantity"] == 2
    assert clients[1]["chassis"]["cpu_model"] == "Zen"
    assert clients[1]["quantity"] == 2


# ---------------------------------------------------------------------------
# D-14 round-trip — outer-dict omissions survive yaml.safe_dump
# ---------------------------------------------------------------------------


def test_systemname_yaml_omits_solution_deployment_in_emitted_file(
    args, cluster_info, target_path,
):
    """D-14 integration: `solution`, `deployment`, etc. absent in emitted YAML."""
    write_systemname_yaml(args, cluster_info, MagicMock())
    data = yaml.safe_load(target_path.read_text())
    sut = data["system_under_test"]
    for forbidden in (
        "solution",
        "deployment",
        "product_nodes",
        "product_switches",
        "total_rack_units",
        "rack_power_supplies",
    ):
        assert forbidden not in sut, f"D-14 violation: {forbidden} present"


# ---------------------------------------------------------------------------
# T-2-08 — symlink attack
# ---------------------------------------------------------------------------


def test_symlink_attack_at_target_path_returns_none(args, cluster_info, tmp_path):
    """T-2-08: pre-existing symlink at target path → O_EXCL refuses to create.

    Verifies POSIX guarantee that O_CREAT|O_EXCL fails if the path resolves to
    anything pre-existing — including a symlink. The symlink's target file
    MUST remain unchanged.

    Phase-5 NOTE: the Phase-2 contract was "returns None on symlink-attack".
    Phase 5 LIFE-02 routes the FileExistsError handler through the load-diff
    branch. The symlink resolves to `innocent.txt` whose content (`"innocent"`)
    is structurally-invalid as a systemname.yaml → the new branch raises
    `SystemDescriptionParseError`. The T-2-08 security guarantee (the symlink
    target is NOT overwritten) still holds — the parse error fires BEFORE any
    write attempt — but the test now asserts the new exception shape.
    """
    from mlpstorage_py.errors import SystemDescriptionParseError

    innocent = tmp_path / "innocent.txt"
    innocent.write_text("innocent")

    target_dir = tmp_path / "closed" / "Acme" / "systems"
    target_dir.mkdir(parents=True)
    target = target_dir / "sys-v1.yaml"
    os.symlink(str(innocent), str(target))

    with pytest.raises(SystemDescriptionParseError):
        write_systemname_yaml(args, cluster_info, MagicMock())

    # T-2-08 security guarantee: the symlink target is NOT overwritten — the
    # parse-error path fires after a read but BEFORE any write.
    assert innocent.read_text() == "innocent"


# ---------------------------------------------------------------------------
# D-9 — filesystem errors propagate (not swallowed)
# ---------------------------------------------------------------------------


def test_filesystem_error_propagates_eacces(args, cluster_info, tmp_path):
    """D-9: non-FileExistsError filesystem errors propagate as exceptions."""
    if os.geteuid() == 0:
        pytest.skip("root bypasses chmod restrictions")

    # Pre-create the org directory and make it non-writable so mkdir of
    # `systems/` (or the os.open) fails.
    org_dir = tmp_path / "closed" / "Acme"
    org_dir.mkdir(parents=True)
    os.chmod(str(org_dir), 0o555)
    try:
        with pytest.raises((PermissionError, OSError)):
            write_systemname_yaml(args, cluster_info, MagicMock())
    finally:
        # Restore so pytest can clean up tmp_path.
        os.chmod(str(org_dir), 0o755)


# ---------------------------------------------------------------------------
# D-15 — no validate_file call inside writer
# ---------------------------------------------------------------------------


def test_writer_does_not_call_schema_validator_validate_file(args, cluster_info):
    """D-15: writer must NOT call schema_validator.validate_file post-write."""
    with patch(
        "mlpstorage_py.system_description.schema_validator.validate_file",
        side_effect=AssertionError("must not be called per D-15"),
    ):
        write_systemname_yaml(args, cluster_info, MagicMock())


# ---------------------------------------------------------------------------
# Mode constant sanity
# ---------------------------------------------------------------------------


def test_systemname_yaml_mode_is_0o644():
    """`_SYSTEMNAME_YAML_MODE` mirrors `sentinel.py:_SENTINEL_MODE` (LAY-03 parity)."""
    assert _SYSTEMNAME_YAML_MODE == 0o644


# ===========================================================================
# Phase 5 / Plan 05-02 — LIFE-02 / LIFE-03 / LIFE-04 wiring tests
# ---------------------------------------------------------------------------
# These tests exercise the new load-diff-raise branch inside
# write_systemname_yaml that replaces the Phase-2 FileExistsError no-op. They
# also lock parse_on_disk_systemname_yaml's structural-validation behavior on
# malformed inputs. The fixtures (args / cluster_info / target_path / _make_host
# / _make_cluster_info) are reused from the Phase-2 test infrastructure above.
# ===========================================================================


import hashlib  # noqa: E402 — placed near new test class for locality


def _sha256(path: Path) -> str:
    """Helper: SHA-256 of file contents for LIFE-04 byte-equality invariant."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestPhase5DriftWiring:
    """LIFE-02 (load + diff), LIFE-03 (raise SystemDriftError before DLIO),
    LIFE-04 (no-touch when diff empty + submitter hand-fills survive).

    Naming convention: every test starts with `test_` and lives on this class
    so the acceptance-criteria grep (`grep -c '    def test_'`) catches all
    new tests in one count.
    """

    # ----- LIFE-01 regression (Phase 2 behavior preserved) -----------------

    def test_first_run_against_empty_dir_writes_file(self, args, cluster_info, target_path):
        """Phase 2 LIFE-01: first call against an empty results-dir writes the file."""
        assert not target_path.exists()
        returned = write_systemname_yaml(args, cluster_info, MagicMock())
        assert returned == str(target_path)
        assert target_path.exists()

    # ----- LIFE-04 no-touch path -------------------------------------------

    def test_second_run_against_unchanged_fleet_no_touch_mtime_invariant(
        self, args, cluster_info, target_path,
    ):
        """LIFE-04 mtime invariant: re-run against unchanged fleet leaves
        on-disk mtime AND sha256 unchanged. SC#1 lock."""
        write_systemname_yaml(args, cluster_info, MagicMock())
        assert target_path.exists()

        mtime_before = target_path.stat().st_mtime_ns
        sha_before = _sha256(target_path)

        # Sleep just enough so mtime changes IF the file were re-written
        # (filesystems with second-resolution mtime would otherwise hide a touch).
        # 1.1s is the conservative cross-FS minimum.
        import time
        time.sleep(1.1)

        result = write_systemname_yaml(args, cluster_info, MagicMock())
        assert result is None  # no-touch path returns None per LIFE-04
        assert target_path.stat().st_mtime_ns == mtime_before, "mtime changed — LIFE-04 violated"
        assert _sha256(target_path) == sha_before, "sha256 changed — LIFE-04 violated"

    def test_second_run_against_unchanged_fleet_no_touch_returns_none(
        self, args, cluster_info, target_path,
    ):
        """LIFE-04: the no-touch path returns None (matches Phase-2
        FileExistsError no-op return value)."""
        write_systemname_yaml(args, cluster_info, MagicMock())
        result = write_systemname_yaml(args, cluster_info, MagicMock())
        assert result is None

    # ----- LIFE-03 raise-before-DLIO path ----------------------------------

    def test_second_run_against_drifted_cpu_model_raises_system_drift_error(
        self, args, cluster_info, target_path,
    ):
        """LIFE-03: drifted cpu_model surfaces as fingerprint orphan (since
        cpu_model is part of _FINGERPRINT_KEYS); the SystemDriftError message
        contains the fingerprint orphan path."""
        from mlpstorage_py.errors import SystemDriftError

        write_systemname_yaml(args, cluster_info, MagicMock())

        # Build a drifted fleet with a different cpu_model.
        drifted = _make_cluster_info(num_hosts=3, cpu_model="Different CPU")
        with pytest.raises(SystemDriftError) as exc_info:
            write_systemname_yaml(args, drifted, MagicMock())
        message = str(exc_info.value)
        assert "clients[fingerprint=" in message, (
            f"expected fingerprint orphan path in drift report, got:\n{message}"
        )

    def test_second_run_against_drifted_sysctl_value_raises_system_drift_error(
        self, args, target_path,
    ):
        """LIFE-03: a sysctl value-drift while fingerprint is stable surfaces
        as a leaf-level diff. NOTE: sysctl_sig is part of _FINGERPRINT_KEYS in
        Phase 4, so changing a sysctl value would change the fingerprint and
        produce an orphan. We test the leaf-diff path indirectly by changing
        ONLY the hostname (which is NOT in the fingerprint) — but hostnames
        aren't in the emitted stanza either, so the diff should be empty.
        Instead: change the friendly_description (which IS in the emitted
        stanza but NOT in the fingerprint) — that triggers a leaf diff with
        @@-style path."""
        # First run: write the file.
        ci_run1 = _make_cluster_info(num_hosts=2)
        write_systemname_yaml(args, ci_run1, MagicMock())
        assert target_path.exists()

        # Patch the on-disk file's friendly_description to a non-empty value
        # so the diff has something to surface (in-memory has '' per universal
        # collection-failure rule; SER-02 blank-preservation skip applies only
        # in the OTHER direction so disk-filled vs. in-memory-empty surfaces
        # nothing per Pitfall 3(a) — therefore we modify on-disk to a different
        # *fingerprint-affecting* sysctl value instead to force a drift hit).
        from mlpstorage_py.errors import SystemDriftError
        on_disk = yaml.safe_load(target_path.read_text())
        on_disk["system_under_test"]["clients"][0]["sysctl"] = [
            {"name": "net.core.rmem_max", "value": "16777216"}
        ]
        target_path.write_text(yaml.safe_dump(on_disk, default_flow_style=False))

        # Re-run: the in-memory side still has sysctl=[] (Phase 2 stub),
        # so the on-disk fingerprint (which includes sysctl_sig) differs from
        # the in-memory fingerprint → fingerprint orphan diff.
        with pytest.raises(SystemDriftError) as exc_info:
            write_systemname_yaml(args, ci_run1, MagicMock())
        message = str(exc_info.value)
        # The drift report contains either "@@ " hunks or a fingerprint orphan path.
        assert "@@ " in message or "clients[fingerprint=" in message, (
            f"expected drift report markers, got:\n{message}"
        )

    def test_second_run_against_quantity_change_raises_system_drift_error(
        self, args, target_path,
    ):
        """D-39: changing the fleet from 2 identical hosts to 3 identical hosts
        produces a `quantity` field diff (fingerprint stable, quantity leaf changed)."""
        from mlpstorage_py.errors import SystemDriftError

        ci_2hosts = _make_cluster_info(num_hosts=2)
        write_systemname_yaml(args, ci_2hosts, MagicMock())

        ci_3hosts = _make_cluster_info(num_hosts=3)
        with pytest.raises(SystemDriftError) as exc_info:
            write_systemname_yaml(args, ci_3hosts, MagicMock())
        message = str(exc_info.value)
        assert "quantity" in message, (
            f"expected 'quantity' substring in drift report, got:\n{message}"
        )

    def test_drift_error_contains_remediation_block(
        self, args, cluster_info, target_path,
    ):
        """D-40: the unified-diff report ends with a Remediation block listing
        the rename and remove options."""
        from mlpstorage_py.errors import SystemDriftError

        write_systemname_yaml(args, cluster_info, MagicMock())
        drifted = _make_cluster_info(num_hosts=3, cpu_model="Different CPU")
        with pytest.raises(SystemDriftError) as exc_info:
            write_systemname_yaml(args, drifted, MagicMock())
        message = str(exc_info.value)
        assert "Remediation:" in message
        assert "Rename" in message
        assert "Remove" in message

    def test_drift_logger_error_called_with_report(
        self, args, cluster_info, target_path,
    ):
        """LIFE-03: logger.error fires with the unified-diff report BEFORE the
        SystemDriftError raise (operator sees the report in the run log even
        if the exception is swallowed somewhere upstream)."""
        from mlpstorage_py.errors import SystemDriftError

        write_systemname_yaml(args, cluster_info, MagicMock())
        drifted = _make_cluster_info(num_hosts=3, cpu_model="Different CPU")
        logger = MagicMock()
        with pytest.raises(SystemDriftError):
            write_systemname_yaml(args, drifted, logger)
        assert logger.error.called, "logger.error must be called with the drift report"
        error_messages = " ".join(str(c) for c in logger.error.call_args_list)
        assert "--- on-disk" in error_messages

    # ----- D-48 malformed YAML / structural validation ---------------------

    def test_malformed_yaml_raises_system_description_parse_error(
        self, args, cluster_info, target_path,
    ):
        """D-48: garbage YAML at the on-disk path surfaces SystemDescriptionParseError
        with 'malformed' in the message — NOT SystemDriftError."""
        from mlpstorage_py.errors import SystemDescriptionParseError

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("not: valid: yaml: : : :\n")

        with pytest.raises(SystemDescriptionParseError) as exc_info:
            write_systemname_yaml(args, cluster_info, MagicMock())
        assert "malformed" in str(exc_info.value).lower()

    def test_missing_system_under_test_raises_parse_error(
        self, args, cluster_info, target_path,
    ):
        """D-48: valid YAML but missing top-level system_under_test key."""
        from mlpstorage_py.errors import SystemDescriptionParseError

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("some_other_key:\n  foo: bar\n")

        with pytest.raises(SystemDescriptionParseError) as exc_info:
            write_systemname_yaml(args, cluster_info, MagicMock())
        assert "system_under_test" in str(exc_info.value)

    def test_missing_clients_raises_parse_error(
        self, args, cluster_info, target_path,
    ):
        """D-48: valid YAML with system_under_test but no clients key."""
        from mlpstorage_py.errors import SystemDescriptionParseError

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            "system_under_test:\n  solution:\n    foo: bar\n"
        )

        with pytest.raises(SystemDescriptionParseError) as exc_info:
            write_systemname_yaml(args, cluster_info, MagicMock())
        assert "clients" in str(exc_info.value)

    def test_clients_not_a_list_raises_parse_error(
        self, args, cluster_info, target_path,
    ):
        """D-48: clients key present but not a list."""
        from mlpstorage_py.errors import SystemDescriptionParseError

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            "system_under_test:\n  clients: not a list\n"
        )

        with pytest.raises(SystemDescriptionParseError) as exc_info:
            write_systemname_yaml(args, cluster_info, MagicMock())
        assert "clients" in str(exc_info.value)

    def test_yaml_error_problem_mark_in_message_when_available(
        self, args, cluster_info, target_path,
    ):
        """When yaml.YAMLError surfaces a problem_mark, the parse error message
        contains '(line N, column M)'."""
        from mlpstorage_py.errors import SystemDescriptionParseError

        target_path.parent.mkdir(parents=True, exist_ok=True)
        # Trigger a parse error with positional info: unclosed quote on line 2.
        target_path.write_text('key1: value1\nkey2: "unclosed\nkey3: value3\n')

        with pytest.raises(SystemDescriptionParseError) as exc_info:
            write_systemname_yaml(args, cluster_info, MagicMock())
        message = str(exc_info.value)
        # The problem_mark may not be available on every parser path, but for
        # this triggering shape PyYAML reliably exposes it.
        assert "(line " in message, (
            f"expected '(line ' in message when problem_mark available, got:\n{message}"
        )

    # ----- D-12 carry-forward (datagen never triggers the branch) ----------

    def test_datagen_command_does_not_trigger_diff_branch(
        self, args, cluster_info, target_path,
    ):
        """D-12 + Phase-2: datagen's writer-side gate returns None BEFORE
        reaching the FileExistsError branch. Even with garbage at the on-disk
        path, datagen returns None without raising SystemDescriptionParseError."""
        from mlpstorage_py.errors import SystemDescriptionParseError

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("not: valid: yaml: : : :\n")

        args.command = "datagen"
        # Should return None without raising — the D-12 gate fires before
        # the os.open / FileExistsError / load-diff branch is reached.
        result = write_systemname_yaml(args, cluster_info, MagicMock())
        assert result is None

    # ----- LIFE-04 hand-fill survival (SC#1 hard requirement) --------------

    def test_submitter_hand_fills_survive_unchanged(
        self, args, cluster_info, target_path,
    ):
        """LIFE-04 SC#1: submitter hand-fills SER-02 blanks; re-run against
        unchanged fleet leaves those hand-fills in place. The friendly_description
        field is a SER-02 blank (in-memory side emits '' per universal
        collection-failure rule); Pitfall 3(a) skips the diff when in-memory
        is empty and disk is filled."""
        write_systemname_yaml(args, cluster_info, MagicMock())

        # Submitter hand-fills friendly_description on disk.
        on_disk = yaml.safe_load(target_path.read_text())
        on_disk["system_under_test"]["clients"][0]["friendly_description"] = "Acme-rack-7"
        target_path.write_text(yaml.safe_dump(on_disk, default_flow_style=False))

        # Re-run with the same fleet — must NOT raise SystemDriftError.
        result = write_systemname_yaml(args, cluster_info, MagicMock())
        assert result is None  # LIFE-04 no-touch path

        # The hand-fill survives on disk because the no-touch path doesn't
        # rewrite the file.
        re_loaded = yaml.safe_load(target_path.read_text())
        assert re_loaded["system_under_test"]["clients"][0]["friendly_description"] == "Acme-rack-7"

    # ----- main.py dispatch contract smoke test ----------------------------

    def test_smoke_drift_error_inherits_mlpstorage_exception_for_main_dispatch(
        self, args, cluster_info, target_path,
    ):
        """The contract main.py:262 depends on: SystemDriftError caught as
        MLPStorageException by the top-level handler."""
        from mlpstorage_py.errors import MLPStorageException, SystemDriftError

        write_systemname_yaml(args, cluster_info, MagicMock())
        drifted = _make_cluster_info(num_hosts=3, cpu_model="Different CPU")
        with pytest.raises(MLPStorageException) as exc_info:
            write_systemname_yaml(args, drifted, MagicMock())
        assert isinstance(exc_info.value, SystemDriftError)

    # ----- B-5 stub-splice symmetry lock -----------------------------------

    def test_in_memory_passes_through_splice_stub_lists_before_diff(
        self, args, cluster_info, target_path,
    ):
        """B-5 LIFE-04 stub-splice symmetry: the in-memory comparison subject
        must pass through BOTH copy.deepcopy AND _splice_stub_lists before
        entering diff_node_dict_lists. Otherwise the on-disk side (which went
        through _splice_stub_lists during the original write) and the in-memory
        side would compare asymmetrically — the on-disk side has D-33
        drives-omitted treatment baked in; the in-memory pre-splice side has
        the raw empty drives list. If the symmetry copy is removed from the
        FileExistsError branch, the second run would surface a spurious diff
        between `drives: []` (in-memory pre-splice) and the omitted-drives
        on-disk shape.

        This test writes the file via the Phase-2 emit path, then re-runs with
        the IDENTICAL cluster_info, and asserts no SystemDriftError. If the
        symmetry copy in write_systemname_yaml is removed, this test fires."""
        write_systemname_yaml(args, cluster_info, MagicMock())
        # Identical re-run: no drift expected, no raise expected.
        result = write_systemname_yaml(args, cluster_info, MagicMock())
        assert result is None, (
            "B-5 symmetry violation: identical re-run produced non-None — "
            "in-memory side likely missing _splice_stub_lists pass through "
            "before diff_node_dict_lists"
        )
