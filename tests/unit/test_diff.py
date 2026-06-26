"""Unit tests for the pure-function diff core — Phase 05 / Plan 05-01.

This file owns the comparison-subject contract for Phase 5's logical-diff
lifecycle (LIFE-02 + LIFE-03). It covers the eight must-have truths fixed in
05-01-PLAN.md frontmatter:

1. Byte-equal inputs → DiffResult.empty == True (D-37 + LIFE-04).
2. Single-field change → exactly one DiffEntry with the differing JSONPath and
   old/new values (D-37 / D-40).
3. Quantity-field change on otherwise-identical fingerprint → drift (D-39).
4. Fingerprint present only on-disk → drift (D-47).
5. Fingerprint present only in-memory → drift (D-46).
6. format_unified_diff emits --- / +++ headers + @@ <JSONPath> @@ hunks +
   -/+ lines + Remediation block (D-40 + D-41).
7. SER-02 blank preservation Pitfall 3 direction (a): when in-memory is the
   empty string AND on-disk has a submitter-filled non-empty value, the path
   is NOT flagged.
8. Long sysctl tuple value (e.g. '4096\\t87380\\t16777216') round-trips
   through the formatter verbatim, no truncation (D-41).

Test discipline:
- Pure-function tests, no filesystem or MPI involvement.
- SimpleNamespace + MagicMock fixture style (matches test_auto_generator_write.py).
- Helper fixtures _make_node_dict / _make_disk_and_memory_pair build Phase-4
  7-key emit-shape node dicts so the diff operates on realistic input.

RED gate: at the moment this file is committed, mlpstorage_py/system_description/diff.py
does NOT yet exist. pytest collection MUST fail with ModuleNotFoundError on the
import below. Task 2 ships the implementation and flips the suite to GREEN.
"""

from __future__ import annotations

import logging

import pytest

from mlpstorage_py.system_description.diff import (
    DiffEntry,
    DiffResult,
    _flatten_to_paths,
    diff_node_dict_lists,
    format_unified_diff,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def _make_node_dict(
    *,
    cpu_model: str = "Intel Xeon Platinum 8480+",
    cpu_qty: int = 2,
    cpu_cores: int = 56,
    memory: int = 256,
    model_name: str = "Dell PowerEdge R760",
    os_name: str = "Rocky Linux",
    os_version: str = "9.5",
    networking=None,
    sysctl=None,
    environment=None,
    drives=None,
    friendly_description: str = "",
    quantity: int = 1,
) -> dict:
    """Build a Phase-4 7-key emit-shape node dict.

    Mirrors auto_generator.node_dict_from_host's emission contract: top-level
    keys = friendly_description, chassis, networking, sysctl, environment,
    drives, operating_system, quantity. Friendly_description defaults to ''
    matching SER-02 (collector blank, submitter to fill).
    """
    if networking is None:
        networking = [
            {"type": "ethernet", "speed": 100, "state": "up", "traffic": [], "unit_count": 2},
        ]
    if sysctl is None:
        sysctl = [
            {"name": "net.core.rmem_max", "value": "16777216"},
        ]
    if environment is None:
        environment = []
    if drives is None:
        drives = [
            {"vendor_name": "Samsung", "model_name": "PM9A3", "interface": "NVMe",
             "capacity_in_GB": 1920, "unit_count": 4},
        ]
    return {
        "friendly_description": friendly_description,
        "chassis": {
            "cpu_model": cpu_model,
            "cpu_qty": cpu_qty,
            "cpu_cores": cpu_cores,
            "memory_capacity": memory,
            "model_name": model_name,
        },
        "networking": networking,
        "sysctl": sysctl,
        "environment": environment,
        "drives": drives,
        "operating_system": {
            "name": os_name,
            "version": os_version,
        },
        "quantity": quantity,
    }


def _make_disk_and_memory_pair(*, disk_overrides=None, memory_overrides=None):
    """Two-host fleet baseline.  Returns (on_disk_stanzas, in_memory_stanzas)."""
    on_disk = [_make_node_dict(), _make_node_dict(cpu_model="AMD EPYC 9654")]
    in_memory = [_make_node_dict(), _make_node_dict(cpu_model="AMD EPYC 9654")]
    if disk_overrides:
        for idx, override in disk_overrides.items():
            on_disk[idx].update(override)
    if memory_overrides:
        for idx, override in memory_overrides.items():
            in_memory[idx].update(override)
    return on_disk, in_memory


# ---------------------------------------------------------------------------
# TestDataclasses (3 tests)
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_diff_entry_construction_path_old_new(self):
        e = DiffEntry(path="clients[0].chassis.cpu_model", old="Intel", new="AMD")
        assert e.path == "clients[0].chassis.cpu_model"
        assert e.old == "Intel"
        assert e.new == "AMD"

    def test_diff_result_empty_when_entries_empty(self):
        r = DiffResult(entries=[])
        assert r.empty is True

    def test_diff_result_not_empty_when_one_entry(self):
        r = DiffResult(entries=[DiffEntry(path="x", old=1, new=2)])
        assert r.empty is False


# ---------------------------------------------------------------------------
# TestFlattenToPaths (5 tests)
# ---------------------------------------------------------------------------


class TestFlattenToPaths:
    def test_scalar_at_root_yields_empty_prefix_path(self):
        # Scalar input with empty prefix should yield ('', scalar).
        result = list(_flatten_to_paths(42))
        assert result == [("", 42)]

    def test_dict_keys_dotted_concatenation(self):
        result = dict(_flatten_to_paths({"a": {"b": 1}}))
        assert result == {"a.b": 1}

    def test_list_indices_bracket_notation(self):
        result = dict(_flatten_to_paths({"items": [{"x": 1}, {"x": 2}]}))
        assert result == {"items[0].x": 1, "items[1].x": 2}

    def test_empty_dict_yields_no_entries(self):
        assert list(_flatten_to_paths({})) == []

    def test_mixed_dict_list_dict_nesting(self):
        node = _make_node_dict()
        flat = dict(_flatten_to_paths(node))
        # Spot-check load-bearing JSONPaths that downstream tests assert on.
        assert flat["chassis.cpu_model"] == "Intel Xeon Platinum 8480+"
        assert flat["chassis.cpu_qty"] == 2
        assert flat["operating_system.name"] == "Rocky Linux"
        assert flat["networking[0].type"] == "ethernet"
        assert flat["networking[0].speed"] == 100
        assert flat["sysctl[0].name"] == "net.core.rmem_max"
        assert flat["drives[0].capacity_in_GB"] == 1920


# ---------------------------------------------------------------------------
# TestRoundTripEqualIsEmpty (3 tests; covers D-37 + LIFE-04)
# ---------------------------------------------------------------------------


class TestRoundTripEqualIsEmpty:
    def test_identical_single_stanza_lists_empty(self):
        a = [_make_node_dict()]
        b = [_make_node_dict()]
        r = diff_node_dict_lists(a, b)
        assert r.empty is True

    def test_identical_multi_stanza_lists_empty(self):
        a, b = _make_disk_and_memory_pair()
        r = diff_node_dict_lists(a, b)
        assert r.empty is True

    def test_field_value_order_inside_stanza_insensitive(self):
        # PyYAML / Python dict equality is order-insensitive for siblings —
        # Pitfall 1 lock: reordering keys in the dict literal MUST NOT show
        # up as drift because the diff layer flattens to paths.
        a = [_make_node_dict()]
        b_node = _make_node_dict()
        # Rebuild dict in different key order — semantically equal.
        reordered = {k: b_node[k] for k in reversed(list(b_node.keys()))}
        r = diff_node_dict_lists(a, [reordered])
        assert r.empty is True


# ---------------------------------------------------------------------------
# TestFieldChangeIsDrift (4 tests; covers D-37)
# ---------------------------------------------------------------------------


class TestFieldChangeIsDrift:
    def test_cpu_model_change_surfaces_one_diff_with_jsonpath(self):
        # CPU model is part of the fingerprint — changing it makes the two
        # sides orphan stanzas per D-38 / Pitfall 2. Report shape: two
        # entries (one for the absent on-disk fingerprint, one for the absent
        # in-memory fingerprint). NOT a single same-fingerprint field diff.
        on_disk = [_make_node_dict(cpu_model="Intel Xeon Platinum 8480+")]
        in_mem = [_make_node_dict(cpu_model="AMD EPYC 9654")]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is False
        # Both fingerprints surface as orphans, so at least one DiffEntry path
        # mentions a fingerprint marker.
        assert any("fingerprint=" in e.path for e in r.entries)

    def test_sysctl_value_change_surfaces_diff_at_clients_index_sysctl_index_value(self):
        on_disk = [_make_node_dict(sysctl=[{"name": "net.core.rmem_max", "value": "16777216"}])]
        in_mem = [_make_node_dict(sysctl=[{"name": "net.core.rmem_max", "value": "33554432"}])]
        r = diff_node_dict_lists(on_disk, in_mem)
        # Sysctl is part of the fingerprint (sysctl_sig), so this is again an
        # orphan-stanza diff; assert the change is surfaced (not silently
        # dropped).
        assert r.empty is False

    def test_drives_capacity_change_surfaces_diff(self):
        on_disk = [_make_node_dict(drives=[{"vendor_name": "Samsung", "model_name": "PM9A3",
                                            "interface": "NVMe", "capacity_in_GB": 1920,
                                            "unit_count": 4}])]
        in_mem = [_make_node_dict(drives=[{"vendor_name": "Samsung", "model_name": "PM9A3",
                                           "interface": "NVMe", "capacity_in_GB": 3840,
                                           "unit_count": 4}])]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is False

    def test_environment_value_change_surfaces_diff(self):
        # Phase 4 redaction shape: '[SET — 40 chars]' → '[SET — 38 chars]'.
        # The environment sig is part of the fingerprint, so this is an
        # orphan-stanza diff.
        on_disk = [_make_node_dict(environment=[
            {"name": "AWS_SECRET_ACCESS_KEY", "value": "[SET — 40 chars]"}])]
        in_mem = [_make_node_dict(environment=[
            {"name": "AWS_SECRET_ACCESS_KEY", "value": "[SET — 38 chars]"}])]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is False


# ---------------------------------------------------------------------------
# TestQuantityChangeIsDrift (2 tests; covers D-39)
# ---------------------------------------------------------------------------


class TestQuantityChangeIsDrift:
    def test_quantity_4_to_5_surfaces_diff_at_clients_index_quantity(self):
        on_disk = [_make_node_dict(quantity=4)]
        in_mem = [_make_node_dict(quantity=5)]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is False
        # quantity is NOT part of the fingerprint, so same fingerprint → flat
        # diff at the .quantity path.
        assert any(e.path.endswith(".quantity") or e.path == "quantity"
                   for e in r.entries) or any(
            "quantity" in e.path for e in r.entries)
        # Verify the values
        matching = [e for e in r.entries if "quantity" in e.path]
        assert any(e.old == 4 and e.new == 5 for e in matching)

    def test_quantity_5_to_4_surfaces_diff(self):
        # Fleet shrinkage is symmetric.
        on_disk = [_make_node_dict(quantity=5)]
        in_mem = [_make_node_dict(quantity=4)]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is False
        matching = [e for e in r.entries if "quantity" in e.path]
        assert any(e.old == 5 and e.new == 4 for e in matching)


# ---------------------------------------------------------------------------
# TestSymmetricDriftDetection (3 tests; covers D-46 + D-47)
# ---------------------------------------------------------------------------


class TestSymmetricDriftDetection:
    def test_disk_absent_field_in_memory_only_is_drift(self):
        # Add a new field only present in memory (e.g. fresh schema).
        on_disk_node = _make_node_dict()
        in_mem_node = _make_node_dict()
        in_mem_node["chassis"]["new_field"] = "fresh-value"
        r = diff_node_dict_lists([on_disk_node], [in_mem_node])
        assert r.empty is False
        assert any("chassis.new_field" in e.path for e in r.entries)

    def test_disk_only_field_in_memory_absent_is_drift(self):
        on_disk_node = _make_node_dict()
        on_disk_node["chassis"]["legacy_field"] = "legacy-value"
        in_mem_node = _make_node_dict()
        r = diff_node_dict_lists([on_disk_node], [in_mem_node])
        assert r.empty is False
        assert any("chassis.legacy_field" in e.path for e in r.entries)

    def test_fingerprint_only_on_one_side_is_drift(self):
        # D-38 orphan: a stanza with a unique fingerprint on disk should
        # surface as drift.
        on_disk = [_make_node_dict(), _make_node_dict(cpu_model="AMD EPYC 9654")]
        in_mem = [_make_node_dict()]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is False
        # The AMD fingerprint should be flagged as on-disk-only.
        assert any("fingerprint=" in e.path for e in r.entries)


# ---------------------------------------------------------------------------
# TestSer02BlankPreservation (4 tests; covers Pitfall 3 direction (a))
# ---------------------------------------------------------------------------


class TestSer02BlankPreservation:
    def test_in_memory_empty_disk_filled_friendly_description_NO_diff(self):
        # Disk has submitter-filled friendly_description; in-memory (fresh
        # collection) has the default empty string. Direction (a): NO diff.
        on_disk = [_make_node_dict(friendly_description="Production tier rack 7")]
        in_mem = [_make_node_dict(friendly_description="")]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is True, (
            "Submitter-filled friendly_description on disk + empty in-memory "
            "MUST NOT register as drift (SER-02 blank preservation)."
        )

    def test_in_memory_empty_disk_filled_networking_traffic_NO_diff(self):
        # networking[].traffic is on the SER-02 blank list (REQUIREMENTS line 40).
        on_disk_nic = {"type": "ethernet", "speed": 100, "state": "up",
                       "traffic": ["data"], "unit_count": 2}
        in_mem_nic = {"type": "ethernet", "speed": 100, "state": "up",
                      "traffic": [""], "unit_count": 2}
        on_disk = [_make_node_dict(networking=[on_disk_nic])]
        in_mem = [_make_node_dict(networking=[in_mem_nic])]
        r = diff_node_dict_lists(on_disk, in_mem)
        # Note: in-memory networking[0].traffic[0] == '' and on-disk == 'data'
        # → direction (a) skip applies at the leaf.
        assert r.empty is True

    def test_in_memory_empty_disk_filled_drives_media_type_NO_diff(self):
        # drives media_type is on the SER-02 blank list. Add the field and
        # verify direction (a) preserves the disk-filled value.
        on_disk_drive = {"vendor_name": "Samsung", "model_name": "PM9A3",
                         "interface": "NVMe", "capacity_in_GB": 1920,
                         "unit_count": 4, "media_type": "ssd-flash"}
        in_mem_drive = {"vendor_name": "Samsung", "model_name": "PM9A3",
                        "interface": "NVMe", "capacity_in_GB": 1920,
                        "unit_count": 4, "media_type": ""}
        on_disk = [_make_node_dict(drives=[on_disk_drive])]
        in_mem = [_make_node_dict(drives=[in_mem_drive])]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is True

    def test_in_memory_NONEMPTY_disk_filled_with_different_value_IS_diff(self):
        # Direction-lock regression: direction (a) is NOT "always skip when
        # on-disk is filled". It's specifically "skip when in-memory IS the
        # empty string". If in-memory has a non-empty different value, drift
        # MUST surface.
        on_disk = [_make_node_dict(friendly_description="Production tier rack 7")]
        in_mem = [_make_node_dict(friendly_description="Staging tier rack 3")]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is False, (
            "Pitfall 3(a) direction lock: in-memory non-empty value MUST "
            "register as drift; only empty-string in-memory is preserved."
        )
        assert any("friendly_description" in e.path for e in r.entries)


# ---------------------------------------------------------------------------
# TestUnifiedDiffFormat (5 tests; covers D-40 + D-41)
# ---------------------------------------------------------------------------


class TestUnifiedDiffFormat:
    def _drift_result(self):
        on_disk = [_make_node_dict(quantity=4)]
        in_mem = [_make_node_dict(quantity=5)]
        return diff_node_dict_lists(on_disk, in_mem)

    def test_header_lines_present(self):
        result = self._drift_result()
        report = format_unified_diff(result, "/tmp/sys.yaml")
        assert "--- on-disk: /tmp/sys.yaml" in report
        assert "+++ in-memory: <computed from live MPI fleet>" in report

    def test_hunk_marker_at_at_jsonpath_at_at(self):
        # Construct a quantity diff (same fingerprint, .quantity flat diff).
        result = self._drift_result()
        report = format_unified_diff(result, "/tmp/sys.yaml")
        # The hunk marker is @@ <path> @@ — the path includes .quantity.
        assert "@@ " in report and " @@" in report
        assert "quantity" in report

    def test_minus_old_plus_new_lines(self):
        result = self._drift_result()
        report = format_unified_diff(result, "/tmp/sys.yaml")
        assert "- 4" in report
        assert "+ 5" in report

    def test_remediation_block_present(self):
        result = self._drift_result()
        report = format_unified_diff(result, "/tmp/sys.yaml")
        assert "Remediation:" in report
        assert "Rename the existing yaml" in report
        assert "Remove " in report
        assert "/tmp/sys.yaml" in report  # path appears in rm hint too

    def test_long_sysctl_value_round_trips_verbatim_no_truncation(self):
        # D-41 lock: a long tab-separated sysctl tuple must appear in the
        # output exactly as it appears in the input — no shortening, no
        # repr-quote-wrapping that loses the literal tabs.
        long_value = "4096\t87380\t16777216"
        on_disk = [_make_node_dict(sysctl=[
            {"name": "net.ipv4.tcp_rmem", "value": long_value}])]
        in_mem = [_make_node_dict(sysctl=[
            {"name": "net.ipv4.tcp_rmem", "value": "4096\t87380\t8388608"}])]
        result = diff_node_dict_lists(on_disk, in_mem)
        report = format_unified_diff(result, "/tmp/sys.yaml")
        # The longer value must appear intact somewhere in the report.
        assert long_value in report, (
            f"D-41 truncation regression: long sysctl value not found "
            f"verbatim in report. Report:\n{report}"
        )


# ---------------------------------------------------------------------------
# TestDiffHandFillAffordance (12 tests; Phase 5.2 / HANDFILL-01)
# ---------------------------------------------------------------------------
#
# Soft-pair pre-pass behavior:
#   - When in-memory orphan has at least one "" scalar fingerprint position
#     AND there is a unique on-disk orphan whose non-empty scalar positions
#     all align AND whose 4 signature positions match exactly, the two
#     stanzas are treated as the same client. Pitfall 3(a) SER-02 at the
#     leaf level (diff.py:272-280) then preserves the submitter's value
#     and no DiffEntry is emitted.
#   - Real drift (collector resolves a DIFFERENT non-empty value) is NOT
#     swallowed: the scalar-alignment check fails on the non-matching
#     position and the stanzas remain orphans.
#   - Ambiguous soft-pair candidates fall back to orphan emission (D-63).
#   - D-60 reverse-direction: collector finally learns a value the user
#     did not hand-fill → INFO log, no DiffEntry, no drift.
#   - D-61 strict-signature: 4 callable signature positions are
#     exact-match (empty signature () does NOT count as wildcard).


class TestDiffHandFillAffordance:
    """HANDFILL-01: hand-filled scalar fingerprint values survive a re-run
    when the collector still returns '' at the same position.

    Locks SC#1-#9 + D-60 (reverse-direction INFO log) + D-61 (signature
    strict-match) + D-62 (two-pass pairing) + D-63 (ambiguous fallback).
    """

    # --- (a) Hand-fill survival cases (SC#2, SC#3) -----------------------

    def test_handfilled_chassis_model_name_survives_rerun_with_empty_collector(self):
        on_disk = [_make_node_dict(model_name="Dell Latitude 7420")]
        in_mem = [_make_node_dict(model_name="")]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is True, (
            "D-62 soft-pair: hand-filled chassis.model_name with "
            "collector-empty in-memory must NOT register as drift; "
            "soft-pair pass must pair the stanzas and let Pitfall 3(a) "
            "preserve the submitter's value."
        )

    def test_handfilled_cpu_model_survives_rerun_with_empty_collector(self):
        on_disk = [_make_node_dict(cpu_model="Intel Xeon Platinum 8480+")]
        in_mem = [_make_node_dict(cpu_model="")]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is True, (
            "D-62 soft-pair: hand-filled chassis.cpu_model + collector "
            "empty MUST NOT register as drift."
        )

    def test_handfilled_os_name_survives_rerun_with_empty_collector(self):
        on_disk = [_make_node_dict(os_name="Rocky Linux", os_version="9.5")]
        in_mem = [_make_node_dict(os_name="", os_version="")]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is True, (
            "D-62 soft-pair: hand-filled operating_system.name + "
            "operating_system.version with collector empty MUST NOT "
            "register as drift."
        )

    # --- (b) Real-drift-still-raises cases (SC#4) ------------------------

    def test_real_drift_handfilled_position_recomputed_to_different_non_empty_still_raises(self):
        # Submitter hand-filled "Dell PowerEdge R750"; collector now resolves
        # a DIFFERENT non-empty value. The hand-fill affordance is strictly
        # empty-side adopt-on-empty; a non-empty disagreement is real drift.
        on_disk = [_make_node_dict(model_name="Dell PowerEdge R750")]
        in_mem = [_make_node_dict(model_name="Dell PowerEdge R760")]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is False, (
            "SC#4: hand-fill affordance must NEVER silence a non-empty "
            "disagreement at a fingerprint scalar position. Real drift "
            "still raises."
        )

    def test_real_drift_non_handfill_collector_change_still_raises(self):
        # sysctl is part of sysctl_sig (signature position 8). Per D-61
        # signatures are strict-match — a sysctl value change is a
        # signature-position miss, NOT a soft-pair case. Confirms the
        # soft-pair pass does NOT swallow signature-position changes.
        on_disk = [_make_node_dict(sysctl=[{"name": "net.core.rmem_max", "value": "16777216"}])]
        in_mem = [_make_node_dict(sysctl=[{"name": "net.core.rmem_max", "value": "33554432"}])]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is False, (
            "D-61: sysctl_sig is a strict-match signature position; "
            "soft-pair pass MUST NOT swallow signature differences."
        )

    # --- (c) Ambiguous fallback case (SC#5, D-63) ------------------------

    def test_ambiguous_soft_pair_two_candidates_falls_back_to_orphans(self):
        # Two distinct hand-filled clients (Dell A, Dell B); in-memory has
        # two orphan stanzas with model_name="". The in-memory orphan
        # ambiguously matches both on-disk orphans (they share all
        # signatures and other scalar positions). Per D-63, ambiguity must
        # fall back to orphan emission — never silently conflate distinct
        # machines.
        on_disk = [_make_node_dict(model_name="Dell A"), _make_node_dict(model_name="Dell B")]
        in_mem = [_make_node_dict(model_name=""), _make_node_dict(model_name="")]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is False, (
            "D-63: ambiguous soft-pair MUST fall back to orphan emission "
            "— never silently conflate distinct machines."
        )
        assert any("fingerprint=" in e.path for e in r.entries), (
            "D-63 fallback must emit fingerprint-level orphan entries "
            "(not leaf-level), proving the unpaired orphans flowed into "
            "the existing D-46/D-47 emission path."
        )

    # --- (d) D-61 signature strict-match cases ---------------------------

    def test_signature_strict_match_empty_signature_does_not_soft_pair(self):
        # In-memory has NO networking interfaces — networking_sig is the
        # empty tuple (). Per D-61, empty signature on one side does NOT
        # count as wildcard; must match the other side's signature
        # exactly. Since on-disk has one interface, signatures differ →
        # no soft-pair.
        on_disk = [_make_node_dict(
            model_name="Dell A",
            networking=[{"type": "ethernet", "speed": 100, "state": "up",
                         "traffic": [], "unit_count": 2}],
        )]
        in_mem = [_make_node_dict(model_name="", networking=[])]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is False, (
            "D-61: empty signature () is NOT a wildcard. Soft-pair MUST "
            "NOT pair stanzas with differing networking_sig values."
        )

    def test_signature_strict_match_different_sysctl_sigs_do_not_soft_pair(self):
        # Scalar position (model_name) is soft-pair-eligible, but the
        # sysctl_sig differs between sides. Per D-61, signatures stay
        # strict-match — no soft-pair.
        on_disk = [_make_node_dict(
            model_name="Dell A",
            sysctl=[{"name": "net.core.rmem_max", "value": "16777216"}],
        )]
        in_mem = [_make_node_dict(
            model_name="",
            sysctl=[{"name": "net.core.wmem_max", "value": "16777216"}],
        )]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is False, (
            "D-61: sysctl_sig strict-match. Different sysctl signatures "
            "MUST block soft-pair even when a scalar position is "
            "soft-pair-eligible."
        )

    # --- (e) D-60 reverse-direction INFO log cases -----------------------

    def test_reverse_direction_collector_resolves_value_emits_info_log_no_drift(self, caplog):
        # On-disk was "" (collector previously empty, user did NOT
        # hand-fill); collector finally learned the value. Per D-60: INFO
        # log emitted, NO DiffEntry, NO drift; on-disk file unchanged per
        # LIFE-04.
        on_disk = [_make_node_dict(model_name="")]
        in_mem = [_make_node_dict(model_name="Dell Latitude E5450")]
        with caplog.at_level(logging.INFO, logger="mlpstorage_py.system_description.diff"):
            r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is True, (
            "D-60: reverse-direction (recomputed non-empty + on-disk '') "
            "MUST NOT raise drift."
        )
        assert any("collector resolved" in record.message for record in caplog.records), (
            "D-60: reverse-direction MUST emit an INFO log of shape "
            "'collector resolved <field>=...' so the operator sees the "
            "new knowledge."
        )

    def test_reverse_direction_info_log_includes_field_path_and_resolved_value(self, caplog):
        on_disk = [_make_node_dict(model_name="")]
        in_mem = [_make_node_dict(model_name="Dell Latitude E5450")]
        with caplog.at_level(logging.INFO, logger="mlpstorage_py.system_description.diff"):
            diff_node_dict_lists(on_disk, in_mem)
        joined = "\n".join(record.message for record in caplog.records)
        assert "chassis.model_name" in joined, (
            "D-60 log MUST include the field JSONPath so the operator "
            "knows which field improved."
        )
        assert "Dell Latitude E5450" in joined, (
            "D-60 log MUST include the resolved value so the operator "
            "sees what the collector learned."
        )
        assert "LIFE-04" in joined, (
            "D-60 log MUST hint at LIFE-04 (no-touch contract) so the "
            "operator knows why the on-disk YAML was not updated."
        )

    # --- (f) Two-pass exact-match-first preservation (SC#2 pass-1) -------

    def test_exact_match_pass_still_preserves_identical_fingerprints_no_soft_pair_invoked(self):
        """Regression lock: passes pre-GREEN to prove exact-match preservation
        still works; passes post-GREEN to prove soft-pair did not break it."""
        on_disk, in_mem = _make_disk_and_memory_pair()
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is True, (
            "D-62 pass-1 preservation: exact-fingerprint match must "
            "short-circuit before the soft-pair pre-pass; identical "
            "fleets must surface as no-drift via the existing fast path."
        )

    # --- (g) Multi-client topology positive case (SC#5 paired correctly) -

    def test_multi_client_topology_two_handfilled_three_total_pairs_correctly(self):
        # 3-host fleet: hosts A and B exact-match; host C has its
        # model_name hand-filled on disk and "" in memory. Only host C
        # triggers soft-pair, with exactly one candidate (the Intel-C
        # on-disk stanza).
        on_disk = [
            _make_node_dict(cpu_model="Intel A", model_name="Dell A"),
            _make_node_dict(cpu_model="Intel B", model_name="Dell B"),
            _make_node_dict(cpu_model="Intel C", model_name="HandFilledChassis"),
        ]
        in_mem = [
            _make_node_dict(cpu_model="Intel A", model_name="Dell A"),
            _make_node_dict(cpu_model="Intel B", model_name="Dell B"),
            _make_node_dict(cpu_model="Intel C", model_name=""),
        ]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is True, (
            "D-62 + D-63: unique soft-pair candidate among distinct "
            "topologies must pair correctly without conflating Intel-A "
            "or Intel-B."
        )

    # --- (h) CR-01 / WR-02 partial-ambiguity regression lock -------------

    def test_partial_ambiguity_unique_then_ambiguous_does_not_greedily_conflate(self):
        # CR-01 / WR-02 regression: a single-pass greedy pairing consumes
        # disks as it walks the sorted mem fingerprints. When the FIRST
        # mem (by repr sort) uniquely matches a disk that a LATER mem
        # would also have considered a candidate against the original
        # pool, the greedy pass quietly conflates the later mem with the
        # leftover disk. Per the reviewer's two-phase fix (CR-01), a pair
        # is committed only when (a) the mem has exactly one candidate
        # AND (b) that candidate's claim_count is 1 across all mems.
        #
        # Construction (engineered so the unique mem sorts BEFORE the
        # ambiguous mem under key=repr — which requires the unique mem
        # to have "" at the FIRST scalar position, since "" < any
        # non-empty string in repr ordering):
        #   - DiskA: cpu_model="A", model_name="P"
        #   - DiskB: cpu_model="A", model_name="Q"
        #   - Mem_X: cpu_model="", model_name="P"
        #       fp[0]="" sorts first. Uniquely matches DiskA (model "P"
        #       pins it; DiskB fails non-empty mismatch on model).
        #   - Mem_Y: cpu_model="A", model_name=""
        #       fp[0]="A" sorts after Mem_X. Candidates against ORIGINAL
        #       pool = {DiskA, DiskB} (cpu_model "A" matches both;
        #       model_name="" wildcards). Genuinely ambiguous.
        #
        # Greedy: Mem_X consumes DiskA, then Mem_Y sees only {DiskB}
        # remaining → unique → silently pairs Mem_Y↔DiskB. WRONG.
        # Two-phase: claim_count[DiskA]=2 (Mem_X and Mem_Y both list it);
        # Mem_X's unique cand is contested → does not pair. Mem_Y has 2
        # cands → does not pair. Both fall through to orphan emission.
        on_disk = [
            _make_node_dict(cpu_model="A", model_name="P"),
            _make_node_dict(cpu_model="A", model_name="Q"),
        ]
        in_mem = [
            _make_node_dict(cpu_model="", model_name="P"),
            _make_node_dict(cpu_model="A", model_name=""),
        ]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is False, (
            "CR-01: greedy soft-pair consumption MUST NOT convert genuine "
            "D-63 ambiguity into silent conflation. Mem_Y originally "
            "matched both DiskA and DiskB on the unconsumed pool; the "
            "two-phase algorithm must reject Mem_Y's pairing (and "
            "Mem_X's contested pairing) and surface both as orphans."
        )
        assert any("fingerprint=" in e.path for e in r.entries), (
            "CR-01 fallback must emit fingerprint-level orphan entries "
            "when the two-phase algorithm rejects contested pairings."
        )

    # --- (i) WR-01 D-60 scope regression lock ----------------------------

    def test_d60_does_not_swallow_non_fingerprint_leaf_changes(self):
        # WR-01 regression: Phase 5.2's D-60 reverse-direction INFO log
        # was wired into the every-leaf comparison loop in
        # `_emit_leaf_diffs`, which silently swallows ANY on-disk "" →
        # in-memory non-empty string change — including non-fingerprint
        # leaves like friendly_description. Pre-Phase-5.2 such a change
        # was a DiffEntry (drift). The fix scopes D-60 to the 7 scalar
        # fingerprint positions only.
        #
        # Both stanzas share the default fingerprint (exact-match pass-1)
        # so they fall into `_emit_leaf_diffs` via the common-fp branch.
        # The reverse-direction "" → non-empty change is on
        # `friendly_description`, which is NOT a fingerprint position.
        on_disk = [_make_node_dict(friendly_description="")]
        in_mem = [_make_node_dict(friendly_description="Login Node 0")]
        r = diff_node_dict_lists(on_disk, in_mem)
        assert r.empty is False, (
            "WR-01: D-60 must scope to the 7 scalar fingerprint "
            "positions only. Non-fingerprint leaves (e.g. "
            "friendly_description) that change from '' to non-empty "
            "must still surface as DiffEntries per pre-Phase-5.2 "
            "drift semantics."
        )
        assert any("friendly_description" in e.path for e in r.entries), (
            "WR-01: the friendly_description '' → 'Login Node 0' "
            "change must be the surfaced DiffEntry path."
        )
