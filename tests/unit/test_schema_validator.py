"""
Unit tests for system_description.schema_validator.

One test (or small group) per validation rule, organized by rule group.
Each test uses copy.deepcopy on a minimal valid base document and mutates
exactly the field(s) under test — avoids cross-test contamination and
makes the intent of each test explicit.

Rule reference:
  Phase 1 (structural)   — base schema types, enums, required fields, min values
  Rules  1– 6 (onprem)   — rack_units/power required; total_rack_units required; switch power
  Rules  7–10 (cloud)    — switches/rack_power_supplies forbidden; rack_units/power forbidden
  Rule  11               — total_rack_units arithmetic (clients excluded)
  Rule  12               — power_device.min_psus_active <= sum(psus_configured unit_counts)
  Rule  13               — remap_time biconditional with simultaneous flags
  Rule  14               — client_footprint/client_installation n/a pairing
  Rules 15–17            — networking required for remote/remote_and_local; switches forbidden for local
"""

import copy
import pytest

from mlpstorage_py.system_description.schema_validator import validate_dict


# ---------------------------------------------------------------------------
# Minimal building blocks
# ---------------------------------------------------------------------------

def _psu(unit_count=2) -> dict:
    return {
        "unit_count": unit_count,
        "inlet_voltage": 240,
        "nameplate_power_watts": 750,
        "efficiency": "Platinum",
    }


def _power(min_active=1) -> dict:
    return {"min_psus_active": min_active, "psus_configured": [_psu()]}


def _networking() -> list:
    return [{"unit_count": 1, "type": "ethernet", "speed": 100, "traffic": ["data"]}]


def _os() -> dict:
    return {"name": "Rocky Linux", "version": "9.5"}


def _onprem_chassis(rack_units=1) -> dict:
    return {
        "model_name": "R640",
        "rack_units": rack_units,
        "cpu_model": "Xeon Gold",
        "cpu_qty": 2,
        "cpu_cores": 32,
        "memory_capacity": 128,
        "power": _power(),
    }


def _cloud_chassis() -> dict:
    return {
        "model_name": "i7i.8xlarge",
        "cpu_model": "Graviton3",
        "cpu_qty": 1,
        "cpu_cores": 32,
        "memory_capacity": 256,
    }


def _onprem_node(rack_units=1, quantity=1) -> dict:
    return {
        "friendly_description": "Server",
        "quantity": quantity,
        "chassis": _onprem_chassis(rack_units),
        "networking": _networking(),
        "operating_system": _os(),
    }


def _onprem_client(rack_units=1, quantity=1) -> dict:
    return {
        "friendly_description": "Client",
        "quantity": quantity,
        "chassis": _onprem_chassis(rack_units),
        "networking": _networking(),
        "operating_system": _os(),
    }


def _cloud_node() -> dict:
    return {
        "friendly_description": "Cloud instance",
        "quantity": 1,
        "chassis": _cloud_chassis(),
        "networking": _networking(),
        "operating_system": _os(),
    }


def _cloud_client() -> dict:
    return {
        "friendly_description": "Cloud client",
        "quantity": 1,
        "chassis": _cloud_chassis(),
        "networking": _networking(),
        "operating_system": _os(),
    }


def _switch(rack_units=2, unit_count=1) -> dict:
    return {
        "unit_count": unit_count,
        "vendor_name": "Cisco",
        "model_name": "Nexus 5000",
        "ports": [{"unit_count": 48, "type": "ethernet", "speed": 100, "traffic": ["data"]}],
        "rack_units": rack_units,
        "power": _power(),
    }


def _drive_performance() -> dict:
    return {
        "seq_read_MBps": 14000,
        "seq_write_MBps": 10000,
        "random_read_IOPS": 2700000,
        "random_write_IOPS": 400000,
        "form_factor": "U.2",
        "interface_speed": "PCIe Gen5 x4",
    }


def _drive(with_performance=False, **overrides) -> dict:
    d = {
        "unit_count": 8,
        "vendor_name": "Micron",
        "model_name": "X9000 TLC",
        "interface": "nvme",
        "media_type": "TLC",
        "capacity_in_GB": 8000,
    }
    if with_performance:
        d["performance"] = _drive_performance()
    d.update(overrides)
    return d


def _capabilities(sw=True, sr=True, remap=0) -> dict:
    return {
        "multi_host": True,
        "simultaneous_write": sw,
        "simultaneous_read": sr,
        "remap_time_in_seconds": remap,
    }


def _architecture(storage_location="remote", footprint="open_source", installation="in_box") -> dict:
    return {
        "storage_location": storage_location,
        "benchmark_API": "file",
        "product_API": "file",
        "client_footprint": footprint,
        "client_installation": installation,
    }


def _solution(storage_location="remote", sw=True, sr=True, remap=0,
              footprint="open_source", installation="in_box") -> dict:
    return {
        "submission_name": "test",
        "friendly_description": "Test submission",
        "architecture": _architecture(storage_location, footprint, installation),
        "capabilities": _capabilities(sw, sr, remap),
    }


def _onprem_doc(storage_location="remote") -> dict:
    """Minimal valid onprem document: 1 product node (1 RU × qty 1), 1 client group."""
    return {
        "system_under_test": {
            "solution": _solution(storage_location),
            "deployment": "onprem",
            "product_nodes": [_onprem_node()],
            "total_rack_units": 1,
            "clients": [_onprem_client()],
        }
    }


def _cloud_doc() -> dict:
    """Minimal valid cloud document."""
    return {
        "system_under_test": {
            "solution": _solution(),
            "deployment": "cloud",
            "product_nodes": [_cloud_node()],
            "clients": [_cloud_client()],
        }
    }


def _local_doc() -> dict:
    """Minimal valid onprem/local document (no networking, no switches, n/a footprint)."""
    return {
        "system_under_test": {
            "solution": _solution(
                storage_location="local",
                sw=False, sr=False, remap=30,
                footprint="n/a", installation="n/a",
            ),
            "deployment": "onprem",
            "product_nodes": [
                {
                    "friendly_description": "Drive node",
                    "quantity": 1,
                    "chassis": _onprem_chassis(),
                    "operating_system": _os(),
                }
            ],
            "total_rack_units": 1,
            "clients": [
                {
                    "friendly_description": "Client",
                    "quantity": 1,
                    "chassis": _onprem_chassis(),
                    "operating_system": _os(),
                }
            ],
        }
    }


# ---------------------------------------------------------------------------
# Test assertion helpers
# ---------------------------------------------------------------------------

def ok(doc: dict) -> None:
    errors = validate_dict(copy.deepcopy(doc))
    assert errors == [], f"Expected no errors but got:\n" + "\n".join(f"  {e}" for e in errors)


def fail(doc: dict, *phrases: str) -> None:
    """Assert validation fails; optionally verify that every phrase appears in the error list."""
    errors = validate_dict(copy.deepcopy(doc))
    assert errors, "Expected validation errors but got none"
    for phrase in phrases:
        assert any(phrase in e for e in errors), (
            f"Expected {phrase!r} in errors but got:\n" + "\n".join(f"  {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# Happy paths — valid documents that must pass
# ---------------------------------------------------------------------------

class TestHappyPaths:
    def test_onprem_remote(self):
        ok(_onprem_doc())

    def test_onprem_remote_and_local(self):
        ok(_onprem_doc(storage_location="remote_and_local"))

    def test_onprem_local(self):
        ok(_local_doc())

    def test_cloud(self):
        ok(_cloud_doc())

    def test_onprem_with_one_switch(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_switches"] = [_switch(rack_units=2, unit_count=1)]
        doc["system_under_test"]["total_rack_units"] = 1 + 2   # 1 node + 1×2RU switch
        ok(doc)

    def test_onprem_with_rack_power_supplies(self):
        doc = _onprem_doc()
        doc["system_under_test"]["rack_power_supplies"] = [_power()]
        ok(doc)

    def test_onprem_multiple_node_groups(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"] = [
            _onprem_node(rack_units=2, quantity=3),   # 6 RU
            _onprem_node(rack_units=1, quantity=2),   # 2 RU
        ]
        doc["system_under_test"]["total_rack_units"] = 8
        ok(doc)

    def test_psu_min_equals_total(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["chassis"]["power"]["min_psus_active"] = 2
        ok(doc)

    def test_capabilities_write_false_nonzero_remap(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["capabilities"] = _capabilities(sw=False, sr=True, remap=10)
        ok(doc)

    def test_capabilities_read_false_nonzero_remap(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["capabilities"] = _capabilities(sw=True, sr=False, remap=10)
        ok(doc)

    def test_capabilities_both_false_nonzero_remap(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["capabilities"] = _capabilities(sw=False, sr=False, remap=15)
        ok(doc)

    def test_na_pairing_both_na(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["architecture"]["client_footprint"] = "n/a"
        doc["system_under_test"]["solution"]["architecture"]["client_installation"] = "n/a"
        ok(doc)

    def test_na_pairing_neither_na(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["architecture"]["client_footprint"] = "closed_source"
        doc["system_under_test"]["solution"]["architecture"]["client_installation"] = "installable"
        ok(doc)

    def test_cloud_no_total_rack_units(self):
        doc = _cloud_doc()
        doc["system_under_test"].pop("total_rack_units", None)
        ok(doc)

    def test_onprem_empty_product_switches_list(self):
        """An explicit empty product_switches list must not trigger rule 6."""
        doc = _onprem_doc()
        doc["system_under_test"]["product_switches"] = []
        ok(doc)


# ---------------------------------------------------------------------------
# Strict-extras: unknown fields must surface as errors, not be silently dropped
# ---------------------------------------------------------------------------

class TestExtraFieldsForbidden:
    def test_extra_field_on_psu_rejected(self):
        """Regression: power_capacity_watts was historically tolerated and silently dropped."""
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["chassis"]["power"]["psus_configured"][0]["power_capacity_watts"] = 1800
        fail(doc, "power_capacity_watts")

    def test_extra_field_on_chassis_rejected(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["chassis"]["bogus_field"] = "oops"
        fail(doc, "bogus_field")

    def test_extra_field_on_solution_rejected(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["typo_field"] = True
        fail(doc, "typo_field")

    def test_extra_field_at_root_rejected(self):
        doc = _onprem_doc()
        doc["unexpected_top_level_key"] = 42
        fail(doc, "unexpected_top_level_key")


# ---------------------------------------------------------------------------
# Phase 1: base structural / type / enum / constraint validation
# ---------------------------------------------------------------------------

class TestBaseSchemaValidation:
    def test_missing_system_under_test(self):
        fail({}, "system_under_test")

    def test_missing_submission_name(self):
        doc = _onprem_doc()
        del doc["system_under_test"]["solution"]["submission_name"]
        fail(doc, "submission_name")

    def test_empty_submission_name(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["submission_name"] = ""
        fail(doc, "submission_name")

    def test_missing_deployment(self):
        doc = _onprem_doc()
        del doc["system_under_test"]["deployment"]
        fail(doc, "deployment")

    def test_invalid_deployment_value(self):
        doc = _onprem_doc()
        doc["system_under_test"]["deployment"] = "hybrid"
        fail(doc, "deployment")

    def test_missing_clients(self):
        doc = _onprem_doc()
        del doc["system_under_test"]["clients"]
        fail(doc, "clients")

    def test_invalid_storage_location(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["architecture"]["storage_location"] = "offsite"
        fail(doc, "storage_location")

    def test_invalid_benchmark_api(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["architecture"]["benchmark_API"] = "nvme"
        fail(doc, "benchmark_API")

    def test_invalid_product_api(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["architecture"]["product_API"] = "tape"
        fail(doc, "product_API")

    def test_invalid_client_footprint(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["architecture"]["client_footprint"] = "freeware"
        fail(doc, "client_footprint")

    def test_invalid_client_installation(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["architecture"]["client_installation"] = "burned_in"
        fail(doc, "client_installation")

    def test_remap_time_negative(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["capabilities"]["remap_time_in_seconds"] = -1
        fail(doc)

    def test_invalid_network_type(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["networking"][0]["type"] = "fibrechannel"
        fail(doc)

    def test_invalid_traffic_type(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["networking"][0]["traffic"] = ["storage"]
        fail(doc)

    def test_empty_traffic_list(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["networking"][0]["traffic"] = []
        fail(doc, "traffic")

    def test_invalid_drive_interface(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["drives"] = [{
            "unit_count": 1, "vendor_name": "Micron", "model_name": "9400",
            "interface": "fibrechannel", "media_type": "TLC", "capacity_in_GB": 8000,
        }]
        fail(doc)

    def test_invalid_power_efficiency(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["chassis"]["power"] \
            ["psus_configured"][0]["efficiency"] = "Diamond"
        fail(doc)

    def test_chassis_rack_units_zero(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["chassis"]["rack_units"] = 0
        fail(doc)

    def test_node_quantity_zero(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["quantity"] = 0
        fail(doc, "quantity")

    def test_empty_psus_configured(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["chassis"]["power"]["psus_configured"] = []
        fail(doc, "psus_configured")

    def test_missing_operating_system(self):
        doc = _onprem_doc()
        del doc["system_under_test"]["product_nodes"][0]["operating_system"]
        fail(doc, "operating_system")

    def test_missing_solution_architecture(self):
        doc = _onprem_doc()
        del doc["system_under_test"]["solution"]["architecture"]
        fail(doc, "architecture")

    def test_missing_solution_capabilities(self):
        doc = _onprem_doc()
        del doc["system_under_test"]["solution"]["capabilities"]
        fail(doc, "capabilities")

    def test_empty_switch_ports(self):
        doc = _onprem_doc()
        sw = _switch()
        sw["ports"] = []
        doc["system_under_test"]["product_switches"] = [sw]
        doc["system_under_test"]["total_rack_units"] = 1 + 2
        fail(doc, "ports")

    def test_network_speed_zero(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["networking"][0]["speed"] = 0
        fail(doc)

    def test_memory_capacity_zero(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["chassis"]["memory_capacity"] = 0
        fail(doc)


# ---------------------------------------------------------------------------
# Rules 1–6: onprem deployment requirements
# ---------------------------------------------------------------------------

class TestOnpremDeploymentRules:
    def test_rule1_node_missing_rack_units(self):
        doc = _onprem_doc()
        del doc["system_under_test"]["product_nodes"][0]["chassis"]["rack_units"]
        fail(doc, "product_nodes[0].chassis.rack_units")

    def test_rule2_node_missing_power(self):
        doc = _onprem_doc()
        del doc["system_under_test"]["product_nodes"][0]["chassis"]["power"]
        fail(doc, "product_nodes[0].chassis.power")

    def test_rule3_client_missing_rack_units(self):
        doc = _onprem_doc()
        del doc["system_under_test"]["clients"][0]["chassis"]["rack_units"]
        fail(doc, "clients[0].chassis.rack_units")

    def test_rule4_client_missing_power(self):
        doc = _onprem_doc()
        del doc["system_under_test"]["clients"][0]["chassis"]["power"]
        fail(doc, "clients[0].chassis.power")

    def test_rule5_missing_total_rack_units(self):
        doc = _onprem_doc()
        del doc["system_under_test"]["total_rack_units"]
        fail(doc, "total_rack_units")

    def test_rules1_2_second_node_checked(self):
        """Validator must check every node, not just the first."""
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"].append(_onprem_node())
        doc["system_under_test"]["total_rack_units"] = 2
        del doc["system_under_test"]["product_nodes"][1]["chassis"]["rack_units"]
        fail(doc, "product_nodes[1].chassis.rack_units")

    def test_rules3_4_second_client_checked(self):
        """Validator must check every client, not just the first."""
        doc = _onprem_doc()
        doc["system_under_test"]["clients"].append(_onprem_client())
        del doc["system_under_test"]["clients"][1]["chassis"]["power"]
        fail(doc, "clients[1].chassis.power")

    def test_rule6_switch_missing_power(self):
        doc = _onprem_doc()
        sw = _switch()
        del sw["power"]
        doc["system_under_test"]["product_switches"] = [sw]
        doc["system_under_test"]["total_rack_units"] = 1 + 2
        fail(doc, "product_switches[0].power")

    def test_rule6_switch_missing_rack_units(self):
        # rack_units is unconditionally required by the base SwitchDescription schema
        doc = _onprem_doc()
        sw = _switch()
        del sw["rack_units"]
        doc["system_under_test"]["product_switches"] = [sw]
        doc["system_under_test"]["total_rack_units"] = 1
        fail(doc, "rack_units")

    def test_rule6_second_switch_missing_power(self):
        """Validator must check every switch."""
        doc = _onprem_doc()
        sw2 = _switch(rack_units=1)
        del sw2["power"]
        doc["system_under_test"]["product_switches"] = [_switch(rack_units=2), sw2]
        doc["system_under_test"]["total_rack_units"] = 1 + 2 + 1
        fail(doc, "product_switches[1].power")

    def test_rule6_not_triggered_when_no_switches(self):
        ok(_onprem_doc())


# ---------------------------------------------------------------------------
# Rules 7–10: cloud deployment requirements
# ---------------------------------------------------------------------------

class TestCloudDeploymentRules:
    def test_rule7_cloud_with_product_switches(self):
        doc = _cloud_doc()
        doc["system_under_test"]["product_switches"] = [_switch()]
        fail(doc, "product_switches")

    def test_rule8_cloud_with_rack_power_supplies(self):
        doc = _cloud_doc()
        doc["system_under_test"]["rack_power_supplies"] = [_power()]
        fail(doc, "rack_power_supplies")

    def test_rule9_cloud_node_with_rack_units(self):
        doc = _cloud_doc()
        doc["system_under_test"]["product_nodes"][0]["chassis"]["rack_units"] = 2
        fail(doc, "product_nodes[0].chassis.rack_units")

    def test_rule10_cloud_node_with_power(self):
        doc = _cloud_doc()
        doc["system_under_test"]["product_nodes"][0]["chassis"]["power"] = _power()
        fail(doc, "product_nodes[0].chassis.power")

    def test_rule9_cloud_client_with_rack_units(self):
        doc = _cloud_doc()
        doc["system_under_test"]["clients"][0]["chassis"]["rack_units"] = 1
        fail(doc, "clients[0].chassis.rack_units")

    def test_rule10_cloud_client_with_power(self):
        doc = _cloud_doc()
        doc["system_under_test"]["clients"][0]["chassis"]["power"] = _power()
        fail(doc, "clients[0].chassis.power")

    def test_rule9_cloud_second_node_with_rack_units(self):
        """Validator checks every node."""
        doc = _cloud_doc()
        doc["system_under_test"]["product_nodes"].append(_cloud_node())
        doc["system_under_test"]["product_nodes"][1]["chassis"]["rack_units"] = 1
        fail(doc, "product_nodes[1].chassis.rack_units")


# ---------------------------------------------------------------------------
# Rule 11: total_rack_units arithmetic
# ---------------------------------------------------------------------------

class TestRackUnitsArithmetic:
    def test_correct_single_node(self):
        ok(_onprem_doc())   # 1 node × 1 RU × qty 1 = 1

    def test_sum_too_low(self):
        doc = _onprem_doc()
        doc["system_under_test"]["total_rack_units"] = 0
        fail(doc, "total_rack_units")

    def test_sum_too_high(self):
        doc = _onprem_doc()
        doc["system_under_test"]["total_rack_units"] = 99
        fail(doc, "total_rack_units")

    def test_correct_with_quantity(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"] = [_onprem_node(rack_units=1, quantity=4)]
        doc["system_under_test"]["total_rack_units"] = 4
        ok(doc)

    def test_incorrect_with_quantity(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"] = [_onprem_node(rack_units=1, quantity=4)]
        doc["system_under_test"]["total_rack_units"] = 3
        fail(doc, "total_rack_units")

    def test_correct_with_switches(self):
        doc = _onprem_doc()
        # 1 node×1RU×1 + 2 switches×2RU each = 1 + 4 = 5
        doc["system_under_test"]["product_switches"] = [_switch(rack_units=2, unit_count=2)]
        doc["system_under_test"]["total_rack_units"] = 5
        ok(doc)

    def test_incorrect_excludes_switch_rus(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_switches"] = [_switch(rack_units=2, unit_count=2)]
        doc["system_under_test"]["total_rack_units"] = 1   # omits switch contribution
        fail(doc, "total_rack_units")

    def test_clients_excluded_from_sum(self):
        """Large client fleet must not change the required total_rack_units value."""
        doc = _onprem_doc()
        # Add 10 more clients each taking 2 RU — total_rack_units must still be 1
        doc["system_under_test"]["clients"].append(_onprem_client(rack_units=2, quantity=10))
        doc["system_under_test"]["total_rack_units"] = 1
        ok(doc)

    def test_multiple_node_groups_correct(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"] = [
            _onprem_node(rack_units=2, quantity=4),   # 8 RU
            _onprem_node(rack_units=1, quantity=3),   # 3 RU
        ]
        doc["system_under_test"]["total_rack_units"] = 11
        ok(doc)

    def test_multiple_node_groups_off_by_one(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"] = [
            _onprem_node(rack_units=2, quantity=4),
            _onprem_node(rack_units=1, quantity=3),
        ]
        doc["system_under_test"]["total_rack_units"] = 10   # should be 11
        fail(doc, "total_rack_units")

    def test_nodes_and_switches_combined(self):
        doc = _onprem_doc()
        # 2 nodes×2RU×3 = 12, plus 1 switch×4RU×2 = 8 → total 20
        doc["system_under_test"]["product_nodes"] = [_onprem_node(rack_units=2, quantity=3)]
        doc["system_under_test"]["product_switches"] = [_switch(rack_units=4, unit_count=2)]
        doc["system_under_test"]["total_rack_units"] = 6 + 8
        ok(doc)


# ---------------------------------------------------------------------------
# Rule 12: power_device.min_psus_active <= total installed PSU count
# ---------------------------------------------------------------------------

class TestPsuCount:
    def test_min_one_total_two(self):
        ok(_onprem_doc())   # _power() default: min=1, unit_count=2

    def test_min_equals_total(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["chassis"]["power"]["min_psus_active"] = 2
        ok(doc)

    def test_min_zero_invalid(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["chassis"]["power"]["min_psus_active"] = 0
        fail(doc, "min_psus_active")

    def test_min_exceeds_total(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["chassis"]["power"]["min_psus_active"] = 3
        fail(doc, "min_psus_active")

    def test_multi_entry_sum_valid(self):
        """Two PSU entries with unit_count=1 each sum to 2; min=2 is valid."""
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["chassis"]["power"] = {
            "min_psus_active": 2,
            "psus_configured": [_psu(unit_count=1), _psu(unit_count=1)],
        }
        ok(doc)

    def test_multi_entry_sum_exceeded(self):
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["chassis"]["power"] = {
            "min_psus_active": 3,
            "psus_configured": [_psu(unit_count=1), _psu(unit_count=1)],
        }
        fail(doc, "min_psus_active")

    def test_switch_psu_count_enforced(self):
        doc = _onprem_doc()
        sw = _switch()
        sw["power"]["min_psus_active"] = 5   # exceeds unit_count=2
        doc["system_under_test"]["product_switches"] = [sw]
        doc["system_under_test"]["total_rack_units"] = 1 + 2
        fail(doc, "min_psus_active")

    def test_client_psu_count_enforced(self):
        doc = _onprem_doc()
        doc["system_under_test"]["clients"][0]["chassis"]["power"]["min_psus_active"] = 9
        fail(doc, "min_psus_active")


# ---------------------------------------------------------------------------
# Rule 13: remap_time_in_seconds biconditional
#   both flags true  → remap == 0
#   either flag false → remap > 0
# ---------------------------------------------------------------------------

class TestRemapTimeBiconditional:
    def test_both_true_remap_zero_passes(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["capabilities"] = _capabilities(sw=True, sr=True, remap=0)
        ok(doc)

    def test_both_true_remap_nonzero_fails(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["capabilities"] = _capabilities(sw=True, sr=True, remap=5)
        fail(doc, "remap_time_in_seconds")

    def test_write_false_read_true_remap_zero_fails(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["capabilities"] = _capabilities(sw=False, sr=True, remap=0)
        fail(doc, "remap_time_in_seconds")

    def test_write_false_read_true_remap_nonzero_passes(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["capabilities"] = _capabilities(sw=False, sr=True, remap=15)
        ok(doc)

    def test_write_true_read_false_remap_zero_fails(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["capabilities"] = _capabilities(sw=True, sr=False, remap=0)
        fail(doc, "remap_time_in_seconds")

    def test_write_true_read_false_remap_nonzero_passes(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["capabilities"] = _capabilities(sw=True, sr=False, remap=15)
        ok(doc)

    def test_both_false_remap_zero_fails(self):
        doc = _local_doc()
        doc["system_under_test"]["solution"]["capabilities"]["remap_time_in_seconds"] = 0
        fail(doc, "remap_time_in_seconds")

    def test_both_false_remap_nonzero_passes(self):
        ok(_local_doc())   # _local_doc has both=False, remap=30


# ---------------------------------------------------------------------------
# Rule 14: client_footprint n/a ↔ client_installation n/a
# ---------------------------------------------------------------------------

class TestNaPairing:
    def test_footprint_na_installation_not_na(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["architecture"]["client_footprint"] = "n/a"
        doc["system_under_test"]["solution"]["architecture"]["client_installation"] = "in_box"
        fail(doc, "n/a")

    def test_footprint_not_na_installation_na(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["architecture"]["client_footprint"] = "open_source"
        doc["system_under_test"]["solution"]["architecture"]["client_installation"] = "n/a"
        fail(doc, "n/a")

    def test_both_na_passes(self):
        ok(_local_doc())   # _local_doc uses n/a / n/a

    def test_neither_na_passes(self):
        ok(_onprem_doc())   # open_source / in_box

    def test_closed_source_installable_passes(self):
        doc = _onprem_doc()
        doc["system_under_test"]["solution"]["architecture"]["client_footprint"] = "closed_source"
        doc["system_under_test"]["solution"]["architecture"]["client_installation"] = "installable"
        ok(doc)


# ---------------------------------------------------------------------------
# Rules 15–17: networking requirements driven by storage_location
# ---------------------------------------------------------------------------

class TestNetworkingRequirements:
    # Rule 15: remote → product_nodes must have networking
    def test_rule15_remote_node_missing_networking(self):
        doc = _onprem_doc(storage_location="remote")
        del doc["system_under_test"]["product_nodes"][0]["networking"]
        fail(doc, "product_nodes[0].networking")

    def test_rule15_remote_node_empty_networking(self):
        doc = _onprem_doc(storage_location="remote")
        doc["system_under_test"]["product_nodes"][0]["networking"] = []
        fail(doc, "product_nodes[0].networking")

    def test_rule15_remote_and_local_node_missing_networking(self):
        doc = _onprem_doc(storage_location="remote_and_local")
        del doc["system_under_test"]["product_nodes"][0]["networking"]
        fail(doc, "product_nodes[0].networking")

    # Rule 16: remote → clients must have networking
    def test_rule16_remote_client_missing_networking(self):
        doc = _onprem_doc(storage_location="remote")
        del doc["system_under_test"]["clients"][0]["networking"]
        fail(doc, "clients[0].networking")

    def test_rule16_remote_client_empty_networking(self):
        doc = _onprem_doc(storage_location="remote")
        doc["system_under_test"]["clients"][0]["networking"] = []
        fail(doc, "clients[0].networking")

    def test_rule16_remote_and_local_client_missing_networking(self):
        doc = _onprem_doc(storage_location="remote_and_local")
        del doc["system_under_test"]["clients"][0]["networking"]
        fail(doc, "clients[0].networking")

    def test_rule16_second_client_missing_networking(self):
        """Every client in the list is checked."""
        doc = _onprem_doc(storage_location="remote")
        doc["system_under_test"]["clients"].append(_onprem_client())
        del doc["system_under_test"]["clients"][1]["networking"]
        fail(doc, "clients[1].networking")

    # Rule 17: local → product_switches must be absent
    def test_rule17_local_with_switches_fails(self):
        doc = _local_doc()
        sw = _switch(rack_units=2, unit_count=1)
        doc["system_under_test"]["product_switches"] = [sw]
        doc["system_under_test"]["total_rack_units"] = 1 + 2
        fail(doc, "product_switches")

    # local storage should not require networking (rules 15/16 don't apply)
    def test_local_no_networking_required(self):
        ok(_local_doc())

    # cloud inherits the remote networking requirement (storage_location=remote)
    def test_cloud_node_missing_networking(self):
        doc = _cloud_doc()
        del doc["system_under_test"]["product_nodes"][0]["networking"]
        fail(doc, "product_nodes[0].networking")

    def test_cloud_client_missing_networking(self):
        doc = _cloud_doc()
        del doc["system_under_test"]["clients"][0]["networking"]
        fail(doc, "clients[0].networking")


# ---------------------------------------------------------------------------
# DriveInstance: media_type required; performance block optional but, if
# present, its four throughput/IOPS fields are required.
# ---------------------------------------------------------------------------

class TestDriveInstance:
    def _doc_with_drives(self, *drives) -> dict:
        doc = _onprem_doc()
        doc["system_under_test"]["product_nodes"][0]["drives"] = list(drives)
        return doc

    # ----- media_type (required) -----

    def test_drive_without_performance_passes(self):
        ok(self._doc_with_drives(_drive()))

    def test_drive_missing_media_type_fails(self):
        d = _drive()
        del d["media_type"]
        fail(self._doc_with_drives(d), "media_type")

    def test_drive_invalid_media_type_fails(self):
        fail(self._doc_with_drives(_drive(media_type="SLC")), "media_type")

    def test_drive_media_type_HDD_passes(self):
        ok(self._doc_with_drives(_drive(media_type="HDD")))

    def test_drive_media_type_QLC_passes(self):
        ok(self._doc_with_drives(_drive(media_type="QLC")))

    def test_drive_media_type_other_passes(self):
        ok(self._doc_with_drives(_drive(media_type="other")))

    # ----- performance block (optional) -----

    def test_drive_with_full_performance_passes(self):
        ok(self._doc_with_drives(_drive(with_performance=True)))

    def test_drive_performance_without_optional_fields_passes(self):
        d = _drive(with_performance=True)
        del d["performance"]["form_factor"]
        del d["performance"]["interface_speed"]
        ok(self._doc_with_drives(d))

    def test_drive_performance_missing_seq_read_fails(self):
        d = _drive(with_performance=True)
        del d["performance"]["seq_read_MBps"]
        fail(self._doc_with_drives(d), "seq_read_MBps")

    def test_drive_performance_missing_seq_write_fails(self):
        d = _drive(with_performance=True)
        del d["performance"]["seq_write_MBps"]
        fail(self._doc_with_drives(d), "seq_write_MBps")

    def test_drive_performance_missing_random_read_fails(self):
        d = _drive(with_performance=True)
        del d["performance"]["random_read_IOPS"]
        fail(self._doc_with_drives(d), "random_read_IOPS")

    def test_drive_performance_missing_random_write_fails(self):
        d = _drive(with_performance=True)
        del d["performance"]["random_write_IOPS"]
        fail(self._doc_with_drives(d), "random_write_IOPS")

    def test_drive_performance_zero_seq_read_fails(self):
        d = _drive(with_performance=True)
        d["performance"]["seq_read_MBps"] = 0
        fail(self._doc_with_drives(d), "seq_read_MBps")

    def test_drive_performance_negative_iops_fails(self):
        d = _drive(with_performance=True)
        d["performance"]["random_write_IOPS"] = -1
        fail(self._doc_with_drives(d), "random_write_IOPS")

    def test_drive_performance_invalid_form_factor_fails(self):
        d = _drive(with_performance=True)
        d["performance"]["form_factor"] = "PCIe"
        fail(self._doc_with_drives(d), "form_factor")

    def test_drive_performance_valid_form_factors(self):
        for ff in ["U.2", "U.3", "M.2", "E1.S", "E1.L", "E3.S", "E3.L",
                   "2.5in", "3.5in", "AIC", "other"]:
            d = _drive(with_performance=True)
            d["performance"]["form_factor"] = ff
            ok(self._doc_with_drives(d))

    def test_drive_performance_empty_interface_speed_fails(self):
        d = _drive(with_performance=True)
        d["performance"]["interface_speed"] = ""
        fail(self._doc_with_drives(d), "interface_speed")

    def test_mixed_drive_groups(self):
        """Drives list can mix entries with and without the performance block."""
        ok(self._doc_with_drives(
            _drive(with_performance=True),
            _drive(model_name="X1650 QLC", media_type="QLC", capacity_in_GB=24000),
        ))
