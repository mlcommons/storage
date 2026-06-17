"""Tests for SystemYamlSchemaCheck (Phase 2 Plan 02-02).

Covers ROADMAP Phase 2 success criterion #7: the schema validation pass that
runs once before the per-benchmark loader loop (D-A1) emits rule-ID-tagged
violations for every Pydantic schema error found in systems/<name>.yaml files.

References:
  - D-A1: SystemYamlSchemaCheck instantiated in main.py before for-loop
  - D-A2: SCHEMA_ERROR_RULE_MAP + unmapped-fallback ("2.1.7","systemsDirectoryFiles")
  - D-A3: CHKPT-04 and CHKPT-05 are runtime cross-checks; schema handles presence+type

Test coverage:
  TestSchemaCheck_DefaultClean      — positive case (valid YAML -> zero violations)
  TestSchemaCheck_BadType           — type errors map to 4.7.4
  TestSchemaCheck_MissingField      — missing-required-field errors
  TestSchemaCheck_Rule13CrossField  — Capabilities.check_remap_time cross-field -> 4.7.3
  TestSchemaCheck_UnmappedFallback  — unmapped loc -> 2.1.7 (Resolution A via bad deployment)
  TestSchemaCheck_OpenSubtree       — open/ subtree also walked (D-A1 both closed+open)
  TestSchemaCheck_AccumulateDontAbort — two errors in one YAML -> two records (QUAL-01)
"""

import os
import pathlib

import pytest
import yaml

from mlpstorage_py.submission_checker.checks.system_yaml_schema_checks import (
    SystemYamlSchemaCheck,
)
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.tests.conftest import build_submission

# mock_logger fixture comes from conftest.py automatically (no import needed).


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_check(mock_logger, root_path: pathlib.Path) -> SystemYamlSchemaCheck:
    """Instantiate SystemYamlSchemaCheck wired to a mock_logger."""
    config = Config(version="v2.0", submitters=["Acme"], skip_output_file=True)
    return SystemYamlSchemaCheck(
        log=mock_logger,
        config=config,
        root_path=str(root_path),
    )


# ---------------------------------------------------------------------------
# TestSchemaCheck_DefaultClean — positive case
# ---------------------------------------------------------------------------

class TestSchemaCheck_DefaultClean:
    """Default fixture produces a schema-valid system YAML -> zero violations."""

    def test_default_fixture_passes(self, tmp_path, mock_logger):
        """A clean build_submission tree with schema-valid system YAML passes."""
        root = build_submission(tmp_path)
        check = _make_check(mock_logger, root)
        result = check()
        assert result is True, f"expected True, got {result!r}"
        assert mock_logger.errors == [], (
            f"expected zero violations, got: {mock_logger.errors}"
        )


# ---------------------------------------------------------------------------
# TestSchemaCheck_BadType — type-error violations -> 4.7.4
# ---------------------------------------------------------------------------

class TestSchemaCheck_BadType:
    """Type errors on capability booleans map to rule 4.7.4.

    Pydantic v2 coerces truthy/falsy strings ('yes', 'true', 'false', 'no') and
    int 0/1 to bool without error (lax validation mode).  Tests use values that
    actually trigger a Pydantic ValidationError at the field loc:
      - String 'maybe' / 'not-a-bool' → 'unable to interpret input'
      - Int 2 (not 0 or 1) → 'unable to interpret input'
    These produce loc 'system_under_test -> solution -> capabilities -> <field>'
    which maps to ('4.7.4', 'checkpointSimultaneousRwSupport') in
    SCHEMA_ERROR_RULE_MAP.
    """

    def test_simultaneous_write_invalid_string_emits_4_7_4(
        self, tmp_path, mock_logger
    ):
        """simultaneous_write='not-a-bool' (unrecognised string) emits [4.7.4 ...]."""
        root = build_submission(
            tmp_path,
            system_yaml_bad_capabilities={"simultaneous_write": "not-a-bool"},
        )
        check = _make_check(mock_logger, root)
        result = check()
        assert result is False, "expected False (schema errors present)"
        assert any(
            m.startswith("[4.7.4 checkpointSimultaneousRwSupport]")
            for m in mock_logger.errors
        ), f"expected [4.7.4 ...] prefix in errors; got: {mock_logger.errors}"

    def test_simultaneous_read_invalid_int_emits_4_7_4(
        self, tmp_path, mock_logger
    ):
        """simultaneous_read=2 (int outside [0,1]) emits [4.7.4 ...]."""
        root = build_submission(
            tmp_path,
            # simultaneous_write stays True so Rule-13 doesn't also fire;
            # only simultaneous_read gets an invalid int
            system_yaml_bad_capabilities={"simultaneous_read": 2},
        )
        check = _make_check(mock_logger, root)
        result = check()
        assert result is False
        assert any(
            m.startswith("[4.7.4 checkpointSimultaneousRwSupport]")
            for m in mock_logger.errors
        ), f"expected [4.7.4 ...] in errors; got: {mock_logger.errors}"

    def test_multi_host_invalid_string_emits_4_7_4(self, tmp_path, mock_logger):
        """multi_host='maybe' (unrecognised string) emits [4.7.4 ...]."""
        root = build_submission(
            tmp_path,
            system_yaml_bad_capabilities={"multi_host": "maybe"},
        )
        check = _make_check(mock_logger, root)
        result = check()
        assert result is False
        assert any(
            m.startswith("[4.7.4 checkpointSimultaneousRwSupport]")
            for m in mock_logger.errors
        ), f"expected [4.7.4 ...] in errors; got: {mock_logger.errors}"


# ---------------------------------------------------------------------------
# TestSchemaCheck_MissingField — missing required fields
# ---------------------------------------------------------------------------

class TestSchemaCheck_MissingField:
    """Missing required capability fields -> appropriate rule tags."""

    def test_missing_remap_time_emits_4_7_3(self, tmp_path, mock_logger):
        """Removing remap_time_in_seconds (required field) emits [4.7.3 ...]."""
        root = build_submission(
            tmp_path,
            system_yaml_bad_capabilities={"remove": ["remap_time_in_seconds"]},
        )
        check = _make_check(mock_logger, root)
        result = check()
        assert result is False
        assert any(
            m.startswith("[4.7.3 checkpointRemappingTimeReporting]")
            for m in mock_logger.errors
        ), f"expected [4.7.3 ...] in errors; got: {mock_logger.errors}"

    def test_missing_multi_host_emits_4_7_4(self, tmp_path, mock_logger):
        """Removing multi_host (required field) emits [4.7.4 ...]."""
        root = build_submission(
            tmp_path,
            system_yaml_bad_capabilities={"remove": ["multi_host"]},
        )
        check = _make_check(mock_logger, root)
        result = check()
        assert result is False
        assert any(
            m.startswith("[4.7.4 checkpointSimultaneousRwSupport]")
            for m in mock_logger.errors
        ), f"expected [4.7.4 ...] in errors; got: {mock_logger.errors}"


# ---------------------------------------------------------------------------
# TestSchemaCheck_Rule13CrossField — empirical loc verified in Task 1
# ---------------------------------------------------------------------------

class TestSchemaCheck_Rule13CrossField:
    """Rule-13 cross-field violation (Capabilities.check_remap_time) -> 4.7.3.

    Empirical verification (2026-06-10): the Pydantic v2 model_validator(mode='after')
    on Capabilities fires at loc 'system_under_test -> solution -> capabilities',
    which is mapped to ('4.7.3', 'checkpointRemappingTimeReporting') in
    SCHEMA_ERROR_RULE_MAP.  This test asserts on the emitted rule-ID prefix rather
    than on the raw loc string so it remains correct regardless of minor Pydantic
    version differences.
    """

    def test_rule13_emits_4_7_3(self, tmp_path, mock_logger):
        """system_yaml_rule13_violation=True emits [4.7.3 ...] via cross-field check."""
        root = build_submission(tmp_path, system_yaml_rule13_violation=True)
        check = _make_check(mock_logger, root)
        result = check()
        assert result is False
        assert any(
            m.startswith("[4.7.3 checkpointRemappingTimeReporting]")
            for m in mock_logger.errors
        ), f"expected [4.7.3 ...] in errors; got: {mock_logger.errors}"


# ---------------------------------------------------------------------------
# TestSchemaCheck_UnmappedFallback — unknown loc -> 2.1.7 (Resolution A)
# ---------------------------------------------------------------------------

class TestSchemaCheck_UnmappedFallback:
    """Unmapped Pydantic loc string falls back to rule 2.1.7.

    Resolution A (review-incorporation): Pydantic v2 default is extra='ignore'
    and schema_validator.py declares no model_config = ConfigDict(extra='forbid')
    on any model.  The original approach of injecting an unknown field passes
    vacuously (Pydantic silently ignores it, zero errors emitted).  Instead, we
    use a TYPE error at a known-but-unmapped field: system_under_test.deployment
    is typed DeploymentMode (enum, schema_validator.py:57).  Setting it to int
    12345 emits a ValidationError at loc 'system_under_test -> deployment' which
    is NOT in SCHEMA_ERROR_RULE_MAP (only the four capability paths and the
    Rule-13 cross-field key are mapped).  The violation therefore falls through to
    ("2.1.7", "systemsDirectoryFiles") per D-A2.

    Empirical verification (2026-06-10): validate_dict({'system_under_test':
    {'deployment': 12345, ...}}) -> error at loc 'system_under_test -> deployment'.
    """

    def test_bad_deployment_emits_2_1_7(self, tmp_path, mock_logger):
        """deployment=12345 (non-enum) triggers unmapped loc -> [2.1.7 ...] fallback."""
        root = build_submission(tmp_path, system_yaml_bad_deployment=12345)
        check = _make_check(mock_logger, root)
        result = check()
        assert result is False
        assert any(
            m.startswith("[2.1.7 systemsDirectoryFiles]")
            for m in mock_logger.errors
        ), f"expected [2.1.7 ...] fallback in errors; got: {mock_logger.errors}"


# ---------------------------------------------------------------------------
# TestSchemaCheck_OpenSubtree — open/ subtree also walked (D-A1)
# ---------------------------------------------------------------------------

class TestSchemaCheck_OpenSubtree:
    """D-A1 requires walking both closed/ AND open/ subtrees."""

    def test_open_subtree_yaml_also_validated(self, tmp_path, mock_logger):
        """A Rule-13-violating YAML under open/ surfaces a violation citing open/."""
        # Build a base tree with a valid closed/ YAML
        root = build_submission(tmp_path)

        # Manually add an open/<submitter>/systems/<name>.yaml that violates Rule 13
        open_systems = root / "open" / "Acme" / "systems"
        open_systems.mkdir(parents=True)
        rule13_caps = {
            "multi_host": True,
            "simultaneous_write": True,
            "simultaneous_read": True,
            "remap_time_in_seconds": 5,  # Rule-13 violation
        }
        bad_yaml = {
            "system_under_test": {
                "solution": {
                    "submission_name": "acme-open-v1",
                    "friendly_description": "Open submission test system",
                    "architecture": {
                        "storage_location": "local",
                        "benchmark_API": "file",
                        "product_API": "file",
                        "client_footprint": "open_source",
                        "client_installation": "in_box",
                    },
                    "capabilities": rule13_caps,
                },
                "deployment": "cloud",
                "clients": [
                    {
                        "friendly_description": "Client",
                        "quantity": 1,
                        "chassis": {
                            "model_name": "Server",
                            "cpu_model": "Xeon",
                            "cpu_qty": 2,
                            "cpu_cores": 32,
                            "memory_capacity": 128,
                        },
                        "operating_system": {"name": "Ubuntu", "version": "22.04"},
                    }
                ],
            }
        }
        yaml_file = open_systems / "acme-open-v1.yaml"
        yaml_file.write_text(
            yaml.dump(bad_yaml, default_flow_style=False), encoding="utf-8"
        )

        check = _make_check(mock_logger, root)
        result = check()
        assert result is False, "expected False (open/ YAML has Rule-13 violation)"
        # The path in the error message must contain the open/ segment
        open_errors = [
            m for m in mock_logger.errors
            if "open" + os.sep in m or "open/" in m
        ]
        assert len(open_errors) >= 1, (
            f"expected at least one error mentioning 'open/'; got: {mock_logger.errors}"
        )


# ---------------------------------------------------------------------------
# TestSchemaCheck_AccumulateDontAbort — two violations in one YAML (QUAL-01)
# ---------------------------------------------------------------------------

class TestSchemaCheck_AccumulateDontAbort:
    """Two violations in one YAML file produce two error records (QUAL-01)."""

    def test_multiple_errors_in_one_yaml(self, tmp_path, mock_logger):
        """Two violations in one YAML produce two error records (no short-circuit)."""
        # Build a base tree then overwrite the system YAML with two violations:
        #   1. simultaneous_write='yes' (type error -> 4.7.4)
        #   2. multi_host removed (missing required field -> 4.7.4)
        root = build_submission(tmp_path)
        yaml_path = list((root / "closed").glob("*/systems/*.yaml"))[0]
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        caps = data["system_under_test"]["solution"]["capabilities"]
        caps["simultaneous_write"] = "yes"   # type error
        del caps["multi_host"]               # missing required field
        yaml_path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")

        check = _make_check(mock_logger, root)
        result = check()
        assert result is False
        assert len(mock_logger.errors) >= 2, (
            f"expected at least 2 error records (accumulate-don't-abort); "
            f"got {len(mock_logger.errors)}: {mock_logger.errors}"
        )
