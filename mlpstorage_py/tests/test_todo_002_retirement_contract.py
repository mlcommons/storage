"""Tripwire contract for retiring the TODO-002 deferred-runtime-cross-check
stubs.

The TODO-002 pattern (see ``training_checks.py`` and ``checkpointing_checks.py``)
covers three Rules.md rules whose runtime cross-check is deferred because
``summary.json`` doesn't currently expose the per-host data the check would
need:

  * 3.3.5 trainingDistributedDataAccessibility (TrainingCheck)
  * 3.3.7 trainingNodeCapabilityConsistency    (TrainingCheck)
  * 4.7.4 checkpointSimultaneousRwSupport      (CheckpointingCheck)

Each stub body emits an INFO-level note containing the marker ``TODO-002``,
returns True, and contributes no violation. The schema validator is the
"structural anchor" — for 4.7.4 specifically, three field-level entries in
``SystemYamlSchemaCheck.SCHEMA_ERROR_RULE_MAP`` tag schema errors with the
rule ID even though the runtime body is a stub.

Retirement of any of these stubs MUST satisfy all of the contract
properties pinned below. The intent is that a partial retirement (e.g.
"I upgraded the runtime body but forgot to keep the schema anchor", or
"I removed the TODO-002 marker but the body still returns True without
checking anything") fails this test loudly. Anyone touching a stub is
forced through this file, which doubles as the retirement checklist.

Retirement checklist (read this BEFORE flipping any assertion):

  1. The rule's @rule(...) decoration must remain in place — coverage
     reports rely on ``discover_rules`` finding the binding.
  2. The runtime body must emit a violation (``log_violation`` /
     ``log.error``) when the runtime data disagrees with the YAML
     declaration; mere ``log.info`` is not retirement.
  3. The runtime body must drop the ``TODO-002`` marker AND its
     ``runtime ... cross-check deferred`` phrasing.
  4. For 4.7.4 specifically: the three field-level entries in
     ``SCHEMA_ERROR_RULE_MAP`` (``simultaneous_write`` /
     ``simultaneous_read`` / ``multi_host``) must remain — they are the
     anchor that catches declaration-side defects that the runtime check
     can't see (e.g. a missing field in the YAML before any benchmark
     runs).
  5. For 3.3.5 / 3.3.7: the rule currently has NO schema anchor (no
     SCHEMA_ERROR_RULE_MAP entry). Retirement means the runtime check
     becomes the sole enforcement; if you also add a schema anchor as
     defense-in-depth, register it in ``EXPECTED_SCHEMA_ANCHORS`` below.
  6. The corresponding pinning tests
     (``TestChkpt05DeferredFollowUp`` in
     ``test_checkpointing_check_phase2.py`` and the
     ``test_3_3_5_*`` / ``test_3_3_7_*`` tests in
     ``test_training_check_retrofit.py``) must be flipped from
     "asserts deferred" to "asserts live", in the same commit.
  7. Remove the rule_id from ``DEFERRED_RULES`` in this file (last,
     once 1-6 are complete).

This file is intentionally separate from the existing per-method
pinning tests: those test "the current body emits info"; this file
tests "the contract that ties the three stubs together still holds",
so partial retirement still trips one of the cross-rule invariants.
"""

from __future__ import annotations

import inspect

import pytest

from mlpstorage_py.submission_checker.checks.checkpointing_checks import (
    CheckpointingCheck,
)
from mlpstorage_py.submission_checker.checks.system_yaml_schema_checks import (
    SystemYamlSchemaCheck,
)
from mlpstorage_py.submission_checker.checks.training_checks import (
    TrainingCheck,
)
from mlpstorage_py.submission_checker.rule_registry import discover_rules


# ---------------------------------------------------------------------------
# Contract registry — single source of truth for the TODO-002 retirement set
# ---------------------------------------------------------------------------

# (rule_id, rule_name, owning_check_class, method_name)
DEFERRED_RULES: list[tuple[str, str, type, str]] = [
    ("3.3.5",
     "trainingDistributedDataAccessibility",
     TrainingCheck,
     "distributed_data_accessibility_check"),
    ("3.3.7",
     "trainingNodeCapabilityConsistency",
     TrainingCheck,
     "node_capability_consistency_check"),
    ("4.7.4",
     "checkpointSimultaneousRwSupport",
     CheckpointingCheck,
     "simultaneous_rw_support"),
]

# rule_id -> set of SCHEMA_ERROR_RULE_MAP loc-string keys expected to point
# at it. Only 4.7.4 has a schema anchor today; 3.3.5 / 3.3.7 are pure-runtime
# rules whose Phase-2 "structural anchor" is the existence of the clients
# block (no per-rule schema field). If a future retirement adds a schema
# anchor for either, extend this dict in the same commit.
EXPECTED_SCHEMA_ANCHORS: dict[str, set[str]] = {
    "4.7.4": {
        "system_under_test -> solution -> capabilities -> simultaneous_write",
        "system_under_test -> solution -> capabilities -> simultaneous_read",
        "system_under_test -> solution -> capabilities -> multi_host",
    },
    # 3.3.5 / 3.3.7 intentionally absent — no schema anchor today.
}


# ---------------------------------------------------------------------------
# Contract assertions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule_id,rule_name,check_class,method_name",
                         DEFERRED_RULES,
                         ids=[r[0] for r in DEFERRED_RULES])
def test_rule_decoration_remains_in_place(
    rule_id, rule_name, check_class, method_name,
):
    """Contract item 1: ``@rule(rule_id, rule_name)`` is still on the method.

    Retirement upgrades the body, NOT the decoration. If the decoration is
    removed, ``mlpstorage rules-coverage`` will silently drop the rule from
    its inventory.
    """
    method = getattr(check_class, method_name, None)
    assert method is not None, (
        f"{check_class.__name__}.{method_name} is missing — retirement must "
        f"keep the method (it's the binding point for @rule)."
    )
    assert getattr(method, "__rule_id__", None) == rule_id, (
        f"{check_class.__name__}.{method_name}.__rule_id__ is "
        f"{getattr(method, '__rule_id__', None)!r}, expected {rule_id!r}. "
        f"Retirement must keep the @rule(...) decoration intact."
    )
    assert getattr(method, "__rule_name__", None) == rule_name, (
        f"{check_class.__name__}.{method_name}.__rule_name__ is "
        f"{getattr(method, '__rule_name__', None)!r}, expected {rule_name!r}."
    )

    rules = discover_rules(check_class)
    assert rule_id in rules, (
        f"discover_rules({check_class.__name__}) does not list {rule_id}; "
        f"got {sorted(rules)}. The drift gate "
        f"(`mlpstorage rules-coverage`) would silently lose coverage of "
        f"this rule."
    )


@pytest.mark.parametrize("rule_id,rule_name,check_class,method_name",
                         DEFERRED_RULES,
                         ids=[r[0] for r in DEFERRED_RULES])
def test_body_still_carries_todo_002_marker(
    rule_id, rule_name, check_class, method_name,
):
    """Contract item 3: the body emits an INFO line containing ``TODO-002``.

    The marker IS the deferral signal. When a retirement drops it (which is
    the explicit way to mark retirement), this assertion fails and the
    retirer is forced to update the contract registry above.
    """
    method = getattr(check_class, method_name)
    source = inspect.getsource(method)
    assert "TODO-002" in source, (
        f"{check_class.__name__}.{method_name} no longer contains the "
        f"'TODO-002' marker. If you are RETIRING this stub, also:\n"
        f"  - remove ({rule_id!r}, {rule_name!r}, ...) from DEFERRED_RULES "
        f"in this file,\n"
        f"  - flip the corresponding pinning tests in "
        f"test_training_check_retrofit.py / "
        f"test_checkpointing_check_phase2.py from 'asserts deferred' to "
        f"'asserts live',\n"
        f"  - keep the @rule decoration in place,\n"
        f"  - confirm the body now emits log_violation on detected "
        f"mismatches (not log.info).\n"
        f"See this test class's module docstring for the full checklist."
    )


@pytest.mark.parametrize("rule_id,rule_name,check_class,method_name",
                         DEFERRED_RULES,
                         ids=[r[0] for r in DEFERRED_RULES])
def test_body_does_not_log_violation_while_deferred(
    rule_id, rule_name, check_class, method_name,
):
    """Sanity: a deferred body must not call ``log_violation`` — that's the
    very thing it's deferring.

    Catches the half-retirement case where someone adds violation logic but
    forgets to drop the TODO-002 marker; the previous test would still pass
    but this one fails, surfacing the inconsistency.
    """
    method = getattr(check_class, method_name)
    source = inspect.getsource(method)
    assert "log_violation" not in source, (
        f"{check_class.__name__}.{method_name} contains 'log_violation' "
        f"but still carries the TODO-002 marker. Either the deferral is "
        f"over (drop the marker and update DEFERRED_RULES + the pinning "
        f"tests) or the violation call is wrong (remove it). See this "
        f"file's module docstring for the retirement checklist."
    )


def test_schema_anchors_present_for_rules_that_declare_them():
    """Contract item 4: every rule listed in ``EXPECTED_SCHEMA_ANCHORS``
    has its loc-string entries in ``SCHEMA_ERROR_RULE_MAP``.

    For 4.7.4 specifically, this guarantees that even with the runtime
    check deferred, a missing-or-mistyped ``simultaneous_write`` /
    ``simultaneous_read`` / ``multi_host`` field in the system YAML still
    surfaces as a ``[4.7.4 ...]`` ERROR. Retirement of the runtime check
    must NOT remove these entries — they cover declaration-side defects
    that the runtime check can't see.
    """
    schema_map = SystemYamlSchemaCheck.SCHEMA_ERROR_RULE_MAP
    for rule_id, expected_loc_strings in EXPECTED_SCHEMA_ANCHORS.items():
        for loc_str in expected_loc_strings:
            assert loc_str in schema_map, (
                f"SCHEMA_ERROR_RULE_MAP is missing the anchor entry "
                f"{loc_str!r} for rule {rule_id}. Without it, schema "
                f"errors on that field fall back to '2.1.7 "
                f"systemsDirectoryFiles' and the rule loses its only "
                f"current enforcement."
            )
            mapped = schema_map[loc_str]
            assert mapped[0] == rule_id, (
                f"SCHEMA_ERROR_RULE_MAP[{loc_str!r}] = {mapped}; expected "
                f"rule_id {rule_id!r}. The schema anchor must point at the "
                f"rule whose runtime body is deferred."
            )


def test_rules_without_schema_anchor_have_no_silent_coverage():
    """Forward-looking: rules in ``DEFERRED_RULES`` but NOT in
    ``EXPECTED_SCHEMA_ANCHORS`` (today: 3.3.5 and 3.3.7) genuinely have
    no Phase-2 enforcement.

    This assertion documents the gap explicitly: it succeeds today, but
    if someone adds a schema anchor for 3.3.5 / 3.3.7 (which would be a
    real improvement), they must add the loc strings to
    ``EXPECTED_SCHEMA_ANCHORS`` so this contract test enforces the
    anchor going forward. The failure message tells them exactly what to
    do.
    """
    schema_map = SystemYamlSchemaCheck.SCHEMA_ERROR_RULE_MAP
    actually_mapped: dict[str, set[str]] = {}
    for loc_str, (mapped_rule_id, _name) in schema_map.items():
        actually_mapped.setdefault(mapped_rule_id, set()).add(loc_str)

    for rule_id, _name, _cls, _method in DEFERRED_RULES:
        expected = EXPECTED_SCHEMA_ANCHORS.get(rule_id, set())
        actual = actually_mapped.get(rule_id, set())
        new_anchors = actual - expected
        assert not new_anchors, (
            f"Rule {rule_id} now has SCHEMA_ERROR_RULE_MAP entries "
            f"{sorted(new_anchors)} that aren't recorded in "
            f"EXPECTED_SCHEMA_ANCHORS. If this is intentional (defense in "
            f"depth for the deferred runtime check), add the loc strings "
            f"to EXPECTED_SCHEMA_ANCHORS[{rule_id!r}] in this file so the "
            f"anchor is locked in as part of the retirement contract."
        )
