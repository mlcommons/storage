"""``@rule`` edition gates: a check whose logic changed between rules
editions is bound per edition, so ``validate`` runs only the rules of the
edition a submission declares (design D-15 of
.planning/rules-editions-and-comparability-classes.md; §4 tier 3 of
.planning/results-archive-and-editions-proposal.md).

``@rule(rule_id, rule_name, since=None, until=None)`` gains an optional
half-open edition range ``[since, until)``. The default (no range) is every
edition, so the existing bindings are unchanged. Two methods may bind the
same rule id when their ranges do not overlap: the old logic ``until="4.0"``,
the new logic ``since="4.0"``.

Covered here:
- ``EditionGate``: numeric edition ordering (``3.0 < 3.10 < 4.0``),
  half-open membership, the open gate, rendering, rejected ids;
- the decorator: default gate, ``since`` / ``until`` attached, function
  returned unchanged, malformed or inverted ranges rejected at decoration;
- ``discover_rule_bindings`` lists every binding with its gate;
  ``discover_rules`` projects onto one edition (the current one by default;
  several methods may still implement facets of one rule id, as 4.7.1 does,
  and the last by name wins exactly as before); ``gated_overlaps`` reports
  two *gated* bindings of one rule id whose ranges overlap (the
  replace-a-rule mistake);
- ``BaseCheck.run_checks`` skips a registered check whose gate excludes
  ``config.edition`` (debug line, result unaffected) and runs it otherwise;
  no ``config`` or a non-string edition means the current edition;
- the shipped check classes have no conflicting bindings for any edition
  in the table;
- README and ManPage say only the checks bound to the declared edition
  run; Rules.md states the same gate declaratively (only the rules bound
  to that edition apply).
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import pytest

from mlpstorage_py.config import RULES_EDITION
from mlpstorage_py.editions import load_editions
from mlpstorage_py.submission_checker.checks.base import BaseCheck
from mlpstorage_py.submission_checker.configuration.configuration import Config
from mlpstorage_py.submission_checker.rule_registry import (
    ALL_EDITIONS,
    EditionGate,
    RuleBinding,
    discover_rule_bindings,
    discover_rules,
    edition_key,
    gated_overlaps,
    rule,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_PHRASE = "only the checks bound to that edition run"
# Rules.md is declarative (no run-time tool behaviour); it states the same
# gate as a predicate on the submission.
RULES_PHRASE = "only the rules bound to that edition apply"


# ---------------------------------------------------------------------------
# EditionGate
# ---------------------------------------------------------------------------

class TestEditionGate:
    def test_edition_key_orders_numerically(self):
        assert edition_key("3.0") < edition_key("3.10") < edition_key("4.0")
        assert edition_key("0.5") < edition_key("1.0") < edition_key("2.0") < edition_key("3.0")
        assert edition_key("3.0") == edition_key("3.0")

    @pytest.mark.parametrize("bad", ["", "v3.0", "3", "3.0.1", "three", "3.", ".0", "3.0 ", None, 3.0])
    def test_edition_key_rejects_non_edition_ids(self, bad):
        with pytest.raises(ValueError):
            edition_key(bad)

    def test_open_gate_applies_to_every_edition(self):
        assert ALL_EDITIONS == EditionGate(None, None)
        for eid in ("0.5", "1.0", "2.0", "3.0", "3.1", "4.0", "12.7"):
            assert ALL_EDITIONS.applies_to(eid)
        assert ALL_EDITIONS.is_open

    def test_half_open_range(self):
        gate = EditionGate("3.0", "4.0")
        assert not gate.applies_to("2.0")
        assert gate.applies_to("3.0")
        assert gate.applies_to("3.5")
        assert gate.applies_to("3.10")
        assert not gate.applies_to("4.0")
        assert not gate.applies_to("5.0")
        assert not gate.is_open

    def test_since_only_and_until_only(self):
        assert EditionGate("4.0", None).applies_to("4.0")
        assert EditionGate("4.0", None).applies_to("9.0")
        assert not EditionGate("4.0", None).applies_to("3.0")
        assert EditionGate(None, "4.0").applies_to("3.0")
        assert EditionGate(None, "4.0").applies_to("0.5")
        assert not EditionGate(None, "4.0").applies_to("4.0")

    def test_gate_rejects_bad_bounds(self):
        with pytest.raises(ValueError):
            EditionGate("v3.0", None)
        with pytest.raises(ValueError):
            EditionGate(None, "4")
        with pytest.raises(ValueError):
            EditionGate("4.0", "3.0")
        with pytest.raises(ValueError):
            EditionGate("3.0", "3.0")

    def test_gate_renders_for_reports(self):
        assert str(ALL_EDITIONS) == "all editions"
        assert str(EditionGate("3.0", None)) == "editions >= 3.0"
        assert str(EditionGate(None, "4.0")) == "editions < 4.0"
        assert str(EditionGate("3.0", "4.0")) == "editions >= 3.0 and < 4.0"

    def test_gate_is_hashable_and_frozen(self):
        assert hash(EditionGate("3.0", None)) == hash(EditionGate("3.0", None))
        with pytest.raises(Exception):
            EditionGate("3.0", None).since = "2.0"  # type: ignore[misc]

    def test_applies_to_rejects_bad_edition(self):
        with pytest.raises(ValueError):
            EditionGate("3.0", None).applies_to("v3.0")


# ---------------------------------------------------------------------------
# @rule decorator
# ---------------------------------------------------------------------------

class TestRuleDecoratorGate:
    def test_two_argument_form_is_ungated(self):
        @rule("2.1.2", "topLevelSubdirectories")
        def check(self):
            return True

        assert check.__rule_id__ == "2.1.2"
        assert check.__rule_name__ == "topLevelSubdirectories"
        assert check.__rule_gate__ is ALL_EDITIONS

    def test_since_and_until_attach_a_gate(self):
        @rule("4.2.1", "auThreshold", since="3.0", until="4.0")
        def check(self):
            return True

        assert check.__rule_gate__ == EditionGate("3.0", "4.0")

    def test_since_only(self):
        @rule("4.2.1", "auThreshold", since="4.0")
        def check(self):
            return True

        assert check.__rule_gate__ == EditionGate("4.0", None)
        assert check.__rule_gate__.applies_to("4.0")
        assert not check.__rule_gate__.applies_to("3.0")

    def test_until_only(self):
        @rule("4.2.1", "auThreshold", until="4.0")
        def check(self):
            return True

        assert check.__rule_gate__ == EditionGate(None, "4.0")

    def test_bounds_are_keyword_only(self):
        with pytest.raises(TypeError):
            rule("4.2.1", "auThreshold", "3.0")  # type: ignore[misc]

    @pytest.mark.parametrize("kwargs", [
        {"since": "v3.0"}, {"until": "4"}, {"since": 3.0}, {"until": ""},
        {"since": "4.0", "until": "3.0"}, {"since": "3.0", "until": "3.0"},
    ])
    def test_bad_bounds_rejected_at_decoration(self, kwargs):
        with pytest.raises(ValueError):
            rule("4.2.1", "auThreshold", **kwargs)

    def test_function_returned_unchanged(self):
        def original(self, a, b=2):
            return a + b

        sig = inspect.signature(original)
        decorated = rule("X", "Y", since="3.0")(original)
        assert decorated is original
        assert inspect.signature(decorated) == sig
        assert decorated(None, 1) == 3

    def test_gate_visible_through_bound_method(self):
        class C:
            @rule("1.1", "gated", until="4.0")
            def m(self):
                return "ran"

        bound = C().m
        assert bound.__rule_gate__ == EditionGate(None, "4.0")
        assert bound() == "ran"


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

class _Replaced:
    """One rule id, old logic until 4.0, new logic from 4.0; a neighbour ungated."""

    @rule("2.1.1", "neighbour")
    def neighbour_check(self):
        pass

    @rule("4.2.1", "auThreshold", until="4.0")
    def au_threshold_v3_check(self):
        pass

    @rule("4.2.1", "auThreshold", since="4.0")
    def au_threshold_v4_check(self):
        pass

    @rule("9.9.9", "futureOnly", since="4.0")
    def future_only_check(self):
        pass

    def plain(self):
        pass


class TestDiscovery:
    def test_bindings_list_every_binding_with_gate(self):
        bindings = discover_rule_bindings(_Replaced)
        assert all(isinstance(b, RuleBinding) for b in bindings)
        assert [(b.rule_id, b.method_name) for b in bindings] == [
            ("2.1.1", "neighbour_check"),
            ("4.2.1", "au_threshold_v3_check"),
            ("4.2.1", "au_threshold_v4_check"),
            ("9.9.9", "future_only_check"),
        ]
        by_method = {b.method_name: b for b in bindings}
        assert by_method["neighbour_check"].gate is ALL_EDITIONS
        assert by_method["neighbour_check"].rule_name == "neighbour"
        assert by_method["au_threshold_v3_check"].gate == EditionGate(None, "4.0")
        assert by_method["au_threshold_v4_check"].gate == EditionGate("4.0", None)

    def test_bindings_empty_without_rules(self):
        class Plain:
            def m(self):
                pass

        assert discover_rule_bindings(Plain) == []

    def test_discover_rules_defaults_to_current_edition(self):
        assert RULES_EDITION == "3.0"
        assert discover_rules(_Replaced) == discover_rules(_Replaced, edition=RULES_EDITION)
        assert discover_rules(_Replaced) == {
            "2.1.1": ("neighbour", "neighbour_check"),
            "4.2.1": ("auThreshold", "au_threshold_v3_check"),
        }

    def test_discover_rules_projects_onto_requested_edition(self):
        assert discover_rules(_Replaced, edition="4.0") == {
            "2.1.1": ("neighbour", "neighbour_check"),
            "4.2.1": ("auThreshold", "au_threshold_v4_check"),
            "9.9.9": ("futureOnly", "future_only_check"),
        }
        assert discover_rules(_Replaced, edition="2.0") == {
            "2.1.1": ("neighbour", "neighbour_check"),
            "4.2.1": ("auThreshold", "au_threshold_v3_check"),
        }

    def test_facets_of_one_rule_keep_last_by_name(self):
        """Two ungated methods may implement one rule id (4.7.1 today);
        discover_rules keeps one entry per id, the last by method name,
        exactly as before the gate existed."""
        class Facets:
            @rule("4.7.1", "checkpointCacheFlushValidation")
            def cache_flush_validation(self):
                pass

            @rule("4.7.1", "checkpointCacheFlushValidation")
            def checkpoint_invocation_structure(self):
                pass

        assert discover_rules(Facets) == {
            "4.7.1": ("checkpointCacheFlushValidation", "checkpoint_invocation_structure")}
        assert len(discover_rule_bindings(Facets)) == 2
        assert gated_overlaps(Facets) == []

    def test_gated_overlaps_reports_two_gated_bindings_that_meet(self):
        class Overlap:
            @rule("2.1.1", "dup", until="4.0")
            def a(self):
                pass

            @rule("2.1.1", "dup", since="3.0")
            def b(self):
                pass

            @rule("2.1.1", "dup")            # an ungated facet never conflicts
            def c(self):
                pass

        overlaps = gated_overlaps(Overlap)
        assert len(overlaps) == 1
        rule_id, first, second = overlaps[0]
        assert rule_id == "2.1.1"
        assert {first.method_name, second.method_name} == {"a", "b"}

    def test_gated_overlaps_empty_for_disjoint_ranges(self):
        assert gated_overlaps(_Replaced) == []

        class Touching:
            @rule("1.1", "x", since="3.0", until="4.0")
            def a(self):
                pass

            @rule("1.1", "x", since="4.0", until="5.0")
            def b(self):
                pass

            @rule("1.1", "x", since="5.0")
            def c(self):
                pass

        assert gated_overlaps(Touching) == []

    def test_gated_overlaps_ignores_different_rule_ids(self):
        class Different:
            @rule("1.1", "x", since="3.0")
            def a(self):
                pass

            @rule("1.2", "y", since="3.0")
            def b(self):
                pass

        assert gated_overlaps(Different) == []

    def test_discover_rules_rejects_bad_edition(self):
        with pytest.raises(ValueError):
            discover_rules(_Replaced, edition="v3.0")


# ---------------------------------------------------------------------------
# BaseCheck runtime gate
# ---------------------------------------------------------------------------

class _FakeConfig:
    def __init__(self, edition):
        self.edition = edition


class _GatedCheck(BaseCheck):
    def __init__(self, log, config=None):
        super().__init__(log, "/tree/closed/Acme")
        if config is not None:
            self.config = config
        self.name = "gated checks"
        self.ran = []
        self.checks = [self.old_check, self.new_check, self.always_check]

    @rule("4.2.1", "auThreshold", until="4.0")
    def old_check(self):
        self.ran.append("old")
        return True

    @rule("4.2.1", "auThreshold", since="4.0")
    def new_check(self):
        self.ran.append("new")
        return False

    @rule("2.1.1", "always")
    def always_check(self):
        self.ran.append("always")
        return True


class TestBaseCheckGate:
    def test_runs_only_the_bindings_of_the_configured_edition(self, caplog):
        chk = _GatedCheck(logging.getLogger("t"), _FakeConfig("3.0"))
        with caplog.at_level(logging.DEBUG, logger="t"):
            assert chk.run_checks() is True
        assert chk.ran == ["old", "always"]
        skipped = [r for r in caplog.records if "new_check" in r.getMessage()]
        assert len(skipped) == 1
        assert skipped[0].levelno == logging.DEBUG
        assert "4.2.1" in skipped[0].getMessage() and "3.0" in skipped[0].getMessage()

    def test_future_edition_runs_the_new_binding(self):
        chk = _GatedCheck(logging.getLogger("t"), _FakeConfig("4.0"))
        assert chk.run_checks() is False   # new_check returns False and must count
        assert chk.ran == ["new", "always"]

    def test_skipped_check_never_counts_as_failure(self):
        chk = _GatedCheck(logging.getLogger("t"), _FakeConfig("2.0"))
        assert chk.run_checks() is True
        assert chk.ran == ["old", "always"]

    def test_without_config_the_current_edition_applies(self):
        chk = _GatedCheck(logging.getLogger("t"))
        assert not hasattr(chk, "config")
        chk.run_checks()
        assert chk.ran == ["old", "always"]

    def test_non_string_edition_means_current_edition(self):
        from unittest.mock import MagicMock
        chk = _GatedCheck(logging.getLogger("t"), MagicMock())
        chk.run_checks()
        assert chk.ran == ["old", "always"]

    def test_real_config_edition_drives_the_gate(self):
        chk = _GatedCheck(logging.getLogger("t"), Config(edition="3.0"))
        chk.run_checks()
        assert chk.ran == ["old", "always"]

    def test_rule_applies_helper(self):
        chk = _GatedCheck(logging.getLogger("t"), _FakeConfig("3.0"))
        assert chk.edition == "3.0"
        assert chk.rule_applies(chk.old_check)
        assert not chk.rule_applies(chk.new_check)
        assert chk.rule_applies(chk.always_check)
        assert chk.rule_applies(lambda: True)      # undecorated callables always run

    def test_execute_override_still_receives_applicable_checks(self):
        seen = []

        class Custom(_GatedCheck):
            def execute(self, check):
                seen.append(check.__name__)
                return super().execute(check)

        chk = Custom(logging.getLogger("t"), _FakeConfig("3.0"))
        chk.run_checks()
        assert seen == ["old_check", "always_check"]

    def test_gated_out_check_is_not_exception_wrapped(self, caplog):
        """A skipped check produces no ERROR line of any kind."""
        chk = _GatedCheck(logging.getLogger("t"), _FakeConfig("3.0"))
        with caplog.at_level(logging.DEBUG, logger="t"):
            chk.run_checks()
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# shipped check classes
# ---------------------------------------------------------------------------

def _shipped_check_classes():
    from mlpstorage_py.submission_checker.checks.checkpointing_checks import CheckpointingCheck
    from mlpstorage_py.submission_checker.checks.directory_checks import DirectoryCheck
    from mlpstorage_py.submission_checker.checks.edition_checks import EditionCheck
    from mlpstorage_py.submission_checker.checks.kvcache_checks import KVCacheCheck
    from mlpstorage_py.submission_checker.checks.pool_structure_checks import PoolStructureCheck
    from mlpstorage_py.submission_checker.checks.provenance_checks import ProvenanceCheck
    from mlpstorage_py.submission_checker.checks.submission_structure_checks import SubmissionStructureCheck
    from mlpstorage_py.submission_checker.checks.system_yaml_schema_checks import SystemYamlSchemaCheck
    from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
    from mlpstorage_py.submission_checker.checks.vdb_checks import VdbCheck
    return [SubmissionStructureCheck, SystemYamlSchemaCheck, PoolStructureCheck, ProvenanceCheck,
            EditionCheck, DirectoryCheck, TrainingCheck, CheckpointingCheck, VdbCheck, KVCacheCheck]


class TestShippedClasses:
    @pytest.mark.parametrize("cls", _shipped_check_classes(), ids=lambda c: c.__name__)
    def test_no_overlapping_gated_bindings(self, cls):
        assert gated_overlaps(cls) == []

    @pytest.mark.parametrize("cls", _shipped_check_classes(), ids=lambda c: c.__name__)
    def test_discovery_works_for_every_table_edition(self, cls):
        for eid in load_editions().editions:
            assert isinstance(discover_rules(cls, edition=eid), dict)

    @pytest.mark.parametrize("cls", _shipped_check_classes(), ids=lambda c: c.__name__)
    def test_current_edition_projection_matches_the_pre_gate_view(self, cls):
        """No shipped binding is gated yet, so the current-edition view is the
        whole binding set (one entry per rule id, last by name)."""
        expected = {}
        for b in discover_rule_bindings(cls):
            expected[b.rule_id] = (b.rule_name, b.method_name)
        assert discover_rules(cls) == expected

    @pytest.mark.parametrize("cls", _shipped_check_classes(), ids=lambda c: c.__name__)
    def test_every_binding_reaches_the_current_edition_or_a_later_one(self, cls):
        """A binding closed before the current edition is dead code: it can
        never run (no earlier edition is checkable by this tool)."""
        for b in discover_rule_bindings(cls):
            assert b.gate.until is None or edition_key(b.gate.until) > edition_key(RULES_EDITION), \
                f"{cls.__name__}.{b.method_name} ({b.rule_id}) is gated {b.gate} and can never run"

    def test_coverage_tool_maps_the_current_edition_view(self):
        """The Rules.md coverage tool reports the current edition's bindings:
        exactly the union of discover_rules() over its seven check classes."""
        from mlpstorage_py.submission_checker.tools.rules_coverage import _collect_check_method_coverage
        coverage = _collect_check_method_coverage()
        expected = {}
        for cls in _shipped_check_classes():
            if cls.__name__ in ("PoolStructureCheck", "ProvenanceCheck", "EditionCheck"):
                continue   # not in the coverage tool's class list
            for rule_id, (_name, method) in discover_rules(cls).items():
                expected[rule_id] = f"{cls.__name__}.{method}"
        assert coverage == expected


# ---------------------------------------------------------------------------
# docs
# ---------------------------------------------------------------------------

class TestDocs:
    @pytest.mark.parametrize("doc", ["README.md", "ManPage.md"])
    def test_docs_say_only_the_declared_editions_checks_run(self, doc):
        text = (PROJECT_ROOT / doc).read_text(encoding="utf-8")
        assert DOC_PHRASE in text, f"{doc} does not say '{DOC_PHRASE}'"

    def test_rules_md_states_the_gate_declaratively(self):
        text = (PROJECT_ROOT / "Rules.md").read_text(encoding="utf-8")
        assert RULES_PHRASE in text, f"Rules.md does not say '{RULES_PHRASE}'"
        assert DOC_PHRASE not in text, "Rules.md describes tool behaviour"

    def test_registry_docstring_documents_the_gate(self):
        import mlpstorage_py.submission_checker.rule_registry as reg
        assert "since" in (reg.rule.__doc__ or "") and "until" in (reg.rule.__doc__ or "")
        assert "half-open" in (reg.__doc__ or "").lower() or "half-open" in (reg.rule.__doc__ or "").lower()
