#!/usr/bin/env python3
"""Tests for mlpstorage_py.submission_checker.tools.rules_coverage.

Covers Plan 03-04 success criteria:

* QUAL-03 first clause: ``reconcile()`` returns ``unmapped == set()`` against
  the current retrofitted code + registries (in-process gatekeeper).
* QUAL-03 second clause: injecting an unmapped ID via a fake Rules.md
  produces ``"2.1.99"`` in the unmapped set AND surfaces it via the CLI's
  exit-1 path (subprocess regression).
* Gemini-suggested upgrade #1 (drift warning): a stale
  ``OUT_OF_SCOPE_RULES`` entry fires a ``log.warning`` AND keeps exit code
  unchanged; a stale ``STUB_COVERAGE`` entry fires a warning that names the
  stub class.

The two drift-warning tests use ``monkeypatch.setattr`` to mutate
``coverage_mapping``'s top-level constants. ``reconcile()``'s helpers
re-import the module at call time, so the patches are observed.

Run with:
    pytest mlpstorage_py/tests/test_rules_coverage.py -v
"""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from mlpstorage_py.submission_checker.tools.rules_coverage import reconcile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_MD_PATH = PROJECT_ROOT / "Rules.md"


# ---------------------------------------------------------------------------
# In-process reconciliation tests (the gatekeeper signal for QUAL-03).
# Wall-time budget per case is < 100 ms (CONTEXT.md line 162).
# ---------------------------------------------------------------------------


class TestRulesCoverageReconciliation:
    """In-process tests against the live reconcile() helper."""

    def test_every_rules_md_id_is_mapped(self):
        """Gatekeeper: every live Rules.md ID has a disposition.

        Fails loudly with the unmapped-ID list when a retrofit regresses
        (e.g., a contributor removes an ``@rule`` decorator without
        registering a stub or out-of-scope entry).
        """
        result = reconcile()
        assert result["unmapped"] == set(), (
            "Unmapped Rules.md IDs (every ID needs an @rule binding, stub "
            "entry, or OUT_OF_SCOPE_RULES entry): "
            "{}".format(sorted(result["unmapped"]))
        )

    def test_reconcile_returns_rows_for_every_rules_md_id(self):
        """Row count covers every Rules.md §2/§3/§4 ID (>= 50; current 57)."""
        result = reconcile()
        assert len(result["rows"]) >= 50, (
            "Expected at least 50 rows, got {}".format(len(result["rows"]))
        )

    def test_struct_2_1_2_is_check_method_disposition(self):
        """2.1.2 topLevelSubdirectories: check-method disposition."""
        result = reconcile()
        by_id = {row[0]: row for row in result["rows"]}
        assert "2.1.2" in by_id, "2.1.2 missing from rows"
        rule_id, rule_name, disposition, source = by_id["2.1.2"]
        assert disposition == "check method", (
            "Expected 'check method' disposition for 2.1.2, got {!r}".format(
                disposition
            )
        )
        assert "SubmissionStructureCheck" in source, (
            "Expected SubmissionStructureCheck as source for 2.1.2, "
            "got {!r}".format(source)
        )

    def test_4_7_3_is_check_method_disposition_or_schema_check(self):
        """4.7.3 wins as check-method per priority order D-A4.

        4.7.3 is BOTH @rule-decorated on CheckpointingCheck AND a value in
        SCHEMA_ERROR_RULE_MAP. The priority order (check-method >
        schema-check) ensures the check-method wins; this test pins that
        invariant so a future contributor cannot accidentally invert the
        priority.
        """
        result = reconcile()
        by_id = {row[0]: row for row in result["rows"]}
        assert "4.7.3" in by_id, "4.7.3 missing from rows"
        _rule_id, _rule_name, disposition, _source = by_id["4.7.3"]
        assert disposition == "check method", (
            "Expected check-method to win priority for 4.7.3, got {!r}".format(
                disposition
            )
        )

    def test_inject_unmapped_id_returns_in_unmapped_set(self, tmp_path):
        """Injecting an extra Rules.md line produces the ID in unmapped.

        Builds a fake Rules.md by copying the real one and appending one
        line that matches the locked regex. Asserts the new ID surfaces
        in ``result["unmapped"]`` (the gatekeeper signal that drives the
        CLI's exit-1 path).
        """
        fake_md = tmp_path / "fake.md"
        original = RULES_MD_PATH.read_text(encoding="utf-8")
        # Locked regex: ^([234]\.\d+\.\d+)\.\s+\*\*([a-zA-Z][a-zA-Z0-9]+)\*\*
        fake_md.write_text(
            original + "\n2.1.99. **fakeRule** -- placeholder for testing\n",
            encoding="utf-8",
        )
        result = reconcile(rules_md_path=str(fake_md))
        assert "2.1.99" in result["unmapped"], (
            "Expected 2.1.99 in unmapped set, got {}".format(
                sorted(result["unmapped"])
            )
        )

    def test_baseline_no_stale_entries(self):
        """At Phase 3 land time both registries are empty → no stale entries.

        Locks the Phase-3 invariant: a future contributor who adds an entry
        to either registry that doesn't appear in Rules.md will fail this
        test (and the drift warning will fire in CI).
        """
        result = reconcile()
        assert result["stale_oos"] == set(), (
            "Expected no stale OUT_OF_SCOPE_RULES entries at Phase 3 land "
            "time, got {}".format(sorted(result["stale_oos"]))
        )
        assert result["stale_stubs"] == {}, (
            "Expected no stale STUB_COVERAGE entries at Phase 3 land time, "
            "got {}".format(result["stale_stubs"])
        )

    def test_drift_stale_oos_emits_warning_and_keeps_exit_0(
        self, monkeypatch, caplog
    ):
        """Gemini upgrade #1: stale OUT_OF_SCOPE_RULES entry fires a warning.

        Exit code (driven by ``unmapped``) is unchanged because every live
        Rules.md ID is still mapped; the drift signal is informational.
        """
        monkeypatch.setattr(
            "mlpstorage_py.submission_checker.coverage_mapping.OUT_OF_SCOPE_RULES",
            {"9.9.9": "deleted from spec"},
        )
        with caplog.at_level(logging.WARNING, logger="rules_coverage"):
            result = reconcile()

        assert "9.9.9" in result["stale_oos"], (
            "Expected 9.9.9 in stale_oos, got {}".format(sorted(result["stale_oos"]))
        )
        # Live IDs unaffected — exit code path stays 0.
        assert result["unmapped"] == set(), (
            "Drift warnings must not affect unmapped set; got {}".format(
                sorted(result["unmapped"])
            )
        )
        # Locked-wording warning emitted via log.warning (captured via caplog).
        matched = [
            rec for rec in caplog.records
            if rec.levelname == "WARNING"
            and "OUT_OF_SCOPE_RULES contains stale rule_id 9.9.9" in rec.getMessage()
        ]
        assert matched, (
            "Expected one WARNING-level log record naming the stale OUT_OF_SCOPE "
            "rule_id 9.9.9; got records: {}".format(
                [(r.levelname, r.getMessage()) for r in caplog.records]
            )
        )

    def test_drift_stale_stub_emits_warning_naming_the_stub(
        self, monkeypatch, caplog
    ):
        """Gemini upgrade #1: stale STUB_COVERAGE entry fires a warning naming
        the stub class so contributors know where to delete from.
        """
        monkeypatch.setattr(
            "mlpstorage_py.submission_checker.coverage_mapping.STUB_COVERAGE",
            {"VdbCheck": ["9.9.8"], "KVCacheCheck": []},
        )
        with caplog.at_level(logging.WARNING, logger="rules_coverage"):
            result = reconcile()

        assert result["stale_stubs"] == {"VdbCheck": ["9.9.8"]}, (
            "Expected stale_stubs == {{'VdbCheck': ['9.9.8']}}, got {}".format(
                result["stale_stubs"]
            )
        )
        assert result["unmapped"] == set(), (
            "Drift warnings must not affect unmapped set; got {}".format(
                sorted(result["unmapped"])
            )
        )
        matched = [
            rec for rec in caplog.records
            if rec.levelname == "WARNING"
            and "STUB_COVERAGE contains stale rule_id 9.9.8 on stub VdbCheck"
            in rec.getMessage()
        ]
        assert matched, (
            "Expected one WARNING-level log record naming the stale "
            "STUB_COVERAGE rule_id 9.9.8 on stub VdbCheck; got records: "
            "{}".format([(r.levelname, r.getMessage()) for r in caplog.records])
        )


# ---------------------------------------------------------------------------
# CLI subprocess tests — verify the exit-code contract end-to-end (D-A5).
# ---------------------------------------------------------------------------


class TestRulesCoverageCli:
    """End-to-end CLI tests via subprocess (mirrors test_code_checksum.py)."""

    def test_cli_exits_0_with_table(self):
        """CLI on the live Rules.md prints a Markdown table and exits 0."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlpstorage_py.submission_checker.tools.rules_coverage",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            "Expected exit 0; got {}. stderr: {}".format(
                result.returncode, result.stderr
            )
        )
        assert "| Rule ID" in result.stdout, (
            "Expected Markdown header '| Rule ID' in stdout; got: "
            "{!r}".format(result.stdout[:300])
        )

    def test_cli_exits_1_when_rules_md_has_unmapped_id(self, tmp_path):
        """CLI on a Rules.md with an injected unmapped ID exits 1.

        The fake Rules.md copies the real one and appends a fabricated
        ``2.1.99. **fakeRule**`` line. The CLI must exit 1 AND name the
        unmapped ID in its diagnostic output (stdout or stderr).
        """
        fake_md = tmp_path / "fake.md"
        original = RULES_MD_PATH.read_text(encoding="utf-8")
        fake_md.write_text(
            original + "\n2.1.99. **fakeRule** -- placeholder for testing\n",
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlpstorage_py.submission_checker.tools.rules_coverage",
                "--rules-md",
                str(fake_md),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, (
            "Expected exit 1 when an unmapped ID is injected; got {}. "
            "stdout head: {!r} stderr head: {!r}".format(
                result.returncode, result.stdout[:300], result.stderr[:300]
            )
        )
        combined = result.stdout + result.stderr
        assert "2.1.99" in combined, (
            "Expected '2.1.99' in CLI output; got stdout: {!r} stderr: {!r}".format(
                result.stdout[:300], result.stderr[:300]
            )
        )
        assert "no mapping found" in combined, (
            "Expected the locked 'no mapping found' wording in CLI output"
        )
