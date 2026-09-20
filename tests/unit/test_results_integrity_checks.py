"""
LEAF-01 / RPT-01 — two validator rules that catch what reportgen used to
lose silently (mlcommons/storage#835 item 3, #836 item 4).

LEAF-01 ``runLeafMetadata``: a ``kv_cache`` or ``vector_database`` ``run``
leaf must carry exactly one ``<type>_<timestamp>_metadata.json``. Those two
benchmark types have no other signal that identifies the run — DLIO training
and checkpointing leaves identify themselves through their Hydra configs —
so without the file reportgen drops the run and the workload publishes with
empty metric columns (#835). The rule surfaces that at validation time, and
names the consequence and the remedy.

RPT-01 ``rollupTableAgreement``: every minted ``Public ID`` in an
organization's rollup table (``<mode>/<org>/results/results.json``) must
appear in a workload-level results table under that organization's
``results/<system>/`` tree, and every minted ID in a workload-level table
must appear in the rollup. #836's original shape was a populated rollup row
above an empty workload table; the two paths that disagreed were fixed in
reportgen, and this rule makes any recurrence fail validation. Rows with an
empty ``Public ID`` (datagen / datasize command tables) are identity-only
by design and are ignored.
"""

from __future__ import annotations

import json
import os

import pytest

from mlpstorage_py.submission_checker.checks.results_integrity_checks import (
    ResultsIntegrityCheck,
)
from mlpstorage_py.submission_checker.configuration.configuration import Config


class _Log:
    def __init__(self):
        self.errors, self.warnings, self.infos = [], [], []

    def error(self, msg, *args, **kw):
        self.errors.append(msg % args if args else msg)

    def warning(self, msg, *args, **kw):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args, **kw):
        self.infos.append(msg % args if args else msg)

    def debug(self, *a, **kw):
        pass


TS = "20260708_234249"


def _mk(root, rel, files: dict | None = None):
    """Create ``root/rel`` and write each ``name -> content`` (dict → JSON)."""
    path = os.path.join(root, rel)
    os.makedirs(path, exist_ok=True)
    for name, content in (files or {}).items():
        with open(os.path.join(path, name), "w") as fh:
            if isinstance(content, (dict, list)):
                json.dump(content, fh)
            else:
                fh.write(content)
    return path


def _check(root):
    log = _Log()
    config = Config(submitters=["Acme"], skip_output_file=True)
    check = ResultsIntegrityCheck(log, config, str(root))
    ok = check()
    return ok, log


def _rule_lines(log, rule_id):
    return [e for e in log.errors if e.startswith(f"[{rule_id} ")]


# ---------------------------------------------------------------------------
# LEAF-01
# ---------------------------------------------------------------------------

class TestRunLeafMetadata:
    SYS = "closed/Acme/results/sys-v1"

    def test_kv_cache_run_leaf_without_metadata_fails_and_names_consequence(self, tmp_path):
        leaf = _mk(tmp_path, f"{self.SYS}/kv_cache/llama3.1-8b/run/{TS}",
                   {"summary.json": {"schema_version": 1, "options": {}}})
        ok, log = _check(tmp_path)
        assert ok is False
        lines = _rule_lines(log, "LEAF-01")
        assert len(lines) == 1, log.errors
        line = lines[0]
        assert leaf in line
        assert "BLANK" in line and "kv_cache_" in line and "summary.json" in line

    def test_vector_database_run_leaf_without_metadata_fails(self, tmp_path):
        _mk(tmp_path, f"{self.SYS}/vector_database/milvus/DISKANN/run/{TS}",
            {"summary.json": {"throughput_qps": 1}})
        ok, log = _check(tmp_path)
        assert ok is False
        assert len(_rule_lines(log, "LEAF-01")) == 1, log.errors

    def test_leaf_with_neither_summary_nor_metadata_fails_as_incomplete(self, tmp_path):
        _mk(tmp_path, f"{self.SYS}/kv_cache/llama3.1-8b/run/{TS}")
        ok, log = _check(tmp_path)
        assert ok is False
        lines = _rule_lines(log, "LEAF-01")
        assert len(lines) == 1 and "no summary.json" in lines[0], log.errors

    def test_two_metadata_files_fail(self, tmp_path):
        _mk(tmp_path, f"{self.SYS}/kv_cache/llama3.1-8b/run/{TS}", {
            "summary.json": {},
            f"kv_cache_{TS}_metadata.json": {"verification": "CLOSED"},
            "kv_cache_20260708_000000_metadata.json": {"verification": "CLOSED"},
        })
        ok, log = _check(tmp_path)
        assert ok is False
        lines = _rule_lines(log, "LEAF-01")
        assert len(lines) == 1 and "2 " in lines[0], log.errors

    def test_complete_leaf_passes(self, tmp_path):
        _mk(tmp_path, f"{self.SYS}/kv_cache/llama3.1-8b/run/{TS}", {
            "summary.json": {},
            f"kv_cache_{TS}_metadata.json": {"verification": "CLOSED"},
        })
        _mk(tmp_path, f"{self.SYS}/vector_database/milvus/DISKANN/run/{TS}", {
            "summary.json": {},
            f"vector_database_{TS}_metadata.json": {"verification": "CLOSED"},
        })
        ok, log = _check(tmp_path)
        assert ok is True, log.errors
        assert not _rule_lines(log, "LEAF-01")

    def test_datagen_leaf_and_dlio_leaves_are_out_of_scope(self, tmp_path):
        """kv_cache datagen and every training / checkpointing leaf identify
        themselves without the file (or are owned by other rules)."""
        _mk(tmp_path, f"{self.SYS}/kv_cache/llama3.1-8b/datagen/{TS}")
        _mk(tmp_path, f"{self.SYS}/training/unet3d/run/{TS}", {"summary.json": {}})
        _mk(tmp_path, f"{self.SYS}/checkpointing/llama3-8b/{TS}", {"summary.json": {}})
        ok, log = _check(tmp_path)
        assert not _rule_lines(log, "LEAF-01"), log.errors


# ---------------------------------------------------------------------------
# RPT-01
# ---------------------------------------------------------------------------

def _row(pid, **extra):
    r = {"Public ID": pid, "Organization": "Acme", "Division": "CLOSED",
         "Benchmark Type": "kv_cache", "Model": ""}
    r.update(extra)
    return r


class TestRollupTableAgreement:
    ORG = "closed/Acme/results"

    def test_rollup_id_without_a_workload_row_fails(self, tmp_path):
        """#836's shape: the org rollup carries a row whose workload table
        does not."""
        _mk(tmp_path, self.ORG, {"results.json": [_row("v3.0-0001"), _row("v3.0-0002")]})
        _mk(tmp_path, f"{self.ORG}/sys-v1/kv_cache/llama3.1-8b",
            {"results.json": [_row("v3.0-0001")]})
        _mk(tmp_path, f"{self.ORG}/sys-v1/kv_cache/llama3.1-8b/run/{TS}", {
            "summary.json": {}, f"kv_cache_{TS}_metadata.json": {}})
        ok, log = _check(tmp_path)
        assert ok is False
        lines = _rule_lines(log, "RPT-01")
        assert len(lines) == 1, log.errors
        assert "v3.0-0002" in lines[0] and "v3.0-0001" not in lines[0]
        assert os.path.join(self.ORG, "results.json") in lines[0]

    def test_workload_row_missing_from_rollup_fails(self, tmp_path):
        _mk(tmp_path, self.ORG, {"results.json": [_row("v3.0-0001")]})
        _mk(tmp_path, f"{self.ORG}/sys-v1/kv_cache/llama3.1-8b",
            {"results.json": [_row("v3.0-0001"), _row("v3.0-0009")]})
        ok, log = _check(tmp_path)
        assert ok is False
        lines = _rule_lines(log, "RPT-01")
        assert len(lines) == 1 and "v3.0-0009" in lines[0], log.errors

    def test_blank_id_rows_are_ignored(self, tmp_path):
        """datagen / datasize command tables hold identity-only rows with no
        Public ID; they never reach the rollup by design."""
        _mk(tmp_path, self.ORG, {"results.json": [_row("v3.0-0001")]})
        _mk(tmp_path, f"{self.ORG}/sys-v1/training/unet3d/run",
            {"results.json": [_row("v3.0-0001", **{"Benchmark Type": "training"})]})
        _mk(tmp_path, f"{self.ORG}/sys-v1/training/unet3d/datagen",
            {"results.json": [_row("", **{"Benchmark Type": "training"})]})
        _mk(tmp_path, f"{self.ORG}/sys-v1/training/unet3d/datasize",
            {"results.json": [_row("", **{"Benchmark Type": "training"})]})
        ok, log = _check(tmp_path)
        assert ok is True, log.errors

    def test_agreeing_tables_pass_and_org_without_rollup_is_silent(self, tmp_path):
        _mk(tmp_path, self.ORG, {"results.json": [_row("v3.0-0001"), _row("v3.0-0002")]})
        _mk(tmp_path, f"{self.ORG}/sys-v1/kv_cache/llama3.1-8b",
            {"results.json": [_row("v3.0-0001")]})
        _mk(tmp_path, f"{self.ORG}/sys-v2/vector_database/milvus/DISKANN",
            {"results.json": [_row("v3.0-0002", **{"Benchmark Type": "vector_database"})]})
        # A second org that never ran reportgen: nothing to compare, no finding.
        _mk(tmp_path, f"closed/Beta/results/sys-b/kv_cache/llama3.1-8b/run/{TS}", {
            "summary.json": {}, f"kv_cache_{TS}_metadata.json": {}})
        ok, log = _check(tmp_path)
        assert ok is True, log.errors
        assert not _rule_lines(log, "RPT-01")

    def test_unreadable_table_fails(self, tmp_path):
        _mk(tmp_path, self.ORG, {"results.json": "{not json"})
        ok, log = _check(tmp_path)
        assert ok is False
        assert len(_rule_lines(log, "RPT-01")) == 1, log.errors
