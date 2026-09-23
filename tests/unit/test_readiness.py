"""Submission readiness: results, their runs and their paperwork.

``mlpstorage status`` / ``submit`` design (2026-09-23, PR 1 of 4): the
vocabulary Rules.md §1.3 pins (*run* / *result* / *submission*), the per-result
run count in the rules editions table (``checker.runs_per_result``), and the
shared readiness evaluator ``mlpstorage_py/readiness.py`` that the two verbs
and the post-run recap will print from.

Covered here:
- editions.yaml 3.0 carries ``runs_per_result``; the parser validates it
  (keys = the edition's workload families, positive integers); ``Config``
  reads it; the validator's 2.1.17 / 5.3.1 counts come from the table and
  the retired constant is gone; the docs name the block's new content;
- Rules.md 1.3 and its RulesCommentary entry exist and say what they must;
- finding parsing: the validator's locked ``[<id> <name>] <path>: <msg>``
  line into a ``Finding``;
- classification against scripted findings: per-run status (ok / failed /
  running / invalid / extra), the counted set, RUNS n/m per family, the SUBMIT
  token precedence (short > invalid > paperwork > ready, ``-`` for whatif),
  paperwork attribution (system YAML fields, PDF, code images), dropped
  rollup / count / whatif findings, tree-level problems, JSON shape;
- the real submission checker on the definition-of-done fixture tree:
  captured findings, nothing leaks to stdout / stderr, logger state restored.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
import os
import re
from pathlib import Path

import pytest
import yaml

from mlpstorage_py.config import RULES_EDITION
from mlpstorage_py.results_dir import write_sentinel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDITIONS_YAML = PROJECT_ROOT / "mlpstorage_py" / "rules" / "editions.yaml"

V30_RUNS_PER_RESULT = {"training": 6, "checkpointing": 2, "vector_database": 5, "kv_cache": 1}

ORG = "Acme"
SYS = "sys-1"
HASH_A = "a" * 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_table(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load(EDITIONS_YAML.read_text())
    mutate(data)
    p = tmp_path / "editions.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def _table_with(tmp_path: Path, mutate):
    from mlpstorage_py.editions import load_editions
    return load_editions(_write_table(tmp_path, mutate))


def _leaf(rd, rel, *, exit_status=0, metadata=True, summary=True, verification="CLOSED",
          accelerator="b200", writes=None, reads=None, pointer=HASH_A, benchmark_type=None):
    """One run leaf with the tool's metadata shape (enough for the ledger and
    the evaluator; the validator is scripted in these tests)."""
    leaf = os.path.join(rd, *rel.split("/"))
    os.makedirs(leaf, exist_ok=True)
    ts = rel.rsplit("/", 1)[1]
    family = rel.split("/")[4]
    if metadata:
        body = {"run_datetime": ts, "command": "run", "verification": verification,
                "benchmark_type": benchmark_type or family, "args": {}}
        if exit_status is not None:
            body["exit_status"] = exit_status
        if accelerator:
            body["accelerator"] = accelerator
            body["args"]["accelerator_type"] = accelerator
        if writes is not None or reads is not None:
            body["parameters"] = {"checkpoint": {"num_checkpoints_write": writes or 0,
                                                 "num_checkpoints_read": reads or 0}}
            body["args"]["num_checkpoints_write"] = writes or 0
            body["args"]["num_checkpoints_read"] = reads or 0
        with open(os.path.join(leaf, f"{family}_{ts}_metadata.json"), "w") as fh:
            json.dump(body, fh)
    if summary:
        with open(os.path.join(leaf, "summary.json"), "w") as fh:
            json.dump({"metric": {}}, fh)
    if pointer:
        with open(os.path.join(leaf, ".mlps-code-image"), "w") as fh:
            fh.write(f"md5-tree-v2:{pointer}")
    return leaf


def _system(rd, division, name, *, yaml_ok=True, pdf=True):
    systems = os.path.join(rd, division, ORG, "systems")
    os.makedirs(systems, exist_ok=True)
    if yaml_ok:
        with open(os.path.join(systems, f"{name}.yaml"), "w") as fh:
            fh.write("system_under_test: {}\n")
    if pdf:
        with open(os.path.join(systems, f"{name}.pdf"), "wb") as fh:
            fh.write(b"%PDF-1.4\n")


def _ts(i: int, day: int = 1) -> str:
    return f"202609{day:02d}_{10 + i:02d}0000"


def _training(rd, model, n_ok, *, division="closed", system=SYS, accelerator="b200",
              start=0, **kw):
    leaves = []
    for i in range(n_ok):
        rel = f"{division}/{ORG}/results/{system}/training/{model}/run/{_ts(start + i)}"
        leaves.append(_leaf(rd, rel, accelerator=accelerator, **kw))
    return leaves


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    cfg = tmp_path / "xdg"
    cfg.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg))
    monkeypatch.delenv("MLPSTORAGE_RESULTS_DIR", raising=False)
    return cfg


@pytest.fixture
def rd(tmp_path, xdg):
    """An initialized, empty results-dir."""
    path = str(tmp_path / "results")
    os.makedirs(path)
    write_sentinel(path, ORG)
    return path


@pytest.fixture
def scripted(monkeypatch):
    """Replace the validator capture with a scripted list of findings.

    Returns a function that sets the script; each entry is either a Finding
    or a ``(level, rule_id, rule_name, path, message)`` tuple.
    """
    import mlpstorage_py.readiness as readiness

    script: list = []

    def fake_collect(results_dir):
        out = []
        for item in script:
            if isinstance(item, readiness.Finding):
                out.append(item)
            else:
                level, rule_id, rule_name, path, message = item
                out.append(readiness.Finding(level=level, rule_id=rule_id, rule_name=rule_name,
                                             path=path, message=message))
        return out

    monkeypatch.setattr(readiness, "collect_findings", fake_collect)

    def set_script(items):
        script[:] = list(items)
    return set_script


def _evaluate(rd):
    from mlpstorage_py.readiness import evaluate
    return evaluate(rd)


def _row(sub, *, benchmark, model, division="closed", system=SYS):
    rows = [r for r in sub.results
            if r.benchmark == benchmark and r.model == model
            and r.division == division and r.systemname == system]
    assert len(rows) == 1, [r.key for r in sub.results]
    return rows[0]


# ===========================================================================
# Editions table: runs_per_result
# ===========================================================================

class TestEditionsRunsPerResult:
    def test_current_edition_carries_the_block(self):
        from mlpstorage_py.editions import checker_parameters
        assert checker_parameters().runs_per_result == V30_RUNS_PER_RESULT

    def test_field_is_a_checker_value_field(self):
        from mlpstorage_py.editions import CHECKER_FIELDS, CHECKER_VALUE_FIELDS, CheckerParameters
        assert "runs_per_result" in CHECKER_VALUE_FIELDS
        assert "runs_per_result" in CHECKER_FIELDS
        assert "runs_per_result" in {f.name for f in dataclasses_fields(CheckerParameters)}

    def test_a_checker_block_must_carry_it(self, tmp_path):
        from mlpstorage_py.editions import EditionsError

        def mutate(data):
            del data["editions"]["3.0"]["checker"]["runs_per_result"]
        with pytest.raises(EditionsError, match="runs_per_result"):
            _table_with(tmp_path, mutate)

    def test_keys_must_name_exactly_the_editions_families(self, tmp_path):
        from mlpstorage_py.editions import EditionsError

        def drop(data):
            del data["editions"]["3.0"]["checker"]["runs_per_result"]["kv_cache"]
        with pytest.raises(EditionsError, match="runs_per_result"):
            _table_with(tmp_path, drop)

        def add(data):
            data["editions"]["3.0"]["checker"]["runs_per_result"]["bert"] = 3
        with pytest.raises(EditionsError, match="runs_per_result"):
            _table_with(tmp_path, add)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "6", True])
    def test_values_must_be_positive_integers(self, tmp_path, bad):
        from mlpstorage_py.editions import EditionsError

        def mutate(data):
            data["editions"]["3.0"]["checker"]["runs_per_result"]["training"] = bad
        with pytest.raises(EditionsError, match="runs_per_result"):
            _table_with(tmp_path, mutate)

    def test_a_different_table_changes_the_value(self, tmp_path):
        def mutate(data):
            data["editions"]["3.0"]["checker"]["runs_per_result"]["training"] = 11
        table = _table_with(tmp_path, mutate)
        assert table.checker_parameters("3.0").runs_per_result["training"] == 11

    def test_config_reads_the_block(self):
        from mlpstorage_py.submission_checker.configuration.configuration import Config
        cfg = Config(submitters=None)
        assert cfg.get_runs_per_result("training") == 6
        assert cfg.get_runs_per_result("checkpointing") == 2
        assert cfg.get_runs_per_result("vector_database") == 5
        assert cfg.get_runs_per_result("kv_cache") == 1
        with pytest.raises(KeyError):
            cfg.get_runs_per_result("bert")

    def test_validator_counts_come_from_the_table(self):
        from mlpstorage_py.submission_checker import constants
        from mlpstorage_py.submission_checker.checks import directory_checks, vdb_checks
        assert not hasattr(constants, "RUN_TIMESTAMP_COUNT")
        assert "get_runs_per_result" in inspect.getsource(
            directory_checks.DirectoryCheck.run_files_timestamp_check)
        assert "get_runs_per_result" in inspect.getsource(vdb_checks.VdbCheck.vdb_run_count)
        assert "!= 5" not in inspect.getsource(vdb_checks.VdbCheck.vdb_run_count)

    def test_validator_2_1_17_uses_a_different_tables_count(self, tmp_path):
        """A table with training: 3 makes a three-leaf run phase pass 2.1.17
        and a six-leaf one fail it -- the count is the table's, not a constant."""
        from unittest.mock import MagicMock, patch

        from mlpstorage_py.submission_checker.checks.directory_checks import DirectoryCheck
        from mlpstorage_py.submission_checker.configuration.configuration import Config
        from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs
        from mlpstorage_py.tests.conftest import MockLogger

        def mutate(data):
            data["editions"]["3.0"]["checker"]["runs_per_result"]["training"] = 3
        table = _table_with(tmp_path, mutate)

        def make(n):
            folder = tmp_path / f"wl{n}"
            (folder / "run").mkdir(parents=True)
            files = [(None, None, _ts(i)) for i in range(n)]
            return SubmissionLogs(run_files=files, datagen_files=[], datasize_files=[],
                                  loader_metadata=LoaderMetadata(folder=str(folder), mode="training"))

        with patch("mlpstorage_py.submission_checker.configuration.configuration.load_editions",
                   return_value=table):
            cfg = Config(submitters=None)
        for n, ok in ((3, True), (6, False)):
            log = MockLogger()
            check = DirectoryCheck(log, cfg, make(n))
            assert check.run_files_timestamp_check() is ok, (n, log.errors)

    @pytest.mark.parametrize("doc", ["Rules.md", "README.md", "ManPage.md"])
    def test_docs_name_the_runs_per_result_block(self, doc):
        text = (PROJECT_ROOT / doc).read_text()
        lines = [ln for ln in text.splitlines() if "editions.yaml" in ln and "checker" in ln]
        assert any("runs per result" in ln or "runs_per_result" in ln for ln in lines), doc


def dataclasses_fields(cls):
    import dataclasses
    return dataclasses.fields(cls)


# ===========================================================================
# Rules.md 1.3 vocabulary
# ===========================================================================

class TestRulesMdVocabulary:
    def _block(self):
        text = (PROJECT_ROOT / "Rules.md").read_text()
        assert "1.3. **runResultSubmission**" in text
        block = text.split("1.3. **runResultSubmission**", 1)[1]
        return block.split("\n# 2.", 1)[0]

    def test_rule_defines_the_three_nouns(self):
        block = self._block()
        for noun in ("*run*", "*result*", "*submission*"):
            assert noun in block, noun
        assert "*timestamp directory*" in block
        assert "*workload directory*" in block
        assert "results.csv" in block

    def test_rule_points_at_the_table_and_the_count_rules(self):
        block = self._block()
        assert "runs_per_result" in block
        assert "editions.yaml" in block
        for ref in ("2.1.17", "4.7.1", "5.3.1", "3.3.8", "4.3.6"):
            assert ref in block, ref

    def test_commentary_entry_exists(self):
        text = (PROJECT_ROOT / "RulesCommentary.md").read_text()
        assert "## 1.3 runResultSubmission" in text
        block = self._block()
        assert "(Commentary: `RulesCommentary.md` §1.3.)" in block


# ===========================================================================
# Finding parsing
# ===========================================================================

class TestFindingParsing:
    def test_locked_format(self):
        from mlpstorage_py.readiness import parse_finding
        f = parse_finding("error", "[2.1.7 systemsDirectoryFiles] /r/closed/Acme/systems/a.yaml: "
                                   "system_under_test -> solution: Field required")
        assert f is not None
        assert (f.level, f.rule_id, f.rule_name) == ("error", "2.1.7", "systemsDirectoryFiles")
        assert f.path == "/r/closed/Acme/systems/a.yaml"
        assert f.message == "system_under_test -> solution: Field required"

    def test_check_ids_and_warnings(self):
        from mlpstorage_py.readiness import parse_finding
        f = parse_finding("warning", "[CHECK-01 poolPointerResolution] /r/closed/Acme/results/s/x: run leaf x has no pointer.")
        assert f.rule_id == "CHECK-01" and f.level == "warning"

    def test_non_rule_lines_are_not_findings(self):
        from mlpstorage_py.readiness import parse_finding
        assert parse_finding("error", "Some directory checks failed for: /r/x") is None
        assert parse_finding("error", "SUMMARY: submission has errors") is None
        assert parse_finding("info", "SUMMARY: submission looks OK") is None


# ===========================================================================
# Classification against scripted findings
# ===========================================================================

class TestResultsAndRuns:
    def test_complete_training_result_is_ready(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 6)
        sub = _evaluate(rd)
        assert sub.orgname == ORG
        assert sub.edition == RULES_EDITION
        row = _row(sub, benchmark="training", model="unet3d")
        assert (row.have, row.required) == (6, 6)
        assert row.accelerator == "b200"
        assert row.submit == "ready"
        assert row.note == ""
        assert [r.status for r in row.runs] == ["ok"] * 6
        assert all(r.counted for r in row.runs)
        assert sub.ready_count == 1 and sub.total_results == 1

    def test_short_training_result(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        _training(rd, "retinanet", 3)
        row = _row(_evaluate(rd), benchmark="training", model="retinanet")
        assert (row.have, row.required) == (3, 6)
        assert row.submit == "short"
        assert "3 more run" in row.note

    def test_failed_run_is_not_counted_and_named(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        _training(rd, "retinanet", 3)
        _leaf(rd, f"closed/{ORG}/results/{SYS}/training/retinanet/run/{_ts(3)}",
              exit_status=6, summary=False)
        row = _row(_evaluate(rd), benchmark="training", model="retinanet")
        assert (row.have, row.required) == (3, 6)
        failed = [r for r in row.runs if r.status == "failed"]
        assert len(failed) == 1 and not failed[0].counted
        assert f"run {failed[0].id} failed" in row.note
        assert "runs rm" in row.note

    def test_extra_runs_beyond_the_count(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 7)
        row = _row(_evaluate(rd), benchmark="training", model="unet3d")
        assert (row.have, row.required) == (6, 6)
        extras = [r for r in row.runs if r.status == "extra"]
        assert len(extras) == 1 and not extras[0].counted
        # the newest run is the extra one; the six oldest form the result
        assert extras[0].leaf.endswith(_ts(6))
        assert row.submit == "invalid"
        assert "1 extra run" in row.note

    def test_running_leaf_is_neither_counted_nor_failed(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 5)
        _leaf(rd, f"closed/{ORG}/results/{SYS}/training/unet3d/run/{_ts(5)}",
              metadata=False, summary=False)
        row = _row(_evaluate(rd), benchmark="training", model="unet3d")
        running = [r for r in row.runs if r.status == "running"]
        assert len(running) == 1 and not running[0].counted
        assert (row.have, row.required) == (5, 6)
        assert row.submit == "short"

    def test_invalid_verification_marks_the_run_invalid(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 5)
        _leaf(rd, f"closed/{ORG}/results/{SYS}/training/unet3d/run/{_ts(5)}",
              verification="INVALID")
        row = _row(_evaluate(rd), benchmark="training", model="unet3d")
        bad = [r for r in row.runs if r.status == "invalid"]
        assert len(bad) == 1 and not bad[0].counted
        assert "INVALID" in bad[0].reason
        assert (row.have, row.required) == (5, 6)

    def test_open_verification_under_closed_is_invalid(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 6, verification="OPEN")
        row = _row(_evaluate(rd), benchmark="training", model="unet3d")
        assert all(r.status == "invalid" for r in row.runs)
        assert "OPEN" in row.runs[0].reason
        assert row.submit == "short"

    def test_open_verification_under_open_is_ok(self, rd, scripted):
        scripted([])
        _system(rd, "open", SYS)
        _training(rd, "unet3d", 6, division="open", verification="OPEN")
        row = _row(_evaluate(rd), benchmark="training", model="unet3d", division="open")
        assert all(r.status == "ok" for r in row.runs)
        assert row.submit == "ready"

    def test_whatif_results_are_dashed(self, rd, scripted):
        scripted([])
        _training(rd, "unet3d", 2, division="whatif", verification=None)
        row = _row(_evaluate(rd), benchmark="training", model="unet3d", division="whatif")
        assert row.submit == "-"
        assert (row.have, row.required) == (2, 6)

    def test_datagen_and_datasize_leaves_are_not_runs_of_a_result(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 6)
        _leaf(rd, f"closed/{ORG}/results/{SYS}/training/unet3d/datagen/{_ts(0)}",
              summary=False)
        row = _row(_evaluate(rd), benchmark="training", model="unet3d")
        assert len(row.runs) == 6

    def test_accelerators_split_results(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 6, accelerator="b200")
        _training(rd, "unet3d", 2, accelerator="mi355", start=6)
        sub = _evaluate(rd)
        accels = sorted((r.accelerator, r.have) for r in sub.results)
        assert accels == [("b200", 6), ("mi355", 2)]

    def test_results_are_sorted(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        _system(rd, "closed", "sys-0")
        _training(rd, "unet3d", 1)
        _training(rd, "retinanet", 1)
        _training(rd, "unet3d", 1, system="sys-0")
        _leaf(rd, f"closed/{ORG}/results/{SYS}/kv_cache/llama3.1-8b/run/{_ts(0)}", accelerator=None)
        sub = _evaluate(rd)
        keys = [(r.systemname, r.benchmark, r.model) for r in sub.results]
        assert keys == sorted(keys)


class TestCheckpointingPhases:
    def _ckpt(self, rd, model, *, writes, reads, i=0, **kw):
        return _leaf(rd, f"closed/{ORG}/results/{SYS}/checkpointing/{model}/{_ts(i)}",
                     writes=writes, reads=reads, **kw)

    def test_combined_invocation_covers_both_phases(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        self._ckpt(rd, "llama3-8b", writes=10, reads=10)
        row = _row(_evaluate(rd), benchmark="checkpointing", model="llama3-8b")
        assert (row.have, row.required) == (2, 2)
        assert row.submit == "ready"
        assert row.runs[0].counted

    def test_write_only_is_short_of_the_read_phase(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        self._ckpt(rd, "llama3-70b", writes=10, reads=0)
        row = _row(_evaluate(rd), benchmark="checkpointing", model="llama3-70b")
        assert (row.have, row.required) == (1, 2)
        assert row.submit == "short"
        assert "read phase" in row.note

    def test_two_phase_invocations_are_complete(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        self._ckpt(rd, "llama3-70b", writes=10, reads=0, i=0)
        self._ckpt(rd, "llama3-70b", writes=0, reads=10, i=1)
        row = _row(_evaluate(rd), benchmark="checkpointing", model="llama3-70b")
        assert (row.have, row.required) == (2, 2)
        assert row.submit == "ready"
        assert [r.counted for r in row.runs] == [True, True]

    def test_third_invocation_is_extra(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        self._ckpt(rd, "llama3-8b", writes=10, reads=10, i=0)
        self._ckpt(rd, "llama3-8b", writes=10, reads=10, i=1)
        self._ckpt(rd, "llama3-8b", writes=10, reads=10, i=2)
        row = _row(_evaluate(rd), benchmark="checkpointing", model="llama3-8b")
        assert (row.have, row.required) == (2, 2)
        assert [r.status for r in row.runs] == ["ok", "extra", "extra"]
        assert row.submit == "invalid"
        assert "2 extra run" in row.note


class TestVdbAndKvcache:
    def test_vdb_needs_five(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        for i in range(4):
            _leaf(rd, f"closed/{ORG}/results/{SYS}/vector_database/milvus/DISKANN/run/{_ts(i)}",
                  accelerator=None)
        row = _row(_evaluate(rd), benchmark="vector_database", model="milvus/DISKANN")
        assert (row.have, row.required) == (4, 5)
        assert row.accelerator is None
        assert row.submit == "short"

    def test_kvcache_needs_one(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        _leaf(rd, f"closed/{ORG}/results/{SYS}/kv_cache/llama3.1-8b/run/{_ts(0)}", accelerator=None)
        row = _row(_evaluate(rd), benchmark="kv_cache", model="llama3.1-8b")
        assert (row.have, row.required) == (1, 1)
        assert row.submit == "ready"


class TestFindingsAttribution:
    def _sys_yaml(self, rd, division="closed", name=SYS):
        return os.path.join(rd, division, ORG, "systems", f"{name}.yaml")

    def _leaf_path(self, rd, i=0, model="unet3d"):
        return os.path.join(rd, "closed", ORG, "results", SYS, "training", model, "run", _ts(i))

    def test_blank_system_yaml_fields_are_paperwork(self, rd, scripted):
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 6)
        y = self._sys_yaml(rd)
        scripted([
            ("error", "2.1.7", "systemsDirectoryFiles", y, "system_under_test -> solution: Field required"),
            ("error", "2.1.7", "systemsDirectoryFiles", y,
             "clients -> 0 -> friendly_description: String should have at least 1 character"),
            ("error", "4.7.3", "checkpointRemappingTimeReporting", y,
             "system_under_test -> solution -> capabilities -> remap_time_in_seconds: Field required"),
        ])
        sub = _evaluate(rd)
        pw = sub.paperwork[("closed", SYS)]
        assert pw.blank_fields == [
            "system_under_test -> solution",
            "clients -> 0 -> friendly_description",
            "system_under_test -> solution -> capabilities -> remap_time_in_seconds",
        ]
        assert not pw.pdf_missing and not pw.yaml_missing
        row = _row(sub, benchmark="training", model="unet3d")
        assert row.submit == "paperwork"
        assert f"{SYS}.yaml: 3 blank fields" in row.note
        assert sub.ready_count == 0
        # paperwork is a property of the system, never of a run
        assert all(r.status == "ok" for r in row.runs)

    def test_missing_pdf_and_yaml(self, rd, scripted):
        _system(rd, "closed", SYS, pdf=False)
        _system(rd, "closed", "sys-2", yaml_ok=False)
        _training(rd, "unet3d", 6)
        _training(rd, "unet3d", 6, system="sys-2")
        systems = os.path.join(rd, "closed", ORG, "systems")
        scripted([
            ("error", "2.1.7", "systemsDirectoryFiles", os.path.join(systems, f"{SYS}.pdf"),
             f"{SYS}.yaml has no matching {SYS}.pdf in systems/"),
            ("error", "2.1.8", "resultsDirectorySystems", os.path.join(rd, "closed", ORG, "results", "sys-2"),
             "results/sys-2/ exists but systems/sys-2.yaml is missing"),
        ])
        sub = _evaluate(rd)
        assert sub.paperwork[("closed", SYS)].pdf_missing
        assert sub.paperwork[("closed", "sys-2")].yaml_missing
        assert _row(sub, benchmark="training", model="unet3d").submit == "paperwork"
        assert _row(sub, benchmark="training", model="unet3d", system="sys-2").submit == "paperwork"
        assert f"{SYS}.pdf missing" in _row(sub, benchmark="training", model="unet3d").note

    def test_short_beats_paperwork(self, rd, scripted):
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 2)
        scripted([("error", "2.1.7", "systemsDirectoryFiles", self._sys_yaml(rd),
                   "system_under_test -> solution: Field required")])
        row = _row(_evaluate(rd), benchmark="training", model="unet3d")
        assert row.submit == "short"
        assert row.note.startswith("4 more run")

    def test_code_image_findings_are_result_paperwork(self, rd, scripted):
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 6)
        scripted([("error", "CHECK-01", "poolPointerResolution", self._leaf_path(rd, 2),
                   "run leaf %s .mlps-code-image references hash aaaa that no pool holds" % _ts(2))])
        row = _row(_evaluate(rd), benchmark="training", model="unet3d")
        assert row.submit == "paperwork"
        assert "code image" in row.note
        assert all(r.status == "ok" for r in row.runs)
        assert row.paperwork and "CHECK-01" in row.paperwork[0].rule_id

    def test_rule_violation_on_a_leaf_makes_the_run_invalid(self, rd, scripted):
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 6)
        scripted([("error", "3.3.2", "trainingAcceleratorUtilizationCheck", self._leaf_path(rd, 4),
                   "mean AU 0.81 is below the unet3d minimum 0.90")])
        row = _row(_evaluate(rd), benchmark="training", model="unet3d")
        bad = [r for r in row.runs if r.status == "invalid"]
        assert len(bad) == 1 and bad[0].leaf.endswith(_ts(4))
        assert "3.3.2" in bad[0].reason
        assert (row.have, row.required) == (5, 6)
        assert row.submit == "short"

    def test_violation_on_a_file_inside_a_leaf_attributes_to_the_leaf(self, rd, scripted):
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 6)
        inner = os.path.join(self._leaf_path(rd, 1), "dlio_config", "config.yaml")
        scripted([("error", "3.6.2", "trainingClosedSubmissionParameters", inner,
                   "dataset.num_files_train overridden")])
        row = _row(_evaluate(rd), benchmark="training", model="unet3d")
        assert [r.status for r in row.runs].count("invalid") == 1

    def test_workload_level_violation_marks_the_result_invalid(self, rd, scripted):
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 6)
        wl = os.path.join(rd, "closed", ORG, "results", SYS, "training", "unet3d")
        scripted([("error", "2.1.12", "trainingPhases", wl, "missing phase directory 'datasize'")])
        row = _row(_evaluate(rd), benchmark="training", model="unet3d")
        assert row.submit == "invalid"
        assert "2.1.12" in row.note
        assert all(r.status == "ok" for r in row.runs)
        assert (row.have, row.required) == (6, 6)

    def test_count_and_rollup_findings_are_folded_or_dropped(self, rd, scripted):
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 3)
        run_dir = os.path.join(rd, "closed", ORG, "results", SYS, "training", "unet3d", "run")
        scripted([
            ("error", "2.1.17", "runTimestamps", run_dir, "Expected 6 run files, but found 3."),
            ("error", "2.1.16", "runResultsJson", run_dir, "no results.json"),
            ("error", "RPT-01", "rollupTableAgreement", os.path.join(rd, "closed", ORG, "results", SYS),
             "results.csv disagrees"),
        ])
        sub = _evaluate(rd)
        row = _row(sub, benchmark="training", model="unet3d")
        assert row.submit == "short"
        assert not row.problems
        assert sub.tree_problems == []
        assert "2.1.17" not in row.note and "results.json" not in row.note

    def test_tree_level_findings(self, rd, scripted):
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 6)
        scripted([
            ("error", "EDN-01", "rulesEdition", os.path.join(rd, "closed", ORG, "submission.yaml"),
             "declares rules edition 9.9"),
            ("error", "2.1.2", "topLevelSubdirectories", os.path.join(rd, "stray"),
             "unexpected top-level directory 'stray'"),
        ])
        sub = _evaluate(rd)
        assert [f.rule_id for f in sub.tree_problems] == ["EDN-01", "2.1.2"]
        # a tree-level error blocks submit for every result
        assert _row(sub, benchmark="training", model="unet3d").submit == "ready"
        assert not sub.submittable

    def test_whatif_findings_are_dropped(self, rd, scripted):
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 6)
        _training(rd, "unet3d", 1, division="whatif", verification=None)
        scripted([
            ("error", "2.1.2", "topLevelSubdirectories", os.path.join(rd, "whatif"),
             "unexpected top-level directory 'whatif'"),
            ("error", "3.3.2", "trainingAcceleratorUtilizationCheck",
             os.path.join(rd, "whatif", ORG, "results", SYS, "training", "unet3d", "run", _ts(0)), "low AU"),
        ])
        sub = _evaluate(rd)
        assert sub.tree_problems == []
        assert _row(sub, benchmark="training", model="unet3d", division="whatif").runs[0].status == "ok"

    def test_warnings_never_change_the_token(self, rd, scripted):
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 6)
        scripted([
            ("warning", "EDN-03", "dlioRevision", self._leaf_path(rd, 0), "unlisted DLIO revision"),
            ("warning", "2.1.7", "systemsDirectoryFiles", self._sys_yaml(rd), "something soft"),
        ])
        sub = _evaluate(rd)
        row = _row(sub, benchmark="training", model="unet3d")
        assert row.submit == "ready"
        assert all(r.status == "ok" for r in row.runs)
        assert len(sub.warnings) == 2
        assert sub.submittable

    def test_run_notes_list_every_reason(self, rd, scripted):
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 6)
        leaf = self._leaf_path(rd, 0)
        scripted([
            ("error", "3.3.2", "trainingAcceleratorUtilizationCheck", leaf, "low AU"),
            ("error", "3.6.2", "trainingClosedSubmissionParameters", leaf, "param overridden"),
        ])
        row = _row(_evaluate(rd), benchmark="training", model="unet3d")
        bad = [r for r in row.runs if r.status == "invalid"][0]
        assert [f.rule_id for f in bad.problems] == ["3.3.2", "3.6.2"]


class TestSubmissionShape:
    def test_to_dict_is_json_serialisable_and_complete(self, rd, scripted):
        _system(rd, "closed", SYS)
        _training(rd, "unet3d", 6)
        scripted([("error", "2.1.7", "systemsDirectoryFiles",
                   os.path.join(rd, "closed", ORG, "systems", f"{SYS}.yaml"),
                   "system_under_test -> solution: Field required")])
        sub = _evaluate(rd)
        data = json.loads(json.dumps(sub.to_dict()))
        assert data["results_dir"] == rd
        assert data["orgname"] == ORG
        assert data["rules_edition"] == RULES_EDITION
        assert data["submittable"] is False
        (row,) = data["results"]
        assert row["runs_required"] == 6 and row["runs_have"] == 6
        assert row["submit"] == "paperwork"
        assert row["runs"][0]["status"] == "ok" and row["runs"][0]["counted"] is True
        assert data["paperwork"][0]["systemname"] == SYS
        assert data["paperwork"][0]["blank_fields"] == ["system_under_test -> solution"]

    def test_paperwork_items_are_human_lines(self, rd, scripted):
        _system(rd, "closed", SYS, pdf=False)
        _training(rd, "unet3d", 6)
        systems = os.path.join(rd, "closed", ORG, "systems")
        scripted([
            ("error", "2.1.7", "systemsDirectoryFiles", os.path.join(systems, f"{SYS}.yaml"),
             "system_under_test -> solution: Field required"),
            ("error", "2.1.7", "systemsDirectoryFiles", os.path.join(systems, f"{SYS}.pdf"),
             f"{SYS}.yaml has no matching {SYS}.pdf in systems/"),
        ])
        sub = _evaluate(rd)
        items = sub.paperwork[("closed", SYS)].items()
        assert items == [f"systems/{SYS}.yaml: 1 blank field", f"systems/{SYS}.pdf missing"]

    def test_evaluate_result_finds_the_row_of_a_leaf(self, rd, scripted):
        scripted([])
        _system(rd, "closed", SYS)
        leaves = _training(rd, "unet3d", 6)
        _training(rd, "retinanet", 2)
        from mlpstorage_py.readiness import evaluate_result
        row = evaluate_result(rd, leaves[3])
        assert row is not None and row.model == "unet3d"
        assert evaluate_result(rd, os.path.join(rd, "nowhere")) is None

    def test_empty_results_dir(self, rd, scripted):
        scripted([])
        sub = _evaluate(rd)
        assert sub.results == [] and sub.paperwork == {}
        assert sub.total_results == 0 and sub.ready_count == 0

    def test_evaluate_requires_an_initialized_results_dir(self, tmp_path, scripted):
        scripted([])
        from mlpstorage_py.readiness import evaluate
        from mlpstorage_py.results_dir.errors import ResultsDirNotInitializedError
        bare = tmp_path / "bare"
        bare.mkdir()
        with pytest.raises(ResultsDirNotInitializedError):
            evaluate(str(bare))


# ===========================================================================
# The real submission checker
# ===========================================================================

def _dod_tree(tmp_path, **overrides) -> str:
    """The definition-of-done fixture tree as an initialized results-dir:
    sentinel added, ``metadata.json`` renamed to the tool's
    ``<family>_<ts>_metadata.json`` so the ledger sees it."""
    from mlpstorage_py.tests.conftest import build_submission

    root = build_submission(tmp_path, **overrides)
    rd = str(root)
    write_sentinel(rd, "Acme")
    for dirpath, _dirs, files in os.walk(os.path.join(rd, "closed")):
        if "metadata.json" in files:
            parts = dirpath.replace(os.sep, "/").split("/")
            family = parts[parts.index("results") + 2] if "results" in parts else "training"
            os.rename(os.path.join(dirpath, "metadata.json"),
                      os.path.join(dirpath, f"{family}_{parts[-1]}_metadata.json"))
    return rd


class TestRealValidator:
    def test_capture_is_silent_and_restores_logging(self, tmp_path, xdg, capsys):
        from mlpstorage_py.readiness import evaluate
        rd = _dod_tree(tmp_path)
        main_log = logging.getLogger("main")
        loader_log = logging.getLogger("Loader")
        before = (main_log.propagate, list(main_log.handlers), loader_log.propagate)
        sub = evaluate(rd)
        assert (main_log.propagate, list(main_log.handlers), loader_log.propagate) == before
        out = capsys.readouterr()
        assert out.out == "" and out.err == ""
        assert not list(tmp_path.glob("summary.csv"))
        assert sub.total_results >= 2

    def test_dod_tree_results(self, tmp_path, xdg):
        """The definition-of-done "good" fixture is known-noisy under
        ``validate`` (its run leaves carry no DLIO log files and the tree has
        no code-image pool): the evaluator must say exactly that -- every run
        invalid under the leaf-files rules, the missing pool at tree level --
        rather than count the leaves as ok."""
        from mlpstorage_py.readiness import evaluate
        sub = evaluate(_dod_tree(tmp_path))
        train = _row(sub, benchmark="training", model="unet3d", system="acme-storage-v1")
        assert train.required == 6 and len(train.runs) == 6
        assert all(r.status == "invalid" and "[2.1.19]" in r.reason for r in train.runs)
        assert (train.have, train.submit) == (0, "short")
        ckpt = _row(sub, benchmark="checkpointing", model="llama3-8b", system="acme-storage-v1")
        assert ckpt.required == 2 and len(ckpt.runs) == 2
        assert all(r.status == "invalid" and "2.1.25" in {p.rule_id for p in r.problems}
                   for r in ckpt.runs)
        assert "CHECK-01" in {f.rule_id for f in sub.tree_problems}
        # 2.1.18 names the bare timestamp directory; unique in the tree, it
        # lands on the run rather than at tree level
        assert "2.1.18" not in {f.rule_id for f in sub.tree_problems}
        assert any("2.1.18" in {p.rule_id for p in r.problems} for r in train.runs)

    def test_missing_pdf_is_paperwork(self, tmp_path, xdg):
        from mlpstorage_py.readiness import evaluate
        sub = evaluate(_dod_tree(tmp_path, missing_systems_pdf=True))
        pw = sub.paperwork[("closed", "acme-storage-v1")]
        assert pw.pdf_missing and not pw.yaml_missing
        assert pw.items() == ["systems/acme-storage-v1.pdf missing"]
        train = _row(sub, benchmark="training", model="unet3d", system="acme-storage-v1")
        assert train.system_paperwork is pw
        # the runs are invalid on this fixture, so short wins over paperwork
        assert train.submit == "short"

    def test_schema_error_is_a_blank_field(self, tmp_path, xdg):
        from mlpstorage_py.readiness import evaluate
        sub = evaluate(_dod_tree(tmp_path, system_yaml_bad_deployment=12345))
        pw = sub.paperwork[("closed", "acme-storage-v1")]
        assert pw.blank_fields == ["system_under_test -> deployment"]

    def test_findings_carry_rule_ids_from_the_checker(self, tmp_path, xdg):
        from mlpstorage_py.readiness import collect_findings
        findings = collect_findings(_dod_tree(tmp_path, missing_systems_pdf=True))
        ids = {f.rule_id for f in findings}
        assert "2.1.7" in ids
        assert all(re.fullmatch(r"[0-9.]+[a-z]?|[A-Z]+-\d\d", f.rule_id) for f in findings), ids
