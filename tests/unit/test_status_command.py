"""``mlpstorage status`` and the per-run readiness token (status-and-submit PR 2).

Design (2026-09-23, decision 3): ``status`` is the habitual per-result view of
a results-dir. One row per result (Rules.md 1.3), RUNS ``n/m`` with ``m`` from
the edition's ``runs_per_result``, a SUBMIT token, a NOTE, a de-duplicated
paperwork footer and a "Next:" line; ``--runs`` expands each result into its
run rows. The same table is printed for the result a ``run`` just added to,
after the run finishes. ``runs list`` gains the per-run token as a SUBMIT
column and stays flat.

Covered here:
- the parser: ``status`` is a top-level leaf with the ``runs list`` filters,
  ``--submit``, ``--runs`` and ``--json``;
- the table: per-division header, columns, tokens, NOTE, whatif section,
  ready count, paperwork footer, "Next:" line, ``--runs`` expansion;
- filters and ``--json``;
- the results-dir gates shared with ``runs`` and history silence;
- the post-run recap from ``run_benchmark``: printed for ``run`` only, silent
  under ``--quiet``, never changes the exit code;
- ``runs list``: SUBMIT column and JSON key;
- ``readiness.run_tokens`` / ``readiness.leaf_key``;
- ``--help_all`` tree, block and context tokens.
"""

from __future__ import annotations

import json
import os
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mlpstorage_py.config import EXIT_CODE
from mlpstorage_py.results_dir import write_sentinel

from tests.unit.test_readiness import (  # noqa: F401  (fixtures by import)
    HASH_A,
    ORG,
    SYS,
    _leaf,
    _system,
    _training,
    _ts,
    rd,
    scripted,
    xdg,
)
from tests.unit.test_runs_management import (  # noqa: F401  (fixture by import)
    LEAF_TRAIN_RUN,
    LEAF_TRAIN_RUN_2,
    tree,
)


def _main(argv):
    from mlpstorage_py import main as main_mod

    with patch("sys.argv", ["mlpstorage"] + list(argv)), \
         patch.object(main_mod, "apply_logging_options"):
        return main_mod.main()


def _lines(text: str):
    return [line.rstrip() for line in text.splitlines()]


def _table_rows(out: str):
    """The result rows of the first table: header cells split on 2+ spaces."""
    rows = []
    for line in _lines(out):
        if line.startswith("SYSTEM"):
            rows.append(("header", line))
        elif rows and line and not line.startswith(" ") and not line.startswith("whatif/") \
                and not line.startswith("closed/") and not line.startswith("open/"):
            rows.append(("row", line))
        elif rows and not line:
            break
    return rows


def _systems_finding(rd, division, name, loc):
    return ("error", "2.1.7", "systemYamlValid",
            os.path.join(rd, division, ORG, "systems", f"{name}.yaml"), f"{loc}: Field required")


def _pdf_finding(rd, division, name):
    return ("error", "2.1.8", "systemPdfPresent",
            os.path.join(rd, division, ORG, "systems", f"{name}.yaml"),
            f"{division}/{ORG}/systems/{name}.yaml has no matching {name}.pdf")


def _failed_id(rd):
    """The ledger ID of the failed retinanet run in ``two_results``."""
    from mlpstorage_py.runs.ledger import sync
    return next(r.id for r in sync(rd) if r.leaf.endswith(_ts(5)))


@pytest.fixture
def two_results(rd, scripted):
    """closed/sys-1: unet3d complete (6 ok), retinanet short (2 ok, 1 failed);
    the system's YAML has two blank fields and no PDF."""
    _training(rd, "unet3d", 6)
    _training(rd, "retinanet", 2)
    _leaf(rd, f"closed/{ORG}/results/{SYS}/training/retinanet/run/{_ts(5)}", exit_status=6, summary=False)
    _system(rd, "closed", SYS, pdf=False)
    scripted([
        _systems_finding(rd, "closed", SYS, "system_under_test -> vendor"),
        _systems_finding(rd, "closed", SYS, "system_under_test -> product"),
        _pdf_finding(rd, "closed", SYS),
    ])
    return rd


# ===========================================================================
# Parser
# ===========================================================================

class TestStatusParser:
    def _leaf(self):
        from mlpstorage_py.cli_parser import build_parser
        import argparse
        parser = build_parser()
        top = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        return top.choices["status"]

    def test_status_is_a_top_level_leaf(self):
        import argparse
        sub = self._leaf()
        assert not any(isinstance(a, argparse._SubParsersAction) for a in sub._actions)

    def test_flags(self):
        sub = self._leaf()
        longs = {o for a in sub._actions for o in a.option_strings if o.startswith("--")}
        assert {"--results-dir", "--mode", "--benchmark", "--model", "--systemname",
                "--submit", "--runs", "--json"} <= longs
        assert "--status" not in longs  # per-run ledger states are `runs list` territory

    def test_short_aliases(self):
        sub = self._leaf()
        pairs = {tuple(a.option_strings) for a in sub._actions if len(a.option_strings) == 2}
        assert ("--results-dir", "-rd") in pairs
        assert ("--systemname", "-sn") in pairs

    def test_submit_choices(self):
        sub = self._leaf()
        action = next(a for a in sub._actions if "--submit" in a.option_strings)
        assert list(action.choices) == ["short", "invalid", "paperwork", "ready"]

    def test_parses(self):
        from mlpstorage_py.cli_parser import build_parser
        args = build_parser().parse_args(["status", "--runs", "--json", "--submit", "short",
                                          "--benchmark", "vectordb", "-sn", "x"])
        assert args.mode == "status" and args.runs and args.json
        assert args.submit == "short" and args.benchmark == "vectordb" and args.systemname == "x"


# ===========================================================================
# The table
# ===========================================================================

class TestStatusTable:
    def test_header_line_names_division_org_edition_and_counts(self, two_results, capsys):
        assert _main(["status", "-rd", two_results]) == EXIT_CODE.SUCCESS
        out = capsys.readouterr().out
        first = _lines(out)[0]
        assert first.startswith(f"closed/{ORG}")
        assert "rules edition 3.0" in first
        assert "2 results" in first and "9 runs" in first

    def test_columns_and_rows(self, two_results, capsys):
        _main(["status", "-rd", two_results])
        out = capsys.readouterr().out
        rows = _table_rows(out)
        assert rows[0][1].split() == ["SYSTEM", "BENCHMARK", "MODEL", "ACCEL", "RUNS", "SUBMIT", "NOTE"]
        body = [r[1] for r in rows[1:]]
        assert len(body) == 2
        retina = next(l for l in body if "retinanet" in l)
        unet = next(l for l in body if "unet3d" in l)
        assert retina.split()[:6] == [SYS, "training", "retinanet", "b200", "2/6", "short"]
        assert "4 more runs needed" in retina
        failed = _failed_id(two_results)
        assert f"run {failed} failed" in retina and f"mlpstorage runs rm {failed}" in retina
        assert unet.split()[:6] == [SYS, "training", "unet3d", "b200", "6/6", "paperwork"]

    def test_paperwork_note_is_short_and_footer_carries_it_once(self, two_results, capsys):
        _main(["status", "-rd", two_results])
        out = capsys.readouterr().out
        unet = next(l for l in _lines(out) if "unet3d" in l and l.startswith(SYS))
        # the row says what, briefly; the footer says where
        assert f"{SYS}.yaml: 2 blank fields" in unet
        assert f"{SYS}.pdf missing" in unet
        assert "systems/" not in unet
        footer = [l for l in _lines(out) if l.startswith("Paperwork")]
        assert len(footer) == 1
        assert f"systems/{SYS}.yaml: 2 blank fields" in footer[0]
        assert f"systems/{SYS}.pdf missing" in footer[0]
        assert "closed" in footer[0] and SYS in footer[0]

    def test_ready_count_and_next_line(self, two_results, capsys):
        _main(["status", "-rd", two_results])
        out = capsys.readouterr().out
        assert "0 of 2 results ready" in out
        nxt = [l for l in _lines(out) if l.startswith("Next:")]
        assert len(nxt) == 1 and nxt[0] == _lines(out)[-1]
        assert "mlpstorage status --runs" in nxt[0]

    def test_all_ready_points_at_submit_dry_run(self, rd, scripted, capsys):
        _training(rd, "unet3d", 6)
        _system(rd, "closed", SYS)
        scripted([])
        _main(["status", "-rd", rd])
        out = capsys.readouterr().out
        row = next(l for l in _lines(out) if l.startswith(SYS))
        assert row.split()[:6] == [SYS, "training", "unet3d", "b200", "6/6", "ready"]
        assert "1 of 1 result ready" in out
        assert "Paperwork" not in out
        nxt = [l for l in _lines(out) if l.startswith("Next:")][0]
        assert "mlpstorage submit --dry-run" in nxt

    def test_paperwork_only_next_says_fill_it_in(self, rd, scripted, capsys):
        _training(rd, "unet3d", 6)
        _system(rd, "closed", SYS, pdf=False)
        scripted([_pdf_finding(rd, "closed", SYS)])
        _main(["status", "-rd", rd])
        out = capsys.readouterr().out
        nxt = [l for l in _lines(out) if l.startswith("Next:")][0]
        assert "paperwork" in nxt.lower() and "mlpstorage status" in nxt

    def test_whatif_results_are_a_separate_section_with_dash(self, rd, scripted, capsys):
        _training(rd, "unet3d", 6)
        _training(rd, "unet3d", 1, division="whatif")
        _system(rd, "closed", SYS)
        scripted([])
        _main(["status", "-rd", rd])
        out = capsys.readouterr().out
        lines = _lines(out)
        assert lines[0].startswith(f"closed/{ORG}")
        whatif_hdr = next(l for l in lines if l.startswith(f"whatif/{ORG}"))
        assert "1 result" in whatif_hdr and "never packaged" in whatif_hdr
        whatif_row = lines[lines.index(whatif_hdr) + 3]
        assert whatif_row.split()[:6] == [SYS, "training", "unet3d", "b200", "1/6", "-"]
        assert "1 of 1 result ready" in out  # whatif never counts

    def test_open_division_gets_its_own_header(self, rd, scripted, capsys):
        _training(rd, "unet3d", 6)
        _training(rd, "unet3d", 6, division="open")
        _system(rd, "closed", SYS)
        _system(rd, "open", SYS)
        scripted([])
        _main(["status", "-rd", rd])
        lines = _lines(capsys.readouterr().out)
        assert lines[0].startswith(f"closed/{ORG}")
        assert any(l.startswith(f"open/{ORG}") for l in lines)
        assert "2 of 2 results ready" in "\n".join(lines)

    def test_checkpointing_counts_phases(self, rd, scripted, capsys):
        _leaf(rd, f"closed/{ORG}/results/{SYS}/checkpointing/llama3-70b/{_ts(0)}", writes=10, reads=0)
        _system(rd, "closed", SYS)
        scripted([])
        _main(["status", "-rd", rd])
        row = next(l for l in _lines(capsys.readouterr().out) if l.startswith(SYS))
        assert row.split()[:6] == [SYS, "checkpointing", "llama3-70b", "b200", "1/2", "short"]
        assert "read phase missing" in row

    def test_vdb_row_has_no_accelerator(self, rd, scripted, capsys):
        for i in range(5):
            _leaf(rd, f"closed/{ORG}/results/{SYS}/vector_database/milvus/DISKANN/run/{_ts(i)}",
                  accelerator=None)
        _system(rd, "closed", SYS)
        scripted([])
        _main(["status", "-rd", rd])
        row = next(l for l in _lines(capsys.readouterr().out) if l.startswith(SYS))
        assert row.split()[:6] == [SYS, "vector_database", "milvus/DISKANN", "-", "5/5", "ready"]

    def test_tree_problems_are_listed(self, rd, scripted, capsys):
        _training(rd, "unet3d", 6)
        _system(rd, "closed", SYS)
        scripted([("error", "2.1.1", "layoutTop", rd, "unexpected top-level entry `junk`")])
        _main(["status", "-rd", rd])
        out = capsys.readouterr().out
        assert "Tree problems" in out
        assert "[2.1.1] unexpected top-level entry `junk`" in out
        nxt = [l for l in _lines(out) if l.startswith("Next:")][0]
        assert "mlpstorage validate" in nxt

    def test_warnings_are_counted_not_listed(self, rd, scripted, capsys):
        _training(rd, "unet3d", 6)
        _system(rd, "closed", SYS)
        scripted([("warning", "3.4.1", "trainingAU", f"{rd}/closed/{ORG}/results/{SYS}/training/unet3d/run/{_ts(0)}", "AU low")])
        _main(["status", "-rd", rd])
        out = capsys.readouterr().out
        assert "1 warning" in out and "mlpstorage validate" in out
        assert "AU low" not in out

    def test_empty_tree(self, rd, scripted, capsys):
        scripted([])
        assert _main(["status", "-rd", rd]) == EXIT_CODE.SUCCESS
        assert "no runs" in capsys.readouterr().out.lower()


class TestStatusRuns:
    def test_runs_expands_each_result(self, two_results, capsys):
        assert _main(["status", "--runs", "-rd", two_results]) == EXIT_CODE.SUCCESS
        lines = _lines(capsys.readouterr().out)
        header_idx = next(i for i, l in enumerate(lines) if l.startswith("SYSTEM"))
        run_header = lines[header_idx + 1]
        assert run_header.startswith("  ") and run_header.split() == ["ID", "STATUS", "STARTED", "COUNTED", "NOTE"]
        run_rows = [l for l in lines if l.startswith("  ") and l.split() and l.split()[0].isdigit()]
        assert len(run_rows) == 9
        failed = next(l for l in run_rows if l.split()[0] == str(_failed_id(two_results)))
        assert failed.split()[1] == "failed" and failed.split()[4] == "-"
        assert "exit status 6" in failed
        counted = [l for l in run_rows if l.split()[4] == "yes"]
        assert len(counted) == 8
        first = run_rows[0].split()
        assert first[1] == "ok" and first[2] == "2026-09-01" and first[3] == "10:00:00"

    def test_run_rows_follow_their_result(self, two_results, capsys):
        _main(["status", "--runs", "-rd", two_results])
        lines = _lines(capsys.readouterr().out)
        idx_retina = next(i for i, l in enumerate(lines) if l.startswith(SYS) and "retinanet" in l)
        following = []
        for l in lines[idx_retina + 1:]:
            if not l.startswith("  "):
                break
            following.append(l)
        assert len(following) == 3
        assert any(l.split()[1] == "failed" for l in following)

    def test_extra_runs_are_marked(self, rd, scripted, capsys):
        _training(rd, "unet3d", 7)
        _system(rd, "closed", SYS)
        scripted([])
        _main(["status", "--runs", "-rd", rd])
        lines = _lines(capsys.readouterr().out)
        extra = next(l for l in lines if l.startswith("  ") and " extra " in l)
        assert extra.split()[4] == "-" and "beyond the 6 runs" in extra


class TestStatusFilters:
    @pytest.mark.parametrize("flags, expected_models", [
        (["--model", "unet3d"], ["unet3d"]),
        (["--submit", "short"], ["retinanet"]),
        (["--submit", "paperwork"], ["unet3d"]),
        (["--submit", "ready"], []),
        (["--benchmark", "training"], ["retinanet", "unet3d"]),
        (["--benchmark", "vectordb"], []),
        (["--systemname", SYS], ["retinanet", "unet3d"]),
        (["-sn", "other"], []),
        (["--mode", "closed"], ["retinanet", "unet3d"]),
        (["--mode", "open"], []),
    ])
    def test_filters(self, two_results, capsys, flags, expected_models):
        assert _main(["status", "--json", "-rd", two_results] + flags) == EXIT_CODE.SUCCESS
        data = json.loads(capsys.readouterr().out)
        assert sorted(r["model"] for r in data["results"]) == expected_models

    def test_nothing_matches_says_so(self, two_results, capsys):
        assert _main(["status", "-rd", two_results, "--submit", "ready"]) == EXIT_CODE.SUCCESS
        assert "no results match" in capsys.readouterr().out.lower()


class TestStatusJson:
    def test_shape(self, two_results, capsys):
        assert _main(["status", "--json", "-rd", two_results]) == EXIT_CODE.SUCCESS
        data = json.loads(capsys.readouterr().out)
        assert data["orgname"] == ORG and data["rules_edition"] == "3.0"
        assert data["submittable"] is False
        assert {r["model"]: r["submit"] for r in data["results"]} == {"unet3d": "paperwork", "retinanet": "short"}
        unet = next(r for r in data["results"] if r["model"] == "unet3d")
        assert unet["runs_have"] == 6 and unet["runs_required"] == 6
        assert len(unet["runs"]) == 6 and all(r["counted"] for r in unet["runs"])
        assert data["paperwork"][0]["systemname"] == SYS
        assert data["paperwork"][0]["blank_fields"] == ["system_under_test -> vendor", "system_under_test -> product"]
        assert data["tree_problems"] == [] and data["warnings"] == []

    def test_json_is_the_only_stdout(self, two_results, capsys):
        _main(["status", "--json", "-rd", two_results])
        json.loads(capsys.readouterr().out)  # nothing else on stdout


class TestStatusGates:
    def test_uses_recorded_default_results_dir(self, tree, scripted, capsys):
        scripted([])
        assert _main(["status"]) == EXIT_CODE.SUCCESS
        assert f"closed/{ORG}" in capsys.readouterr().out

    def test_no_results_dir_anywhere_is_actionable(self, xdg, caplog):
        import logging
        with caplog.at_level(logging.ERROR, logger="MLPerfStorage"):
            rc = _main(["status"])
        assert rc != EXIT_CODE.SUCCESS
        assert "mlpstorage init" in " ".join(r.getMessage() for r in caplog.records)

    def test_uninitialized_results_dir_is_refused(self, tmp_path, xdg, caplog):
        import logging
        bare = tmp_path / "bare"
        bare.mkdir()
        with caplog.at_level(logging.ERROR, logger="MLPerfStorage"):
            rc = _main(["status", "-rd", str(bare)])
        assert rc != EXIT_CODE.SUCCESS
        assert "has not been initialized" in " ".join(r.getMessage() for r in caplog.records)
        assert not (bare / ".mlps").exists()

    def test_not_recorded_in_history(self, tree, scripted):
        from mlpstorage_py.history import history_file_for
        scripted([])
        _main(["status"])
        hist = history_file_for(tree)
        assert not os.path.exists(hist) or open(hist).read().strip() == ""

    def test_exit_zero_even_when_nothing_is_ready(self, two_results):
        assert _main(["status", "-rd", two_results]) == EXIT_CODE.SUCCESS


# ===========================================================================
# Post-run recap
# ===========================================================================

class _Bench:
    def __init__(self, leaf, rc=EXIT_CODE.SUCCESS):
        self.metadata_file_path = os.path.join(leaf, "x_metadata.json")
        self.run_result_output = leaf
        self.exit_status = None
        self._rc = rc

    def run(self):
        return self._rc

    def write_metadata(self):
        pass


def _run_benchmark(args, bench):
    from mlpstorage_py import main as main_mod

    fake = SimpleNamespace(TrainingBenchmark=lambda a, run_datetime, logger: bench,
                           VectorDBBenchmark=None, CheckpointingBenchmark=None, KVCacheBenchmark=None)
    with patch.dict("sys.modules", {"mlpstorage_py.benchmarks": fake}), \
         patch.object(main_mod, "_check_and_migrate_legacy_layout"), \
         patch.object(main_mod, "capture_or_verify_code_image"), \
         patch.object(main_mod, "attach_run_log_files", return_value=None):
        return main_mod.run_benchmark(args, "20260901_100000")


def _args(tree, **kw):
    base = dict(benchmark="training", skip_validation=True, verify_lockfile=None,
                mode="closed", command="run", results_dir=tree, quiet=False)
    base.update(kw)
    return Namespace(**base)


class TestPostRunStatus:
    def test_prints_the_result_of_this_run(self, tree, scripted, capsys):
        scripted([])
        leaf = os.path.join(tree, LEAF_TRAIN_RUN)
        rc = _run_benchmark(_args(tree), _Bench(leaf))
        assert rc == EXIT_CODE.SUCCESS
        out = capsys.readouterr().out
        lines = [l for l in _lines(out) if l]
        assert lines[0].startswith(f"closed/{ORG}")
        rows = [l for l in lines if l.startswith("sys-1")]
        assert len(rows) == 1
        assert rows[0].split()[:6] == ["sys-1", "training", "unet3d", "b200", "1/6", "short"]
        assert "run 3 failed" in rows[0]  # LEAF_TRAIN_RUN_2, exit status 6
        assert any(l.startswith("Next:") for l in lines)
        assert "milvus" not in out  # other results of the tree are not printed

    def test_failed_run_still_gets_the_recap_and_its_exit_code(self, tree, scripted, capsys):
        scripted([])
        leaf = os.path.join(tree, LEAF_TRAIN_RUN_2)
        rc = _run_benchmark(_args(tree), _Bench(leaf, rc=EXIT_CODE.FAILURE))
        assert rc == EXIT_CODE.FAILURE
        assert "unet3d" in capsys.readouterr().out

    def test_silent_under_quiet(self, tree, scripted, capsys):
        scripted([])
        leaf = os.path.join(tree, LEAF_TRAIN_RUN)
        _run_benchmark(_args(tree, quiet=True), _Bench(leaf))
        assert capsys.readouterr().out == ""

    def test_silent_for_non_run_commands(self, tree, scripted, capsys):
        scripted([])
        leaf = os.path.join(tree, LEAF_TRAIN_RUN)
        _run_benchmark(_args(tree, command="datagen"), _Bench(leaf))
        assert capsys.readouterr().out == ""

    def test_evaluation_failure_never_changes_the_exit_code(self, tree, capsys, caplog, monkeypatch):
        import logging
        import mlpstorage_py.readiness as readiness

        def boom(_rd):
            raise RuntimeError("checker exploded")
        monkeypatch.setattr(readiness, "collect_findings", boom)
        leaf = os.path.join(tree, LEAF_TRAIN_RUN)
        with caplog.at_level(logging.WARNING, logger="MLPerfStorage"):
            rc = _run_benchmark(_args(tree), _Bench(leaf))
        assert rc == EXIT_CODE.SUCCESS
        assert capsys.readouterr().out == ""
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "readiness" in text.lower() and "checker exploded" in text

    def test_no_results_dir_on_args_is_a_no_op(self, tmp_path, capsys):
        args = Namespace(benchmark="training", skip_validation=True, verify_lockfile=None,
                         mode="closed", command="run")
        rc = _run_benchmark(args, _Bench(str(tmp_path)))
        assert rc == EXIT_CODE.SUCCESS
        assert capsys.readouterr().out == ""


# ===========================================================================
# runs list: per-run SUBMIT token
# ===========================================================================

class TestRunsListToken:
    def test_submit_column(self, tree, scripted, capsys):
        scripted([])
        assert _main(["runs", "list"]) == EXIT_CODE.SUCCESS
        lines = _lines(capsys.readouterr().out)
        header = lines[0].split()
        assert header[:3] == ["ID", "STATUS", "SUBMIT"]
        by_id = {l.split()[0]: l.split() for l in lines[1:] if l}
        assert by_id["1"][1:3] == ["complete", "-"]          # datagen: no result
        assert by_id["2"][1:3] == ["complete", "ok"]
        assert by_id["3"][1:3] == ["failed", "failed"]
        assert by_id["4"][1:3] == ["complete", "extra"]      # open checkpointing, no phase counts in its metadata
        assert by_id["5"][1:3] == ["complete", "ok"]
        assert by_id["6"][1:3] == ["incomplete", "-"]        # whatif: never packaged

    def test_json_key(self, tree, scripted, capsys):
        scripted([])
        assert _main(["runs", "list", "--json"]) == EXIT_CODE.SUCCESS
        data = {d["id"]: d for d in json.loads(capsys.readouterr().out)}
        assert data[2]["submit"] == "ok" and data[3]["submit"] == "failed"
        assert data[1]["submit"] is None   # datagen leaf: no result
        assert data[6]["submit"] == "-"    # whatif run: never packaged

    def test_extra_and_invalid_tokens(self, tree, scripted, capsys):
        from tests.unit.test_runs_management import _make_leaf
        scripted([("error", "3.4.1", "trainingAU", os.path.join(tree, LEAF_TRAIN_RUN), "AU below 90%")])
        for i in range(7):
            _make_leaf(tree, f"closed/{ORG}/results/sys-1/training/unet3d/run/2026091{i}_100000",
                       exit_status=0, pointer=HASH_A)
        _main(["runs", "list", "--json", "--model", "unet3d", "--status", "complete"])
        data = {d["leaf"]: d["submit"] for d in json.loads(capsys.readouterr().out)}
        assert data[LEAF_TRAIN_RUN] == "invalid"
        assert [v for v in data.values() if v].count("extra") == 1  # 7 ok runs, 6 counted

    def test_datagen_only_tree_shows_dash_not_question_mark(self, tmp_path, xdg, scripted, capsys):
        from tests.unit.test_runs_management import LEAF_TRAIN_DATAGEN, _make_leaf
        from mlpstorage_py.results_dir.user_config import record_results_dir
        rd = str(tmp_path / "dg")
        os.makedirs(rd)
        write_sentinel(rd, ORG)
        record_results_dir(rd)
        _make_leaf(rd, LEAF_TRAIN_DATAGEN, summary=False, pointer=HASH_A)
        scripted([])
        assert _main(["runs", "list"]) == EXIT_CODE.SUCCESS
        lines = [l for l in _lines(capsys.readouterr().out) if l]
        assert len(lines) == 2 and lines[1].split()[2] == "-"

    def test_evaluator_failure_leaves_the_listing_usable(self, tree, capsys, caplog, monkeypatch):
        import logging
        import mlpstorage_py.readiness as readiness

        def boom(_rd):
            raise RuntimeError("checker exploded")
        monkeypatch.setattr(readiness, "collect_findings", boom)
        with caplog.at_level(logging.WARNING, logger="MLPerfStorage"):
            assert _main(["runs", "list"]) == EXIT_CODE.SUCCESS
        lines = _lines(capsys.readouterr().out)
        assert len([l for l in lines if l]) == 7
        assert all(l.split()[2] == "?" for l in lines[1:] if l)
        assert "checker exploded" in " ".join(r.getMessage() for r in caplog.records)


class TestReadinessHelpers:
    def test_run_tokens(self, two_results):
        from mlpstorage_py.readiness import evaluate, run_tokens
        tokens = run_tokens(evaluate(two_results))
        assert len(tokens) == 9
        assert tokens[f"closed/{ORG}/results/{SYS}/training/retinanet/run/{_ts(5)}"] == "failed"
        assert set(tokens.values()) == {"ok", "failed"}

    def test_run_tokens_whatif_is_dash(self, rd, scripted):
        from mlpstorage_py.readiness import evaluate, run_tokens
        _training(rd, "unet3d", 1, division="whatif")
        scripted([])
        assert list(run_tokens(evaluate(rd)).values()) == ["-"]

    def test_same_reason_runs_share_one_note_clause(self, rd, scripted):
        from mlpstorage_py.readiness import evaluate
        _training(rd, "unet3d", 6)
        _system(rd, "closed", SYS)
        leaves = [f"{rd}/closed/{ORG}/results/{SYS}/training/unet3d/run/{_ts(i)}" for i in range(6)]
        scripted([("error", "2.1.19", "trainingRunFiles", leaf, "training_run.stdout.log not found")
                  for leaf in leaves[:3]])
        row = evaluate(rd).results[0]
        assert row.submit == "short" and row.have == 3
        assert row.note.count("invalid") == 1
        assert "runs 1, 2, 3 invalid: [2.1.19] training_run.stdout.log not found; remove or redo them (mlpstorage runs rm 1 2 3)" in row.note

    def test_leaf_key(self, two_results):
        from mlpstorage_py.readiness import leaf_key
        rel = f"closed/{ORG}/results/{SYS}/training/unet3d/run/{_ts(0)}"
        assert leaf_key(two_results, os.path.join(two_results, rel)) == rel
        assert leaf_key(two_results, os.path.join(two_results, rel, "summary.json")) == rel
        assert leaf_key(two_results, two_results) is None
        assert leaf_key(two_results, "/elsewhere") is None


# ===========================================================================
# Help surface
# ===========================================================================

class TestHelpSurface:
    def test_help_all_has_a_status_block_and_tree_entry(self):
        from mlpstorage_py.cli.help_formatter import HELP_ALL_TEXT
        assert "\nSTATUS\n" in HELP_ALL_TEXT
        assert "├── status" in HELP_ALL_TEXT
        assert "mlpstorage status [OPTIONS]" in HELP_ALL_TEXT

    def test_parity_maps_the_leaf(self):
        from tests.unit.test_help_all_parity import _block_for
        assert _block_for(("status",)) == "STATUS"

    def test_context_tokens(self):
        from mlpstorage_py.cli.help_formatter import get_context_help_tokens
        assert "status" in get_context_help_tokens([])
        assert get_context_help_tokens(["status"]) is None
