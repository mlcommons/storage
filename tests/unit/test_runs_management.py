"""
``mlpstorage runs`` — run management inside a results-dir.

Covers the run ledger (``<results-dir>/.mlps/runs.jsonl``), the status
derivation for a run leaf, the ``runs list|show|rm|purge|gc`` commands
end to end through ``main()``, and the two hooks in the benchmark path
(ledger registration at leaf reservation, ``exit_status`` in metadata).
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


# ---------------------------------------------------------------------------
# Fixture tree builders
# ---------------------------------------------------------------------------

ORG = "Acme"
FULL_HASH_A = "a" * 32
FULL_HASH_B = "b" * 32

LEAF_TRAIN_RUN = f"closed/{ORG}/results/sys-1/training/unet3d/run/20260901_100000"
LEAF_TRAIN_RUN_2 = f"closed/{ORG}/results/sys-1/training/unet3d/run/20260902_100000"
LEAF_TRAIN_DATAGEN = f"closed/{ORG}/results/sys-1/training/unet3d/datagen/20260831_090000"
LEAF_CKPT = f"open/{ORG}/results/sys-2/checkpointing/llama3-8b/20260903_110000"
LEAF_VDB = f"closed/{ORG}/results/sys-1/vector_database/milvus/DISKANN/run/20260904_120000"
LEAF_KV = f"whatif/{ORG}/results/sys-1/kv_cache/llama3.1-8b/run/20260905_130000"


def _make_leaf(rd, rel, *, metadata=True, summary=True, exit_status=None,
               pointer=None, extra_files=()):
    leaf = os.path.join(rd, rel)
    os.makedirs(leaf, exist_ok=True)
    ts = rel.rsplit("/", 1)[1]
    if metadata:
        body = {"run_datetime": ts, "command": "run", "runtime": 12.5,
                "executed_command": "mpirun ... dlio_benchmark",
                "num_processes": 8, "accelerator": "b200"}
        if exit_status is not None:
            body["exit_status"] = exit_status
        with open(os.path.join(leaf, f"training_{ts}_metadata.json"), "w") as fh:
            json.dump(body, fh)
    if summary:
        with open(os.path.join(leaf, "summary.json"), "w") as fh:
            json.dump({"metric": {"train_au_percentage": [95.0]}}, fh)
    if pointer:
        with open(os.path.join(leaf, ".mlps-code-image"), "w") as fh:
            fh.write(f"md5-tree-v2:{pointer}")
    for name in extra_files:
        with open(os.path.join(leaf, name), "w") as fh:
            fh.write("x" * 1024)
    return leaf


def _make_pool_image(rd, full_hash, org=ORG):
    pool_root = os.path.join(rd, org)
    os.makedirs(pool_root, exist_ok=True)
    with open(os.path.join(pool_root, ".mlps-image-pool"), "w") as fh:
        fh.write("version=1\ncreated=2026-09-01T00:00:00Z\n")
    image = os.path.join(pool_root, f"code-{full_hash[:8]}")
    os.makedirs(image, exist_ok=True)
    with open(os.path.join(image, ".code-hash.json"), "w") as fh:
        json.dump({"hash": full_hash, "algorithm": "md5-tree-v2",
                   "captured_at": "2026-09-01T00:00:00Z",
                   "mlpstorage_version": "3.0.46", "git_sha": None}, fh)
    with open(os.path.join(image, "setup.py"), "w") as fh:
        fh.write("# image\n")
    return image


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    cfg = tmp_path / "xdg"
    cfg.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg))
    monkeypatch.delenv("MLPSTORAGE_RESULTS_DIR", raising=False)
    return cfg


@pytest.fixture
def tree(tmp_path, xdg):
    """An initialized results-dir with six runs across every layout, two
    pool images (one orphan), and the tree recorded as the user default."""
    from mlpstorage_py.results_dir.user_config import record_results_dir

    rd = str(tmp_path / "results")
    os.makedirs(rd)
    write_sentinel(rd, ORG)
    record_results_dir(rd)
    _make_leaf(rd, LEAF_TRAIN_DATAGEN, summary=False, pointer=FULL_HASH_A)
    _make_leaf(rd, LEAF_TRAIN_RUN, exit_status=0, pointer=FULL_HASH_A,
               extra_files=("0_output.json",))
    _make_leaf(rd, LEAF_TRAIN_RUN_2, summary=False, exit_status=6, pointer=FULL_HASH_A)
    _make_leaf(rd, LEAF_CKPT, exit_status=0, pointer=FULL_HASH_A)
    _make_leaf(rd, LEAF_VDB, exit_status=0, pointer=FULL_HASH_A)
    _make_leaf(rd, LEAF_KV, metadata=False, summary=False, pointer=FULL_HASH_A)
    _make_pool_image(rd, FULL_HASH_A)
    _make_pool_image(rd, FULL_HASH_B)  # orphan: no leaf points at it
    return rd


def _main(argv):
    from mlpstorage_py import main as main_mod

    with patch("sys.argv", ["mlpstorage"] + list(argv)), \
         patch.object(main_mod, "apply_logging_options"):
        return main_mod.main()


# ---------------------------------------------------------------------------
# Leaf path parsing and discovery
# ---------------------------------------------------------------------------

class TestLeafParsing:
    def test_training_leaf(self):
        from mlpstorage_py.runs.ledger import parse_leaf

        info = parse_leaf(LEAF_TRAIN_RUN)
        assert info == {
            "mode": "closed", "orgname": ORG, "systemname": "sys-1",
            "benchmark": "training", "model": "unet3d", "command": "run",
            "run_datetime": "20260901_100000",
        }

    def test_checkpointing_leaf_has_no_command_segment(self):
        from mlpstorage_py.runs.ledger import parse_leaf

        info = parse_leaf(LEAF_CKPT)
        assert info["benchmark"] == "checkpointing"
        assert info["model"] == "llama3-8b"
        assert info["command"] == "run"
        assert info["mode"] == "open"
        assert info["systemname"] == "sys-2"

    def test_vector_database_leaf_model_is_engine_and_index(self):
        from mlpstorage_py.runs.ledger import parse_leaf

        info = parse_leaf(LEAF_VDB)
        assert info["benchmark"] == "vector_database"
        assert info["model"] == "milvus/DISKANN"
        assert info["command"] == "run"

    def test_kv_cache_leaf(self):
        from mlpstorage_py.runs.ledger import parse_leaf

        info = parse_leaf(LEAF_KV)
        assert info["benchmark"] == "kv_cache"
        assert info["model"] == "llama3.1-8b"
        assert info["mode"] == "whatif"

    @pytest.mark.parametrize("rel", [
        "closed/Acme/results/sys-1/training/unet3d/run",          # no timestamp
        "closed/Acme/code-aaaaaaaa",                               # pool image
        ".mlps/trash/20260910_000000/" + LEAF_TRAIN_RUN,           # trashed
        "systems/sys-1.yaml",
        "closed/Acme/results/sys-1/training/unet3d/run/20260901_100000/0_output.json",
    ])
    def test_non_canonical_paths_are_rejected(self, rel):
        from mlpstorage_py.runs.ledger import parse_leaf

        assert parse_leaf(rel) is None

    def test_iter_leaves_finds_every_layout_and_nothing_else(self, tree):
        from mlpstorage_py.runs.ledger import iter_leaves

        found = iter_leaves(tree)
        assert sorted(found) == sorted([
            LEAF_TRAIN_DATAGEN, LEAF_TRAIN_RUN, LEAF_TRAIN_RUN_2,
            LEAF_CKPT, LEAF_VDB, LEAF_KV,
        ])

    def test_iter_leaves_ignores_trash_and_pool(self, tree):
        from mlpstorage_py.runs.ledger import iter_leaves

        trashed = os.path.join(tree, ".mlps", "trash", "20260910_000000", LEAF_TRAIN_RUN)
        os.makedirs(trashed)
        assert LEAF_TRAIN_RUN in iter_leaves(tree)
        assert not any(rel.startswith(".mlps") for rel in iter_leaves(tree))


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

class TestLedger:
    def test_sync_adopts_leaves_with_sequential_ids_in_timestamp_order(self, tree):
        from mlpstorage_py.runs.ledger import ledger_path, sync

        records = sync(tree)
        assert [r.id for r in records] == [1, 2, 3, 4, 5, 6]
        assert [r.leaf for r in records] == [
            LEAF_TRAIN_DATAGEN, LEAF_TRAIN_RUN, LEAF_TRAIN_RUN_2,
            LEAF_CKPT, LEAF_VDB, LEAF_KV,
        ]
        assert os.path.isfile(ledger_path(tree))
        assert ledger_path(tree) == os.path.join(tree, ".mlps", "runs.jsonl")

    def test_ids_are_stable_across_syncs(self, tree):
        from mlpstorage_py.runs.ledger import sync

        first = {r.leaf: r.id for r in sync(tree)}
        second = {r.leaf: r.id for r in sync(tree)}
        assert first == second

    def test_hand_deleted_leaf_disappears_and_its_id_is_never_reused(self, tree):
        import shutil

        from mlpstorage_py.runs.ledger import sync

        sync(tree)
        shutil.rmtree(os.path.join(tree, LEAF_TRAIN_DATAGEN))
        after = sync(tree)
        assert 1 not in [r.id for r in after]
        new_leaf = f"closed/{ORG}/results/sys-1/training/unet3d/run/20260906_100000"
        _make_leaf(tree, new_leaf, exit_status=0)
        again = {r.leaf: r.id for r in sync(tree)}
        assert again[new_leaf] == 7

    def test_register_run_is_idempotent_and_returns_the_id(self, tree):
        from mlpstorage_py.runs.ledger import register_run, sync

        leaf = os.path.join(tree, LEAF_TRAIN_RUN)
        first = register_run(tree, leaf)
        assert first == 1
        assert register_run(tree, leaf) == 1
        # sync adopts the rest after the explicit registration
        ids = {r.leaf: r.id for r in sync(tree)}
        assert ids[LEAF_TRAIN_RUN] == 1
        assert len(ids) == 6

    def test_register_run_ignores_a_non_canonical_leaf(self, tree):
        from mlpstorage_py.runs.ledger import ledger_path, register_run

        stray = os.path.join(tree, "scratch", "20260901_100000")
        os.makedirs(stray)
        assert register_run(tree, stray) is None
        assert not os.path.exists(ledger_path(tree))

    def test_ledger_lines_are_json_events(self, tree):
        from mlpstorage_py.runs.ledger import ledger_path, sync

        sync(tree)
        with open(ledger_path(tree)) as fh:
            events = [json.loads(line) for line in fh if line.strip()]
        assert all(e["event"] == "registered" for e in events)
        assert {e["id"] for e in events} == {1, 2, 3, 4, 5, 6}
        assert all("leaf" in e and "at" in e for e in events)


# ---------------------------------------------------------------------------
# Status derivation
# ---------------------------------------------------------------------------

class TestRunStatus:
    def _status(self, tree, rel):
        from mlpstorage_py.runs.ledger import parse_leaf, run_status

        return run_status(os.path.join(tree, rel), parse_leaf(rel))

    def test_exit_status_zero_is_complete(self, tree):
        assert self._status(tree, LEAF_TRAIN_RUN) == "complete"

    def test_nonzero_exit_status_is_failed(self, tree):
        assert self._status(tree, LEAF_TRAIN_RUN_2) == "failed"

    def test_missing_metadata_is_incomplete(self, tree):
        assert self._status(tree, LEAF_KV) == "incomplete"

    def test_legacy_run_without_summary_is_failed(self, tmp_path):
        rd = str(tmp_path)
        _make_leaf(rd, LEAF_TRAIN_RUN, summary=False)  # no exit_status key
        assert self._status(rd, LEAF_TRAIN_RUN) == "failed"

    def test_legacy_run_with_summary_is_complete(self, tmp_path):
        rd = str(tmp_path)
        _make_leaf(rd, LEAF_TRAIN_RUN)
        assert self._status(rd, LEAF_TRAIN_RUN) == "complete"

    def test_legacy_training_datagen_needs_no_summary(self, tmp_path):
        rd = str(tmp_path)
        _make_leaf(rd, LEAF_TRAIN_DATAGEN, summary=False)
        assert self._status(rd, LEAF_TRAIN_DATAGEN) == "complete"


# ---------------------------------------------------------------------------
# Benchmark-side hooks
# ---------------------------------------------------------------------------

class TestBenchmarkHooks:
    def test_metadata_carries_exit_status(self):
        from mlpstorage_py.benchmarks.base import Benchmark
        from mlpstorage_py.config import BENCHMARK_TYPES

        shim = SimpleNamespace(
            BENCHMARK_TYPE=BENCHMARK_TYPES.training,
            args=Namespace(model="unet3d", command="run", num_processes=1,
                           accelerator_type="b200"),
            run_datetime="20260901_100000",
            run_result_output="/tmp/x",
            runtime=1.0,
            verification=None,
            command_output_files=[],
            exit_status=6,
            _apply_dotted_overrides=Benchmark._apply_dotted_overrides,
        )
        meta = Benchmark.metadata.fget(shim)
        assert meta["exit_status"] == 6

    def test_constructing_a_benchmark_registers_the_leaf(self, tree):
        from mlpstorage_py.benchmarks.base import Benchmark
        from mlpstorage_py.config import BENCHMARK_TYPES
        from mlpstorage_py.runs.ledger import sync

        class _B(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training

            def _run(self):
                return 0

        args = Namespace(
            mode="closed", dry_run=False, orgname=ORG, systemname="sys-1",
            debug=False, verbose=False, what_if=False, stream_log_level="INFO",
            results_dir=tree, model="unet3d", command="run",
            num_processes=1, accelerator_type="b200",
        )
        sync(tree)  # six runs already known
        bench = _B(args, run_datetime="20260907_100000")
        assert bench.run_id == 7
        rel = os.path.relpath(bench.run_result_output, tree)
        assert {r.id: r.leaf for r in sync(tree)}[7] == rel
        assert bench.exit_status is None

    def test_run_benchmark_records_exit_status_before_metadata(self, tmp_path):
        """``run_benchmark`` writes metadata in a ``finally``; the exit
        status of ``benchmark.run()`` must be on the instance by then."""
        from mlpstorage_py import main as main_mod

        seen = {}

        class _Bench:
            metadata_file_path = str(tmp_path / "m.json")
            run_result_output = str(tmp_path)

            def __init__(self, args, run_datetime, logger):
                self.exit_status = None

            def run(self):
                return EXIT_CODE.SUCCESS

            def write_metadata(self):
                seen["exit_status"] = self.exit_status

        fake_benchmarks = SimpleNamespace(
            TrainingBenchmark=_Bench, VectorDBBenchmark=_Bench,
            CheckpointingBenchmark=_Bench, KVCacheBenchmark=_Bench,
        )
        args = Namespace(benchmark="training", skip_validation=True,
                         verify_lockfile=None, mode="closed", command="run")
        with patch.dict("sys.modules", {"mlpstorage_py.benchmarks": fake_benchmarks}), \
             patch.object(main_mod, "_check_and_migrate_legacy_layout"), \
             patch.object(main_mod, "capture_or_verify_code_image"), \
             patch.object(main_mod, "attach_run_log_files", return_value=None):
            rc = main_mod.run_benchmark(args, "20260901_100000")
        assert rc == EXIT_CODE.SUCCESS
        assert seen["exit_status"] == 0


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

class TestRunsParser:
    @pytest.mark.parametrize("argv", [
        ["runs", "list"],
        ["runs", "list", "--json", "--status", "failed", "--mode", "closed",
         "--benchmark", "training", "--model", "unet3d", "--systemname", "sys-1"],
        ["runs", "show", "3"],
        ["runs", "rm", "1", "2", "--yes"],
        ["runs", "rm", "--status", "failed", "--older-than", "7d", "--keep-last", "2", "-y"],
        ["runs", "purge", "--yes"],
        ["runs", "gc", "--yes"],
        ["runs", "list", "--results-dir", "/somewhere"],
        ["runs", "gc", "-rd", "/somewhere"],
    ])
    def test_subcommands_parse(self, argv, xdg):
        from mlpstorage_py.cli_parser import build_parser

        args = build_parser().parse_args(argv)
        assert args.mode == "runs"
        assert args.command == argv[1]

    def test_results_dir_flag_is_labelled_as_the_flag(self, xdg):
        from mlpstorage_py.cli_parser import parse_arguments

        with patch("sys.argv", ["mlpstorage", "runs", "list", "-rd", "/somewhere"]):
            args = parse_arguments()
        assert args.results_dir == "/somewhere"
        assert args.results_dir_source == "--results-dir"

    def test_context_help_lists_runs_subcommands(self):
        from mlpstorage_py.cli.help_formatter import get_context_help_tokens

        assert get_context_help_tokens(["runs"]) == "next: list | show | rm | purge | gc"
        assert get_context_help_tokens(["runs", "list"]) is None
        assert " runs " in get_context_help_tokens([])

    def test_help_all_documents_runs(self):
        from mlpstorage_py.cli.help_formatter import HELP_ALL_TEXT, SYNOPSIS_TEXT

        for block in ("RUNS_LIST", "RUNS_SHOW", "RUNS_RM", "RUNS_PURGE", "RUNS_GC"):
            assert f"\n{block}\n" in HELP_ALL_TEXT
        assert "runs" in SYNOPSIS_TEXT


# ---------------------------------------------------------------------------
# End to end through main()
# ---------------------------------------------------------------------------

class TestRunsList:
    def test_lists_every_run_with_id_and_status(self, tree, capsys):
        assert _main(["runs", "list"]) == EXIT_CODE.SUCCESS
        out = capsys.readouterr().out
        rows = [line for line in out.splitlines() if line.strip()]
        assert len(rows) == 7  # header + six runs
        assert rows[0].split()[0] == "ID"
        assert "complete" in out and "failed" in out and "incomplete" in out
        assert "milvus/DISKANN" in out
        assert "code-aaaaaaaa" in out

    def test_json_output(self, tree, capsys):
        assert _main(["runs", "list", "--json"]) == EXIT_CODE.SUCCESS
        data = json.loads(capsys.readouterr().out)
        assert [d["id"] for d in data] == [1, 2, 3, 4, 5, 6]
        by_id = {d["id"]: d for d in data}
        assert by_id[3]["status"] == "failed"
        assert by_id[6]["status"] == "incomplete"
        assert by_id[4]["mode"] == "open"
        assert by_id[4]["systemname"] == "sys-2"
        assert by_id[5]["model"] == "milvus/DISKANN"
        assert by_id[2]["code_image"] == "code-aaaaaaaa"
        assert by_id[2]["leaf"] == LEAF_TRAIN_RUN
        assert by_id[2]["size_bytes"] > 0

    @pytest.mark.parametrize("flags, expected_ids", [
        (["--status", "failed"], [3]),
        (["--mode", "open"], [4]),
        (["--benchmark", "training"], [1, 2, 3]),
        (["--benchmark", "vectordb"], [5]),
        (["--benchmark", "kvcache"], [6]),
        (["--model", "unet3d", "--status", "complete"], [1, 2]),
        (["--systemname", "sys-2"], [4]),
    ])
    def test_filters(self, tree, capsys, flags, expected_ids):
        assert _main(["runs", "list", "--json"] + flags) == EXIT_CODE.SUCCESS
        data = json.loads(capsys.readouterr().out)
        assert [d["id"] for d in data] == expected_ids

    def test_empty_tree_says_so(self, tmp_path, xdg, capsys):
        from mlpstorage_py.results_dir.user_config import record_results_dir

        rd = str(tmp_path / "empty")
        os.makedirs(rd)
        write_sentinel(rd, ORG)
        record_results_dir(rd)
        assert _main(["runs", "list"]) == EXIT_CODE.SUCCESS
        assert "no runs" in capsys.readouterr().out.lower()

    def test_runs_commands_are_not_recorded_in_history(self, tree):
        from mlpstorage_py.history import history_file_for

        _main(["runs", "list"])
        hist = history_file_for(tree)
        assert not os.path.exists(hist) or open(hist).read().strip() == ""

    def test_no_results_dir_anywhere_is_actionable(self, xdg, caplog):
        import logging

        with caplog.at_level(logging.ERROR, logger="MLPerfStorage"):
            rc = _main(["runs", "list"])
        assert rc != EXIT_CODE.SUCCESS
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "mlpstorage init" in text

    def test_uninitialized_results_dir_is_refused(self, tmp_path, xdg, caplog):
        import logging

        rd = tmp_path / "bare"
        rd.mkdir()
        with caplog.at_level(logging.ERROR, logger="MLPerfStorage"):
            rc = _main(["runs", "list", "-rd", str(rd)])
        assert rc != EXIT_CODE.SUCCESS
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "has not been initialized" in text
        assert not os.path.exists(rd / ".mlps")


class TestRunsShow:
    def test_show_prints_leaf_metadata_and_pointer(self, tree, capsys):
        assert _main(["runs", "show", "2"]) == EXIT_CODE.SUCCESS
        out = capsys.readouterr().out
        assert LEAF_TRAIN_RUN in out
        assert "complete" in out
        assert "code-aaaaaaaa" in out
        assert "0_output.json" in out
        assert "summary.json" in out
        assert "mpirun ... dlio_benchmark" in out

    def test_show_flags_a_dangling_pointer(self, tree, capsys):
        _make_leaf(tree, f"closed/{ORG}/results/sys-1/training/unet3d/run/20260908_100000",
                   exit_status=0, pointer="c" * 32)
        assert _main(["runs", "show", "7"]) == EXIT_CODE.SUCCESS
        out = capsys.readouterr().out
        assert "code-cccccccc" in out
        assert "missing" in out.lower()

    def test_unknown_id_is_an_error(self, tree, capsys):
        assert _main(["runs", "show", "99"]) == EXIT_CODE.INVALID_ARGUMENTS


class TestRunsRm:
    def _trash_entries(self, tree):
        root = os.path.join(tree, ".mlps", "trash")
        if not os.path.isdir(root):
            return []
        found = []
        for batch in os.listdir(root):
            for dirpath, dirnames, _ in os.walk(os.path.join(root, batch)):
                for d in dirnames:
                    rel = os.path.relpath(os.path.join(dirpath, d), os.path.join(root, batch))
                    if rel.count("/") >= 4 and d[:8].isdigit():
                        found.append(rel)
        return found

    def test_rm_moves_the_leaf_into_trash(self, tree, capsys):
        assert _main(["runs", "rm", "2", "--yes"]) == EXIT_CODE.SUCCESS
        assert not os.path.exists(os.path.join(tree, LEAF_TRAIN_RUN))
        assert self._trash_entries(tree) == [LEAF_TRAIN_RUN]
        # the moved leaf is intact
        batch = os.listdir(os.path.join(tree, ".mlps", "trash"))[0]
        moved = os.path.join(tree, ".mlps", "trash", batch, LEAF_TRAIN_RUN)
        assert os.path.isfile(os.path.join(moved, "summary.json"))
        # sentinel and pool untouched
        assert os.path.isfile(os.path.join(tree, "mlperf-results.yaml"))
        assert os.path.isdir(os.path.join(tree, ORG, "code-aaaaaaaa"))
        out = capsys.readouterr().out
        assert "purge" in out

    def test_rm_hides_the_run_and_keeps_its_id_retired(self, tree, capsys):
        _main(["runs", "rm", "2", "--yes"])
        capsys.readouterr()
        _main(["runs", "list", "--json"])
        ids = [d["id"] for d in json.loads(capsys.readouterr().out)]
        assert ids == [1, 3, 4, 5, 6]

    def test_rm_records_the_move_in_the_ledger(self, tree):
        from mlpstorage_py.runs.ledger import ledger_path

        _main(["runs", "rm", "2", "--yes"])
        with open(ledger_path(tree)) as fh:
            events = [json.loads(l) for l in fh if l.strip()]
        trashed = [e for e in events if e["event"] == "trashed"]
        assert len(trashed) == 1
        assert trashed[0]["id"] == 2
        assert trashed[0]["to"].startswith(".mlps/trash/")

    def test_rm_without_yes_and_no_tty_moves_nothing(self, tree, caplog):
        import logging

        with caplog.at_level(logging.ERROR, logger="MLPerfStorage"):
            rc = _main(["runs", "rm", "2"])
        assert rc == EXIT_CODE.INVALID_ARGUMENTS
        assert os.path.isdir(os.path.join(tree, LEAF_TRAIN_RUN))
        assert self._trash_entries(tree) == []
        assert "--yes" in " ".join(r.getMessage() for r in caplog.records)

    def test_rm_by_status(self, tree):
        assert _main(["runs", "rm", "--status", "failed", "--yes"]) == EXIT_CODE.SUCCESS
        assert not os.path.exists(os.path.join(tree, LEAF_TRAIN_RUN_2))
        assert os.path.isdir(os.path.join(tree, LEAF_TRAIN_RUN))
        assert self._trash_entries(tree) == [LEAF_TRAIN_RUN_2]

    def test_rm_by_status_incomplete(self, tree):
        assert _main(["runs", "rm", "--status", "incomplete", "--yes"]) == EXIT_CODE.SUCCESS
        assert self._trash_entries(tree) == [LEAF_KV]

    def test_rm_older_than_with_keep_last(self, tree):
        # every run is older than "today"; keep the newest two
        assert _main(["runs", "rm", "--older-than", "1d", "--keep-last", "2", "--yes"]) == EXIT_CODE.SUCCESS
        remaining = sorted(os.path.relpath(p, tree) for p in [
            os.path.join(tree, r) for r in
            (LEAF_TRAIN_DATAGEN, LEAF_TRAIN_RUN, LEAF_TRAIN_RUN_2, LEAF_CKPT, LEAF_VDB, LEAF_KV)
        ] if os.path.isdir(p))
        assert remaining == sorted([LEAF_VDB, LEAF_KV])

    def test_rm_older_than_absolute_date(self, tree):
        assert _main(["runs", "rm", "--older-than", "2026-09-02", "--yes"]) == EXIT_CODE.SUCCESS
        assert sorted(self._trash_entries(tree)) == sorted([LEAF_TRAIN_DATAGEN, LEAF_TRAIN_RUN])

    def test_rm_with_no_selection_is_an_error(self, tree):
        assert _main(["runs", "rm", "--yes"]) == EXIT_CODE.INVALID_ARGUMENTS

    def test_rm_unknown_id_is_an_error_and_moves_nothing(self, tree):
        assert _main(["runs", "rm", "2", "99", "--yes"]) == EXIT_CODE.INVALID_ARGUMENTS
        assert os.path.isdir(os.path.join(tree, LEAF_TRAIN_RUN))

    def test_rm_warns_about_stale_rollups(self, tree, caplog):
        import logging

        model_dir = os.path.join(tree, f"closed/{ORG}/results/sys-1/training/unet3d")
        with open(os.path.join(model_dir, "results.json"), "w") as fh:
            fh.write("{}")
        with caplog.at_level(logging.WARNING, logger="MLPerfStorage"):
            _main(["runs", "rm", "2", "--yes"])
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "results.json" in text and "reportgen" in text


class TestRunsPurge:
    def test_purge_empties_the_trash(self, tree, capsys):
        _main(["runs", "rm", "2", "3", "--yes"])
        trash = os.path.join(tree, ".mlps", "trash")
        assert os.listdir(trash)
        assert _main(["runs", "purge", "--yes"]) == EXIT_CODE.SUCCESS
        assert not os.path.isdir(trash) or os.listdir(trash) == []
        assert os.path.isfile(os.path.join(tree, "mlperf-results.yaml"))

    def test_purge_without_yes_and_no_tty_deletes_nothing(self, tree):
        _main(["runs", "rm", "2", "--yes"])
        assert _main(["runs", "purge"]) == EXIT_CODE.INVALID_ARGUMENTS
        assert os.listdir(os.path.join(tree, ".mlps", "trash"))

    def test_purge_of_empty_trash_is_fine(self, tree, capsys):
        assert _main(["runs", "purge", "--yes"]) == EXIT_CODE.SUCCESS


class TestRunsGc:
    def test_gc_trashes_orphan_images_only(self, tree, capsys):
        assert _main(["runs", "gc", "--yes"]) == EXIT_CODE.SUCCESS
        assert not os.path.isdir(os.path.join(tree, ORG, "code-bbbbbbbb"))
        assert os.path.isdir(os.path.join(tree, ORG, "code-aaaaaaaa"))
        assert os.path.isfile(os.path.join(tree, ORG, ".mlps-image-pool"))
        batch = os.listdir(os.path.join(tree, ".mlps", "trash"))[0]
        assert os.path.isfile(os.path.join(
            tree, ".mlps", "trash", batch, ORG, "code-bbbbbbbb", ".code-hash.json"))
        assert "code-bbbbbbbb" in capsys.readouterr().out

    def test_gc_keeps_images_referenced_only_from_trash(self, tree):
        # every leaf pointing at A goes to the trash; A must survive gc
        _main(["runs", "rm", "1", "2", "3", "4", "5", "6", "--yes"])
        assert _main(["runs", "gc", "--yes"]) == EXIT_CODE.SUCCESS
        assert os.path.isdir(os.path.join(tree, ORG, "code-aaaaaaaa"))
        assert not os.path.isdir(os.path.join(tree, ORG, "code-bbbbbbbb"))

    def test_gc_without_yes_and_no_tty_moves_nothing(self, tree):
        assert _main(["runs", "gc"]) == EXIT_CODE.INVALID_ARGUMENTS
        assert os.path.isdir(os.path.join(tree, ORG, "code-bbbbbbbb"))

    def test_gc_with_nothing_to_collect(self, tree, capsys):
        _main(["runs", "gc", "--yes"])
        capsys.readouterr()
        assert _main(["runs", "gc", "--yes"]) == EXIT_CODE.SUCCESS
        assert "nothing" in capsys.readouterr().out.lower()
