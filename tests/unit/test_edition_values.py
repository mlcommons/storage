"""Remaining edition-varying values move into the rules editions table
(design §6, D-16 / D-17 of .planning/rules-editions-and-comparability-classes.md).

Five value blocks join the ``checker:`` of every checkable edition in
``mlpstorage_py/rules/editions.yaml``: the Rules.md 3.3.2 AU minimums per
training model, the Table 2 CLOSED process counts and checkpoint sizes, the
Table 3 simulated-accelerator memories and the 6.3.2.1 KVCache sequence
locks. ``workloads:`` of edition 3.0 gains an ``open:`` division. The
submission checker reads them through its per-submission ``Config`` (so a
future edition can change a value without touching a neighbour), the runtime
reads the current edition's block, and the constants they replace are gone.

Covered here:
- the table: 3.0 carries the five blocks with the former constants' values,
  historical editions carry none, the model / accelerator keys must agree with
  ``workloads:``, malformed blocks are rejected;
- Edition / table helpers: models and accelerators per division, the
  tree-wide vocabulary as the union over checkable editions;
- constants: the seven retired names are gone, the kept ones stay in lockstep;
- ``Config``: getters read the edition's block (and follow a different table);
- submission checks 4.3.4 / 4.3.3 / 3.3.2 / 2.1.10 / 2.1.11 / 2.1.21 read the table;
- runtime: checkpointing run checker, KVCache sequence locks, CLI choices and
  defaults, ``--help_all`` tree, datagen allowlist, run summary;
- docs name the new contents of the block.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from mlpstorage_py.config import RULES_EDITION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDITIONS_YAML = PROJECT_ROOT / "mlpstorage_py" / "rules" / "editions.yaml"

V30_VALUES = {
    "training_au_thresholds": {"unet3d": 0.90, "retinanet": 0.85},
    "closed_mpi_processes": {"llama3-8b": 8, "llama3-70b": 64, "llama3-405b": 512, "llama3-1t": 1024},
    "checkpoint_size_gb": {"llama3-8b": 105, "llama3-70b": 912, "llama3-405b": 5290, "llama3-1t": 18000},
    "accelerator_memory_gb": {"b200": 180, "mi355": 288, "h100": 80, "a100": 80},
    "kvcache_closed_sequence": {"seed": 42, "trials": 3, "inter_option_delay_s": 90},
}
LLAMA = ["llama3-8b", "llama3-70b", "llama3-405b", "llama3-1t"]


def _table():
    from mlpstorage_py.editions import load_editions
    return load_editions()


def _write_table(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load(EDITIONS_YAML.read_text())
    mutate(data)
    p = tmp_path / "editions.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def _table_with(tmp_path: Path, mutate):
    from mlpstorage_py.editions import load_editions
    return load_editions(_write_table(tmp_path, mutate))


def _blocks(tmp_path: Path, **blocks):
    """A table whose 3.0 checker blocks are REPLACED by ``blocks``."""
    def mutate(data):
        data["editions"]["3.0"]["checker"].update(blocks)
    return _table_with(tmp_path, mutate)


def _values(tmp_path: Path, **changes):
    """A table whose 3.0 checker block carries ``changes`` (deep-merged into
    the shipped values, so a partial mapping keeps the other keys)."""
    def mutate(data):
        checker = data["editions"]["3.0"]["checker"]
        for k, v in changes.items():
            if isinstance(v, dict) and isinstance(checker.get(k), dict):
                checker[k] = {**checker[k], **v}
            else:
                checker[k] = v
    return _table_with(tmp_path, mutate)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


class TestTable:
    def test_current_edition_carries_the_five_value_blocks(self):
        checker = _table().editions[RULES_EDITION].checker
        for name, expected in V30_VALUES.items():
            assert getattr(checker, name) == expected, name

    def test_the_five_names_are_checker_fields(self):
        from mlpstorage_py.editions import CHECKER_FIELDS, CHECKER_VALUE_FIELDS, CheckerParameters
        assert set(CHECKER_VALUE_FIELDS) == set(V30_VALUES)
        assert set(CHECKER_VALUE_FIELDS) <= set(CHECKER_FIELDS)
        assert set(CHECKER_FIELDS) <= {f.name for f in dataclasses.fields(CheckerParameters)}

    def test_historical_editions_carry_no_values(self):
        table = _table()
        for eid in ("0.5", "1.0", "2.0"):
            assert table.editions[eid].checker is None, eid

    def test_30_open_division_mirrors_closed(self):
        e = _table().editions["3.0"]
        assert e.workloads["open"] == e.workloads["closed"]

    @pytest.mark.parametrize("name", sorted(V30_VALUES))
    def test_a_value_block_is_required(self, tmp_path, name):
        from mlpstorage_py.editions import EditionsError

        def drop(data):
            del data["editions"]["3.0"]["checker"][name]
        with pytest.raises(EditionsError, match=name):
            _table_with(tmp_path, drop)

    def test_au_thresholds_must_name_exactly_the_training_models(self, tmp_path):
        from mlpstorage_py.editions import EditionsError
        with pytest.raises(EditionsError, match="training_au_thresholds"):
            _values(tmp_path, training_au_thresholds={"cosmoflow": 0.7})  # an extra model
        with pytest.raises(EditionsError, match="training_au_thresholds"):
            _blocks(tmp_path, training_au_thresholds={"unet3d": 0.9})  # a missing model

    def test_au_threshold_is_a_fraction(self, tmp_path):
        from mlpstorage_py.editions import EditionsError
        with pytest.raises(EditionsError, match="training_au_thresholds"):
            _values(tmp_path, training_au_thresholds={"unet3d": 90})
        with pytest.raises(EditionsError, match="training_au_thresholds"):
            _values(tmp_path, training_au_thresholds={"unet3d": 0})

    @pytest.mark.parametrize("name", ["closed_mpi_processes", "checkpoint_size_gb"])
    def test_checkpoint_blocks_must_name_exactly_the_checkpointing_models(self, tmp_path, name):
        from mlpstorage_py.editions import EditionsError
        with pytest.raises(EditionsError, match=name):
            _blocks(tmp_path, **{name: {"llama3-8b": 8}})  # missing models
        with pytest.raises(EditionsError, match=name):
            _values(tmp_path, **{name: {"llama4-1b": 1}})  # an extra model

    def test_closed_mpi_processes_are_positive_integers(self, tmp_path):
        from mlpstorage_py.editions import EditionsError
        with pytest.raises(EditionsError, match="closed_mpi_processes"):
            _values(tmp_path, closed_mpi_processes={"llama3-8b": 8.5})
        with pytest.raises(EditionsError, match="closed_mpi_processes"):
            _values(tmp_path, closed_mpi_processes={"llama3-8b": 0})

    def test_checkpoint_size_is_positive(self, tmp_path):
        from mlpstorage_py.editions import EditionsError
        with pytest.raises(EditionsError, match="checkpoint_size_gb"):
            _values(tmp_path, checkpoint_size_gb={"llama3-8b": -1})

    def test_accelerator_memory_must_cover_every_workload_accelerator(self, tmp_path):
        from mlpstorage_py.editions import EditionsError

        def drop(data):
            del data["editions"]["3.0"]["checker"]["accelerator_memory_gb"]["mi355"]
        with pytest.raises(EditionsError, match="accelerator_memory_gb"):
            _table_with(tmp_path, drop)
        with pytest.raises(EditionsError, match="accelerator_memory_gb"):
            _values(tmp_path, accelerator_memory_gb={"b200": 0})
        # An accelerator no workload lists (h100 in 3.0) is still allowed.
        assert _table().editions["3.0"].checker.accelerator_memory_gb["h100"] == 80

    def test_kvcache_sequence_has_exactly_three_keys(self, tmp_path):
        from mlpstorage_py.editions import EditionsError
        with pytest.raises(EditionsError, match="kvcache_closed_sequence"):
            _blocks(tmp_path, kvcache_closed_sequence={"seed": 42, "trials": 3})  # missing key
        with pytest.raises(EditionsError, match="kvcache_closed_sequence"):
            _values(tmp_path, kvcache_closed_sequence={"loops": 2})  # extra key
        with pytest.raises(EditionsError, match="kvcache_closed_sequence"):
            _values(tmp_path, kvcache_closed_sequence={"trials": 0})
        with pytest.raises(EditionsError, match="kvcache_closed_sequence"):
            _values(tmp_path, kvcache_closed_sequence={"seed": "x"})

    def test_a_value_block_must_be_a_mapping(self, tmp_path):
        from mlpstorage_py.editions import EditionsError
        with pytest.raises(EditionsError, match="accelerator_memory_gb"):
            _blocks(tmp_path, accelerator_memory_gb=[180, 288])

    def test_a_different_table_carries_different_values(self, tmp_path):
        t = _values(tmp_path, accelerator_memory_gb={"b200": 192}, closed_mpi_processes={"llama3-70b": 32})
        c = t.editions["3.0"].checker
        assert c.accelerator_memory_gb["b200"] == 192
        assert c.closed_mpi_processes["llama3-70b"] == 32
        assert c.closed_mpi_processes["llama3-8b"] == 8


# ---------------------------------------------------------------------------
# Edition / table helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_models_per_division(self):
        e = _table().editions["3.0"]
        assert e.models("training", "closed") == ["unet3d", "retinanet"]
        assert e.models("training", "open") == ["unet3d", "retinanet"]
        assert e.models("checkpointing", "closed") == LLAMA
        assert e.models("kv_cache", "closed") == ["llama3.1-8b"]
        assert e.models("vector_database", "closed") == ["milvus"]
        assert e.models("training", "whatif") == []
        assert e.models("nope", "closed") == []

    def test_models_union_over_divisions(self):
        e = _table().editions["3.0"]
        assert e.models("training") == ["unet3d", "retinanet"]
        assert e.models("checkpointing") == LLAMA
        e2 = _table().editions["2.0"]
        assert e2.models("training") == ["unet3d", "resnet50", "cosmoflow"]
        assert e2.models("training", "open") == []

    def test_accelerators(self):
        e = _table().editions["3.0"]
        assert e.accelerators("training", "closed") == ["b200", "mi355"]
        assert e.accelerators("checkpointing", "open") == ["b200", "mi355"]
        assert e.accelerators() == ["b200", "mi355"]
        assert e.accelerators("kv_cache") == []
        assert _table().editions["2.0"].accelerators() == ["a100", "h100"]

    def test_tree_wide_vocabulary_is_the_union_over_checkable_editions(self):
        t = _table()
        assert t.families() == frozenset({"training", "checkpointing", "vector_database", "kv_cache"})
        assert t.vocabulary("training") == frozenset({"unet3d", "retinanet"})
        assert t.vocabulary("checkpointing") == frozenset(LLAMA)
        assert t.vocabulary("nope") == frozenset()
        # Historical (uncheckable) editions do not widen the tree-wide vocabulary.
        assert "cosmoflow" not in t.vocabulary("training")
        assert "bert" not in t.vocabulary("training")

    def test_vocabulary_widens_with_a_second_checkable_edition(self, tmp_path):
        def add(data):
            e = yaml.safe_load(yaml.safe_dump(data["editions"]["3.0"]))
            e["status"] = "historical"
            e["workloads"]["closed"]["training"]["flux"] = ["b200"]
            e["checker"]["training_au_thresholds"]["flux"] = 0.8
            data["editions"]["4.0"] = e
        t = _table_with(tmp_path, add)
        assert t.vocabulary("training") == frozenset({"unet3d", "retinanet", "flux"})
        assert t.editions["3.0"].models("training") == ["unet3d", "retinanet"]

    def test_current_edition_helper(self):
        from mlpstorage_py.editions import current_edition
        e = current_edition()
        assert e.id == RULES_EDITION
        assert e is _table().editions[RULES_EDITION]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    @pytest.mark.parametrize("name", [
        "MODELS_CLOSED", "MODELS_OPEN", "ACCELERATORS_CLOSED", "LLM_MODELS_CLOSED",
        "ACCELERATOR_MEMORY_GB", "LLM_CHECKPOINT_SIZE_GB",
    ])
    def test_config_constants_are_gone(self, name):
        import mlpstorage_py.config as config
        assert not hasattr(config, name), name

    def test_closed_mpi_processes_constant_is_gone(self):
        import mlpstorage_py.submission_checker.constants as constants
        assert not hasattr(constants, "CLOSED_MPI_PROCESSES")

    def test_llm_allowed_values_closed_count_is_in_lockstep(self):
        from mlpstorage_py.config import LLM_ALLOWED_VALUES
        counts = _table().editions[RULES_EDITION].checker.closed_mpi_processes
        assert {m: v[3] for m, v in LLM_ALLOWED_VALUES.items()} == counts

    def test_kvcache_model_default_is_the_closed_workload(self):
        from mlpstorage_py.config import KVCACHE_MODEL_DEFAULT
        assert _table().editions[RULES_EDITION].models("kv_cache", "closed") == [KVCACHE_MODEL_DEFAULT]

    def test_every_tool_accelerator_and_llm_has_a_table_entry(self):
        from mlpstorage_py.config import ACCELERATORS, LLM_MODELS
        checker = _table().editions[RULES_EDITION].checker
        assert set(checker.accelerator_memory_gb) == set(ACCELERATORS)
        assert set(checker.checkpoint_size_gb) == set(LLM_MODELS)
        assert set(checker.closed_mpi_processes) == set(LLM_MODELS)

    def test_no_module_still_imports_the_retired_names(self):
        pattern = re.compile(r"\b(MODELS_CLOSED|MODELS_OPEN|ACCELERATORS_CLOSED|LLM_MODELS_CLOSED|"
                             r"ACCELERATOR_MEMORY_GB|LLM_CHECKPOINT_SIZE_GB|CLOSED_MPI_PROCESSES)\b")
        offenders = []
        for p in (PROJECT_ROOT / "mlpstorage_py").rglob("*.py"):
            if "/tests/" in str(p):
                continue
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if pattern.search(line) and not line.lstrip().startswith("#"):
                    offenders.append(f"{p.relative_to(PROJECT_ROOT)}:{i}: {line.strip()}")
        assert offenders == [], "\n".join(offenders)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _config(**kw):
    from mlpstorage_py.submission_checker.configuration.configuration import Config
    return Config(submitters=["Acme"], skip_output_file=True, **kw)


class TestConfig:
    def test_edition_entry(self):
        c = _config()
        assert c.edition_entry is _table().editions[RULES_EDITION]
        assert c.edition_entry.checker is c.checker

    @pytest.mark.parametrize("size,count", [("8b", 8), ("70b", 64), ("405b", 512), ("1t", 1024)])
    def test_closed_mpi_processes_by_short_key(self, size, count):
        assert _config().get_closed_mpi_processes(size) == count

    def test_closed_mpi_processes_unknown_key_raises(self):
        with pytest.raises(KeyError):
            _config().get_closed_mpi_processes("99x")

    def test_accelerator_memory(self):
        assert _config().get_accelerator_memory_gb() == V30_VALUES["accelerator_memory_gb"]

    def test_training_au_threshold(self):
        c = _config()
        assert c.get_training_au_threshold("unet3d") == 0.90
        assert c.get_training_au_threshold("retinanet") == 0.85
        assert c.get_training_au_threshold("cosmoflow") is None

    def test_models(self):
        c = _config()
        assert c.models("checkpointing") == LLAMA
        assert c.models("training", "closed") == ["unet3d", "retinanet"]

    def test_values_follow_the_table(self, tmp_path, monkeypatch):
        import mlpstorage_py.submission_checker.configuration.configuration as cfg
        t = _values(tmp_path, accelerator_memory_gb={"b200": 192}, closed_mpi_processes={"llama3-70b": 32},
                    training_au_thresholds={"unet3d": 0.5})
        monkeypatch.setattr(cfg, "load_editions", lambda: t)
        c = _config()
        assert c.get_accelerator_memory_gb()["b200"] == 192
        assert c.get_closed_mpi_processes("70b") == 32
        assert c.get_training_au_threshold("unet3d") == 0.5


# ---------------------------------------------------------------------------
# Submission checks
# ---------------------------------------------------------------------------


def _fake_table(tmp_path, **changes):
    return _values(tmp_path, **changes)


def _checkpointing_check(config, *, model="llama3-70b", num_processes=64, accelerator="b200",
                         num_accelerators=None, checkpoint_size_gb=912, verification="CLOSED"):
    from mlpstorage_py.submission_checker.checks.checkpointing_checks import CheckpointingCheck
    from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs
    from mlpstorage_py.tests.conftest import MockLogger
    log = MockLogger()
    metadata = {
        "verification": verification,
        "accelerator": accelerator,
        "override_parameters": {"checkpoint.mode": "combined"},
        "args": {"model": model, "num_processes": num_processes, "accelerator_type": accelerator},
    }
    summary = {
        "num_accelerators": num_processes if num_accelerators is None else num_accelerators,
        "metric": {"checkpoint_size_GB": checkpoint_size_gb},
    }
    logs = SubmissionLogs(
        checkpoint_files=[(summary, metadata, "20260901_100000")], system_file={},
        loader_metadata=LoaderMetadata(division="closed", submitter="Acme", system="sys-1",
                                       mode="checkpointing", benchmark=model, folder="/fake/path"),
    )
    return CheckpointingCheck(log=log, config=config, submissions_logs=logs), log


def _training_check(config, *, model="unet3d", au_mean=95.0, expectation="success"):
    from mlpstorage_py.submission_checker.checks.training_checks import TrainingCheck
    from mlpstorage_py.submission_checker.loader import LoaderMetadata, SubmissionLogs
    from mlpstorage_py.tests.conftest import MockLogger
    log = MockLogger()
    summary = {"metric": {"train_au_mean_percentage": au_mean, "train_au_meet_expectation": expectation}}
    logs = SubmissionLogs(
        run_files=[(summary, {"args": {"model": model}}, "20260901_100000")], datagen_files=[],
        datasize_files=[], system_file={},
        loader_metadata=LoaderMetadata(division="closed", submitter="Acme", system="sys-1",
                                       mode="training", benchmark=model, folder="/fake/path"),
    )
    return TrainingCheck(log=log, config=config, submissions_logs=logs), log


def _config_on(monkeypatch, table):
    import mlpstorage_py.submission_checker.configuration.configuration as cfg
    monkeypatch.setattr(cfg, "load_editions", lambda: table)
    return _config()


class TestSubmissionChecks:
    def test_4_3_4_reads_the_edition_memory_table(self, tmp_path, monkeypatch):
        # 8 x b200 at 180 GB = 1440 GB covers 912 GB under the shipped table ...
        check, log = _checkpointing_check(_config(), num_processes=8)
        assert check.aggregate_accelerator_memory() is True
        assert log.errors == []
        # ... and fails when the submission's edition says b200 has 100 GB.
        config = _config_on(monkeypatch, _fake_table(tmp_path, accelerator_memory_gb={"b200": 100}))
        check, log = _checkpointing_check(config, num_processes=8)
        assert check.aggregate_accelerator_memory() is False
        assert any(m.startswith("[4.3.4 checkpointAggregateAcceleratorMemory]") and "100" in m
                   for m in log.errors), log.errors

    def test_4_3_4_unknown_accelerator_lists_the_edition_table(self, tmp_path, monkeypatch):
        config = _config_on(monkeypatch, _fake_table(tmp_path, accelerator_memory_gb={"tpu": 512}))
        check, log = _checkpointing_check(config, accelerator="gb300")
        assert check.aggregate_accelerator_memory() is False
        assert any("tpu" in m and "gb300" in m for m in log.errors), log.errors

    def test_4_6_1_reads_the_edition_process_counts(self, tmp_path, monkeypatch):
        check, log = _checkpointing_check(_config(), model="llama3-70b", num_processes=64)
        assert check.closed_mpi_processes() is True
        config = _config_on(monkeypatch, _fake_table(tmp_path, closed_mpi_processes={"llama3-70b": 32}))
        check, log = _checkpointing_check(config, model="llama3-70b", num_processes=64)
        assert check.closed_mpi_processes() is False
        assert any(m.startswith("[4.6.1 checkpointClosedMpiProcesses]") and "32" in m for m in log.errors), log.errors

    def test_4_3_3_reads_the_edition_models(self, tmp_path, monkeypatch):
        check, log = _checkpointing_check(_config(), model="llama3-70b")
        assert check.model_configuration_req() is True

        def only_8b(data):
            for div in ("closed", "open"):
                data["editions"]["3.0"]["workloads"][div]["checkpointing"] = {"llama3-8b": ["b200", "mi355"]}
            for name in ("closed_mpi_processes", "checkpoint_size_gb"):
                block = data["editions"]["3.0"]["checker"][name]
                data["editions"]["3.0"]["checker"][name] = {"llama3-8b": block["llama3-8b"]}
        config = _config_on(monkeypatch, _table_with(tmp_path, only_8b))
        check, log = _checkpointing_check(config, model="llama3-70b")
        assert check.model_configuration_req() is False
        assert any(m.startswith("[4.3.3 checkpointModelConfigurationReq]") for m in log.errors), log.errors

    def test_3_3_2_dlio_verdict_still_fails_a_run(self):
        check, log = _training_check(_config(), au_mean=80.0, expectation="fail")
        assert check.accelerator_utilization_check() is False
        msgs = [m for m in log.errors if m.startswith("[3.3.2 trainingAcceleratorUtilizationCheck]")]
        assert len(msgs) == 1 and "expected 'success'" in msgs[0], log.errors

    def test_3_3_2_mean_au_below_the_edition_minimum_fails_despite_dlio_success(self):
        check, log = _training_check(_config(), model="unet3d", au_mean=88.0, expectation="success")
        assert check.accelerator_utilization_check() is False
        msgs = [m for m in log.errors if m.startswith("[3.3.2 trainingAcceleratorUtilizationCheck]")]
        assert len(msgs) == 1 and "88.00" in msgs[0] and "90" in msgs[0], log.errors
        # retinanet's minimum is 85, so the same 88 passes there.
        check, log = _training_check(_config(), model="retinanet", au_mean=88.0)
        assert check.accelerator_utilization_check() is True
        assert log.errors == []

    def test_3_3_2_minimum_follows_the_edition(self, tmp_path, monkeypatch):
        config = _config_on(monkeypatch, _fake_table(tmp_path, training_au_thresholds={"unet3d": 0.95}))
        check, log = _training_check(config, model="unet3d", au_mean=92.0)
        assert check.accelerator_utilization_check() is False
        assert any("95" in m for m in log.errors), log.errors

    def test_3_3_2_at_the_minimum_passes(self):
        check, log = _training_check(_config(), model="unet3d", au_mean=90.0)
        assert check.accelerator_utilization_check() is True
        assert log.errors == []

    def test_3_3_2_unknown_model_trusts_dlio(self):
        check, log = _training_check(_config(), model="cosmoflow", au_mean=10.0)
        assert check.accelerator_utilization_check() is True

    def test_structure_vocabulary_comes_from_the_table(self):
        import mlpstorage_py.submission_checker.checks.submission_structure_checks as ssc
        t = _table()
        assert ssc._VALID_WORKLOAD_CATEGORIES == t.families()
        assert ssc._VALID_TRAINING_WORKLOADS == t.vocabulary("training")
        assert ssc._VALID_CHECKPOINTING_WORKLOADS == t.vocabulary("checkpointing")
        src = inspect.getsource(ssc)
        assert 'frozenset({"unet3d", "retinanet"})' not in src
        assert '"llama3-405b"' not in src.split("class SubmissionStructureCheck")[0]


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


def _params_with(tmp_path, **changes):
    return _values(tmp_path, **changes).editions["3.0"].checker


class TestRuntimeCheckpointing:
    def _run(self, *, model, num_processes, accelerator):
        from mlpstorage_py.config import BENCHMARK_TYPES
        from mlpstorage_py.rules.models import BenchmarkRun, BenchmarkRunData
        data = BenchmarkRunData(benchmark_type=BENCHMARK_TYPES.checkpointing, model=model, command="run",
                                run_datetime="20260917_120000", num_processes=num_processes, parameters={},
                                override_parameters={}, accelerator=accelerator)
        return BenchmarkRun.from_data(data, MagicMock())

    def test_accelerator_memory_reads_the_current_edition(self, tmp_path):
        from mlpstorage_py.config import PARAM_VALIDATION
        from mlpstorage_py.rules.run_checkers.checkpointing import CheckpointingRunRulesChecker
        run = self._run(model="llama3-70b", num_processes=8, accelerator="b200")
        assert CheckpointingRunRulesChecker(run, logger=MagicMock()).check_accelerator_memory() is None
        params = _params_with(tmp_path, accelerator_memory_gb={"b200": 100})
        with patch("mlpstorage_py.rules.run_checkers.checkpointing.checker_parameters", return_value=params):
            issue = CheckpointingRunRulesChecker(run, logger=MagicMock()).check_accelerator_memory()
        assert issue is not None and issue.validation == PARAM_VALIDATION.INVALID
        assert "100" in issue.message and "912" in issue.message

    def test_checkpoint_size_reads_the_current_edition(self, tmp_path):
        from mlpstorage_py.rules.run_checkers.checkpointing import CheckpointingRunRulesChecker
        run = self._run(model="llama3-70b", num_processes=8, accelerator="b200")
        params = _params_with(tmp_path, checkpoint_size_gb={"llama3-70b": 5000})
        with patch("mlpstorage_py.rules.run_checkers.checkpointing.checker_parameters", return_value=params):
            issue = CheckpointingRunRulesChecker(run, logger=MagicMock()).check_accelerator_memory()
        assert issue is not None and "5000" in issue.message

    def test_closed_process_count_reads_the_current_edition(self, tmp_path):
        from mlpstorage_py.config import PARAM_VALIDATION
        from mlpstorage_py.rules.run_checkers.checkpointing import CheckpointingRunRulesChecker
        run = self._run(model="llama3-70b", num_processes=64, accelerator="b200")
        assert CheckpointingRunRulesChecker(run, logger=MagicMock()).check_num_processes() is None
        params = _params_with(tmp_path, closed_mpi_processes={"llama3-70b": 16})
        with patch("mlpstorage_py.rules.run_checkers.checkpointing.checker_parameters", return_value=params):
            checker = CheckpointingRunRulesChecker(run, logger=MagicMock())
            issue = checker.check_num_processes()
            assert issue is not None and issue.validation == PARAM_VALIDATION.OPEN and "16" in issue.message
            run16 = self._run(model="llama3-70b", num_processes=16, accelerator="b200")
            assert CheckpointingRunRulesChecker(run16, logger=MagicMock()).check_num_processes() is None


class TestRuntimeKvcache:
    def _bm(self, tmp_path, **overrides):
        from tests.unit.test_benchmarks_kvcache import _make_run_benchmark
        bm = _make_run_benchmark(tmp_path, what_if=False)
        bm.args.mode = "closed"
        for k, v in overrides.items():
            setattr(bm.args, k, v)
        return bm

    def test_closed_locks_come_from_the_current_edition(self, tmp_path):
        bm = self._bm(tmp_path, inter_option_delay=90, trials=3, seed=42)
        params = _params_with(tmp_path, kvcache_closed_sequence={"seed": 7, "trials": 2, "inter_option_delay_s": 45})
        with patch("mlpstorage_py.benchmarks.kvcache.checker_parameters", return_value=params), \
             patch.object(bm, "logger") as log:
            assert bm._execute_run() == 1
        log.error.assert_called()
        # The first lock checked is --seed: the run's 42 is now illegal (must be 7).
        assert "must be 7, got 42" in str(log.error.call_args_list[-1])
        # With the seed matching, the delay lock is what fires.
        second = tmp_path / "second"
        second.mkdir()
        bm = self._bm(second, inter_option_delay=90, trials=2, seed=7)
        with patch("mlpstorage_py.benchmarks.kvcache.checker_parameters", return_value=params), \
             patch.object(bm, "logger") as log:
            assert bm._execute_run() == 1
        assert "must be 45, got 90" in str(log.error.call_args_list[-1])

    def test_closed_effective_values_come_from_the_current_edition(self, tmp_path):
        bm = self._bm(tmp_path, inter_option_delay=None, trials=None, seed=None)
        params = _params_with(tmp_path, kvcache_closed_sequence={"seed": 7, "trials": 1, "inter_option_delay_s": 0})
        seeds, sleeps = [], []

        def fake_execute(cmd, **kwargs):
            seeds.append(cmd)
            return ("", "", 0)
        with patch("mlpstorage_py.benchmarks.kvcache.checker_parameters", return_value=params), \
             patch.object(bm, "_execute_command", side_effect=fake_execute), \
             patch.object(bm, "_interruptible_sleep", side_effect=lambda s: sleeps.append(s)), \
             patch.object(bm, "_aggregate_option_results", return_value={
                 "option": 1, "aggregated_read_bandwidth_gbps": 0.0, "aggregated_write_bandwidth_gbps": 0.0,
                 "aggregated_avg_throughput_tokens_per_sec": 0.0, "aggregated_storage_throughput_tokens_per_sec": 0.0,
                 "aggregated_p95_latency_ms": 0.0, "rank_count": 2, "trial_count": 1, "partial_failure": False,
                 "missing_files": [], "cpu_tier_ranks": []}), \
             patch.object(bm, "_write_run_summary"), patch.object(bm, "write_metadata"):
            assert bm._execute_run() == 0
        assert len(seeds) == 3, seeds  # trials=1 from the table, not 3
        assert all("--seed-base 7" in c or "--seed-base=7" in c for c in seeds), seeds
        assert sleeps == [0, 0]

    def test_closed_model_is_the_edition_workload(self, tmp_path):
        from mlpstorage_py.config import KVCACHE_MODEL_DEFAULT
        from tests.unit.test_benchmarks_kvcache import _make_run_benchmark
        bm = _make_run_benchmark(tmp_path, what_if=False)
        assert bm.args.model == "llama3.1-8b" == KVCACHE_MODEL_DEFAULT

    def test_help_text_and_closed_defaults_carry_the_edition_values(self):
        from mlpstorage_py.cli.kvcache_args import KVCACHE_HELP_MESSAGES
        text = KVCACHE_HELP_MESSAGES["inter_option_delay"]
        assert "90" in text and "20" not in text, text
        assert "42" in KVCACHE_HELP_MESSAGES["seed"]
        assert "3" in KVCACHE_HELP_MESSAGES["trials"]
        from mlpstorage_py.cli_parser import parse_arguments
        argv = ["mlpstorage", "closed", "kvcache", "run", "-rd", "/tmp", "-sn", "sys-v1"]
        with patch("sys.argv", argv):
            args = parse_arguments()
        assert (args.seed, args.trials, args.inter_option_delay) == (42, 3, 90)

    def test_run_summary_effective_defaults_come_from_the_current_edition(self, tmp_path):
        from mlpstorage_py import run_summary
        args = Namespace(seed=None, trials=None, inter_option_delay=None, model="llama3.1-8b",
                         mode="closed", hosts=["localhost"], npernode=None, num_processes=None)
        params = _params_with(tmp_path, kvcache_closed_sequence={"seed": 7, "trials": 2, "inter_option_delay_s": 45})
        lines = []
        with patch("mlpstorage_py.run_summary.checker_parameters", return_value=params):
            run_summary._print_kvcache_section(args, lines)
        text = "\n".join(lines)
        assert "45  [default]" in text and "7  [default]" in text and "2  [default]" in text, text


class TestRuntimeCli:
    def _edition_with(self, **training_models):
        e = _table().editions["3.0"]
        workloads = yaml.safe_load(yaml.safe_dump(e.workloads))
        workloads["closed"]["training"].update(training_models)
        return dataclasses.replace(e, workloads=workloads)

    def test_training_choices_come_from_the_current_edition(self):
        from mlpstorage_py.cli_parser import parse_arguments
        argv = ["mlpstorage", "closed", "training", "resnet50", "datasize", "-cm", "64", "-at", "b200", "-ma", "4",
                "-rd", "/tmp", "-sn", "sys-v1"]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit):
                parse_arguments()
        with patch("mlpstorage_py.cli.training_args.current_edition",
                   return_value=self._edition_with(resnet50=["b200", "mi355"])), patch("sys.argv", argv):
            assert parse_arguments().model == "resnet50"

    def test_training_accelerator_choices_come_from_the_current_edition(self):
        from mlpstorage_py.cli_parser import parse_arguments
        argv = ["mlpstorage", "closed", "training", "unet3d", "run", "-cm", "64", "-at", "h100", "-na", "4",
                "-rd", "/tmp", "-sn", "sys-v1", "-dd", "/tmp", "file"]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit):
                parse_arguments()
        with patch("mlpstorage_py.cli.training_args.current_edition",
                   return_value=self._edition_with(unet3d=["b200", "mi355", "h100"])), patch("sys.argv", argv):
            assert parse_arguments().accelerator_type == "h100"

    def test_checkpointing_choices_come_from_the_current_edition(self):
        from mlpstorage_py.cli_parser import parse_arguments
        argv = ["mlpstorage", "closed", "checkpointing", "run", "-cm", "64", "-m", "llama3-8b", "-np", "8",
                "-at", "h100", "-cf", "/tmp", "-rd", "/tmp", "-sn", "sys-v1", "file"]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit):
                parse_arguments()
        e = _table().editions["3.0"]
        workloads = yaml.safe_load(yaml.safe_dump(e.workloads))
        workloads["closed"]["checkpointing"]["llama3-8b"].append("h100")
        with patch("mlpstorage_py.cli.checkpointing_args.current_edition",
                   return_value=dataclasses.replace(e, workloads=workloads)), patch("sys.argv", argv):
            assert parse_arguments().accelerator_type == "h100"

    def test_whatif_keeps_the_tool_capabilities(self):
        from mlpstorage_py.cli_parser import parse_arguments
        from mlpstorage_py.config import ACCELERATORS, MODELS
        for model in MODELS:
            argv = ["mlpstorage", "whatif", "training", model, "datasize", "-cm", "64", "-at", ACCELERATORS[0],
                    "-ma", "4", "-rd", "/tmp", "-sn", "sys-v1"]
            with patch("sys.argv", argv):
                assert parse_arguments().model == model

    def test_help_all_tree_reads_the_table(self):
        from mlpstorage_py.cli import help_formatter
        e = _table().editions["3.0"]
        assert help_formatter._TRAINING_MODELS_CLOSED_OPEN == frozenset(e.models("training", "closed")) | \
            frozenset(e.models("training", "open"))
        assert "frozenset(('unet3d', 'retinanet'))" not in inspect.getsource(help_formatter)

    def test_datagen_allowlist_reads_the_table(self):
        from mlpstorage_py.errors import ConfigurationError
        from mlpstorage_py.rules import datagen_hierarchy
        datagen_hierarchy.validate_supported_model("unet3d", "closed")
        with pytest.raises(ConfigurationError):
            datagen_hierarchy.validate_supported_model("resnet50", "open")
        with patch("mlpstorage_py.rules.datagen_hierarchy.current_edition",
                   return_value=TestRuntimeCli()._edition_with(resnet50=["b200"])):
            datagen_hierarchy.validate_supported_model("resnet50", "closed")
        assert "MODELS_CLOSED" not in inspect.getsource(datagen_hierarchy)

    def test_training_submission_checker_supported_models(self):
        from mlpstorage_py.rules.submission_checkers.training import TrainingSubmissionRulesChecker
        assert list(TrainingSubmissionRulesChecker.supported_models) == ["unet3d", "retinanet"]


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------


class TestDocs:
    @pytest.mark.parametrize("doc", ["Rules.md", "README.md", "ManPage.md"])
    def test_docs_name_the_new_block_contents(self, doc):
        text = (PROJECT_ROOT / doc).read_text()
        lines = [ln for ln in text.splitlines() if "editions.yaml" in ln and "checker" in ln]
        assert lines, f"{doc}: no line describes the checker block"
        joined = " ".join(lines)
        for needle in ("accelerator memor", "process count", "AU"):
            assert needle in joined, f"{doc}: {needle!r} missing from the checker-block description"

    def test_rules_md_3_3_2_records_the_minimums(self):
        text = (PROJECT_ROOT / "Rules.md").read_text()
        block = text.split("3.3.2. **trainingAcceleratorUtilizationCheck**", 1)[1].split("3.3.3.", 1)[0]
        assert "unet3d" in block and "retinanet" in block and "editions.yaml" in block, block

    def test_manpage_kvcache_help_no_longer_says_20(self):
        text = (PROJECT_ROOT / "ManPage.md").read_text()
        assert "fixed at 20" not in text and "default 20" not in text
