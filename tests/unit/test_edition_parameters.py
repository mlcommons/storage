"""Per-edition checker parameters and the per-submission ``Config``
(retires ``submission_checker.constants.VERSIONS``; design §5 of
.planning/rules-editions-and-comparability-classes.md).

The six ``VERSIONS``-keyed required-file / required-folder dicts move into
``mlpstorage_py/rules/editions.yaml`` as a ``checker:`` block per edition.
Only an edition this tool can check carries one (3.0 today); an edition
without one is "not checkable by this tool" and its ``tool:`` line says
which tool is. ``Config`` is built from the table -- once tree-wide for the
pre-loop checks, then per submission from the ``rules_edition`` its
``submission.yaml`` declares -- and the reviewer-asserted
``--mlperf-version`` flag is gone: the submission declares its edition.

Covered here:
- the table: 3.0 carries the six lists byte-for-byte as the former
  constants, every regex compiles, historical editions carry none, a
  malformed block is rejected;
- ``constants``: ``VERSIONS`` / ``DEFAULT_SPEC_VERSION`` / the six dicts are
  gone, ``SYSTEM_PATH`` is one layout string;
- ``Config``: default is the current edition, explicit edition, uncheckable
  or unknown edition raises, ``for_edition`` keeps the tree-wide options;
- ``Loader`` takes no version;
- ``main.run``: no manifest -> current edition; manifest declaring the
  current edition -> that edition; manifest declaring a known but
  uncheckable edition -> EDN-04 once, submission skipped, exit 1; manifest
  declaring an unknown edition -> EDN-01 only, validated tree-wide;
- the runtime datagen validator reads the table, not the constants;
- CLI / docs: ``--mlperf-version`` and standalone ``--version`` rejected,
  README / ManPage / ``--help_all`` no longer list them, Rules.md and
  ManPage document EDN-04, the old spec-version test module is gone.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import logging
import re
from pathlib import Path

import pytest
import yaml

from mlpstorage_py.config import RULES_EDITION
from mlpstorage_py.provenance import (
    MANIFEST_FILENAME,
    write_submission_manifest,
)
from mlpstorage_py.submission_checker.rule_registry import discover_rules

from tests.unit.test_leaf_provenance import (  # noqa: E402
    HASH_A, UNET3D_B200, _image, _leaf, _org,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDITIONS_YAML = PROJECT_ROOT / "mlpstorage_py" / "rules" / "editions.yaml"

LEAF_CLOSED_RUN = "closed/Acme/results/sys-1/training/unet3d/run/20260901_100000"

# The former constants.py dicts, byte-for-byte (the v3.0 validator-diff
# depends on these regexes staying exactly as they were).
V30_CHECKER = {
    "datagen_required_files": [r"training_datagen\.stdout\.log$", r"training_datagen\.stderr\.log$",
                               r"dlio\.log$", r"training_.*_metadata\.json$"],
    "datagen_required_folders": ["dlio_config"],
    "run_required_files": [r"training_run\.stdout.log", r"training_run\.stderr.log", r".*output\.json",
                           r".*per_epoch_stats\.json", r".*summary\.json", r"dlio\.log"],
    "run_required_folders": ["dlio_config"],
    "checkpoint_required_files": [r"checkpointing_run\.stdout\.log", r"checkpointing_run\.stderr\.log",
                                  r".*output\.json", r".*per_epoch_stats\.json", r".*summary\.json",
                                  r"dlio\.log"],
    "checkpoint_required_folders": ["dlio_config"],
}
CHECKER_FIELDS = tuple(V30_CHECKER)


def _table():
    from mlpstorage_py.editions import load_editions
    return load_editions()


def _write_table(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load(EDITIONS_YAML.read_text())
    mutate(data)
    p = tmp_path / "editions.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


class TestTable:
    def test_current_edition_carries_the_checker_block(self):
        checker = _table().editions[RULES_EDITION].checker
        assert checker is not None
        for name in CHECKER_FIELDS:
            assert getattr(checker, name) == V30_CHECKER[name], name
        for name in ("datagen_required_files", "run_required_files", "checkpoint_required_files"):
            for pattern in getattr(checker, name):
                re.compile(pattern)

    def test_historical_editions_are_not_checkable_by_this_tool(self):
        table = _table()
        for eid in ("0.5", "1.0", "2.0"):
            assert table.editions[eid].checker is None, eid
            assert table.checker_parameters(eid) is None, eid
            assert not table.is_checkable(eid), eid
        assert table.is_checkable(RULES_EDITION)
        assert not table.is_checkable("9.9")

    def test_checker_parameters_defaults_to_the_current_edition(self):
        from mlpstorage_py.editions import checker_parameters
        table = _table()
        assert checker_parameters() == table.editions[table.current_edition].checker
        assert table.checker_parameters() is table.editions[table.current_edition].checker
        assert checker_parameters("2.0") is None

    def test_a_checker_block_must_carry_all_six_lists(self, tmp_path):
        from mlpstorage_py.editions import EditionsError, load_editions

        def drop(data):
            del data["editions"]["3.0"]["checker"]["run_required_files"]
        with pytest.raises(EditionsError, match="run_required_files"):
            load_editions(_write_table(tmp_path, drop))

    def test_a_checker_regex_must_compile(self, tmp_path):
        from mlpstorage_py.editions import EditionsError, load_editions

        def bad(data):
            data["editions"]["3.0"]["checker"]["datagen_required_files"] = ["("]
        with pytest.raises(EditionsError, match="datagen_required_files"):
            load_editions(_write_table(tmp_path, bad))

    def test_a_checker_list_must_be_a_list_of_strings(self, tmp_path):
        from mlpstorage_py.editions import EditionsError, load_editions

        def bad(data):
            data["editions"]["3.0"]["checker"]["run_required_folders"] = "dlio_config"
        with pytest.raises(EditionsError, match="run_required_folders"):
            load_editions(_write_table(tmp_path, bad))

    def test_the_current_edition_must_be_checkable(self, tmp_path):
        from mlpstorage_py.editions import EditionsError, load_editions

        def drop(data):
            del data["editions"]["3.0"]["checker"]
        with pytest.raises(EditionsError, match="checker"):
            load_editions(_write_table(tmp_path, drop))


# ---------------------------------------------------------------------------
# constants.py
# ---------------------------------------------------------------------------


class TestConstants:
    @pytest.mark.parametrize("name", [
        "VERSIONS", "DEFAULT_SPEC_VERSION", "_derive_default_spec_version",
        "DATAGEN_REQUIRED_FILES", "DATAGEN_REQUIRED_FOLDERS",
        "RUN_REQUIRED_FILES", "RUN_REQUIRED_FOLDERS",
        "CHECKPOINT_REQUIRED_FILES", "CHECKPOINT_REQUIRED_FOLDERS",
    ])
    def test_version_keyed_names_are_gone(self, name):
        from mlpstorage_py.submission_checker import constants
        assert not hasattr(constants, name), name

    def test_system_path_is_one_layout_string(self):
        from mlpstorage_py.submission_checker.constants import SYSTEM_PATH
        assert isinstance(SYSTEM_PATH, str)
        assert SYSTEM_PATH.format(division="closed", submitter="Acme", system="sys-1") == \
            "closed/Acme/systems/sys-1.yaml"


# ---------------------------------------------------------------------------
# Config / Loader
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_edition_is_the_current_one(self):
        from mlpstorage_py.submission_checker.configuration.configuration import Config
        c = Config(submitters=None)
        assert c.edition == RULES_EDITION
        assert c.get_datagen_required_files() == V30_CHECKER["datagen_required_files"]
        assert c.get_datagen_required_folders() == V30_CHECKER["datagen_required_folders"]
        assert c.get_run_required_files() == V30_CHECKER["run_required_files"]
        assert c.get_run_required_folders() == V30_CHECKER["run_required_folders"]
        assert c.get_checkpoint_required_files() == V30_CHECKER["checkpoint_required_files"]
        assert c.get_checkpoint_required_folders() == V30_CHECKER["checkpoint_required_folders"]

    def test_explicit_edition(self):
        from mlpstorage_py.submission_checker.configuration.configuration import Config
        assert Config(edition="3.0", submitters=None).edition == "3.0"
        assert Config(edition=3.0, submitters=None).edition == "3.0"   # YAML floats normalise

    def test_uncheckable_edition_raises(self):
        from mlpstorage_py.editions import EditionsError, UncheckableEditionError
        from mlpstorage_py.submission_checker.configuration.configuration import Config
        assert issubclass(UncheckableEditionError, EditionsError)
        with pytest.raises(UncheckableEditionError, match=r"2\.0.*mlpstorage 2\.0"):
            Config(edition="2.0", submitters=None)
        with pytest.raises(UncheckableEditionError, match=r"9\.9"):
            Config(edition="9.9", submitters=None)

    def test_for_edition_keeps_the_tree_wide_options(self):
        from mlpstorage_py.editions import UncheckableEditionError
        from mlpstorage_py.submission_checker.configuration.configuration import Config
        c = Config(submitters=["Acme"], skip_output_file=True)
        d = c.for_edition("3.0")
        assert d.edition == "3.0"
        assert d.submitters == ["Acme"] and d.skip_output_file is True
        assert d.check_submitter("Acme") and not d.check_submitter("Other")
        with pytest.raises(UncheckableEditionError):
            c.for_edition("2.0")

    def test_signatures_carry_no_version(self):
        from mlpstorage_py.submission_checker.configuration.configuration import Config
        from mlpstorage_py.submission_checker.loader import Loader
        assert "version" not in inspect.signature(Config.__init__).parameters
        assert "edition" in inspect.signature(Config.__init__).parameters
        assert "version" not in inspect.signature(Loader.__init__).parameters

    def test_loader_uses_the_single_layout_path(self, tmp_path):
        from mlpstorage_py.submission_checker.configuration.configuration import Config
        from mlpstorage_py.submission_checker.loader import Loader
        loader = Loader(root=str(tmp_path), config=Config(submitters=None))
        assert loader.system_log_path == str(tmp_path / "{division}/{submitter}/systems/{system}.yaml")


# ---------------------------------------------------------------------------
# main.run: per-submission Config
# ---------------------------------------------------------------------------


class _Recorder:
    """Stand-in for MODE_TO_CHECKERS entries: records the Config each
    submission was checked with and passes."""
    seen: list = []

    def __init__(self, log, config, logs):
        _Recorder.seen.append((logs.loader_metadata.submitter, config))

    def __call__(self):
        return True


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "sub"
    _org(root)
    _image(root, HASH_A)
    _leaf(root, LEAF_CLOSED_RUN, parameters=UNET3D_B200)
    return root


def _run(root: Path, tmp_path: Path, monkeypatch) -> int:
    from mlpstorage_py.submission_checker import main as checker_main
    _Recorder.seen = []
    monkeypatch.setattr(checker_main, "MODE_TO_CHECKERS", {"training": [_Recorder]})
    args = argparse.Namespace(input=str(root), submitters=None,
                              csv=str(tmp_path / "out.csv"), skip_output_file=True)
    return checker_main.run(args)


def _declare(root: Path, edition) -> Path:
    path = write_submission_manifest(root, "closed", "Acme")
    data = yaml.safe_load(path.read_text())
    data["rules_edition"] = edition
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


class TestPerSubmissionConfig:
    def test_rule_ids(self):
        from mlpstorage_py.submission_checker.checks.edition_checks import EditionCheck
        found = discover_rules(EditionCheck)
        assert found["EDN-04"][0] == "checkableEdition"

    def test_no_manifest_validates_under_the_current_edition(self, tmp_path, monkeypatch, caplog):
        root = _tree(tmp_path)
        with caplog.at_level(logging.DEBUG):
            _run(root, tmp_path, monkeypatch)
        assert [s for s, _ in _Recorder.seen] == ["Acme"]
        assert _Recorder.seen[0][1].edition == RULES_EDITION
        assert "[EDN-04" not in caplog.text

    def test_manifest_declaring_the_current_edition(self, tmp_path, monkeypatch, caplog):
        root = _tree(tmp_path)
        _declare(root, RULES_EDITION)
        with caplog.at_level(logging.DEBUG):
            _run(root, tmp_path, monkeypatch)
        assert [s for s, _ in _Recorder.seen] == ["Acme"]
        assert _Recorder.seen[0][1].edition == RULES_EDITION
        assert "[EDN-04" not in caplog.text

    def test_manifest_declaring_an_uncheckable_edition(self, tmp_path, monkeypatch, caplog):
        root = _tree(tmp_path)
        manifest = _declare(root, "2.0")
        with caplog.at_level(logging.DEBUG):
            rc = _run(root, tmp_path, monkeypatch)
        assert rc == 1
        lines = [l for l in caplog.text.splitlines() if "[EDN-04 checkableEdition]" in l]
        assert len(lines) == 1, lines
        assert str(manifest) in lines[0] and "2.0" in lines[0] and "mlpstorage 2.0" in lines[0]
        assert "[EDN-01" not in caplog.text
        assert _Recorder.seen == []          # workload checks skipped
        assert "skipping" in caplog.text.lower() and LEAF_CLOSED_RUN.split("/results/")[0] in caplog.text

    def test_manifest_declaring_an_unknown_edition_is_edn01_only(self, tmp_path, monkeypatch, caplog):
        root = _tree(tmp_path)
        _declare(root, "9.9")
        with caplog.at_level(logging.DEBUG):
            rc = _run(root, tmp_path, monkeypatch)
        assert rc == 1
        assert "[EDN-01 rulesEdition]" in caplog.text
        assert "[EDN-04" not in caplog.text
        assert [s for s, _ in _Recorder.seen] == ["Acme"]
        assert _Recorder.seen[0][1].edition == RULES_EDITION

    def test_config_is_built_once_per_org(self, tmp_path, monkeypatch):
        root = _tree(tmp_path)
        _leaf(root, "closed/Acme/results/sys-1/training/unet3d/run/20260902_100000", parameters=UNET3D_B200)
        _declare(root, RULES_EDITION)
        _run(root, tmp_path, monkeypatch)
        configs = {id(c) for _, c in _Recorder.seen}
        assert len(_Recorder.seen) >= 1 and len(configs) == 1

    def test_pre_loop_checks_keep_the_tree_wide_config(self):
        from mlpstorage_py.submission_checker import main as checker_main
        src = inspect.getsource(checker_main.run)
        assert "EditionCheck(log, config, args.input)" in src
        assert "config_for_submission(" in src


# ---------------------------------------------------------------------------
# Runtime datagen validator
# ---------------------------------------------------------------------------


class TestRuntimeLeafValidator:
    def test_reads_the_edition_table_not_the_constants(self):
        from mlpstorage_py.rules import datagen_hierarchy as dh
        src = inspect.getsource(dh)
        assert "submission_checker.constants" not in src
        assert "_select_regex_set" not in src
        assert "checker_parameters" in src

    def test_validate_leaves_still_enforce_the_v30_sets(self, tmp_path):
        from mlpstorage_py.rules.datagen_hierarchy import (
            validate_checkpoint_leaf, validate_datagen_leaf, validate_run_leaf,
        )
        leaf = tmp_path / "leaf"
        (leaf / "dlio_config").mkdir(parents=True)
        for n in ("config.yaml", "hydra.yaml", "overrides.yaml"):
            (leaf / "dlio_config" / n).write_text("")
        for n in ("training_datagen.stdout.log", "training_datagen.stderr.log", "dlio.log",
                  "training_20260901_100000_metadata.json", "training_run.stdout.log",
                  "training_run.stderr.log", "checkpointing_run.stdout.log",
                  "checkpointing_run.stderr.log", "x_output.json", "x_per_epoch_stats.json",
                  "x_summary.json"):
            (leaf / n).write_text("")
        assert validate_datagen_leaf(str(leaf)) == []
        assert validate_run_leaf(str(leaf)) == []
        assert validate_checkpoint_leaf(str(leaf)) == []
        (leaf / "dlio.log").unlink()
        assert validate_run_leaf(str(leaf)) != []


# ---------------------------------------------------------------------------
# CLI and docs
# ---------------------------------------------------------------------------


class TestCliAndDocs:
    def test_validate_parser_rejects_mlperf_version(self):
        from mlpstorage_py.cli.utility_args import add_validate_arguments
        parser = argparse.ArgumentParser()
        add_validate_arguments(parser)
        ns = parser.parse_args(["some-dir"])
        assert not hasattr(ns, "version")
        with pytest.raises(SystemExit):
            parser.parse_args(["some-dir", "--mlperf-version", "v3.0"])

    def test_standalone_parser_rejects_version(self, monkeypatch, capsys):
        from mlpstorage_py.submission_checker import main as checker_main
        monkeypatch.setattr("sys.argv", ["submission_checker", "--input", "d", "--version", "v3.0"])
        with pytest.raises(SystemExit):
            checker_main.get_args()
        monkeypatch.setattr("sys.argv", ["submission_checker", "--input", "d"])
        assert not hasattr(checker_main.get_args(), "version")

    def test_help_all_readme_and_manpage_no_longer_list_the_flag(self):
        from mlpstorage_py.cli import help_formatter
        assert "--mlperf-version" not in inspect.getsource(help_formatter)
        for doc in ("README.md", "ManPage.md"):
            assert "--mlperf-version" not in (PROJECT_ROOT / doc).read_text(), doc

    def test_rules_md_and_manpage_document_edn04_and_the_checker_block(self):
        rules = (PROJECT_ROOT / "Rules.md").read_text()
        assert "EDN-04 (`checkableEdition`)" in rules
        assert "checker" in rules.split("EDN-04")[0].rsplit("**Rules editions", 1)[1]
        man = (PROJECT_ROOT / "ManPage.md").read_text()
        assert "EDN-04" in man

    def test_old_spec_version_test_module_is_gone(self):
        assert not (PROJECT_ROOT / "mlpstorage_py" / "tests" / "test_default_spec_version.py").exists()
