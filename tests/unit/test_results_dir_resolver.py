"""
Results-dir resolution: one recorded default instead of ``--results-dir`` on
every command.

``mlpstorage init <orgname> [path]`` records the initialized path in the
per-user config file (``$XDG_CONFIG_HOME/mlpstorage/config.yaml``, default
``~/.config/mlpstorage/config.yaml``). Every later command resolves its
results-dir in this order and says which tier won:

    --results-dir flag  >  MLPSTORAGE_RESULTS_DIR  >  user config file

With no ``[path]``, ``init`` uses ``~/mlpstorage-results``.

Closes two gaps from the 2026-09-17 hygiene survey:

* GAP A — ``datasize``/``datagen``/``configview`` used to run without a
  results-dir and fell through to a stale "export MLPSTORAGE_ORGNAME" error.
  Every closed/open/whatif command now requires a resolved results-dir, and
  orgname comes only from the sentinel.
* GAP B — command history lived at ``~/mlps_history`` while the ManPage
  promised it inside the results-dir. It now lives at
  ``<results-dir>/.mlps/history`` and ``history show/rerun`` resolve the
  results-dir the same way as everything else.
"""
from __future__ import annotations

import os
import re
from argparse import Namespace
from unittest.mock import patch

import pytest
import yaml


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    """Isolate the per-user config under a fresh XDG_CONFIG_HOME."""
    home = tmp_path / "xdg"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    monkeypatch.delenv("MLPSTORAGE_RESULTS_DIR", raising=False)
    return home


def _init(orgname, path=None):
    from mlpstorage_py.results_dir.init import run_init

    return run_init(Namespace(mode="init", orgname=orgname, path=path))


# --------------------------------------------------------------------------- #
# User config file                                                             #
# --------------------------------------------------------------------------- #


class TestUserConfigFile:
    def test_path_honours_xdg_config_home(self, xdg):
        from mlpstorage_py.results_dir.user_config import user_config_path

        assert user_config_path() == str(xdg / "mlpstorage" / "config.yaml")

    def test_path_defaults_to_dot_config_under_home(self, tmp_path, monkeypatch):
        from mlpstorage_py.results_dir.user_config import user_config_path

        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert user_config_path() == str(
            tmp_path / ".config" / "mlpstorage" / "config.yaml"
        )

    def test_relative_xdg_config_home_is_ignored(self, tmp_path, monkeypatch):
        """The XDG spec says a relative XDG_CONFIG_HOME must be ignored."""
        from mlpstorage_py.results_dir.user_config import user_config_path

        monkeypatch.setenv("XDG_CONFIG_HOME", "relative/dir")
        monkeypatch.setenv("HOME", str(tmp_path))
        assert user_config_path() == str(
            tmp_path / ".config" / "mlpstorage" / "config.yaml"
        )

    def test_missing_file_reads_as_empty(self, xdg):
        from mlpstorage_py.results_dir.user_config import read_user_config

        assert read_user_config() == {}

    def test_record_results_dir_creates_file_and_keeps_other_keys(self, xdg, tmp_path):
        from mlpstorage_py.results_dir.user_config import (
            read_user_config,
            record_results_dir,
            user_config_path,
        )

        cfg = xdg / "mlpstorage" / "config.yaml"
        cfg.parent.mkdir()
        cfg.write_text("systemname: keep-me\n")

        written = record_results_dir(str(tmp_path / "r"))

        assert written == user_config_path()
        data = read_user_config()
        assert data["results_dir"] == str(tmp_path / "r")
        assert data["systemname"] == "keep-me"
        # No temp file left behind from the atomic write.
        assert sorted(p.name for p in cfg.parent.iterdir()) == ["config.yaml"]

    def test_record_results_dir_stores_absolute_path(self, xdg, tmp_path, monkeypatch):
        from mlpstorage_py.results_dir.user_config import (
            read_user_config,
            record_results_dir,
        )

        monkeypatch.chdir(tmp_path)
        record_results_dir("rel")
        assert read_user_config()["results_dir"] == str(tmp_path / "rel")

    def test_malformed_yaml_is_a_configuration_error(self, xdg):
        from mlpstorage_py.errors import ConfigurationError
        from mlpstorage_py.results_dir.user_config import read_user_config

        cfg = xdg / "mlpstorage" / "config.yaml"
        cfg.parent.mkdir()
        cfg.write_text("results_dir: [unterminated\n")
        with pytest.raises(ConfigurationError) as exc:
            read_user_config()
        assert str(cfg) in str(exc.value)

    def test_non_mapping_yaml_is_a_configuration_error(self, xdg):
        from mlpstorage_py.errors import ConfigurationError
        from mlpstorage_py.results_dir.user_config import read_user_config

        cfg = xdg / "mlpstorage" / "config.yaml"
        cfg.parent.mkdir()
        cfg.write_text("- just\n- a list\n")
        with pytest.raises(ConfigurationError):
            read_user_config()


# --------------------------------------------------------------------------- #
# Resolution order                                                             #
# --------------------------------------------------------------------------- #


class TestResolveResultsDir:
    def test_nothing_configured_resolves_to_none(self, xdg):
        from mlpstorage_py.results_dir.user_config import resolve_results_dir

        assert resolve_results_dir(None) == (None, None)

    def test_config_file_is_the_last_tier(self, xdg, tmp_path):
        from mlpstorage_py.results_dir.user_config import (
            record_results_dir,
            resolve_results_dir,
            user_config_path,
        )

        record_results_dir(str(tmp_path / "cfg"))
        assert resolve_results_dir(None) == (str(tmp_path / "cfg"), user_config_path())

    def test_env_beats_config_file(self, xdg, tmp_path, monkeypatch):
        from mlpstorage_py.results_dir.user_config import (
            record_results_dir,
            resolve_results_dir,
        )

        record_results_dir(str(tmp_path / "cfg"))
        monkeypatch.setenv("MLPSTORAGE_RESULTS_DIR", str(tmp_path / "env"))
        assert resolve_results_dir(None) == (str(tmp_path / "env"), "MLPSTORAGE_RESULTS_DIR")

    def test_flag_beats_env_and_config(self, xdg, tmp_path, monkeypatch):
        from mlpstorage_py.results_dir.user_config import (
            record_results_dir,
            resolve_results_dir,
        )

        record_results_dir(str(tmp_path / "cfg"))
        monkeypatch.setenv("MLPSTORAGE_RESULTS_DIR", str(tmp_path / "env"))
        assert resolve_results_dir(str(tmp_path / "flag")) == (
            str(tmp_path / "flag"),
            "--results-dir",
        )

    def test_config_value_expands_tilde(self, xdg, tmp_path, monkeypatch):
        from mlpstorage_py.results_dir.user_config import resolve_results_dir

        monkeypatch.setenv("HOME", str(tmp_path))
        cfg = xdg / "mlpstorage" / "config.yaml"
        cfg.parent.mkdir()
        cfg.write_text("results_dir: ~/mine\n")
        path, _ = resolve_results_dir(None)
        assert path == str(tmp_path / "mine")


# --------------------------------------------------------------------------- #
# init: optional path, records the default                                     #
# --------------------------------------------------------------------------- #


class TestInitRecordsDefault:
    def test_init_records_results_dir_in_user_config(self, xdg, tmp_path):
        from mlpstorage_py.config import EXIT_CODE
        from mlpstorage_py.results_dir.user_config import read_user_config

        target = tmp_path / "results"
        assert _init("Acme", str(target)) == EXIT_CODE.SUCCESS
        assert read_user_config()["results_dir"] == str(target)

    def test_idempotent_reinit_also_records(self, xdg, tmp_path):
        """Re-running init on an already-initialized tree is how a user
        switches their recorded default between two trees."""
        from mlpstorage_py.results_dir.user_config import (
            read_user_config,
            record_results_dir,
        )

        target = tmp_path / "results"
        _init("Acme", str(target))
        record_results_dir(str(tmp_path / "elsewhere"))
        _init("Acme", str(target))
        assert read_user_config()["results_dir"] == str(target)

    def test_refused_init_does_not_touch_config(self, xdg, tmp_path):
        from mlpstorage_py.results_dir.errors import DoubleInitError
        from mlpstorage_py.results_dir.user_config import read_user_config

        target = tmp_path / "results"
        _init("Acme", str(target))
        other = tmp_path / "other"
        _init("Acme", str(other))
        with pytest.raises(DoubleInitError):
            _init("Globex", str(target))
        assert read_user_config()["results_dir"] == str(other)

    def test_init_without_path_uses_home_default(self, xdg, tmp_path, monkeypatch):
        from mlpstorage_py.config import EXIT_CODE
        from mlpstorage_py.results_dir import MLPERF_RESULTS_FILENAME
        from mlpstorage_py.results_dir.user_config import (
            DEFAULT_RESULTS_DIR,
            read_user_config,
        )

        monkeypatch.setenv("HOME", str(tmp_path))
        assert DEFAULT_RESULTS_DIR == "~/mlpstorage-results"
        assert _init("Acme") == EXIT_CODE.SUCCESS
        default = tmp_path / "mlpstorage-results"
        assert (default / MLPERF_RESULTS_FILENAME).is_file()
        assert read_user_config()["results_dir"] == str(default)

    def test_init_parser_path_is_optional(self):
        from mlpstorage_py.cli_parser import build_parser

        args = build_parser().parse_args(["init", "Acme"])
        assert args.mode == "init"
        assert args.orgname == "Acme"
        assert args.path is None

    def test_init_confirms_recorded_default(self, xdg, tmp_path, caplog):
        import logging

        target = tmp_path / "results"
        with caplog.at_level(logging.INFO, logger="mlpstorage_py"):
            _init("Acme", str(target))
        assert any(
            str(target) in r.getMessage() and "config.yaml" in r.getMessage()
            for r in caplog.records
        ), [r.getMessage() for r in caplog.records]


# --------------------------------------------------------------------------- #
# Sentinel version 3                                                           #
# --------------------------------------------------------------------------- #


class TestSentinelVersionThree:
    def test_current_version_is_three(self):
        from mlpstorage_py.results_dir import MLPERF_RESULTS_VERSION

        assert MLPERF_RESULTS_VERSION == 3

    def test_version_one_trees_still_resolve(self, tmp_path):
        """v3.0-era trees carry version 1; they are read identically."""
        from mlpstorage_py.results_dir import resolve_orgname

        (tmp_path / "mlperf-results.yaml").write_text(
            "mlperf_results_version: 1\norgname: Acme\n"
            "initialized_at: '2026-01-01T00:00:00+00:00'\n"
            "initialized_by: mlpstorage 3.0.46\n"
        )
        assert resolve_orgname(str(tmp_path)) == "Acme"


# --------------------------------------------------------------------------- #
# CLI: every benchmark command requires a resolved results-dir (GAP A)         #
# --------------------------------------------------------------------------- #

_DATASIZE_ARGV = {
    "training": ["closed", "training", "unet3d", "datasize",
                 "--accelerator-type", "b200", "--max-accelerators", "8",
                 "--client-host-memory-in-gb", "64", "--systemname", "s"],
    "checkpointing": ["closed", "checkpointing", "datasize",
                      "--model", "llama3-8b", "--client-host-memory-in-gb", "64",
                      "--num-processes", "8"],
    "vectordb": ["closed", "vectordb", "datasize"],
    "kvcache": ["closed", "kvcache", "datasize"],
}


class TestEveryBenchmarkCommandNeedsResultsDir:
    @pytest.mark.parametrize("benchmark", sorted(_DATASIZE_ARGV))
    def test_datasize_without_any_results_dir_is_a_loud_error(
        self, xdg, benchmark, capsys, monkeypatch
    ):
        from mlpstorage_py.cli import common_args as common_args_mod
        from mlpstorage_py.cli_parser import parse_arguments
        from mlpstorage_py.config import EXIT_CODE

        monkeypatch.setattr(common_args_mod, "ENV_FALLBACK_RESULTS_DIR", "")
        monkeypatch.setattr("sys.argv", ["mlpstorage"] + _DATASIZE_ARGV[benchmark])
        with pytest.raises(SystemExit) as exc:
            parse_arguments()
        assert exc.value.code == EXIT_CODE.INVALID_ARGUMENTS
        err = capsys.readouterr().err
        assert "error: --results-dir/-rd is required" in err
        assert "mlpstorage init" in err
        assert "MLPSTORAGE_RESULTS_DIR" in err

    @pytest.mark.parametrize("benchmark", sorted(_DATASIZE_ARGV))
    def test_datasize_resolves_from_user_config(
        self, xdg, benchmark, tmp_path, monkeypatch
    ):
        from mlpstorage_py.cli import common_args as common_args_mod
        from mlpstorage_py.cli_parser import parse_arguments
        from mlpstorage_py.results_dir.user_config import (
            record_results_dir,
            user_config_path,
        )

        monkeypatch.setattr(common_args_mod, "ENV_FALLBACK_RESULTS_DIR", "")
        record_results_dir(str(tmp_path / "cfg"))
        monkeypatch.setattr("sys.argv", ["mlpstorage"] + _DATASIZE_ARGV[benchmark])
        args = parse_arguments()
        assert args.results_dir == str(tmp_path / "cfg")
        assert args.results_dir_source == user_config_path()

    def test_flag_on_command_line_is_labelled_as_the_flag(self, xdg, tmp_path, monkeypatch):
        from mlpstorage_py.cli_parser import parse_arguments
        from mlpstorage_py.results_dir.user_config import record_results_dir

        record_results_dir(str(tmp_path / "cfg"))
        monkeypatch.setattr(
            "sys.argv",
            ["mlpstorage"] + _DATASIZE_ARGV["kvcache"] + ["-rd", str(tmp_path / "flag")],
        )
        args = parse_arguments()
        assert args.results_dir == str(tmp_path / "flag")
        assert args.results_dir_source == "--results-dir"

    def test_env_var_is_read_at_parse_time_and_labelled(self, xdg, tmp_path, monkeypatch):
        from mlpstorage_py.cli import common_args as common_args_mod
        from mlpstorage_py.cli_parser import parse_arguments

        monkeypatch.setattr(common_args_mod, "ENV_FALLBACK_RESULTS_DIR", "")
        monkeypatch.setenv("MLPSTORAGE_RESULTS_DIR", str(tmp_path / "env"))
        monkeypatch.setattr("sys.argv", ["mlpstorage"] + _DATASIZE_ARGV["kvcache"])
        args = parse_arguments()
        assert args.results_dir == str(tmp_path / "env")
        assert args.results_dir_source == "MLPSTORAGE_RESULTS_DIR"

    def test_reportgen_resolves_from_user_config(self, xdg, tmp_path, monkeypatch):
        from mlpstorage_py.cli import common_args as common_args_mod
        from mlpstorage_py.cli_parser import parse_arguments
        from mlpstorage_py.results_dir.user_config import record_results_dir

        monkeypatch.setattr(common_args_mod, "ENV_FALLBACK_RESULTS_DIR", "")
        record_results_dir(str(tmp_path / "cfg"))
        monkeypatch.setattr("sys.argv", ["mlpstorage", "reports", "reportgen"])
        args = parse_arguments()
        assert args.results_dir == str(tmp_path / "cfg")


# --------------------------------------------------------------------------- #
# main: status line + history inside the results-dir (GAP B)                   #
# --------------------------------------------------------------------------- #


class _Stop(Exception):
    pass


def _run_main_until_gate(argv):
    """Drive ``_main_impl`` through argument parsing, the results-dir status
    line, history recording and the LAY-03 gate; stop before any benchmark
    plumbing runs."""
    from mlpstorage_py import main as main_mod

    with patch("sys.argv", argv), \
         patch.object(main_mod, "update_args", side_effect=_Stop), \
         patch.object(main_mod, "apply_logging_options"):
        try:
            main_mod._main_impl()
        except _Stop:
            pass


class TestMainResolution:
    def test_status_line_names_the_source(self, xdg, tmp_path, caplog):
        import logging

        from mlpstorage_py.results_dir.user_config import user_config_path

        target = tmp_path / "results"
        _init("Acme", str(target))
        with caplog.at_level(logging.INFO, logger="MLPerfStorage"):
            _run_main_until_gate(["mlpstorage"] + _DATASIZE_ARGV["kvcache"])
        lines = [r.getMessage() for r in caplog.records]
        assert any(
            f"results-dir: {target} (from {user_config_path()})" in line
            for line in lines
        ), lines

    def test_history_lives_inside_the_results_dir(self, xdg, tmp_path):
        from mlpstorage_py.history import history_file_for

        target = tmp_path / "results"
        _init("Acme", str(target))
        _run_main_until_gate(["mlpstorage"] + _DATASIZE_ARGV["kvcache"])

        hist = target / ".mlps" / "history"
        assert history_file_for(str(target)) == str(hist)
        assert hist.is_file()
        assert "kvcache datasize" in hist.read_text()

    def test_no_history_written_into_an_uninitialized_dir(self, xdg, tmp_path, monkeypatch):
        """Writing ``.mlps/`` into an un-initialized dir would make it
        non-empty and block the very ``init`` the error tells the user to run."""
        from mlpstorage_py.errors import ConfigurationError

        uninit = tmp_path / "uninit"
        uninit.mkdir()
        monkeypatch.setenv("MLPSTORAGE_RESULTS_DIR", str(uninit))
        with pytest.raises(ConfigurationError, match="has not been initialized"):
            _run_main_until_gate(["mlpstorage"] + _DATASIZE_ARGV["kvcache"])
        assert list(uninit.iterdir()) == []

    def test_history_show_resolves_results_dir_without_a_flag(self, xdg, tmp_path, capsys):
        from mlpstorage_py import main as main_mod

        target = tmp_path / "results"
        _init("Acme", str(target))
        _run_main_until_gate(["mlpstorage"] + _DATASIZE_ARGV["kvcache"])

        with patch("sys.argv", ["mlpstorage", "history", "show"]), \
             patch.object(main_mod, "apply_logging_options"):
            rc = main_mod._main_impl()
        assert rc == 0
        assert "kvcache datasize" in capsys.readouterr().out

    def test_history_show_without_any_results_dir_is_actionable(self, xdg):
        from mlpstorage_py import main as main_mod
        from mlpstorage_py.errors import ConfigurationError

        with patch("sys.argv", ["mlpstorage", "history", "show"]), \
             patch.object(main_mod, "apply_logging_options"), \
             pytest.raises(ConfigurationError, match="mlpstorage init"):
            main_mod._main_impl()

    def test_no_global_history_file_constant_remains(self):
        from mlpstorage_py import config, history, main

        for mod in (config, history, main):
            assert not hasattr(mod, "HISTFILE"), mod.__name__


# --------------------------------------------------------------------------- #
# Orgname comes only from the sentinel                                         #
# --------------------------------------------------------------------------- #


class TestOrgnameEnvVarGone:
    def test_config_no_longer_declares_the_env_var(self):
        from mlpstorage_py import config

        assert not hasattr(config, "MLPSTORAGE_ORGNAME_ENVVAR")
        assert "MLPSTORAGE_ORGNAME" not in config.MANPAGE_ENV_VAR_TIERS
        assert "MLPSTORAGE_ORGNAME" not in config._LEGACY_ENVVAR_MAP.values()
        assert "MLPSTORAGE_ORGNAME" not in config._LEGACY_ENVVAR_MAP

    def test_code_image_ignores_the_env_var(self, tmp_path):
        from types import SimpleNamespace

        from mlpstorage_py.errors import ConfigurationError
        from mlpstorage_py.submission_checker.tools.code_image import (
            capture_or_verify_code_image,
        )

        args = SimpleNamespace(
            mode="closed", command="datagen", results_dir=str(tmp_path),
            benchmark="training", model="unet3d", orgname=None,
        )
        with pytest.raises(ConfigurationError) as exc:
            capture_or_verify_code_image(
                args, {"MLPSTORAGE_ORGNAME": "acme"}, _quiet_logger()
            )
        assert "MLPSTORAGE_ORGNAME" not in str(exc.value)
        assert "mlpstorage init" in (exc.value.suggestion or "")

    def test_manpage_has_no_orgname_env_row(self):
        text = open(os.path.join(_repo_root(), "ManPage.md"), encoding="utf-8").read()
        assert not re.search(r"^\|\s*`MLPSTORAGE_ORGNAME`", text, re.M)


def _quiet_logger():
    import logging

    return logging.getLogger("test_results_dir_resolver.quiet")


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
