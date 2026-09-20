"""
Per-user config file: environment-describing defaults beyond ``results_dir``.

``~/.config/mlpstorage/config.yaml`` (written by ``mlpstorage init``, hand
edits kept) may now carry defaults for every flag that describes the
*environment* the benchmark runs in — never one that picks the *workload*.
One layered resolver applies every tier in a single order for every key:

    command-line flag  >  --config-file YAML  >  MLPSTORAGE_* env var
                       >  ~/.config/mlpstorage/config.yaml  >  built-in default

That fold also fixes ``--config-file``: it used to be applied AFTER argparse
and silently overwrote flags typed on the command line, the opposite of what
the README promised.

Keys the per-user file may carry (all optional, all only where the leaf
command has the flag): ``results_dir``, ``systemname``, ``data_dir``,
``checkpoint_folder``, ``hosts``, ``mpi_bin``, ``mpi_btl``, ``oversubscribe``,
``allow_run_as_root``, ``mpi_params``, ``dlio_bin_path``, ``exec_type``,
``color``, ``stream_log_level``. A workload-selecting key (``model``,
``accelerator_type``, ``num_accelerators``, ...) in that file is a hard error
so a stale file can never reshape a run; an unknown key is a warning.
"""
from __future__ import annotations

import logging
import os
from unittest.mock import patch

import pytest
import yaml


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    """Fresh XDG_CONFIG_HOME; no MLPSTORAGE_* env vars leak in."""
    home = tmp_path / "xdg"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    for name in (
        "MLPSTORAGE_RESULTS_DIR", "MLPSTORAGE_SYSTEMNAME",
        "MLPSTORAGE_DATA_DIR", "MLPSTORAGE_CHECKPOINT_FOLDER",
    ):
        monkeypatch.delenv(name, raising=False)
    # argparse captured the env vars at import time; neutralise that copy.
    from mlpstorage_py.cli import common_args, training_args, checkpointing_args

    monkeypatch.setattr(common_args, "ENV_FALLBACK_RESULTS_DIR", "")
    monkeypatch.setattr(common_args, "ENV_FALLBACK_SYSTEMNAME", "")
    monkeypatch.setattr(training_args, "ENV_FALLBACK_DATA_DIR", "")
    monkeypatch.setattr(checkpointing_args, "ENV_FALLBACK_CHECKPOINT_FOLDER", "")
    return home


def _write_user_config(values):
    from mlpstorage_py.results_dir.user_config import user_config_path, write_user_config

    write_user_config(values)
    return user_config_path()


def _parse(argv, monkeypatch):
    from mlpstorage_py.cli_parser import parse_arguments

    monkeypatch.setattr("sys.argv", ["mlpstorage"] + argv)
    return parse_arguments()


TRAIN_RUN = ["closed", "training", "unet3d", "run", "file",
             "-na", "2", "-at", "b200", "-cm", "64", "-dd", "/data"]
TRAIN_DATASIZE = ["closed", "training", "unet3d", "datasize",
                  "-at", "b200", "--max-accelerators", "8", "-cm", "64"]
CKPT_RUN = ["closed", "checkpointing", "run", "file",
            "-m", "llama3-8b", "-at", "b200", "-np", "8", "-cm", "64"]
KV_DATASIZE = ["closed", "kvcache", "datasize"]
REPORTGEN = ["reports", "reportgen"]


# --------------------------------------------------------------------------- #
# The probe: which dests did the command line actually supply?                 #
# --------------------------------------------------------------------------- #


class TestExplicitDests:
    def test_store_flags_and_positionals(self):
        from mlpstorage_py.cli.config_layers import explicit_dests

        got = explicit_dests(TRAIN_RUN + ["-rd", "/r", "-sn", "s", "--hosts", "a", "b"])
        assert {"mode", "benchmark", "model", "command", "data_access_protocol",
                "num_accelerators", "accelerator_type", "client_host_memory_in_gb",
                "data_dir", "results_dir", "systemname", "hosts"} <= got
        # Nothing the command line did not name.
        assert not {"mpi_bin", "mpi_btl", "oversubscribe", "color",
                    "stream_log_level", "dlio_bin_path", "exec_type",
                    "loops", "allow_invalid_params"} & got

    def test_store_true_and_append_actions(self):
        from mlpstorage_py.cli.config_layers import explicit_dests

        got = explicit_dests(CKPT_RUN + ["-rd", "/r", "-sn", "s", "-cf", "/c",
                                         "--oversubscribe", "--mpi-params=-x y",
                                         "--verbose"])
        assert {"oversubscribe", "mpi_params", "verbose", "checkpoint_folder"} <= got
        assert not {"allow_run_as_root", "debug", "mpi_bin"} & got

    def test_utility_commands(self):
        from mlpstorage_py.cli.config_layers import explicit_dests

        assert "systemname" not in explicit_dests(REPORTGEN)
        assert "systemname" in explicit_dests(REPORTGEN + ["-sn", "s"])


# --------------------------------------------------------------------------- #
# The per-user file supplies environment defaults                              #
# --------------------------------------------------------------------------- #


class TestUserConfigSuppliesDefaults:
    def test_systemname(self, xdg, tmp_path, monkeypatch):
        path = _write_user_config({"results_dir": str(tmp_path), "systemname": "lab-a"})
        args = _parse(KV_DATASIZE, monkeypatch)
        assert args.systemname == "lab-a"
        assert args.config_sources["systemname"] == path
        assert args.config_sources["results_dir"] == path

    def test_hosts_as_list_and_as_comma_string(self, xdg, tmp_path, monkeypatch):
        _write_user_config({"results_dir": str(tmp_path), "systemname": "s",
                            "hosts": ["n1", "n2:4"]})
        args = _parse(TRAIN_RUN, monkeypatch)
        assert args.hosts == ["n1", "n2:4"]
        assert args.num_client_hosts == 2

        _write_user_config({"hosts": "n1,n2,n3"})
        args = _parse(TRAIN_RUN, monkeypatch)
        assert args.hosts == ["n1", "n2", "n3"]

    def test_mpi_settings(self, xdg, tmp_path, monkeypatch):
        _write_user_config({
            "results_dir": str(tmp_path), "systemname": "s",
            "mpi_bin": "mpiexec", "mpi_btl": "tcp", "oversubscribe": True,
            "allow_run_as_root": True, "mpi_params": "-genv FI_PROVIDER=tcp",
        })
        args = _parse(TRAIN_RUN, monkeypatch)
        assert args.mpi_bin == "mpiexec"
        assert args.mpi_btl == "tcp"
        assert args.oversubscribe is True
        assert args.allow_run_as_root is True
        assert args.mpi_params == ["-genv FI_PROVIDER=tcp"]

    def test_mpi_params_list_form(self, xdg, tmp_path, monkeypatch):
        _write_user_config({"results_dir": str(tmp_path), "systemname": "s",
                            "mpi_params": ["-x A=1", "-x B=2"]})
        args = _parse(TRAIN_RUN, monkeypatch)
        assert args.mpi_params == ["-x A=1", "-x B=2"]

    def test_dlio_bin_path_and_exec_type_are_type_converted(self, xdg, tmp_path, monkeypatch):
        from mlpstorage_py.config import EXEC_TYPE

        _write_user_config({"results_dir": str(tmp_path), "systemname": "s",
                            "dlio_bin_path": "/opt/dlio/bin", "exec_type": "docker"})
        args = _parse(TRAIN_RUN, monkeypatch)
        assert args.dlio_bin_path == "/opt/dlio/bin"
        assert args.exec_type is EXEC_TYPE.DOCKER

    def test_color_and_stream_log_level(self, xdg, tmp_path, monkeypatch):
        _write_user_config({"results_dir": str(tmp_path), "systemname": "s",
                            "color": "never", "stream_log_level": "DEBUG"})
        args = _parse(KV_DATASIZE, monkeypatch)
        assert args.color == "never"
        assert args.stream_log_level == "DEBUG"

    def test_data_dir_satisfies_file_mode_training(self, xdg, tmp_path, monkeypatch):
        _write_user_config({"results_dir": str(tmp_path), "systemname": "s",
                            "data_dir": "/mnt/unet3d"})
        argv = [a for a in TRAIN_RUN if a not in ("-dd", "/data")]
        args = _parse(argv, monkeypatch)
        assert args.data_dir == "/mnt/unet3d"

    def test_checkpoint_folder_satisfies_checkpointing_run(self, xdg, tmp_path, monkeypatch):
        _write_user_config({"results_dir": str(tmp_path), "systemname": "s",
                            "checkpoint_folder": "/mnt/ckpt"})
        args = _parse(CKPT_RUN, monkeypatch)
        assert args.checkpoint_folder == "/mnt/ckpt"

    def test_keys_the_leaf_lacks_are_ignored(self, xdg, tmp_path, monkeypatch):
        # reportgen has no --hosts / --mpi-btl; the file is still valid there.
        _write_user_config({"results_dir": str(tmp_path), "hosts": ["n1"],
                            "mpi_btl": "tcp", "checkpoint_folder": "/c"})
        args = _parse(REPORTGEN, monkeypatch)
        assert not hasattr(args, "hosts")
        assert "hosts" not in args.config_sources

    def test_unknown_key_is_a_warning_not_an_error(self, xdg, tmp_path, monkeypatch, capsys):
        _write_user_config({"results_dir": str(tmp_path), "systemname": "s",
                            "sytemname": "typo"})
        args = _parse(KV_DATASIZE, monkeypatch)
        assert args.systemname == "s"
        err = capsys.readouterr().err
        assert "sytemname" in err and "warning" in err.lower()


# --------------------------------------------------------------------------- #
# Guards: the per-user file never picks the workload                           #
# --------------------------------------------------------------------------- #


class TestUserConfigGuards:
    @pytest.mark.parametrize("key,value", [
        ("model", "retinanet"),
        ("accelerator_type", "mi355"),
        ("num_accelerators", 8),
        ("client_host_memory_in_gb", 512),
        ("mode", "open"),
        ("params", {"reader.read_threads": 8}),
    ])
    def test_workload_key_is_a_hard_error(self, xdg, tmp_path, monkeypatch, capsys, key, value):
        from mlpstorage_py.config import EXIT_CODE

        path = _write_user_config({"results_dir": str(tmp_path), "systemname": "s", key: value})
        with pytest.raises(SystemExit) as exc:
            _parse(TRAIN_RUN, monkeypatch)
        assert exc.value.code == EXIT_CODE.INVALID_ARGUMENTS
        err = capsys.readouterr().err
        assert key in err and path in err
        assert "--config-file" in err  # where such a key IS allowed

    def test_invalid_choice_is_a_hard_error(self, xdg, tmp_path, monkeypatch, capsys):
        from mlpstorage_py.config import EXIT_CODE

        _write_user_config({"results_dir": str(tmp_path), "systemname": "s", "mpi_btl": "fast"})
        with pytest.raises(SystemExit) as exc:
            _parse(TRAIN_RUN, monkeypatch)
        assert exc.value.code == EXIT_CODE.INVALID_ARGUMENTS
        err = capsys.readouterr().err
        assert "mpi_btl" in err and "fast" in err and "tcp" in err

    def test_wrong_type_is_a_hard_error(self, xdg, tmp_path, monkeypatch, capsys):
        from mlpstorage_py.config import EXIT_CODE

        _write_user_config({"results_dir": str(tmp_path), "systemname": "s",
                            "oversubscribe": "yes please"})
        with pytest.raises(SystemExit) as exc:
            _parse(TRAIN_RUN, monkeypatch)
        assert exc.value.code == EXIT_CODE.INVALID_ARGUMENTS
        assert "oversubscribe" in capsys.readouterr().err

    def test_allowlist_is_environment_only(self):
        from mlpstorage_py.results_dir.user_config import USER_CONFIG_KEYS

        assert set(USER_CONFIG_KEYS) == {
            "results_dir", "systemname", "data_dir", "checkpoint_folder",
            "hosts", "mpi_bin", "mpi_btl", "oversubscribe", "allow_run_as_root",
            "mpi_params", "dlio_bin_path", "exec_type", "color", "stream_log_level",
        }


# --------------------------------------------------------------------------- #
# One precedence order for every key                                           #
# --------------------------------------------------------------------------- #


class TestPrecedence:
    def test_flag_beats_user_config(self, xdg, tmp_path, monkeypatch):
        _write_user_config({"results_dir": str(tmp_path), "systemname": "from-file",
                            "mpi_btl": "tcp", "oversubscribe": True})
        args = _parse(TRAIN_RUN + ["-sn", "from-flag", "--mpi-btl", "vader"], monkeypatch)
        assert args.systemname == "from-flag"
        assert args.config_sources["systemname"] == "--systemname"
        assert args.mpi_btl == "vader"
        assert args.oversubscribe is True  # file still fills what the flag left

    def test_env_beats_user_config(self, xdg, tmp_path, monkeypatch):
        _write_user_config({"results_dir": str(tmp_path), "systemname": "from-file",
                            "data_dir": "/from-file", "checkpoint_folder": "/from-file"})
        monkeypatch.setenv("MLPSTORAGE_SYSTEMNAME", "from-env")
        monkeypatch.setenv("MLPSTORAGE_CHECKPOINT_FOLDER", "/from-env")
        args = _parse(CKPT_RUN, monkeypatch)
        assert args.systemname == "from-env"
        assert args.config_sources["systemname"] == "MLPSTORAGE_SYSTEMNAME"
        assert args.checkpoint_folder == "/from-env"
        assert args.config_sources["checkpoint_folder"] == "MLPSTORAGE_CHECKPOINT_FOLDER"

    def test_env_is_read_at_parse_time_for_every_env_backed_key(self, xdg, tmp_path, monkeypatch):
        _write_user_config({"results_dir": str(tmp_path)})
        monkeypatch.setenv("MLPSTORAGE_SYSTEMNAME", "from-env")
        monkeypatch.setenv("MLPSTORAGE_DATA_DIR", "/from-env")
        argv = [a for a in TRAIN_RUN if a not in ("-dd", "/data")]
        args = _parse(argv, monkeypatch)
        assert args.systemname == "from-env"
        assert args.data_dir == "/from-env"
        assert args.config_sources["data_dir"] == "MLPSTORAGE_DATA_DIR"

    def test_config_file_flag_beats_env_and_user_config(self, xdg, tmp_path, monkeypatch):
        _write_user_config({"results_dir": str(tmp_path), "systemname": "from-file",
                            "mpi_btl": "tcp"})
        monkeypatch.setenv("MLPSTORAGE_SYSTEMNAME", "from-env")
        override = tmp_path / "run.yaml"
        override.write_text("systemname: from-c\nmpi_btl: vader\n")
        args = _parse(TRAIN_RUN + ["-c", str(override)], monkeypatch)
        assert args.systemname == "from-c"
        assert args.config_sources["systemname"] == "--config-file"
        assert args.mpi_btl == "vader"

    def test_flag_beats_config_file(self, xdg, tmp_path, monkeypatch):
        """The fold: --config-file no longer overwrites what was typed."""
        _write_user_config({"results_dir": str(tmp_path)})
        override = tmp_path / "run.yaml"
        override.write_text("systemname: from-c\nnum_accelerators: 16\nmpi_btl: tcp\n")
        args = _parse(TRAIN_RUN + ["-c", str(override), "-sn", "from-flag"], monkeypatch)
        assert args.systemname == "from-flag"
        assert args.num_accelerators == 2      # -na 2 on the command line wins
        assert args.mpi_btl == "tcp"           # untyped: the file fills it

    def test_config_file_keeps_workload_keys(self, xdg, tmp_path, monkeypatch):
        """--config-file is explicit per invocation, so the README's
        repeatable-knobs example (workload keys + params) stays valid."""
        _write_user_config({"results_dir": str(tmp_path), "systemname": "s"})
        override = tmp_path / "run.yaml"
        override.write_text(yaml.safe_dump({
            "accelerator_type": "mi355", "hosts": ["h1", "h2"],
            "params": {"dataset.num_files_train": 42000},
        }))
        argv = [a for a in TRAIN_RUN if a not in ("-at", "b200")]
        args = _parse(argv + ["-c", str(override)], monkeypatch)
        assert args.accelerator_type == "mi355"
        assert args.hosts == ["h1", "h2"]
        assert args.params == ["dataset.num_files_train=42000"]

    def test_config_file_supplies_data_dir_for_file_mode(self, xdg, tmp_path, monkeypatch):
        _write_user_config({"results_dir": str(tmp_path), "systemname": "s"})
        override = tmp_path / "run.yaml"
        override.write_text("data_dir: /mnt/from-c\n")
        argv = [a for a in TRAIN_RUN if a not in ("-dd", "/data")]
        args = _parse(argv + ["-c", str(override)], monkeypatch)
        assert args.data_dir == "/mnt/from-c"

    def test_results_dir_source_label_is_unchanged(self, xdg, tmp_path, monkeypatch):
        path = _write_user_config({"results_dir": str(tmp_path), "systemname": "s"})
        args = _parse(KV_DATASIZE, monkeypatch)
        assert args.results_dir == str(tmp_path)
        assert args.results_dir_source == path
        monkeypatch.setenv("MLPSTORAGE_RESULTS_DIR", str(tmp_path / "env"))
        args = _parse(KV_DATASIZE, monkeypatch)
        assert args.results_dir_source == "MLPSTORAGE_RESULTS_DIR"


# --------------------------------------------------------------------------- #
# Never silent: main says which keys the files supplied                        #
# --------------------------------------------------------------------------- #


class _Stop(Exception):
    pass


def _run_main_until_gate(argv):
    from mlpstorage_py import main as main_mod

    with patch("sys.argv", argv), \
         patch.object(main_mod, "update_args", side_effect=_Stop), \
         patch.object(main_mod, "apply_logging_options"):
        try:
            main_mod._main_impl()
        except _Stop:
            pass


class TestStatusLine:
    def test_main_names_the_keys_each_file_supplied(self, xdg, tmp_path, caplog):
        from mlpstorage_py.results_dir.init import run_init
        from argparse import Namespace

        target = tmp_path / "results"
        run_init(Namespace(mode="init", orgname="Acme", path=str(target)))
        path = _write_user_config({"systemname": "lab-a", "color": "never"})
        with caplog.at_level(logging.INFO, logger="MLPerfStorage"):
            _run_main_until_gate(["mlpstorage"] + KV_DATASIZE)
        lines = [r.getMessage() for r in caplog.records]
        assert any(f"results-dir: {target} (from {path})" in l for l in lines), lines
        assert any(f"defaults from {path}: color, systemname" in l for l in lines), lines

    def test_no_line_when_the_file_supplied_nothing_else(self, xdg, tmp_path, caplog):
        from mlpstorage_py.results_dir.init import run_init
        from argparse import Namespace

        target = tmp_path / "results"
        run_init(Namespace(mode="init", orgname="Acme", path=str(target)))
        with caplog.at_level(logging.INFO, logger="MLPerfStorage"):
            _run_main_until_gate(["mlpstorage"] + KV_DATASIZE + ["-sn", "s"])
        lines = [r.getMessage() for r in caplog.records]
        assert not any("defaults from" in l for l in lines), lines


# --------------------------------------------------------------------------- #
# Documentation stays in step                                                  #
# --------------------------------------------------------------------------- #


class TestDocs:
    def test_manpage_lists_every_user_config_key(self):
        from mlpstorage_py.results_dir.user_config import USER_CONFIG_KEYS

        with open("ManPage.md", encoding="utf-8") as fh:
            text = fh.read()
        for key in USER_CONFIG_KEYS:
            assert f"`{key}`" in text, key
        assert "--config-file YAML  >  MLPSTORAGE_" in text or \
               "`--config-file` > `MLPSTORAGE_" in text

    def test_help_all_states_the_single_precedence_order(self):
        from mlpstorage_py.cli.help_formatter import HELP_ALL_TEXT

        assert "applied after CLI args" not in HELP_ALL_TEXT
        assert "--config-file" in HELP_ALL_TEXT and "config.yaml" in HELP_ALL_TEXT

    def test_readme_no_longer_claims_yaml_after_cli(self):
        with open("README.md", encoding="utf-8") as fh:
            text = fh.read()
        assert "loaded **after** the CLI arguments" not in text
        assert "argparse defaults  >  environment variables" not in text

    def test_sample_file_says_what_it_is(self):
        with open("mlpstorage.yaml", encoding="utf-8") as fh:
            text = fh.read()
        assert "--config-file" in text
