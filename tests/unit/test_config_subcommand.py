"""
``mlpstorage config`` — manage the per-user config file without an editor.

``~/.config/mlpstorage/config.yaml`` (``$XDG_CONFIG_HOME`` honoured) carries
environment-describing defaults (:data:`USER_CONFIG_KEYS`). Until now only
``mlpstorage init`` wrote it, and only ``results_dir``; every other key was a
hand edit. Four subcommands, none of which needs a results-dir:

    mlpstorage config show  [--json]      every allowed key: file value or [unset],
                                          plus the MLPSTORAGE_* env var that outranks it
    mlpstorage config set   KEY VALUE...  allowlist-, type- and choice-checked as the
                                          flag would be; several values store a list
    mlpstorage config unset KEY           drop one key, keep the rest
    mlpstorage config path                print the file path

``config`` bypasses the tier resolver (``cli.config_layers``) that every
other command runs at parse time: that resolver hard-fails on a workload key
in the file, and ``config unset`` is how you remove one.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import yaml

from mlpstorage_py.config import EXIT_CODE


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    home = tmp_path / "xdg"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    for name in (
        "MLPSTORAGE_RESULTS_DIR", "MLPSTORAGE_SYSTEMNAME",
        "MLPSTORAGE_DATA_DIR", "MLPSTORAGE_CHECKPOINT_FOLDER",
    ):
        monkeypatch.delenv(name, raising=False)
    return home


def _cfg_path():
    from mlpstorage_py.results_dir.user_config import user_config_path
    return user_config_path()


def _write(values):
    from mlpstorage_py.results_dir.user_config import write_user_config
    return write_user_config(values)


def _read():
    with open(_cfg_path(), encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _run(argv, capsys):
    """Dispatch through the real entry point; return (exit code, stdout, stderr)."""
    from mlpstorage_py.results_dir.config_cmd import run_config_command
    from mlpstorage_py.cli_parser import build_parser

    args = build_parser().parse_args(["config"] + argv)
    rc = run_config_command(args, logger=None)
    out = capsys.readouterr()
    return rc, out.out, out.err


# --------------------------------------------------------------------------- #
# Parser surface                                                               #
# --------------------------------------------------------------------------- #


class TestParser:
    def test_four_subcommands(self):
        from mlpstorage_py.cli_parser import build_parser

        p = build_parser()
        assert p.parse_args(["config", "show"]).command == "show"
        assert p.parse_args(["config", "show", "--json"]).json is True
        a = p.parse_args(["config", "set", "hosts", "n1", "n2"])
        assert (a.mode, a.command, a.key, a.values) == ("config", "set", "hosts", ["n1", "n2"])
        assert p.parse_args(["config", "unset", "hosts"]).key == "hosts"
        assert p.parse_args(["config", "path"]).command == "path"

    def test_set_needs_a_value(self):
        from mlpstorage_py.cli_parser import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["config", "set", "hosts"])

    def test_context_help_lists_subcommands(self):
        from mlpstorage_py.cli.help_formatter import get_context_help_tokens

        assert get_context_help_tokens(["config"]) == "next: show | set | unset | path"
        assert get_context_help_tokens(["config", "show"]) is None

    def test_parse_arguments_skips_the_tier_resolver(self, xdg, monkeypatch):
        """A workload key in the file fails every other command at parse
        time; ``config`` must still parse so ``unset`` can remove it."""
        from mlpstorage_py.cli_parser import parse_arguments

        _write({"model": "unet3d"})
        monkeypatch.setattr("sys.argv", ["mlpstorage", "config", "unset", "model"])
        args = parse_arguments()
        assert args.mode == "config" and args.key == "model"
        assert not hasattr(args, "config_sources")

        monkeypatch.setattr("sys.argv", ["mlpstorage", "version"])
        with pytest.raises(SystemExit):
            parse_arguments()


# --------------------------------------------------------------------------- #
# path                                                                         #
# --------------------------------------------------------------------------- #


class TestPath:
    def test_prints_the_path(self, xdg, capsys):
        rc, out, _ = _run(["path"], capsys)
        assert rc == EXIT_CODE.SUCCESS
        assert out.strip() == str(xdg / "mlpstorage" / "config.yaml")


# --------------------------------------------------------------------------- #
# show                                                                         #
# --------------------------------------------------------------------------- #


class TestShow:
    def test_no_file(self, xdg, capsys):
        from mlpstorage_py.results_dir.user_config import USER_CONFIG_KEYS

        rc, out, _ = _run(["show"], capsys)
        assert rc == EXIT_CODE.SUCCESS
        assert _cfg_path() in out
        assert "[not present]" in out
        for key in USER_CONFIG_KEYS:
            assert key in out
        assert out.count("[unset]") == len(USER_CONFIG_KEYS)

    def test_values_and_env_overrides(self, xdg, monkeypatch, capsys):
        _write({"results_dir": "/r", "systemname": "lab-a",
                "hosts": ["n1", "n2"], "oversubscribe": True})
        monkeypatch.setenv("MLPSTORAGE_SYSTEMNAME", "lab-b")
        monkeypatch.setenv("MLPSTORAGE_DATA_DIR", "/d")
        rc, out, _ = _run(["show"], capsys)
        assert rc == EXIT_CODE.SUCCESS
        lines = {ln.split()[0]: ln for ln in out.splitlines() if ln.strip()}
        assert "/r" in lines["results_dir"]
        assert "lab-a" in lines["systemname"]
        assert "outranked by MLPSTORAGE_SYSTEMNAME=lab-b" in lines["systemname"]
        assert "[unset]" in lines["data_dir"]
        assert "MLPSTORAGE_DATA_DIR=/d supplies it" in lines["data_dir"]
        assert "[n1, n2]" in lines["hosts"]
        assert "true" in lines["oversubscribe"]
        assert "[not present]" not in out

    def test_unknown_and_workload_keys_are_diagnosed(self, xdg, capsys):
        _write({"systemname": "s", "model": "unet3d", "colour": "never"})
        rc, out, _ = _run(["show"], capsys)
        assert rc == EXIT_CODE.SUCCESS
        assert "model" in out and "REJECTED" in out
        assert "mlpstorage config unset model" in out
        assert "colour" in out and "ignored" in out

    def test_json(self, xdg, monkeypatch, capsys):
        from mlpstorage_py.results_dir.user_config import USER_CONFIG_KEYS

        _write({"systemname": "s", "hosts": ["n1"], "model": "unet3d", "colour": "x"})
        monkeypatch.setenv("MLPSTORAGE_SYSTEMNAME", "t")
        rc, out, _ = _run(["show", "--json"], capsys)
        assert rc == EXIT_CODE.SUCCESS
        doc = json.loads(out)
        assert doc["path"] == _cfg_path() and doc["exists"] is True
        assert set(doc["keys"]) == set(USER_CONFIG_KEYS)
        assert doc["keys"]["systemname"] == {
            "value": "s", "env": {"name": "MLPSTORAGE_SYSTEMNAME", "value": "t"}}
        assert doc["keys"]["hosts"] == {"value": ["n1"], "env": None}
        assert doc["keys"]["color"] == {"value": None, "env": None}
        assert doc["rejected"] == {"model": "unet3d"}
        assert doc["ignored"] == {"colour": "x"}

    def test_json_no_file(self, xdg, capsys):
        rc, out, _ = _run(["show", "--json"], capsys)
        doc = json.loads(out)
        assert doc["exists"] is False and doc["rejected"] == {} and doc["ignored"] == {}
        assert all(v == {"value": None, "env": None} for v in doc["keys"].values())


# --------------------------------------------------------------------------- #
# set                                                                          #
# --------------------------------------------------------------------------- #


class TestSet:
    def test_scalar_creates_file_and_keeps_header(self, xdg, capsys):
        rc, out, _ = _run(["set", "systemname", "lab-a"], capsys)
        assert rc == EXIT_CODE.SUCCESS
        assert _read() == {"systemname": "lab-a"}
        with open(_cfg_path(), encoding="utf-8") as fh:
            assert fh.readline().startswith("# Per-user mlpstorage defaults")
        assert "systemname" in out and "lab-a" in out and _cfg_path() in out

    def test_keeps_other_keys(self, xdg, capsys):
        _write({"results_dir": "/r", "color": "never"})
        _run(["set", "systemname", "lab-a"], capsys)
        assert _read() == {"results_dir": "/r", "color": "never", "systemname": "lab-a"}

    def test_overwrites_existing(self, xdg, capsys):
        _write({"systemname": "old"})
        _run(["set", "systemname", "new"], capsys)
        assert _read()["systemname"] == "new"

    def test_paths_stored_as_typed(self, xdg, capsys):
        _run(["set", "results_dir", "~/results"], capsys)
        _run(["set", "data_dir", "/mnt/nvme/unet3d"], capsys)
        assert _read() == {"results_dir": "~/results", "data_dir": "/mnt/nvme/unet3d"}

    def test_list_keys(self, xdg, capsys):
        _run(["set", "hosts", "n1", "n2", "n3"], capsys)
        assert _read()["hosts"] == ["n1", "n2", "n3"]
        _run(["set", "hosts", "n1,n2"], capsys)
        assert _read()["hosts"] == ["n1", "n2"]
        _run(["set", "hosts", "n1"], capsys)
        assert _read()["hosts"] == ["n1"]

    def test_append_key(self, xdg, capsys):
        _run(["set", "mpi_params", "-x A=1"], capsys)
        assert _read()["mpi_params"] == "-x A=1"
        _run(["set", "mpi_params", "-x A=1", "-x B=2"], capsys)
        assert _read()["mpi_params"] == ["-x A=1", "-x B=2"]

    @pytest.mark.parametrize("word, expected", [
        ("true", True), ("True", True), ("yes", True), ("1", True), ("on", True),
        ("false", False), ("no", False), ("0", False), ("off", False),
    ])
    def test_booleans(self, xdg, capsys, word, expected):
        _run(["set", "oversubscribe", word], capsys)
        assert _read()["oversubscribe"] is expected

    def test_bad_boolean(self, xdg, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(["set", "allow_run_as_root", "maybe"], capsys)
        assert exc.value.code == EXIT_CODE.INVALID_ARGUMENTS
        assert "allow_run_as_root" in capsys.readouterr().err
        assert not os.path.exists(_cfg_path())

    def test_choice_checked(self, xdg, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(["set", "mpi_bin", "mpi"], capsys)
        assert exc.value.code == EXIT_CODE.INVALID_ARGUMENTS
        err = capsys.readouterr().err
        assert "mpi_bin" in err and "mpirun" in err and "mpiexec" in err
        _run(["set", "mpi_bin", "mpiexec"], capsys)
        assert _read()["mpi_bin"] == "mpiexec"

    def test_scalar_takes_one_value(self, xdg, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(["set", "color", "never", "always"], capsys)
        assert exc.value.code == EXIT_CODE.INVALID_ARGUMENTS
        assert "one value" in capsys.readouterr().err

    def test_workload_key_refused(self, xdg, capsys):
        _write({"systemname": "s"})
        for key in ("model", "accelerator_type", "num_accelerators", "params"):
            with pytest.raises(SystemExit) as exc:
                _run(["set", key, "x"], capsys)
            assert exc.value.code == EXIT_CODE.INVALID_ARGUMENTS
            err = capsys.readouterr().err
            assert key in err and "selects the workload" in err
        assert _read() == {"systemname": "s"}

    def test_unknown_key_refused(self, xdg, capsys):
        with pytest.raises(SystemExit) as exc:
            _run(["set", "colour", "never"], capsys)
        assert exc.value.code == EXIT_CODE.INVALID_ARGUMENTS
        err = capsys.readouterr().err
        assert "colour" in err and "systemname" in err  # names the allowed keys
        assert not os.path.exists(_cfg_path())

    def test_every_allowed_key_is_settable(self, xdg, capsys):
        from mlpstorage_py.results_dir.user_config import USER_CONFIG_KEYS

        sample = {
            "results_dir": "/r", "systemname": "s", "data_dir": "/d",
            "checkpoint_folder": "/c", "hosts": "n1,n2", "mpi_bin": "mpirun",
            "mpi_btl": "tcp", "oversubscribe": "true", "allow_run_as_root": "false",
            "mpi_params": "-x A", "dlio_bin_path": "/opt/dlio", "exec_type": "mpi",
            "color": "never", "stream_log_level": "DEBUG",
        }
        assert set(sample) == set(USER_CONFIG_KEYS)
        for key, value in sample.items():
            rc, _, _ = _run(["set", key, value], capsys)
            assert rc == EXIT_CODE.SUCCESS, key
        assert set(_read()) == set(USER_CONFIG_KEYS)

    def test_set_then_every_command_reads_it(self, xdg, tmp_path, monkeypatch, capsys):
        """What ``set`` writes is what the tier resolver reads back."""
        from mlpstorage_py.cli_parser import parse_arguments
        from mlpstorage_py.cli import common_args

        monkeypatch.setattr(common_args, "ENV_FALLBACK_RESULTS_DIR", "")
        monkeypatch.setattr(common_args, "ENV_FALLBACK_SYSTEMNAME", "")
        _run(["set", "results_dir", str(tmp_path)], capsys)
        _run(["set", "systemname", "lab-a"], capsys)
        _run(["set", "hosts", "n1,n2"], capsys)
        _run(["set", "oversubscribe", "yes"], capsys)
        monkeypatch.setattr("sys.argv", ["mlpstorage", "closed", "training", "unet3d",
                                         "run", "file", "-na", "2", "-at", "b200",
                                         "-cm", "64", "-dd", "/data"])
        args = parse_arguments()
        assert args.systemname == "lab-a"
        assert args.hosts == ["n1", "n2"]
        assert args.oversubscribe is True
        assert args.config_sources["hosts"] == _cfg_path()


# --------------------------------------------------------------------------- #
# unset                                                                        #
# --------------------------------------------------------------------------- #


class TestUnset:
    def test_removes_one_key(self, xdg, capsys):
        _write({"results_dir": "/r", "systemname": "s", "color": "never"})
        rc, out, _ = _run(["unset", "systemname"], capsys)
        assert rc == EXIT_CODE.SUCCESS
        assert _read() == {"results_dir": "/r", "color": "never"}
        assert "systemname" in out

    def test_removes_a_workload_key(self, xdg, capsys):
        _write({"systemname": "s", "model": "unet3d"})
        rc, _, _ = _run(["unset", "model"], capsys)
        assert rc == EXIT_CODE.SUCCESS
        assert _read() == {"systemname": "s"}

    def test_absent_key_is_a_no_op(self, xdg, capsys):
        _write({"systemname": "s"})
        rc, out, _ = _run(["unset", "hosts"], capsys)
        assert rc == EXIT_CODE.SUCCESS
        assert "not set" in out
        assert _read() == {"systemname": "s"}

    def test_no_file_is_a_no_op(self, xdg, capsys):
        rc, out, _ = _run(["unset", "hosts"], capsys)
        assert rc == EXIT_CODE.SUCCESS
        assert not os.path.exists(_cfg_path())

    def test_unset_results_dir_says_how_to_supply_one(self, xdg, capsys):
        from mlpstorage_py.results_dir.user_config import HOW_TO_SUPPLY

        _write({"results_dir": "/r"})
        rc, out, _ = _run(["unset", "results_dir"], capsys)
        assert rc == EXIT_CODE.SUCCESS
        assert HOW_TO_SUPPLY in out
        assert _read() == {}


# --------------------------------------------------------------------------- #
# main dispatch                                                                #
# --------------------------------------------------------------------------- #


class TestMainDispatch:
    def test_config_needs_no_results_dir(self, xdg, monkeypatch, capsys):
        from mlpstorage_py.main import _main_impl, NON_BENCHMARK_NO_ORGNAME_MODES

        assert "config" in NON_BENCHMARK_NO_ORGNAME_MODES
        monkeypatch.setattr("sys.argv", ["mlpstorage", "config", "path"])
        rc = _main_impl()
        out = capsys.readouterr().out
        assert rc == EXIT_CODE.SUCCESS
        assert out.strip().endswith(os.path.join("mlpstorage", "config.yaml"))
        assert "results-dir:" not in out

    def test_config_set_through_main(self, xdg, monkeypatch, capsys):
        from mlpstorage_py.main import _main_impl

        monkeypatch.setattr("sys.argv", ["mlpstorage", "config", "set", "systemname", "lab-a"])
        assert _main_impl() == EXIT_CODE.SUCCESS
        assert _read() == {"systemname": "lab-a"}
