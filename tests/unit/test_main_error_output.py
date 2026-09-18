"""``main()`` error output: one Suggestion line, an end-of-run recap, and no
stray ``print()`` calls on stdout.
"""

from __future__ import annotations

import io
from argparse import Namespace
from unittest.mock import patch

import pytest

from mlpstorage_py import mlps_logging as ml


@pytest.fixture
def console_out(tmp_path, monkeypatch):
    from mlpstorage_py import main as main_mod

    ml.reset_for_tests()
    buf = io.StringIO()
    ml.configure_console(color="never", file=buf)
    ml.setup_logging("MLPerfStorage")
    yield buf
    ml.reset_for_tests()


def _datagen_argv(results_dir: str):
    return [
        "mlpstorage", "closed", "training", "unet3d", "datagen", "file",
        "--data-dir", "/d",
        "--results-dir", results_dir,
        "--systemname", "sys-v1",
        "--num-processes", "1",
    ]


def test_suggestion_is_printed_exactly_once(console_out, tmp_path):
    """The exception body already carries ``Suggestion:``; ``main()`` must not
    log it a second time as a separate INFO line."""
    from mlpstorage_py import main as main_mod

    uninit = tmp_path / "uninit"
    uninit.mkdir()
    with patch("sys.argv", _datagen_argv(str(uninit))):
        rc = main_mod.main()

    assert rc != 0
    out = console_out.getvalue()
    assert out.count("Suggestion:") == 1, out


def test_recap_block_follows_a_failed_invocation(console_out, tmp_path):
    from mlpstorage_py import main as main_mod

    uninit = tmp_path / "uninit"
    uninit.mkdir()
    with patch("sys.argv", _datagen_argv(str(uninit))):
        main_mod.main()

    out = console_out.getvalue()
    assert "1 error" in out
    assert "has not been initialized" in out


def test_update_args_does_not_print_to_stdout(capsys):
    from mlpstorage_py.cli_parser import update_args

    args = Namespace(num_accelerators=4, benchmark="training", command="run")
    update_args(args)

    assert args.num_processes == 4
    assert "Setting attr" not in capsys.readouterr().out
