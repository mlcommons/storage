"""Logging behaviour: TTY-aware colour, per-run log files, and the end-of-run
error recap.

Design (2026-09-17 logging PR):

* Console output goes through ONE shared rich ``Console`` on stderr. Colour is
  applied to the level tag only, never to the message body, so long messages
  stay readable and ``grep`` still works on the text.
* ``--color auto|always|never``: ``auto`` colours only when stderr is a TTY
  (rich also honours ``NO_COLOR``); ``never`` emits no ANSI at all.
* Every record from any ``mlpstorage_py.*`` module logger reaches the console
  (previously ``mlpstorage init``'s confirmation line was silently dropped
  because its ``logging.getLogger(__name__)`` had no handler in its chain).
* A DEBUG-level buffer collects every record from process start. When a run
  leaf is reserved the buffer is flushed into ``<leaf>/mlpstorage.log`` (all
  levels) and ``<leaf>/mlpstorage.errors.log`` (WARNING and above), and later
  records stream straight to those files until detach.
* WARNING+ records are collected for an end-of-invocation recap printed to
  the console; nothing is printed when the run was clean.
"""

from __future__ import annotations

import io
import logging
import re
from argparse import Namespace

import pytest

from mlpstorage_py import mlps_logging as ml

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture
def console_out():
    """Reset shared logging state and route the console into a StringIO."""
    ml.reset_for_tests()
    buf = io.StringIO()
    ml.configure_console(color="never", file=buf)
    logger = ml.setup_logging("MLPerfStorage")
    yield logger, buf
    ml.reset_for_tests()


# --------------------------------------------------------------------------- #
# Level tags and colour                                                       #
# --------------------------------------------------------------------------- #


def test_level_tags_carry_glyphs_for_error_and_warning():
    assert ml.format_level_tag(logging.ERROR) == "✖ ERROR"
    assert ml.format_level_tag(logging.CRITICAL) == "✖ CRITICAL"
    assert ml.format_level_tag(logging.WARNING) == "⚠ WARNING"
    assert ml.format_level_tag(logging.INFO) == "INFO"
    assert ml.format_level_tag(ml.STATUS) == "STATUS"


def test_color_never_emits_no_ansi(console_out):
    logger, buf = console_out
    logger.error("boom")
    out = buf.getvalue()
    assert "✖ ERROR: boom" in out
    assert not ANSI_RE.search(out)


def test_color_auto_on_non_tty_emits_no_ansi():
    ml.reset_for_tests()
    buf = io.StringIO()
    ml.configure_console(color="auto", file=buf)
    logger = ml.setup_logging("MLPerfStorage")
    logger.error("boom")
    assert "boom" in buf.getvalue()
    assert not ANSI_RE.search(buf.getvalue())
    ml.reset_for_tests()


def test_color_always_colours_only_the_level_tag():
    ml.reset_for_tests()
    buf = io.StringIO()
    ml.configure_console(color="always", file=buf)
    logger = ml.setup_logging("MLPerfStorage")
    logger.error("boom with a long body")
    out = buf.getvalue()
    assert ANSI_RE.search(out), "expected ANSI colour on the level tag"
    # The message body must come AFTER the last reset sequence, i.e. unstyled.
    tail = out.rsplit("\x1b[0m", 1)[-1]
    assert "boom with a long body" in tail
    ml.reset_for_tests()


def test_color_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        ml.configure_console(color="sometimes", file=io.StringIO())


# --------------------------------------------------------------------------- #
# Module loggers propagate to the console                                     #
# --------------------------------------------------------------------------- #


def test_package_module_logger_reaches_console(console_out):
    _, buf = console_out
    logging.getLogger("mlpstorage_py.results_dir.init").info("initialized ok")
    assert "initialized ok" in buf.getvalue()


def test_package_module_logger_is_not_duplicated(console_out):
    _, buf = console_out
    logging.getLogger("mlpstorage_py.results_dir.init").warning("once only")
    assert buf.getvalue().count("once only") == 1


# --------------------------------------------------------------------------- #
# Per-run log files                                                           #
# --------------------------------------------------------------------------- #


def test_attach_flushes_buffer_and_splits_errors(console_out, tmp_path):
    logger, _ = console_out
    logger.info("before-info")
    logger.warning("before-warn")
    logging.getLogger("mlpstorage_py.some.module").debug("before-debug")

    files = ml.attach_run_log_files(str(tmp_path))
    logger.error("after-error")
    logger.info("after-info")
    files.detach()
    logger.info("late-info")

    full = (tmp_path / "mlpstorage.log").read_text()
    errors = (tmp_path / "mlpstorage.errors.log").read_text()

    for line in ("before-info", "before-warn", "before-debug", "after-error", "after-info"):
        assert line in full, line
    assert "late-info" not in full

    assert "before-warn" in errors
    assert "after-error" in errors
    assert "before-info" not in errors
    assert "after-info" not in errors
    assert "late-info" not in errors


def test_attach_works_as_context_manager(console_out, tmp_path):
    logger, _ = console_out
    with ml.attach_run_log_files(str(tmp_path)) as files:
        logger.status("inside")
        assert files.log_path == str(tmp_path / "mlpstorage.log")
        assert files.errors_path == str(tmp_path / "mlpstorage.errors.log")
    logger.status("outside")
    full = (tmp_path / "mlpstorage.log").read_text()
    assert "inside" in full
    assert "outside" not in full


def test_run_log_files_are_plain_text_without_ansi(tmp_path):
    ml.reset_for_tests()
    buf = io.StringIO()
    ml.configure_console(color="always", file=buf)
    logger = ml.setup_logging("MLPerfStorage")
    with ml.attach_run_log_files(str(tmp_path)):
        logger.error("colour on console only")
    assert ANSI_RE.search(buf.getvalue())
    assert not ANSI_RE.search((tmp_path / "mlpstorage.log").read_text())
    assert not ANSI_RE.search((tmp_path / "mlpstorage.errors.log").read_text())
    ml.reset_for_tests()


def test_second_run_in_same_process_does_not_replay_first_runs_records(console_out, tmp_path):
    logger, _ = console_out
    leaf1 = tmp_path / "leaf1"
    leaf2 = tmp_path / "leaf2"
    leaf1.mkdir()
    leaf2.mkdir()
    logger.info("startup")
    with ml.attach_run_log_files(str(leaf1)):
        logger.info("run-one")
    logger.info("between")
    with ml.attach_run_log_files(str(leaf2)):
        logger.info("run-two")
    one = (leaf1 / "mlpstorage.log").read_text()
    two = (leaf2 / "mlpstorage.log").read_text()
    assert "startup" in one and "run-one" in one
    assert "run-one" not in two
    assert "between" in two and "run-two" in two


# --------------------------------------------------------------------------- #
# End-of-invocation recap                                                     #
# --------------------------------------------------------------------------- #


def test_recap_lists_warnings_and_errors(console_out):
    logger, buf = console_out
    logger.warning("disk nearly full")
    logger.error("[E101] something broke")
    logger.info("noise")
    buf.truncate(0)
    buf.seek(0)

    n = ml.emit_recap()

    out = buf.getvalue()
    assert n == 2
    assert "1 error" in out and "1 warning" in out
    assert "disk nearly full" in out
    assert "[E101] something broke" in out
    assert "noise" not in out


def test_recap_is_silent_when_clean(console_out):
    logger, buf = console_out
    logger.info("all fine")
    buf.truncate(0)
    buf.seek(0)
    assert ml.emit_recap() == 0
    assert buf.getvalue() == ""


def test_recap_names_the_errors_log_when_a_run_leaf_was_attached(console_out, tmp_path):
    logger, buf = console_out
    with ml.attach_run_log_files(str(tmp_path)):
        logger.error("boom")
    buf.truncate(0)
    buf.seek(0)
    ml.emit_recap()
    assert str(tmp_path / "mlpstorage.errors.log") in buf.getvalue()


# --------------------------------------------------------------------------- #
# apply_logging_options                                                       #
# --------------------------------------------------------------------------- #


def test_apply_logging_options_color_never_disables_ansi():
    ml.reset_for_tests()
    buf = io.StringIO()
    ml.configure_console(color="always", file=buf)
    logger = ml.setup_logging("MLPerfStorage")
    ml.apply_logging_options(logger, Namespace(color="never"))
    logger.error("plain")
    assert "plain" in ml.get_console().file.getvalue()
    assert not ANSI_RE.search(ml.get_console().file.getvalue())
    ml.reset_for_tests()


def test_stream_level_does_not_starve_the_run_log_files(console_out, tmp_path):
    logger, buf = console_out
    ml.apply_logging_options(logger, Namespace(stream_log_level="ERROR"))
    with ml.attach_run_log_files(str(tmp_path)):
        logger.info("quiet on console, loud in file")
    assert "quiet on console" not in buf.getvalue()
    assert "quiet on console" in (tmp_path / "mlpstorage.log").read_text()


def test_debug_option_adds_module_and_line_to_console(console_out):
    logger, buf = console_out
    ml.apply_logging_options(logger, Namespace(debug=True))
    logger.debug("where am i")
    assert re.search(r"test_mlps_logging:\d+", buf.getvalue())


# --------------------------------------------------------------------------- #
# CLI surface                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", ["auto", "always", "never"])
def test_cli_color_flag_parses(value):
    from mlpstorage_py.cli_parser import build_parser

    args = build_parser().parse_args(
        ["closed", "training", "unet3d", "run", "file", "--color", value]
    )
    assert args.color == value


def test_cli_color_flag_defaults_to_auto():
    from mlpstorage_py.cli_parser import build_parser

    args = build_parser().parse_args(["closed", "training", "unet3d", "run", "file"])
    assert args.color == "auto"


def test_cli_color_flag_rejects_unknown_value():
    from mlpstorage_py.cli_parser import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["closed", "training", "unet3d", "run", "file", "--color", "rainbow"]
        )
