"""
Unit tests for the orgname-resolution gate in ``mlpstorage_py.main._main_impl``
— Slice 4 of Phase 1 (canonical-layout-and-init).

LAY-03 enforcement: every command that takes ``--results-dir`` reads orgname
from ``<results-dir>/mlperf-results.yaml`` at startup and fails BEFORE any
benchmark plumbing if the sentinel is missing — with the EXACT actionable
message from CONTEXT.md (backticks around the path, NOT single quotes).

Covers:

- ``test_uninitialized_results_dir_fails`` — gate raises before update_args /
  validate_benchmark_environment / print_run_summary / run_benchmark.
- ``test_failure_message_text`` — verbatim CONTEXT.md / ROADMAP success-criterion #2
  string (backticks; no single quotes; no backslash escapes).
- ``test_gated_commands_fail_uninitialized`` — parameterised over every
  ``--results-dir``-bearing mode (closed, open, whatif, reports, history).
  ``validate`` does NOT take ``--results-dir`` and is therefore NOT
  exercised here (see deviation note in SUMMARY).
- ``test_env_orgname_ignored`` — MLPERF_ORGNAME env var is NEVER consulted.
- ``test_bypass_commands_skip_gate`` — parameterised over the EXACT four
  bypass modes (init, version, lockfile, rules-coverage).
- ``test_initialized_dir_resolves_orgname_to_args`` — happy path: gate
  populates ``args.orgname`` from the sentinel.
- ``test_no_orgname_flag_on_non_init_commands`` — regression of the Slice 2
  check, repeated here for the gate-shape contract.
- ``test_benchmark_init_raises_when_orgname_missing`` — Pitfall 3
  defense-in-depth assertion at ``Benchmark.__init__``.

Refs: 01-canonical-layout-and-init / 01-04-PLAN.md; 01-CONTEXT.md LAY-03 / D-12;
01-RESEARCH.md Pitfalls 1, 3, 6, 8.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from argparse import Namespace
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------- #
# Helpers                                                                      #
# ---------------------------------------------------------------------------- #


def _datagen_argv(results_dir: str) -> list:
    """Construct a minimal valid ``closed training unet3d datagen`` argv.

    Datagen is a deliberately small surface area: model positional,
    data_access_protocol positional, ``--data-dir``, ``--results-dir``,
    ``--systemname``, ``--num-processes``. No ``-cm``/``-na``/``-at``/
    ``--num-client-hosts`` needed for datagen.
    """
    return [
        "mlpstorage", "closed", "training", "unet3d", "datagen", "file",
        "--data-dir", "/d",
        "--results-dir", results_dir,
        "--systemname", "sys-v1",
        "--num-processes", "1",
    ]


def _init_results_dir(tmp_path, orgname: str = "Acme"):
    """Initialise a results-dir with the sentinel by calling run_init directly."""
    from argparse import Namespace as _NS
    from mlpstorage_py.results_dir.init import run_init

    d = tmp_path / "r1"
    d.mkdir()
    run_init(_NS(orgname=orgname, path=str(d)))
    return str(d)


# ---------------------------------------------------------------------------- #
# Tests                                                                        #
# ---------------------------------------------------------------------------- #


def test_uninitialized_results_dir_fails(tmp_path):
    """``_main_impl`` against an uninitialised --results-dir must raise
    ConfigurationError BEFORE update_args / run_benchmark / etc. fire.
    """
    from mlpstorage_py import main as main_mod
    from mlpstorage_py.errors import ConfigurationError

    uninit = tmp_path / "uninit"
    uninit.mkdir()

    with patch.object(main_mod, "update_args") as mock_update_args, \
         patch.object(main_mod, "run_benchmark") as mock_run_benchmark, \
         patch.object(main_mod, "validate_benchmark_environment") as mock_validate_env, \
         patch("mlpstorage_py.run_summary.print_run_summary") as mock_print_summary, \
         patch("sys.argv", _datagen_argv(str(uninit))):
        with pytest.raises(ConfigurationError):
            main_mod._main_impl()

    mock_update_args.assert_not_called()
    mock_run_benchmark.assert_not_called()
    mock_validate_env.assert_not_called()
    mock_print_summary.assert_not_called()


def test_failure_message_text(tmp_path):
    """Error message AND suggestion match CONTEXT.md LAY-03 / ROADMAP success
    criterion #2 verbatim — backticks around the path, NOT single quotes,
    NOT backslash-escaped.
    """
    from mlpstorage_py import main as main_mod
    from mlpstorage_py.errors import ConfigurationError

    uninit = tmp_path / "uninit"
    uninit.mkdir()
    uninit_str = str(uninit)

    with patch("sys.argv", _datagen_argv(uninit_str)):
        with pytest.raises(ConfigurationError) as excinfo:
            main_mod._main_impl()

    err = excinfo.value
    msg = str(err)
    suggestion = err.suggestion or ""

    # Verbatim string assertion (locked spec).
    assert f"results-dir `{uninit_str}` has not been initialized." in msg, (
        f"Expected backtick-delimited path in message; got: {msg!r}"
    )
    assert f"Run `mlpstorage init <orgname> {uninit_str}` first." in suggestion, (
        f"Expected backtick-delimited suggestion; got: {suggestion!r}"
    )

    # Regex assertion (covers message shape).
    assert re.search(r"results-dir `[^`]+` has not been initialized\.", msg), (
        f"Message did not match LAY-03 regex; got: {msg!r}"
    )
    assert re.search(r"Run `mlpstorage init <orgname> [^`]+` first\.", suggestion), (
        f"Suggestion did not match LAY-03 regex; got: {suggestion!r}"
    )

    # Negative: ensure single-quotes form (the !r leak) is NOT used.
    assert f"results-dir '{uninit_str}'" not in msg, (
        "Message uses single quotes; the locked spec is backticks (D-09 / LAY-03)."
    )


@pytest.mark.parametrize(
    "mode,extra_argv",
    [
        # closed/open/whatif: datagen path (minimal universal-arg call).
        ("closed", ["closed", "training", "unet3d", "datagen", "file",
                    "--data-dir", "/d", "--systemname", "sys-v1",
                    "--num-processes", "1"]),
        ("open", ["open", "training", "unet3d", "datagen", "file",
                  "--data-dir", "/d", "--systemname", "sys-v1",
                  "--num-processes", "1"]),
        ("whatif", ["whatif", "training", "unet3d", "datagen", "file",
                    "--data-dir", "/d", "--systemname", "sys-v1",
                    "--num-processes", "1"]),
        # reports reportgen takes --results-dir + --systemname.
        ("reports", ["reports", "reportgen", "--systemname", "sys-v1"]),
        # history show takes --results-dir + --systemname.
        ("history", ["history", "show", "--systemname", "sys-v1"]),
    ],
    ids=["closed", "open", "whatif", "reports", "history"],
)
def test_gated_commands_fail_uninitialized(tmp_path, mode, extra_argv):
    """Every mode that takes ``--results-dir`` (per D-12 gated scope) must
    fail-fast with the LAY-03 actionable error against an uninitialised dir.

    Note: ``validate`` is NOT in this parameter list because the existing
    ``add_validate_arguments`` builder does not register ``--results-dir``
    (it takes a positional ``input`` instead). Documented as Slice-4
    deviation in 01-04-SUMMARY.md.
    """
    from mlpstorage_py import main as main_mod
    from mlpstorage_py.errors import ConfigurationError

    uninit = tmp_path / "uninit"
    uninit.mkdir()

    argv = ["mlpstorage"] + extra_argv + ["--results-dir", str(uninit)]

    # Patch downstream dispatch entry points so an accidental fall-through
    # doesn't run the real ReportGenerator / HistoryTracker. report_generator
    # is lazy-loaded inside the `args.mode == "reports"` branch and depends on
    # the optional [full]-extras `psutil` package; only patch its symbol when
    # the module is actually importable in this venv.
    try:
        import mlpstorage_py.report_generator  # noqa: F401
        with patch.object(main_mod, "update_args"), \
             patch.object(main_mod, "run_benchmark"), \
             patch.object(main_mod, "validate_benchmark_environment"), \
             patch("mlpstorage_py.report_generator.ReportGenerator") as mock_reportgen, \
             patch("mlpstorage_py.history.HistoryTracker.handle_history_command") as mock_history, \
             patch("sys.argv", argv):
            with pytest.raises(ConfigurationError) as excinfo:
                main_mod._main_impl()
            assert "has not been initialized" in str(excinfo.value), (
                f"Expected LAY-03 message for mode={mode}; got: {excinfo.value!r}"
            )
            mock_reportgen.assert_not_called()
            mock_history.assert_not_called()
    except ModuleNotFoundError:
        # psutil-free path: skip the report_generator patch, run without it.
        with patch.object(main_mod, "update_args"), \
             patch.object(main_mod, "run_benchmark"), \
             patch.object(main_mod, "validate_benchmark_environment"), \
             patch("mlpstorage_py.history.HistoryTracker.handle_history_command") as mock_history, \
             patch("sys.argv", argv):
            with pytest.raises(ConfigurationError) as excinfo:
                main_mod._main_impl()
            assert "has not been initialized" in str(excinfo.value), (
                f"Expected LAY-03 message for mode={mode}; got: {excinfo.value!r}"
            )
            mock_history.assert_not_called()


def test_env_orgname_ignored(tmp_path, monkeypatch):
    """MLPERF_ORGNAME env var must NOT be consulted — gate still fails on
    an uninitialised dir even when the var is set.
    """
    from mlpstorage_py import main as main_mod
    from mlpstorage_py.errors import ConfigurationError

    monkeypatch.setenv("MLPERF_ORGNAME", "Acme")

    uninit = tmp_path / "uninit"
    uninit.mkdir()

    with patch("sys.argv", _datagen_argv(str(uninit))):
        with pytest.raises(ConfigurationError) as excinfo:
            main_mod._main_impl()

    assert "has not been initialized" in str(excinfo.value)


@pytest.mark.parametrize(
    "mode,argv",
    [
        ("version", ["mlpstorage", "version"]),
        ("lockfile", ["mlpstorage", "lockfile", "generate", "--output", "/tmp/x.lock"]),
        ("rules-coverage", ["mlpstorage", "rules-coverage"]),
        # init is covered by its own dispatch test in test_init.py but we
        # repeat here for the gate-shape contract.
        ("init", ["mlpstorage", "init", "Acme", "/tmp/init_placeholder_unused"]),
    ],
    ids=["version", "lockfile", "rules-coverage", "init"],
)
def test_bypass_commands_skip_gate(tmp_path, mode, argv):
    """Exactly four bypass modes complete WITHOUT raising
    ResultsDirNotInitializedError even when no sentinel exists.
    """
    from mlpstorage_py import main as main_mod
    from mlpstorage_py.results_dir.errors import ResultsDirNotInitializedError

    # Patch heavy / side-effectful entry points to keep the test hermetic.
    # We assert that the gate (which would raise ConfigurationError on
    # uninitialised dirs) is NOT invoked — but allow the bypass mode's
    # own logic to run / be short-circuited.
    with patch("mlpstorage_py.results_dir.run_init",
               return_value=__import__("mlpstorage_py.config",
                                       fromlist=["EXIT_CODE"]).EXIT_CODE.SUCCESS), \
         patch("mlpstorage_py.results_dir.init.run_init",
               return_value=__import__("mlpstorage_py.config",
                                       fromlist=["EXIT_CODE"]).EXIT_CODE.SUCCESS), \
         patch.object(main_mod, "handle_lockfile_command",
                      return_value=__import__("mlpstorage_py.config",
                                              fromlist=["EXIT_CODE"]).EXIT_CODE.SUCCESS), \
         patch("mlpstorage_py.submission_checker.tools.rules_coverage.run",
               return_value=__import__("mlpstorage_py.config",
                                       fromlist=["EXIT_CODE"]).EXIT_CODE.SUCCESS), \
         patch("sys.argv", argv):
        try:
            main_mod._main_impl()
        except ResultsDirNotInitializedError as e:
            pytest.fail(
                f"Bypass mode {mode!r} unexpectedly tripped sentinel gate: {e}"
            )
        except SystemExit:
            # version mode raises SystemExit(0); that's success for bypass intent.
            pass


def test_initialized_dir_resolves_orgname_to_args(tmp_path):
    """After running ``mlpstorage init Acme <tmp>``, invoking _main_impl with
    --results-dir <tmp> populates args.orgname=="Acme" and lets execution
    proceed past the gate.
    """
    from mlpstorage_py import main as main_mod
    from mlpstorage_py.config import EXIT_CODE

    initialized = _init_results_dir(tmp_path, orgname="Acme")

    captured = {}

    def fake_run_benchmark(args, run_datetime):
        captured["orgname"] = getattr(args, "orgname", None)
        captured["systemname"] = getattr(args, "systemname", None)
        return EXIT_CODE.SUCCESS

    with patch.object(main_mod, "run_benchmark", side_effect=fake_run_benchmark), \
         patch.object(main_mod, "update_args"), \
         patch("mlpstorage_py.run_summary.print_run_summary"), \
         patch("sys.argv", _datagen_argv(initialized)):
        rc = main_mod._main_impl()

    assert rc == EXIT_CODE.SUCCESS
    assert captured.get("orgname") == "Acme", (
        f"Expected args.orgname='Acme' after gate; got: {captured!r}"
    )
    assert captured.get("systemname") == "sys-v1"


def test_no_orgname_flag_on_non_init_commands():
    """Regression of Slice 2's check: no top-level --orgname flag leaks
    into any non-init subcommand. Repeated here for the gate-shape contract.
    """
    # Argparse-layer check.
    from mlpstorage_py.cli_parser import parse_arguments

    for bad_argv in (
        ["mlpstorage", "closed", "training", "unet3d", "datagen", "file",
         "--data-dir", "/d", "--results-dir", "/tmp", "--systemname", "sys-v1",
         "--num-processes", "1", "--orgname", "Acme"],
        ["mlpstorage", "reports", "reportgen", "-rd", "/tmp", "-sn", "sys-v1",
         "--orgname", "Acme"],
        ["mlpstorage", "history", "show", "-rd", "/tmp", "-sn", "sys-v1",
         "--orgname", "Acme"],
    ):
        with patch("sys.argv", bad_argv):
            with pytest.raises(SystemExit):
                parse_arguments()

    # Grep-style check: no `--orgname` literal in non-init CLI builder source.
    import mlpstorage_py.cli as cli_pkg
    cli_dir = os.path.dirname(cli_pkg.__file__)
    offenders = []
    for root, _dirs, files in os.walk(cli_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            if name == "init_args.py":
                # init_args.py registers the positional `orgname` — that is
                # the only sanctioned mention.
                continue
            path = os.path.join(root, name)
            with open(path) as f:
                for lineno, line in enumerate(f, start=1):
                    stripped = line.lstrip()
                    if stripped.startswith("#"):
                        continue
                    if "--orgname" in line:
                        offenders.append(f"{path}:{lineno}: {line.rstrip()}")
    assert offenders == [], (
        f"Found --orgname references in non-init CLI builders: {offenders}"
    )

    # Also check cli_parser.py and main.py.
    for mod_path in ("mlpstorage_py/cli_parser.py", "mlpstorage_py/main.py"):
        with open(mod_path) as f:
            for lineno, line in enumerate(f, start=1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                assert "--orgname" not in line, (
                    f"--orgname found in {mod_path}:{lineno}: {line.rstrip()}"
                )

    # MLPERF_ORGNAME env var must NEVER be referenced in main.py.
    with open("mlpstorage_py/main.py") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            assert "MLPERF_ORGNAME" not in line, (
                f"MLPERF_ORGNAME found in main.py:{lineno}: {line.rstrip()}"
            )


def test_benchmark_init_raises_when_orgname_missing():
    """Defense-in-depth (Pitfall 3): ``Benchmark.__init__`` raises
    ConfigurationError when args.orgname is empty or missing. Production
    callers never trip this — the gate populates args.orgname upstream —
    but the guard catches direct (test-only) instantiations.
    """
    from mlpstorage_py.errors import ConfigurationError

    # Build a TrainingBenchmark with deliberately-empty orgname.
    try:
        from mlpstorage_py.benchmarks import TrainingBenchmark
    except ImportError:
        pytest.skip("TrainingBenchmark import requires optional deps not installed.")

    args = Namespace(
        orgname="",   # the violation
        systemname="sys-v1",
        mode="closed",
        debug=False,
        verbose=False,
        stream_log_level="INFO",
        results_dir="/tmp/r",
        benchmark="training",
        model="unet3d",
        command="run",
        num_processes=1,
        num_accelerators=1,
        accelerator_type="h100",
        client_host_memory_in_gb=64,
        data_dir="/d",
        mpi_bin="mpirun",
        exec_type="mpi",
        loops=1,
        params=None,
        hosts=None,
        allow_invalid_params=False,
        loop_delay=0,
    )

    with pytest.raises(ConfigurationError) as excinfo:
        TrainingBenchmark(args, run_datetime="20260619_000000", logger=MagicMock())

    assert "orgname" in str(excinfo.value).lower(), (
        f"Expected orgname-related error; got: {excinfo.value!r}"
    )


# ---------------------------------------------------------------------------- #
# WR-06 — defense-in-depth orgname re-validation at the gate                   #
# ---------------------------------------------------------------------------- #


def test_gate_rejects_orgname_with_unsafe_characters(tmp_path):
    """WR-06: the gate must re-validate ``args.orgname`` against the canonical
    character class even after ``resolve_orgname`` returns.

    Pydantic v2's full-match regex makes this defensive, but the value
    immediately lands in ``os.path.join(..., orgname, "results", ...)`` —
    so a Pydantic regression (version bump to non-anchored semantics,
    switch to v1 ``regex=`` keyword, etc.) would create a directory-
    traversal vector. Patch ``resolve_orgname`` to return an unsafe value
    and assert the gate refuses it with ``ConfigurationError``.
    """
    from mlpstorage_py import main as main_mod
    from mlpstorage_py.errors import ConfigurationError

    init_dir = _init_results_dir(tmp_path)
    argv = _datagen_argv(init_dir)

    # Simulate a Pydantic-validation bypass / regression by short-circuiting
    # resolve_orgname directly. The full-stack Pydantic path is exercised
    # elsewhere; this test pins the gate's belt-and-suspenders check.
    with patch("mlpstorage_py.main.resolve_orgname", return_value="evil/../traversal"), \
         patch.object(main_mod, "update_args"), \
         patch.object(main_mod, "run_benchmark"), \
         patch("sys.argv", argv):
        with pytest.raises(ConfigurationError) as excinfo:
            main_mod._main_impl()

    msg = str(excinfo.value)
    assert "orgname" in msg.lower()
    assert "invalid" in msg.lower() or "character" in msg.lower(), (
        f"Expected message to flag invalid characters; got: {msg!r}"
    )


# ---------------------------------------------------------------------------- #
# WR-03 — history-rerun must re-dispatch when the replayed mode lands in       #
# a NON_BENCHMARK_NO_ORGNAME bypass mode (version / lockfile / rules-coverage).#
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("replayed_mode", ["version", "lockfile", "rules-coverage"])
def test_history_rerun_redispatches_bypass_mode(replayed_mode, tmp_path):
    """WR-03: a history rerun whose replayed entry is a bypass mode must
    re-route to that mode's handler — not fall through to the benchmark loop.

    Pre-fix, after ``args = new_args`` the only mode the code re-checked
    was ``reports``. A replayed ``version``/``lockfile``/``rules-coverage``
    entry fell straight into ``update_args`` + ``run_benchmark``, which
    either crashed in obvious ways (no ``benchmark`` attribute) or — worse
    — silently re-entered benchmark plumbing with a non-benchmark mode.
    The fix re-dispatches after ``args = new_args`` so the replayed
    bypass mode is honored.
    """
    from mlpstorage_py import main as main_mod
    from mlpstorage_py.config import EXIT_CODE

    # Initial argv: history rerun with --results-dir (orgname gate fires
    # only on the original args, which is fine — history is not bypassed).
    init_dir = _init_results_dir(tmp_path)
    argv = [
        "mlpstorage", "history", "rerun", "1",
        "--results-dir", init_dir,
        "--systemname", "sys-v1",
    ]

    # Build a replayed Namespace that lands in the requested bypass mode.
    # Keep the attributes minimal — only ``mode`` and the bare-minimum
    # things that ``_main_impl`` looks at on the post-history path.
    from argparse import Namespace as _NS
    replayed = _NS(
        mode=replayed_mode,
        debug=False,
        verbose=False,
        stream_log_level="INFO",
        quiet=False,
        # Most bypass modes don't carry --results-dir, but the orgname gate
        # has already run on the ORIGINAL args before we reach history; the
        # replayed mode being in NON_BENCHMARK_NO_ORGNAME_MODES is what
        # signals "do not gate again".
        results_dir=init_dir,
    )

    with patch.object(main_mod, "update_args") as mock_update_args, \
         patch.object(main_mod, "run_benchmark") as mock_run_benchmark, \
         patch("mlpstorage_py.run_summary.print_run_summary") as mock_print_summary, \
         patch("mlpstorage_py.history.HistoryTracker.handle_history_command",
               return_value=replayed), \
         patch("sys.argv", argv):
        # We don't care if the actual bypass handler runs end-to-end here —
        # we only care that the benchmark loop is NOT invoked on a replayed
        # non-benchmark mode. Patch the lockfile/version/rules-coverage
        # handlers so they no-op if reached.
        with patch.object(main_mod, "handle_lockfile_command",
                          return_value=EXIT_CODE.SUCCESS), \
             patch("mlpstorage_py.submission_checker.tools.rules_coverage.run",
                   return_value=EXIT_CODE.SUCCESS, create=True):
            try:
                main_mod._main_impl()
            except SystemExit:
                # ``version`` calls ``sys.exit(0)`` — accept that as success.
                pass
            except Exception:
                # The exact handler may not be wired in the test fixture;
                # the WR-03 contract is purely about NOT touching
                # update_args/run_benchmark/print_run_summary on a replayed
                # bypass mode.
                pass

    mock_update_args.assert_not_called()
    mock_run_benchmark.assert_not_called()
    mock_print_summary.assert_not_called()
