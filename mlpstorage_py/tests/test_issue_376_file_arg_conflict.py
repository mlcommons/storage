#!/usr/bin/env python3
"""
Regression tests for issue #376:

    Unexpected error: argument --file: conflicting option string: --file

Root cause: ``add_universal_arguments`` and ``add_storage_type_arguments``
both declared ``--file`` / ``--object``. Every benchmark subparser
(training, checkpointing, vectordb, kvcache) calls both, so the duplicate
declaration crashed argparse at parser-construction time — which fired on
the very first ``mlpstorage --help`` invocation, before any user args
were even parsed.

After the fix:

* ``--file`` / ``--object`` are declared **only** in
  ``add_storage_type_arguments``.
* All four benchmark subparser builders can be constructed without
  raising ``argparse.ArgumentError``.
* Utility subparsers (reportgen, history, lockfile) that only call
  ``add_universal_arguments`` no longer expose ``--file``/``--object``,
  which is the intended behavior.

These tests pin those invariants so the regression cannot reappear.
"""

import argparse

import pytest


# ---------------------------------------------------------------------------
# Direct reproduction of the original failure mode
# ---------------------------------------------------------------------------

def test_universal_then_storage_type_does_not_conflict():
    """Calling both adders on one parser must not raise ArgumentError."""
    from mlpstorage_py.cli.common_args import (
        add_universal_arguments,
        add_storage_type_arguments,
    )

    parser = argparse.ArgumentParser()
    add_universal_arguments(parser)
    # Pre-fix this line raised:
    #   argparse.ArgumentError: argument --file: conflicting option string: --file
    add_storage_type_arguments(parser)

    ns = parser.parse_args(["--file"])
    assert ns.file is True
    assert ns.object is None


def test_file_declared_in_exactly_one_adder():
    """``--file`` must live in add_storage_type_arguments only."""
    from mlpstorage_py.cli.common_args import (
        add_universal_arguments,
        add_storage_type_arguments,
    )

    universal_parser = argparse.ArgumentParser()
    add_universal_arguments(universal_parser)
    universal_opts = {
        opt for action in universal_parser._actions for opt in action.option_strings
    }
    assert "--file" not in universal_opts, (
        "--file leaked back into add_universal_arguments; this re-introduces "
        "issue #376 because every benchmark subparser then registers --file twice."
    )
    assert "--object" not in universal_opts

    storage_parser = argparse.ArgumentParser()
    add_storage_type_arguments(storage_parser)
    storage_opts = {
        opt for action in storage_parser._actions for opt in action.option_strings
    }
    assert "--file" in storage_opts
    assert "--object" in storage_opts


# ---------------------------------------------------------------------------
# End-to-end: every real benchmark subparser must build cleanly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module_path, builder_attr",
    [
        ("mlpstorage_py.cli.training_args",      "add_training_arguments"),
        ("mlpstorage_py.cli.checkpointing_args", "add_checkpointing_arguments"),
        ("mlpstorage_py.cli.vectordb_args",      "add_vectordb_arguments"),
        ("mlpstorage_py.cli.kvcache_args",       "add_kvcache_arguments"),
    ],
)
def test_benchmark_subparser_builds_without_argparse_conflict(module_path, builder_attr):
    """Each benchmark subparser builder must not raise during construction.

    Pre-fix, all four crashed during the builder call because both
    ``add_universal_arguments`` and ``add_storage_type_arguments`` registered
    ``--file`` on the same subparser.
    """
    import importlib
    module = importlib.import_module(module_path)
    builder = getattr(module, builder_attr, None)
    if builder is None:
        pytest.skip(f"{module_path} has no builder named {builder_attr}")

    # The real CLI wires these builders by passing a subcommand parser
    # (the return value of ``sub_programs.add_parser(...)``), not the
    # ``_SubParsersAction`` itself — mirror that here.
    root = argparse.ArgumentParser()
    sub_programs = root.add_subparsers(dest="program")
    cmd_parser = sub_programs.add_parser(builder_attr.replace("add_", "").replace("_arguments", ""))

    # If this raises argparse.ArgumentError, the regression is back.
    builder(cmd_parser)


# ---------------------------------------------------------------------------
# --debug must work — it was reported as broken in #376, but only because
# the argparse crash fired before --debug could be evaluated.
# ---------------------------------------------------------------------------

def test_debug_flag_parses_after_fix():
    """``--debug`` is in add_universal_arguments; it must parse normally."""
    from mlpstorage_py.cli.common_args import (
        add_universal_arguments,
        add_storage_type_arguments,
    )

    parser = argparse.ArgumentParser()
    add_universal_arguments(parser)
    add_storage_type_arguments(parser)

    ns = parser.parse_args(["--file", "--debug"])
    assert ns.debug is True
