"""
CLI argument parsing for MLPerf Storage benchmarks.

This module provides the main argument parsing entry point,
using modular argument builders from the cli package.
"""

import argparse
import logging
import os
import re
import shlex
import sys

from mlpstorage_py import VERSION
from mlpstorage_py.config import (
    LLM_MODELS,
    VECTORDB_DEFAULT_RUNTIME,
    EXIT_CODE,
    MLPSTORAGE_RESULTS_DIR_ENVVAR,
    MLPSTORAGE_SYSTEMNAME_ENVVAR,
    MLPSTORAGE_CHECKPOINT_FOLDER_ENVVAR,
    _LEGACY_ENVVAR_MAP,
)

# Import modular argument builders from cli package
from mlpstorage_py.cli.config_layers import apply_config_layers, apply_override_file
from mlpstorage_py.results_dir.user_config import HOW_TO_SUPPLY
from mlpstorage_py.cli import (
    HELP_MESSAGES,
    PROGRAM_DESCRIPTIONS,
    MLPStorageHelpFormatter,
    add_universal_arguments,
    add_training_arguments,      validate_training_arguments,
    add_checkpointing_arguments, validate_checkpointing_arguments,
    add_vectordb_arguments,      validate_vectordb_arguments,
    add_kvcache_arguments,       validate_kvcache_arguments,
    add_reports_arguments,
    add_history_arguments,
    add_lockfile_arguments,
    add_init_arguments,
    add_runs_arguments,
    add_status_arguments,
    add_submit_arguments,
    add_config_arguments,
    add_version_arguments,
    add_validate_arguments,
    add_rules_coverage_arguments,
)

# Backwards compatibility aliases
help_messages = HELP_MESSAGES
prog_descriptions = PROGRAM_DESCRIPTIONS


# -----------------------------------------------------------------------------
# _UNIVERSAL_REQUIRED_SPECS — declarative table for the parse-time
# required-universal gate (Plan 05-02 D-09). Each row describes one universal
# CLI flag that a subcommand may opt into as "required-with-env-fallback":
#
#   (marker_attr_name, arg_attr_name, envvar_name_constant,
#    long_flag, short_flag, name_for_template)
#
# - marker_attr_name: the ``args._mlps_req_*`` attribute set by
#   ``add_universal_arguments(..., req_<x>=True)`` on the opt-in subcommand.
# - arg_attr_name: the resolved argparse dest whose value we check for
#   truthiness (env-var-sourced defaults populate this if the CLI flag was
#   omitted).
# - envvar_name_constant: the ``MLPSTORAGE_*`` env-var-name string constant
#   (single source of truth in ``mlpstorage_py.config``); also the lookup
#   key into ``_LEGACY_ENVVAR_MAP`` for the D-05 migration hint.
# - long_flag / short_flag / name_for_template: substituted verbatim into
#   the D-02 error template and D-05 hint template.
#
# Declaration order pins emission order (results, systemname,
# checkpoint-folder). Training's ``--data-dir`` is intentionally absent —
# per D-07 it gates in ``training_args.py`` after YAML merge.
# -----------------------------------------------------------------------------
_UNIVERSAL_REQUIRED_SPECS = (
    ("_mlps_req_results",           "results_dir",       MLPSTORAGE_RESULTS_DIR_ENVVAR,       "--results-dir",       "-rd", "RESULTS_DIR"),
    ("_mlps_req_systemname",        "systemname",        MLPSTORAGE_SYSTEMNAME_ENVVAR,        "--systemname",        "-sn", "SYSTEMNAME"),
    ("_mlps_req_checkpoint_folder", "checkpoint_folder", MLPSTORAGE_CHECKPOINT_FOLDER_ENVVAR, "--checkpoint-folder", "-cf", "CHECKPOINT_FOLDER"),
)


def _apply_formatter(parser):
    """Recursively set MLPStorageHelpFormatter on every parser in the subparser tree."""
    parser.formatter_class = MLPStorageHelpFormatter
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                _apply_formatter(subparser)


def _build_mode_branch(mode_parser, mode):
    """Build the benchmark subparser tree for a given mode (closed/open/whatif).

    Args:
        mode_parser: The argparse subparser for this mode.
        mode: One of 'closed', 'open', 'whatif'.
    """
    benchmarks = mode_parser.add_subparsers(dest="benchmark", required=True)

    training_parser = benchmarks.add_parser(
        "training",
        help="Training benchmark (unet3d, retinanet)"
    )
    checkpointing_parser = benchmarks.add_parser(
        "checkpointing",
        help="Checkpointing benchmark (llama3-8b, llama3-70b, etc.)"
    )
    vectordb_parser = benchmarks.add_parser(
        "vectordb",
        help="Vector database benchmark (PREVIEW)"
    )
    kvcache_parser = benchmarks.add_parser(
        "kvcache",
        help="KV-cache benchmark for LLM inference"
    )

    add_training_arguments(training_parser, mode)
    add_checkpointing_arguments(checkpointing_parser, mode)
    add_vectordb_arguments(vectordb_parser, mode)
    add_kvcache_arguments(kvcache_parser, mode)


def build_parser():
    """Construct the complete mlpstorage argparse tree.

    Exposed separately from parse_arguments() so tests and tooling can walk
    the real parser tree programmatically (e.g. the --help_all parity test).

    Returns:
        argparse.ArgumentParser: The fully assembled parser.
    """
    parser = argparse.ArgumentParser(
        prog="mlpstorage",
        description="Script to launch the MLPerf Storage benchmark"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    # NOTE: VERSION currently returns a wrong string (mlpstorage_py dist name bug).
    # This will be fixed in Phase 2. Do not add logic here to work around it.

    top = parser.add_subparsers(dest="mode", required=True)

    # Three benchmark mode branches
    for mode_name in ("closed", "open", "whatif"):
        mode_parser = top.add_parser(
            mode_name,
            description=f"Run benchmarks in {mode_name} configuration",
            help=f"{mode_name.capitalize()} submission mode"
        )
        _build_mode_branch(mode_parser, mode_name)

    # Utility siblings — top-level, not nested under modes
    reports_parser = top.add_parser("reports", help="Generate a report from benchmark results")
    history_parser = top.add_parser("history", help="Display benchmark history")
    runs_parser = top.add_parser(
        "runs",
        description="List, inspect and remove the runs recorded in a results-dir",
        help="Manage the runs in a results-dir",
    )
    status_parser = top.add_parser(
        "status",
        description="Per-result submission readiness of a results-dir: runs counted, "
                    "SUBMIT token, paperwork still to do",
        help="Show what stands between each result and a submission",
    )
    submit_parser = top.add_parser(
        "submit",
        description="Check a results-dir with the Rules.md checker and, when every result is "
                    "ready, package it (tarball + manifest + checksum) for upload",
        help="Check and package a results-dir for upload",
    )
    lockfile_parser = top.add_parser("lockfile", help="Generate and verify package lockfiles")
    init_parser = top.add_parser(
        "init",
        description="Initialize a results-dir with the mlperf-results.yaml sentinel",
        help="Pin orgname to a results-dir",
    )
    config_parser = top.add_parser(
        "config",
        description="Show or edit the per-user config file (~/.config/mlpstorage/config.yaml)",
        help="Manage the per-user config file",
    )
    version_parser = top.add_parser("version", description="Print the mlpstorage package version", help="Show installed package version and exit")
    validate_parser = top.add_parser(
        "validate",
        description="Validate a submission package against Rules.md (closed/open/whatif hierarchy).",
        help="Validate a submission package against Rules.md",
    )
    rules_coverage_parser = top.add_parser(
        "rules-coverage",
        description="Self-validation: reconcile Rules.md IDs against @rule-decorated check methods.",
        help="Audit which Rules.md IDs are covered by check methods",
    )
    add_reports_arguments(reports_parser)
    add_history_arguments(history_parser)
    add_runs_arguments(runs_parser)
    add_status_arguments(status_parser)
    add_submit_arguments(submit_parser)
    add_lockfile_arguments(lockfile_parser)
    add_init_arguments(init_parser)
    add_config_arguments(config_parser)
    add_version_arguments(version_parser)
    add_validate_arguments(validate_parser)
    add_rules_coverage_arguments(rules_coverage_parser)

    _apply_formatter(parser)
    return parser


def parse_arguments():
    """Parse command-line arguments for MLPerf Storage benchmarks.

    Returns:
        argparse.Namespace: Parsed and validated arguments.
    """
    _argv = sys.argv[1:]

    # HELP-01: --help_all — print full command tree and exit
    if '--help_all' in _argv:
        from mlpstorage_py.cli.help_formatter import HELP_ALL_TEXT
        print(HELP_ALL_TEXT)
        sys.exit(0)

    # HELP-02 / HELP-03: context-sensitive help — bare, --help, AND incomplete paths
    # R-03-01 fix: call get_context_help_tokens unconditionally (not gated on --help presence).
    # Strip help flags first so they don't appear as positionals. Then strip all remaining
    # option-style tokens (anything starting with '-') so that flags like '-cm 64' interspersed
    # between positionals don't confuse the path lookup.
    _help_flags = {'-h', '--help'}
    _stripped = [a for a in _argv if a not in _help_flags]
    _positionals = [a for a in _stripped if not a.startswith('-')]
    from mlpstorage_py.cli.help_formatter import get_context_help_tokens, SYNOPSIS_TEXT
    _msg = get_context_help_tokens(_positionals)
    if _msg is not None:
        # Fire for: bare invocation, --help at any level, AND bare incomplete paths
        # (e.g., 'mlpstorage closed training' with no --help still shows "next: unet3d | retinanet")
        if _help_flags.intersection(_argv):
            print(SYNOPSIS_TEXT)
            print()
        print(_msg + '  (or -h or --help_all for details)')
        sys.exit(0)
    # _msg is None → leaf level OR unrecognized token → fall through to argparse (HELP-03)

    parser = build_parser()

    argv = sys.argv[1:]
    parsed_args = parser.parse_args(argv)

    # NOTE: No post-parse consolidation for data_access_protocol here.
    # add_storage_type_arguments() registers 'data_access_protocol' as a positional;
    # argparse sets it directly to 'file'|'object'|None. The old --file/--object
    # consolidation block is removed entirely.

    # Fill every flag the command line left out, in one order for every key:
    # flag > --config-file YAML > MLPSTORAGE_* env var > ~/.config/mlpstorage/
    # config.yaml > argparse default (cli/config_layers.py). Records
    # ``config_sources`` so main can say where each value came from.
    # ``config`` manages that file and must work on one the resolver
    # refuses (a workload key in it is a hard error there); it is the
    # one mode that skips the layers.
    if parsed_args.mode != "config":
        apply_config_layers(parsed_args, argv)

    # File-mode --data-dir is enforced here, after the layers, so any tier
    # (--config-file, MLPSTORAGE_DATA_DIR, the per-user file) can satisfy it.
    # Object mode is checked in validate_training_arguments.
    if (
        getattr(parsed_args, 'benchmark', None) == 'training'
        and getattr(parsed_args, 'command', None) in ('datagen', 'run')
        and getattr(parsed_args, 'data_access_protocol', None) == 'file'
        and not getattr(parsed_args, 'data_dir', None)
    ):
        parser.error(
            f"--data-dir is required for training {parsed_args.command} with file storage"
        )

    validate_args(parsed_args)
    return parsed_args


def apply_yaml_config_overrides(args, explicit=frozenset()):
    """
    Apply the ``--config-file`` YAML to a parsed namespace.

    Kept as the public name; the work lives in
    ``cli.config_layers.apply_override_file``. ``explicit`` is the set of
    dests the command line typed, which the file must not overwrite. Direct
    callers (tests, tooling) that pass no ``explicit`` get the historical
    "override everything the file names" behaviour.

    Args:
        args (argparse.Namespace): The parsed command-line arguments

    Returns:
        argparse.Namespace: The updated arguments with YAML values applied
    """
    return apply_override_file(args, explicit)


# These are used by the history tracker to know if logging needs to be updated.
logging_options = ['debug', 'verbose', 'stream_log_level']


def validate_args(args):
    """Validate the whole set of args for the different arg suites

    Args:
        args (argparse.Namespace): The parsed command-line arguments
    """
    if getattr(args, 'mode', None) == 'version':
        return
    # CR-02: enforce env-var-aware "required" gates for --results-dir and
    # --systemname after argparse defaults have settled. ``add_universal_arguments``
    # tags the namespace with ``_mlps_req_results`` / ``_mlps_req_systemname``
    # whenever the calling subcommand opted in to the requirement; the actual
    # resolved value (which may come from the env-var-sourced DEFAULT) is
    # checked here. Using argparse ``required=True`` would have short-
    # circuited before the env-var default applied — that is exactly the bug
    # CR-02 fixed.
    _apply_results_dir_resolution(args)
    _check_universal_required_present(args)
    benchmark = getattr(args, 'benchmark', None)
    if benchmark == 'training':
        validate_training_arguments(args)
    if benchmark == 'checkpointing':
        validate_checkpointing_arguments(args)
    if benchmark == 'vectordb':
        validate_vectordb_arguments(args)
    if benchmark == 'kvcache':
        validate_kvcache_arguments(args)


def _apply_results_dir_resolution(args):
    """Label ``args.results_dir_source`` for ``main``.

    The value itself was filled by ``apply_config_layers`` (``--results-dir``
    > ``results_dir`` in the ``--config-file`` YAML > ``MLPSTORAGE_RESULTS_DIR``
    > ``results_dir`` recorded by ``mlpstorage init``). ``None`` when nothing
    supplied a value. Commands without a ``--results-dir`` flag are left alone.
    """
    if not hasattr(args, 'results_dir'):
        return
    sources = getattr(args, 'config_sources', None) or {}
    args.results_dir = args.results_dir or ""
    args.results_dir_source = sources.get('results_dir') if args.results_dir else None


def _check_universal_required_present(args):
    """Enforce post-parse non-empty checks for the required universal flags.

    Covers three universals per Plan 05-02 / D-09:
      * ``--results-dir / -rd`` (marker: ``_mlps_req_results``)
      * ``--systemname / -sn`` (marker: ``_mlps_req_systemname``)
      * ``--checkpoint-folder / -cf`` (marker: ``_mlps_req_checkpoint_folder``)

    Training's ``--data-dir`` does NOT flow through this gate — it is
    checked in ``training_args.py`` AFTER YAML config merge (D-07), because
    a ``--config-file`` may legitimately supply ``data_dir``.

    Subcommands opt in to "required" via
    ``add_universal_arguments(..., req_results=True, ...)`` and equivalents,
    which set ``args._mlps_req_*`` markers via ``parser.set_defaults``. The
    resolved argparse dest (``results_dir`` / ``systemname`` /
    ``checkpoint_folder``) may be populated by an ``MLPSTORAGE_*`` env-var
    fallback; we treat empty-string and ``None`` as missing.

    Emission model (D-01 / D-02 / D-04 / D-05):
      * One ``error:`` line PER missing flag, using the D-02 verbatim
        template.
      * Immediately BELOW each error line, if the corresponding legacy
        ``MLPERF_*`` env var is set AND the new ``MLPSTORAGE_*`` is not,
        an adjacent ``hint:`` line using the D-05 verbatim template.
      * All missing universals are checked and reported BEFORE
        ``sys.exit()`` (D-01 aggregate-before-exit; T-05-06). No first-
        error short-circuit.
      * Exit code is ``EXIT_CODE.INVALID_ARGUMENTS`` (=2, D-03).

    ``MLPSTORAGE_CHECKPOINT_FOLDER`` has no legacy ``MLPERF_*``
    predecessor (D-08) and is intentionally absent from
    ``_LEGACY_ENVVAR_MAP``, so the hint is never emitted for
    checkpoint-folder — this falls out of the map lookup without a
    special case.
    """
    pending = []
    for (
        marker_attr,
        arg_attr,
        envvar_name,
        long_flag,
        short_flag,
        name_for_template,
    ) in _UNIVERSAL_REQUIRED_SPECS:
        if not getattr(args, marker_attr, False):
            continue
        if getattr(args, arg_attr, None):
            continue
        # D-02 verbatim template — kept on one physical line so the
        # phase-wide grep for the exact template string finds it. The
        # results-dir line also names the third tier, `mlpstorage init`.
        if arg_attr == "results_dir":
            error_line = f"{long_flag}/{short_flag} is required: {HOW_TO_SUPPLY}"
        else:
            error_line = f"{long_flag}/{short_flag} is required: pass it on the command line or set MLPSTORAGE_{name_for_template}"
        # D-04 / D-05: adjacent migration hint when legacy env is set and
        # the new env is not. checkpoint_folder has no legacy pair, so its
        # envvar_name is not a key in _LEGACY_ENVVAR_MAP and the hint stays
        # None without any special-case code.
        hint_line = None
        legacy_name = _LEGACY_ENVVAR_MAP.get(envvar_name)
        if legacy_name is not None:
            legacy_val = os.environ.get(legacy_name, "")
            new_val = os.environ.get(envvar_name, "")
            if legacy_val and not new_val:
                # D-05 verbatim template — single physical line for the
                # phase-wide grep to pin the exact wording.
                hint_line = f"hint: MLPERF_{name_for_template} is set but is no longer read; rename it to MLPSTORAGE_{name_for_template}"
        pending.append((error_line, hint_line))

    if pending:
        for error_line, hint_line in pending:
            print(f"error: {error_line}", file=sys.stderr)
            if hint_line is not None:
                print(hint_line, file=sys.stderr)
        sys.exit(EXIT_CODE.INVALID_ARGUMENTS)


def update_args(args):
    """
    This method is an interface between the CLI and the benchmark class.
    """
    if not hasattr(args, 'num_processes'):
        # Different commands for training use different nomenclature for the number of mpi processes to use
        # Training = num_accelerators
        # Datasize = max_accelerators
        # Datagen = num_processes
        # Checkpoint = num_processes
        # We want to consistently use num_processes in code but the different options for the CLI
        for arg in ['num_processes', 'num_accelerators', 'max_accelerators']:
            if hasattr(args, arg) and type(getattr(args, arg)) is int:
                logging.getLogger(__name__).debug(f'Setting num_processes from {arg}={getattr(args, arg)}')
                setattr(args, 'num_processes', int(getattr(args, arg)))
                break

    if hasattr(args, 'runtime') and hasattr(args, 'queries'):
        # For VectorDB we need runtime or queries. If none defined use a default runtime
        if not args.runtime and not args.queries:
            args.runtime = VECTORDB_DEFAULT_RUNTIME  # Default runtime if not provided

    # Check for list of lists in params and flatten them
    if hasattr(args, 'params') and args.params:
        flattened_params = [item for sublist in args.params for item in sublist]
        # Each token must be of the form KEY=VALUE. argparse with nargs="+"
        # silently accepts space-separated pairs (--param KEY VALUE), which
        # used to surface as "not enough values to unpack" inside the DLIO
        # param processor (issue #469). Catch it here with a message that
        # actually names the offending token and the right syntax.
        bad = [tok for tok in flattened_params if '=' not in tok]
        if bad:
            print(
                "Error: --params expects KEY=VALUE tokens joined with '=' "
                "(no space).\n"
                f"  Offending token(s): {bad}\n"
                "  Wrong: --param dataset.num_files_train 35000\n"
                "  Right: --param dataset.num_files_train=35000\n"
                "  Multiple overrides: --params A=1 B=2 C=3"
            )
            sys.exit(EXIT_CODE.INVALID_ARGUMENTS)

        # storage#795: reject known-typo dotted keys at CLI-parse time so the
        # user doesn't wait through MPI cluster collection, code-image
        # capture, and results-directory creation only to be told the
        # parameter was disallowed. All three keys are auto-injected by the
        # tool in --object mode, so the user typically does not need to pass
        # them at all.
        from mlpstorage_py.rules.param_hints import KNOWN_PARAM_TYPOS
        typo_hits = []
        for tok in flattened_params:
            key = tok.split('=', 1)[0]
            canonical = KNOWN_PARAM_TYPOS.get(key)
            if canonical:
                typo_hits.append((key, canonical))
        if typo_hits:
            lines = ["Error: --params contains dotted key(s) that are not real DLIO parameters:"]
            for typed, canonical in typo_hits:
                lines.append(f"  '{typed}' → did you mean '{canonical}'?")
            lines.append(
                "  Note: storage.storage_options.* keys are auto-injected by "
                "the tool in --object mode; you typically do not need to pass "
                "them explicitly."
            )
            print("\n".join(lines))
            sys.exit(EXIT_CODE.INVALID_ARGUMENTS)

        setattr(args, 'params', flattened_params)

    if hasattr(args, 'mpi_params') and args.mpi_params:
        # --mpi-params is collected with action="append" as a list of raw
        # strings, each potentially containing several space-separated MPI
        # flags, e.g. ["-genv PMI_VERSION=2 -genv FI_PROVIDER=tcp"].
        # MPI flags begin with '-', so nargs="+" used to reject them with
        # "expected at least one argument" (see issue #422). We now accept the
        # whole string and tokenize it here with shlex so quoting is honored
        # and downstream (generate_mpi_prefix_cmd) receives a flat token list.
        flattened_mpi_params = []
        for chunk in args.mpi_params:
            if isinstance(chunk, (list, tuple)):
                # Backwards-compat: tolerate the old nested-list shape.
                for item in chunk:
                    flattened_mpi_params.extend(shlex.split(item))
            else:
                flattened_mpi_params.extend(shlex.split(chunk))
        setattr(args, 'mpi_params', flattened_mpi_params)

    if hasattr(args, 'hosts') and args.hosts is not None:
        # Accept any of the following equivalent forms and normalize to a clean list:
        #   --hosts h1 h2 h3              -> ['h1', 'h2', 'h3']
        #   --hosts h1,h2,h3              -> ['h1', 'h2', 'h3']
        #   --hosts 'h1 h2 h3'            -> ['h1', 'h2', 'h3']   (quoted, e.g. from YAML)
        #   --hosts='h1,h2,h3'            -> ['h1', 'h2', 'h3']   (DLIO subprocess form)
        #   --hosts='h1 h2 h3'            -> ['h1', 'h2', 'h3']   (quoted after '=')
        # This defends against the argparse + nargs='+' + '=' interaction documented in
        # https://github.com/mlcommons/storage/issues/322.
        raw = args.hosts if isinstance(args.hosts, list) else [args.hosts]
        normalized = []
        for item in raw:
            if not isinstance(item, str):
                continue
            for tok in re.split(r'[,\s]+', item.strip()):
                if tok:
                    normalized.append(tok)
        if not normalized:
            print("ERROR: --hosts is empty after parsing", file=sys.stderr)
            sys.exit(EXIT_CODE.INVALID_ARGUMENTS)
        args.hosts = normalized

        if getattr(args, 'num_client_hosts', None) is None:
            setattr(args, "num_client_hosts", len(args.hosts))


if __name__ == "__main__":
    args = parse_arguments()
    import pprint
    pprint.pprint(vars(args))
