"""
Utility CLI argument builders for non-benchmark commands.

This module defines the CLI arguments for utility commands like
reports and history.
"""

from mlpstorage_py.cli.common_args import (
    HELP_MESSAGES,
    add_universal_arguments,
)


def add_reports_arguments(parser):
    """Add reports command arguments to the parser.

    Args:
        parser: Argparse subparser for the reports command.
    """
    reports_subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Sub-commands"
    )
    parser.required = True

    reportgen = reports_subparsers.add_parser(
        'reportgen',
        help=HELP_MESSAGES['reportgen']
    )

    reportgen.add_argument(
        '--output-dir',
        type=str,
        help=HELP_MESSAGES['output_dir']
    )

    add_universal_arguments(reportgen, req_results=True)


def add_history_arguments(parser):
    """Add history command arguments to the parser.

    Args:
        parser: Argparse subparser for the history command.
    """
    history_subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Sub-commands"
    )
    parser.required = True

    history = history_subparsers.add_parser(
        'show',
        help="Show command history"
    )
    history.add_argument(
        '--limit', '-n',
        type=int,
        help="Limit to the N most recent commands"
    )
    history.add_argument(
        '--id', '-i',
        type=int,
        help="Show a specific command by ID"
    )

    rerun = history_subparsers.add_parser(
        'rerun',
        help="Re-run a command from history"
    )
    rerun.add_argument(
        'rerun_id',
        type=int,
        help="ID of the command to re-run"
    )

    for _parser in [history, rerun]:
        add_universal_arguments(_parser, req_results=True)


def add_version_arguments(parser):
    """Add version command arguments to the parser.

    No subcommands or flags — version is printed and the process exits in main.py.
    """
    pass


def add_validate_arguments(parser):
    """Add ``validate`` (submission-checker) command arguments to the parser.

    Mirrors the standalone ``submission_checker`` CLI argument surface so a
    single namespace can be passed to ``submission_checker.main.run``.
    """
    parser.add_argument(
        "input",
        help="Path to a submission directory (closed/<submitter> or open/<submitter>)."
    )
    parser.add_argument(
        "--submitters",
        default=None,
        help="Comma-separated list of submitters to check (default: all submitters under the input dir)."
    )
    # Lazy import — DEFAULT_SPEC_VERSION reads through to constants.py, which
    # pulls in submission_checker module deps. Keeping the import inside the
    # arg builder avoids paying that cost at top-level CLI parser construction
    # time (matters for `mlpstorage --help_all` and similar fast paths).
    from mlpstorage_py.submission_checker.constants import DEFAULT_SPEC_VERSION
    parser.add_argument(
        "--mlperf-version",
        dest="version",
        default=DEFAULT_SPEC_VERSION,
        help=(
            "MLPerf Storage spec version that the submission package claims "
            "to conform to (default: %(default)s, derived from this "
            "package's release version's major.minor)."
        ),
    )
    parser.add_argument(
        "--csv",
        default="summary.csv",
        help="Path to write the summary CSV (default: summary.csv in the current directory)."
    )
    parser.add_argument(
        "--skip-output-file",
        action="store_true",
        help="Do not write a per-submission output file alongside the CSV summary."
    )
    parser.add_argument(
        "--reference-checksum",
        default=None,
        help="Override the bundled REFERENCE_CHECKSUMS for the code/ tree MD5 check."
    )


def add_rules_coverage_arguments(parser):
    """Add ``rules-coverage`` command arguments to the parser.

    The tool reconciles every Rules.md §2/§3/§4 ID against the live
    @rule-decorated check methods plus the OUT_OF_SCOPE_RULES /
    STUB_COVERAGE / SCHEMA_ERROR_RULE_MAP registries.
    """
    parser.add_argument(
        "--rules-md",
        dest="rules_md",
        default=None,
        help="Path to Rules.md (defaults to the project-root Rules.md)."
    )
