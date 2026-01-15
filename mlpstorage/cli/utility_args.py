"""
Utility CLI argument builders for non-benchmark commands.

This module defines the CLI arguments for utility commands like
reports and history.
"""

from mlpstorage.cli.common_args import (
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

    add_universal_arguments(reportgen)


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
        add_universal_arguments(_parser)
