"""
CLI argument builder for ``mlpstorage runs`` (run management).

``runs`` is a top-level utility sibling of ``history`` and ``reports``. Its
subcommands take ``--results-dir/-rd`` (resolved like everywhere else:
flag > ``MLPSTORAGE_RESULTS_DIR`` > the default recorded by ``mlpstorage
init``) and nothing else from the universal set — they manage a tree,
they do not emit into one.
"""

from __future__ import annotations

from mlpstorage_py.cli.common_args import add_results_dir_argument
from mlpstorage_py.runs.ledger import MODES, STATUSES

BENCHMARK_CHOICES = ("training", "checkpointing", "vectordb", "kvcache")


def _add_selection_filters(parser, *, with_status=True):
    parser.add_argument(
        "--mode",
        dest="mode_filter",
        choices=MODES,
        default=None,
        help="Only runs from this submission mode",
    )
    parser.add_argument(
        "--benchmark",
        choices=BENCHMARK_CHOICES,
        default=None,
        help="Only runs of this benchmark",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Only runs of this model (vectordb: <engine>/<index>, e.g. milvus/DISKANN)",
    )
    parser.add_argument(
        "--systemname", "-sn",
        default=None,
        help="Only runs from this system-under-test",
    )
    if with_status:
        parser.add_argument(
            "--status",
            choices=STATUSES,
            default=None,
            help="Only runs in this state",
        )


def add_runs_arguments(parser):
    """Register ``list``, ``show``, ``rm``, ``purge`` and ``gc``."""
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")
    parser.required = True

    list_parser = subparsers.add_parser(
        "list",
        help="List the runs in the results-dir with their IDs and status",
    )
    add_results_dir_argument(list_parser)
    _add_selection_filters(list_parser)
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the runs as a JSON array instead of a table",
    )

    show_parser = subparsers.add_parser(
        "show",
        help="Show one run: identity, status, code image, metadata excerpt, files",
    )
    add_results_dir_argument(show_parser)
    show_parser.add_argument("run_id", type=int, help="Run ID from `mlpstorage runs list`")

    rm_parser = subparsers.add_parser(
        "rm",
        help="Move runs to <results-dir>/.mlps/trash (restore by hand; `runs purge` deletes)",
    )
    add_results_dir_argument(rm_parser)
    rm_parser.add_argument(
        "run_ids",
        type=int,
        nargs="*",
        help="Run IDs to remove (omit to select by --status / --older-than)",
    )
    rm_parser.add_argument(
        "--status",
        choices=STATUSES,
        default=None,
        help="Remove runs in this state",
    )
    rm_parser.add_argument(
        "--older-than",
        default=None,
        metavar="AGE|DATE",
        help="Remove runs started before this: 12h, 7d, 2w, or a date (2026-09-01)",
    )
    rm_parser.add_argument(
        "--keep-last",
        type=int,
        default=None,
        metavar="N",
        help="Keep the N newest runs of the selection",
    )
    rm_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Do not ask for confirmation (required when stdin is not a terminal)",
    )

    purge_parser = subparsers.add_parser(
        "purge",
        help="Permanently delete everything in <results-dir>/.mlps/trash",
    )
    add_results_dir_argument(purge_parser)
    purge_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Do not ask for confirmation (required when stdin is not a terminal)",
    )

    gc_parser = subparsers.add_parser(
        "gc",
        help="Move code-image pool directories no run points at into the trash",
    )
    add_results_dir_argument(gc_parser)
    gc_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Do not ask for confirmation (required when stdin is not a terminal)",
    )
    return parser
