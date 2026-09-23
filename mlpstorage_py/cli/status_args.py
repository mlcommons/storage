"""
CLI argument builder for ``mlpstorage status`` (per-result submission readiness).

``status`` is a top-level utility sibling of ``runs``: it reads an
initialized results-dir (``--results-dir/-rd``, resolved flag >
``MLPSTORAGE_RESULTS_DIR`` > the default recorded by ``mlpstorage init``)
and prints one row per result. Its selection flags are the ``runs list``
ones, minus ``--status`` (a per-run ledger state) and plus ``--submit`` (the
per-result token).
"""

from __future__ import annotations

from mlpstorage_py.cli.common_args import add_results_dir_argument
from mlpstorage_py.cli.runs_args import add_selection_filters
from mlpstorage_py.readiness import SUBMIT_INVALID, SUBMIT_PAPERWORK, SUBMIT_READY, SUBMIT_SHORT

SUBMIT_CHOICES = (SUBMIT_SHORT, SUBMIT_INVALID, SUBMIT_PAPERWORK, SUBMIT_READY)


def add_status_arguments(parser):
    """Register the ``status`` flags on ``parser`` (a leaf: no subcommands)."""
    add_results_dir_argument(parser)
    add_selection_filters(parser, with_status=False)
    parser.add_argument(
        "--submit",
        choices=SUBMIT_CHOICES,
        default=None,
        help="Only results with this SUBMIT token",
    )
    parser.add_argument(
        "--runs",
        action="store_true",
        help="Expand each result into its runs (ID, status, started, counted)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the readiness of the selected results as JSON instead of a table",
    )
    return parser
