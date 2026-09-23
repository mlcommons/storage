"""
CLI argument builder for ``mlpstorage submit`` (check, package, record).

``submit`` is a top-level utility sibling of ``status``: it reads an
initialized results-dir (``--results-dir/-rd``, resolved flag >
``MLPSTORAGE_RESULTS_DIR`` > the default recorded by ``mlpstorage init``),
regenerates the rollup tables, runs the Rules.md checker and, when every
result is ready, writes the package. ``--dry-run`` stops before writing.
"""

from __future__ import annotations

from mlpstorage_py.cli.common_args import add_results_dir_argument


def add_submit_arguments(parser):
    """Register the ``submit`` flags on ``parser`` (a leaf: no subcommands)."""
    add_results_dir_argument(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Regenerate the rollups and run the checker, describe the package, write nothing",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Where to write the package: a directory, or a .tar.gz file path "
             "(default <results-dir>/.mlps/packages/<org>-<edition>-<timestamp>.tar.gz)",
    )
    return parser
