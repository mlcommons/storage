"""
CLI argument builder for the ``mlpstorage init`` subcommand.

Registers two positionals — ``orgname`` and ``path`` — on the supplied
subparser. No flags, no ``--results-dir`` (the ``path`` positional IS the
results-dir target), no universal-arguments (per PATTERNS.md row
``cli/init_args.py``: no sentinel exists yet at init time, so the
``--results-dir`` defaulting logic does not apply).

Naming distinction (RESEARCH.md Pitfall 5): the ``<orgname>`` positional is
the same identity as the per-submission ``submitter`` name in Rules.md §2.1.5,
but it is **pinned to one results-dir at init time** — never passed per-run.
See CONTEXT.md "Locked Decisions → mlpstorage init" for the rationale.

Refs: 01-canonical-layout-and-init / 01-02-PLAN.md Task 1; PATTERNS.md row
``cli/init_args.py``; CONTEXT.md "Locked Decisions"; RESEARCH.md Pitfalls 3+5.
"""

from __future__ import annotations


def add_init_arguments(parser):
    """Add ``orgname`` and ``path`` positionals to the ``init`` subparser.

    Args:
        parser: The ``init`` subparser created by ``cli_parser.py`` via
            ``top.add_parser("init", ...)``.

    Returns:
        The same parser, for chaining if a caller wants it.
    """
    parser.add_argument(
        "orgname",
        type=str,
        help=(
            "Organization name to pin to this results-dir "
            "(Rules.md §2.1.5 submitter). This is the same identity as the "
            "per-submission 'submitter' name, but it is pinned to one "
            "results-dir at init time — not passed per-run."
        ),
    )
    parser.add_argument(
        "path",
        type=str,
        help=(
            "Filesystem path to initialize as a results-dir. If the path does "
            "not exist, it will be created (the parent directory must already "
            "exist). If the path exists and is empty, it is initialized in "
            "place; if it already contains an mlperf-results.yaml sentinel "
            "with a matching orgname, init exits 0 (idempotent)."
        ),
    )
    return parser
