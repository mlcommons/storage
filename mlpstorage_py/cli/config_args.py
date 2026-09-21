"""
CLI argument builder for ``mlpstorage config`` (the per-user config file).

``config`` is a top-level utility sibling of ``init``. It manages
``$XDG_CONFIG_HOME/mlpstorage/config.yaml`` (``~/.config/mlpstorage/config.yaml``)
— the file ``mlpstorage init`` writes ``results_dir`` into and every command
reads environment defaults from — so no key needs a hand edit. It takes no
``--results-dir`` and nothing from the universal set: it never touches a
results tree.
"""

from __future__ import annotations

from mlpstorage_py.results_dir.user_config import USER_CONFIG_KEYS

_KEYS_HELP = ", ".join(USER_CONFIG_KEYS)


def add_config_arguments(parser):
    """Register ``show``, ``set``, ``unset`` and ``path``."""
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")
    parser.required = True

    show_parser = subparsers.add_parser(
        "show",
        help="Print every key the file may carry, its value, and any env var that outranks it",
    )
    show_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Machine-readable form of the same report",
    )

    set_parser = subparsers.add_parser(
        "set",
        help="Record a default in the per-user config file",
    )
    set_parser.add_argument(
        "key",
        help=f"Key to set; one of: {_KEYS_HELP}",
    )
    set_parser.add_argument(
        "values",
        nargs="+",
        metavar="VALUE",
        help=(
            "The value. Several values (or one comma-separated value for hosts) "
            "store a list; true/false for the boolean keys. Checked as the flag would be."
        ),
    )

    unset_parser = subparsers.add_parser(
        "unset",
        help="Remove a key from the per-user config file",
    )
    unset_parser.add_argument(
        "key",
        help="Key to remove; any key, so a stray workload key can be cleared",
    )

    subparsers.add_parser(
        "path",
        help="Print the per-user config file path",
    )
    return parser
