#!/usr/bin/env python3
"""Backend-agnostic collection administration CLI.

Provides subcommands for inspecting and managing collections across
any registered vector-database backend (Milvus, pgvector, Elasticsearch,
etc.)  All heavy lifting delegates to the :class:`VectorDBBackend`
admin methods so behaviour is consistent across databases.

Usage examples::

    # Interactive mode -- discover backends, pick one, browse collections
    collection-admin interactive

    # List all collections on a Milvus server
    collection-admin --backend milvus list

    # Detailed info for one collection
    collection-admin --backend milvus info my_collection

    # Show indexes
    collection-admin --backend pgvector indexes my_collection

    # Collection statistics
    collection-admin --backend elasticsearch stats my_collection

    # Drop a collection (requires --yes for safety)
    collection-admin --backend milvus drop my_collection --yes

    # Drop an index
    collection-admin --backend pgvector drop-index my_collection

Connection parameters are sourced from environment variables using the
``{BACKEND}__{PARAM}`` convention (see ``_env.py``), from a ``.env``
file, or from ``--param key=value`` CLI flags.
"""

from __future__ import annotations

import sys

# ------------------------------------------------------------------
# Direct-execution bootstrap (same pattern as run_benchmark.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import importlib
    import pathlib

    _this = pathlib.Path(__file__).resolve()
    _pkg_root = str(_this.parent.parent.parent)
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)

    _mod = importlib.import_module("vdbbench.benchmark.collection_admin")
    raise SystemExit(_mod.main())

# ------------------------------------------------------------------
# Normal imports (only reached when loaded as a package member).
# ------------------------------------------------------------------

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tabulate import tabulate as _tabulate

from .backends import registry, get_backend
from .backends._env import load_env_file, env_for_backend
from .backends.base import BackendDescriptor, VectorDBBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# =====================================================================
# Output formatting helpers
# =====================================================================

def _json_out(data: Any) -> None:
    """Print *data* as indented JSON to stdout."""
    print(json.dumps(data, indent=2, default=str))


def _table_out(rows: List[Dict[str, Any]], keys: Optional[List[str]] = None) -> None:
    """Print rows as a simple aligned table."""
    if not rows:
        print("(no results)")
        return

    keys = keys or list(rows[0].keys())
    # Column widths
    widths = {k: len(k) for k in keys}
    for row in rows:
        for k in keys:
            widths[k] = max(widths[k], len(str(row.get(k, ""))))

    header = "  ".join(k.ljust(widths[k]) for k in keys)
    sep = "  ".join("-" * widths[k] for k in keys)
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row.get(k, "")).ljust(widths[k]) for k in keys))


# =====================================================================
# Backend connection helper
# =====================================================================

def _connect_backend(
    backend_name: str,
    extra_params: Optional[Dict[str, str]] = None,
) -> VectorDBBackend:
    """Instantiate, connect, and return a backend.

    Connection parameters come from (highest-precedence-first):
    1. ``--param key=value`` CLI flags (*extra_params*).
    2. Environment variables (``{BACKEND}__{PARAM}``).
    3. Defaults from the backend descriptor.
    """
    load_env_file()

    desc = registry.get(backend_name)
    if desc is None:
        available = ", ".join(registry.names()) or "(none)"
        print(f"Unknown backend '{backend_name}'.  Available: {available}",
              file=sys.stderr)
        sys.exit(1)

    # Merge env + CLI overrides
    conn = env_for_backend(backend_name, desc)
    if extra_params:
        conn.update(extra_params)

    backend = desc.backend_class()
    backend.connect(**conn)
    return backend


# =====================================================================
# Non-interactive subcommand handlers
# =====================================================================

def _cmd_list(backend: VectorDBBackend, args: argparse.Namespace) -> None:
    """``list`` -- show all collections."""
    names = backend.list_collections()
    if args.json:
        _json_out(names)
        return
    if not names:
        print("(no collections found)")
        return
    for n in sorted(names):
        print(n)


def _cmd_info(backend: VectorDBBackend, args: argparse.Namespace) -> None:
    """``info`` -- detailed metadata for one collection."""
    info = backend.get_collection_info(args.collection)
    if args.json:
        _json_out(info)
        return

    print(f"\nCollection: {info['name']}")
    print(f"  Rows:       {info.get('row_count', '?'):,}")
    print(f"  Dimension:  {info.get('dimension') or '?'}")
    print(f"  Metric:     {info.get('metric_type') or '?'}")
    print(f"  Index type: {info.get('index_type') or '?'}")

    schema = info.get("schema", [])
    if schema:
        print("\n  Schema:")
        for fld in schema:
            extras = []
            if fld.get("dim"):
                extras.append(f"dim={fld['dim']}")
            if fld.get("is_primary"):
                extras.append("PK")
            suffix = f"  ({', '.join(extras)})" if extras else ""
            print(f"    - {fld['name']}: {fld.get('dtype', '?')}{suffix}")

    for key in ("num_partitions", "partitions"):
        if key in info:
            print(f"  {key}: {info[key]}")
    print()


def _cmd_indexes(backend: VectorDBBackend, args: argparse.Namespace) -> None:
    """``indexes`` -- list indexes on a collection."""
    indexes = backend.list_indexes(args.collection)
    if args.json:
        _json_out(indexes)
        return
    if not indexes:
        print(f"No indexes found on '{args.collection}'")
        return
    _table_out(indexes)


def _cmd_stats(backend: VectorDBBackend, args: argparse.Namespace) -> None:
    """``stats`` -- operational statistics for a collection."""
    stats = backend.get_collection_stats(args.collection)
    if args.json:
        _json_out(stats)
        return
    for k, v in stats.items():
        label = k.replace("_", " ").title()
        if isinstance(v, int) and v > 999:
            print(f"  {label}: {v:,}")
        else:
            print(f"  {label}: {v}")


def _cmd_drop(backend: VectorDBBackend, args: argparse.Namespace) -> None:
    """``drop`` -- drop a collection (destructive!)."""
    name = args.collection
    if not backend.collection_exists(name):
        print(f"Collection '{name}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not args.yes:
        try:
            answer = input(f"Really DROP collection '{name}'? (yes/[no]) > ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer.lower() != "yes":
            print("Aborted.")
            return

    backend.drop_collection(name)
    print(f"Dropped: {name}")


def _cmd_drop_index(backend: VectorDBBackend, args: argparse.Namespace) -> None:
    """``drop-index`` -- drop an index from a collection."""
    name = args.collection
    idx = getattr(args, "index_name", None)

    if not args.yes:
        target = f"index '{idx}'" if idx else "the vector index"
        try:
            answer = input(
                f"Really DROP {target} on '{name}'? (yes/[no]) > "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer.lower() != "yes":
            print("Aborted.")
            return

    backend.drop_index(name, index_name=idx)
    print(f"Dropped index on '{name}'")


# =====================================================================
# Interactive mode -- backend discovery, health-check, menus
# =====================================================================

@dataclass
class BackendStatus:
    """Result of probing one backend."""
    name: str
    display_name: str
    configured: bool = False
    healthy: bool = False
    error: str = ""
    conn_params: Dict[str, Any] = field(default_factory=dict)
    descriptor: Optional[BackendDescriptor] = None


def discover_backends(env_path: Optional[str] = None) -> List[BackendStatus]:
    """Probe every active backend and return their status.

    For each active backend registered in the global registry:

    1. Load connection params from ``.env`` / environment variables.
    2. If at least one connection parameter is configured, attempt
       ``connect()`` followed by ``disconnect()`` as a health check.
    3. If no env vars are set, fall back to the defaults declared in the
       backend descriptor and try to connect anyway -- but mark it as
       *not explicitly configured*.
    """
    load_env_file(env_path)

    results: List[BackendStatus] = []
    for desc in registry.list_backends():
        status = BackendStatus(
            name=desc.name,
            display_name=desc.display_name,
            descriptor=desc,
        )

        # Gather connection params from env
        env_params = env_for_backend(desc.name, desc)
        status.configured = bool(env_params)

        # Build full param set: defaults + env overrides
        conn: Dict[str, Any] = {}
        for p in desc.connection_params:
            if p.default is not None:
                conn[p.name] = p.default
        conn.update(env_params)
        status.conn_params = conn

        # Attempt ping
        try:
            backend = desc.backend_class()
            backend.connect(**conn)
            backend.disconnect()
            status.healthy = True
        except Exception as exc:
            status.healthy = False
            status.error = str(exc)

        results.append(status)

    return results


def _sep(text: str) -> str:
    """Return a ``─`` line matching the widest line in *text*."""
    width = max((len(l) for l in text.splitlines()), default=0)
    return "─" * width


def pick_backend(statuses: List[BackendStatus]) -> Optional[BackendStatus]:
    """Display a table of backends and let the user choose one.

    Only healthy backends are selectable.  Returns ``None`` if the user
    cancels or no healthy backends exist.
    """
    headers = ["Idx", "Backend", "Configured", "Status", "Details"]
    rows = []
    for i, s in enumerate(statuses):
        configured = "Yes" if s.configured else "defaults"
        if s.healthy:
            status_str = "Healthy"
            detail = ", ".join(f"{k}={v}" for k, v in s.conn_params.items()
                               if v is not None and k != "password")
        else:
            status_str = "Unreachable"
            detail = s.error[:60] if s.error else ""
        rows.append([i, s.display_name, configured, status_str, detail])

    table = _tabulate(rows, headers=headers, tablefmt="github")
    sep = _sep(table)
    print(f"\n{sep}")
    print(table)
    print(sep)

    healthy_ids = [i for i, s in enumerate(statuses) if s.healthy]
    if not healthy_ids:
        print("\nNo healthy backends found.  Check your .env configuration.")
        return None

    print(f"\nHealthy backends: {', '.join(str(i) for i in healthy_ids)}")
    while True:
        try:
            choice = input("Select backend idx (or q to quit) > ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if choice.lower() == "q":
            return None
        try:
            idx = int(choice)
        except ValueError:
            print(f"Invalid input '{choice}'. Enter a backend idx or q to quit.")
            continue
        if idx < 0 or idx >= len(statuses):
            print(f"Index {idx} out of range. Select an idx between 0 and {len(statuses) - 1}.")
            continue
        if not statuses[idx].healthy:
            print(f"Backend '{statuses[idx].display_name}' is not healthy. Select a healthy idx.")
            continue
        return statuses[idx]


def _connect_from_status(status: BackendStatus) -> VectorDBBackend:
    """Instantiate and connect a backend from its discovered status."""
    backend = status.descriptor.backend_class()
    backend.connect(**status.conn_params)
    return backend


def pick_collection(
    backend: VectorDBBackend,
    backend_name: str,
) -> Optional[str]:
    """List collections on the backend and let the user choose one.

    Returns the collection *name* or ``None`` if cancelled.
    """
    try:
        names = backend.list_collections()
    except Exception as exc:
        print(f"Failed to list collections: {exc}")
        return None

    if not names:
        print(f"\nNo collections found on '{backend_name}'.")
        return None

    headers = ["Idx", "Collection", "Rows", "Dim", "Index", "Metric"]
    rows = []
    for i, name in enumerate(sorted(names)):
        try:
            info = backend.get_collection_info(name)
            row_count = (f"{info.get('row_count', '?'):,}"
                         if isinstance(info.get('row_count'), int) else "?")
            dim = info.get("dimension") or "?"
            idx_type = info.get("index_type") or "?"
            metric = info.get("metric_type") or "?"
        except Exception:
            row_count = "?"
            dim = "?"
            idx_type = "?"
            metric = "?"
        rows.append([i, name, row_count, dim, idx_type, metric])

    table = _tabulate(rows, headers=headers, tablefmt="github")
    sep = _sep(table)
    print(f"\n{sep}")
    print(table)
    print(sep)

    while True:
        try:
            choice = input("\nSelect collection idx (or b=back, q=quit) > ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if choice.lower() == "b":
            return None
        if choice.lower() == "q":
            print("Bye.")
            sys.exit(0)
        try:
            idx = int(choice)
        except ValueError:
            print(f"Invalid input '{choice}'. Enter a collection idx, b, or q.")
            continue
        if idx < 0 or idx >= len(rows):
            print(f"Index {idx} out of range. Select an idx between 0 and {len(rows) - 1}.")
            continue
        return rows[idx][1]  # collection name


# ── Interactive operation helpers ──────────────────────────────────

def _iop_info(backend: VectorDBBackend, collection: str) -> None:
    """Display detailed collection info."""
    try:
        info = backend.get_collection_info(collection)
    except Exception as exc:
        print(f"Failed to get info: {exc}")
        return

    print(f"\n{'='*70}")
    print(f"Collection: {info['name']}")
    print(f"{'='*70}")
    row_count = info.get("row_count", "?")
    if isinstance(row_count, int):
        print(f"Rows:       {row_count:,}")
    else:
        print(f"Rows:       {row_count}")
    print(f"Dimension:  {info.get('dimension') or '?'}")
    print(f"Metric:     {info.get('metric_type') or '?'}")
    print(f"Index type: {info.get('index_type') or '?'}")

    schema = info.get("schema", [])
    if schema:
        print("\nSchema:")
        for fld in schema:
            extras = []
            if fld.get("dim"):
                extras.append(f"dim={fld['dim']}")
            if fld.get("is_primary"):
                extras.append("PK")
            suffix = f"  ({', '.join(extras)})" if extras else ""
            print(f"  - {fld['name']}: {fld.get('dtype', '?')}{suffix}")

    if "num_partitions" in info:
        print(f"\nPartitions: {info['num_partitions']}")
        for p in info.get("partitions", []):
            print(f"  - {p}")
    print(f"{'='*70}\n")


def _iop_stats(backend: VectorDBBackend, collection: str) -> None:
    """Display operational statistics."""
    try:
        stats = backend.get_collection_stats(collection)
    except Exception as exc:
        print(f"Failed to get stats: {exc}")
        return

    print(f"\nStats for '{collection}':")
    for k, v in stats.items():
        label = k.replace("_", " ").title()
        if isinstance(v, int) and v > 999:
            print(f"  {label}: {v:,}")
        else:
            print(f"  {label}: {v}")
    print()


def _iop_indexes(backend: VectorDBBackend, collection: str) -> None:
    """List indexes on a collection."""
    try:
        indexes = backend.list_indexes(collection)
    except Exception as exc:
        print(f"Failed to list indexes: {exc}")
        return

    if not indexes:
        print(f"No indexes on '{collection}'.")
        return

    print(f"\nIndexes on '{collection}':")
    print(_tabulate(
        [{k: v for k, v in idx.items()} for idx in indexes],
        headers="keys",
        tablefmt="github",
    ))
    print()


def _iop_compact(backend: VectorDBBackend, collection: str) -> None:
    """Trigger compaction (if supported)."""
    try:
        print(f"Starting compaction on '{collection}'...")
        backend.compact(collection)
        print("Compaction completed.")
    except NotImplementedError:
        print("Compaction is not supported by this backend.")
    except Exception as exc:
        print(f"Compact failed: {exc}")


def _iop_drop_index(backend: VectorDBBackend, collection: str) -> None:
    """Drop the vector index from a collection."""
    try:
        confirm = input(
            f"Really DROP the index on '{collection}'? (yes/[no]) > "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        confirm = ""
    if confirm.lower() != "yes":
        print("Aborted.")
        return

    try:
        backend.drop_index(collection)
        print(f"Index dropped on '{collection}'.")
    except NotImplementedError:
        print("drop_index is not supported by this backend.")
    except Exception as exc:
        print(f"Drop index failed: {exc}")


def _iop_delete(backend: VectorDBBackend, collection: str) -> None:
    """Drop (delete) a collection entirely."""
    try:
        confirm = input(
            f"Really DROP collection '{collection}'? "
            "This is irreversible. (yes/[no]) > "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        confirm = ""
    if confirm.lower() != "yes":
        print("Aborted; collection kept.")
        return

    try:
        backend.drop_collection(collection)
        print(f"Collection '{collection}' dropped.")
    except Exception as exc:
        print(f"Delete failed: {exc}")


_INTERACTIVE_OPS = {
    "i": ("info", "Detailed collection info", _iop_info),
    "s": ("stats", "Operational statistics", _iop_stats),
    "x": ("indexes", "List indexes", _iop_indexes),
    "c": ("compact", "Trigger compaction", _iop_compact),
    "di": ("drop-index", "Drop the vector index", _iop_drop_index),
    "d": ("delete", "Drop the collection", _iop_delete),
    "b": ("back", "Back to collection list", None),
    "q": ("quit", "Exit", None),
}


def operations_menu(
    backend: VectorDBBackend,
    collection: str,
    backend_name: str,
) -> bool:
    """Run the operations loop for a single collection.

    Returns ``True`` to go back to the collection picker,
    ``False`` to exit.
    """
    while True:
        header = f"  [{backend_name}] Collection: '{collection}'"
        cmd_lines = [f"    {key:<4}  {name:<12}  {desc}"
                     for key, (name, desc, _) in _INTERACTIVE_OPS.items()]
        body = "\n".join([header, "  Available commands:"] + cmd_lines)
        sep = _sep(body)
        print(f"\n{sep}")
        print(body)
        print(sep)

        try:
            choice = input("Enter command > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False

        if choice == "q":
            print("Bye.")
            sys.exit(0)

        if choice == "b":
            return True

        entry = _INTERACTIVE_OPS.get(choice)
        if entry is None:
            print(f"Unknown command '{choice}'. Enter one of: "
                  f"{', '.join(_INTERACTIVE_OPS.keys())}")
            continue

        _, _, handler = entry
        if handler is not None:
            handler(backend, collection)

            # If the collection was deleted, return to the picker
            if choice == "d":
                return True


def _cmd_interactive(args: argparse.Namespace) -> int:
    """``interactive`` -- menu-driven backend and collection manager."""
    env_path = getattr(args, "env_file", None)

    print("Discovering backends...")
    statuses = discover_backends(env_path=env_path)

    if not statuses:
        print("No backends registered.  Is the benchmark package installed?")
        return 1

    backend: Optional[VectorDBBackend] = None
    current_status: Optional[BackendStatus] = None

    while True:
        # ── backend picker ────────────────────────────────────────
        if backend is not None:
            print(f"\nCurrently connected to: {current_status.display_name}")
            try:
                switch = input("Switch backend? (y/[n]) > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if switch == "y":
                try:
                    backend.disconnect()
                except Exception:
                    pass
                backend = None

        if backend is None:
            chosen = pick_backend(statuses)
            if chosen is None:
                print("Bye.")
                break
            try:
                backend = _connect_from_status(chosen)
                current_status = chosen
                print(f"\nConnected to {chosen.display_name}.")
            except Exception as exc:
                print(f"Connection failed: {exc}")
                continue

        # ── collection picker ─────────────────────────────────────
        col_name = pick_collection(backend, current_status.display_name)
        if col_name is None:
            try:
                backend.disconnect()
            except Exception:
                pass
            backend = None
            continue

        # ── operations menu ───────────────────────────────────────
        go_back = operations_menu(backend, col_name, current_status.display_name)
        if not go_back:
            break

    # Cleanup
    if backend is not None:
        try:
            backend.disconnect()
        except Exception:
            pass

    return 0


# =====================================================================
# Argument parser
# =====================================================================

_EPILOG = """\
concepts:
  collection  The data container that holds vectors and their metadata
              (IDs, dimensions, schema).  Mapped to a Milvus Collection,
              a PostgreSQL table (pgvector), or an Elasticsearch index.
              Dropping a collection permanently destroys all stored data.

  index       A search-acceleration structure (e.g. HNSW, IVF_FLAT,
              DISKANN) built on a collection's vector field.  Enables
              fast approximate nearest-neighbor (ANN) queries.  Created
              automatically with the collection.  Dropping an index
              removes only the search structure -- the underlying data
              remains intact and can be re-indexed.
"""


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="collection_admin",
        description="Backend-agnostic vector-DB collection administration.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend", "-b",
        default=None,
        help="Backend name (e.g. milvus, pgvector, elasticsearch). "
             "Required for non-interactive commands.",
    )
    parser.add_argument(
        "--param", "-p",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra connection parameter (repeatable).",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        default=False,
        help="Output results as JSON.",
    )

    sub = parser.add_subparsers(dest="command")

    # -- interactive --
    p_ia = sub.add_parser(
        "interactive",
        help="Menu-driven interactive mode: discover backends, browse "
             "collections, run operations.",
    )
    p_ia.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file (default: auto-detect).",
    )

    # -- list --
    sub.add_parser("list", help="List all collections on the server.")

    # -- info --
    p_info = sub.add_parser("info", help="Show detailed collection metadata.")
    p_info.add_argument("collection", help="Collection name.")

    # -- indexes --
    p_idx = sub.add_parser("indexes", help="List indexes on a collection.")
    p_idx.add_argument("collection", help="Collection name.")

    # -- stats --
    p_stats = sub.add_parser("stats", help="Show collection statistics.")
    p_stats.add_argument("collection", help="Collection name.")

    # -- drop --
    p_drop = sub.add_parser(
        "drop",
        help="Drop a collection -- permanently deletes all data and indexes.",
    )
    p_drop.add_argument("collection", help="Collection name.")
    p_drop.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Skip confirmation prompt.",
    )

    # -- drop-index --
    p_di = sub.add_parser(
        "drop-index",
        help="Drop an index from a collection -- data is kept and can be re-indexed.",
    )
    p_di.add_argument("collection", help="Collection name.")
    p_di.add_argument(
        "--index-name", "-i",
        default=None,
        help="Specific index to drop (default: primary vector index).",
    )
    p_di.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Skip confirmation prompt.",
    )

    return parser


def _parse_params(raw: List[str]) -> Dict[str, str]:
    """Parse ``--param KEY=VALUE`` arguments into a dict."""
    result: Dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            print(f"Invalid --param format (expected KEY=VALUE): {item}",
                  file=sys.stderr)
            sys.exit(1)
        key, _, value = item.partition("=")
        result[key.strip()] = value.strip()
    return result


# =====================================================================
# Main entry point
# =====================================================================

_DISPATCH = {
    "list": _cmd_list,
    "info": _cmd_info,
    "indexes": _cmd_indexes,
    "stats": _cmd_stats,
    "drop": _cmd_drop,
    "drop-index": _cmd_drop_index,
}


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, connect to the backend, and dispatch."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Default to interactive when no subcommand given
    if not args.command:
        args.command = "interactive"

    # ── Interactive mode (no --backend required) ──────────────────
    if args.command == "interactive":
        return _cmd_interactive(args)

    # ── Non-interactive commands require --backend ────────────────
    if not args.backend:
        parser.error("--backend/-b is required for non-interactive commands.")

    extra = _parse_params(args.param)
    backend = _connect_backend(args.backend, extra)

    try:
        handler = _DISPATCH[args.command]
        handler(backend, args)
    except NotImplementedError as exc:
        print(f"Not supported: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.error("Error: %s", exc, exc_info=True)
        return 1
    finally:
        backend.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
