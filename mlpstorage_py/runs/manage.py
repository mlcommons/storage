"""
``mlpstorage runs list|show|rm|purge|gc`` handlers.

Nothing here deletes a run outright. ``rm`` moves leaves to
``<results-dir>/.mlps/trash/<batch>/<original relative path>`` and ``gc``
moves orphaned code-image pool directories the same way; only ``purge``
deletes, and only from the trash. The sentinel, ``systems/`` and the pool
roots are never touched by ``rm``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import sys
from typing import Dict, List, Optional, Tuple

from mlpstorage_py.config import EXIT_CODE
from mlpstorage_py.runs.ledger import (
    BENCHMARK_ALIASES,
    MODES,
    RunRecord,
    STATUSES,
    dir_size,
    iter_leaves,
    mark_trashed,
    read_metadata,
    read_pointer_hash,
    run_status,
    sync,
    trash_root,
)

_POOL_SENTINEL = ".mlps-image-pool"
_HASH_FILENAME = ".code-hash.json"
_POINTER_FILENAME = ".mlps-code-image"
_RESERVED_TOP_LEVEL = set(MODES) | {"systems"}
_ROLLUP_FILES = ("results.json", "results.csv")
_OLDER_THAN_RELATIVE = re.compile(r"^(\d+)\s*([hdw])$")


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def _human_size(num: int) -> str:
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _started_display(record: RunRecord) -> str:
    started = record.started()
    return started.strftime("%Y-%m-%d %H:%M:%S") if started else record.run_datetime


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join("{:<%d}" % w for w in widths)
    print(fmt.format(*headers).rstrip())
    for row in rows:
        print(fmt.format(*row).rstrip())


def _confirm(prompt: str, yes: bool, logger) -> bool:
    if yes:
        return True
    if sys.stdin is not None and sys.stdin.isatty():
        try:
            answer = input(f"{prompt} [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        return answer in ("y", "yes")
    logger.error(
        f"{prompt} — refusing without confirmation because stdin is not a "
        f"terminal. Re-run with --yes to confirm."
    )
    return False


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------

def _pool_roots(results_dir: str) -> List[Tuple[str, str]]:
    """``[(name, path)]`` for every top-level directory carrying a
    ``.mlps-image-pool`` sentinel: the tree-wide ``code-images/`` and any
    per-organization ``<orgname>/`` pool an earlier release wrote."""
    roots = []
    try:
        entries = sorted(os.listdir(results_dir))
    except OSError:
        return roots
    for entry in entries:
        if entry.startswith(".") or entry in _RESERVED_TOP_LEVEL:
            continue
        path = os.path.join(results_dir, entry)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, _POOL_SENTINEL)):
            roots.append((entry, path))
    return roots


def _find_pool_image(results_dir: str, full_hash: str) -> Optional[str]:
    name = f"code-{full_hash[:8]}"
    for _org, root in _pool_roots(results_dir):
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def _new_trash_batch(results_dir: str) -> str:
    batch = os.path.join(trash_root(results_dir), _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(batch, exist_ok=True)
    return batch


def _move_into_trash(results_dir: str, batch: str, rel: str) -> str:
    """Move ``<results-dir>/<rel>`` to ``<batch>/<rel>``; returns the
    results-dir-relative destination."""
    src = os.path.join(results_dir, *rel.split("/"))
    dest = os.path.join(batch, *rel.split("/"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        os.rename(src, dest)
    except OSError:
        shutil.move(src, dest)
    return os.path.relpath(dest, results_dir).replace(os.sep, "/")


def _describe(results_dir: str, record: RunRecord) -> dict:
    leaf_path = record.path(results_dir)
    pointer = read_pointer_hash(leaf_path)
    row = dict(record.info)
    row.update({
        "id": record.id,
        "leaf": record.leaf,
        "status": run_status(leaf_path, record.info),
        "code_image": f"code-{pointer[:8]}" if pointer else None,
        "code_hash": pointer,
        "size_bytes": dir_size(leaf_path),
        "registered_at": record.registered_at,
    })
    return row


def _parse_older_than(text: str) -> _dt.datetime:
    match = _OLDER_THAN_RELATIVE.match(text.strip().lower())
    if match:
        count, unit = int(match.group(1)), match.group(2)
        delta = {"h": _dt.timedelta(hours=count),
                 "d": _dt.timedelta(days=count),
                 "w": _dt.timedelta(weeks=count)}[unit]
        return _dt.datetime.now() - delta
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y%m%d_%H%M%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return _dt.datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(
        f"--older-than {text!r}: use a count with a unit (12h, 7d, 2w) or a "
        f"date (2026-09-01, 20260901, 20260901_120000)."
    )


def _matches_filters(row: dict, args) -> bool:
    mode = getattr(args, "mode_filter", None)
    if mode and row["mode"] != mode:
        return False
    benchmark = getattr(args, "benchmark", None)
    if benchmark and row["benchmark"] != BENCHMARK_ALIASES.get(benchmark, benchmark):
        return False
    model = getattr(args, "model", None)
    if model and row["model"] != model:
        return False
    systemname = getattr(args, "systemname", None)
    if systemname and row["systemname"] != systemname:
        return False
    status = getattr(args, "status", None)
    if status and row["status"] != status:
        return False
    return True


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def _run_tokens(results_dir: str, logger) -> Optional[Dict[str, str]]:
    """Per-run readiness token by leaf (``ok`` / ``failed`` / ``running`` /
    ``invalid`` / ``extra``, ``-`` for whatif), from the same evaluator
    ``mlpstorage status`` prints; ``None`` when the evaluator fails, so the
    listing (what ``rm`` acts on) stays usable (an empty map is a tree
    without run leaves)."""
    from mlpstorage_py.readiness import evaluate, run_tokens
    try:
        return dict(run_tokens(evaluate(results_dir)))
    except Exception as exc:  # noqa: BLE001 -- listing must not depend on the checker
        logger.warning(f"Could not evaluate submission readiness for the SUBMIT column: {exc}")
        return None


def _cmd_list(args, results_dir: str, logger) -> int:
    rows = [_describe(results_dir, r) for r in sync(results_dir)]
    tokens = _run_tokens(results_dir, logger)
    for row in rows:
        row["submit"] = tokens.get(row["leaf"]) if tokens is not None else None
    rows = [row for row in rows if _matches_filters(row, args)]
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2))
        return EXIT_CODE.SUCCESS
    if not rows:
        print(f"No runs in {results_dir}.")
        return EXIT_CODE.SUCCESS
    headers = ["ID", "STATUS", "SUBMIT", "MODE", "SYSTEM", "BENCHMARK", "MODEL", "COMMAND", "STARTED", "CODE", "SIZE"]
    unknown = "?" if tokens is None else "-"
    table = []
    for row in rows:
        table.append([
            str(row["id"]), row["status"], row["submit"] or unknown, row["mode"], row["systemname"],
            row["benchmark"], row["model"], row["command"],
            _started_display(RunRecord(id=row["id"], leaf=row["leaf"], registered_at="")),
            row["code_image"] or "-", _human_size(row["size_bytes"]),
        ])
    _print_table(headers, table)
    return EXIT_CODE.SUCCESS


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

def _cmd_show(args, results_dir: str, logger) -> int:
    records = {r.id: r for r in sync(results_dir)}
    record = records.get(args.run_id)
    if record is None:
        logger.error(f"No run with ID {args.run_id} in {results_dir} (see `mlpstorage runs list`).")
        return EXIT_CODE.INVALID_ARGUMENTS
    row = _describe(results_dir, record)
    leaf_path = record.path(results_dir)

    print(f"Run {record.id}: {record.leaf}")
    print(f"  path:        {leaf_path}")
    print(f"  status:      {row['status']}")
    print(f"  mode:        {row['mode']}")
    print(f"  orgname:     {row['orgname']}")
    print(f"  systemname:  {row['systemname']}")
    print(f"  benchmark:   {row['benchmark']}")
    print(f"  model:       {row['model']}")
    print(f"  command:     {row['command']}")
    print(f"  started:     {_started_display(record)}")
    print(f"  size:        {_human_size(row['size_bytes'])}")
    print(f"  registered:  {record.registered_at or '-'}")

    if row["code_hash"]:
        image = _find_pool_image(results_dir, row["code_hash"])
        where = image if image else "MISSING — no pool image with this hash in the tree"
        print(f"  code image:  {row['code_image']} ({row['code_hash']}) → {where}")
    else:
        print("  code image:  none (no .mlps-code-image pointer in the leaf)")

    from mlpstorage_py.provenance import (
        PROVENANCE_FILENAME, ProvenanceError, UNKNOWN, read_leaf_provenance,
    )
    try:
        stamp = read_leaf_provenance(leaf_path, results_dir)
    except ProvenanceError as e:
        print(f"  provenance:  MALFORMED — {e}")
    else:
        if os.path.isfile(os.path.join(leaf_path, PROVENANCE_FILENAME)):
            print(f"  provenance:  {PROVENANCE_FILENAME}")
        else:
            print("  provenance:  derived (leaf predates provenance stamping; nothing written)")
        sha = stamp.tool["git_sha"]
        commit = stamp.dlio["commit"]
        lib = stamp.storage_library
        lib_text = lib["name"] if lib["name"] == "none" else f"{lib['name']} {lib.get('version', UNKNOWN)}"
        print(f"    rules edition: {stamp.rules_edition}   layout: {stamp.layout_version}")
        print(f"    tool:          mlpstorage {stamp.tool['version']} "
              f"({sha if sha == UNKNOWN else sha[:8]})")
        print(f"    dlio:          {stamp.dlio['version']} @ "
              f"{commit if commit == UNKNOWN else commit[:8]}")
        print(f"    storage lib:   {lib_text}")
        print(f"    core config:   {stamp.core_config['hash']} ({stamp.core_config['allowlist']})")
        from mlpstorage_py.editions import describe_class
        metadata = read_metadata(leaf_path) or {}
        accelerator = metadata.get("accelerator")
        print(f"    class:         {describe_class(stamp, division=row['mode'], family=row['benchmark'], model=row['model'], accelerator=accelerator if isinstance(accelerator, str) and accelerator else None)}")

    metadata = read_metadata(leaf_path)
    if metadata is None:
        print("  metadata:    none (no *_metadata.json — still running, or killed before it was written)")
    else:
        print("  metadata:")
        for key in ("exit_status", "executed_command", "runtime", "num_processes",
                    "accelerator", "invocation_start_time", "invocation_end_time", "verification"):
            if key in metadata:
                print(f"    {key}: {metadata[key]}")

    print("  files:")
    entries = []
    for dirpath, _dirnames, filenames in os.walk(leaf_path):
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, leaf_path)
            try:
                size = os.lstat(full).st_size
            except OSError:
                size = 0
            entries.append((rel, size))
    for rel, size in sorted(entries):
        print(f"    {_human_size(size):>10}  {rel}")
    return EXIT_CODE.SUCCESS


# ---------------------------------------------------------------------------
# rm
# ---------------------------------------------------------------------------

def _rollups_above(results_dir: str, record: RunRecord) -> List[str]:
    """reportgen rollup files sitting on the leaf's ancestors, up to the
    ``results/<systemname>`` directory."""
    parts = record.leaf.split("/")
    stop = 4  # index of <systemname>
    found = []
    for depth in range(len(parts) - 1, stop, -1):
        parent = os.path.join(results_dir, *parts[:depth])
        for name in _ROLLUP_FILES:
            path = os.path.join(parent, name)
            if os.path.isfile(path):
                found.append(path)
    return found


def _cmd_rm(args, results_dir: str, logger) -> int:
    present = {r.id: r for r in sync(results_dir)}
    ids = list(getattr(args, "run_ids", None) or [])
    status = getattr(args, "status", None)
    older_than = getattr(args, "older_than", None)
    keep_last = getattr(args, "keep_last", None)

    if not ids and not status and not older_than:
        logger.error(
            "runs rm: nothing selected. Give run IDs (see `mlpstorage runs list`) "
            "and/or narrow with --status / --older-than."
        )
        return EXIT_CODE.INVALID_ARGUMENTS

    if ids:
        unknown = [i for i in ids if i not in present]
        if unknown:
            logger.error(
                f"runs rm: no run with ID {', '.join(str(i) for i in unknown)} "
                f"in {results_dir}; nothing was moved."
            )
            return EXIT_CODE.INVALID_ARGUMENTS
        selected = [present[i] for i in sorted(set(ids))]
    else:
        selected = list(present.values())

    rows = {r.id: _describe(results_dir, r) for r in selected}
    if status:
        selected = [r for r in selected if rows[r.id]["status"] == status]
    if older_than:
        try:
            cutoff = _parse_older_than(older_than)
        except ValueError as exc:
            logger.error(str(exc))
            return EXIT_CODE.INVALID_ARGUMENTS
        selected = [r for r in selected if r.started() is not None and r.started() < cutoff]
    if keep_last:
        newest_first = sorted(selected, key=lambda r: (r.run_datetime, r.id), reverse=True)
        selected = sorted(newest_first[keep_last:], key=lambda r: r.id)

    if not selected:
        print("Nothing to remove: no run matches the selection.")
        return EXIT_CODE.SUCCESS

    total = sum(rows[r.id]["size_bytes"] for r in selected)
    print(f"Runs to move to {trash_root(results_dir)}:")
    for r in selected:
        print(f"  {r.id:>4}  {rows[r.id]['status']:<10}  {_human_size(rows[r.id]['size_bytes']):>10}  {r.leaf}")
    print(f"  {len(selected)} run(s), {_human_size(total)}")

    if not _confirm(f"Move {len(selected)} run(s) to the trash?", getattr(args, "yes", False), logger):
        return EXIT_CODE.INVALID_ARGUMENTS

    batch = _new_trash_batch(results_dir)
    stale_rollups = []
    for r in selected:
        for path in _rollups_above(results_dir, r):
            if path not in stale_rollups:
                stale_rollups.append(path)
        dest = _move_into_trash(results_dir, batch, r.leaf)
        mark_trashed(results_dir, r.id, dest)
        logger.verbose(f"run {r.id}: {r.leaf} → {dest}")

    print(
        f"Moved {len(selected)} run(s) to {batch}. `mlpstorage runs purge` "
        f"deletes the trash for good; move a leaf back by hand to restore it."
    )
    for path in stale_rollups:
        logger.warning(
            f"{path} was generated before this removal and still counts the "
            f"removed run(s); rerun `mlpstorage reports reportgen` to refresh it."
        )
    return EXIT_CODE.SUCCESS


# ---------------------------------------------------------------------------
# purge
# ---------------------------------------------------------------------------

def _cmd_purge(args, results_dir: str, logger) -> int:
    root = trash_root(results_dir)
    try:
        batches = sorted(e for e in os.listdir(root) if os.path.isdir(os.path.join(root, e)))
    except FileNotFoundError:
        batches = []
    if not batches:
        print(f"Trash is empty ({root}).")
        return EXIT_CODE.SUCCESS

    sizes = {b: dir_size(os.path.join(root, b)) for b in batches}
    print(f"Trash batches under {root}:")
    for b in batches:
        print(f"  {b}  {_human_size(sizes[b]):>10}")
    print(f"  {len(batches)} batch(es), {_human_size(sum(sizes.values()))}")

    if not _confirm("Delete everything in the trash permanently?", getattr(args, "yes", False), logger):
        return EXIT_CODE.INVALID_ARGUMENTS

    for b in batches:
        shutil.rmtree(os.path.join(root, b))
    print(f"Deleted {len(batches)} batch(es), {_human_size(sum(sizes.values()))}.")
    return EXIT_CODE.SUCCESS


# ---------------------------------------------------------------------------
# gc
# ---------------------------------------------------------------------------

def _referenced_hashes(results_dir: str) -> set:
    referenced = set()
    for rel in iter_leaves(results_dir):
        full_hash = read_pointer_hash(os.path.join(results_dir, *rel.split("/")))
        if full_hash:
            referenced.add(full_hash)
    # Trashed leaves keep their images alive: restoring a leaf must not
    # leave it with a dangling pointer. Purge first, then gc.
    for dirpath, _dirnames, filenames in os.walk(trash_root(results_dir)):
        if _POINTER_FILENAME in filenames:
            full_hash = read_pointer_hash(dirpath)
            if full_hash:
                referenced.add(full_hash)
    return referenced


def _cmd_gc(args, results_dir: str, logger) -> int:
    referenced = _referenced_hashes(results_dir)
    orphans: List[Tuple[str, str, int]] = []  # (root name, image dir name, size)
    for root_name, root in _pool_roots(results_dir):
        for entry in sorted(os.listdir(root)):
            image = os.path.join(root, entry)
            if not entry.startswith("code-") or not os.path.isdir(image):
                continue
            try:
                with open(os.path.join(image, _HASH_FILENAME), "r", encoding="utf-8") as fh:
                    stored = json.load(fh).get("hash")
            except (OSError, ValueError, AttributeError):
                logger.warning(
                    f"{image}: no readable {_HASH_FILENAME}; leaving it alone "
                    f"(`mlpstorage validate` reports it as CHECK-02)."
                )
                continue
            if stored not in referenced:
                orphans.append((root_name, entry, dir_size(image)))

    if not orphans:
        print("Nothing to collect: every code-image pool directory is referenced by a run.")
        return EXIT_CODE.SUCCESS

    print(f"Orphaned code images (no run leaf, in the tree or its trash, points at them):")
    for root_name, entry, size in orphans:
        print(f"  {root_name}/{entry}  {_human_size(size):>10}")
    print(f"  {len(orphans)} image(s), {_human_size(sum(s for _, _, s in orphans))}")

    if not _confirm(f"Move {len(orphans)} orphaned image(s) to the trash?", getattr(args, "yes", False), logger):
        return EXIT_CODE.INVALID_ARGUMENTS

    batch = _new_trash_batch(results_dir)
    for root_name, entry, _size in orphans:
        dest = _move_into_trash(results_dir, batch, f"{root_name}/{entry}")
        logger.verbose(f"{root_name}/{entry} → {dest}")
    print(f"Moved {len(orphans)} image(s) to {batch}. `mlpstorage runs purge` deletes them for good.")
    return EXIT_CODE.SUCCESS


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

_HANDLERS = {
    "list": _cmd_list,
    "show": _cmd_show,
    "rm": _cmd_rm,
    "purge": _cmd_purge,
    "gc": _cmd_gc,
}


def run_runs_command(args, results_dir: str, logger) -> int:
    """Entry point from ``main``: ``args.command`` picks the handler."""
    handler = _HANDLERS.get(getattr(args, "command", None))
    if handler is None:
        logger.error(f"runs: unknown subcommand {getattr(args, 'command', None)!r}")
        return EXIT_CODE.INVALID_ARGUMENTS
    return handler(args, results_dir, logger)
