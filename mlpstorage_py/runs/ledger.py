"""
Run ledger: stable IDs for the run leaves of a results-dir.

Layout
------
``<results-dir>/.mlps/runs.jsonl`` holds one JSON object per line, append
only. Two event kinds exist::

    {"event": "registered", "id": 7, "leaf": "closed/Acme/results/sys/training/unet3d/run/20260901_100000", "at": "..."}
    {"event": "trashed",    "id": 7, "to": ".mlps/trash/20260910_120000/closed/Acme/...", "at": "..."}

IDs are assigned once and never reused: a leaf removed by hand simply
stops being listed, and its number stays retired. The tree itself is the
source of truth for *existence* — every ``runs`` command first walks the
tree, adopts leaves the ledger has not seen (older trees, or runs whose
registration failed) and hides ledger entries whose leaf is gone.

The ``Benchmark`` base class calls :func:`register_run` right after it
reserves the leaf, so an in-flight run already has its ID while it runs.

Only canonical leaves are tracked, i.e. paths of the Rules.md §2.1 shape
``<mode>/<orgname>/results/<systemname>/<benchmark>/.../<YYYYMMDD_HHMMSS>``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

LEDGER_RELPATH = os.path.join(".mlps", "runs.jsonl")
TRASH_RELPATH = os.path.join(".mlps", "trash")

MODES = ("closed", "open", "whatif")

#: Directory names of the benchmark tails under ``results/<systemname>/``.
BENCHMARK_DIRS = ("training", "checkpointing", "vector_database", "kv_cache")

#: User-facing benchmark names (the CLI positionals) → directory names.
BENCHMARK_ALIASES = {
    "training": "training",
    "checkpointing": "checkpointing",
    "vectordb": "vector_database",
    "vector_database": "vector_database",
    "kvcache": "kv_cache",
    "kv_cache": "kv_cache",
}

STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_INCOMPLETE = "incomplete"
STATUSES = (STATUS_COMPLETE, STATUS_FAILED, STATUS_INCOMPLETE)

_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
_METADATA_SUFFIX = "_metadata.json"
_SUMMARY_FILENAME = "summary.json"
_POINTER_FILENAME = ".mlps-code-image"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def ledger_path(results_dir: str) -> str:
    """``<results-dir>/.mlps/runs.jsonl``."""
    return os.path.join(results_dir, LEDGER_RELPATH)


def trash_root(results_dir: str) -> str:
    """``<results-dir>/.mlps/trash``."""
    return os.path.join(results_dir, TRASH_RELPATH)


def _now() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Leaf paths
# ---------------------------------------------------------------------------

def parse_leaf(rel: str) -> Optional[dict]:
    """Decode a results-dir-relative leaf path into its identity fields.

    Returns ``None`` for anything that is not a canonical run leaf (pool
    images, trash, ``systems/``, files inside a leaf, ...).

    Shapes recognised (``<ts>`` = ``YYYYMMDD_HHMMSS``)::

        <mode>/<org>/results/<system>/training/<model>/<command>/<ts>
        <mode>/<org>/results/<system>/checkpointing/<model>/<ts>
        <mode>/<org>/results/<system>/vector_database/<engine>/<index>/<command>/<ts>
        <mode>/<org>/results/<system>/kv_cache/<model>/<command>/<ts>
    """
    parts = rel.replace(os.sep, "/").strip("/").split("/")
    if len(parts) < 7 or parts[0] not in MODES or parts[2] != "results":
        return None
    if not _TIMESTAMP_RE.match(parts[-1]):
        return None
    mode, orgname, _results, systemname, benchmark = parts[:5]
    tail = parts[5:-1]
    if benchmark == "training" or benchmark == "kv_cache":
        if len(tail) != 2:
            return None
        model, command = tail
    elif benchmark == "checkpointing":
        if len(tail) != 1:
            return None
        model, command = tail[0], "run"
    elif benchmark == "vector_database":
        if len(tail) != 3:
            return None
        model, command = f"{tail[0]}/{tail[1]}", tail[2]
    else:
        return None
    return {
        "mode": mode,
        "orgname": orgname,
        "systemname": systemname,
        "benchmark": benchmark,
        "model": model,
        "command": command,
        "run_datetime": parts[-1],
    }


def iter_leaves(results_dir: str) -> List[str]:
    """Every canonical run leaf under ``results_dir``, as sorted
    results-dir-relative POSIX paths. ``.mlps/`` (trash, ledger), pool
    images and anything outside ``<mode>/<org>/results/`` are skipped."""
    found: List[str] = []
    for mode in MODES:
        mode_dir = os.path.join(results_dir, mode)
        if not os.path.isdir(mode_dir):
            continue
        for org in sorted(os.listdir(mode_dir)):
            results_root = os.path.join(mode_dir, org, "results")
            if not os.path.isdir(results_root):
                continue
            for dirpath, dirnames, _files in os.walk(results_root):
                prune = []
                for d in dirnames:
                    if _TIMESTAMP_RE.match(d):
                        rel = os.path.relpath(os.path.join(dirpath, d), results_dir)
                        rel = rel.replace(os.sep, "/")
                        if parse_leaf(rel) is not None:
                            found.append(rel)
                        prune.append(d)
                    elif d.startswith("."):
                        prune.append(d)
                for d in prune:
                    dirnames.remove(d)
    return sorted(found)


def _relative_leaf(results_dir: str, leaf_path: str) -> Optional[str]:
    """``leaf_path`` relative to ``results_dir`` when it is a canonical leaf
    inside the tree, else ``None``."""
    root = os.path.realpath(results_dir)
    leaf = os.path.realpath(leaf_path)
    try:
        rel = os.path.relpath(leaf, root)
    except ValueError:  # different drives on Windows
        return None
    if rel.startswith(".."):
        return None
    rel = rel.replace(os.sep, "/")
    return rel if parse_leaf(rel) is not None else None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    id: int
    leaf: str
    registered_at: str
    trashed_at: Optional[str] = None
    trashed_to: Optional[str] = None
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.info:
            self.info = parse_leaf(self.leaf) or {}

    def path(self, results_dir: str) -> str:
        return os.path.join(results_dir, *self.leaf.split("/"))

    @property
    def run_datetime(self) -> str:
        return self.info.get("run_datetime", "")

    def started(self) -> Optional[_dt.datetime]:
        try:
            return _dt.datetime.strptime(self.run_datetime, "%Y%m%d_%H%M%S")
        except ValueError:
            return None


def _iter_events(path: str) -> Iterable[dict]:
    try:
        fh = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        return
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn line from a crash mid-append; skip it
            if isinstance(event, dict):
                yield event


def read_ledger(results_dir: str) -> Dict[int, RunRecord]:
    """Fold the event log into ``{id: RunRecord}``."""
    records: Dict[int, RunRecord] = {}
    for event in _iter_events(ledger_path(results_dir)):
        kind = event.get("event")
        try:
            run_id = int(event.get("id"))
        except (TypeError, ValueError):
            continue
        if kind == "registered" and isinstance(event.get("leaf"), str):
            records[run_id] = RunRecord(
                id=run_id, leaf=event["leaf"], registered_at=str(event.get("at", "")),
            )
        elif kind == "trashed" and run_id in records:
            records[run_id].trashed_at = str(event.get("at", ""))
            records[run_id].trashed_to = event.get("to")
    return records


def _append_events(results_dir: str, events: List[dict]) -> None:
    path = ledger_path(results_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, sort_keys=True) + "\n")
        fh.flush()


class _Locked:
    """Advisory lock on the ledger so two runs starting in the same
    instant cannot both take the same next ID."""

    def __init__(self, results_dir: str):
        self._lock_path = ledger_path(results_dir) + ".lock"
        self._fh = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self._lock_path), exist_ok=True)
        self._fh = open(self._lock_path, "a+")
        try:
            import fcntl
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        return self

    def __exit__(self, *exc):
        try:
            import fcntl
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        self._fh.close()
        return False


def _next_id(records: Dict[int, RunRecord]) -> int:
    return (max(records) + 1) if records else 1


def register_run(results_dir: str, leaf_path: str) -> Optional[int]:
    """Give ``leaf_path`` an ID (idempotent) and return it.

    Returns ``None`` — and writes nothing — when the leaf is not a
    canonical run leaf inside ``results_dir``.
    """
    rel = _relative_leaf(results_dir, leaf_path)
    if rel is None:
        return None
    with _Locked(results_dir):
        records = read_ledger(results_dir)
        for record in records.values():
            if record.leaf == rel and record.trashed_at is None:
                return record.id
        run_id = _next_id(records)
        _append_events(results_dir, [
            {"event": "registered", "id": run_id, "leaf": rel, "at": _now()},
        ])
    return run_id


def sync(results_dir: str) -> List[RunRecord]:
    """Reconcile the ledger with the tree and return the runs that exist.

    Leaves the ledger has never seen are registered (in timestamp order,
    so a tree that predates the ledger gets chronological IDs); ledger
    entries whose leaf is gone are hidden. A leaf that reappears after a
    ``trashed`` event (someone moved it back by hand) is listed under its
    old ID.
    """
    on_disk = set(iter_leaves(results_dir))
    with _Locked(results_dir):
        records = read_ledger(results_dir)
        known = {}
        for record in records.values():
            # the newest registration of a leaf wins (re-adoption after a
            # hand restore of a trashed leaf registers it again)
            known[record.leaf] = record
        unknown = sorted(on_disk - set(known), key=lambda rel: (parse_leaf(rel)["run_datetime"], rel))
        if unknown:
            next_id = _next_id(records)
            events = []
            for rel in unknown:
                events.append({"event": "registered", "id": next_id, "leaf": rel, "at": _now()})
                records[next_id] = RunRecord(id=next_id, leaf=rel, registered_at=events[-1]["at"])
                known[rel] = records[next_id]
                next_id += 1
            _append_events(results_dir, events)
    present = [known[rel] for rel in on_disk]
    return sorted(present, key=lambda r: r.id)


def mark_trashed(results_dir: str, run_id: int, dest_rel: str) -> None:
    """Record that run ``run_id`` was moved to ``dest_rel`` (results-dir
    relative)."""
    _append_events(results_dir, [
        {"event": "trashed", "id": run_id, "to": dest_rel.replace(os.sep, "/"), "at": _now()},
    ])


# ---------------------------------------------------------------------------
# Leaf inspection
# ---------------------------------------------------------------------------

def metadata_file(leaf_path: str) -> Optional[str]:
    try:
        names = sorted(os.listdir(leaf_path))
    except OSError:
        return None
    for name in names:
        if name.endswith(_METADATA_SUFFIX):
            return os.path.join(leaf_path, name)
    return None


def read_metadata(leaf_path: str) -> Optional[dict]:
    path = metadata_file(leaf_path)
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def run_status(leaf_path: str, info: Optional[dict]) -> str:
    """``complete`` / ``failed`` / ``incomplete`` for one leaf.

    - no readable ``*_metadata.json`` → ``incomplete`` (still running, or
      killed before the ``finally`` that writes metadata);
    - metadata with an ``exit_status`` (every run since this ledger
      landed) → ``complete`` when it is 0, else ``failed``;
    - older metadata: run-phase leaves need DLIO's ``summary.json``;
      training/checkpointing ``datagen``/``datasize`` never produce one and
      count as complete.
    """
    metadata = read_metadata(leaf_path)
    if metadata is None:
        return STATUS_INCOMPLETE
    exit_status = metadata.get("exit_status")
    if isinstance(exit_status, int) and not isinstance(exit_status, bool):
        return STATUS_COMPLETE if exit_status == 0 else STATUS_FAILED
    info = info or {}
    if info.get("benchmark") in ("training", "checkpointing") \
            and info.get("command") in ("datagen", "datasize"):
        return STATUS_COMPLETE
    if os.path.isfile(os.path.join(leaf_path, _SUMMARY_FILENAME)):
        return STATUS_COMPLETE
    return STATUS_FAILED


def read_pointer_hash(leaf_path: str) -> Optional[str]:
    """The 32-hex md5-tree-v2 hash from the leaf's ``.mlps-code-image``,
    or ``None`` when absent or malformed."""
    try:
        with open(os.path.join(leaf_path, _POINTER_FILENAME), "r", encoding="utf-8") as fh:
            line = fh.read().strip()
    except OSError:
        return None
    _alg, _sep, hex_part = line.partition(":")
    if _alg != "md5-tree-v2" or not re.fullmatch(r"[0-9a-f]{32}", hex_part):
        return None
    return hex_part


def dir_size(path: str) -> int:
    """Bytes held by every regular file under ``path``."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total
