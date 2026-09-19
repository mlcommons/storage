"""Rules editions table and declared comparability classes.

Reads ``mlpstorage_py/rules/editions.yaml`` (schema ``mlps-rules-editions/1``),
the machine-readable companion of Rules.md "Rules editions and comparability
classes". The file is data the working group edits by pull request; this
module validates it, answers "which edition is this" and "which class does
this run belong to", and never writes anything.

A **rules edition** (``"3.0"``) is the version of Rules.md a result was
produced under -- the comparability key across rounds, stamped into every
run leaf's ``provenance.json`` (``config.RULES_EDITION``) and every
``submission.yaml``. It is not the tool version.

A **comparability class** is the WG's assertion that runs of one
(family, model, emulated accelerator) whose ``core-config-v1`` hash is one of
the class's ``core_configs`` executed the same workload in every edition the
class lists. Rows in one class compare; nothing else does. Classification is
a pure function of (stamp, table) and is never cached into evidence files.

Design and survey: .planning/rules-editions-and-comparability-classes.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from mlpstorage_py.config import RULES_EDITION
from mlpstorage_py.provenance import UNKNOWN, load_allowlists

EDITIONS_FILE = Path(__file__).resolve().parent / "rules" / "editions.yaml"
EDITIONS_SCHEMA = "mlps-rules-editions/1"
ANY_ACCELERATOR = "any"
CLASS_FAMILIES = ("training", "checkpointing")
EDITION_STATUSES = ("historical", "current")
_ACCELERATORS = ("b200", "mi355", "h100", "a100", ANY_ACCELERATOR)
_CLASS_ID_RE = re.compile(r"[a-z0-9.]+(-[a-z0-9]+)*-[A-Z]")
_HASH_RE = re.compile(r"[0-9a-f]{16}")
_SHA_RE = re.compile(r"[0-9a-f]{40}")


class EditionsError(ValueError):
    """The editions table is missing, unreadable or inconsistent."""


@dataclass(frozen=True)
class Edition:
    id: str
    status: str
    results_repo: str
    tool: str
    layout_versions: List[int]
    workloads: Dict[str, Dict[str, Dict[str, List[str]]]]
    dlio_revisions: List[Dict[str, str]]
    storage_libraries: Dict[str, List[str]]
    notes: str = ""

    @property
    def dlio_commits(self) -> frozenset:
        return frozenset(r["commit"] for r in self.dlio_revisions)


@dataclass(frozen=True)
class ComparabilityClass:
    id: str
    family: str
    model: str
    accelerator: str
    allowlist: str
    core_configs: Tuple[str, ...]
    editions: Tuple[str, ...]
    reference: Optional[str] = None
    notes: str = ""

    def matches(self, *, family: str, model: str, accelerator: Optional[str],
                core_config: str, edition: Optional[str] = None) -> bool:
        if self.family != family or self.model != model:
            return False
        if self.accelerator != ANY_ACCELERATOR and self.accelerator != accelerator:
            return False
        if core_config not in self.core_configs:
            return False
        if edition is not None and edition not in self.editions:
            return False
        return True


@dataclass(frozen=True)
class EditionsTable:
    schema: str
    current_edition: str
    editions: Dict[str, Edition]
    classes: List[ComparabilityClass]
    path: Path = field(default=EDITIONS_FILE, compare=False)

    def edition(self, edition_id: Any) -> Optional[Edition]:
        return self.editions.get(str(edition_id)) if edition_id is not None else None

    def classes_for(self, edition_id: str) -> List[ComparabilityClass]:
        return [c for c in self.classes if edition_id in c.editions]

    def classify(self, *, family: str, model: str, accelerator: Optional[str],
                 core_config: Any, edition: Optional[str] = None) -> Optional[ComparabilityClass]:
        """The class a run belongs to, or ``None``.

        ``edition=None`` matches across every edition (used for derived stamps
        that carry no edition). ``UNKNOWN`` in ``core_config`` or ``edition``
        never classifies.
        """
        if not isinstance(core_config, str) or core_config == UNKNOWN:
            return None
        if edition is not None and (edition == UNKNOWN or edition not in self.editions):
            return None
        for c in self.classes:
            if c.matches(family=family, model=model, accelerator=accelerator,
                         core_config=core_config, edition=edition):
                return c
        return None

    def classify_stamp(self, stamp, *, family: str, model: str,
                       accelerator: Optional[str]) -> Optional[ComparabilityClass]:
        """Classify a ``RunProvenance`` under its own stamped edition."""
        edition = getattr(stamp, "rules_edition", UNKNOWN)
        if edition == UNKNOWN:
            return None
        return self.classify(family=family, model=model, accelerator=accelerator,
                             core_config=(stamp.core_config or {}).get("hash", UNKNOWN),
                             edition=edition)

    def accepts_dlio(self, edition_id: Any, commit: Any) -> bool:
        e = self.edition(edition_id)
        if e is None or not isinstance(commit, str) or commit == UNKNOWN:
            return False
        return commit in e.dlio_commits


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise EditionsError(msg)


def _parse_edition(eid: Any, raw: Any, where: str) -> Edition:
    _require(isinstance(raw, dict), f"{where}: edition {eid!r} must be a mapping")
    eid = str(eid)
    status = raw.get("status")
    _require(status in EDITION_STATUSES, f"{where}: edition {eid}: status must be one of {EDITION_STATUSES}")
    _require(isinstance(raw.get("results_repo"), str) and raw["results_repo"],
             f"{where}: edition {eid}: results_repo is required")
    layout = raw.get("layout_versions") or []
    _require(isinstance(layout, list) and all(isinstance(v, int) for v in layout),
             f"{where}: edition {eid}: layout_versions must be a list of integers")
    workloads = raw.get("workloads") or {}
    _require(isinstance(workloads, dict), f"{where}: edition {eid}: workloads must be a mapping")
    for division, families in workloads.items():
        _require(isinstance(families, dict), f"{where}: edition {eid}: workloads.{division} must be a mapping")
        for fam, models in families.items():
            _require(isinstance(models, dict), f"{where}: edition {eid}: workloads.{division}.{fam} must be a mapping")
            for model, accels in models.items():
                _require(isinstance(accels, list) and all(isinstance(a, str) for a in accels),
                         f"{where}: edition {eid}: workloads.{division}.{fam}.{model} must list accelerators")
    revs = raw.get("dlio_revisions") or []
    _require(isinstance(revs, list), f"{where}: edition {eid}: dlio_revisions must be a list")
    for r in revs:
        _require(isinstance(r, dict) and isinstance(r.get("commit"), str) and _SHA_RE.fullmatch(r["commit"]),
                 f"{where}: edition {eid}: every dlio_revisions entry needs a 40-hex commit")
        _require(isinstance(r.get("source"), str) and r["source"].startswith("https://"),
                 f"{where}: edition {eid}: dlio revision {r.get('commit')} needs an https source")
    libs = raw.get("storage_libraries") or {}
    _require(isinstance(libs, dict) and all(isinstance(v, list) for v in libs.values()),
             f"{where}: edition {eid}: storage_libraries must map name -> list of versions")
    return Edition(
        id=eid, status=status, results_repo=raw["results_repo"], tool=str(raw.get("tool", UNKNOWN)),
        layout_versions=list(layout), workloads=workloads,
        dlio_revisions=[{"commit": r["commit"], "source": r["source"], "version": str(r.get("version", UNKNOWN))}
                        for r in revs],
        storage_libraries={str(k): [str(v) for v in vs] for k, vs in libs.items()},
        notes=str(raw.get("notes", "") or ""),
    )


def _parse_class(raw: Any, editions: Dict[str, Edition], allowlists: Dict[str, Any],
                 where: str) -> ComparabilityClass:
    _require(isinstance(raw, dict), f"{where}: every class must be a mapping")
    cid = raw.get("id")
    _require(isinstance(cid, str) and _CLASS_ID_RE.fullmatch(cid),
             f"{where}: class id {cid!r} must look like <model>-<accelerator>-<Letter>")
    _require(raw.get("family") in CLASS_FAMILIES, f"{where}: class {cid}: family must be one of {CLASS_FAMILIES}")
    _require(isinstance(raw.get("model"), str) and raw["model"], f"{where}: class {cid}: model is required")
    _require(raw.get("accelerator") in _ACCELERATORS,
             f"{where}: class {cid}: accelerator must be one of {_ACCELERATORS}")
    _require(raw.get("allowlist") in allowlists,
             f"{where}: class {cid}: allowlist {raw.get('allowlist')!r} is not in core_config_keys.yaml")
    hashes = raw.get("core_configs")
    _require(isinstance(hashes, list) and hashes and all(isinstance(h, str) and _HASH_RE.fullmatch(h) for h in hashes),
             f"{where}: class {cid}: core_configs must be a non-empty list of 16-hex hashes")
    _require(len(set(hashes)) == len(hashes), f"{where}: class {cid}: duplicate core_configs entry")
    eds = raw.get("editions")
    _require(isinstance(eds, list) and eds and all(isinstance(e, (str, float, int)) for e in eds),
             f"{where}: class {cid}: editions must be a non-empty list")
    eds = [str(e) for e in eds]
    for e in eds:
        _require(e in editions, f"{where}: class {cid}: edition {e!r} is not in the editions table")
    ref = raw.get("reference")
    _require(ref is None or (isinstance(ref, str) and ref), f"{where}: class {cid}: reference must be a path")
    return ComparabilityClass(
        id=cid, family=raw["family"], model=raw["model"], accelerator=raw["accelerator"],
        allowlist=raw["allowlist"], core_configs=tuple(hashes), editions=tuple(eds),
        reference=ref, notes=str(raw.get("notes", "") or ""),
    )


def _load(path: Path) -> EditionsTable:
    where = str(path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as e:
        raise EditionsError(f"{where}: cannot read editions table: {e}") from e
    _require(isinstance(data, dict), f"{where}: editions table must be a mapping")
    _require(data.get("schema") == EDITIONS_SCHEMA,
             f"{where}: unsupported editions schema {data.get('schema')!r} (expected {EDITIONS_SCHEMA!r})")
    raw_editions = data.get("editions")
    _require(isinstance(raw_editions, dict) and raw_editions, f"{where}: editions must be a non-empty mapping")
    editions = {str(k): _parse_edition(k, v, where) for k, v in raw_editions.items()}
    current = data.get("current_edition")
    _require(current is not None and str(current) in editions,
             f"{where}: current_edition {current!r} is not in the editions table")
    current = str(current)
    _require(editions[current].status == "current", f"{where}: current_edition {current} must have status current")
    raw_classes = data.get("classes")
    _require(isinstance(raw_classes, list), f"{where}: classes must be a list")
    allowlists = load_allowlists().get("allowlists", {})
    classes = [_parse_class(c, editions, allowlists, where) for c in raw_classes]
    ids = [c.id for c in classes]
    _require(len(set(ids)) == len(ids), f"{where}: duplicate class id(s): {sorted({i for i in ids if ids.count(i) > 1})}")
    _require(ids == sorted(ids), f"{where}: classes must be sorted by id")
    claimed: Dict[Tuple[str, str, str, str, str], str] = {}
    for c in classes:
        for e in c.editions:
            for h in c.core_configs:
                key = (c.family, c.model, c.accelerator, e, h)
                _require(key not in claimed, f"{where}: classes {claimed.get(key)} and {c.id} both claim {key}")
                claimed[key] = c.id
    return EditionsTable(schema=EDITIONS_SCHEMA, current_edition=current, editions=editions,
                         classes=classes, path=path)


_CACHE: Dict[str, EditionsTable] = {}


def load_editions(path=None) -> EditionsTable:
    """Load and validate the editions table (the shipped one by default, cached)."""
    p = Path(path) if path is not None else EDITIONS_FILE
    key = str(p)
    if path is None and key in _CACHE:
        return _CACHE[key]
    table = _load(p)
    if path is None:
        _CACHE[key] = table
    return table


def describe_class(stamp, *, family: str, model: str, accelerator: Optional[str]) -> str:
    """One line for ``mlpstorage runs show``: the class id, or why there is none."""
    try:
        table = load_editions()
    except EditionsError as e:
        return f"unavailable ({e})"
    core = (getattr(stamp, "core_config", None) or {}).get("hash", UNKNOWN)
    if not isinstance(core, str) or core == UNKNOWN:
        return "n/a (no core-config hash for this family)"
    edition = getattr(stamp, "rules_edition", UNKNOWN)
    if edition == UNKNOWN:
        c = table.classify(family=family, model=model, accelerator=accelerator, core_config=core)
        if c is None:
            return "unclassified (matched by hash across every edition; stamp has no edition)"
        return f"{c.id} (matched by hash across every edition; stamp has no edition)"
    if edition not in table.editions:
        return f"unclassified (rules edition {edition} is not in {EDITIONS_FILE.name})"
    c = table.classify(family=family, model=model, accelerator=accelerator, core_config=core, edition=edition)
    return c.id if c is not None else f"unclassified (no class for edition {edition} {family}/{model}/{accelerator or UNKNOWN} {core})"


def _self_check_current_edition() -> None:
    """``current_edition`` must equal the tool constant (guarded at import so the
    two cannot drift silently; the unit test asserts the same)."""
    try:
        table = load_editions()
    except EditionsError:
        return
    if table.current_edition != RULES_EDITION:
        raise EditionsError(
            f"{EDITIONS_FILE}: current_edition {table.current_edition!r} != config.RULES_EDITION {RULES_EDITION!r}")


_self_check_current_edition()
