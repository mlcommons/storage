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
(division, family, model, emulated accelerator) whose ``core-config-v1`` hash
is one of the class's ``core_configs`` executed the same workload in every
edition the class lists. Rows in one class compare; nothing else does, and
comparisons across divisions are never declared. Classification is
a pure function of (stamp, table) and is never cached into evidence files.

An edition's **checker parameters** (``checker:``) are what ``validate`` needs
to check a submission of that edition: the required files and folders of
every datagen / run / checkpoint leaf, the Rules.md 3.3.2 AU minimum per
training model, the Table 2 CLOSED process counts and checkpoint sizes, the
Table 3 simulated-accelerator memory and the 6.3.2.1 KVCache sequence locks.
Only an edition this tool can check carries them; ``Config`` is built from
them, once tree-wide for the current edition and once per submission from
the edition its ``submission.yaml`` declares. There is no reviewer-side
edition flag. The runtime reads the current edition's block for the same
values (:func:`current_edition`, :func:`checker_parameters`).

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
CLASS_FAMILIES = ("training", "checkpointing", "vector_database", "kv_cache")
CLASS_DIVISIONS = ("closed", "open", "whatif")
EDITION_STATUSES = ("historical", "current")
_ACCELERATORS = ("b200", "mi355", "h100", "a100", ANY_ACCELERATOR)
_CLASS_ID_RE = re.compile(r"[a-z0-9.]+(-[a-z0-9]+)*-[A-Z]")
_HASH_RE = re.compile(r"[0-9a-f]{16}")
_SHA_RE = re.compile(r"[0-9a-f]{40}")


CHECKER_LIST_FIELDS = ("datagen_required_files", "datagen_required_folders",
                       "run_required_files", "run_required_folders",
                       "checkpoint_required_files", "checkpoint_required_folders")
_CHECKER_REGEX_FIELDS = ("datagen_required_files", "run_required_files", "checkpoint_required_files")
# The edition-varying values (design D-16): mappings keyed by model or
# accelerator name exactly as ``workloads:`` spells them.
CHECKER_VALUE_FIELDS = ("training_au_thresholds", "closed_mpi_processes", "checkpoint_size_gb",
                        "accelerator_memory_gb", "kvcache_closed_sequence")
CHECKER_FIELDS = CHECKER_LIST_FIELDS + CHECKER_VALUE_FIELDS
KVCACHE_SEQUENCE_KEYS = ("seed", "trials", "inter_option_delay_s")


class EditionsError(ValueError):
    """The editions table is missing, unreadable or inconsistent."""


class UncheckableEditionError(EditionsError):
    """A rules edition this tool cannot check: unknown to the table, or listed
    without ``checker:`` parameters (a historical edition checked by its own
    tool)."""


@dataclass(frozen=True)
class CheckerParameters:
    """What ``validate`` requires of every leaf of one edition, and the
    edition's values of the rules whose numbers may change between editions."""
    datagen_required_files: List[str]
    datagen_required_folders: List[str]
    run_required_files: List[str]
    run_required_folders: List[str]
    checkpoint_required_files: List[str]
    checkpoint_required_folders: List[str]
    #: Rules.md 3.3.2 -- minimum mean AU per training model, as a fraction.
    training_au_thresholds: Dict[str, float]
    #: Rules.md Table 2 "Total Processes" per checkpointing model (4.6.1).
    closed_mpi_processes: Dict[str, int]
    #: Rules.md Table 2 "Checkpoint size" per checkpointing model, GB (4.3.4 pre-flight).
    checkpoint_size_gb: Dict[str, float]
    #: Rules.md Table 3 -- memory per simulated accelerator, GB (4.3.4).
    accelerator_memory_gb: Dict[str, float]
    #: Rules.md 6.3.2.1 -- ``seed`` / ``trials`` / ``inter_option_delay_s`` of a CLOSED kv_cache run.
    kvcache_closed_sequence: Dict[str, int]


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
    checker: Optional[CheckerParameters] = None

    @property
    def checkable(self) -> bool:
        return self.checker is not None

    @property
    def dlio_commits(self) -> frozenset:
        return frozenset(r["commit"] for r in self.dlio_revisions)

    def models(self, family: str, division: Optional[str] = None) -> List[str]:
        """The models ``workloads:`` sanctions for ``family`` in ``division``
        (every division when ``None``), in first-appearance order."""
        out: List[str] = []
        for div, families in self.workloads.items():
            if division is not None and div != division:
                continue
            for model in (families.get(family) or {}):
                if model not in out:
                    out.append(model)
        return out

    def accelerators(self, family: Optional[str] = None, division: Optional[str] = None) -> List[str]:
        """The emulated accelerators ``workloads:`` sanctions (for one family
        and / or division when given), in first-appearance order."""
        out: List[str] = []
        for div, families in self.workloads.items():
            if division is not None and div != division:
                continue
            for fam, models in families.items():
                if family is not None and fam != family:
                    continue
                for accels in models.values():
                    for a in accels:
                        if a not in out:
                            out.append(a)
        return out


@dataclass(frozen=True)
class ComparabilityClass:
    id: str
    division: str
    family: str
    model: str
    accelerator: str
    allowlist: str
    core_configs: Tuple[str, ...]
    editions: Tuple[str, ...]
    reference: Optional[str] = None
    notes: str = ""

    def matches(self, *, division: str, family: str, model: str, accelerator: Optional[str],
                core_config: str, edition: Optional[str] = None) -> bool:
        if self.division != division or self.family != family or self.model != model:
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

    def is_checkable(self, edition_id: Any) -> bool:
        e = self.edition(edition_id)
        return e is not None and e.checkable

    def checkable_editions(self) -> List[Edition]:
        return [e for e in self.editions.values() if e.checkable]

    def families(self) -> frozenset:
        """The workload families (results/<system>/ directory names) of every
        edition this tool can check, all divisions -- the tree-wide 2.1.10
        vocabulary."""
        return frozenset(fam for e in self.checkable_editions()
                         for families in e.workloads.values() for fam in families)

    def vocabulary(self, family: str) -> frozenset:
        """The workload directory names of ``family`` across every edition this
        tool can check, all divisions -- the tree-wide 2.1.11 / 2.1.21
        vocabulary. Historical editions checked by their own tool do not widen
        it."""
        return frozenset(m for e in self.checkable_editions() for m in e.models(family))

    def checker_parameters(self, edition_id: Any = None) -> Optional[CheckerParameters]:
        """The ``checker:`` block of an edition (the current one by default), or
        ``None`` when the table does not know the edition or lists it without one."""
        e = self.edition(self.current_edition if edition_id is None else edition_id)
        return e.checker if e is not None else None

    def require_checkable(self, edition_id: Any = None) -> CheckerParameters:
        """``checker_parameters`` that raises :class:`UncheckableEditionError`
        with the reason (unknown edition, or which tool checks it)."""
        eid = self.current_edition if edition_id is None else str(edition_id)
        e = self.edition(eid)
        if e is None:
            raise UncheckableEditionError(
                f"rules edition {eid} is not in {self.path.name} (known: {', '.join(sorted(self.editions))})")
        if e.checker is None:
            raise UncheckableEditionError(
                f"rules edition {eid} cannot be checked by this tool ({self.path.name} lists it "
                f"without checker parameters; it is checked by its own tool: {e.tool})")
        return e.checker

    def classify(self, *, division: str, family: str, model: str, accelerator: Optional[str],
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
            if c.matches(division=division, family=family, model=model, accelerator=accelerator,
                         core_config=core_config, edition=edition):
                return c
        return None

    def classify_stamp(self, stamp, *, division: str, family: str, model: str,
                       accelerator: Optional[str]) -> Optional[ComparabilityClass]:
        """Classify a ``RunProvenance`` under its own stamped edition."""
        edition = getattr(stamp, "rules_edition", UNKNOWN)
        if edition == UNKNOWN:
            return None
        return self.classify(division=division, family=family, model=model, accelerator=accelerator,
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
    checker = _parse_checker(eid, raw.get("checker"), where)
    edition = Edition(
        id=eid, status=status, results_repo=raw["results_repo"], tool=str(raw.get("tool", UNKNOWN)),
        layout_versions=list(layout), workloads=workloads,
        dlio_revisions=[{"commit": r["commit"], "source": r["source"], "version": str(r.get("version", UNKNOWN))}
                        for r in revs],
        storage_libraries={str(k): [str(v) for v in vs] for k, vs in libs.items()},
        notes=str(raw.get("notes", "") or ""),
        checker=checker,
    )
    if checker is not None:
        _check_values_against_workloads(edition, where)
    return edition


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _parse_value_map(eid: str, name: str, raw: Any, where: str, *, integer: bool = False,
                     fraction: bool = False) -> Dict[str, Any]:
    """A ``checker:`` value block: a non-empty mapping of name -> positive
    number (int when ``integer``; in (0, 1] when ``fraction``)."""
    _require(isinstance(raw, dict) and raw, f"{where}: edition {eid}: checker.{name} must be a non-empty mapping")
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        _require(isinstance(k, str) and k, f"{where}: edition {eid}: checker.{name}: keys must be names")
        if integer:
            _require(isinstance(v, int) and not isinstance(v, bool) and v > 0,
                     f"{where}: edition {eid}: checker.{name}.{k} must be a positive integer, got {v!r}")
        elif fraction:
            _require(_is_number(v) and 0 < v <= 1,
                     f"{where}: edition {eid}: checker.{name}.{k} must be a fraction in (0, 1], got {v!r}")
        else:
            _require(_is_number(v) and v > 0,
                     f"{where}: edition {eid}: checker.{name}.{k} must be a positive number, got {v!r}")
        out[k] = v
    return out


def _parse_kvcache_sequence(eid: str, raw: Any, where: str) -> Dict[str, int]:
    name = "kvcache_closed_sequence"
    _require(isinstance(raw, dict), f"{where}: edition {eid}: checker.{name} must be a mapping")
    _require(set(raw) == set(KVCACHE_SEQUENCE_KEYS),
             f"{where}: edition {eid}: checker.{name} must carry exactly {list(KVCACHE_SEQUENCE_KEYS)}, "
             f"got {sorted(raw)}")
    for k, v in raw.items():
        _require(isinstance(v, int) and not isinstance(v, bool),
                 f"{where}: edition {eid}: checker.{name}.{k} must be an integer, got {v!r}")
    _require(raw["trials"] > 0, f"{where}: edition {eid}: checker.{name}.trials must be positive")
    _require(raw["inter_option_delay_s"] >= 0,
             f"{where}: edition {eid}: checker.{name}.inter_option_delay_s must not be negative")
    return {k: int(raw[k]) for k in KVCACHE_SEQUENCE_KEYS}


def _parse_checker(eid: str, raw: Any, where: str) -> Optional[CheckerParameters]:
    if raw is None:
        return None
    _require(isinstance(raw, dict), f"{where}: edition {eid}: checker must be a mapping")
    unknown = sorted(set(raw) - set(CHECKER_FIELDS))
    _require(not unknown, f"{where}: edition {eid}: checker has unknown key(s) {unknown}")
    missing = [n for n in CHECKER_FIELDS if n not in raw]
    _require(not missing, f"{where}: edition {eid}: checker is missing {missing}")
    values: Dict[str, Any] = {}
    for name in CHECKER_LIST_FIELDS:
        v = raw.get(name)
        _require(isinstance(v, list) and v and all(isinstance(x, str) and x for x in v),
                 f"{where}: edition {eid}: checker.{name} must be a non-empty list of strings")
        if name in _CHECKER_REGEX_FIELDS:
            for pattern in v:
                try:
                    re.compile(pattern)
                except re.error as e:
                    raise EditionsError(
                        f"{where}: edition {eid}: checker.{name}: {pattern!r} is not a regex ({e})") from e
        values[name] = list(v)
    values["training_au_thresholds"] = _parse_value_map(eid, "training_au_thresholds",
                                                        raw["training_au_thresholds"], where, fraction=True)
    values["closed_mpi_processes"] = _parse_value_map(eid, "closed_mpi_processes",
                                                      raw["closed_mpi_processes"], where, integer=True)
    values["checkpoint_size_gb"] = _parse_value_map(eid, "checkpoint_size_gb", raw["checkpoint_size_gb"], where)
    values["accelerator_memory_gb"] = _parse_value_map(eid, "accelerator_memory_gb",
                                                       raw["accelerator_memory_gb"], where)
    values["kvcache_closed_sequence"] = _parse_kvcache_sequence(eid, raw["kvcache_closed_sequence"], where)
    return CheckerParameters(**values)


def _check_values_against_workloads(edition: Edition, where: str) -> None:
    """The value blocks are keyed by the edition's own vocabulary: AU minimums
    name exactly its training models, the Table 2 blocks exactly its
    checkpointing models, and every accelerator any workload lists has a
    memory entry (extra accelerators, e.g. whatif-only ones, are allowed)."""
    eid, c = edition.id, edition.checker
    training = set(edition.models("training"))
    checkpointing = set(edition.models("checkpointing"))
    _require(set(c.training_au_thresholds) == training,
             f"{where}: edition {eid}: checker.training_au_thresholds must name exactly the edition's training "
             f"models {sorted(training)}, got {sorted(c.training_au_thresholds)}")
    for name in ("closed_mpi_processes", "checkpoint_size_gb"):
        keys = set(getattr(c, name))
        _require(keys == checkpointing,
                 f"{where}: edition {eid}: checker.{name} must name exactly the edition's checkpointing models "
                 f"{sorted(checkpointing)}, got {sorted(keys)}")
    missing = sorted(set(edition.accelerators()) - set(c.accelerator_memory_gb))
    _require(not missing,
             f"{where}: edition {eid}: checker.accelerator_memory_gb lacks the workload accelerator(s) {missing}")


def _parse_class(raw: Any, editions: Dict[str, Edition], allowlists: Dict[str, Any],
                 where: str) -> ComparabilityClass:
    _require(isinstance(raw, dict), f"{where}: every class must be a mapping")
    cid = raw.get("id")
    _require(isinstance(cid, str) and _CLASS_ID_RE.fullmatch(cid),
             f"{where}: class id {cid!r} must look like <model>-<accelerator>-<Letter>")
    _require(raw.get("division") in CLASS_DIVISIONS,
             f"{where}: class {cid}: division must be one of {CLASS_DIVISIONS}")
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
        id=cid, division=raw["division"], family=raw["family"], model=raw["model"],
        accelerator=raw["accelerator"],
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
    _require(editions[current].checkable, f"{where}: current_edition {current} must carry checker parameters")
    raw_classes = data.get("classes")
    _require(isinstance(raw_classes, list), f"{where}: classes must be a list")
    allowlists = load_allowlists().get("allowlists", {})
    classes = [_parse_class(c, editions, allowlists, where) for c in raw_classes]
    ids = [c.id for c in classes]
    _require(len(set(ids)) == len(ids), f"{where}: duplicate class id(s): {sorted({i for i in ids if ids.count(i) > 1})}")
    _require(ids == sorted(ids), f"{where}: classes must be sorted by id")
    claimed: Dict[Tuple[str, str, str, str, str, str], str] = {}
    for c in classes:
        for e in c.editions:
            for h in c.core_configs:
                key = (c.division, c.family, c.model, c.accelerator, e, h)
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


def checker_parameters(edition_id: Any = None) -> Optional[CheckerParameters]:
    """The shipped table's ``checker:`` block for an edition (current by default)."""
    return load_editions().checker_parameters(edition_id)


def current_edition() -> Edition:
    """The shipped table's entry for the edition this tool implements -- what
    the runtime (CLI choices, pre-flight gates, CLOSED defaults) reads."""
    table = load_editions()
    return table.editions[table.current_edition]


def describe_class(stamp, *, division: str, family: str, model: str, accelerator: Optional[str]) -> str:
    """One line for ``mlpstorage runs show``: the class id, or why there is none."""
    try:
        table = load_editions()
    except EditionsError as e:
        return f"unavailable ({e})"
    core = (getattr(stamp, "core_config", None) or {}).get("hash", UNKNOWN)
    if not isinstance(core, str) or core == UNKNOWN:
        return "n/a (the leaf recorded no hashable workload; core-config hash unknown)"
    edition = getattr(stamp, "rules_edition", UNKNOWN)
    if edition == UNKNOWN:
        c = table.classify(division=division, family=family, model=model, accelerator=accelerator,
                           core_config=core)
        if c is None:
            return "unclassified (matched by hash across every edition; stamp has no edition)"
        return f"{c.id} (matched by hash across every edition; stamp has no edition)"
    if edition not in table.editions:
        return f"unclassified (rules edition {edition} is not in {EDITIONS_FILE.name})"
    c = table.classify(division=division, family=family, model=model, accelerator=accelerator,
                       core_config=core, edition=edition)
    return c.id if c is not None else (f"unclassified (no class for edition {edition} {division} "
                                       f"{family}/{model}/{accelerator or UNKNOWN} {core})")


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
