"""Submission readiness: results, their runs and their paperwork.

Rules.md 1.3 names three things: a *run* (one leaf, one ledger ID), a
*result* (the runs of one division / system / benchmark / workload /
accelerator that become one ``results.csv`` row) and a *submission* (one
organization's results-dir). This module scores a results-dir at the result
level -- "how many runs do I have, how many does the edition ask for, and
what else stands between this result and an upload" -- for ``mlpstorage
status``, the per-result recap printed after every run, and ``mlpstorage
submit --dry-run``.

Two sources feed it and nothing else does:

* the run ledger (``mlpstorage_py.runs.ledger``): which leaves exist, their
  IDs, whether each finished (``exit_status``), and what the run-time
  verifier concluded (``verification`` in the leaf's metadata);
* the submission checker behind ``mlpstorage validate``
  (``mlpstorage_py.submission_checker.main.run``), executed in-process on
  the results-dir with its findings captured instead of printed. Readiness
  never re-implements a rule: whatever ``validate`` would say about the
  tree, this module attributes to a run, a result, a system's paperwork or
  the tree as a whole.

Severity is a matter of *stage*, not of level (design decision 1,
2026-09-23):

* **paperwork** -- fixable by editing files, never stops a run, hard only at
  submit time: the system description YAML's blank or invalid fields, a
  missing ``systems/<name>.pdf``, a leaf whose code image or provenance
  stamp the pool cannot resolve;
* **rerun problems** -- a run that failed, one the verifier or ``validate``
  found INVALID, one that qualifies for OPEN only under ``closed``, and
  extra runs beyond the edition's count (Rules.md 2.1.17 requires *exactly*
  six): the run must be removed (``mlpstorage runs rm``) or redone.

The SUBMIT token of a result is, in precedence order, ``short`` (fewer
counted runs than the edition asks for), ``invalid`` (complete, but a run
must go or a workload-level rule failed), ``paperwork``, ``ready``; a
``whatif`` result shows ``-``. Warnings from the checker are carried but
never change a token.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from mlpstorage_py.config import RULES_EDITION
from mlpstorage_py.editions import EditionsError, load_editions
from mlpstorage_py.provenance import MANIFEST_FILENAME, ProvenanceError, read_submission_manifest
from mlpstorage_py.results_dir import resolve_orgname
from mlpstorage_py.runs.ledger import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    RunRecord,
    parse_leaf,
    read_metadata,
    run_status,
    sync,
)

# Per-run STATUS tokens (``status --runs``).
RUN_OK = "ok"
RUN_FAILED = "failed"
RUN_RUNNING = "running"
RUN_INVALID = "invalid"
RUN_EXTRA = "extra"

# Per-result SUBMIT tokens, in precedence order.
SUBMIT_SHORT = "short"
SUBMIT_INVALID = "invalid"
SUBMIT_PAPERWORK = "paperwork"
SUBMIT_READY = "ready"
SUBMIT_NA = "-"

# The divisions a package may carry (Rules.md 2.1.2); ``whatif`` runs live in
# a results-dir but never in a package, so nothing ``validate`` says about
# them counts.
_PACKAGE_DIVISIONS = ("closed", "open")
_WHATIF = "whatif"

# Rules whose only content is the per-result count: RUNS n/m already says it.
COUNT_RULES = frozenset({"2.1.17", "2.1.23", "5.3.1"})
# Rules about reportgen's rollup files, which ``submit`` regenerates before
# packaging; a results-dir mid-flight has none and that is not a problem.
ROLLUP_RULES = frozenset({"2.1.16", "2.1.22", "RPT-01"})
# Paperwork of a *system*: its description YAML and PDF in ``systems/``.
SYSTEM_PAPERWORK_RULES = frozenset({"2.1.7", "2.1.8", "4.7.3", "4.7.4"})
# Paperwork of a *run leaf* (attributed to its result): code image and stamp.
LEAF_PAPERWORK_RULES = frozenset({"CHECK-01", "CHECK-02", "CHECK-03", "CHECK-04", "PROV-01"})

_FINDING_RE = re.compile(r"\[(?P<id>[^\s\]]+) (?P<name>[^\s\]]+)\] (?P<path>.*?): (?P<msg>.*)", re.S)
_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
_PDF_MISSING_RE = re.compile(r"\.yaml has no matching .*\.pdf|systems/.*\.pdf is missing")
_YAML_MISSING_RE = re.compile(r"\.pdf has no matching .*\.yaml|systems/.*\.yaml is missing")
_LOC_RE = re.compile(r"^[A-Za-z_][\w\[\]]*(?: -> [\w\[\]]+)*$")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """One line the submission checker emitted in its locked format
    ``[<rule_id> <rule_name>] <path>: <message>``."""
    level: str
    rule_id: str
    rule_name: str
    path: str
    message: str

    def to_dict(self) -> dict:
        return {"level": self.level, "rule_id": self.rule_id, "rule_name": self.rule_name,
                "path": self.path, "message": self.message}

    def short(self) -> str:
        return f"[{self.rule_id}] {self.message}"


@dataclass
class RunReadiness:
    id: int
    leaf: str
    status: str
    counted: bool = False
    reason: str = ""
    started: str = ""
    problems: List[Finding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "leaf": self.leaf, "status": self.status, "counted": self.counted,
                "reason": self.reason, "started": self.started,
                "problems": [f.to_dict() for f in self.problems]}


@dataclass
class Paperwork:
    """What a submitter must author or fix for one system before an upload."""
    division: str
    systemname: str
    yaml_missing: bool = False
    pdf_missing: bool = False
    blank_fields: List[str] = field(default_factory=list)
    other: List[Finding] = field(default_factory=list)

    @property
    def key(self) -> Tuple[str, str]:
        return (self.division, self.systemname)

    def items(self) -> List[str]:
        out = []
        if self.yaml_missing:
            out.append(f"systems/{self.systemname}.yaml missing")
        if self.blank_fields:
            n = len(self.blank_fields)
            out.append(f"systems/{self.systemname}.yaml: {n} blank field{'s' if n != 1 else ''}")
        if self.pdf_missing:
            out.append(f"systems/{self.systemname}.pdf missing")
        for f in self.other:
            out.append(f"systems/: {f.short()}")
        return out

    def to_dict(self) -> dict:
        return {"division": self.division, "systemname": self.systemname,
                "yaml_missing": self.yaml_missing, "pdf_missing": self.pdf_missing,
                "blank_fields": list(self.blank_fields), "other": [f.to_dict() for f in self.other],
                "items": self.items()}


@dataclass
class ResultReadiness:
    division: str
    orgname: str
    systemname: str
    benchmark: str
    model: str
    accelerator: Optional[str]
    required: int
    runs: List[RunReadiness] = field(default_factory=list)
    have: int = 0
    submit: str = SUBMIT_NA
    note: str = ""
    problems: List[Finding] = field(default_factory=list)
    paperwork: List[Finding] = field(default_factory=list)
    system_paperwork: Optional[Paperwork] = None

    @property
    def key(self) -> tuple:
        return (self.division, self.orgname, self.systemname, self.benchmark, self.model,
                self.accelerator or "")

    def to_dict(self) -> dict:
        return {"division": self.division, "orgname": self.orgname, "systemname": self.systemname,
                "benchmark": self.benchmark, "model": self.model, "accelerator": self.accelerator,
                "runs_required": self.required, "runs_have": self.have, "submit": self.submit,
                "note": self.note, "runs": [r.to_dict() for r in self.runs],
                "problems": [f.to_dict() for f in self.problems],
                "paperwork": [f.to_dict() for f in self.paperwork]}


@dataclass
class SubmissionReadiness:
    results_dir: str
    orgname: str
    edition: str
    results: List[ResultReadiness] = field(default_factory=list)
    paperwork: Dict[Tuple[str, str], Paperwork] = field(default_factory=dict)
    tree_problems: List[Finding] = field(default_factory=list)
    warnings: List[Finding] = field(default_factory=list)

    @property
    def package_results(self) -> List[ResultReadiness]:
        return [r for r in self.results if r.division != _WHATIF]

    @property
    def total_results(self) -> int:
        return len(self.package_results)

    @property
    def ready_count(self) -> int:
        return sum(1 for r in self.package_results if r.submit == SUBMIT_READY)

    @property
    def submittable(self) -> bool:
        rows = self.package_results
        return bool(rows) and not self.tree_problems and all(r.submit == SUBMIT_READY for r in rows)

    def to_dict(self) -> dict:
        return {"results_dir": self.results_dir, "orgname": self.orgname,
                "rules_edition": self.edition, "submittable": self.submittable,
                "results": [r.to_dict() for r in self.results],
                "paperwork": [p.to_dict() for _k, p in sorted(self.paperwork.items())],
                "tree_problems": [f.to_dict() for f in self.tree_problems],
                "warnings": [f.to_dict() for f in self.warnings]}


# ---------------------------------------------------------------------------
# The checker's findings
# ---------------------------------------------------------------------------

def parse_finding(level: str, text: str) -> Optional[Finding]:
    """A ``Finding`` from one checker log line, or ``None`` for the lines that
    are not rule findings (``Some X checks failed for:``, ``SUMMARY:``)."""
    m = _FINDING_RE.match(text.strip())
    if not m:
        return None
    return Finding(level=level, rule_id=m.group("id"), rule_name=m.group("name"),
                   path=m.group("path"), message=m.group("msg").strip())


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.findings: List[Finding] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
        except Exception:  # a torn format string; not a finding
            return
        f = parse_finding(record.levelname.lower(), text)
        if f is not None:
            self.findings.append(f)


def collect_findings(results_dir: str) -> List[Finding]:
    """Run the submission checker (the one behind ``mlpstorage validate``)
    on ``results_dir`` and return its findings instead of printing them.

    The checker's loggers are detached from the console for the duration of
    the call and restored afterwards; its CSV goes to a temporary directory.
    """
    from mlpstorage_py.submission_checker.main import run as checker_run

    root = os.path.abspath(results_dir)
    capture = _Capture()
    main_log = logging.getLogger("main")
    loader_log = logging.getLogger("Loader")
    saved = (main_log.propagate, loader_log.propagate)
    main_log.addHandler(capture)
    main_log.propagate = False
    loader_log.propagate = False
    try:
        with tempfile.TemporaryDirectory(prefix="mlps-readiness-") as td:
            checker_run(Namespace(input=root, submitters=None,
                                  csv=os.path.join(td, "summary.csv"),
                                  skip_output_file=False, reference_checksum=None))
    finally:
        main_log.removeHandler(capture)
        main_log.propagate, loader_log.propagate = saved
    return capture.findings


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def _relative_parts(root: str, path: str) -> Optional[List[str]]:
    try:
        rel = os.path.relpath(os.path.abspath(path), root)
    except ValueError:
        return None
    if rel == "." or rel.startswith(".."):
        return None
    return rel.replace(os.sep, "/").split("/")


def _leaf_of(parts: List[str]) -> Optional[str]:
    """The run leaf (results-dir relative) a path lies in, or ``None``."""
    for i in range(5, len(parts)):
        if _TIMESTAMP_RE.match(parts[i]):
            rel = "/".join(parts[:i + 1])
            if parse_leaf(rel) is not None:
                return rel
            return None
    return None


def _workload_of(parts: List[str]) -> Optional[Tuple[str, str, str, str]]:
    """``(division, systemname, benchmark, model)`` for a path under
    ``<division>/<org>/results/<system>/<benchmark>/<model...>``, with
    ``model`` empty for a path that stops above the workload."""
    if len(parts) < 4 or parts[0] not in _PACKAGE_DIVISIONS or parts[2] != "results":
        return None
    division, systemname = parts[0], parts[3]
    if len(parts) < 5:
        return (division, systemname, "", "")
    benchmark = parts[4]
    if benchmark == "vector_database":
        model = "/".join(parts[5:7]) if len(parts) >= 7 else ""
    else:
        model = parts[5] if len(parts) >= 6 else ""
    return (division, systemname, benchmark, model)


def _system_of_paperwork(parts: List[str]) -> Optional[Tuple[str, str]]:
    """``(division, systemname)`` for a path under ``systems/`` or
    ``results/<system>``."""
    if len(parts) < 4 or parts[0] not in _PACKAGE_DIVISIONS:
        return None
    if parts[2] == "systems":
        stem, _ext = os.path.splitext(parts[3])
        return (parts[0], stem) if stem else None
    if parts[2] == "results":
        return (parts[0], parts[3])
    return None


def _blank_field(message: str) -> Optional[str]:
    """The schema location a system-YAML finding names, or ``None``."""
    loc, sep, _rest = message.partition(": ")
    if not sep:
        return None
    loc = loc.strip()
    return loc if _LOC_RE.match(loc) else None


# ---------------------------------------------------------------------------
# Runs and results
# ---------------------------------------------------------------------------

def _accelerator_of(metadata: dict) -> Optional[str]:
    for value in (metadata.get("accelerator"), (metadata.get("args") or {}).get("accelerator_type")):
        if isinstance(value, str) and value:
            return value
    return None


def _checkpoint_counts(metadata: dict) -> Tuple[int, int]:
    params = ((metadata.get("parameters") or {}).get("checkpoint") or {})
    args = metadata.get("args") or {}

    def _int(*values):
        for v in values:
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        return 0
    return (_int(params.get("num_checkpoints_write"), args.get("num_checkpoints_write")),
            _int(params.get("num_checkpoints_read"), args.get("num_checkpoints_read")))


def _run_readiness(results_dir: str, record: RunRecord, metadata: Optional[dict]) -> RunReadiness:
    leaf_path = record.path(results_dir)
    raw = run_status(leaf_path, record.info)
    started = record.started()
    run = RunReadiness(id=record.id, leaf=record.leaf, status=RUN_OK,
                       started=started.strftime("%Y-%m-%d %H:%M:%S") if started else record.run_datetime)
    if raw == STATUS_FAILED:
        run.status = RUN_FAILED
        exit_status = (metadata or {}).get("exit_status")
        run.reason = f"exit status {exit_status}" if exit_status is not None else "no summary.json"
        return run
    if raw != STATUS_COMPLETE:
        run.status = RUN_RUNNING
        run.reason = "no metadata yet (still running, or killed before it was written)"
        return run
    division = record.info.get("mode")
    verification = (metadata or {}).get("verification")
    if division != _WHATIF and isinstance(verification, str):
        if verification.upper() == "INVALID":
            run.status = RUN_INVALID
            run.reason = "verified INVALID at run time (see runs show)"
        elif verification.upper() == "OPEN" and division == "closed":
            run.status = RUN_INVALID
            run.reason = "qualifies for OPEN only (verified OPEN at run time under closed)"
    return run


def _count_runs(result: ResultReadiness, metadata_by_id: Dict[int, dict]) -> None:
    """Pick the counted set and mark the extras (Rules.md 1.3)."""
    ok = sorted((r for r in result.runs if r.status == RUN_OK),
                key=lambda r: (r.leaf.rsplit("/", 1)[1], r.id))
    if result.benchmark == "checkpointing":
        phases: set = set()
        for run in ok:
            writes, reads = _checkpoint_counts(metadata_by_id.get(run.id) or {})
            supplies = set()
            if writes > 0 and "write" not in phases:
                supplies.add("write")
            if reads > 0 and "read" not in phases:
                supplies.add("read")
            if supplies:
                run.counted = True
                phases |= supplies
            else:
                run.status = RUN_EXTRA
                run.reason = "both phases already supplied by earlier runs"
        result.have = len(phases)
        return
    for i, run in enumerate(ok):
        if i < result.required:
            run.counted = True
        else:
            run.status = RUN_EXTRA
            run.reason = f"beyond the {result.required} runs one result holds"
    result.have = min(len(ok), result.required)


def _missing_phases(result: ResultReadiness, metadata_by_id: Dict[int, dict]) -> List[str]:
    have = set()
    for run in result.runs:
        if run.counted:
            writes, reads = _checkpoint_counts(metadata_by_id.get(run.id) or {})
            if writes > 0:
                have.add("write")
            if reads > 0:
                have.add("read")
    return [p for p in ("write", "read") if p not in have]


def _rm_hint(ids: Iterable[int]) -> str:
    return "mlpstorage runs rm " + " ".join(str(i) for i in ids)


def _score(result: ResultReadiness, metadata_by_id: Dict[int, dict]) -> None:
    """The SUBMIT token and NOTE of a result from its runs, problems and
    paperwork."""
    parts: List[str] = []
    missing = result.required - result.have
    if missing > 0:
        if result.benchmark == "checkpointing":
            phases = _missing_phases(result, metadata_by_id)
            if phases:
                parts.append(f"{' and '.join(p + ' phase' for p in phases)} missing "
                             f"({missing} more run{'s' if missing != 1 else ''} needed)")
            else:
                parts.append(f"{missing} more run{'s' if missing != 1 else ''} needed")
        else:
            parts.append(f"{missing} more run{'s' if missing != 1 else ''} needed")
    failed = [r for r in result.runs if r.status == RUN_FAILED]
    invalid = [r for r in result.runs if r.status == RUN_INVALID]
    extra = [r for r in result.runs if r.status == RUN_EXTRA]
    for run in failed:
        parts.append(f"run {run.id} failed ({run.reason}); remove it ({_rm_hint([run.id])})")
    for run in invalid:
        parts.append(f"run {run.id} invalid: {run.reason}; remove or redo it ({_rm_hint([run.id])})")
    if extra:
        n = len(extra)
        parts.append(f"{n} extra run{'s' if n != 1 else ''} ({', '.join(str(r.id) for r in extra)}); "
                     f"keep exactly {result.required} ({_rm_hint(r.id for r in extra)})")
    for f in result.problems:
        parts.append(f.short())

    if result.division == _WHATIF:
        result.submit = SUBMIT_NA
        result.note = "; ".join(parts)
        return
    if missing > 0:
        result.submit = SUBMIT_SHORT
    elif failed or invalid or extra or result.problems:
        result.submit = SUBMIT_INVALID
    else:
        paper: List[str] = []
        if result.paperwork:
            paper.append("code image: " + "; ".join(f.short() for f in result.paperwork))
        if result.system_paperwork is not None:
            paper.extend(result.system_paperwork.items())
        if paper:
            result.submit = SUBMIT_PAPERWORK
            parts.extend(paper)
        else:
            result.submit = SUBMIT_READY
    result.note = "; ".join(parts)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _declared_edition(results_dir: str, orgname: str) -> str:
    for division in _PACKAGE_DIVISIONS:
        path = os.path.join(results_dir, division, orgname, MANIFEST_FILENAME)
        if os.path.isfile(path):
            try:
                declared = read_submission_manifest(path).get("rules_edition")
            except ProvenanceError:
                continue
            if isinstance(declared, str) and declared:
                return declared
    return RULES_EDITION


def _runs_per_result(edition: str) -> Dict[str, int]:
    try:
        table = load_editions()
    except EditionsError:
        return {}
    params = table.checker_parameters(edition) or table.checker_parameters()
    return dict(params.runs_per_result) if params is not None else {}


def evaluate(results_dir: str) -> SubmissionReadiness:
    """Score every result in an initialized results-dir.

    Raises ``ResultsDirNotInitializedError`` when the directory carries no
    ``mlperf-results.yaml`` sentinel.
    """
    root = os.path.abspath(results_dir)
    orgname = resolve_orgname(root)
    edition = _declared_edition(root, orgname)
    counts = _runs_per_result(edition)
    sub = SubmissionReadiness(results_dir=results_dir, orgname=orgname, edition=edition)

    # Runs from the ledger, grouped into results.
    by_key: Dict[tuple, ResultReadiness] = {}
    run_index: Dict[str, Tuple[ResultReadiness, RunReadiness]] = {}
    metadata_by_id: Dict[int, dict] = {}
    for record in sync(root):
        info = record.info
        if not info or info.get("command") != "run":
            continue
        metadata = read_metadata(record.path(root))
        if metadata is not None:
            metadata_by_id[record.id] = metadata
        accelerator = _accelerator_of(metadata or {})
        if info["benchmark"] in ("vector_database", "kv_cache"):
            accelerator = None
        key = (info["mode"], info["orgname"], info["systemname"], info["benchmark"], info["model"],
               accelerator or "")
        result = by_key.get(key)
        if result is None:
            result = ResultReadiness(division=info["mode"], orgname=info["orgname"],
                                     systemname=info["systemname"], benchmark=info["benchmark"],
                                     model=info["model"], accelerator=accelerator,
                                     required=counts.get(info["benchmark"], 1))
            by_key[key] = result
        run = _run_readiness(root, record, metadata)
        result.runs.append(run)
        run_index[record.leaf] = (result, run)

    # A run whose metadata is not written yet (still running, or killed
    # early) records no accelerator; it belongs to the sibling result when
    # there is exactly one, and to a row of its own otherwise.
    for key in [k for k in by_key if k[5] == "" and k[3] in ("training", "checkpointing")]:
        siblings = [k for k in by_key if k[:5] == key[:5] and k[5] != ""]
        if len(siblings) == 1:
            target = by_key[siblings[0]]
            for run in by_key[key].runs:
                target.runs.append(run)
                run_index[run.leaf] = (target, run)
            del by_key[key]

    leaf_by_timestamp: Dict[str, List[str]] = {}
    for leaf in run_index:
        leaf_by_timestamp.setdefault(leaf.rsplit("/", 1)[1], []).append(leaf)

    # The checker's findings, attributed.
    for f in collect_findings(root):
        parts = _relative_parts(root, f.path)
        if parts is None and _TIMESTAMP_RE.match(f.path.strip()):
            # 2.1.18 / 2.1.24 name the bare timestamp directory; unique
            # in the tree it names the run, otherwise nobody.
            candidates = leaf_by_timestamp.get(f.path.strip(), [])
            if len(candidates) == 1:
                parts = candidates[0].split("/")
        if parts is not None and parts[0] == _WHATIF:
            continue
        if f.level == "warning":
            sub.warnings.append(f)
            continue
        if f.level != "error":
            continue
        if f.rule_id in COUNT_RULES or f.rule_id in ROLLUP_RULES:
            continue
        attributed = False
        if parts is not None:
            if f.rule_id in SYSTEM_PAPERWORK_RULES:
                key = _system_of_paperwork(parts)
                if key is not None:
                    paper = sub.paperwork.get(key)
                    if paper is None:
                        paper = sub.paperwork[key] = Paperwork(division=key[0], systemname=key[1])
                    if _PDF_MISSING_RE.search(f.message):
                        paper.pdf_missing = True
                    elif _YAML_MISSING_RE.search(f.message):
                        paper.yaml_missing = True
                    else:
                        loc = _blank_field(f.message)
                        if loc is not None:
                            paper.blank_fields.append(loc)
                        else:
                            paper.other.append(f)
                    attributed = True
            else:
                leaf = _leaf_of(parts)
                workload = _workload_of(parts)
                if leaf is not None and leaf in run_index:
                    result, run = run_index[leaf]
                    if f.rule_id in LEAF_PAPERWORK_RULES:
                        result.paperwork.append(f)
                    else:
                        run.problems.append(f)
                        if run.status == RUN_OK:
                            run.status = RUN_INVALID
                            run.reason = f.short()
                    attributed = True
                elif workload is not None and f.rule_id not in LEAF_PAPERWORK_RULES:
                    division, systemname, benchmark, model = workload
                    for result in by_key.values():
                        if (result.division, result.systemname) != (division, systemname):
                            continue
                        if benchmark and result.benchmark != benchmark:
                            continue
                        if model and result.model != model:
                            continue
                        result.problems.append(f)
                        attributed = True
        if not attributed:
            sub.tree_problems.append(f)

    for result in by_key.values():
        result.system_paperwork = sub.paperwork.get((result.division, result.systemname))
        _count_runs(result, metadata_by_id)
        result.runs.sort(key=lambda r: (r.leaf.rsplit("/", 1)[1], r.id))
        _score(result, metadata_by_id)
    sub.results = sorted(by_key.values(), key=lambda r: r.key)
    return sub


def evaluate_result(results_dir: str, leaf_path: str) -> Optional[ResultReadiness]:
    """The result that the run leaf ``leaf_path`` belongs to, scored, or
    ``None`` when the leaf is not a run of the results-dir."""
    root = os.path.abspath(results_dir)
    parts = _relative_parts(root, leaf_path)
    leaf = _leaf_of(parts) if parts else None
    if leaf is None:
        return None
    for result in evaluate(results_dir).results:
        if any(run.leaf == leaf for run in result.runs):
            return result
    return None
