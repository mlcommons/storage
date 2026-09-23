"""``mlpstorage submit``: check, package, record, and say how to upload.

The submitter's verb over their own initialized results-dir (design decision
4, 2026-09-23). ``validate`` is the reviewer's tool over an arbitrary tree;
``submit`` runs the same checker -- never a second implementation -- but
over the results-dir the sentinel names, with the refusal rendered as the
``status`` table, and with a package at the end.

The steps, in order:

1. Regenerate the rollup tables and the per-organization ``submission.yaml``
   (``reports reportgen``), because the rules about them (2.1.16, 2.1.22,
   RPT-01, PROV-02) can only be judged on fresh files. This happens in
   ``--dry-run`` too: the check must be the real check.
2. Run the checker once (``readiness.collect_findings``) and score the tree
   from that list (``readiness.evaluate``). Refuse with exit 1 and the
   ``status`` table while any error remains: a short or invalid result, a
   system's paperwork, a tree problem, or a checker error the table does not
   carry. Warnings never block; ``whatif`` results are never packaged and
   nothing the checker says about them counts. There is no ``--force``.
3. Plan the package: ``<org>/closed/<org>/**`` and ``<org>/open/<org>/**``
   exactly as they stand, plus ``<org>/code-images/`` holding the marker and
   only the images that the packaged leaves point at (resolved against the
   tree-wide pool first, the v3.0 per-organization pool second, Rules.md
   2.1.6). Never ``whatif/``, ``.mlps/`` or the sentinel: the package is what
   Rules.md 2.1.2 names and nothing else.
4. ``--dry-run`` stops here and says what it would write. Otherwise write
   the gzipped tarball, a ``.sha256`` sidecar (``sha256sum`` format) and a
   ``.manifest.json`` (``mlps-submission-package/1``: the results, the
   images and every file with its size and digest), append one line to
   ``<results-dir>/.mlps/submissions.jsonl`` (the packaging ledger, mirroring
   ``runs.jsonl``), and print the manual upload instructions. There is no
   uploader yet; ``UPLOAD_INSTRUCTIONS`` is the plug-in point for one.

Output goes to stdout like ``status``; the logger (stderr) carries the
results-dir line, reportgen's own progress and any warning.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import io
import json
import os
import tarfile
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mlpstorage_py import VERSION
from mlpstorage_py.config import EXIT_CODE
import mlpstorage_py.readiness as readiness
from mlpstorage_py.readiness import (
    ROLLUP_RULES,
    SUBMIT_INVALID,
    SUBMIT_PAPERWORK,
    SUBMIT_SHORT,
    SubmissionReadiness,
)
from mlpstorage_py.runs.ledger import read_pointer_hash
from mlpstorage_py.status import render

LEDGER_RELPATH = os.path.join(".mlps", "submissions.jsonl")
PACKAGES_RELPATH = os.path.join(".mlps", "packages")
PACKAGE_SCHEMA = "mlps-submission-package/1"

_PACKAGE_DIVISIONS = ("closed", "open")
_POOL_DIRNAME = "code-images"
_POOL_MARKER = ".mlps-image-pool"
_POINTER_FILENAME = ".mlps-code-image"
_TARBALL_SUFFIXES = (".tar.gz", ".tgz")

UPLOAD_INSTRUCTIONS = (
    "Upload the package and its .sha256 through the MLCommons submission UI for this round\n"
    "(the link is in the round's call for submissions). Every upload replaces the previous\n"
    "one, so run `mlpstorage submit` and upload again whenever a result changes; uploading\n"
    "every day or two keeps a usable package on file. Keep the manifest with your records:\n"
    "it lists every file in the package with its checksum."
)


class SubmitError(Exception):
    """The package cannot be built from this tree (a pointer no pool resolves)."""


# ---------------------------------------------------------------------------
# Rollups
# ---------------------------------------------------------------------------

def regenerate_rollups(results_dir: str, logger) -> List[str]:
    """``reports reportgen`` over ``results_dir``: per-model and per-org
    ``results.{csv,json}`` and each organization's ``submission.yaml``.
    Raises on any failure; the caller turns that into a refusal.

    reportgen's own report (the results table, skipped run directories)
    goes to stdout; here it is captured and logged at debug level so the
    ``status`` table is the first thing a submitter reads. Its warnings
    still reach stderr through ``logger``."""
    from mlpstorage_py.report_generator import ReportGenerator

    args = Namespace(mode="reports", command="reportgen", results_dir=results_dir,
                     systemname=None, debug=False)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        rc = ReportGenerator(results_dir, args, logger=logger).generate_reports()
    report = captured.getvalue().strip()
    if report:
        logger.debug(f"reports reportgen output:\n{report}")
    if rc != EXIT_CODE.SUCCESS:
        raise RuntimeError(f"reports reportgen exited {rc}")
    return []


# ---------------------------------------------------------------------------
# The package plan
# ---------------------------------------------------------------------------

@dataclass
class PackageEntry:
    """One tar member: a file or directory copied from ``source``, or a
    small file synthesized from ``data``."""
    arcname: str
    source: Optional[str] = None
    data: Optional[bytes] = None

    @property
    def is_file(self) -> bool:
        return self.data is not None or (self.source is not None and os.path.isfile(self.source))

    @property
    def size(self) -> int:
        if self.data is not None:
            return len(self.data)
        return os.path.getsize(self.source) if self.is_file else 0


@dataclass
class PackagePlan:
    orgname: str
    edition: str
    divisions: List[str] = field(default_factory=list)
    systems: Dict[str, List[str]] = field(default_factory=dict)
    images: List[str] = field(default_factory=list)
    entries: List[PackageEntry] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return sum(1 for e in self.entries if e.is_file)

    @property
    def size_bytes(self) -> int:
        return sum(e.size for e in self.entries if e.is_file)


def _walk_sorted(top: str):
    """``(dirpath, dirnames, filenames)`` like ``os.walk``, in sorted order
    so the tarball is the same for the same tree."""
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames.sort()
        yield dirpath, dirnames, sorted(filenames)


def _resolve_image(root: str, orgname: str, full_hash: str) -> Optional[str]:
    """The pool directory ``full_hash`` names: the tree-wide pool first, the
    organization's own pool second (Rules.md 2.1.6)."""
    name = f"code-{full_hash[:8]}"
    for candidate in (os.path.join(root, _POOL_DIRNAME, name), os.path.join(root, orgname, name)):
        if os.path.isdir(candidate):
            return candidate
    return None


def _pool_marker(root: str, orgname: str, now: _dt.datetime) -> PackageEntry:
    arcname = f"{orgname}/{_POOL_DIRNAME}/{_POOL_MARKER}"
    for candidate in (os.path.join(root, _POOL_DIRNAME, _POOL_MARKER),
                      os.path.join(root, orgname, _POOL_MARKER)):
        if os.path.isfile(candidate):
            return PackageEntry(arcname, source=candidate)
    data = f"mlpstorage_version={VERSION}\npackaged_at={now.replace(microsecond=0).isoformat()}\n"
    return PackageEntry(arcname, data=data.encode("utf-8"))


def plan_package(results_dir: str, sub: SubmissionReadiness, now: _dt.datetime) -> PackagePlan:
    """Everything the tarball will hold, in order."""
    root = os.path.abspath(results_dir)
    org = sub.orgname
    plan = PackagePlan(orgname=org, edition=sub.edition)
    plan.entries.append(PackageEntry(org, source=root))
    hashes: Dict[str, List[str]] = {}

    for division in _PACKAGE_DIVISIONS:
        org_dir = os.path.join(root, division, org)
        if not os.path.isdir(org_dir):
            continue
        plan.divisions.append(division)
        systems_dir = os.path.join(org_dir, "systems")
        plan.systems[division] = sorted(
            os.path.splitext(n)[0] for n in os.listdir(systems_dir) if n.endswith(".yaml")
        ) if os.path.isdir(systems_dir) else []
        plan.entries.append(PackageEntry(f"{org}/{division}", source=os.path.join(root, division)))
        for dirpath, dirnames, filenames in _walk_sorted(org_dir):
            rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
            plan.entries.append(PackageEntry(f"{org}/{rel}", source=dirpath))
            for name in filenames:
                path = os.path.join(dirpath, name)
                plan.entries.append(PackageEntry(f"{org}/{rel}/{name}", source=path))
                if name == _POINTER_FILENAME:
                    full_hash = read_pointer_hash(dirpath)
                    if full_hash:
                        hashes.setdefault(full_hash, []).append(f"{rel}")

    plan.entries.append(PackageEntry(f"{org}/{_POOL_DIRNAME}", source=root))
    plan.entries.append(_pool_marker(root, org, now))
    for full_hash in sorted(hashes):
        image = _resolve_image(root, org, full_hash)
        if image is None:
            leaves = ", ".join(hashes[full_hash][:3])
            raise SubmitError(f"no pool image for code-{full_hash[:8]} (pointed at by {leaves})")
        name = os.path.basename(image)
        plan.images.append(name)
        plan.entries.append(PackageEntry(f"{org}/{_POOL_DIRNAME}/{name}", source=image))
        for dirpath, dirnames, filenames in _walk_sorted(image):
            rel = os.path.relpath(dirpath, image).replace(os.sep, "/")
            prefix = f"{org}/{_POOL_DIRNAME}/{name}" + ("" if rel == "." else f"/{rel}")
            if rel != ".":
                plan.entries.append(PackageEntry(prefix, source=dirpath))
            for fname in filenames:
                plan.entries.append(PackageEntry(f"{prefix}/{fname}", source=os.path.join(dirpath, fname)))
    return plan


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _human_size(num: int) -> str:
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def package_path(results_dir: str, out: Optional[str], orgname: str, edition: str,
                 now: _dt.datetime) -> str:
    """Where the tarball goes: ``--out`` as a file when it ends in
    ``.tar.gz``/``.tgz``, as a directory otherwise; the default is
    ``<results-dir>/.mlps/packages/<org>-<edition>-<timestamp>.tar.gz``."""
    name = f"{orgname}-{edition}-{now.strftime('%Y%m%d_%H%M%S')}.tar.gz"
    if not out:
        return os.path.join(os.path.abspath(results_dir), PACKAGES_RELPATH, name)
    out = os.path.abspath(os.path.expanduser(out))
    if out.endswith(_TARBALL_SUFFIXES) and not os.path.isdir(out):
        return out
    return os.path.join(out, name)


def sidecar_paths(tarball: str) -> Tuple[str, str]:
    """``(<stem>.sha256, <stem>.manifest.json)`` beside the tarball."""
    stem = tarball
    for suffix in _TARBALL_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem + ".sha256", stem + ".manifest.json"


@dataclass
class WrittenPackage:
    path: str
    sha256: str
    size_bytes: int
    files: List[dict]


def write_tarball(plan: PackagePlan, path: str, now: _dt.datetime) -> WrittenPackage:
    """Write ``plan`` to ``path`` (gzipped tar, written as ``.part`` and
    renamed into place) and return its digest, size and per-file inventory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    files: List[dict] = []
    mtime = int(now.timestamp())
    with tarfile.open(tmp, "w:gz") as tar:
        for entry in plan.entries:
            if entry.data is not None:
                info = tarfile.TarInfo(entry.arcname)
                info.size = len(entry.data)
                info.mtime = mtime
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(entry.data))
                files.append({"path": entry.arcname, "size": len(entry.data),
                              "sha256": hashlib.sha256(entry.data).hexdigest()})
                continue
            tar.add(entry.source, arcname=entry.arcname, recursive=False)
            if os.path.isfile(entry.source):
                files.append({"path": entry.arcname, "size": os.path.getsize(entry.source),
                              "sha256": _sha256_file(entry.source)})
    os.replace(tmp, path)
    return WrittenPackage(path=path, sha256=_sha256_file(path), size_bytes=os.path.getsize(path),
                          files=files)


def _results_inventory(sub: SubmissionReadiness) -> List[dict]:
    out = []
    for r in sub.package_results:
        out.append({"division": r.division, "systemname": r.systemname, "benchmark": r.benchmark,
                    "model": r.model, "accelerator": r.accelerator, "runs": len(r.runs),
                    "run_ids": sorted(run.id for run in r.runs),
                    "leaves": sorted(run.leaf for run in r.runs)})
    return out


def write_sidecars(written: WrittenPackage, plan: PackagePlan, sub: SubmissionReadiness,
                   results_dir: str, now: _dt.datetime) -> Tuple[str, str]:
    checksum_path, manifest_path = sidecar_paths(written.path)
    basename = os.path.basename(written.path)
    with open(checksum_path, "w", encoding="utf-8") as fh:
        fh.write(f"{written.sha256}  {basename}\n")
    manifest = {
        "schema": PACKAGE_SCHEMA,
        "orgname": plan.orgname,
        "rules_edition": plan.edition,
        "created_at": now.replace(microsecond=0).isoformat(),
        "created_by": f"mlpstorage {VERSION}",
        "results_dir": os.path.abspath(results_dir),
        "package": basename,
        "sha256": written.sha256,
        "size_bytes": written.size_bytes,
        "file_count": len(written.files),
        "divisions": list(plan.divisions),
        "systems": {d: list(names) for d, names in plan.systems.items()},
        "code_images": list(plan.images),
        "results": _results_inventory(sub),
        "files": written.files,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return checksum_path, manifest_path


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def ledger_path(results_dir: str) -> str:
    """``<results-dir>/.mlps/submissions.jsonl``."""
    return os.path.join(results_dir, LEDGER_RELPATH)


def read_ledger(results_dir: str) -> List[dict]:
    path = ledger_path(results_dir)
    out: List[dict] = []
    try:
        fh = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        return out
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn line from a crash mid-append
            if isinstance(event, dict):
                out.append(event)
    return out


def record_package(results_dir: str, written: WrittenPackage, plan: PackagePlan,
                   sub: SubmissionReadiness, now: _dt.datetime) -> int:
    """Append a ``packaged`` event and return its ID (1, 2, ... per tree)."""
    events = read_ledger(results_dir)
    ids = [int(e["id"]) for e in events if isinstance(e.get("id"), int)]
    next_id = (max(ids) + 1) if ids else 1
    package = sub.package_results
    event = {
        "event": "packaged",
        "id": next_id,
        "at": now.replace(microsecond=0).isoformat(),
        "package": written.path,
        "sha256": written.sha256,
        "size_bytes": written.size_bytes,
        "file_count": len(written.files),
        "orgname": plan.orgname,
        "rules_edition": plan.edition,
        "divisions": list(plan.divisions),
        "results": len(package),
        "runs": sum(len(r.runs) for r in package),
        "tool": f"mlpstorage {VERSION}",
    }
    path = ledger_path(results_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")
    return next_id


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def _refusal_summary(sub: SubmissionReadiness) -> str:
    package = sub.package_results
    counts = {SUBMIT_SHORT: 0, SUBMIT_INVALID: 0, SUBMIT_PAPERWORK: 0}
    for r in package:
        if r.submit in counts:
            counts[r.submit] += 1
    parts = []
    if counts[SUBMIT_SHORT]:
        parts.append(f"{_plural(counts[SUBMIT_SHORT], 'result')} short")
    if counts[SUBMIT_INVALID]:
        parts.append(f"{counts[SUBMIT_INVALID]} invalid")
    if counts[SUBMIT_PAPERWORK]:
        n = counts[SUBMIT_PAPERWORK]
        parts.append(f"{n} need{'s' if n == 1 else ''} paperwork")
    if sub.tree_problems:
        parts.append(_plural(len(sub.tree_problems), "tree problem"))
    head = ", ".join(parts) if parts else "the checker reports errors"
    return f"Not submittable: {head}; {_plural(len(sub.errors), 'checker error')} in all."


def _contents_line(plan: PackagePlan, sub: SubmissionReadiness) -> str:
    parts = []
    for division in plan.divisions:
        rows = [r for r in sub.package_results if r.division == division]
        runs = sum(len(r.runs) for r in rows)
        parts.append(f"{division}/{plan.orgname}: {_plural(len(plan.systems.get(division, [])), 'system')}, "
                     f"{_plural(len(rows), 'result')}, {_plural(runs, 'run')}")
    parts.append(f"{_POOL_DIRNAME}: {_plural(len(plan.images), 'image')}")
    parts.append(f"{_plural(plan.file_count, 'file')}, {_human_size(plan.size_bytes)}")
    return "  " + "; ".join(parts)


def run_submit_command(args, results_dir: str, logger) -> int:
    """``mlpstorage submit [--dry-run] [--out PATH]`` on an initialized
    ``results_dir``. Exit 0 on a pass or a clean dry run, 1 otherwise."""
    root = os.path.abspath(results_dir)
    dry_run = bool(getattr(args, "dry_run", False))
    now = _dt.datetime.now()

    # 1. Fresh rollups and manifests, so the rules about them are judged on
    #    what the package will carry.
    logger.status("Regenerating rollup tables and submission manifests (reports reportgen)...")
    try:
        regenerate_rollups(root, logger)
    except Exception as exc:  # noqa: BLE001 -- anything here is a refusal, not a crash
        logger.error(f"Could not regenerate the rollup tables before checking: {exc}")
        print(f"Not submittable: the rollup tables could not be regenerated ({exc}); "
              f"see `mlpstorage reports reportgen -rd {results_dir}`.")
        return EXIT_CODE.GENERAL_ERROR

    # 2. One pass of the checker, scored per result.
    sub = readiness.evaluate(root, findings=readiness.collect_findings(root))
    package = sub.package_results
    if not package:
        why = " (whatif results are never packaged)" if sub.results else ""
        print(f"Nothing to submit: no closed or open results in {results_dir}{why}.")
        return EXIT_CODE.GENERAL_ERROR

    for line in render(sub, next_line=False):
        print(line)
    if not sub.submittable:
        # The table carries runs, results and paperwork; a rollup rule that
        # still fails after regeneration has no row to sit in.
        rollup = [f for f in sub.errors if f.rule_id in ROLLUP_RULES]
        if rollup:
            print("Rollup tables (regenerated, still failing):")
            for f in rollup:
                print(f"  {f.short()}")
        print(_refusal_summary(sub))
        print("Next: fix the above, then mlpstorage submit --dry-run")
        return EXIT_CODE.GENERAL_ERROR

    # 3. The plan.
    try:
        plan = plan_package(root, sub, now)
    except SubmitError as exc:
        print(f"Not submittable: {exc}.")
        print(f"Next: mlpstorage validate {results_dir}")
        return EXIT_CODE.GENERAL_ERROR
    path = package_path(results_dir, getattr(args, "out", None), sub.orgname, sub.edition, now)

    if dry_run:
        print(f"Would write {path}")
        print(_contents_line(plan, sub))
        print("Dry run: no package written. `mlpstorage submit` builds it.")
        return EXIT_CODE.SUCCESS

    # 4. Package, sidecars, ledger, instructions.
    written = write_tarball(plan, path, now)
    checksum_path, manifest_path = write_sidecars(written, plan, sub, root, now)
    submission_id = record_package(root, written, plan, sub, now)
    print(f"Package:   {written.path}  ({_human_size(written.size_bytes)}, "
          f"{_plural(len(written.files), 'file')})")
    print(f"Checksum:  {checksum_path}  (sha256 {written.sha256})")
    print(f"Manifest:  {manifest_path}")
    print(_contents_line(plan, sub))
    print(f"Recorded as submission {submission_id} in {os.path.join(results_dir, LEDGER_RELPATH)}")
    print("")
    print(UPLOAD_INSTRUCTIONS)
    return EXIT_CODE.SUCCESS
