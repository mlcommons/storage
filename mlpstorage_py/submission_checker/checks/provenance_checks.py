"""ProvenanceCheck — PROV-01 leafProvenance and PROV-02 submissionManifest.

Runs as a pre-loop check in ``main.py:run()`` after ``PoolStructureCheck``.
Both rules verify the files ``mlpstorage_py/provenance.py`` writes:

- PROV-01: a run leaf's ``provenance.json`` must parse, and its ``tool.code_image``
  must agree with the leaf's ``.mlps-code-image`` pointer. A leaf whose
  ``*_metadata.json`` declares the sidecar (``provenance_file``) but has none
  is an error. A leaf that never declared one predates stamping and draws no
  line at all — the frozen v3.0 tree validates byte-identically.
- PROV-02: ``<mode>/<org>/submission.yaml``, when present, must parse, must
  list exactly the run leaves the tree holds, and its ``rules_edition`` must
  match every stamped leaf. A tree without manifests draws no line.

Accumulate-don't-abort: every violation in the tree is reported before the
method returns ``False``; nothing raises out of a rule body. Messages use the
locked ``[<rule_id> <rule_name>] <path>: <msg>`` format via ``log_violation``.
"""

from __future__ import annotations

import os
from pathlib import Path

from mlpstorage_py.provenance import (
    MANIFEST_FILENAME,
    METADATA_PROVENANCE_KEY,
    PROVENANCE_FILENAME,
    ProvenanceError,
    _load_stamp_file,
    _MODES,
    has_leaf_provenance,
    pointer_hash,
    read_submission_manifest,
)
from mlpstorage_py.runs.ledger import iter_leaves, read_metadata

from .base import BaseCheck
from ..configuration.configuration import Config
from ..rule_registry import rule


class ProvenanceCheck(BaseCheck):
    """Validate per-leaf provenance stamps and per-organization manifests."""

    def __init__(self, log, config: Config, root_path: str):
        super().__init__(log=log, path=root_path)
        self.config = config
        self.root_path = root_path
        self.name = "provenance checks"
        self.init_checks()

    def init_checks(self):
        self.checks = [
            self.leaf_provenance_check,
            self.submission_manifest_check,
        ]

    # ------------------------------------------------------------------
    # PROV-01 — leafProvenance
    # ------------------------------------------------------------------

    @rule("PROV-01", "leafProvenance")
    def leaf_provenance_check(self):
        valid = True
        root = Path(self.root_path)
        for rel in iter_leaves(self.root_path):
            leaf = root / rel
            sidecar = leaf / PROVENANCE_FILENAME
            if not sidecar.is_file():
                metadata = read_metadata(str(leaf)) or {}
                if metadata.get(METADATA_PROVENANCE_KEY):
                    self.log_violation(
                        "PROV-01", "leafProvenance", str(leaf),
                        "run leaf %s: its metadata declares a %s sidecar "
                        "(%s = %r) but the file is absent.",
                        rel, PROVENANCE_FILENAME, METADATA_PROVENANCE_KEY,
                        metadata.get(METADATA_PROVENANCE_KEY),
                    )
                    valid = False
                # Undeclared and absent: the leaf predates stamping. Silent.
                continue
            try:
                stamp = _load_stamp_file(sidecar)
            except ProvenanceError as e:
                self.log_violation("PROV-01", "leafProvenance", str(leaf), "%s", str(e))
                valid = False
                continue
            pointed = pointer_hash(leaf)
            if pointed is not None:
                expected = f"md5-tree-v2:{pointed}"
                if stamp.tool.get("code_image") != expected:
                    self.log_violation(
                        "PROV-01", "leafProvenance", str(leaf),
                        "run leaf %s: %s names code image %s but .mlps-code-image "
                        "points at %s.",
                        rel, PROVENANCE_FILENAME, stamp.tool.get("code_image"), expected,
                    )
                    valid = False
        return valid

    # ------------------------------------------------------------------
    # PROV-02 — submissionManifest
    # ------------------------------------------------------------------

    @rule("PROV-02", "submissionManifest")
    def submission_manifest_check(self):
        valid = True
        root = Path(self.root_path)
        all_leaves = iter_leaves(self.root_path)
        for mode in _MODES:
            mode_dir = root / mode
            if not mode_dir.is_dir():
                continue
            try:
                orgs = sorted(p for p in os.listdir(mode_dir) if (mode_dir / p).is_dir())
            except OSError:
                continue
            for org in orgs:
                manifest_path = mode_dir / org / MANIFEST_FILENAME
                if not manifest_path.is_file():
                    continue  # no manifest: the tree predates them. Silent.
                try:
                    manifest = read_submission_manifest(manifest_path)
                except ProvenanceError as e:
                    self.log_violation("PROV-02", "submissionManifest",
                                       str(manifest_path), "%s", str(e))
                    valid = False
                    continue
                org_prefix = f"{mode}/{org}/"
                walked = {rel[len(org_prefix):] for rel in all_leaves
                          if rel.startswith(org_prefix + "results/")}
                listed = {r.get("leaf") for r in (manifest.get("runs") or [])
                          if isinstance(r, dict)}
                if listed != walked:
                    missing = sorted(walked - listed)
                    stale = sorted(listed - walked)
                    self.log_violation(
                        "PROV-02", "submissionManifest", str(manifest_path),
                        "%s lists %d run leaves but the tree holds %d "
                        "(not in manifest: %s; listed but absent: %s). Re-run "
                        "`mlpstorage reports reportgen` to refresh it.",
                        MANIFEST_FILENAME, len(listed), len(walked),
                        ", ".join(missing) or "-", ", ".join(str(s) for s in stale) or "-",
                    )
                    valid = False
                declared = manifest.get("rules_edition")
                for rel_org in sorted(walked):
                    leaf = mode_dir / org / rel_org
                    if not has_leaf_provenance(leaf):
                        continue  # derived stamps carry no edition (PROV-01 owns the file)
                    try:
                        stamp = _load_stamp_file(leaf / PROVENANCE_FILENAME)
                    except ProvenanceError:
                        continue
                    if stamp.rules_edition != declared:
                        self.log_violation(
                            "PROV-02", "submissionManifest", str(leaf),
                            "run leaf %s is stamped rules edition %s but %s declares %s.",
                            rel_org, stamp.rules_edition, MANIFEST_FILENAME, declared,
                        )
                        valid = False
        return valid
