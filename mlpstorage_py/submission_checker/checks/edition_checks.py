"""EditionCheck — EDN-01 rulesEdition, EDN-02 comparabilityClass, EDN-03 dlioRevision,
EDN-04 checkableEdition.

Runs as a pre-loop check in ``main.py:run()`` after ``ProvenanceCheck``.
All three rules read the leaf ``provenance.json`` stamps and the
``<mode>/<org>/submission.yaml`` manifests against the rules editions table
(``mlpstorage_py/rules/editions.yaml``, loader ``mlpstorage_py/editions.py``):

- EDN-01: a stamped leaf, or a manifest, declaring a rules edition the table
  does not know is an ERROR.
- EDN-02: a stamped ``run`` leaf with a known edition and a core-config hash
  that resolves to no comparability class for its (edition, division, family,
  model, emulated accelerator) is an ERROR under ``closed/`` and INFO elsewhere.
  datasize/datagen leaves and families without an allowlist (hash
  ``unknown``) are skipped.
- EDN-03: a stamped ``run`` leaf whose DLIO commit is not in its edition's
  accepted list is a WARNING (``unknown`` commits are skipped).

Leaves without a stamp (derived provenance, rules edition ``unknown``) draw
no line at all, so the frozen v3.0 tree validates byte-identically.
Accumulate-don't-abort; messages use the locked
``[<rule_id> <rule_name>] <path>: <msg>`` format.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from mlpstorage_py.editions import EditionsError, EditionsTable, load_editions
from mlpstorage_py.provenance import (
    MANIFEST_FILENAME,
    PROVENANCE_FILENAME,
    UNKNOWN,
    ProvenanceError,
    RunProvenance,
    _load_stamp_file,
    _MODES,
    has_leaf_provenance,
    read_submission_manifest,
)
from mlpstorage_py.runs.ledger import iter_leaves, parse_leaf, read_metadata

from .base import BaseCheck
from ..configuration.configuration import Config
from ..rule_registry import rule

_RULE_FILE = "mlpstorage_py/rules/editions.yaml"


class EditionCheck(BaseCheck):
    """Validate declared rules editions and comparability classes."""

    def __init__(self, log, config: Config, root_path: str):
        super().__init__(log=log, path=root_path)
        self.config = config
        self.root_path = root_path
        self.name = "rules edition checks"
        self._table: Optional[EditionsTable] = None
        self._table_error: Optional[str] = None
        self._stamps: Optional[Dict[str, Tuple[RunProvenance, dict, dict]]] = None
        self.init_checks()

    def init_checks(self):
        self.checks = [
            self.rules_edition_check,
            self.checkable_edition_check,
            self.comparability_class_check,
            self.dlio_revision_check,
        ]

    # ------------------------------------------------------------------
    # shared state
    # ------------------------------------------------------------------

    def _load_table(self) -> Optional[EditionsTable]:
        if self._table is None and self._table_error is None:
            try:
                self._table = load_editions()
            except EditionsError as e:
                self._table_error = str(e)
        return self._table

    def _stamped_leaves(self) -> Dict[str, Tuple[RunProvenance, dict, dict]]:
        """rel -> (stamp, parse_leaf info, metadata) for every leaf that carries a
        parseable ``provenance.json`` (PROV-01 owns malformed ones)."""
        if self._stamps is not None:
            return self._stamps
        root = Path(self.root_path)
        found: Dict[str, Tuple[RunProvenance, dict, dict]] = {}
        for rel in iter_leaves(self.root_path):
            leaf = root / rel
            if not has_leaf_provenance(leaf):
                continue
            try:
                stamp = _load_stamp_file(leaf / PROVENANCE_FILENAME)
            except ProvenanceError:
                continue
            found[rel] = (stamp, parse_leaf(rel) or {}, read_metadata(str(leaf)) or {})
        self._stamps = found
        return found

    @staticmethod
    def _accelerator(metadata: dict) -> str:
        acc = metadata.get("accelerator")
        return acc if isinstance(acc, str) and acc else UNKNOWN

    # ------------------------------------------------------------------
    # EDN-01 — rulesEdition
    # ------------------------------------------------------------------

    @rule("EDN-01", "rulesEdition")
    def rules_edition_check(self):
        valid = True
        table = self._load_table()
        if table is None:
            self.log_violation("EDN-01", "rulesEdition", self.root_path,
                               "cannot load the rules editions table: %s", self._table_error)
            return False
        root = Path(self.root_path)
        for rel, (stamp, _info, _md) in sorted(self._stamped_leaves().items()):
            edition = stamp.rules_edition
            if edition == UNKNOWN or edition in table.editions:
                continue
            self.log_violation(
                "EDN-01", "rulesEdition", str(root / rel),
                "run leaf %s: %s declares rules edition %s, which %s does not know "
                "(known: %s).", rel, PROVENANCE_FILENAME, edition, _RULE_FILE,
                ", ".join(sorted(table.editions)))
            valid = False
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
                    continue
                try:
                    manifest = read_submission_manifest(manifest_path)
                except ProvenanceError:
                    continue  # PROV-02 owns the file
                declared = manifest.get("rules_edition")
                declared = str(declared) if declared is not None else UNKNOWN
                if declared == UNKNOWN or declared in table.editions:
                    continue
                self.log_violation(
                    "EDN-01", "rulesEdition", str(manifest_path),
                    "%s declares rules edition %s, which %s does not know (known: %s).",
                    MANIFEST_FILENAME, declared, _RULE_FILE, ", ".join(sorted(table.editions)))
                valid = False
        return valid

    # ------------------------------------------------------------------
    # EDN-04 — checkableEdition
    # ------------------------------------------------------------------

    @rule("EDN-04", "checkableEdition")
    def checkable_edition_check(self):
        """A manifest declaring an edition the table knows but lists without
        checker parameters: this tool cannot check the submission, so
        ``main.run`` skips its workload checks (EDN-01 owns unknown editions)."""
        valid = True
        table = self._load_table()
        if table is None:
            return True  # EDN-01 reported the unloadable table
        root = Path(self.root_path)
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
                    continue
                try:
                    manifest = read_submission_manifest(manifest_path)
                except ProvenanceError:
                    continue  # PROV-02 owns the file
                declared = manifest.get("rules_edition")
                declared = str(declared) if declared is not None else UNKNOWN
                edition = table.edition(declared)
                if edition is None or edition.checkable:
                    continue
                self.log_violation(
                    "EDN-04", "checkableEdition", str(manifest_path),
                    "%s declares rules edition %s, which this tool cannot check (%s lists it "
                    "without checker parameters; it is checked by its own tool: %s); the "
                    "submission's workload checks are skipped.",
                    MANIFEST_FILENAME, declared, _RULE_FILE, edition.tool)
                valid = False
        return valid

    # ------------------------------------------------------------------
    # EDN-02 — comparabilityClass
    # ------------------------------------------------------------------

    @rule("EDN-02", "comparabilityClass")
    def comparability_class_check(self):
        valid = True
        table = self._load_table()
        if table is None:
            return True  # EDN-01 already reported the table problem
        root = Path(self.root_path)
        for rel, (stamp, info, metadata) in sorted(self._stamped_leaves().items()):
            edition = stamp.rules_edition
            if edition == UNKNOWN or edition not in table.editions:
                continue  # EDN-01 territory
            if info.get("command") != "run":
                continue
            core = (stamp.core_config or {}).get("hash", UNKNOWN)
            if not isinstance(core, str) or core == UNKNOWN:
                continue  # the leaf recorded no hashable workload (e.g. a pre-allowlist kv_cache / vector_database leaf)
            family, model = info.get("benchmark", UNKNOWN), info.get("model", UNKNOWN)
            accelerator = self._accelerator(metadata)
            division = info.get("mode", UNKNOWN)
            if table.classify(division=division, family=family, model=model, accelerator=accelerator,
                              core_config=core, edition=edition) is not None:
                continue
            msg = ("run leaf %s: no comparability class in %s for rules edition %s, "
                   "%s %s/%s on %s with core-config %s (allowlist %s); the run is not the "
                   "sanctioned workload, or the table needs a new class.")
            args = (rel, _RULE_FILE, edition, division, family, model, accelerator, core,
                    (stamp.core_config or {}).get("allowlist", UNKNOWN))
            if info.get("mode") == "closed":
                self.log_violation("EDN-02", "comparabilityClass", str(root / rel), msg, *args)
                valid = False
            else:
                self.info_violation("EDN-02", "comparabilityClass", str(root / rel), msg, *args)
        return valid

    # ------------------------------------------------------------------
    # EDN-03 — dlioRevision
    # ------------------------------------------------------------------

    @rule("EDN-03", "dlioRevision")
    def dlio_revision_check(self):
        table = self._load_table()
        if table is None:
            return True
        root = Path(self.root_path)
        for rel, (stamp, info, _md) in sorted(self._stamped_leaves().items()):
            edition = stamp.rules_edition
            if edition == UNKNOWN or edition not in table.editions:
                continue
            if info.get("command") != "run":
                continue
            commit = (stamp.dlio or {}).get("commit", UNKNOWN)
            if not isinstance(commit, str) or commit == UNKNOWN:
                continue
            if table.accepts_dlio(edition, commit):
                continue
            self.warn_violation(
                "EDN-03", "dlioRevision", str(root / rel),
                "run leaf %s ran DLIO commit %s (%s), which rules edition %s has not "
                "listed as an accepted revision in %s.",
                rel, commit[:8], (stamp.dlio or {}).get("source", UNKNOWN), edition, _RULE_FILE)
        return True
