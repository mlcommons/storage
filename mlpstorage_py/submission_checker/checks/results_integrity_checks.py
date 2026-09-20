"""Results-integrity rules: LEAF-01 and RPT-01.

Both rules exist because ``reportgen`` used to lose a run without saying so
(mlcommons/storage#835, #836). Reportgen now names every drop and keeps its
leaf and rollup tables in step; these two checks make the same conditions
fail *validation*, so a reviewer sees them at submission time rather than
by grepping a report log.

LEAF-01 ``runLeafMetadata``
    A ``kv_cache`` or ``vector_database`` ``run`` leaf must carry exactly
    one ``<type>_<timestamp>_metadata.json``. Those two benchmark types have
    no other signal that identifies the run — DLIO training and
    checkpointing leaves identify themselves through their Hydra configs, so
    the file is effectively optional for them and they are out of scope
    here. Without it reportgen cannot tell what the leaf is, drops it, and
    the workload's metric columns publish blank (#835 suggestion 3). A leaf
    with two metadata files is dropped for the same reason and fails too.

RPT-01 ``rollupTableAgreement``
    For every organization rollup table ``<mode>/<org>/results/results.json``
    the set of minted ``Public ID`` values must equal the union of minted
    IDs across every workload-level ``results.json`` under that
    organization's ``results/<system>/`` tree. #836's original shape was a
    rollup row assembled from one of several runs above a workload table
    holding zero rows; both were fixed in reportgen, and this rule fails any
    recurrence (#836 suggestion 4). Rows whose ``Public ID`` is empty are the
    identity-only datagen / datasize command tables and are ignored. An
    organization with no rollup (reportgen never ran) draws no finding —
    other rules own the presence of ``results.json``.

Tree-level, same shape as ``ProvenanceCheck`` / ``EditionCheck``: runs once
before the per-benchmark loader loop in ``main.run``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Set, Tuple

from mlpstorage_py.runs.ledger import iter_leaves, parse_leaf

from .base import BaseCheck
from ..configuration.configuration import Config
from ..rule_registry import rule

_METADATA_SUFFIX = "_metadata.json"
_RESULTS_TABLE = "results.json"
_LEAF_01_TYPES = ("kv_cache", "vector_database")
_MODES = ("closed", "open", "whatif")


class ResultsIntegrityCheck(BaseCheck):
    """LEAF-01 / RPT-01 (see module docstring)."""

    def __init__(self, log, config: Config, root_path: str):
        super().__init__(log=log, path=root_path)
        self.config = config
        self.root_path = root_path
        self.name = "results integrity checks"
        self.init_checks()

    def init_checks(self):
        self.checks = [
            self.run_leaf_metadata_check,
            self.rollup_table_agreement_check,
        ]

    # ------------------------------------------------------------------
    # LEAF-01 — runLeafMetadata
    # ------------------------------------------------------------------

    @rule("LEAF-01", "runLeafMetadata")
    def run_leaf_metadata_check(self):
        valid = True
        for rel in iter_leaves(self.root_path):
            info = parse_leaf(rel) or {}
            if info.get("benchmark") not in _LEAF_01_TYPES or info.get("command") != "run":
                continue
            leaf = os.path.join(self.root_path, rel)
            try:
                names = sorted(os.listdir(leaf))
            except OSError:
                continue
            metadata_files = [n for n in names if n.endswith(_METADATA_SUFFIX)]
            if len(metadata_files) == 1:
                continue
            valid = False
            bt = info["benchmark"]
            expected = f"{bt}_{info['run_datetime']}{_METADATA_SUFFIX}"
            if not metadata_files:
                if "summary.json" in names:
                    self.log_violation(
                        "LEAF-01", "runLeafMetadata", leaf,
                        "%s run leaf has a summary.json but no %s. Nothing else "
                        "identifies a %s run, so reportgen drops this leaf and the "
                        "workload's metric columns publish BLANK in results.csv. "
                        "Restore %s — the measured data does not need to be "
                        "regenerated. (mlcommons/storage#835)",
                        bt, _METADATA_SUFFIX, bt, expected,
                    )
                else:
                    self.log_violation(
                        "LEAF-01", "runLeafMetadata", leaf,
                        "%s run leaf has no summary.json and no %s: it is not a "
                        "completed run. Remove the directory or restore the run's "
                        "files (%s and summary.json). (mlcommons/storage#835)",
                        bt, _METADATA_SUFFIX, expected,
                    )
            else:
                self.log_violation(
                    "LEAF-01", "runLeafMetadata", leaf,
                    "%d %s files found (%s) and only one may identify a run; "
                    "reportgen drops the leaf and the workload's metric columns "
                    "publish BLANK in results.csv. Keep only %s.",
                    len(metadata_files), _METADATA_SUFFIX, ", ".join(metadata_files),
                    expected,
                )
        return valid

    # ------------------------------------------------------------------
    # RPT-01 — rollupTableAgreement
    # ------------------------------------------------------------------

    def _minted_ids(self, table_path: str) -> Tuple[Set[str], bool]:
        """(minted Public IDs, readable). Blank IDs are skipped by design."""
        try:
            with open(table_path, "r", encoding="utf-8") as fh:
                rows = json.load(fh)
        except (OSError, ValueError) as e:
            self.log_violation(
                "RPT-01", "rollupTableAgreement", table_path,
                "results table cannot be read (%s); re-run reportgen.", e,
            )
            return set(), False
        if not isinstance(rows, list):
            self.log_violation(
                "RPT-01", "rollupTableAgreement", table_path,
                "results table is not a list of rows; re-run reportgen.",
            )
            return set(), False
        ids: Set[str] = set()
        for row in rows:
            if isinstance(row, dict):
                pid = row.get("Public ID")
                if pid:
                    ids.add(str(pid))
        return ids, True

    @staticmethod
    def _workload_tables(results_root: str) -> Iterable[str]:
        """Every ``results.json`` below ``results_root/<system>/`` — the
        workload-level tables (model dir, command dir, or any intermediate
        level reportgen writes). The org rollup sits *at* ``results_root``
        and is excluded."""
        for system in sorted(os.listdir(results_root)):
            system_dir = os.path.join(results_root, system)
            if not os.path.isdir(system_dir) or system.startswith("."):
                continue
            for dirpath, dirnames, files in os.walk(system_dir):
                dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
                if _RESULTS_TABLE in files:
                    yield os.path.join(dirpath, _RESULTS_TABLE)

    @rule("RPT-01", "rollupTableAgreement")
    def rollup_table_agreement_check(self):
        valid = True
        for mode in _MODES:
            mode_dir = os.path.join(self.root_path, mode)
            if not os.path.isdir(mode_dir):
                continue
            for org in sorted(os.listdir(mode_dir)):
                results_root = os.path.join(mode_dir, org, "results")
                rollup = os.path.join(results_root, _RESULTS_TABLE)
                if not os.path.isfile(rollup):
                    continue  # reportgen never ran here; not this rule's business
                rollup_ids, ok = self._minted_ids(rollup)
                if not ok:
                    valid = False
                    continue
                leaf_ids: Set[str] = set()
                homes: Dict[str, List[str]] = {}
                for table in self._workload_tables(results_root):
                    ids, ok = self._minted_ids(table)
                    if not ok:
                        valid = False
                        continue
                    leaf_ids |= ids
                    for pid in ids:
                        homes.setdefault(pid, []).append(os.path.relpath(table, results_root))
                orphans = sorted(rollup_ids - leaf_ids)
                if orphans:
                    valid = False
                    self.log_violation(
                        "RPT-01", "rollupTableAgreement", rollup,
                        "%d rollup row%s (%s) appear%s in no workload-level "
                        "results.json under %s/. A rollup row must be backed by "
                        "the workload table it summarizes; re-run reportgen and, "
                        "if the row persists, the workload's run layout is not "
                        "one reportgen can place. (mlcommons/storage#836)",
                        len(orphans), "" if len(orphans) == 1 else "s",
                        ", ".join(orphans), "s" if len(orphans) == 1 else "",
                        os.path.relpath(results_root, self.root_path),
                    )
                missing = sorted(leaf_ids - rollup_ids)
                if missing:
                    valid = False
                    self.log_violation(
                        "RPT-01", "rollupTableAgreement", rollup,
                        "%d workload-table row%s (%s) %s missing from the rollup: %s. "
                        "Re-run reportgen so the rollup and the workload tables "
                        "agree. (mlcommons/storage#836)",
                        len(missing), "" if len(missing) == 1 else "s",
                        ", ".join(missing), "is" if len(missing) == 1 else "are",
                        "; ".join(f"{pid} in {', '.join(homes.get(pid, []))}" for pid in missing),
                    )
        return valid
