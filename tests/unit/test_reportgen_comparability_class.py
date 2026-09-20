"""results.csv carries the rules edition and the comparability class.

The unified results archive (one table for every round) needs two columns
the v3.0 webpage tables never had: the *rules edition* a row was produced
under (the table's version column) and the *comparability class* the
working group declared for the workload it ran (the only key on which rows
compare, within a round or across rounds). Both are looked up at report
time from each run leaf's provenance stamp and ``editions.yaml`` -- the
class is never written into a leaf or manifest (PR #856 D-5).

Row semantics (one row aggregates several ``run`` leaves):

- ``Rules Edition``: the edition every contributing run leaf declares. An
  unstamped leaf falls back to its submission's ``submission.yaml``
  (PROV-02 already forces the two to agree). Blank when unknown or when
  the leaves disagree.
- ``Comparability Class``: the one class every contributing run leaf
  falls into. Blank when no leaf classifies (OPEN variants, unknown
  hashes), and blank *with a warning* when the leaves land in different
  classes or only some of them classify -- such a row is not comparable.
- datagen / datasize leaves never contribute; division is part of the
  class key (an OPEN run using a CLOSED configuration is unclassified).
"""

from __future__ import annotations

import csv
import json
import logging
import pathlib
import shutil
from argparse import Namespace
from types import SimpleNamespace

import pytest

from mlpstorage_py.config import BENCHMARK_TYPES
from mlpstorage_py.provenance import (
    RunProvenance, compute_core_config, write_leaf_provenance,
    write_submission_manifest,
)
from mlpstorage_py.report_generator import ReportGenerator, _FINAL_SCHEMA

from tests.unit.test_leaf_provenance import UNET3D_B200, _stamp  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "sample_results" / "multi_orgname"

UNET3D_HASH = compute_core_config(UNET3D_B200, "training")["hash"]
_VARIANT = json.loads(json.dumps(UNET3D_B200))
_VARIANT["reader"]["batch_size"] = 8
VARIANT_HASH = compute_core_config(_VARIANT, "training")["hash"]

ACME_LEAF = "closed/acme/results/system-a/training/unet3d/run/20260706_100000"
BETA_LEAF = "closed/beta_corp/results/system-b/training/unet3d/run/20260706_101500"


def _training_stamp(core_hash: str, **over) -> RunProvenance:
    d = _stamp(core_config={"algorithm": "core-config-v1", "allowlist": "training@1",
                            "hash": core_hash, "keys": ["framework"]})
    d.update(over)
    return RunProvenance.from_dict(d)


def _prepare_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    dest = tmp_path / "repo_root"
    shutil.copytree(_FIXTURE, dest)
    for org, system in (("acme", "system-a"), ("beta_corp", "system-b")):
        systems = dest / "closed" / org / "systems"
        systems.mkdir(parents=True, exist_ok=True)
        (systems / f"{system}.yaml").write_text("system_under_test: {}\n")
    return dest


def _set_metadata(leaf: pathlib.Path, **fields) -> None:
    path = next(leaf.glob("*_metadata.json"))
    md = json.loads(path.read_text())
    md.update(fields)
    path.write_text(json.dumps(md, indent=2))


def _run_reportgen(root: pathlib.Path) -> ReportGenerator:
    gen = ReportGenerator(str(root), args=Namespace(debug=False), validate_structure=False)
    assert gen.generate_reports() == 0
    return gen


def _rows_by_org(root: pathlib.Path):
    rows = json.loads((root / "results.json").read_text())
    return {r["Organization"]: r for r in rows}


def _run(leaf: pathlib.Path, *, command="run", model="unet3d", accelerator="b200",
         family=BENCHMARK_TYPES.training, parameters=None):
    return SimpleNamespace(result_dir=str(leaf), command=command, model=model,
                           accelerator=accelerator, benchmark_type=family,
                           parameters=parameters or {})


def _bare_generator(tmp_path: pathlib.Path, root: pathlib.Path | None = None) -> ReportGenerator:
    from unittest.mock import patch
    results_dir = root if root is not None else tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    with patch.object(ReportGenerator, "accumulate_results"):
        with patch.object(ReportGenerator, "print_results"):
            return ReportGenerator(str(results_dir), validate_structure=False)


class TestSchema:
    def test_edition_and_class_follow_model(self):
        i = _FINAL_SCHEMA.index("Model")
        assert _FINAL_SCHEMA[i + 1] == "Rules Edition"
        assert _FINAL_SCHEMA[i + 2] == "Comparability Class"
        assert _FINAL_SCHEMA[i + 3] == "Name"


class TestEndToEnd:
    def test_stamped_closed_run_carries_edition_and_class(self, tmp_path):
        root = _prepare_tree(tmp_path)
        acme = root / ACME_LEAF
        _set_metadata(acme, accelerator="b200")
        write_leaf_provenance(acme, _training_stamp(UNET3D_HASH))
        _run_reportgen(root)

        rows = _rows_by_org(root)
        assert rows["acme"]["Rules Edition"] == "3.0"
        assert rows["acme"]["Comparability Class"] == "unet3d-b200-A"
        # beta_corp is unstamped, has no manifest and its fixture parameters
        # hash to no class: both cells stay blank rather than guessed.
        assert rows["beta_corp"]["Rules Edition"] == ""
        assert rows["beta_corp"]["Comparability Class"] == ""

        with open(root / "results.csv", newline="") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames
            csv_rows = {r["Organization"]: r for r in reader}
        assert header == _FINAL_SCHEMA
        assert csv_rows["acme"]["Rules Edition"] == "3.0"
        assert csv_rows["acme"]["Comparability Class"] == "unet3d-b200-A"

    def test_unstamped_leaf_takes_edition_from_manifest_and_class_from_derived_hash(self, tmp_path):
        root = _prepare_tree(tmp_path)
        acme = root / ACME_LEAF
        _set_metadata(acme, accelerator="b200", parameters=UNET3D_B200)
        write_submission_manifest(root, "closed", "acme")
        _run_reportgen(root)

        rows = _rows_by_org(root)
        assert rows["acme"]["Rules Edition"] == "3.0"
        assert rows["acme"]["Comparability Class"] == "unet3d-b200-A"

    def test_unstamped_leaf_without_manifest_matches_class_by_hash_only(self, tmp_path):
        root = _prepare_tree(tmp_path)
        acme = root / ACME_LEAF
        _set_metadata(acme, accelerator="b200", parameters=UNET3D_B200)
        _run_reportgen(root)

        rows = _rows_by_org(root)
        assert rows["acme"]["Rules Edition"] == ""
        assert rows["acme"]["Comparability Class"] == "unet3d-b200-A"


class TestRowRollup:
    def test_all_leaves_in_one_class(self, tmp_path):
        root = _prepare_tree(tmp_path)
        leaves = []
        for ts in ("20260706_100000", "20260707_100000"):
            leaf = root / ACME_LEAF.replace("20260706_100000", ts)
            leaf.mkdir(parents=True, exist_ok=True)
            write_leaf_provenance(leaf, _training_stamp(UNET3D_HASH))
            leaves.append(leaf)
        gen = _bare_generator(tmp_path, root)
        cols = gen._edition_columns("closed", [_run(l) for l in leaves])
        assert cols == {"rules_edition": "3.0", "comparability_class": "unet3d-b200-A"}

    def test_mixed_classes_blank_with_warning(self, tmp_path, caplog):
        root = _prepare_tree(tmp_path)
        a = root / ACME_LEAF
        b = root / ACME_LEAF.replace("20260706_100000", "20260707_100000")
        b.mkdir(parents=True, exist_ok=True)
        write_leaf_provenance(a, _training_stamp(UNET3D_HASH))
        write_leaf_provenance(b, _training_stamp(VARIANT_HASH))
        gen = _bare_generator(tmp_path, root)
        with caplog.at_level(logging.WARNING, logger=gen.logger.name):
            cols = gen._edition_columns("closed", [_run(a), _run(b)])
        assert cols == {"rules_edition": "3.0", "comparability_class": ""}
        assert any("comparability class" in r.getMessage().lower() for r in caplog.records)

    def test_mixed_editions_blank_edition(self, tmp_path, caplog):
        root = _prepare_tree(tmp_path)
        a = root / ACME_LEAF
        b = root / ACME_LEAF.replace("20260706_100000", "20260707_100000")
        b.mkdir(parents=True, exist_ok=True)
        write_leaf_provenance(a, _training_stamp(UNET3D_HASH))
        write_leaf_provenance(b, _training_stamp(UNET3D_HASH, rules_edition="2.0"))
        gen = _bare_generator(tmp_path, root)
        with caplog.at_level(logging.WARNING, logger=gen.logger.name):
            cols = gen._edition_columns("closed", [_run(a), _run(b)])
        assert cols["rules_edition"] == ""
        assert any("rules edition" in r.getMessage().lower() for r in caplog.records)

    def test_open_run_with_closed_hash_is_unclassified(self, tmp_path):
        root = _prepare_tree(tmp_path)
        leaf = root / ACME_LEAF.replace("closed/", "open/")
        leaf.mkdir(parents=True, exist_ok=True)
        write_leaf_provenance(leaf, _training_stamp(UNET3D_HASH))
        gen = _bare_generator(tmp_path, root)
        cols = gen._edition_columns("open", [_run(leaf)])
        assert cols == {"rules_edition": "3.0", "comparability_class": ""}

    def test_accelerator_is_part_of_the_key(self, tmp_path):
        root = _prepare_tree(tmp_path)
        leaf = root / ACME_LEAF
        write_leaf_provenance(leaf, _training_stamp(UNET3D_HASH))
        gen = _bare_generator(tmp_path, root)
        cols = gen._edition_columns("closed", [_run(leaf, accelerator="h100")])
        assert cols == {"rules_edition": "3.0", "comparability_class": ""}

    def test_auxiliary_leaves_do_not_contribute(self, tmp_path):
        root = _prepare_tree(tmp_path)
        leaf = root / ACME_LEAF.replace("/run/", "/datagen/")
        leaf.mkdir(parents=True, exist_ok=True)
        write_leaf_provenance(leaf, _training_stamp(UNET3D_HASH))
        gen = _bare_generator(tmp_path, root)
        cols = gen._edition_columns("closed", [_run(leaf, command="datagen")])
        assert cols == {"rules_edition": "", "comparability_class": ""}

    def test_missing_leaf_dir_is_blank_not_fatal(self, tmp_path):
        gen = _bare_generator(tmp_path)
        cols = gen._edition_columns("closed", [_run(tmp_path / "nowhere")])
        assert cols == {"rules_edition": "", "comparability_class": ""}
        cols = gen._edition_columns("closed", [SimpleNamespace(result_dir=None, command="run")])
        assert cols == {"rules_edition": "", "comparability_class": ""}

    def test_vdb_model_token_is_engine_slash_index(self, tmp_path):
        """The class key's vdb model is the leaf's ``<engine>/<index>`` path
        token, exactly as the validator (EDN-02) and ``runs show`` spell it."""
        from mlpstorage_py.editions import load_editions
        vdb_class = next(c for c in load_editions().classes if c.id == "milvus-diskann-A")
        root = _prepare_tree(tmp_path)
        leaf = root / "closed/acme/results/system-a/vector_database/milvus/DISKANN/run/20260706_100000"
        leaf.mkdir(parents=True, exist_ok=True)
        d = _stamp(core_config={"algorithm": "core-config-v1", "allowlist": vdb_class.allowlist,
                                "hash": vdb_class.core_configs[0], "keys": ["database.database"]})
        write_leaf_provenance(leaf, RunProvenance.from_dict(d))
        gen = _bare_generator(tmp_path, root)
        cols = gen._edition_columns("closed", [_run(leaf, model="", accelerator="",
                                                    family=BENCHMARK_TYPES.vector_database,
                                                    parameters={"engine": "milvus", "index_type": "DISKANN"})])
        assert cols == {"rules_edition": "3.0", "comparability_class": "milvus-diskann-A"}
