"""Public ID generation for the v3.0 results tables.

The ``Public ID`` column is no longer a manual placeholder. reportgen numbers
the published rows itself:

- Format ``v3.0-NNNN`` — a hardcoded version prefix plus a 4-digit
  zero-padded counter.
- The counter starts at 1 **after** the rows have been sorted, so IDs follow
  the established rollup sort key (category, orgname, systemname,
  benchmark_type, model, accelerator).
- IDs are **fully regenerated** on every run. Nothing is persisted between
  runs; inserting a row that sorts earlier renumbers everything after it.
- Only ``run`` rows are numbered. Auxiliary-phase rows (datagen/datasize —
  the R3 / issue-#771/#791 6-element workload keys) are excluded from the
  rollups and carry a blank Public ID in their per-phase leaf files.

Cross-table identity is the load-bearing property: because a row dict is
shared by reference between its per-model table, its per-org rollup and the
global rollup, one assignment pass gives the same row the same ID everywhere.
Numbering each file independently would give one row three different IDs.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
from argparse import Namespace

import pytest

from mlpstorage_py.report_generator import ReportGenerator

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_FIXTURES_ROOT = _REPO_ROOT / "tests" / "fixtures" / "sample_results"

_PUBLIC_ID_RE = re.compile(r"^v3\.0-\d{4}$")


def _prepare_tree(tmp_path: pathlib.Path, name: str = "repo_root") -> pathlib.Path:
    dest = tmp_path / name
    shutil.copytree(_FIXTURES_ROOT / "multi_orgname", dest)
    return dest


def _run(root: pathlib.Path) -> None:
    gen = ReportGenerator(str(root), args=Namespace(debug=False),
                          validate_structure=False)
    rc = gen.generate_reports()
    assert rc == 0, f"generate_reports returned {rc}"


def _rows(path: pathlib.Path) -> list:
    return json.loads(path.read_text())


def _global_rows(root: pathlib.Path) -> list:
    return _rows(root / "results.json")


def _org_rollup_rows(root: pathlib.Path, org: str) -> list:
    return _rows(root / "closed" / org / "results" / "results.json")


def _per_model_rows(root: pathlib.Path, org: str, system: str) -> list:
    return _rows(root / "closed" / org / "results" / system
                 / "training" / "unet3d" / "run" / "results.json")


class TestPublicIdFormat:
    def test_ids_match_v3_0_four_digit_format(self, tmp_path):
        root = _prepare_tree(tmp_path)
        _run(root)
        for row in _global_rows(root):
            assert _PUBLIC_ID_RE.match(row["Public ID"]), (
                f"malformed Public ID {row['Public ID']!r}"
            )

    def test_numbering_starts_at_one_and_is_contiguous(self, tmp_path):
        root = _prepare_tree(tmp_path)
        _run(root)
        ids = [r["Public ID"] for r in _global_rows(root)]
        assert ids == [f"v3.0-{i:04d}" for i in range(1, len(ids) + 1)]

    def test_ids_follow_the_rollup_sort_order(self, tmp_path):
        """acme sorts before beta_corp, so acme takes 0001."""
        root = _prepare_tree(tmp_path)
        _run(root)
        by_org = {r["Organization"]: r["Public ID"] for r in _global_rows(root)}
        assert by_org == {"acme": "v3.0-0001", "beta_corp": "v3.0-0002"}


class TestCrossTableIdentity:
    def test_same_row_carries_same_id_in_every_table(self, tmp_path):
        """A row's ID must agree across per-model, per-org and global tables."""
        root = _prepare_tree(tmp_path)
        _run(root)
        global_by_org = {
            r["Organization"]: r["Public ID"] for r in _global_rows(root)
        }
        for org, system in (("acme", "system-a"), ("beta_corp", "system-b")):
            per_model = _per_model_rows(root, org, system)
            assert len(per_model) == 1
            org_rollup = _org_rollup_rows(root, org)
            assert len(org_rollup) == 1
            assert per_model[0]["Public ID"] == global_by_org[org]
            assert org_rollup[0]["Public ID"] == global_by_org[org]

    def test_per_model_tables_are_not_numbered_independently(self, tmp_path):
        """Each per-model table holds one row; numbering them per-file would
        make every one of them 'v3.0-0001'."""
        root = _prepare_tree(tmp_path)
        _run(root)
        acme = _per_model_rows(root, "acme", "system-a")[0]["Public ID"]
        beta = _per_model_rows(root, "beta_corp", "system-b")[0]["Public ID"]
        assert acme != beta, (
            "per-model tables were numbered independently — both rows got "
            f"{acme!r}"
        )


class TestFullRegeneration:
    def test_rerunning_produces_identical_ids(self, tmp_path):
        root = _prepare_tree(tmp_path)
        _run(root)
        first = {r["Organization"]: r["Public ID"] for r in _global_rows(root)}
        _run(root)
        second = {r["Organization"]: r["Public ID"] for r in _global_rows(root)}
        assert first == second

    def test_inserting_an_earlier_org_renumbers(self, tmp_path):
        """IDs are positional, not sticky: a new org sorting before acme
        pushes acme from 0001 to 0002."""
        root = _prepare_tree(tmp_path)
        _run(root)
        assert _per_model_rows(root, "acme", "system-a")[0]["Public ID"] == (
            "v3.0-0001"
        )
        # Clone acme's whole org tree under a name that sorts first.
        shutil.copytree(root / "closed" / "acme", root / "closed" / "aaa_corp")
        _run(root)
        ids = [r["Public ID"] for r in _global_rows(root)]
        assert ids == [f"v3.0-{i:04d}" for i in range(1, len(ids) + 1)]
        by_org = {r["Organization"]: r["Public ID"] for r in _global_rows(root)}
        assert by_org["aaa_corp"] == "v3.0-0001"
        assert by_org["acme"] == "v3.0-0002"
        assert by_org["beta_corp"] == "v3.0-0003"


class TestOnlyRunRowsAreNumbered:
    def _tree_with_datagen(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Clone acme's run invocation into a datagen phase directory.

        Mirrors the R3 helper in test_reportgen_output_shape.py: same
        metadata shape, command='datagen', no accelerator, no summary.json.
        """
        dest = _prepare_tree(tmp_path)
        run_dir = (dest / "closed" / "acme" / "results" / "system-a"
                   / "training" / "unet3d" / "run" / "20260706_100000")
        datagen_dir = (dest / "closed" / "acme" / "results" / "system-a"
                       / "training" / "unet3d" / "datagen" / "20260706_090000")
        datagen_dir.mkdir(parents=True)
        metadata = json.loads(
            (run_dir / "training_unet3d_metadata.json").read_text()
        )
        metadata["command"] = "datagen"
        metadata["accelerator"] = None
        metadata["run_datetime"] = "20260706_090000"
        (datagen_dir / "training_unet3d_metadata.json").write_text(
            json.dumps(metadata)
        )
        return dest

    def test_datagen_leaf_row_has_blank_public_id(self, tmp_path):
        root = self._tree_with_datagen(tmp_path)
        _run(root)
        datagen_rows = _rows(
            root / "closed" / "acme" / "results" / "system-a"
            / "training" / "unet3d" / "datagen" / "results.json"
        )
        assert datagen_rows, "expected a per-phase leaf row for datagen"
        for row in datagen_rows:
            assert row["Public ID"] == "", (
                f"auxiliary row should not be numbered, got {row['Public ID']!r}"
            )

    def test_auxiliary_rows_do_not_consume_numbers(self, tmp_path):
        """The datagen row must not shift beta_corp's ID."""
        root = self._tree_with_datagen(tmp_path)
        _run(root)
        by_org = {r["Organization"]: r["Public ID"] for r in _global_rows(root)}
        assert by_org == {"acme": "v3.0-0001", "beta_corp": "v3.0-0002"}
