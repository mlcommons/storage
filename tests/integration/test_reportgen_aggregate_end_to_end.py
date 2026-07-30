"""
Phase 6 end-to-end integration test for the score-aggregation pipeline.

Exercises the FULL production ReportGenerator pipeline (constructor
runs accumulate_results + print_results, no patches on the writer /
aggregation helper / grouping key derivation) against canonical
multi-benchmark-type fixture trees copied under tmp_path.

Concerns pinned here (see .planning/phases/06-score-aggregation/06-CONTEXT.md):

- D-02: Bottom-up build. Per-model results.{csv,json} at each
        <...>/<model>/ path IS the source of truth; top-level file
        is a strict collection of every per-model row.
- D-03: Reconstruct from scratch on every reportgen invocation.
        Deleted run subdirs disappear from the next report (canonical
        always-reconstruct assertion). Empty-model dirs still get
        results.{csv,json} regenerated: CSV = header row only,
        JSON = [].
- D-04: Unrelated files under <results-dir> are preserved. Reportgen
        only writes to per-model results.{csv,json} and top-level
        results.{csv,json} paths — never prunes.

Style precedent
---------------
- Fixture planting: shutil.copytree from tests/fixtures/sample_results/
  into tmp_path. The delete step uses shutil.rmtree.
- Full-pipeline construction: ReportGenerator(str(results_dir),
  args=argparse.Namespace(debug=False)) with NO patch.object on
  accumulate_results or print_results. This is the writer-path +
  aggregator-path integration surface — patching either off would
  defeat the test's purpose.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import shutil
from argparse import Namespace
from typing import Any, Dict, List, Set, Tuple

import pytest

from mlpstorage_py.report_generator import ReportGenerator


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_FIXTURES_ROOT = _REPO_ROOT / "tests" / "fixtures" / "sample_results"


# --------------------------------------------------------------------------- #
# Fixture planting helpers                                                    #
# --------------------------------------------------------------------------- #


def _plant_multi_orgname(results_root: pathlib.Path) -> None:
    """Copy multi_orgname fixture flattened into a canonical tree.

    Source layout (fixture):
        multi_orgname/closed/<org>/results/<sys>/training/unet3d/run/<ts>/

    We copy the `closed/` subtree directly beneath `results_root` so the
    canonical-tree resolver recognizes it and orgname derivation via
    _derive_orgname_from_path picks up both orgs.
    """
    src = _FIXTURES_ROOT / "multi_orgname"
    for div_dir in src.iterdir():
        if div_dir.is_dir():
            shutil.copytree(div_dir, results_root / div_dir.name)


def _load_json(path: pathlib.Path) -> Any:
    """Load JSON from path."""
    with open(path, "r") as fh:
        return json.load(fh)


def _read_csv_header(path: pathlib.Path) -> List[str]:
    """Return the CSV header row from path."""
    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh)
        return next(reader)


def _row_identity(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Extract a stable identity tuple from a fixed-schema row dict.

    Column-parity: the emitted file no longer carries the machine-key
    identity prefix. Identity now reads the fixed-schema discriminator
    columns (Division / Organization / Benchmark Type / Model). systemname
    and accelerator are no longer standalone emitted columns.
    """
    return (
        str(row.get('Division', '') or ''),
        str(row.get('Organization', '') or ''),
        str(row.get('Benchmark Type', '') or ''),
        str(row.get('Model', '') or ''),
    )


# --------------------------------------------------------------------------- #
# TestBottomUpBuild — D-02                                                    #
# --------------------------------------------------------------------------- #


class TestBottomUpBuild:
    """D-02: per-model file is source of truth; top-level = collection."""

    def test_per_model_files_written_before_top_level_and_contents_match(
        self, tmp_path
    ):
        # Plant the multi_orgname closed/ tree — it has two orgs, each
        # with one training/unet3d workload. That gives us two per-model
        # files (one per orgname) that must union into the top-level.
        results_root = tmp_path / "results_root"
        results_root.mkdir()
        _plant_multi_orgname(results_root)

        args = Namespace(debug=False)
        gen = ReportGenerator(
            str(results_root), args=args, validate_structure=False,
        )
        rc = gen.generate_reports()
        assert rc == 0, f"generate_reports failed with rc={rc}"

        # D-02 (a): Per-model results.{csv,json} at each per-org
        # per-model path. For training, Rules.md 2.1.16 mandates the
        # rollup lives inside the <model>/run/ phase directory.
        # multi_orgname has:
        #   closed/acme/results/system-a/training/unet3d/run/
        #   closed/beta_corp/results/system-b/training/unet3d/run/
        acme_dir = (
            results_root / "closed" / "acme" / "results" / "system-a"
            / "training" / "unet3d" / "run"
        )
        beta_dir = (
            results_root / "closed" / "beta_corp" / "results" / "system-b"
            / "training" / "unet3d" / "run"
        )
        for per_model_dir in (acme_dir, beta_dir):
            assert (per_model_dir / "results.csv").exists(), (
                f"D-02 violated: expected per-model results.csv at {per_model_dir}"
            )
            assert (per_model_dir / "results.json").exists(), (
                f"D-02 violated: expected per-model results.json at {per_model_dir}"
            )

        # D-02 (b): top-level results.{csv,json} at global_summary_dir.
        top_json = pathlib.Path(gen.global_summary_dir) / "results.json"
        top_csv = pathlib.Path(gen.global_summary_dir) / "results.csv"
        assert top_json.exists(), f"expected top-level {top_json}"
        assert top_csv.exists(), f"expected top-level {top_csv}"

        # D-02 (c) bottom-up integrity: the top-level JSON row set must
        # equal the UNION of every per-model JSON row set. The
        # multi_orgname layout puts each org under its own results/
        # subtree, so the resolver picks ONE org's results/ folder as
        # global_summary_dir. Aggregate every per-model file discovered
        # by walking results_root and confirm the top-level rows are a
        # subset of that union AND every row in top_json has a
        # matching per-model row.
        per_model_rows: List[Dict[str, Any]] = []
        for per_model_dir in (acme_dir, beta_dir):
            per_model_rows.extend(_load_json(per_model_dir / "results.json"))

        top_rows = _load_json(top_json)

        # Every top-level row must appear in some per-model file — the
        # collection step MUST NOT invent rows.
        per_model_identities: Set[Tuple[str, ...]] = {
            _row_identity(r) for r in per_model_rows
        }
        for row in top_rows:
            ident = _row_identity(row)
            assert ident in per_model_identities, (
                f"D-02 violated: top-level row {ident} has no matching per-model row. "
                f"Per-model identities: {per_model_identities}"
            )


# --------------------------------------------------------------------------- #
# TestDeleteAndRerun — D-03 canonical                                         #
# --------------------------------------------------------------------------- #


class TestDeleteAndRerun:
    """D-03: reconstruct from scratch — deleted run vanishes on rerun."""

    def test_deleted_run_subdir_absent_from_next_report(self, tmp_path):
        # Build the canonical multi-workload tree using multi_orgname's
        # two-orgname training trees. N = 2 workloads (one per org).
        results_root = tmp_path / "results_root"
        results_root.mkdir()
        _plant_multi_orgname(results_root)

        # First reportgen invocation.
        args = Namespace(debug=False)
        gen1 = ReportGenerator(
            str(results_root), args=args, validate_structure=False,
        )
        rc1 = gen1.generate_reports()
        assert rc1 == 0

        top_json_path = pathlib.Path(gen1.global_summary_dir) / "results.json"
        rows_before = _load_json(top_json_path)
        n = len(rows_before)
        assert n >= 2, (
            f"expected at least 2 rows in top-level before deletion, got {n}. "
            f"Rows: {rows_before}"
        )

        # Capture identity of each workload BEFORE deletion so we can
        # assert the deleted workload's row is absent AFTER.
        identities_before = {_row_identity(r) for r in rows_before}
        # beta_corp's workload lives at beta_corp/results/system-b/training/unet3d/run/<ts>/.
        # Delete the whole run tree so the workload disappears.
        beta_workload_root = (
            results_root / "closed" / "beta_corp" / "results" / "system-b"
            / "training" / "unet3d"
        )
        # Remove the ENTIRE per-model dir (including the earlier-emitted
        # results.{csv,json}) so the reconstruction step sees an
        # empty-of-real-runs subtree that _enumerate_on_disk_model_dirs
        # won't find (parent training/ dir survives; but there's no
        # unet3d/ child anymore).
        shutil.rmtree(beta_workload_root)
        assert not beta_workload_root.exists()

        # Second reportgen invocation — full pipeline runs against the
        # mutated tree.
        gen2 = ReportGenerator(
            str(results_root), args=args, validate_structure=False,
        )
        rc2 = gen2.generate_reports()
        assert rc2 == 0

        top_json_path_after = pathlib.Path(gen2.global_summary_dir) / "results.json"
        rows_after = _load_json(top_json_path_after)

        # D-03 canonical: (a) N-1 rows; (b) deleted row absent.
        assert len(rows_after) == n - 1, (
            f"D-03 violated: expected {n-1} rows after deletion, got "
            f"{len(rows_after)}. Before: {rows_before}. After: {rows_after}"
        )

        # The deleted workload's identity carried Organization='beta_corp',
        # Benchmark Type='training', Model='Unet3D' (display label). Any row
        # bearing ('beta_corp', 'training', 'Unet3D') must be absent.
        for row in rows_after:
            assert not (
                row.get('Organization') == 'beta_corp'
                and row.get('Benchmark Type') == 'training'
                and row.get('Model') == 'Unet3D'
            ), (
                f"D-03 violated: deleted workload row still present in "
                f"top-level after rerun. Row: {row}"
            )

        # And the beta_corp row that WAS in the before-set must be
        # missing from the after-set (row-identity subset check).
        identities_after = {_row_identity(r) for r in rows_after}
        beta_identity = next(
            (
                ident for ident in identities_before
                if ident[1] == 'beta_corp'  # Organization
                and ident[2] == 'training'  # Benchmark Type
                and ident[3] == 'Unet3D'    # Model (display label)
            ),
            None,
        )
        assert beta_identity is not None, (
            f"test setup issue: beta_corp workload not in before-set. "
            f"Before identities: {identities_before}"
        )
        assert beta_identity not in identities_after, (
            f"D-03 violated: beta_corp workload identity {beta_identity} "
            f"still present after deletion. After identities: {identities_after}"
        )


# --------------------------------------------------------------------------- #
# TestEmptyModelDir — D-03 corner                                             #
# --------------------------------------------------------------------------- #


class TestEmptyModelDir:
    """D-03 corner: empty per-model dir emits header-only CSV + [] JSON."""

    def test_empty_model_dir_emits_header_only_csv_and_empty_list_json(
        self, tmp_path
    ):
        # empty_model fixture layout:
        #   empty_model/training/unet3d/run/.gitkeep
        # Copy the tree so the results_dir has a canonical training/<model>/run/
        # shape with no run timestamp subdirs. _enumerate_on_disk_model_dirs
        # walks the tree, finds training/unet3d/run/ (Rules.md 2.1.16 —
        # training per-model rollup lives at <model>/run/), and
        # _emit_empty_model_dirs writes results.csv + results.json there.
        src = _FIXTURES_ROOT / "empty_model"
        results_root = tmp_path / "results_root"
        shutil.copytree(src, results_root)

        # Remove the .gitkeep so nothing in the run/ dir masquerades as
        # a valid run summary. accumulate_results MAY still walk this
        # tree — its 'no runs found' path is what we exercise here.
        gitkeep = results_root / "training" / "unet3d" / "run" / ".gitkeep"
        if gitkeep.exists():
            gitkeep.unlink()

        args = Namespace(debug=False)
        gen = ReportGenerator(
            str(results_root), args=args, validate_structure=False,
        )
        rc = gen.generate_reports()
        assert rc == 0

        # The empty per-model dir MUST have results.csv (header row
        # only, 1 line) and results.json ([]). For training, this is at
        # <model>/run/ per Rules.md 2.1.16.
        per_model_dir = results_root / "training" / "unet3d" / "run"
        csv_path = per_model_dir / "results.csv"
        json_path = per_model_dir / "results.json"
        assert csv_path.exists(), (
            f"D-03 corner violated: expected {csv_path} (header only). "
            f"Directory: {list(per_model_dir.iterdir())}"
        )
        assert json_path.exists(), (
            f"D-03 corner violated: expected {json_path} ([] JSON). "
            f"Directory: {list(per_model_dir.iterdir())}"
        )

        # CSV: header row only — exactly one line, non-empty, carrying the
        # full fixed webpage-parity schema (column-parity: header is fixed
        # even for an empty model dir).
        from mlpstorage_py.report_generator import _FINAL_SCHEMA
        with open(csv_path, "r", newline="") as fh:
            lines = [line.rstrip("\r\n") for line in fh if line.strip()]
        assert len(lines) == 1, (
            f"D-03 corner violated: expected 1 header line in {csv_path}, "
            f"got {len(lines)} lines: {lines}"
        )
        header = _read_csv_header(csv_path)
        assert header == _FINAL_SCHEMA, (
            f"empty-model CSV header must be the fixed schema: {header}"
        )

        # JSON: contents == [].
        loaded = _load_json(json_path)
        assert loaded == [], (
            f"D-03 corner violated: expected [] in {json_path}, got {loaded!r}"
        )


# --------------------------------------------------------------------------- #
# TestPreservesUnrelatedFiles — D-04                                          #
# --------------------------------------------------------------------------- #


class TestPreservesUnrelatedFiles:
    """D-04: reportgen never prunes unrelated files under <results-dir>."""

    def test_reportgen_does_not_prune_unrelated_files(self, tmp_path):
        # Build a canonical tree AND plant unrelated files that
        # reportgen has no business touching.
        results_root = tmp_path / "results_root"
        results_root.mkdir()
        _plant_multi_orgname(results_root)

        # Plant unrelated files.
        readme = results_root / "README.md"
        readme.write_text("Unrelated readme content — must survive reportgen.")
        orphan = results_root / "orphan_report.json"
        orphan.write_text('{"note": "orphan file — must survive reportgen"}')
        # Also plant an unrelated file DEEP in the tree so we cover
        # the case where reportgen walks a subtree it emits to.
        deep_unrelated = (
            results_root / "closed" / "acme" / "results" / "system-a"
            / "training" / "unet3d" / "notes.txt"
        )
        deep_unrelated.write_text("Unrelated per-model notes — must survive.")

        args = Namespace(debug=False)
        gen = ReportGenerator(
            str(results_root), args=args, validate_structure=False,
        )
        rc = gen.generate_reports()
        assert rc == 0

        # D-04: all planted unrelated files STILL exist verbatim.
        for planted in (readme, orphan, deep_unrelated):
            assert planted.exists(), (
                f"D-04 violated: unrelated file {planted} was pruned by reportgen"
            )
        # Contents unchanged (spot-check).
        assert readme.read_text() == (
            "Unrelated readme content — must survive reportgen."
        )
        assert '"orphan file — must survive reportgen"' in orphan.read_text()
        assert deep_unrelated.read_text() == (
            "Unrelated per-model notes — must survive."
        )


# --------------------------------------------------------------------------- #
# TestLeafAndRollupAgree — issue #836 suggestion 3                            #
# --------------------------------------------------------------------------- #


def _plant_flattened_kvcache(results_root: pathlib.Path) -> pathlib.Path:
    """Plant a kv_cache run whose leaf is NOT named after its timestamp.

    Reproduces ``closed/ANL/results/crux-eagle`` from #836: three runs
    written canonically as ``<system>/kv_cache/llama3.1-8b/run/<ts>/`` on
    three different machines, then merged into one system directory at
    packaging time with each ``run/<ts>/`` pair collapsed into a single
    directory renamed after its topology.

    Returns the model directory that owns the planted runs.
    """
    src = _FIXTURES_ROOT / "kvcache" / "llama3-8b" / "run" / "20260704_140000"
    model_dir = (
        results_root / "closed" / "ANL" / "results" / "crux-eagle"
        / "kv_cache" / "llama3-8b-10u"
    )
    model_dir.mkdir(parents=True)
    for leaf_name, run_datetime in (
        ("1nodex8ppn", "20260723_062638"),
        ("8nodex8ppn", "20260723_191548"),
    ):
        leaf = model_dir / leaf_name
        shutil.copytree(src, leaf)
        # Each leaf carries its own run_datetime, as the real tree does —
        # the directory name no longer encodes it.
        for name in ("summary.json", "kvcache_llama3-8b_metadata.json"):
            path = leaf / name
            blob = _load_json(path)
            blob["run_datetime"] = run_datetime
            with open(path, "w") as fh:
                json.dump(blob, fh)
    return model_dir


class TestLeafAndRollupAgree:
    """A workload's own table must hold the row its rollups publish (#836).

    Two paths disagreed about what constitutes a run. Extraction works
    off each leaf's self-describing ``*_metadata.json``, so a
    non-canonically-named leaf yields a ``BenchmarkRun`` and reaches the
    rollups. Row PLACEMENT went through ``_model_group_folder``, which
    hops a fixed two levels up from the leaf on the assumption that it
    sits at ``<model>/<command>/<ts>/``. When the leaf sits one level
    shallower, those two hops land ON the ``kv_cache`` benchmark-type
    directory instead of the model directory — so the row was written to
    ``<system>/kv_cache/results.csv``, a path the canonical layout has no
    table at, while the real model directory got a header-only file from
    the D-03 empty-model-dir pass.

    Observed in the v3.0 tree as an empty
    ``crux-eagle/kv_cache/llama3-8b-10u/results.csv`` beside a populated
    ``crux-eagle/kv_cache/results.csv`` and a populated top-level row.
    ``crux-eagle/kv_cache/results.csv`` is the only such file in the
    tree — every other system's kv_cache table lives one level deeper,
    inside its model directory.
    """

    def test_noncanonical_leaf_row_lands_in_the_model_dir(self, tmp_path):
        results_root = tmp_path / "results_root"
        results_root.mkdir()
        model_dir = _plant_flattened_kvcache(results_root)

        args = Namespace(debug=False)
        gen = ReportGenerator(
            str(results_root), args=args, validate_structure=False,
        )
        assert gen.generate_reports() == 0

        # The runs were extracted and aggregated — that is the premise,
        # not the thing under test.
        top_rows = _load_json(pathlib.Path(gen.global_summary_dir) / "results.json")
        kv_rows = [r for r in top_rows if r.get("Benchmark Type") == "kv_cache"]
        assert len(kv_rows) == 1, (
            f"expected the group to publish one row upstream; got {kv_rows}"
        )

        # ...so the model directory's own table must hold it.
        leaf_rows = _load_json(model_dir / "results.json")
        assert len(leaf_rows) == 1, (
            f"{model_dir.name}/results.json is empty while the top-level "
            f"table publishes {len(kv_rows)} kv_cache row(s) — the two "
            f"tables disagree about whether this workload exists"
        )
        assert leaf_rows[0]["Public ID"] == kv_rows[0]["Public ID"]

    def test_no_table_is_written_at_the_benchmark_type_dir(self, tmp_path):
        """No ``<system>/kv_cache/results.csv`` — that path owns no table.

        The canonical model-level rollup lives at
        ``<system>/kv_cache/<model>/``. A table one level above it is not
        a second opinion, it is a phantom: nothing reads it, and it
        contradicts the model directory beside it.
        """
        results_root = tmp_path / "results_root"
        results_root.mkdir()
        model_dir = _plant_flattened_kvcache(results_root)
        benchmark_type_dir = model_dir.parent

        args = Namespace(debug=False)
        gen = ReportGenerator(
            str(results_root), args=args, validate_structure=False,
        )
        assert gen.generate_reports() == 0

        for name in ("results.csv", "results.json"):
            stray = benchmark_type_dir / name
            assert not stray.exists(), (
                f"reportgen wrote {stray}, one level above the model "
                f"directory that owns the workload's table"
            )
