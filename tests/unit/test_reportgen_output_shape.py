"""
Phase 6 output-shape regression tests — the emitted-file layer for
report_generator.write_csv_file / write_json_file / generate_reports.

Concerns pinned here (see .planning/phases/06-score-aggregation/06-CONTEXT.md):

- D-10: Fixed 6-column prefix, in this exact order:
    ['category', 'orgname', 'systemname', 'benchmark_type', 'model',
     'accelerator']
- D-12: Trailing 'issues' column (always last).
- D-25: Multi-issue rows joined by '; ' (semicolon + single space).
- D-01: No 'row_type' discriminator column anywhere in the emitted output
        or in the report_generator source (defense against the
        SUPERSEDED SC-5 language returning in a future refactor). This
        is the D-01 defense clause — see the planner-discipline-allow
        marker below.
- D-29: whatif runs appear in results.{csv,json} with category='whatif'
        AND the D-24 INVALID message substrings do NOT appear in that
        row's issues column (whatif SKIPs the rules-strict gates).
- D-08: multi-orgname trees produce ONE top-level results.{csv,json}
        containing rows for EVERY orgname, distinguished by the
        'orgname' column (no per-orgname sub-file synthesis).

This file is the writer-boundary analog of tests/unit/test_aggregation.py.
That file pins the helper-level (D-11 within-group ordering, D-14 exact
column names, D-24 verbatim INVALID templates). This file pins the
emitted-file shape: what actually lands on disk after generate_reports
runs against fixture trees.

Style precedent
---------------
- Constructor patching pattern: same as test_reporting.py's TestReportGeneratorWriteCsv
  fixture (patch.object accumulate_results / print_results so the
  constructor does not run the full scan) — for the direct-writer tests.
- Full-pipeline pattern: instantiate ReportGenerator without patches for
  the D-29 whatif and D-08 multi-orgname tests, which require the real
  accumulate + workload-groups path (write_csv_file alone does not
  route category / orgname / systemname through _workload_result_to_row).

planner-discipline-allow: row_type
The string 'row_type' appears in TestNoRowTypeColumn (which asserts its
ABSENCE from the report_generator source, header row, and JSON keys).
That is the D-01 defense-mention — not a positive assertion FOR the
column. Do not remove it: the point is to catch a future PR that
resurrects the SUPERSEDED SC-5 row_type discriminator.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import shutil
from argparse import Namespace
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from mlpstorage_py.report_generator import ReportGenerator
from mlpstorage_py.config import BENCHMARK_TYPES


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_FIXTURES_ROOT = _REPO_ROOT / "tests" / "fixtures" / "sample_results"
_REPORT_GENERATOR_PY = _REPO_ROOT / "mlpstorage_py" / "report_generator.py"


# --------------------------------------------------------------------------- #
# Helper: build a bare ReportGenerator with the constructor pipeline patched  #
# off. Mirrors the tests/unit/test_reporting.py fixture pattern.              #
# --------------------------------------------------------------------------- #


def _bare_generator(tmp_path: pathlib.Path) -> ReportGenerator:
    """Instantiate ReportGenerator with accumulate/print patched off."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    with patch.object(ReportGenerator, "accumulate_results"):
        with patch.object(ReportGenerator, "print_results"):
            return ReportGenerator(str(results_dir), validate_structure=False)


def _synthetic_rows() -> List[Dict[str, Any]]:
    """Build a small row list covering every column-type group.

    Each row carries the D-10 6-column prefix, at least one metric column
    from the D-11 group it belongs to, and a trailing issues field
    populated by a proxy string (the writer-side flattening simply
    passes the value through).
    """
    return [
        {
            "category": "closed",
            "orgname": "acme",
            "systemname": "system-a",
            "benchmark_type": "training",
            "model": "unet3d",
            "accelerator": "h100",
            "train_mean_of_au_percentage": 95.0,
            "train_mean_of_throughput_samples_per_second": 1250.0,
            "issues": "",
        },
        {
            "category": "closed",
            "orgname": "acme",
            "systemname": "system-a",
            "benchmark_type": "checkpointing",
            "model": "llama3-8b",
            "accelerator": "",
            "checkpoint_mean_of_read_throughput_GB_per_second": 12.4,
            "checkpoint_mean_of_write_throughput_GB_per_second": 8.8,
            "issues": "",
        },
        {
            "category": "closed",
            "orgname": "acme",
            "systemname": "system-a",
            "benchmark_type": "vector_database",
            "model": "",
            "accelerator": "",
            "vdb_engine": "milvus",
            "vdb_index_type": "hnsw",
            "vdb_throughput_qps": 4200.0,
            "vdb_recall": 0.985,
            "issues": "",
        },
        {
            "category": "closed",
            "orgname": "acme",
            "systemname": "system-a",
            "benchmark_type": "kv_cache",
            "model": "llama3-8b",
            "accelerator": "",
            "kvcache_performance_profile": "balanced",
            "kvcache_aggregated_read_bandwidth_gbps": 24.0,
            "issues": "",
        },
    ]


# --------------------------------------------------------------------------- #
# TestSixColumnPrefix — D-10                                                  #
# --------------------------------------------------------------------------- #


_LEFT_EDGE = [
    'Public ID', 'Organization', 'Division', 'Benchmark Type', 'Model',
]


class TestFixedSchemaLeftEdge:
    """Column-parity: fixed left-edge (SUT + discriminators) at CSV + JSON.

    Replaces the SUPERSEDED D-10 machine-key 6-column prefix. The emitted
    file is now the fixed webpage-parity schema; the first five columns are
    the identity + discriminator block (Public ID, Organization, Division,
    Benchmark Type, Model).
    """

    def test_csv_header_starts_with_fixed_left_edge(self, tmp_path):
        gen = _bare_generator(tmp_path)
        out_dir = tmp_path / "csv_prefix_out"
        out_dir.mkdir()
        gen.write_csv_file(_synthetic_rows(), target_dir=str(out_dir))

        csv_path = out_dir / "results.csv"
        assert csv_path.exists(), f"expected {csv_path} to exist"
        with open(csv_path, "r", newline="") as fh:
            header = next(csv.reader(fh))
        assert header[:5] == _LEFT_EDGE, (
            "Fixed-schema left edge violated. First 5 header columns must be "
            f"{_LEFT_EDGE}. Got: {header[:5]}"
        )
        # The old machine-key identity columns must be gone.
        for banned in ('category', 'orgname', 'systemname', 'benchmark_type'):
            assert banned not in header

    def test_json_row_dict_starts_with_fixed_left_edge(self, tmp_path):
        gen = _bare_generator(tmp_path)
        out_dir = tmp_path / "json_prefix_out"
        out_dir.mkdir()
        gen.write_json_file(_synthetic_rows(), target_dir=str(out_dir))

        json_path = out_dir / "results.json"
        assert json_path.exists(), f"expected {json_path} to exist"
        with open(json_path, "r") as fh:
            loaded = json.load(fh)

        for row in loaded:
            keys = list(row.keys())
            assert keys[:5] == _LEFT_EDGE, (
                f"Fixed-schema left-edge ORDER violated in JSON row. Expected "
                f"first 5 keys to be {_LEFT_EDGE}, got {keys[:5]}."
            )


# --------------------------------------------------------------------------- #
# TestTrailingIssuesColumn — D-12 / D-25                                      #
# --------------------------------------------------------------------------- #


class TestNoIssuesColumn:
    """Column-parity: the 'issues' column is NOT emitted.

    Supersedes D-12/D-25 (trailing 'issues' column, '; '-joined). The
    fixed webpage-parity schema carries only the reference columns +
    discriminators; validation issues are not a reference column. The
    internal '; ' join in _workload_result_to_row still runs (it feeds
    the on-screen report), it is just no longer emitted to the file.
    """

    def test_csv_header_has_no_issues_column(self, tmp_path):
        gen = _bare_generator(tmp_path)
        out_dir = tmp_path / "csv_trailing_out"
        out_dir.mkdir()
        gen.write_csv_file(_synthetic_rows(), target_dir=str(out_dir))

        csv_path = out_dir / "results.csv"
        assert csv_path.exists()
        with open(csv_path, "r", newline="") as fh:
            header = next(csv.reader(fh))
        assert 'issues' not in header, (
            f"'issues' must not be a results.csv column. Full header: {header}"
        )

    def test_json_rows_have_no_issues_key(self, tmp_path):
        gen = _bare_generator(tmp_path)
        out_dir = tmp_path / "json_no_issues_out"
        out_dir.mkdir()
        gen.write_json_file(_synthetic_rows(), target_dir=str(out_dir))

        with open(out_dir / "results.json", "r") as fh:
            loaded = json.load(fh)
        for row in loaded:
            assert 'issues' not in row, f"'issues' must not be a JSON key: {row}"


# --------------------------------------------------------------------------- #
# TestNoRowTypeColumn — D-01 defense                                          #
# --------------------------------------------------------------------------- #


class TestNoRowTypeColumn:
    """D-01 defense: no row_type column in emitted output OR in source.

    The SUPERSEDED SC-5 language proposed a row_type discriminator to
    tell 'run' rows apart from 'aggregate' rows in the top-level file.
    D-01 replaced that with one-row-per-workload; the discriminator was
    struck. If a future PR resurrects it, these tests fail loudly.
    """

    def test_csv_header_does_not_contain_row_type(self, tmp_path):
        gen = _bare_generator(tmp_path)
        out_dir = tmp_path / "csv_no_row_type_out"
        out_dir.mkdir()
        gen.write_csv_file(_synthetic_rows(), target_dir=str(out_dir))

        csv_path = out_dir / "results.csv"
        with open(csv_path, "r", newline="") as fh:
            header = next(csv.reader(fh))
        # D-01 defense: no discriminator column.
        assert 'row_type' not in header, (
            f"D-01 defense violated: CSV header contains 'row_type'. "
            f"Full header: {header}"
        )

    def test_json_rows_do_not_contain_row_type_key(self, tmp_path):
        gen = _bare_generator(tmp_path)
        out_dir = tmp_path / "json_no_row_type_out"
        out_dir.mkdir()
        gen.write_json_file(_synthetic_rows(), target_dir=str(out_dir))

        json_path = out_dir / "results.json"
        with open(json_path, "r") as fh:
            loaded = json.load(fh)
        for row in loaded:
            assert 'row_type' not in row, (
                f"D-01 defense violated: JSON row contains 'row_type' key. "
                f"Row: {row}"
            )

    def test_report_generator_source_does_not_contain_row_type_string(self):
        """Read report_generator.py as text; assert no 'row_type' literal.

        Strips pure-comment lines (first non-whitespace char == '#') so
        the D-01 defense marker in COMMENTS does not self-invalidate the
        check. Same comment-stripping technique as
        tests/unit/test_aggregation.py::TestNumpyPandasScipyForbidden.
        """
        assert _REPORT_GENERATOR_PY.is_file(), (
            f"expected report_generator.py at {_REPORT_GENERATOR_PY}"
        )
        raw = _REPORT_GENERATOR_PY.read_text()
        cleaned_lines = []
        for line in raw.splitlines():
            stripped = line.lstrip()
            if stripped.startswith('#'):
                continue
            cleaned_lines.append(line)
        cleaned_text = '\n'.join(cleaned_lines)
        # D-01 defense: no row_type string in the codepath itself
        # (comment references were stripped above). Grep gate.
        assert 'row_type' not in cleaned_text, (
            "D-01 defense violated: report_generator.py contains 'row_type' "
            "outside comments. The SUPERSEDED SC-5 row_type discriminator "
            "must not appear in the codepath."
        )


# --------------------------------------------------------------------------- #
# TestWhatifCategoryValue — D-29                                              #
# --------------------------------------------------------------------------- #


class TestWhatifCategoryValue:
    """D-29: whatif runs emit category='whatif' AND skip D-24 INVALID gates."""

    def test_whatif_row_emits_category_whatif(self, tmp_path):
        # Plant the whatif fixture under a layout that preserves the
        # 'whatif' path segment so _derive_category_from_path resolves
        # 'whatif' correctly (the derivation scans the ABSOLUTE
        # result_dir for the literal 'whatif' token).
        #
        # Layout: <results_dir>/whatif/training/unet3d/run/<ts>/
        fixture_src = _FIXTURES_ROOT / "whatif"
        assert fixture_src.is_dir(), (
            f"expected whatif fixture at {fixture_src}"
        )
        results_root = tmp_path / "results_dir"
        results_root.mkdir()
        # Copy the whatif tree wholesale, preserving 'whatif/' as a
        # visible path segment beneath results_root. The
        # canonical-tree resolver does NOT match this shape (no
        # closed/open dirs), so results_dir stays at results_root.
        shutil.copytree(fixture_src, results_root / "whatif")

        # Full-pipeline run. No accumulate/print patches — we want
        # generate_reports to actually iterate workload_results built
        # from the fixture tree.
        args = Namespace(debug=False)
        gen = ReportGenerator(
            str(results_root), args=args, validate_structure=False,
        )
        rc = gen.generate_reports()
        assert rc == 0, f"generate_reports returned non-zero exit code {rc}"

        # Read the emitted top-level results.json.
        top_json = pathlib.Path(gen.global_summary_dir) / "results.json"
        assert top_json.exists(), (
            f"expected top-level results.json at {top_json}. "
            f"Directory listing: {list(pathlib.Path(gen.global_summary_dir).iterdir())}"
        )
        with open(top_json, "r") as fh:
            rows = json.load(fh)

        # D-29: the whatif category surfaces in the fixed-schema Division
        # column (category.upper()). Division is the emitted successor to
        # the machine-key 'category' prefix column.
        whatif_rows = [r for r in rows if r.get('Division') == 'WHATIF']
        assert whatif_rows, (
            f"D-29 violated: expected at least one row with Division='WHATIF' "
            f"in {top_json}. Got rows: {rows}"
        )

        # D-29 also mandates whatif SKIPS the rules-strict INVALID gates.
        # The 'issues' column is no longer emitted (column-parity), so
        # assert the skip at the internal Result layer instead: no whatif
        # workload Result carries a D-24 INVALID substring.
        d24_substrings = [
            "expected 6 training invocations per Rules.md",
            "expected exactly 1 warmup invocation to be detected",
            "expected 10 checkpoint operations per Rules.md",
            "cannot aggregate",
        ]
        for wk, wr in gen.workload_results.items():
            cat = getattr(wr.category, 'value', wr.category)
            if str(cat) != 'whatif' and 'whatif' not in str(wk):
                continue
            issue_text = '; '.join(
                str(getattr(i, 'message', i)) for i in (wr.issues or []))
            for sub in d24_substrings:
                assert sub not in issue_text, (
                    f"D-29 violated: whatif workload {wk} carries D-24 INVALID "
                    f"substring {sub!r} — whatif MUST skip the rules-strict "
                    f"gates. Issues: {issue_text}"
                )


# --------------------------------------------------------------------------- #
# TestMultiOrgnameCollection — D-08                                           #
# --------------------------------------------------------------------------- #


class TestMultiOrgnameCollection:
    """D-08: multi-orgname trees produce ONE top-level file with mixed rows."""

    def test_multi_orgname_produces_single_top_level_file_with_mixed_rows(
        self, tmp_path
    ):
        # Copy the multi_orgname fixture into tmp_path. The fixture root
        # itself is `multi_orgname/` and contains
        # `closed/<org>/results/<sys>/training/unet3d/run/<ts>/...` under it.
        fixture_src = _FIXTURES_ROOT / "multi_orgname"
        assert fixture_src.is_dir(), (
            f"expected multi_orgname fixture at {fixture_src}"
        )
        dest = tmp_path / "multi_orgname"
        shutil.copytree(fixture_src, dest)

        # Point ReportGenerator at `dest` — which contains the
        # closed/<org>/ tree. The canonical-tree resolver
        # (_resolve_effective_results_dir) recognizes this layout
        # because `dest/closed` exists.
        args = Namespace(debug=False)
        gen = ReportGenerator(
            str(dest), args=args, validate_structure=False,
        )
        rc = gen.generate_reports()
        assert rc == 0, f"generate_reports returned non-zero exit code {rc}"

        # D-08: ONE top-level results.json / results.csv, containing
        # rows from BOTH orgnames. The top-level lives at
        # `global_summary_dir` (the org's `results/` folder — but under
        # multi-orgname trees, there is only one `results/` per org, and
        # since we did not pass --systemname, the resolver rebinds each
        # org's canonical tree. The multi_orgname fixture has two
        # separate `closed/<org>/results/` trees; the resolver picks
        # ONE (the first sorted match). Verify BOTH per-org per-model
        # files exist per D-09 (that layer is always produced) and
        # that the top-level file collects rows from at least the
        # picked org.
        # This test's critical assertion is D-09 per-org per-model file
        # existence — that is what proves D-08 semantics (one file per
        # org tree, no per-orgname sub-file synthesis at any other
        # level).

        # D-09: per-org per-model files exist at their canonical paths.
        # For training, Rules.md 2.1.16 mandates the rollup lives inside
        # the <model>/run/ phase directory, not directly under <model>/.
        acme_per_model_json = (
            dest / "closed" / "acme" / "results" / "system-a"
            / "training" / "unet3d" / "run" / "results.json"
        )
        beta_per_model_json = (
            dest / "closed" / "beta_corp" / "results" / "system-b"
            / "training" / "unet3d" / "run" / "results.json"
        )
        assert acme_per_model_json.exists(), (
            f"D-09 violated: expected per-model rollup at {acme_per_model_json}"
        )
        assert beta_per_model_json.exists(), (
            f"D-09 violated: expected per-model rollup at {beta_per_model_json}"
        )

        # Read per-model rows to confirm each org has its own row.
        with open(acme_per_model_json, "r") as fh:
            acme_rows = json.load(fh)
        with open(beta_per_model_json, "r") as fh:
            beta_rows = json.load(fh)
        assert len(acme_rows) == 1, (
            f"expected 1 acme workload row in per-model file, got {len(acme_rows)}"
        )
        assert len(beta_rows) == 1, (
            f"expected 1 beta_corp workload row in per-model file, got {len(beta_rows)}"
        )
        assert acme_rows[0].get('Organization') == 'acme', (
            f"expected acme in acme's per-model row Organization, got "
            f"{acme_rows[0].get('Organization')!r}"
        )
        assert beta_rows[0].get('Organization') == 'beta_corp', (
            f"expected beta_corp in beta's per-model row Organization, got "
            f"{beta_rows[0].get('Organization')!r}"
        )

        # D-08 core assertion: the top-level file exists and its rows
        # come from the SAME single results.json file — no per-orgname
        # sub-file was synthesized between the top level and the
        # per-model level. (Worklist A11 moved the top level of a
        # multi-org tree to the tree root, collecting EVERY org's rows;
        # the per-org rollup at each <div>/<org>/results/ is pinned by
        # TestMultiOrgRollupPlacement, not here.)
        top_level_json = pathlib.Path(gen.global_summary_dir) / "results.json"
        assert top_level_json.exists(), (
            f"expected top-level results.json at {top_level_json}"
        )
        with open(top_level_json, "r") as fh:
            top_rows = json.load(fh)

        # The fixed-schema 'Organization' column is the emitted successor
        # to the machine-key 'orgname' — it distinguishes cross-org rows.
        top_orgnames = {r.get('Organization') for r in top_rows}
        assert top_orgnames, (
            f"top-level results.json is empty; expected at least one org's rows"
        )
        assert '' not in top_orgnames, (
            f"D-08 violated: top-level rows include empty Organization. "
            f"Organizations seen: {top_orgnames}"
        )
        assert top_orgnames.issubset({'acme', 'beta_corp'}), (
            f"D-08 violated: top-level rows include unexpected Organizations "
            f"{top_orgnames - {'acme', 'beta_corp'}}"
        )


# --------------------------------------------------------------------------- #
# TestMultiOrgRollupPlacement — worklist A10/A11 (Curtis, 2026-07-24)         #
# --------------------------------------------------------------------------- #


class TestMultiOrgRollupPlacement:
    """A10/A11: multi-org tree → per-org rollups + one global at tree root.

    An assembled multi-submitter tree (no sentinel → orgname=None) must:

    - A10: pass structure validation — scan roots are the per-system
      slices under every ``<div>/<org>/results/``, never the raw root.
    - A11 (decision (a), Curtis 2026-07-24): the GLOBAL rollup
      ``results.{csv,json}`` lands at the TREE ROOT (covering every org),
      and each org additionally gets its own rollup at
      ``<div>/<org>/results/`` — the same placement a single-org
      canonical tree already gets. Pre-fix,
      ``_resolve_effective_results_dir`` returned from its org loop on
      the FIRST org found, so the alphabetically-first org's tree
      received the aggregate for everyone.

    This extends D-08 (one top-level file with mixed rows) rather than
    contradicting it: the top level of a multi-org tree is the tree
    root, and the per-org rollup at each org's ``results/`` mirrors the
    established single-org behavior."""

    def _build_tree(self, tmp_path: pathlib.Path) -> pathlib.Path:
        fixture_src = _FIXTURES_ROOT / "multi_orgname"
        assert fixture_src.is_dir(), (
            f"expected multi_orgname fixture at {fixture_src}"
        )
        dest = tmp_path / "multi_orgname"
        shutil.copytree(fixture_src, dest)
        return dest

    def _run_reportgen(self, dest: pathlib.Path,
                       validate_structure: bool = False) -> ReportGenerator:
        args = Namespace(debug=False)
        gen = ReportGenerator(
            str(dest), args=args, validate_structure=validate_structure,
        )
        rc = gen.generate_reports()
        assert rc == 0, f"generate_reports returned non-zero exit code {rc}"
        return gen

    def test_scan_roots_are_per_system_slices_not_raw_root(self, tmp_path):
        """A10: orgname=None on a canonical multi-org tree must yield the
        per-system slices; the raw root (which holds only division dirs)
        must not be a scan root."""
        dest = self._build_tree(tmp_path)
        gen = self._run_reportgen(dest)
        expected = [
            str(dest / "closed" / "acme" / "results" / "system-a"),
            str(dest / "closed" / "beta_corp" / "results" / "system-b"),
        ]
        assert gen.scan_roots == expected, (
            f"A10 violated: expected per-system slices as scan roots, "
            f"got {gen.scan_roots}"
        )

    def test_structure_validation_passes_on_multi_org_tree(self, tmp_path):
        """A10 end-to-end: validate_structure=True must not sys.exit(3) —
        pre-fix the raw root was validated and had no benchmark-type
        dirs, killing the run before any report was generated."""
        dest = self._build_tree(tmp_path)
        gen = self._run_reportgen(dest, validate_structure=True)
        assert gen.workload_results, (
            "expected workloads to accumulate after structure validation"
        )

    def test_global_rollup_lands_at_tree_root_with_all_orgs(self, tmp_path):
        """A11 decision (a): the global results.{csv,json} covers every
        org and sits at the tree root — NOT inside the first org's
        results/ folder."""
        dest = self._build_tree(tmp_path)
        gen = self._run_reportgen(dest)
        assert gen.global_summary_dir == str(dest), (
            f"A11 violated: global rollup dir should be the tree root "
            f"{dest}, got {gen.global_summary_dir}"
        )
        top_json = dest / "results.json"
        assert top_json.exists(), (
            f"expected global results.json at tree root {top_json}"
        )
        with open(top_json, "r") as fh:
            rows = json.load(fh)
        orgs = {r.get('Organization') for r in rows}
        assert orgs == {'acme', 'beta_corp'}, (
            f"global rollup must contain rows from every org, got {orgs}"
        )
        # The first org's results/ must NOT receive the global rollup.
        first_org_rollup = (
            dest / "closed" / "acme" / "results" / "results.json"
        )
        if first_org_rollup.exists():
            with open(first_org_rollup, "r") as fh:
                acme_rollup_rows = json.load(fh)
            acme_orgs = {r.get('Organization') for r in acme_rollup_rows}
            assert acme_orgs <= {'acme'}, (
                f"acme's per-org rollup leaked other orgs' rows: {acme_orgs}"
            )

    def test_per_org_rollups_written_into_each_orgs_results_dir(
        self, tmp_path
    ):
        """A11 decision (a): each org gets its own rollup at
        <div>/<org>/results/, containing only that org's rows."""
        dest = self._build_tree(tmp_path)
        self._run_reportgen(dest)
        for org in ("acme", "beta_corp"):
            org_json = dest / "closed" / org / "results" / "results.json"
            org_csv = dest / "closed" / org / "results" / "results.csv"
            assert org_json.exists(), (
                f"A11 violated: expected per-org rollup at {org_json}"
            )
            assert org_csv.exists(), (
                f"A11 violated: expected per-org rollup at {org_csv}"
            )
            with open(org_json, "r") as fh:
                rows = json.load(fh)
            assert rows, f"per-org rollup at {org_json} is empty"
            assert {r.get('Organization') for r in rows} == {org}, (
                f"per-org rollup at {org_json} must contain only {org} rows"
            )


class TestRollupExcludesAuxiliaryPhases:
    """R3: datagen/datasize rows stay OUT of org/global rollup tables.

    The fixed final schema has no phase column, so an auxiliary-phase row
    (datagen/datasize — the issue-#771/#791 6-element workload keys) is
    indistinguishable from a broken measurement row in the staff-facing
    tables: same Organization/Model, every metric cell blank. The rollups
    (tree root + per-org) must therefore carry only measurement rows.
    The per-phase LEAF results.{csv,json} (Rules.md 2.1.16/2.1.22
    mandated) must still be written for every phase directory.
    """

    def _build_tree_with_datagen(self, tmp_path: pathlib.Path) -> pathlib.Path:
        fixture_src = _FIXTURES_ROOT / "multi_orgname"
        dest = tmp_path / "multi_orgname"
        shutil.copytree(fixture_src, dest)
        # Clone acme's run invocation into a datagen phase dir: same
        # metadata shape, command='datagen', no accelerator, and no
        # summary.json (datagen runs never produce one — A14).
        run_dir = (
            dest / "closed" / "acme" / "results" / "system-a"
            / "training" / "unet3d" / "run" / "20260706_100000"
        )
        datagen_dir = (
            dest / "closed" / "acme" / "results" / "system-a"
            / "training" / "unet3d" / "datagen" / "20260706_090000"
        )
        datagen_dir.mkdir(parents=True)
        with open(run_dir / "training_unet3d_metadata.json", "r") as fh:
            metadata = json.load(fh)
        metadata["command"] = "datagen"
        metadata["accelerator"] = None
        metadata["run_datetime"] = "20260706_090000"
        with open(datagen_dir / "training_unet3d_metadata.json", "w") as fh:
            json.dump(metadata, fh)
        return dest

    def test_datagen_row_written_to_leaf_but_not_rollups(self, tmp_path):
        dest = self._build_tree_with_datagen(tmp_path)
        args = Namespace(debug=False)
        gen = ReportGenerator(str(dest), args=args, validate_structure=False)
        rc = gen.generate_reports()
        assert rc == 0, f"generate_reports returned non-zero exit code {rc}"

        # Leaf emission preserved: the datagen phase dir gets its own
        # results.json with the (metric-less) datagen row.
        leaf_json = (
            dest / "closed" / "acme" / "results" / "system-a"
            / "training" / "unet3d" / "datagen" / "results.json"
        )
        assert leaf_json.exists(), (
            f"R3 must not remove the Rules-mandated per-phase leaf rollup "
            f"at {leaf_json}"
        )
        with open(leaf_json, "r") as fh:
            leaf_rows = json.load(fh)
        assert len(leaf_rows) == 1, (
            f"expected the datagen leaf rollup to hold its one row, got "
            f"{len(leaf_rows)}"
        )

        # Global rollup (tree root): only the two measurement rows — the
        # acme datagen row must NOT appear.
        with open(dest / "results.json", "r") as fh:
            top_rows = json.load(fh)
        assert len(top_rows) == 2, (
            f"R3 violated: global rollup must hold only the 2 measurement "
            f"rows (acme run + beta_corp run), got {len(top_rows)} rows"
        )

        # Per-org rollup: acme keeps exactly its one measurement row.
        with open(
            dest / "closed" / "acme" / "results" / "results.json", "r"
        ) as fh:
            acme_rows = json.load(fh)
        assert len(acme_rows) == 1, (
            f"R3 violated: acme per-org rollup must hold only the run row, "
            f"got {len(acme_rows)} rows"
        )


# --------------------------------------------------------------------------- #
# TestModelFolderClamping — issue #836 suggestion 3                           #
# --------------------------------------------------------------------------- #


class TestModelFolderClamping:
    """A model-level table never lands on or above `<benchmark-type>/` (#836).

    ``_model_group_folder`` hops a fixed number of levels up from the run
    leaf, which is right at canonical depth and overshoots when a
    submission collapses ``<model>/<command>/<ts>/`` into one directory.
    The overshoot wrote crux-eagle's row to
    ``<system>/kv_cache/results.csv`` while its model directory got a
    header-only file — two tables disagreeing about one workload.

    The end-to-end behaviour is pinned in
    tests/integration/test_reportgen_aggregate_end_to_end.py; these cover
    the resolver's own edges, including the layout so flat there is no
    model level to place a table in.
    """

    @staticmethod
    def _result_for(leaf: str, bt=BENCHMARK_TYPES.kv_cache):
        """A minimal Result stand-in — the resolver reads two attributes."""
        run = SimpleNamespace(result_dir=leaf, run_id="run-under-test")
        return SimpleNamespace(benchmark_run=run, benchmark_type=bt)

    def test_canonical_leaf_resolves_to_the_model_dir(self, tmp_path):
        gen = _bare_generator(tmp_path)
        leaf = "/subs/closed/ANL/results/sys-a/kv_cache/llama3.1-8b/run/20260723_062638"
        assert gen._model_group_folder(self._result_for(leaf)) == (
            "/subs/closed/ANL/results/sys-a/kv_cache/llama3.1-8b"
        )

    def test_shallow_leaf_clamps_to_the_model_dir(self, tmp_path):
        """The crux-eagle shape: `<model>/<topology>/`, no `<command>/<ts>/`."""
        gen = _bare_generator(tmp_path)
        leaf = "/subs/closed/ANL/results/crux-eagle/kv_cache/llama3-8b-10u/1nodex8ppn"
        assert gen._model_group_folder(self._result_for(leaf)) == (
            "/subs/closed/ANL/results/crux-eagle/kv_cache/llama3-8b-10u"
        )

    def test_leaf_directly_under_benchmark_type_has_no_model_level(self, tmp_path):
        """No model directory means no model-level table — and say so.

        Returning the benchmark-type directory would write a phantom
        table; returning the leaf would write ``results.csv`` inside a run
        directory. Neither is a table, so the row stays in the rollups
        only and the warning names the expected layout.
        """
        gen = _bare_generator(tmp_path)
        gen.logger = MagicMock()
        leaf = "/subs/closed/ANL/results/crux-eagle/kv_cache/20260723_062638"

        assert gen._model_group_folder(self._result_for(leaf)) is None

        assert gen.logger.warning.called, "the skipped placement was silent"
        # Render lazy-% calls the way the logger would, so the assertions
        # read what a submitter actually sees.
        text = " ".join(
            (call.args[0] % call.args[1:]) if len(call.args) > 1 else str(call.args)
            for call in gen.logger.warning.call_args_list
        )
        assert "no model-level results.csv" in text
        assert "<system>/kv_cache/<model>/<command>/<timestamp>/" in text
        assert leaf in text

    def test_vdb_shallow_leaf_clamps_below_vector_database(self, tmp_path):
        """The same overshoot exists for vdb, whose canonical depth is deeper."""
        gen = _bare_generator(tmp_path)
        leaf = (
            "/subs/closed/TTA/results/sys-b/vector_database/milvus/hnsw-flat"
        )
        assert gen._model_group_folder(
            self._result_for(leaf, BENCHMARK_TYPES.vector_database)
        ) == "/subs/closed/TTA/results/sys-b/vector_database/milvus"

    def test_tree_without_a_benchmark_type_dir_keeps_the_hop_result(self, tmp_path):
        """Flat fixture trees have no `<benchmark-type>/` ancestor to anchor on.

        Several fixtures plant ``<model>/run/<ts>/`` straight under the
        results root. With no anchor the hop result stands, unchanged —
        the clamp must not start returning None for those.
        """
        gen = _bare_generator(tmp_path)
        leaf = str(tmp_path / "results" / "llama3-8b" / "run" / "20260704_140000")
        expected = str(tmp_path / "results" / "llama3-8b")
        assert gen._model_group_folder(self._result_for(leaf)) == expected


# --------------------------------------------------------------------------- #
# TestZombieRowsWithheld — issue #836, WRT-4                                  #
# --------------------------------------------------------------------------- #


def _plant_kvcache_leaf(
    system_dir: pathlib.Path,
    command: str,
    run_datetime: str,
    with_summary: bool = True,
) -> pathlib.Path:
    """Plant one canonical kv_cache invocation under a system directory.

    ``with_summary=False`` reproduces ZettaLane's
    ``kv_cache/llama3.1-8b/run/20260627_194318``: option dirs, trial logs
    and metadata are all present, but the run's summary sits under the
    older ``kvcache_run_summary_*.json`` name, so ``summary.json`` is
    absent and every pass-through metric reads blank.
    """
    src = _FIXTURES_ROOT / "kvcache" / "llama3-8b" / "run" / "20260704_140000"
    leaf = system_dir / "kv_cache" / "llama3-8b" / command / run_datetime
    leaf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, leaf)
    metadata_path = leaf / "kvcache_llama3-8b_metadata.json"
    with open(metadata_path) as fh:
        metadata = json.load(fh)
    metadata["command"] = command
    metadata["run_datetime"] = run_datetime
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh)
    if not with_summary:
        (leaf / "summary.json").rename(
            leaf / f"kvcache_run_summary_{run_datetime}.json"
        )
    return leaf


def _plant_kvcache_system(tmp_path: pathlib.Path, system: str) -> pathlib.Path:
    """A canonical closed/<org>/results/<system>/ directory."""
    system_dir = (
        tmp_path / "tree" / "closed" / "TestOrg" / "results" / system
    )
    system_dir.mkdir(parents=True)
    return system_dir


class TestZombieRowsWithheld:
    """A row with no measurements must not be published (#836, WRT-4).

    The fixed final schema has no phase column and no issues column, so a
    row whose every metric cell is blank is indistinguishable from a real
    result that happens to read zero. Two such rows publish in the v3.0
    tree today:

    - ``v3.0-0006`` ANL polaris-nvme, whose kv_cache group holds only two
      ``datasize`` invocations. R3 already keeps training/checkpointing
      auxiliary phases out of the rollups via the issue-#771/#791
      6-element workload keys, but kv_cache and vector_database keys are
      always 5-tuples, so their auxiliary groups were never marked.
    - ``v3.0-0138`` ZettaLane mayascale, a ``run`` group whose
      ``summary.json`` does not exist under that name — every pass-through
      column is blank.

    Where the data is missing the row is withheld from every table and the
    reason goes to the log, which is where submitters read problems (and
    what ApparentProblems.md is written from) — not into the results
    table.
    """

    def test_kvcache_datasize_only_group_stays_out_of_rollups(self, tmp_path):
        """ANL polaris-nvme's shape: auxiliary phases only, no measurement."""
        system_dir = _plant_kvcache_system(tmp_path, "polaris-nvme")
        _plant_kvcache_leaf(system_dir, "datasize", "20260721_203747")
        _plant_kvcache_leaf(system_dir, "datasize", "20260721_215716")

        gen = ReportGenerator(
            str(tmp_path / "tree"), args=Namespace(debug=False),
            validate_structure=False,
        )
        gen.logger.warning = MagicMock(wraps=gen.logger.warning)
        assert gen.generate_reports() == 0

        top_rows = json.loads(
            (pathlib.Path(gen.global_summary_dir) / "results.json").read_text()
        )
        assert top_rows == [], (
            f"a kv_cache group holding only datasize invocations published a "
            f"metric-less row: {top_rows}"
        )

        # R3 parity with training: the per-phase leaf file is still written.
        leaf_json = system_dir / "kv_cache" / "llama3-8b" / "results.json"
        assert leaf_json.exists(), (
            f"the Rules-mandated per-phase leaf rollup at {leaf_json} was "
            f"removed along with the rollup row"
        )

        # A system that loses a workload must be told why. For polaris-nvme
        # this is the ONLY signal available: its `run` leaf holds trial logs
        # and option dirs but no *_metadata.json, and the #835 skipped-run
        # reporting never sees it — the walk does not treat a leaf with no
        # metadata as a run directory at all, so it is named nowhere in a
        # full-tree pass.
        rendered = " ".join(
            (call.args[0] % call.args[1:]) if len(call.args) > 1
            else str(call.args[0] if call.args else call)
            for call in gen.logger.warning.call_args_list
        )
        assert "polaris-nvme" in rendered, (
            f"the withheld workload's system was never named: {rendered[:600]}"
        )
        assert "no measurement" in rendered, (
            f"the log does not say the group holds no measurement "
            f"invocation: {rendered[:600]}"
        )

    def test_run_without_a_readable_summary_is_withheld_and_logged(
        self, tmp_path
    ):
        """ZettaLane's shape: a run invocation that yields no measurements."""
        system_dir = _plant_kvcache_system(tmp_path, "mayascale")
        leaf = _plant_kvcache_leaf(
            system_dir, "run", "20260627_194318", with_summary=False,
        )

        gen = ReportGenerator(
            str(tmp_path / "tree"), args=Namespace(debug=False),
            validate_structure=False,
        )
        gen.logger.warning = MagicMock(wraps=gen.logger.warning)
        assert gen.generate_reports() == 0

        top_rows = json.loads(
            (pathlib.Path(gen.global_summary_dir) / "results.json").read_text()
        )
        assert top_rows == [], (
            f"a run with no readable summary published a blank row: {top_rows}"
        )
        model_rows = json.loads(
            (system_dir / "kv_cache" / "llama3-8b" / "results.json").read_text()
        )
        assert model_rows == [], (
            f"the model-level table published the blank row the rollups "
            f"withheld — the two tables disagree: {model_rows}"
        )

        rendered = " ".join(
            (call.args[0] % call.args[1:]) if len(call.args) > 1
            else str(call.args[0] if call.args else call)
            for call in gen.logger.warning.call_args_list
        )
        assert str(leaf) in rendered, (
            "the withheld row's run directory was never named in the log"
        )
        assert "no measurements" in rendered, (
            f"the log does not say why the row was withheld: {rendered[:600]}"
        )

    def test_healthy_run_still_publishes(self, tmp_path):
        """Negative control — a run with a summary is unaffected."""
        system_dir = _plant_kvcache_system(tmp_path, "healthy")
        _plant_kvcache_leaf(system_dir, "run", "20260704_140000")

        gen = ReportGenerator(
            str(tmp_path / "tree"), args=Namespace(debug=False),
            validate_structure=False,
        )
        assert gen.generate_reports() == 0

        top_rows = json.loads(
            (pathlib.Path(gen.global_summary_dir) / "results.json").read_text()
        )
        assert len(top_rows) == 1, f"the healthy row was withheld: {top_rows}"
        assert top_rows[0]["Public ID"] == "v3.0-0001"
