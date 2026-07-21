"""Task F — SUT (System Under Test) block population in reportgen output.

reportgen must emit the shared System-Under-Test columns (v3.0 results
table, `Results Table Structure.xlsx`) into results.{csv,json} for every
workload row, sourced from the row's `system-description.yaml`:

- ``sut_organization``  ← orgname (path)
- ``sut_name``          ← Hyperlink(text=solution.submission_name,
                                    href=<cat>/<org>/systems/<sys>.yaml)
- ``sut_description``    ← Hyperlink(text=solution.submission_name,
                                    href=<cat>/<org>/systems/<sys>.pdf)
- ``sut_rus``           ← system_under_test.total_rack_units
- blank placeholders (manual fill post-reportgen): ``sut_public_id``,
  ``sut_type``, ``sut_access_protocol``, ``sut_availability``,
  ``sut_integrated_client_storage``, ``sut_usable_capacity_tib``

Encoding (user-confirmed):
- ``.csv``  → hyperlink cell is an HTML anchor string
              ``<a href="...">text</a>``
- ``.json`` → hyperlink cell is a ``{"text": ..., "href": ...}`` object

Href base = repo-root-relative (Option A): ``<results-dir>`` is the
aggregated-submissions repo root, so ``closed/<org>/systems/<sys>.yaml``
is a valid relative URL.

Regression guard: the D-10 6-column prefix stays first and D-12 ``issues``
stays last — the ``sut_*`` columns are a body group between them.
"""

from __future__ import annotations

import csv
import json
import pathlib
import shutil
from argparse import Namespace

import pytest

from mlpstorage_py.report_generator import ReportGenerator

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_FIXTURES_ROOT = _REPO_ROOT / "tests" / "fixtures" / "sample_results"

_SUBMISSION_NAME = "AcmeSysA"
_RACK_UNITS = 14

_ACME_YAML = f"""\
system_under_test:
  solution:
    submission_name: {_SUBMISSION_NAME}
  total_rack_units: {_RACK_UNITS}
"""


def _prepare_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """Copy the multi_orgname fixture and inject acme's system-description.yaml."""
    fixture_src = _FIXTURES_ROOT / "multi_orgname"
    assert fixture_src.is_dir(), f"expected fixture at {fixture_src}"
    dest = tmp_path / "repo_root"
    shutil.copytree(fixture_src, dest)
    systems_dir = dest / "closed" / "acme" / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "system-a.yaml").write_text(_ACME_YAML)
    return dest


def _run_reportgen(root: pathlib.Path) -> None:
    gen = ReportGenerator(str(root), args=Namespace(debug=False),
                          validate_structure=False)
    rc = gen.generate_reports()
    assert rc == 0, f"generate_reports returned {rc}"


def _acme_per_model_dir(root: pathlib.Path) -> pathlib.Path:
    return (root / "closed" / "acme" / "results" / "system-a"
            / "training" / "unet3d" / "run")


class TestSutBlockJson:
    def test_json_sut_columns_populated(self, tmp_path):
        root = _prepare_tree(tmp_path)
        _run_reportgen(root)
        rows = json.loads((_acme_per_model_dir(root) / "results.json").read_text())
        assert len(rows) == 1
        row = rows[0]

        assert row["sut_organization"] == "acme"
        assert row["sut_rus"] == _RACK_UNITS
        # Hyperlinks serialize as {"text","href"} objects in JSON.
        assert row["sut_name"] == {
            "text": _SUBMISSION_NAME,
            "href": "closed/acme/systems/system-a.yaml",
        }
        assert row["sut_description"] == {
            "text": _SUBMISSION_NAME,
            "href": "closed/acme/systems/system-a.pdf",
        }
        # Blank placeholders present-but-empty (manual fill).
        for blank in ("sut_public_id", "sut_type", "sut_access_protocol",
                      "sut_availability", "sut_integrated_client_storage",
                      "sut_usable_capacity_tib"):
            assert row[blank] == "", f"{blank} should be blank, got {row[blank]!r}"


class TestSutBlockCsv:
    def test_csv_hyperlinks_render_as_html_anchor(self, tmp_path):
        root = _prepare_tree(tmp_path)
        _run_reportgen(root)
        with open(_acme_per_model_dir(root) / "results.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        row = rows[0]
        assert row["sut_name"] == (
            f'<a href="closed/acme/systems/system-a.yaml">{_SUBMISSION_NAME}</a>'
        )
        assert row["sut_description"] == (
            f'<a href="closed/acme/systems/system-a.pdf">{_SUBMISSION_NAME}</a>'
        )
        assert row["sut_organization"] == "acme"
        assert row["sut_rus"] == str(_RACK_UNITS)


class TestSutBlockDoesNotBreakColumnInvariants:
    def test_prefix_first_issues_last_sut_in_between(self, tmp_path):
        root = _prepare_tree(tmp_path)
        _run_reportgen(root)
        with open(_acme_per_model_dir(root) / "results.csv", newline="") as fh:
            header = next(csv.reader(fh))
        assert header[:6] == [
            "category", "orgname", "systemname",
            "benchmark_type", "model", "accelerator",
        ], f"D-10 prefix broken: {header[:6]}"
        assert header[-1] == "issues", f"D-12 trailing issues broken: {header[-1]}"
        # sut_ columns exist and sit strictly between prefix and issues.
        sut_idxs = [i for i, c in enumerate(header) if c.startswith("sut_")]
        assert sut_idxs, f"no sut_ columns in header: {header}"
        assert min(sut_idxs) >= 6 and max(sut_idxs) < len(header) - 1
