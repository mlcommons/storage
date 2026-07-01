"""
Tests for ReportGenerator class in mlpstorage.reporting module.

Tests cover:
- Result dataclass
- ReportGenerator initialization
- Report generation
- CSV and JSON file writing
- Results accumulation
- Results printing
"""

import csv
import json
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import asdict
from argparse import Namespace

from mlpstorage_py.report_generator import Result, ReportGenerator
from mlpstorage_py.config import BENCHMARK_TYPES, PARAM_VALIDATION, EXIT_CODE
from mlpstorage_py.rules import Issue


class TestResultDataclass:
    """Tests for Result dataclass."""

    def test_result_creation(self):
        """Should create Result with all fields."""
        mock_run = MagicMock()
        mock_run.run_id = "test_run_id"
        issues = [Issue(PARAM_VALIDATION.OPEN, "Test issue")]

        result = Result(
            multi=False,
            benchmark_type=BENCHMARK_TYPES.training,
            benchmark_command='run',
            benchmark_model='unet3d',
            benchmark_run=mock_run,
            issues=issues,
            category=PARAM_VALIDATION.CLOSED,
            metrics={'throughput': 100.0}
        )

        assert result.multi is False
        assert result.benchmark_type == BENCHMARK_TYPES.training
        assert result.benchmark_command == 'run'
        assert result.benchmark_model == 'unet3d'
        assert result.benchmark_run == mock_run
        assert len(result.issues) == 1
        assert result.category == PARAM_VALIDATION.CLOSED
        assert result.metrics == {'throughput': 100.0}

    def test_result_with_multi_runs(self):
        """Should handle multi=True with list of runs."""
        mock_runs = [MagicMock(), MagicMock()]

        result = Result(
            multi=True,
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            benchmark_command='run',
            benchmark_model='llama3-8b',
            benchmark_run=mock_runs,
            issues=[],
            category=PARAM_VALIDATION.OPEN,
            metrics={}
        )

        assert result.multi is True
        assert len(result.benchmark_run) == 2


class TestReportGeneratorInit:
    """Tests for ReportGenerator initialization."""

    def test_exits_if_results_dir_not_exists(self, tmp_path):
        """Should exit if results directory doesn't exist."""
        with pytest.raises(SystemExit):
            ReportGenerator('/nonexistent/path')

    def test_accepts_custom_logger(self, tmp_path):
        """Should accept custom logger."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        mock_logger = MagicMock()

        with patch.object(ReportGenerator, 'accumulate_results'):
            with patch.object(ReportGenerator, 'print_results'):
                generator = ReportGenerator(str(results_dir), logger=mock_logger, validate_structure=False)

        assert generator.logger == mock_logger

    def test_uses_debug_from_args(self, tmp_path):
        """Should use debug setting from args."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = Namespace(debug=True)

        with patch.object(ReportGenerator, 'accumulate_results'):
            with patch.object(ReportGenerator, 'print_results'):
                generator = ReportGenerator(str(results_dir), args=args, validate_structure=False)

        assert generator.debug is True


class TestReportGeneratorWriteJson:
    """Tests for write_json_file method."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create a ReportGenerator instance."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        with patch.object(ReportGenerator, 'accumulate_results'):
            with patch.object(ReportGenerator, 'print_results'):
                return ReportGenerator(str(results_dir), validate_structure=False)

    def test_writes_json_file(self, generator):
        """Should write results to JSON file."""
        results = [
            {'run_id': 'run1', 'model': 'unet3d'},
            {'run_id': 'run2', 'model': 'resnet50'}
        ]
        generator.write_json_file(results)

        json_file = os.path.join(generator.results_dir, 'results.json')
        assert os.path.exists(json_file)

        with open(json_file, 'r') as f:
            loaded = json.load(f)

        assert len(loaded) == 2
        assert loaded[0]['run_id'] == 'run1'

    def test_json_has_proper_formatting(self, generator):
        """JSON should be properly formatted with indent."""
        results = [{'key': 'value'}]
        generator.write_json_file(results)

        json_file = os.path.join(generator.results_dir, 'results.json')
        with open(json_file, 'r') as f:
            content = f.read()

        # Should have newlines (indicating indentation)
        assert '\n' in content


class TestReportGeneratorWriteCsv:
    """Tests for write_csv_file method."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create a ReportGenerator instance."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        with patch.object(ReportGenerator, 'accumulate_results'):
            with patch.object(ReportGenerator, 'print_results'):
                return ReportGenerator(str(results_dir), validate_structure=False)

    def test_writes_csv_file(self, generator):
        """Should write results to CSV file."""
        results = [
            {'run_id': 'run1', 'model': 'unet3d', 'throughput': 100.0},
            {'run_id': 'run2', 'model': 'resnet50', 'throughput': 200.0}
        ]
        generator.write_csv_file(results)

        csv_file = os.path.join(generator.results_dir, 'results.csv')
        assert os.path.exists(csv_file)

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

    def test_flattens_nested_dicts(self, generator):
        """Should flatten nested dictionaries."""
        results = [
            {'run_id': 'run1', 'metrics': {'throughput': 100.0, 'au': 95.0}}
        ]
        generator.write_csv_file(results)

        csv_file = os.path.join(generator.results_dir, 'results.csv')
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have flattened keys
        assert 'metrics.throughput' in rows[0] or 'throughput' in rows[0]

    def test_handles_nan_values(self, generator):
        """Should remove NaN values."""
        results = [
            {'run_id': 'run1', 'value': float('nan')}
        ]
        generator.write_csv_file(results)

        csv_file = os.path.join(generator.results_dir, 'results.csv')
        assert os.path.exists(csv_file)


class TestReportGeneratorGenerateReports:
    """Tests for generate_reports method."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create a ReportGenerator with mock run results."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Model-level rollup: one results.{json,csv} per <model>/ folder,
        # covering every command's timestamps for that model. The mock
        # leaf must live under the canonical training layout
        # `<results>/training/<model>/<command>/<ts>/` so the grouping
        # helper can walk up two levels to land at `<model>/`.
        self._model_folder = results_dir / "training" / "unet3d"
        leaf = self._model_folder / "run" / "20250111_120000"
        leaf.mkdir(parents=True)

        with patch.object(ReportGenerator, 'accumulate_results'):
            with patch.object(ReportGenerator, 'print_results'):
                gen = ReportGenerator(str(results_dir), validate_structure=False)

        # Add mock run results
        mock_run = MagicMock()
        mock_run.result_dir = str(leaf)
        mock_run.as_dict.return_value = {
            'run_id': 'test_run',
            'benchmark_type': 'training',
            'model': 'unet3d',
            'metrics': {'throughput': 100.0}
        }

        gen.run_results = {
            'test_run': Result(
                multi=False,
                benchmark_type=BENCHMARK_TYPES.training,
                benchmark_command='run',
                benchmark_model='unet3d',
                benchmark_run=mock_run,
                issues=[],
                category=PARAM_VALIDATION.CLOSED,
                metrics={'throughput': 100.0}
            )
        }
        return gen

    def test_returns_success(self, generator):
        """Should return SUCCESS exit code."""
        result = generator.generate_reports()
        assert result == EXIT_CODE.SUCCESS

    def test_creates_json_file(self, generator):
        """Should create results.json inside the <model>/ folder."""
        generator.generate_reports()
        json_file = os.path.join(str(self._model_folder), 'results.json')
        assert os.path.exists(json_file), (
            f"Expected model-level rollup at {json_file}"
        )

    def test_creates_csv_file(self, generator):
        """Should create results.csv inside the <model>/ folder."""
        generator.generate_reports()
        csv_file = os.path.join(str(self._model_folder), 'results.csv')
        assert os.path.exists(csv_file), (
            f"Expected model-level rollup at {csv_file}"
        )

    def test_creates_global_summary_files(self, generator):
        """Should also emit the global <results-dir>/results.{json,csv}
        summary alongside the per-model rollups. Preserves the
        pre-model-rollup contract downstream tooling relies on."""
        generator.generate_reports()
        global_json = os.path.join(generator.results_dir, 'results.json')
        global_csv = os.path.join(generator.results_dir, 'results.csv')
        assert os.path.exists(global_json), (
            f"Expected global summary at {global_json}"
        )
        assert os.path.exists(global_csv), (
            f"Expected global summary at {global_csv}"
        )


class TestReportGeneratorPrintResults:
    """Tests for print_results method."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create a ReportGenerator instance."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        with patch.object(ReportGenerator, 'accumulate_results'):
            with patch.object(ReportGenerator, 'print_results'):
                gen = ReportGenerator(str(results_dir), validate_structure=False)

        return gen

    def test_prints_closed_results(self, generator, capsys):
        """Should print CLOSED results."""
        mock_run = MagicMock()
        mock_run.run_id = "test_run"
        mock_run.benchmark_type = BENCHMARK_TYPES.training
        mock_run.command = 'run'
        mock_run.model = 'unet3d'

        generator.run_results = {
            'test_run': Result(
                multi=False,
                benchmark_type=BENCHMARK_TYPES.training,
                benchmark_command='run',
                benchmark_model='unet3d',
                benchmark_run=mock_run,
                issues=[],
                category=PARAM_VALIDATION.CLOSED,
                metrics={'throughput': 100.0}
            )
        }
        generator.workload_results = {}

        generator.print_results()
        captured = capsys.readouterr()
        assert "CLOSED" in captured.out
        assert "test_run" in captured.out

    def test_prints_issues(self, generator, capsys):
        """Should print issues for results."""
        mock_run = MagicMock()
        mock_run.run_id = "test_run"

        generator.run_results = {
            'test_run': Result(
                multi=False,
                benchmark_type=BENCHMARK_TYPES.training,
                benchmark_command='run',
                benchmark_model='unet3d',
                benchmark_run=mock_run,
                issues=[Issue(PARAM_VALIDATION.OPEN, "Test issue message")],
                category=PARAM_VALIDATION.OPEN,
                metrics={}
            )
        }
        generator.workload_results = {}

        generator.print_results()
        captured = capsys.readouterr()
        assert "Test issue message" in captured.out

    def test_prints_metrics(self, generator, capsys):
        """Should print metrics for results."""
        mock_run = MagicMock()
        mock_run.run_id = "test_run"

        generator.run_results = {
            'test_run': Result(
                multi=False,
                benchmark_type=BENCHMARK_TYPES.training,
                benchmark_command='run',
                benchmark_model='unet3d',
                benchmark_run=mock_run,
                issues=[],
                category=PARAM_VALIDATION.CLOSED,
                metrics={'throughput': 1250.5, 'au_percentage': 95.2}
            )
        }
        generator.workload_results = {}

        generator.print_results()
        captured = capsys.readouterr()
        assert "1,250.5" in captured.out  # Formatted with comma
        assert "95.2%" in captured.out  # Percentage formatted

    def test_prints_metric_lists(self, generator, capsys):
        """Should print metric lists."""
        mock_run = MagicMock()
        mock_run.run_id = "test_run"

        generator.run_results = {
            'test_run': Result(
                multi=False,
                benchmark_type=BENCHMARK_TYPES.training,
                benchmark_command='run',
                benchmark_model='unet3d',
                benchmark_run=mock_run,
                issues=[],
                category=PARAM_VALIDATION.CLOSED,
                metrics={'throughput': [100.0, 200.0, 300.0]}
            )
        }
        generator.workload_results = {}

        generator.print_results()
        captured = capsys.readouterr()
        assert "100.0" in captured.out

    def test_prints_workload_results(self, generator, capsys):
        """Should print workload results."""
        mock_runs = [MagicMock(), MagicMock()]
        mock_runs[0].run_id = "run1"
        mock_runs[0].accelerator = "h100"
        # result_dir + benchmark_type are consumed by _system_scope_key
        # (called from _print_workload_details' warmup badge lookup).
        # Explicit strings prevent MagicMock's auto-attribute machinery
        # from producing unpredictable paths.
        mock_runs[0].result_dir = "/results/training/unet3d/run/20250101_000001"
        mock_runs[0].benchmark_type = BENCHMARK_TYPES.training
        mock_runs[1].run_id = "run2"
        mock_runs[1].accelerator = "h100"
        mock_runs[1].result_dir = "/results/training/unet3d/run/20250101_000002"
        mock_runs[1].benchmark_type = BENCHMARK_TYPES.training

        # Both leaves walk up 4 dirs from `/results/training/unet3d/run/<ts>/`
        # to `/results`, so both share scope '/results'.
        generator.run_results = {
            ('/results', 'run1'): Result(
                multi=False,
                benchmark_type=BENCHMARK_TYPES.training,
                benchmark_command='run',
                benchmark_model='unet3d',
                benchmark_run=mock_runs[0],
                issues=[],
                category=PARAM_VALIDATION.CLOSED,
                metrics={}
            ),
            ('/results', 'run2'): Result(
                multi=False,
                benchmark_type=BENCHMARK_TYPES.training,
                benchmark_command='run',
                benchmark_model='unet3d',
                benchmark_run=mock_runs[1],
                issues=[],
                category=PARAM_VALIDATION.CLOSED,
                metrics={}
            )
        }

        generator.workload_results = {
            ('unet3d', 'h100'): Result(
                multi=True,
                benchmark_type=BENCHMARK_TYPES.training,
                benchmark_command='run',
                benchmark_model='unet3d',
                benchmark_run=mock_runs,
                issues=[],
                category=PARAM_VALIDATION.CLOSED,
                metrics={}
            )
        }

        generator.print_results()
        captured = capsys.readouterr()
        assert "Training" in captured.out
        assert "unet3d" in captured.out


class TestReportGeneratorAccumulateResults:
    """Tests for accumulate_results method."""

    def test_accumulates_from_benchmark_runs(self, tmp_path):
        """Should accumulate results from benchmark runs."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Create mock BenchmarkRun
        mock_run = MagicMock()
        mock_run.run_id = "test_run"
        mock_run.benchmark_type = BENCHMARK_TYPES.training
        mock_run.command = 'run'
        mock_run.model = 'unet3d'
        mock_run.accelerator = 'h100'
        mock_run.metrics = {'throughput': 100.0}

        with patch('mlpstorage_py.report_generator.get_runs_files', return_value=[mock_run]):
            with patch('mlpstorage_py.report_generator.BenchmarkVerifier') as mock_verifier_class:
                mock_verifier = MagicMock()
                mock_verifier.verify.return_value = PARAM_VALIDATION.CLOSED
                mock_verifier.issues = []
                mock_verifier_class.return_value = mock_verifier

                with patch.object(ReportGenerator, 'print_results'):
                    generator = ReportGenerator(str(results_dir), validate_structure=False)

        # run_results is keyed by (system_scope, run_id) tuple so that
        # identical run_ids from different systems don't collide as
        # warmups. There's exactly one entry here; extract it via .values()
        # rather than depending on the exact scope-key string, which is
        # derived from the MagicMock's .result_dir attribute path.
        assert len(generator.run_results) == 1
        (only_result,) = generator.run_results.values()
        assert only_result.category == PARAM_VALIDATION.CLOSED

    def test_groups_by_workload(self, tmp_path):
        """Should group runs by workload (model, accelerator)."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Create two mock runs with same workload
        mock_run1 = MagicMock()
        mock_run1.run_id = "run1"
        mock_run1.benchmark_type = BENCHMARK_TYPES.training
        mock_run1.command = 'run'
        mock_run1.model = 'unet3d'
        mock_run1.accelerator = 'h100'
        mock_run1.metrics = {}

        mock_run2 = MagicMock()
        mock_run2.run_id = "run2"
        mock_run2.benchmark_type = BENCHMARK_TYPES.training
        mock_run2.command = 'run'
        mock_run2.model = 'unet3d'
        mock_run2.accelerator = 'h100'
        mock_run2.metrics = {}

        with patch('mlpstorage_py.report_generator.get_runs_files', return_value=[mock_run1, mock_run2]):
            with patch('mlpstorage_py.report_generator.BenchmarkVerifier') as mock_verifier_class:
                mock_verifier = MagicMock()
                mock_verifier.verify.return_value = PARAM_VALIDATION.CLOSED
                mock_verifier.issues = []
                mock_verifier_class.return_value = mock_verifier

                with patch.object(ReportGenerator, 'print_results'):
                    generator = ReportGenerator(str(results_dir), validate_structure=False)

        # Should have workload result for (unet3d, h100)
        assert ('unet3d', 'h100') in generator.workload_results


class TestReportGeneratorIntegration:
    """Integration tests for ReportGenerator."""

    def test_full_workflow_with_fixture_data(self, tmp_path):
        """Test full workflow with mock fixture data."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Model-level rollups land at `<...>/training/<model>/`, so the
        # mock leaf must live under the canonical layout for the parent
        # arithmetic in _model_group_folder() to resolve correctly.
        model_folder = results_dir / "training" / "unet3d"
        leaf = model_folder / "run" / "20250111_120000"
        leaf.mkdir(parents=True)

        # Create mock benchmark run
        mock_run = MagicMock()
        mock_run.run_id = "training_run_20250111"
        mock_run.benchmark_type = BENCHMARK_TYPES.training
        mock_run.command = 'run'
        mock_run.model = 'unet3d'
        mock_run.accelerator = 'h100'
        mock_run.result_dir = str(leaf)
        mock_run.metrics = {
            'train_throughput_samples_per_second': 1250.5,
            'train_au_percentage': 95.2
        }
        mock_run.as_dict.return_value = {
            'run_id': 'training_run_20250111',
            'benchmark_type': 'training',
            'model': 'unet3d',
            'accelerator': 'h100',
            'metrics': mock_run.metrics
        }

        with patch('mlpstorage_py.report_generator.get_runs_files', return_value=[mock_run]):
            with patch('mlpstorage_py.report_generator.BenchmarkVerifier') as mock_verifier_class:
                mock_verifier = MagicMock()
                mock_verifier.verify.return_value = PARAM_VALIDATION.CLOSED
                mock_verifier.issues = []
                mock_verifier_class.return_value = mock_verifier

                generator = ReportGenerator(str(results_dir), validate_structure=False)

        # Generate reports
        result = generator.generate_reports()
        assert result == EXIT_CODE.SUCCESS

        # Per-model rollup lands inside the <model>/ folder.
        assert os.path.exists(os.path.join(str(model_folder), 'results.json'))
        assert os.path.exists(os.path.join(str(model_folder), 'results.csv'))
        # Global summary is preserved at the results_dir root alongside it.
        assert os.path.exists(os.path.join(str(results_dir), 'results.json'))
        assert os.path.exists(os.path.join(str(results_dir), 'results.csv'))


# ---------------------------------------------------------------------------
# Issue #599 regression tests — canonical-tree support, --output-dir, and
# --systemname filtering. See the issue body for the full reproduction.
# ---------------------------------------------------------------------------


def _write_canonical_run(tmp_path, mode, orgname, systemname,
                         benchmark="training", model="unet3d",
                         timestamp="20260123_120000"):
    """Create one fake run under the canonical layout that `mlpstorage init`
    + `<bench> run` actually produces, and return the run dir."""
    run_dir = (
        tmp_path / mode / orgname / "results" / systemname
        / benchmark / model / "run" / timestamp
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{benchmark}_{model}_metadata.json").write_text("{}")
    (run_dir / "summary.json").write_text("{}")
    return run_dir


def _make_generator(results_dir, *, args=None, validate_structure=False):
    """Build a ReportGenerator with the heavy stages stubbed out so the
    init constructor lands on `self` without touching real result files."""
    with patch.object(ReportGenerator, 'accumulate_results'), \
         patch.object(ReportGenerator, 'print_results'):
        return ReportGenerator(
            str(results_dir), args=args,
            validate_structure=validate_structure,
        )


class TestIssue599CanonicalTreeAccepted:
    """Bug 1: the validator rejected the canonical
    `<results-dir>/<closed|open>/<orgname>/results/<systemname>/<bench>/...`
    layout that the rest of the toolchain produces. After the fix,
    ReportGenerator discovers the canonical slice via discover_scan_roots
    and validates that slice, accepting the tree the user already has."""

    def test_canonical_closed_tree_is_accepted_and_walked(self, tmp_path):
        """The validator must pass and accumulate_results must walk the
        closed slice for the requested system. Pre-fix this raised
        SystemExit at validation time."""
        _write_canonical_run(
            tmp_path, "closed", "Acme", "sysA",
            benchmark="training", model="unet3d",
        )

        args = Namespace(
            debug=False, output_dir=None,
            orgname="Acme", systemname="sysA",
        )

        # Intercept get_runs_files so we can assert it was called with the
        # canonical slice path (NOT the bare tmp_path) — that's the key
        # post-fix contract.
        seen_scan_roots = []
        def fake_get_runs(path, logger=None):
            seen_scan_roots.append(path)
            return []

        with patch('mlpstorage_py.report_generator.get_runs_files',
                   side_effect=fake_get_runs), \
             patch.object(ReportGenerator, 'print_results'):
            gen = ReportGenerator(str(tmp_path), args=args,
                                  validate_structure=True)

        expected_slice = str(
            tmp_path / "closed" / "Acme" / "results" / "sysA"
        )
        assert seen_scan_roots == [expected_slice], (
            f"accumulate_results must walk the canonical slice, "
            f"not the bare results-dir. Saw: {seen_scan_roots!r}"
        )
        assert gen.scan_roots == [expected_slice]

    def test_canonical_tree_does_not_aggregate_other_systems(self, tmp_path):
        """Bug 3: a multi-system tree must aggregate ONLY the requested
        system. Pre-fix, get_runs_files walked everything under
        --results-dir and tagged every run with the requested
        --systemname, mashing systems together."""
        _write_canonical_run(tmp_path, "closed", "Acme", "sysA")
        _write_canonical_run(tmp_path, "closed", "Acme", "sysB")

        args = Namespace(
            debug=False, output_dir=None,
            orgname="Acme", systemname="sysA",
        )
        seen_scan_roots = []
        def fake_get_runs(path, logger=None):
            seen_scan_roots.append(path)
            return []

        with patch('mlpstorage_py.report_generator.get_runs_files',
                   side_effect=fake_get_runs), \
             patch.object(ReportGenerator, 'print_results'):
            ReportGenerator(str(tmp_path), args=args,
                            validate_structure=True)

        # Only sysA's slice may be scanned.
        assert seen_scan_roots == [
            str(tmp_path / "closed" / "Acme" / "results" / "sysA")
        ], seen_scan_roots
        # And definitely NOT sysB's.
        assert not any(
            "sysB" in p for p in seen_scan_roots
        ), seen_scan_roots

    def test_canonical_both_modes_walked(self, tmp_path):
        """When the submitter has staged both closed/ and open/ subtrees
        for the same system, both slices are walked."""
        _write_canonical_run(tmp_path, "closed", "Acme", "sysA")
        _write_canonical_run(tmp_path, "open", "Acme", "sysA")

        args = Namespace(
            debug=False, output_dir=None,
            orgname="Acme", systemname="sysA",
        )
        seen_scan_roots = []
        def fake_get_runs(path, logger=None):
            seen_scan_roots.append(path)
            return []

        with patch('mlpstorage_py.report_generator.get_runs_files',
                   side_effect=fake_get_runs), \
             patch.object(ReportGenerator, 'print_results'):
            ReportGenerator(str(tmp_path), args=args,
                            validate_structure=True)

        assert sorted(seen_scan_roots) == sorted([
            str(tmp_path / "closed" / "Acme" / "results" / "sysA"),
            str(tmp_path / "open" / "Acme" / "results" / "sysA"),
        ])

    def test_flat_layout_still_works_without_orgname(self, tmp_path):
        """Backward-compat: without orgname/systemname (programmatic
        callers, the existing test fixtures, pre-LAY-03 trees), reportgen
        still walks --results-dir directly."""
        # Build a flat tree (no closed/open wrapper).
        run_dir = tmp_path / "training" / "unet3d" / "run" / "20260123_120000"
        run_dir.mkdir(parents=True)
        (run_dir / "training_unet3d_metadata.json").write_text("{}")
        (run_dir / "summary.json").write_text("{}")

        seen_scan_roots = []
        def fake_get_runs(path, logger=None):
            seen_scan_roots.append(path)
            return []

        with patch('mlpstorage_py.report_generator.get_runs_files',
                   side_effect=fake_get_runs), \
             patch.object(ReportGenerator, 'print_results'):
            # No args — exercises the no-orgname/no-systemname fallback.
            ReportGenerator(str(tmp_path), args=None,
                            validate_structure=True)

        assert seen_scan_roots == [str(tmp_path)]


class TestGlobalSummaryLocation:
    """The global results.{json,csv} rollup must NOT land inside the
    per-system slice when a canonical submission tree is in use. It
    belongs one level up, in the `<sentinel>/<div>/<org>/results/`
    directory that is the parent of every `<system>/` subfolder, so a
    submitter sees a single aggregate alongside each per-system slice
    instead of an aggregate buried under one system's directory."""

    def test_global_summary_dir_is_results_parent_under_canonical_tree(
        self, tmp_path
    ):
        _write_canonical_run(tmp_path, "closed", "Acme", "sysA")
        args = Namespace(
            debug=False, output_dir=None,
            orgname="Acme", systemname="sysA",
        )

        with patch('mlpstorage_py.report_generator.get_runs_files',
                   return_value=[]), \
             patch.object(ReportGenerator, 'print_results'):
            gen = ReportGenerator(str(tmp_path), args=args,
                                  validate_structure=True)

        # results_dir is rebound to the per-system slice…
        assert gen.results_dir == str(
            tmp_path / "closed" / "Acme" / "results" / "sysA"
        )
        # …but the global summary lands in the org's results/ parent.
        assert gen.global_summary_dir == str(
            tmp_path / "closed" / "Acme" / "results"
        )
        # And critically, the global summary dir is NOT the system slice.
        assert gen.global_summary_dir != gen.results_dir
        assert "sysA" not in os.path.basename(gen.global_summary_dir)

    def test_global_summary_dir_equals_results_dir_for_flat_layout(
        self, tmp_path
    ):
        """Flat layout (no canonical rebind): global summary dir stays
        equal to results_dir — preserves the pre-canonical-tree
        contract."""
        run_dir = tmp_path / "training" / "unet3d" / "run" / "20260123_120000"
        run_dir.mkdir(parents=True)
        (run_dir / "training_unet3d_metadata.json").write_text("{}")
        (run_dir / "summary.json").write_text("{}")

        with patch('mlpstorage_py.report_generator.get_runs_files',
                   return_value=[]), \
             patch.object(ReportGenerator, 'print_results'):
            gen = ReportGenerator(str(tmp_path), args=None,
                                  validate_structure=True)

        assert gen.global_summary_dir == gen.results_dir == str(tmp_path)
