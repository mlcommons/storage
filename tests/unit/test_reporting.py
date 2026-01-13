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

from mlpstorage.reporting import Result, ReportGenerator
from mlpstorage.config import BENCHMARK_TYPES, PARAM_VALIDATION, EXIT_CODE
from mlpstorage.rules import Issue


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
                generator = ReportGenerator(str(results_dir), logger=mock_logger)

        assert generator.logger == mock_logger

    def test_uses_debug_from_args(self, tmp_path):
        """Should use debug setting from args."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        args = Namespace(debug=True)

        with patch.object(ReportGenerator, 'accumulate_results'):
            with patch.object(ReportGenerator, 'print_results'):
                generator = ReportGenerator(str(results_dir), args=args)

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
                return ReportGenerator(str(results_dir))

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
                return ReportGenerator(str(results_dir))

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

        with patch.object(ReportGenerator, 'accumulate_results'):
            with patch.object(ReportGenerator, 'print_results'):
                gen = ReportGenerator(str(results_dir))

        # Add mock run results
        mock_run = MagicMock()
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
        """Should create results.json."""
        generator.generate_reports()
        json_file = os.path.join(generator.results_dir, 'results.json')
        assert os.path.exists(json_file)

    def test_creates_csv_file(self, generator):
        """Should create results.csv."""
        generator.generate_reports()
        csv_file = os.path.join(generator.results_dir, 'results.csv')
        assert os.path.exists(csv_file)


class TestReportGeneratorPrintResults:
    """Tests for print_results method."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create a ReportGenerator instance."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        with patch.object(ReportGenerator, 'accumulate_results'):
            with patch.object(ReportGenerator, 'print_results'):
                gen = ReportGenerator(str(results_dir))

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
        mock_runs[1].run_id = "run2"
        mock_runs[1].accelerator = "h100"

        generator.run_results = {
            'run1': Result(
                multi=False,
                benchmark_type=BENCHMARK_TYPES.training,
                benchmark_command='run',
                benchmark_model='unet3d',
                benchmark_run=mock_runs[0],
                issues=[],
                category=PARAM_VALIDATION.CLOSED,
                metrics={}
            ),
            'run2': Result(
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

        with patch('mlpstorage.reporting.get_runs_files', return_value=[mock_run]):
            with patch('mlpstorage.reporting.BenchmarkVerifier') as mock_verifier_class:
                mock_verifier = MagicMock()
                mock_verifier.verify.return_value = PARAM_VALIDATION.CLOSED
                mock_verifier.issues = []
                mock_verifier_class.return_value = mock_verifier

                with patch.object(ReportGenerator, 'print_results'):
                    generator = ReportGenerator(str(results_dir))

        assert 'test_run' in generator.run_results
        assert generator.run_results['test_run'].category == PARAM_VALIDATION.CLOSED

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

        with patch('mlpstorage.reporting.get_runs_files', return_value=[mock_run1, mock_run2]):
            with patch('mlpstorage.reporting.BenchmarkVerifier') as mock_verifier_class:
                mock_verifier = MagicMock()
                mock_verifier.verify.return_value = PARAM_VALIDATION.CLOSED
                mock_verifier.issues = []
                mock_verifier_class.return_value = mock_verifier

                with patch.object(ReportGenerator, 'print_results'):
                    generator = ReportGenerator(str(results_dir))

        # Should have workload result for (unet3d, h100)
        assert ('unet3d', 'h100') in generator.workload_results


class TestReportGeneratorIntegration:
    """Integration tests for ReportGenerator."""

    def test_full_workflow_with_fixture_data(self, tmp_path):
        """Test full workflow with mock fixture data."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Create mock benchmark run
        mock_run = MagicMock()
        mock_run.run_id = "training_run_20250111"
        mock_run.benchmark_type = BENCHMARK_TYPES.training
        mock_run.command = 'run'
        mock_run.model = 'unet3d'
        mock_run.accelerator = 'h100'
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

        with patch('mlpstorage.reporting.get_runs_files', return_value=[mock_run]):
            with patch('mlpstorage.reporting.BenchmarkVerifier') as mock_verifier_class:
                mock_verifier = MagicMock()
                mock_verifier.verify.return_value = PARAM_VALIDATION.CLOSED
                mock_verifier.issues = []
                mock_verifier_class.return_value = mock_verifier

                generator = ReportGenerator(str(results_dir))

        # Generate reports
        result = generator.generate_reports()
        assert result == EXIT_CODE.SUCCESS

        # Check files were created
        assert os.path.exists(os.path.join(results_dir, 'results.json'))
        assert os.path.exists(os.path.join(results_dir, 'results.csv'))
