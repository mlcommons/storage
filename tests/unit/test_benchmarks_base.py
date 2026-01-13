"""
Tests for Benchmark base class in mlpstorage.benchmarks.base module.

Tests cover:
- Benchmark initialization
- Output location generation
- Metadata generation and writing
- Command execution
- Benchmark verification
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from argparse import Namespace

from mlpstorage.benchmarks.base import Benchmark
from mlpstorage.config import BENCHMARK_TYPES, PARAM_VALIDATION


class ConcreteBenchmark(Benchmark):
    """Concrete implementation of abstract Benchmark for testing."""

    BENCHMARK_TYPE = BENCHMARK_TYPES.training

    def _run(self):
        """Concrete implementation of abstract _run method."""
        return 0


class TestBenchmarkInit:
    """Tests for Benchmark initialization."""

    @pytest.fixture
    def basic_args(self):
        """Create basic args for benchmark."""
        return Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir='/tmp/results',
            model='unet3d',
            command='run',
            num_processes=8,
            accelerator_type='h100'
        )

    def test_creates_output_directory(self, basic_args, tmp_path):
        """Should create output directory."""
        basic_args.results_dir = str(tmp_path / "results")

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            output_dir = str(tmp_path / "results" / "output")
            mock_gen.return_value = output_dir

            benchmark = ConcreteBenchmark(basic_args)

        assert os.path.exists(output_dir)

    def test_accepts_custom_logger(self, basic_args, tmp_path):
        """Should accept custom logger."""
        basic_args.results_dir = str(tmp_path)
        mock_logger = MagicMock()

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            benchmark = ConcreteBenchmark(basic_args, logger=mock_logger)

        assert benchmark.logger == mock_logger

    def test_uses_provided_run_datetime(self, basic_args, tmp_path):
        """Should use provided run_datetime."""
        basic_args.results_dir = str(tmp_path)

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            benchmark = ConcreteBenchmark(basic_args, run_datetime="20250115_120000")

        assert benchmark.run_datetime == "20250115_120000"

    def test_sets_debug_from_args(self, basic_args, tmp_path):
        """Should set debug from args."""
        basic_args.results_dir = str(tmp_path)
        basic_args.debug = True

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            benchmark = ConcreteBenchmark(basic_args)

        assert benchmark.debug is True

    def test_initializes_command_executor(self, basic_args, tmp_path):
        """Should initialize CommandExecutor."""
        basic_args.results_dir = str(tmp_path)

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            benchmark = ConcreteBenchmark(basic_args)

        assert benchmark.cmd_executor is not None


class TestBenchmarkMetadata:
    """Tests for Benchmark metadata property."""

    @pytest.fixture
    def benchmark(self, tmp_path):
        """Create a benchmark instance."""
        args = Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            command='run',
            num_processes=8,
            accelerator_type='h100'
        )

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            return ConcreteBenchmark(args, run_datetime="20250115_120000")

    def test_includes_benchmark_type(self, benchmark):
        """Metadata should include benchmark_type."""
        assert 'benchmark_type' in benchmark.metadata
        assert benchmark.metadata['benchmark_type'] == 'training'

    def test_includes_model(self, benchmark):
        """Metadata should include model."""
        assert 'model' in benchmark.metadata
        assert benchmark.metadata['model'] == 'unet3d'

    def test_includes_command(self, benchmark):
        """Metadata should include command."""
        assert 'command' in benchmark.metadata
        assert benchmark.metadata['command'] == 'run'

    def test_includes_run_datetime(self, benchmark):
        """Metadata should include run_datetime."""
        assert 'run_datetime' in benchmark.metadata
        assert benchmark.metadata['run_datetime'] == "20250115_120000"

    def test_includes_num_processes(self, benchmark):
        """Metadata should include num_processes."""
        assert 'num_processes' in benchmark.metadata
        assert benchmark.metadata['num_processes'] == 8

    def test_includes_accelerator(self, benchmark):
        """Metadata should include accelerator."""
        assert 'accelerator' in benchmark.metadata
        assert benchmark.metadata['accelerator'] == 'h100'

    def test_includes_result_dir(self, benchmark):
        """Metadata should include result_dir."""
        assert 'result_dir' in benchmark.metadata

    def test_includes_parameters(self, benchmark):
        """Metadata should include parameters."""
        assert 'parameters' in benchmark.metadata

    def test_includes_override_parameters(self, benchmark):
        """Metadata should include override_parameters."""
        assert 'override_parameters' in benchmark.metadata

    def test_includes_runtime(self, benchmark):
        """Metadata should include runtime."""
        assert 'runtime' in benchmark.metadata

    def test_includes_args(self, benchmark):
        """Metadata should include args."""
        assert 'args' in benchmark.metadata


class TestBenchmarkWriteMetadata:
    """Tests for write_metadata method."""

    @pytest.fixture
    def benchmark(self, tmp_path):
        """Create a benchmark instance."""
        args = Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            command='run',
            num_processes=8,
            accelerator_type='h100'
        )

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            return ConcreteBenchmark(args, run_datetime="20250115_120000")

    def test_writes_metadata_file(self, benchmark):
        """Should write metadata to JSON file."""
        benchmark.write_metadata()
        assert os.path.exists(benchmark.metadata_file_path)

    def test_metadata_file_is_valid_json(self, benchmark):
        """Metadata file should be valid JSON."""
        benchmark.write_metadata()

        with open(benchmark.metadata_file_path, 'r') as f:
            data = json.load(f)

        assert 'benchmark_type' in data
        assert data['model'] == 'unet3d'


class TestBenchmarkExecuteCommand:
    """Tests for _execute_command method."""

    @pytest.fixture
    def benchmark(self, tmp_path):
        """Create a benchmark instance."""
        args = Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            command='run',
            num_processes=8,
            accelerator_type='h100'
        )

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            return ConcreteBenchmark(args, run_datetime="20250115_120000")

    def test_what_if_mode_skips_execution(self, benchmark):
        """Should skip execution in what-if mode."""
        benchmark.args.what_if = True

        stdout, stderr, rc = benchmark._execute_command("echo hello")

        assert stdout == ""
        assert stderr == ""
        assert rc == 0

    def test_stores_executed_command(self, benchmark):
        """Should store executed command in __dict__."""
        benchmark.args.what_if = True
        benchmark._execute_command("echo hello")

        assert benchmark.executed_command == "echo hello"

    def test_writes_output_files_with_prefix(self, benchmark):
        """Should write stdout/stderr files when prefix provided."""
        benchmark.cmd_executor.execute = MagicMock(return_value=("output", "error", 0))

        benchmark._execute_command("echo hello", output_file_prefix="test_cmd")

        stdout_file = os.path.join(benchmark.run_result_output, "test_cmd.stdout.log")
        stderr_file = os.path.join(benchmark.run_result_output, "test_cmd.stderr.log")

        assert os.path.exists(stdout_file)
        assert os.path.exists(stderr_file)

        with open(stdout_file, 'r') as f:
            assert f.read() == "output"

    def test_appends_to_command_output_files(self, benchmark):
        """Should track output files in command_output_files list."""
        benchmark.cmd_executor.execute = MagicMock(return_value=("output", "error", 0))

        benchmark._execute_command("echo hello", output_file_prefix="test_cmd")

        assert len(benchmark.command_output_files) == 1
        assert 'command' in benchmark.command_output_files[0]
        assert 'stdout' in benchmark.command_output_files[0]
        assert 'stderr' in benchmark.command_output_files[0]


class TestBenchmarkVerifyBenchmark:
    """Tests for verify_benchmark method."""

    @pytest.fixture
    def benchmark(self, tmp_path):
        """Create a benchmark instance."""
        args = Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            command='run',
            num_processes=8,
            accelerator_type='h100',
            closed=True,
            allow_invalid_params=False
        )

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            return ConcreteBenchmark(args, run_datetime="20250115_120000")

    def test_returns_true_for_closed_verification(self, benchmark):
        """Should return True for CLOSED verification."""
        with patch('mlpstorage.benchmarks.base.BenchmarkVerifier') as mock_verifier_class:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.CLOSED
            mock_verifier_class.return_value = mock_verifier

            result = benchmark.verify_benchmark()

        assert result is True
        assert benchmark.verification == PARAM_VALIDATION.CLOSED

    def test_exits_for_invalid_verification(self, benchmark):
        """Should exit for INVALID verification."""
        with patch('mlpstorage.benchmarks.base.BenchmarkVerifier') as mock_verifier_class:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.INVALID
            mock_verifier_class.return_value = mock_verifier

            with pytest.raises(SystemExit):
                benchmark.verify_benchmark()

    def test_allows_invalid_with_flag(self, benchmark):
        """Should allow invalid params with --allow-invalid-params."""
        benchmark.args.allow_invalid_params = True

        with patch('mlpstorage.benchmarks.base.BenchmarkVerifier') as mock_verifier_class:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.INVALID
            mock_verifier_class.return_value = mock_verifier

            result = benchmark.verify_benchmark()

        assert result is True

    def test_exits_for_open_when_closed_required(self, benchmark):
        """Should exit for OPEN verification when closed is required."""
        benchmark.args.closed = True

        with patch('mlpstorage.benchmarks.base.BenchmarkVerifier') as mock_verifier_class:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.OPEN
            mock_verifier_class.return_value = mock_verifier

            with pytest.raises(SystemExit):
                benchmark.verify_benchmark()

    def test_allows_open_with_open_flag(self, benchmark):
        """Should allow OPEN verification with --open flag."""
        benchmark.args.closed = False

        with patch('mlpstorage.benchmarks.base.BenchmarkVerifier') as mock_verifier_class:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.OPEN
            mock_verifier_class.return_value = mock_verifier

            result = benchmark.verify_benchmark()

        assert result is True

    def test_returns_true_with_closed_false_no_open_attr(self, tmp_path):
        """Should return True and warn when closed=False and no 'open' attr (default state)."""
        # This simulates the default state when neither --closed nor --open is passed
        # The CLI sets closed=False as default, and hasattr(args, 'open') is False
        args = Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            command='run',
            num_processes=8,
            accelerator_type='h100',
            closed=False,  # Default when neither flag is passed
            allow_invalid_params=False
        )
        # Note: 'open' attribute should NOT be present for this test

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            benchmark = ConcreteBenchmark(args, run_datetime="20250115_120000")

        with patch('mlpstorage.benchmarks.base.BenchmarkVerifier') as mock_verifier_class:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.OPEN
            mock_verifier_class.return_value = mock_verifier

            result = benchmark.verify_benchmark()

        # Should return True early with warning (line 170-172 in base.py)
        assert result is True


class TestBenchmarkRun:
    """Tests for run method."""

    @pytest.fixture
    def benchmark(self, tmp_path):
        """Create a benchmark instance."""
        args = Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            command='run',
            num_processes=8,
            accelerator_type='h100'
        )

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            return ConcreteBenchmark(args, run_datetime="20250115_120000")

    def test_calls_run_method(self, benchmark):
        """Should call _run method."""
        with patch.object(benchmark, '_run', return_value=0) as mock_run:
            benchmark.run()
            mock_run.assert_called_once()

    def test_tracks_runtime(self, benchmark):
        """Should track runtime."""
        with patch.object(benchmark, '_run', return_value=0):
            with patch('time.time', side_effect=[100.0, 105.0]):
                benchmark.run()

        assert benchmark.runtime == 5.0

    def test_returns_run_result(self, benchmark):
        """Should return result from _run."""
        with patch.object(benchmark, '_run', return_value=42):
            result = benchmark.run()

        assert result == 42


class TestBenchmarkGenerateOutputLocation:
    """Tests for generate_output_location method."""

    def test_raises_without_benchmark_type(self, tmp_path):
        """Should raise ValueError without BENCHMARK_TYPE."""
        args = Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            command='run',
            num_processes=8,
            accelerator_type='h100'
        )

        # Create a valid benchmark first
        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            benchmark = ConcreteBenchmark(args)

        # Then set BENCHMARK_TYPE to None on the instance
        benchmark.BENCHMARK_TYPE = None

        with pytest.raises(ValueError, match="No benchmark specified"):
            benchmark.generate_output_location()

    def test_calls_generate_output_location(self, tmp_path):
        """Should call generate_output_location function."""
        args = Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            command='run',
            num_processes=8,
            accelerator_type='h100'
        )

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            benchmark = ConcreteBenchmark(args, run_datetime="20250115_120000")
            mock_gen.reset_mock()

            result = benchmark.generate_output_location()

            mock_gen.assert_called_once_with(benchmark, "20250115_120000")


class TestBenchmarkIntegration:
    """Integration tests for Benchmark class."""

    def test_full_workflow(self, tmp_path):
        """Test full benchmark workflow."""
        args = Namespace(
            debug=False,
            verbose=False,
            what_if=True,  # Use what-if to avoid actual execution
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            command='run',
            num_processes=8,
            accelerator_type='h100',
            closed=True,
            allow_invalid_params=False
        )

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            output_dir = tmp_path / "training" / "run" / "unet3d" / "20250115_120000"
            mock_gen.return_value = str(output_dir)

            benchmark = ConcreteBenchmark(args, run_datetime="20250115_120000")

        # Verify benchmark
        with patch('mlpstorage.benchmarks.base.BenchmarkVerifier') as mock_verifier_class:
            mock_verifier = MagicMock()
            mock_verifier.verify.return_value = PARAM_VALIDATION.CLOSED
            mock_verifier_class.return_value = mock_verifier

            result = benchmark.verify_benchmark()
            assert result is True

        # Execute command (in what-if mode)
        stdout, stderr, rc = benchmark._execute_command("test command")
        assert rc == 0

        # Run benchmark
        result = benchmark.run()
        assert result == 0

        # Write metadata
        benchmark.write_metadata()
        assert os.path.exists(benchmark.metadata_file_path)
