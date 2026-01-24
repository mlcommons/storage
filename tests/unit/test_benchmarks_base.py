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
from mlpstorage.config import BENCHMARK_TYPES, PARAM_VALIDATION, EXEC_TYPE


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
            with patch.object(benchmark, '_collect_cluster_start'):
                with patch.object(benchmark, '_collect_cluster_end'):
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


class TestBenchmarkValidation:
    """Tests for benchmark validation integration."""

    @pytest.fixture
    def basic_args(self, tmp_path):
        """Create basic args for benchmark."""
        return Namespace(
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

    def test_validate_environment_called_on_run(self, basic_args, tmp_path):
        """Should call _validate_environment before _run when run() is called."""
        basic_args.results_dir = str(tmp_path)
        call_order = []

        class TrackingBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training

            def _validate_environment(self):
                call_order.append('validate')

            def _run(self):
                call_order.append('run')
                return 0

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            benchmark = TrackingBenchmark(basic_args)

        benchmark.run()

        assert call_order == ['validate', 'run'], "validate should be called before run"

    def test_validate_environment_can_be_overridden(self, basic_args, tmp_path):
        """Should allow subclasses to override _validate_environment."""
        basic_args.results_dir = str(tmp_path)
        validation_called = []

        class CustomValidationBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training

            def _validate_environment(self):
                validation_called.append('custom_validation')

            def _run(self):
                return 0

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            benchmark = CustomValidationBenchmark(basic_args)

        benchmark.run()

        assert 'custom_validation' in validation_called

    def test_validation_error_prevents_run(self, basic_args, tmp_path):
        """Should propagate validation errors and prevent _run from executing."""
        from mlpstorage.errors import DependencyError

        basic_args.results_dir = str(tmp_path)
        run_called = []

        class FailingValidationBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training

            def _validate_environment(self):
                raise DependencyError("Test dependency error")

            def _run(self):
                run_called.append('run')
                return 0

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            benchmark = FailingValidationBenchmark(basic_args)

        with pytest.raises(DependencyError):
            benchmark.run()

        assert run_called == [], "_run should NOT be called when validation fails"

    def test_base_validate_environment_is_noop(self, basic_args, tmp_path):
        """Base class _validate_environment should be a no-op (pass)."""
        basic_args.results_dir = str(tmp_path)

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            benchmark = ConcreteBenchmark(basic_args)

        # Should not raise any exception
        benchmark._validate_environment()

    def test_validation_error_preserves_type(self, basic_args, tmp_path):
        """Should preserve the specific error type from validation."""
        from mlpstorage.errors import ConfigurationError

        basic_args.results_dir = str(tmp_path)

        class ConfigErrorBenchmark(Benchmark):
            BENCHMARK_TYPE = BENCHMARK_TYPES.training

            def _validate_environment(self):
                raise ConfigurationError("Invalid configuration")

            def _run(self):
                return 0

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            benchmark = ConfigErrorBenchmark(basic_args)

        with pytest.raises(ConfigurationError):
            benchmark.run()


class TestBenchmarkCollectionSelection:
    """Tests for benchmark collection method selection."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        logger = MagicMock()
        logger.debug = MagicMock()
        logger.warning = MagicMock()
        logger.status = MagicMock()
        logger.verbose = MagicMock()
        logger.verboser = MagicMock()
        return logger

    def _create_benchmark_with_args(self, tmp_path, mock_logger, **kwargs):
        """Helper to create a benchmark with specific args."""
        defaults = {
            'hosts': None,
            'exec_type': None,
            'command': 'run',
            'debug': False,
            'verbose': False,
            'stream_log_level': 'INFO',
            'results_dir': str(tmp_path),
            'model': 'unet3d',
            'num_processes': 8,
            'what_if': True,  # Prevent actual execution
        }
        defaults.update(kwargs)
        args = Namespace(**defaults)

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            benchmark = ConcreteBenchmark(args, logger=mock_logger, run_datetime='20260124_120000')

        return benchmark

    def test_should_use_ssh_collection_no_hosts(self, tmp_path, mock_logger):
        """Test that SSH collection is not used when no hosts specified."""
        benchmark = self._create_benchmark_with_args(tmp_path, mock_logger, hosts=None)
        assert benchmark._should_use_ssh_collection() is False

    def test_should_use_ssh_collection_empty_hosts(self, tmp_path, mock_logger):
        """Test that SSH collection is not used when hosts list is empty."""
        benchmark = self._create_benchmark_with_args(tmp_path, mock_logger, hosts=[])
        assert benchmark._should_use_ssh_collection() is False

    def test_selects_ssh_collection_with_hosts_no_exec_type(self, tmp_path, mock_logger):
        """Test that SSH collection is used when hosts specified but no exec_type."""
        benchmark = self._create_benchmark_with_args(
            tmp_path, mock_logger,
            hosts=['node1', 'node2'],
            exec_type=None
        )
        assert benchmark._should_use_ssh_collection() is True

    def test_selects_ssh_collection_docker_exec_type(self, tmp_path, mock_logger):
        """Test that SSH collection is used for EXEC_TYPE.DOCKER (non-MPI)."""
        benchmark = self._create_benchmark_with_args(
            tmp_path, mock_logger,
            hosts=['node1', 'node2'],
            exec_type=EXEC_TYPE.DOCKER
        )
        assert benchmark._should_use_ssh_collection() is True

    def test_should_not_use_ssh_collection_mpi_exec_type(self, tmp_path, mock_logger):
        """Test that SSH collection is not used when exec_type is MPI."""
        benchmark = self._create_benchmark_with_args(
            tmp_path, mock_logger,
            hosts=['node1', 'node2'],
            exec_type=EXEC_TYPE.MPI
        )
        assert benchmark._should_use_ssh_collection() is False

    def test_should_not_use_ssh_collection_for_datagen(self, tmp_path, mock_logger):
        """Test that SSH collection is skipped for datagen command."""
        benchmark = self._create_benchmark_with_args(
            tmp_path, mock_logger,
            hosts=['node1', 'node2'],
            command='datagen'
        )
        assert benchmark._should_use_ssh_collection() is False

    def test_should_not_use_ssh_collection_for_configview(self, tmp_path, mock_logger):
        """Test that SSH collection is skipped for configview command."""
        benchmark = self._create_benchmark_with_args(
            tmp_path, mock_logger,
            hosts=['node1', 'node2'],
            command='configview'
        )
        assert benchmark._should_use_ssh_collection() is False

    def test_should_not_use_ssh_collection_when_disabled(self, tmp_path, mock_logger):
        """Test that SSH collection is skipped when explicitly disabled."""
        benchmark = self._create_benchmark_with_args(
            tmp_path, mock_logger,
            hosts=['node1', 'node2'],
            skip_cluster_collection=True
        )
        assert benchmark._should_use_ssh_collection() is False

    def test_should_collect_cluster_info_no_hosts(self, tmp_path, mock_logger):
        """Test that MPI collection returns False when no hosts specified."""
        benchmark = self._create_benchmark_with_args(tmp_path, mock_logger, hosts=None)
        assert benchmark._should_collect_cluster_info() is False

    def test_should_collect_cluster_info_with_hosts_mpi(self, tmp_path, mock_logger):
        """Test that _should_collect_cluster_info is True with hosts and MPI exec_type."""
        benchmark = self._create_benchmark_with_args(
            tmp_path, mock_logger,
            hosts=['node1', 'node2'],
            exec_type=EXEC_TYPE.MPI
        )
        # _should_collect_cluster_info checks for hosts and command
        # exec_type check is in _collect_cluster_information
        assert benchmark._should_collect_cluster_info() is True


class TestBenchmarkClusterSnapshots:
    """Tests for cluster snapshot functionality."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        logger = MagicMock()
        logger.debug = MagicMock()
        logger.warning = MagicMock()
        logger.status = MagicMock()
        logger.verbose = MagicMock()
        logger.verboser = MagicMock()
        return logger

    @patch('mlpstorage.benchmarks.base.SSHClusterCollector')
    def test_collect_cluster_start_uses_ssh(self, mock_ssh_collector_class, tmp_path, mock_logger):
        """Test that _collect_cluster_start uses SSH when appropriate."""
        from mlpstorage.rules.models import ClusterInformation

        # Setup mock collector
        mock_collector = MagicMock()
        mock_collector.is_available.return_value = True
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {'localhost': {'hostname': 'localhost', 'meminfo': {}}}
        mock_result.timestamp = '2026-01-24T12:00:00Z'
        mock_result.errors = []
        mock_collector.collect.return_value = mock_result
        mock_ssh_collector_class.return_value = mock_collector

        args = Namespace(
            hosts=['localhost'],
            exec_type=None,  # Non-MPI
            command='run',
            debug=False,
            verbose=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            num_processes=8,
            what_if=True,
            ssh_username=None,
            cluster_collection_timeout=60,
        )

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            with patch.object(ClusterInformation, 'from_mpi_collection') as mock_from_mpi:
                mock_cluster_info = MagicMock()
                mock_cluster_info.num_hosts = 1
                mock_cluster_info.total_memory_bytes = 16 * 1024**3
                mock_from_mpi.return_value = mock_cluster_info

                benchmark = ConcreteBenchmark(args, logger=mock_logger, run_datetime='20260124_120000')
                benchmark._collect_cluster_start()

                mock_ssh_collector_class.assert_called_once()
                mock_collector.collect.assert_called_once()
                assert hasattr(benchmark, '_cluster_info_start')
                assert benchmark._collection_method == 'ssh'

    @patch('mlpstorage.benchmarks.base.SSHClusterCollector')
    def test_collect_cluster_end_creates_snapshots(self, mock_ssh_collector_class, tmp_path, mock_logger):
        """Test that _collect_cluster_end creates ClusterSnapshots."""
        from mlpstorage.rules.models import ClusterInformation, ClusterSnapshots

        # Setup mock collector
        mock_collector = MagicMock()
        mock_collector.is_available.return_value = True
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {'localhost': {'hostname': 'localhost', 'meminfo': {}}}
        mock_result.timestamp = '2026-01-24T12:00:00Z'
        mock_result.errors = []
        mock_collector.collect.return_value = mock_result
        mock_ssh_collector_class.return_value = mock_collector

        args = Namespace(
            hosts=['localhost'],
            exec_type=None,  # Non-MPI
            command='run',
            debug=False,
            verbose=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            num_processes=8,
            what_if=True,
            ssh_username=None,
            cluster_collection_timeout=60,
        )

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            with patch.object(ClusterInformation, 'from_mpi_collection') as mock_from_mpi:
                mock_cluster_info = MagicMock()
                mock_cluster_info.num_hosts = 1
                mock_cluster_info.total_memory_bytes = 16 * 1024**3
                mock_from_mpi.return_value = mock_cluster_info

                benchmark = ConcreteBenchmark(args, logger=mock_logger, run_datetime='20260124_120000')

                # Simulate start collection
                benchmark._collect_cluster_start()
                assert hasattr(benchmark, '_cluster_info_start')

                # Simulate end collection
                benchmark._collect_cluster_end()

                # Verify ClusterSnapshots was created
                assert hasattr(benchmark, 'cluster_snapshots')
                assert benchmark.cluster_snapshots is not None
                assert benchmark.cluster_snapshots.start is not None
                assert benchmark.cluster_snapshots.collection_method == 'ssh'

    def test_run_calls_start_and_end_collection(self, tmp_path, mock_logger):
        """Test that run() calls _collect_cluster_start and _collect_cluster_end."""
        call_order = []

        class TrackingBenchmark(ConcreteBenchmark):
            def _collect_cluster_start(self):
                call_order.append('start_collection')

            def _collect_cluster_end(self):
                call_order.append('end_collection')

            def _run(self):
                call_order.append('run')
                return 0

        args = Namespace(
            debug=False,
            verbose=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            command='run',
            num_processes=8,
            what_if=True,
        )

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            benchmark = TrackingBenchmark(args, logger=mock_logger, run_datetime='20260124_120000')

        benchmark.run()

        # Verify order: start_collection -> run -> end_collection
        assert call_order == ['start_collection', 'run', 'end_collection']

    def test_metadata_includes_cluster_snapshots(self, tmp_path, mock_logger):
        """Test that metadata property includes cluster_snapshots when available."""
        from mlpstorage.rules.models import ClusterSnapshots

        args = Namespace(
            debug=False,
            verbose=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            command='run',
            num_processes=8,
            what_if=True,
        )

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            benchmark = ConcreteBenchmark(args, logger=mock_logger, run_datetime='20260124_120000')

        # Mock ClusterSnapshots
        mock_start = MagicMock()
        mock_start.as_dict.return_value = {'total_memory_bytes': 16 * 1024**3}
        mock_snapshots = ClusterSnapshots(start=mock_start, collection_method='ssh')
        benchmark.cluster_snapshots = mock_snapshots

        metadata = benchmark.metadata
        assert 'cluster_snapshots' in metadata
        assert metadata['cluster_snapshots']['collection_method'] == 'ssh'

    def test_skips_end_collection_without_start(self, tmp_path, mock_logger):
        """Test that _collect_cluster_end does nothing if start collection was skipped."""
        args = Namespace(
            debug=False,
            verbose=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            command='run',
            num_processes=8,
            what_if=True,
            hosts=None,  # No hosts = no collection
        )

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            benchmark = ConcreteBenchmark(args, logger=mock_logger, run_datetime='20260124_120000')

        # Call end collection without start collection
        benchmark._collect_cluster_end()

        # Should not create cluster_snapshots
        assert not hasattr(benchmark, 'cluster_snapshots') or benchmark.cluster_snapshots is None
