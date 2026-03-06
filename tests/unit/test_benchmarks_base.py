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
import time
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
        # Patch time.time specifically in the base module where runtime tracking happens.
        # Use a counter to provide specific values: start=100.0, end=105.0
        call_count = [0]

        def mock_time():
            call_count[0] += 1
            if call_count[0] == 1:
                return 100.0  # start_time
            elif call_count[0] == 2:
                return 105.0  # end time
            # Fallback for any additional calls
            return 105.0 + call_count[0]

        import mlpstorage.benchmarks.base as base_module
        original_time = base_module.time

        class MockTime:
            @staticmethod
            def time():
                return mock_time()

        with patch.object(benchmark, '_run', return_value=0):
            with patch.object(benchmark, '_collect_cluster_start'):
                with patch.object(benchmark, '_collect_cluster_end'):
                    with patch.object(benchmark, '_start_timeseries_collection'):
                        with patch.object(benchmark, '_stop_timeseries_collection'):
                            with patch.object(benchmark, 'write_timeseries_data'):
                                # Patch the time module in base
                                base_module.time = MockTime
                                try:
                                    benchmark.run()
                                finally:
                                    base_module.time = original_time

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


# =============================================================================
# Time-Series Collection Integration Tests
# =============================================================================

class TestTimeSeriesCollectionIntegration:
    """Tests for time-series collection integration in Benchmark base."""

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

    def _create_benchmark(self, tmp_path, mock_logger, **kwargs):
        """Helper to create a benchmark with specific args."""
        defaults = {
            'debug': False,
            'verbose': False,
            'stream_log_level': 'INFO',
            'results_dir': str(tmp_path),
            'model': 'unet3d',
            'command': 'run',
            'num_processes': 8,
            'what_if': False,
            'hosts': None,
            'skip_timeseries': False,
            'timeseries_interval': 10.0,
            'max_timeseries_samples': 100,
        }
        defaults.update(kwargs)
        args = Namespace(**defaults)

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            benchmark = ConcreteBenchmark(args, logger=mock_logger, run_datetime='20260124_120000')

        return benchmark

    def test_should_collect_timeseries_default_true(self, tmp_path, mock_logger):
        """Time-series collection should be enabled by default for run command."""
        benchmark = self._create_benchmark(tmp_path, mock_logger, command='run')

        assert benchmark._should_collect_timeseries() is True

    def test_should_collect_timeseries_skip_flag(self, tmp_path, mock_logger):
        """Time-series collection should be disabled when skip flag is set."""
        benchmark = self._create_benchmark(tmp_path, mock_logger, skip_timeseries=True)

        assert benchmark._should_collect_timeseries() is False

    def test_should_collect_timeseries_datagen_disabled(self, tmp_path, mock_logger):
        """Time-series collection should be disabled for datagen command."""
        benchmark = self._create_benchmark(tmp_path, mock_logger, command='datagen')

        assert benchmark._should_collect_timeseries() is False

    def test_should_collect_timeseries_whatif_disabled(self, tmp_path, mock_logger):
        """Time-series collection should be disabled in what-if mode."""
        benchmark = self._create_benchmark(tmp_path, mock_logger, what_if=True, command='run')

        assert benchmark._should_collect_timeseries() is False

    def test_start_timeseries_creates_collector(self, tmp_path, mock_logger):
        """_start_timeseries_collection should create a collector."""
        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            command='run',
            timeseries_interval=0.1,
            max_timeseries_samples=10
        )

        benchmark._start_timeseries_collection()

        assert benchmark._timeseries_collector is not None
        assert benchmark._timeseries_collector.is_running

        # Cleanup
        benchmark._timeseries_collector.stop()

    def test_start_timeseries_multihost_with_hosts(self, tmp_path, mock_logger):
        """Should use MultiHostTimeSeriesCollector when hosts are provided."""
        from mlpstorage.cluster_collector import MultiHostTimeSeriesCollector

        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            command='run',
            hosts=['localhost'],
            timeseries_interval=0.1
        )

        benchmark._start_timeseries_collection()

        assert isinstance(benchmark._timeseries_collector, MultiHostTimeSeriesCollector)

        # Cleanup
        benchmark._timeseries_collector.stop()

    def test_start_timeseries_singlehost_without_hosts(self, tmp_path, mock_logger):
        """Should use TimeSeriesCollector when no hosts provided."""
        from mlpstorage.cluster_collector import TimeSeriesCollector

        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            command='run',
            hosts=None,
            timeseries_interval=0.1
        )

        benchmark._start_timeseries_collection()

        assert isinstance(benchmark._timeseries_collector, TimeSeriesCollector)

        # Cleanup
        benchmark._timeseries_collector.stop()

    def test_stop_timeseries_creates_data(self, tmp_path, mock_logger):
        """_stop_timeseries_collection should create TimeSeriesData."""
        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            command='run',
            hosts=None,
            timeseries_interval=0.1
        )

        benchmark._start_timeseries_collection()
        time.sleep(0.25)
        benchmark._stop_timeseries_collection()

        assert benchmark._timeseries_data is not None
        assert benchmark._timeseries_data.num_samples >= 1

    def test_stop_timeseries_multihost_creates_data(self, tmp_path, mock_logger):
        """_stop_timeseries_collection should create TimeSeriesData for multi-host."""
        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            command='run',
            hosts=['localhost'],
            timeseries_interval=0.1
        )

        benchmark._start_timeseries_collection()
        time.sleep(0.25)
        benchmark._stop_timeseries_collection()

        assert benchmark._timeseries_data is not None
        assert 'localhost' in benchmark._timeseries_data.hosts_collected

    def test_write_timeseries_creates_file(self, tmp_path, mock_logger):
        """write_timeseries_data should create JSON file."""
        from mlpstorage.rules.models import TimeSeriesData, TimeSeriesSample

        benchmark = self._create_benchmark(tmp_path, mock_logger)

        # Setup output path
        benchmark.run_result_output = str(tmp_path)
        benchmark.timeseries_file_path = str(tmp_path / 'test_timeseries.json')

        # Create test data
        sample = TimeSeriesSample(
            timestamp='2026-01-24T12:00:00Z',
            hostname='testhost',
            vmstat={'key': 123}
        )
        benchmark._timeseries_data = TimeSeriesData(
            collection_interval_seconds=10.0,
            start_time='2026-01-24T12:00:00Z',
            end_time='2026-01-24T12:01:00Z',
            num_samples=1,
            samples_by_host={'testhost': [sample]},
            collection_method='local',
            hosts_requested=['testhost'],
            hosts_collected=['testhost']
        )

        benchmark.write_timeseries_data()

        assert os.path.exists(benchmark.timeseries_file_path)

        # Verify content
        with open(benchmark.timeseries_file_path) as f:
            data = json.load(f)
        assert data['num_samples'] == 1
        assert 'testhost' in data['samples_by_host']

    def test_timeseries_file_follows_naming_convention(self, tmp_path, mock_logger):
        """Time-series file should follow {benchmark_type}_{datetime}_timeseries.json pattern (HOST-04)."""
        benchmark = self._create_benchmark(tmp_path, mock_logger)
        benchmark.run_result_output = str(tmp_path)

        # Check filename follows pattern
        assert benchmark.timeseries_filename.endswith('_timeseries.json')
        assert benchmark.BENCHMARK_TYPE.value in benchmark.timeseries_filename
        assert benchmark.run_datetime in benchmark.timeseries_filename

    def test_metadata_includes_timeseries_reference(self, tmp_path, mock_logger):
        """metadata property should include time-series data reference (HOST-04)."""
        from mlpstorage.rules.models import TimeSeriesData, TimeSeriesSample

        benchmark = self._create_benchmark(tmp_path, mock_logger)
        benchmark.run_result_output = str(tmp_path)

        sample = TimeSeriesSample(
            timestamp='2026-01-24T12:00:00Z',
            hostname='testhost'
        )
        benchmark._timeseries_data = TimeSeriesData(
            collection_interval_seconds=10.0,
            start_time='2026-01-24T12:00:00Z',
            end_time='2026-01-24T12:01:00Z',
            num_samples=5,
            samples_by_host={'testhost': [sample]},
            collection_method='local',
            hosts_requested=['testhost'],
            hosts_collected=['testhost']
        )

        metadata = benchmark.metadata

        assert 'timeseries_data' in metadata
        assert metadata['timeseries_data']['num_samples'] == 5
        assert metadata['timeseries_data']['interval_seconds'] == 10.0
        assert metadata['timeseries_data']['file'] == benchmark.timeseries_filename

    def test_run_integrates_timeseries_collection(self, tmp_path, mock_logger):
        """run() should start and stop time-series collection."""
        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            command='run',
            hosts=None,
            timeseries_interval=0.1,
            skip_timeseries=False
        )

        # Track calls
        run_called = []
        original_run = benchmark._run

        def tracking_run():
            run_called.append('run')
            time.sleep(0.2)
            return 0

        benchmark._run = tracking_run

        result = benchmark.run()

        assert result == 0
        assert 'run' in run_called
        # Time-series collection should have been performed
        assert benchmark._timeseries_data is not None

    def test_timeseries_uses_background_thread(self, tmp_path, mock_logger):
        """Time-series collection should use background thread (HOST-05 architecture)."""
        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            command='run',
            hosts=None,
            timeseries_interval=0.1
        )

        benchmark._start_timeseries_collection()

        # Verify collector uses threading
        assert hasattr(benchmark._timeseries_collector, '_thread')
        assert benchmark._timeseries_collector._thread.name == 'TimeSeriesCollector'

        # Cleanup
        benchmark._timeseries_collector.stop()

    def test_timeseries_multihost_uses_correct_thread_name(self, tmp_path, mock_logger):
        """MultiHostTimeSeriesCollector should have correct thread name."""
        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            command='run',
            hosts=['localhost'],
            timeseries_interval=0.1
        )

        benchmark._start_timeseries_collection()

        # Verify collector uses threading
        assert hasattr(benchmark._timeseries_collector, '_thread')
        assert benchmark._timeseries_collector._thread.name == 'MultiHostTimeSeriesCollector'

        # Cleanup
        benchmark._timeseries_collector.stop()

    def test_timeseries_skipped_for_datagen_command(self, tmp_path, mock_logger):
        """Time-series collection should be skipped for datagen command."""
        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            command='datagen'
        )

        benchmark._start_timeseries_collection()

        assert benchmark._timeseries_collector is None

    def test_timeseries_skipped_for_configview_command(self, tmp_path, mock_logger):
        """Time-series collection should be skipped for configview command."""
        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            command='configview'
        )

        benchmark._start_timeseries_collection()

        assert benchmark._timeseries_collector is None

    def test_timeseries_stop_without_start_noop(self, tmp_path, mock_logger):
        """_stop_timeseries_collection should be no-op if collector is None."""
        benchmark = self._create_benchmark(tmp_path, mock_logger)
        benchmark._timeseries_collector = None

        # Should not raise
        benchmark._stop_timeseries_collection()

        assert benchmark._timeseries_data is None

    def test_write_timeseries_without_data_noop(self, tmp_path, mock_logger):
        """write_timeseries_data should be no-op if no data collected."""
        benchmark = self._create_benchmark(tmp_path, mock_logger)
        benchmark._timeseries_data = None

        # Should not raise
        benchmark.write_timeseries_data()

        assert not os.path.exists(benchmark.timeseries_file_path)


# =============================================================================
# Progress Integration Tests
# =============================================================================

class TestBenchmarkProgress:
    """Tests for progress indication integration in Benchmark base."""

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

    def _create_benchmark(self, tmp_path, mock_logger, **kwargs):
        """Helper to create a benchmark with specific args."""
        defaults = {
            'debug': False,
            'verbose': False,
            'stream_log_level': 'INFO',
            'results_dir': str(tmp_path),
            'model': 'unet3d',
            'command': 'run',
            'num_processes': 8,
            'what_if': True,
            'hosts': None,
            'skip_timeseries': True,  # Disable for faster tests
        }
        defaults.update(kwargs)
        args = Namespace(**defaults)

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            benchmark = ConcreteBenchmark(args, logger=mock_logger, run_datetime='20260125_120000')

        return benchmark

    @patch('mlpstorage.benchmarks.base.create_stage_progress')
    def test_run_shows_stage_progress(self, mock_stage_progress, tmp_path, mock_logger):
        """run() should use create_stage_progress with expected stages."""
        benchmark = self._create_benchmark(tmp_path, mock_logger)

        # Set up mock context manager
        mock_advance = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_advance)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_stage_progress.return_value = mock_cm

        benchmark.run()

        # Verify create_stage_progress was called with expected stages
        mock_stage_progress.assert_called_once()
        call_args = mock_stage_progress.call_args
        stages = call_args[0][0]  # First positional argument
        assert len(stages) == 4
        assert "Validating environment..." in stages
        assert "Collecting cluster info..." in stages
        assert "Running benchmark..." in stages
        assert "Processing results..." in stages

        # Verify advance_stage was called 4 times (once per stage)
        assert mock_advance.call_count == 4

    @patch('mlpstorage.progress.is_interactive_terminal', return_value=False)
    def test_run_non_interactive_logs_stages(self, mock_is_interactive, tmp_path, mock_logger):
        """run() should log stages in non-interactive mode via logger.status fallback."""
        benchmark = self._create_benchmark(tmp_path, mock_logger)

        benchmark.run()

        # In non-interactive mode, create_stage_progress calls logger.status for each stage
        # Verify at least one stage was logged via status()
        status_calls = [call for call in mock_logger.status.call_args_list]
        stage_logged = any('Stage' in str(call) for call in status_calls)
        assert stage_logged, f"Expected stage log messages, got: {status_calls}"

    @patch('mlpstorage.benchmarks.base.progress_context')
    def test_cluster_collection_shows_spinner(self, mock_progress_context, tmp_path, mock_logger):
        """_collect_cluster_start should use progress_context with spinner (total=None)."""
        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            hosts=['host1', 'host2'],
            skip_cluster_collection=False,
            exec_type=None  # Use SSH collection
        )

        # Set up mock context manager
        mock_update = MagicMock()
        mock_set_desc = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=(mock_update, mock_set_desc))
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_progress_context.return_value = mock_cm

        # Mock the SSH collection to avoid actual SSH calls
        with patch.object(benchmark, '_collect_via_ssh', return_value=None):
            benchmark._collect_cluster_start()

        # Verify progress_context was called with total=None (spinner)
        mock_progress_context.assert_called_once()
        call_kwargs = mock_progress_context.call_args[1]
        assert call_kwargs.get('total') is None

    @patch('mlpstorage.benchmarks.base.progress_context')
    def test_cluster_collection_updates_description_ssh(self, mock_progress_context, tmp_path, mock_logger):
        """_collect_cluster_start should update description to show SSH collection method."""
        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            hosts=['host1', 'host2'],
            skip_cluster_collection=False,
            exec_type=None  # Use SSH collection
        )

        # Set up mock context manager
        mock_update = MagicMock()
        mock_set_desc = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=(mock_update, mock_set_desc))
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_progress_context.return_value = mock_cm

        # Mock the SSH collection to avoid actual SSH calls
        with patch.object(benchmark, '_collect_via_ssh', return_value=None):
            benchmark._collect_cluster_start()

        # Verify set_desc was called with "Collecting via SSH..."
        mock_set_desc.assert_called_with("Collecting via SSH...")

    @patch('mlpstorage.benchmarks.base.progress_context')
    def test_cluster_collection_updates_description_mpi(self, mock_progress_context, tmp_path, mock_logger):
        """_collect_cluster_start should update description to show MPI collection method."""
        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            hosts=['host1', 'host2'],
            skip_cluster_collection=False,
            exec_type=EXEC_TYPE.MPI  # Use MPI collection
        )

        # Set up mock context manager
        mock_update = MagicMock()
        mock_set_desc = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=(mock_update, mock_set_desc))
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_progress_context.return_value = mock_cm

        # Mock the MPI collection to avoid actual MPI calls
        with patch.object(benchmark, '_collect_cluster_information', return_value=None):
            benchmark._collect_cluster_start()

        # Verify set_desc was called with "Collecting via MPI..."
        mock_set_desc.assert_called_with("Collecting via MPI...")

    @patch('mlpstorage.benchmarks.base.create_stage_progress')
    def test_run_progress_cleanup_on_exception(self, mock_stage_progress, tmp_path, mock_logger):
        """Stage progress context should properly exit even when _run() raises exception."""
        # Track whether __exit__ was called
        exit_called = []

        class ExceptionRaisingBenchmark(ConcreteBenchmark):
            def _run(self):
                raise RuntimeError("Test exception")

        mock_advance = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_advance)
        mock_cm.__exit__ = MagicMock(side_effect=lambda *args: exit_called.append(True) or False)
        mock_stage_progress.return_value = mock_cm

        args = Namespace(
            debug=False,
            verbose=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            model='unet3d',
            command='run',
            num_processes=8,
            what_if=True,
            hosts=None,
            skip_timeseries=True,
        )

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
            mock_gen.return_value = str(tmp_path / "output")
            os.makedirs(tmp_path / "output", exist_ok=True)
            benchmark = ExceptionRaisingBenchmark(args, logger=mock_logger, run_datetime='20260125_120000')

        with pytest.raises(RuntimeError, match="Test exception"):
            benchmark.run()

        # Verify the context manager's __exit__ was called (cleanup happened)
        assert len(exit_called) == 1

    @patch('mlpstorage.benchmarks.base.progress_context')
    def test_end_cluster_collection_shows_spinner(self, mock_progress_context, tmp_path, mock_logger):
        """_collect_cluster_end should use progress_context with spinner."""
        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            hosts=['host1'],
            skip_cluster_collection=False,
            exec_type=None  # Use SSH collection
        )

        # Set up mock context manager
        mock_update = MagicMock()
        mock_set_desc = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=(mock_update, mock_set_desc))
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_progress_context.return_value = mock_cm

        # Simulate that start collection was done
        benchmark._cluster_info_start = MagicMock()
        benchmark._collection_method = 'ssh'

        # Mock the SSH collection to avoid actual SSH calls
        with patch.object(benchmark, '_collect_via_ssh', return_value=None):
            benchmark._collect_cluster_end()

        # Verify progress_context was called with total=None (spinner)
        mock_progress_context.assert_called_once()
        call_kwargs = mock_progress_context.call_args[1]
        assert call_kwargs.get('total') is None

    def test_cluster_collection_skipped_logs_debug(self, tmp_path, mock_logger):
        """_collect_cluster_start should log debug when skipping collection."""
        benchmark = self._create_benchmark(
            tmp_path, mock_logger,
            hosts=None  # No hosts = no collection
        )

        benchmark._collect_cluster_start()

        # Should log debug message about skipping
        mock_logger.debug.assert_any_call('Skipping start cluster collection (conditions not met)')
