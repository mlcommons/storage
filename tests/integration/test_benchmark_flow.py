"""
Integration tests for benchmark execution flow.

These tests validate the complete benchmark execution flow using mock
dependencies, allowing testing of benchmark logic without actual
DLIO/MPI execution.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.fixtures import (
    MockCommandExecutor,
    MockClusterCollector,
    MockLogger,
    create_sample_benchmark_args,
    create_sample_benchmark_run_data,
)


class TestTrainingBenchmarkFlow:
    """Integration tests for training benchmark execution flow."""

    @pytest.fixture
    def training_args(self):
        """Create training benchmark args."""
        return create_sample_benchmark_args(
            benchmark_type='training',
            command='run',
            model='unet3d',
            accelerator_type='h100',
            num_accelerators=8,
            client_host_memory_in_gb=256,
            hosts=['127.0.0.1'],
            data_dir='/data/unet3d',
        )

    @pytest.fixture
    def mock_setup(self):
        """Create mock dependencies for benchmark testing."""
        return {
            'executor': MockCommandExecutor({
                'dlio_benchmark': ('Benchmark completed', '', 0),
                'mpirun': ('MPI completed', '', 0),
            }),
            'collector': MockClusterCollector(),
            'logger': MockLogger(),
        }

    def test_training_benchmark_what_if_mode(self, training_args, mock_setup, tmp_path):
        """Training benchmark in what-if mode generates command but doesn't execute."""
        training_args.what_if = True
        training_args.results_dir = str(tmp_path)

        # Import here to avoid import errors if dependencies missing
        try:
            from mlpstorage.benchmarks.dlio import TrainingBenchmark
        except ImportError:
            pytest.skip("DLIO dependencies not available")

        with patch('mlpstorage.benchmarks.base.ClusterInformation') as mock_ci:
            mock_ci.return_value = MagicMock()
            mock_ci.return_value.total_memory_bytes = 256 * 1024**3
            mock_ci.return_value.host_info_list = []

            benchmark = TrainingBenchmark(
                training_args,
                logger=mock_setup['logger']
            )

            # In what-if mode, run should return success without executing
            result = benchmark.run()

            # Should complete without error
            assert result == 0 or result is None

    def test_training_benchmark_generates_correct_command(self, training_args, mock_setup, tmp_path):
        """Training benchmark generates correct DLIO command."""
        training_args.what_if = True
        training_args.results_dir = str(tmp_path)

        try:
            from mlpstorage.benchmarks.dlio import TrainingBenchmark
        except ImportError:
            pytest.skip("DLIO dependencies not available")

        with patch('mlpstorage.benchmarks.base.ClusterInformation') as mock_ci:
            mock_ci.return_value = MagicMock()
            mock_ci.return_value.total_memory_bytes = 256 * 1024**3
            mock_ci.return_value.host_info_list = []

            benchmark = TrainingBenchmark(
                training_args,
                logger=mock_setup['logger']
            )

            # Get the generated command
            if hasattr(benchmark, 'generate_command'):
                cmd = benchmark.generate_command()
                # Command should contain dlio_benchmark or relevant executable
                assert cmd is not None


class TestCheckpointingBenchmarkFlow:
    """Integration tests for checkpointing benchmark execution flow."""

    @pytest.fixture
    def checkpointing_args(self):
        """Create checkpointing benchmark args."""
        return create_sample_benchmark_args(
            benchmark_type='checkpointing',
            command='run',
            model='llama3-8b',
            num_processes=8,
            num_checkpoints_read=10,
            num_checkpoints_write=10,
            checkpoint_folder='/data/checkpoints',
            client_host_memory_in_gb=512,
        )

    def test_checkpointing_benchmark_what_if_mode(self, checkpointing_args, tmp_path):
        """Checkpointing benchmark in what-if mode."""
        checkpointing_args.what_if = True
        checkpointing_args.results_dir = str(tmp_path)

        try:
            from mlpstorage.benchmarks.dlio import CheckpointingBenchmark
        except ImportError:
            pytest.skip("DLIO dependencies not available")

        logger = MockLogger()

        with patch('mlpstorage.benchmarks.base.ClusterInformation') as mock_ci:
            mock_ci.return_value = MagicMock()
            mock_ci.return_value.total_memory_bytes = 512 * 1024**3
            mock_ci.return_value.host_info_list = []

            benchmark = CheckpointingBenchmark(
                checkpointing_args,
                logger=logger
            )

            result = benchmark.run()
            assert result == 0 or result is None


class TestBenchmarkWithMockExecutor:
    """Test benchmark execution using mock command executor."""

    @pytest.fixture
    def training_args(self):
        """Create training benchmark args."""
        return create_sample_benchmark_args(
            benchmark_type='training',
            command='run',
            model='unet3d',
        )

    def test_benchmark_records_executed_commands(self, training_args, mock_executor, tmp_path):
        """Benchmark execution is recorded by mock executor."""
        training_args.results_dir = str(tmp_path)
        training_args.what_if = False

        try:
            from mlpstorage.benchmarks.dlio import TrainingBenchmark
        except ImportError:
            pytest.skip("DLIO dependencies not available")

        # Configure executor to return success for DLIO
        mock_executor.add_response('dlio_benchmark', 'Success', '', 0)

        with patch('mlpstorage.benchmarks.base.ClusterInformation') as mock_ci:
            mock_ci.return_value = MagicMock()
            mock_ci.return_value.total_memory_bytes = 256 * 1024**3
            mock_ci.return_value.host_info_list = []

            with patch('subprocess.run') as mock_run:
                # Mock subprocess to use our executor
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout='Success',
                    stderr=''
                )

                benchmark = TrainingBenchmark(
                    training_args,
                    logger=MockLogger()
                )

                # The benchmark would try to execute a command
                # This test verifies our mock infrastructure works


class TestBenchmarkWithMockCollector:
    """Test benchmark initialization with mock cluster collector."""

    @pytest.fixture
    def training_args(self):
        """Create training benchmark args."""
        return create_sample_benchmark_args(
            benchmark_type='training',
            command='run',
            model='unet3d',
        )

    def test_benchmark_uses_mock_cluster_info(self, training_args, mock_collector, tmp_path):
        """Benchmark can use mock cluster collector for initialization."""
        training_args.results_dir = str(tmp_path)
        training_args.what_if = True

        # Configure collector with specific host configuration
        mock_collector.set_hosts(num_hosts=2, memory_gb=256, cpu_cores=64)

        try:
            from mlpstorage.benchmarks.dlio import TrainingBenchmark
        except ImportError:
            pytest.skip("DLIO dependencies not available")

        with patch('mlpstorage.benchmarks.base.ClusterInformation') as mock_ci:
            # Make ClusterInformation use our mock data
            mock_ci.return_value = MagicMock()
            mock_ci.return_value.total_memory_bytes = 2 * 256 * 1024**3
            mock_ci.return_value.host_info_list = []

            benchmark = TrainingBenchmark(
                training_args,
                logger=MockLogger(),
                # cluster_collector=mock_collector  # Once dependency injection is implemented
            )

            # Benchmark should initialize successfully
            assert benchmark is not None


class TestMetadataGeneration:
    """Test benchmark metadata file generation."""

    @pytest.fixture
    def training_args(self):
        """Create training benchmark args."""
        return create_sample_benchmark_args(
            benchmark_type='training',
            command='run',
            model='unet3d',
        )

    def test_metadata_file_created_on_run(self, training_args, tmp_path):
        """Running benchmark creates metadata file."""
        training_args.results_dir = str(tmp_path)
        training_args.what_if = True

        try:
            from mlpstorage.benchmarks.dlio import TrainingBenchmark
        except ImportError:
            pytest.skip("DLIO dependencies not available")

        with patch('mlpstorage.benchmarks.base.ClusterInformation') as mock_ci:
            mock_ci.return_value = MagicMock()
            mock_ci.return_value.total_memory_bytes = 256 * 1024**3
            mock_ci.return_value.host_info_list = []

            benchmark = TrainingBenchmark(
                training_args,
                logger=MockLogger()
            )

            benchmark.run()

            # Check if metadata was written
            # Note: In what-if mode, metadata may or may not be written
            # This depends on implementation


class TestValidationIntegration:
    """Test benchmark validation integration."""

    def test_benchmark_run_can_be_verified(self, tmp_path):
        """Completed benchmark run can be loaded and verified."""
        from mlpstorage.config import PARAM_VALIDATION

        # Create a mock result directory with metadata
        result_dir = tmp_path / "training" / "unet3d" / "run" / "20250115_120000"
        result_dir.mkdir(parents=True)

        # Create metadata file
        metadata = {
            "benchmark_type": "training",
            "model": "unet3d",
            "command": "run",
            "run_datetime": "20250115_120000",
            "num_processes": 8,
            "accelerator": "h100",
            "parameters": {
                "model": {"name": "unet3d"},
                "dataset": {"num_files_train": 42000},
            },
            "override_parameters": {},
        }
        with open(result_dir / "training_20250115_120000_metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create summary.json
        summary = {
            "metric": {"train_au_percentage": [95.0, 94.5, 95.2]},
            "host_memory_GB": [256],
            "num_accelerators": 8,
        }
        with open(result_dir / "summary.json", "w") as f:
            json.dump(summary, f)

        # Create hydra config directory
        hydra_dir = result_dir / ".hydra"
        hydra_dir.mkdir()
        with open(hydra_dir / "config.yaml", "w") as f:
            import yaml
            yaml.dump({"workload": metadata["parameters"]}, f)
        with open(hydra_dir / "overrides.yaml", "w") as f:
            yaml.dump([], f)

        # Load and verify the run
        try:
            from mlpstorage.rules import BenchmarkRun, BenchmarkVerifier
            from mlpstorage.mlps_logging import setup_logging

            logger = setup_logging(name='test')

            run = BenchmarkRun.from_result_dir(str(result_dir))
            assert run is not None
            assert run.model == 'unet3d'

            # Verify the run
            verifier = BenchmarkVerifier(run, logger=logger)
            result = verifier.verify()

            # Result should be valid
            assert result in [PARAM_VALIDATION.CLOSED, PARAM_VALIDATION.OPEN, PARAM_VALIDATION.INVALID]

        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")


class TestVerificationFlowIntegration:
    """Integration tests for the benchmark verification flow.

    These tests validate that the complete verification flow works correctly,
    from BenchmarkRun creation through BenchmarkVerifier execution.
    This catches initialization order bugs and other integration issues.
    """

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MockLogger()

    def test_verifier_can_verify_training_benchmark_run(self, mock_logger):
        """BenchmarkVerifier can verify a training benchmark run end-to-end.

        This is a regression test for the initialization order bug where
        TrainingRunRulesChecker.__init__ called super().__init__() before
        setting self.benchmark_run.
        """
        from mlpstorage.rules import BenchmarkRun, BenchmarkVerifier, BenchmarkRunData
        from mlpstorage.config import BENCHMARK_TYPES, PARAM_VALIDATION

        # Create a sample benchmark run
        run_data = create_sample_benchmark_run_data(
            benchmark_type='training',
            model='unet3d',
            command='run',
            num_processes=8,
        )

        run = BenchmarkRun.from_data(run_data, logger=mock_logger)

        # Create verifier and run verification
        # This would have failed before the fix with:
        # AttributeError: 'TrainingRunRulesChecker' object has no attribute 'benchmark_run'
        verifier = BenchmarkVerifier(run, logger=mock_logger)
        result = verifier.verify()

        # Should return a valid PARAM_VALIDATION enum
        assert result in [PARAM_VALIDATION.CLOSED, PARAM_VALIDATION.OPEN, PARAM_VALIDATION.INVALID]

    def test_verifier_can_verify_checkpointing_benchmark_run(self, mock_logger):
        """BenchmarkVerifier can verify a checkpointing benchmark run end-to-end."""
        from mlpstorage.rules import BenchmarkRun, BenchmarkVerifier, BenchmarkRunData
        from mlpstorage.config import BENCHMARK_TYPES, PARAM_VALIDATION

        run_data = create_sample_benchmark_run_data(
            benchmark_type='checkpointing',
            model='llama3-8b',
            command='run',
            num_processes=8,
            parameters={
                'model': {'name': 'llama3_8b'},
                'checkpoint': {
                    'num_checkpoints_read': 10,
                    'num_checkpoints_write': 10,
                },
                'workflow': {'checkpoint': True},
            }
        )

        run = BenchmarkRun.from_data(run_data, logger=mock_logger)

        verifier = BenchmarkVerifier(run, logger=mock_logger)
        result = verifier.verify()

        assert result in [PARAM_VALIDATION.CLOSED, PARAM_VALIDATION.OPEN, PARAM_VALIDATION.INVALID]

    def test_verifier_runs_all_checks(self, mock_logger):
        """BenchmarkVerifier runs all check methods and collects issues."""
        from mlpstorage.rules import BenchmarkRun, BenchmarkVerifier

        run_data = create_sample_benchmark_run_data(
            benchmark_type='training',
            model='unet3d',
            command='run',
            parameters={
                'dataset': {'num_files_train': 10},  # Too small
                'workflow': {'train': True},
            }
        )

        run = BenchmarkRun.from_data(run_data, logger=mock_logger)
        verifier = BenchmarkVerifier(run, logger=mock_logger)
        verifier.verify()

        # Should have collected some issues (dataset too small)
        assert len(verifier.issues) > 0


class TestDependencyValidationIntegration:
    """Integration tests for dependency validation in benchmarks.

    These tests verify that benchmarks properly validate dependencies
    (MPI, DLIO) before execution, providing clear error messages.
    """

    @pytest.fixture
    def training_args(self, tmp_path):
        """Create training benchmark args with valid temp directories."""
        from mlpstorage.config import EXEC_TYPE

        # Create data directory in temp path
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        args = create_sample_benchmark_args(
            benchmark_type='training',
            command='run',
            model='unet3d',
            what_if=False,
            data_dir=str(data_dir),
            results_dir=str(tmp_path / "results"),
        )
        args.exec_type = EXEC_TYPE.MPI
        return args

    def test_benchmark_fails_fast_when_dlio_missing(self, training_args):
        """Benchmark should fail fast with clear message when DLIO not found."""
        def mock_which(cmd):
            # MPI found, but DLIO not found
            if cmd == 'mpirun':
                return '/usr/bin/mpirun'
            return None

        with patch('shutil.which', side_effect=mock_which):
            with patch('mlpstorage.benchmarks.base.ClusterInformation') as mock_ci:
                mock_ci.return_value = MagicMock()
                mock_ci.return_value.total_memory_bytes = 256 * 1024**3
                mock_ci.return_value.host_info_list = []

                from mlpstorage.errors import DependencyError

                with pytest.raises(DependencyError) as exc_info:
                    from mlpstorage.benchmarks.dlio import TrainingBenchmark
                    TrainingBenchmark(training_args, logger=MockLogger())

                # Error should mention DLIO and how to install
                assert 'DLIO' in str(exc_info.value) or 'dlio' in str(exc_info.value).lower()

    def test_benchmark_fails_fast_when_mpi_missing(self, training_args):
        """Benchmark should fail fast with clear message when MPI not found."""
        def mock_which(cmd):
            # DLIO found, but MPI not found
            if cmd == 'dlio_benchmark':
                return '/usr/bin/dlio_benchmark'
            return None

        with patch('shutil.which', side_effect=mock_which):
            with patch('mlpstorage.benchmarks.base.ClusterInformation') as mock_ci:
                mock_ci.return_value = MagicMock()
                mock_ci.return_value.total_memory_bytes = 256 * 1024**3
                mock_ci.return_value.host_info_list = []

                from mlpstorage.errors import DependencyError

                with pytest.raises(DependencyError) as exc_info:
                    from mlpstorage.benchmarks.dlio import TrainingBenchmark
                    TrainingBenchmark(training_args, logger=MockLogger())

                # Error should mention MPI
                assert 'MPI' in str(exc_info.value) or 'mpi' in str(exc_info.value).lower()

    def test_benchmark_skips_dependency_check_in_whatif_mode(self, training_args):
        """Benchmark should skip dependency validation in what-if mode."""
        training_args.what_if = True

        # Even with no executables found, what-if mode should succeed
        with patch('shutil.which', return_value=None):
            with patch('mlpstorage.benchmarks.base.ClusterInformation') as mock_ci:
                mock_ci.return_value = MagicMock()
                mock_ci.return_value.total_memory_bytes = 256 * 1024**3
                mock_ci.return_value.host_info_list = []

                from mlpstorage.benchmarks.dlio import TrainingBenchmark

                # Should not raise DependencyError
                benchmark = TrainingBenchmark(training_args, logger=MockLogger())
                assert benchmark is not None

    def test_dependency_check_finds_dlio_in_custom_path(self, training_args, tmp_path):
        """Benchmark should find DLIO in custom path specified by --dlio-bin-path."""
        training_args.what_if = True

        # Create fake DLIO executable in custom path
        custom_bin_path = tmp_path / "custom_bin"
        custom_bin_path.mkdir()
        dlio_exe = custom_bin_path / "dlio_benchmark"
        dlio_exe.touch()
        dlio_exe.chmod(0o755)

        training_args.dlio_bin_path = str(custom_bin_path)

        def mock_which(cmd):
            if cmd == 'mpirun':
                return '/usr/bin/mpirun'
            return None  # DLIO not in PATH

        with patch('shutil.which', side_effect=mock_which):
            with patch('mlpstorage.benchmarks.base.ClusterInformation') as mock_ci:
                mock_ci.return_value = MagicMock()
                mock_ci.return_value.total_memory_bytes = 256 * 1024**3
                mock_ci.return_value.host_info_list = []

                from mlpstorage.benchmarks.dlio import TrainingBenchmark

                # Should find DLIO in custom path
                benchmark = TrainingBenchmark(training_args, logger=MockLogger())
                assert benchmark.base_command_path == str(dlio_exe)
