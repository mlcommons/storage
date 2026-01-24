"""
Tests for rules checker classes in mlpstorage.rules module.

Tests cover:
- RulesChecker base class behavior
- RunRulesChecker individual run checking
- MultiRunRulesChecker multi-run validation
- TrainingRunRulesChecker training-specific rules
- CheckpointingRunRulesChecker checkpointing-specific rules
- Submission rules checkers
"""

import pytest
from unittest.mock import MagicMock, patch

from mlpstorage.config import PARAM_VALIDATION, BENCHMARK_TYPES, UNET
from mlpstorage.rules import (
    Issue,
    RunID,
    RulesChecker,
    RunRulesChecker,
    MultiRunRulesChecker,
    TrainingRunRulesChecker,
    CheckpointingRunRulesChecker,
    CheckpointSubmissionRulesChecker,
    TrainingSubmissionRulesChecker,
    BenchmarkRun,
    BenchmarkRunData,
    ClusterInformation,
    HostInfo,
    HostMemoryInfo,
)


class TestRulesChecker:
    """Tests for RulesChecker base class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    def test_discovers_check_methods(self, mock_logger):
        """RulesChecker discovers all check_* methods."""
        class TestChecker(RulesChecker):
            def check_first(self):
                return None

            def check_second(self):
                return None

            def not_a_check(self):
                pass

        checker = TestChecker(logger=mock_logger)
        # Should find check_first and check_second
        method_names = [m.__name__ for m in checker.check_methods]
        assert 'check_first' in method_names
        assert 'check_second' in method_names
        assert 'not_a_check' not in method_names

    def test_run_checks_collects_issues(self, mock_logger):
        """run_checks collects issues from all check methods."""
        class TestChecker(RulesChecker):
            def check_returns_issue(self):
                return Issue(PARAM_VALIDATION.OPEN, "Test issue")

            def check_returns_list(self):
                return [
                    Issue(PARAM_VALIDATION.CLOSED, "Issue 1"),
                    Issue(PARAM_VALIDATION.CLOSED, "Issue 2")
                ]

            def check_returns_none(self):
                return None

        checker = TestChecker(logger=mock_logger)
        issues = checker.run_checks()

        assert len(issues) == 3
        assert issues[0].message == "Test issue"
        assert issues[1].message == "Issue 1"
        assert issues[2].message == "Issue 2"

    def test_run_checks_handles_exceptions(self, mock_logger):
        """run_checks catches exceptions and creates INVALID issues."""
        class TestChecker(RulesChecker):
            def check_raises_exception(self):
                raise ValueError("Something went wrong")

        checker = TestChecker(logger=mock_logger)
        issues = checker.run_checks()

        assert len(issues) == 1
        assert issues[0].validation == PARAM_VALIDATION.INVALID
        assert "failed with error" in issues[0].message

    def test_issues_are_reset_each_run(self, mock_logger):
        """run_checks resets issues on each call."""
        class TestChecker(RulesChecker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.call_count = 0

            def check_adds_issue(self):
                self.call_count += 1
                return Issue(PARAM_VALIDATION.CLOSED, f"Issue {self.call_count}")

        checker = TestChecker(logger=mock_logger)

        issues1 = checker.run_checks()
        assert len(issues1) == 1

        issues2 = checker.run_checks()
        assert len(issues2) == 1
        # Issues should be reset, not accumulated
        assert issues2[0].message == "Issue 2"


class TestRunRulesChecker:
    """Tests for RunRulesChecker class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def sample_benchmark_run(self, mock_logger):
        """Create a sample BenchmarkRun."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={
                "dataset": {"num_files_train": 400},
                "reader": {"read_threads": 8}
            },
            override_parameters={}
        )
        return BenchmarkRun.from_data(data, mock_logger)

    def test_stores_benchmark_run(self, mock_logger, sample_benchmark_run):
        """RunRulesChecker stores the benchmark run."""
        class TestChecker(RunRulesChecker):
            pass

        checker = TestChecker(sample_benchmark_run, logger=mock_logger)
        assert checker.benchmark_run is sample_benchmark_run


class TestMultiRunRulesChecker:
    """Tests for MultiRunRulesChecker class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def closed_run(self, mock_logger):
        """Create a run with CLOSED category."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={},
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        run.category = PARAM_VALIDATION.CLOSED
        return run

    @pytest.fixture
    def open_run(self, mock_logger):
        """Create a run with OPEN category."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143023",
            num_processes=8,
            parameters={},
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        run.category = PARAM_VALIDATION.OPEN
        return run

    @pytest.fixture
    def invalid_run(self, mock_logger):
        """Create a run with INVALID category."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143024",
            num_processes=8,
            parameters={},
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        run.category = PARAM_VALIDATION.INVALID
        return run

    def test_requires_list_or_tuple(self, mock_logger):
        """MultiRunRulesChecker requires list or tuple of runs."""
        with pytest.raises(TypeError, match="must be a list or tuple"):
            MultiRunRulesChecker("not a list", logger=mock_logger)

    def test_check_runs_valid_all_closed(self, mock_logger, closed_run):
        """check_runs_valid returns CLOSED when all runs are CLOSED."""
        checker = MultiRunRulesChecker([closed_run, closed_run], logger=mock_logger)
        issue = checker.check_runs_valid()

        assert issue.validation == PARAM_VALIDATION.CLOSED
        assert "CLOSED category" in issue.message

    def test_check_runs_valid_with_open(self, mock_logger, closed_run, open_run):
        """check_runs_valid returns OPEN when mix of OPEN and CLOSED."""
        checker = MultiRunRulesChecker([closed_run, open_run], logger=mock_logger)
        issue = checker.check_runs_valid()

        assert issue.validation == PARAM_VALIDATION.OPEN
        assert "OPEN or CLOSED" in issue.message

    def test_check_runs_valid_with_invalid(self, mock_logger, closed_run, invalid_run):
        """check_runs_valid returns INVALID when any run is INVALID."""
        checker = MultiRunRulesChecker([closed_run, invalid_run], logger=mock_logger)
        issue = checker.check_runs_valid()

        assert issue.validation == PARAM_VALIDATION.INVALID
        assert "Invalid runs found" in issue.message


class TestTrainingRunRulesChecker:
    """Tests for TrainingRunRulesChecker class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def valid_training_run(self, mock_logger):
        """Create a valid training benchmark run."""
        # Create cluster info with sufficient memory
        host_memory = HostMemoryInfo.from_total_mem_int(274877906944)  # 256 GB
        host = HostInfo(hostname='host1', memory=host_memory)
        cluster_info = ClusterInformation([host], mock_logger)

        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={
                "dataset": {"num_files_train": 56000, "record_length": 131072},
                "reader": {"read_threads": 8, "batch_size": 7}
            },
            override_parameters={},
            system_info=cluster_info
        )
        return BenchmarkRun.from_data(data, mock_logger)

    def test_check_benchmark_type_valid(self, mock_logger, valid_training_run):
        """check_benchmark_type returns None for training benchmark."""
        checker = TrainingRunRulesChecker(valid_training_run, logger=mock_logger)
        issue = checker.check_benchmark_type()

        assert issue is None

    def test_check_benchmark_type_invalid(self, mock_logger):
        """check_benchmark_type returns INVALID for non-training benchmark."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.checkpointing,  # Wrong type
            model="llama3-8b",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={
                "dataset": {"num_files_train": 400},
                "reader": {"read_threads": 8}
            },
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = TrainingRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_benchmark_type()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
        assert "Invalid benchmark type" in issue.message

    def test_check_num_files_train_missing_dataset(self, mock_logger):
        """check_num_files_train returns INVALID when dataset missing."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={},  # No dataset
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = TrainingRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_num_files_train()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
        assert "Missing dataset parameters" in issue.message

    def test_check_num_files_train_missing_num_files(self, mock_logger):
        """check_num_files_train returns INVALID when num_files_train missing."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={"dataset": {}},  # No num_files_train
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = TrainingRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_num_files_train()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
        assert "Missing num_files_train" in issue.message

    def test_check_allowed_params_closed_param(self, mock_logger):
        """check_allowed_params returns CLOSED for allowed closed params."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={
                "dataset": {"num_files_train": 400},
                "reader": {"read_threads": 8}
            },
            override_parameters={"dataset.num_files_train": "400"}  # Closed allowed
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = TrainingRunRulesChecker(run, logger=mock_logger)
        issues = checker.check_allowed_params()

        assert len(issues) == 1
        assert issues[0].validation == PARAM_VALIDATION.CLOSED

    def test_check_allowed_params_open_param(self, mock_logger):
        """check_allowed_params returns OPEN for open-category params."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={
                "dataset": {"num_files_train": 400},
                "reader": {"read_threads": 8}
            },
            override_parameters={"framework": "pytorch"}  # Open allowed
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = TrainingRunRulesChecker(run, logger=mock_logger)
        issues = checker.check_allowed_params()

        assert len(issues) == 1
        assert issues[0].validation == PARAM_VALIDATION.OPEN

    def test_check_allowed_params_invalid_param(self, mock_logger):
        """check_allowed_params returns INVALID for disallowed params."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={
                "dataset": {"num_files_train": 400},
                "reader": {"read_threads": 8}
            },
            override_parameters={"some.invalid.param": "value"}  # Not allowed
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = TrainingRunRulesChecker(run, logger=mock_logger)
        issues = checker.check_allowed_params()

        assert len(issues) == 1
        assert issues[0].validation == PARAM_VALIDATION.INVALID
        assert "Disallowed parameter" in issues[0].message

    def test_check_allowed_params_ignores_workflow(self, mock_logger):
        """check_allowed_params ignores workflow parameters."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={
                "dataset": {"num_files_train": 400},
                "reader": {"read_threads": 8}
            },
            override_parameters={"workflow.train": "true"}  # Should be ignored
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = TrainingRunRulesChecker(run, logger=mock_logger)
        issues = checker.check_allowed_params()

        assert len(issues) == 0

    def test_check_odirect_unsupported_model(self, mock_logger):
        """check_odirect_supported_model returns INVALID for non-unet3d."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="resnet50",  # Not unet3d
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={
                "dataset": {"num_files_train": 400},
                "reader": {"read_threads": 8, "odirect": True}  # odirect set
            },
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = TrainingRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_odirect_supported_model()

        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
        assert "odirect" in issue.message

    def test_check_odirect_valid_for_unet3d(self, mock_logger):
        """check_odirect_supported_model returns None for unet3d."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model=UNET,  # unet3d
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={
                "dataset": {"num_files_train": 400},
                "reader": {"read_threads": 8, "odirect": True}
            },
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        checker = TrainingRunRulesChecker(run, logger=mock_logger)
        issue = checker.check_odirect_supported_model()

        assert issue is None


class TestTrainingRunRulesCheckerNewModels:
    """Tests for TrainingRunRulesChecker with new models (DLRM, RetinaNet, Flux)."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    def create_benchmark_run(self, mock_logger, model, parameters=None, override_parameters=None):
        """Helper to create a BenchmarkRun for testing."""
        default_params = {
            'dataset': {
                'num_files_train': 1000,
                'data_folder': f'data/{model}/'
            },
            'reader': {
                'batch_size': 16
            },
            'workflow': {}
        }
        if parameters:
            for key, value in parameters.items():
                if key in default_params and isinstance(value, dict):
                    default_params[key].update(value)
                else:
                    default_params[key] = value

        # Create minimal required objects
        host_memory = HostMemoryInfo.from_total_mem_int(137438953472)  # 128 GB
        host_info = HostInfo(
            hostname="test-host",
            memory=host_memory
        )
        cluster_info = ClusterInformation([host_info], mock_logger)

        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model=model,
            command="run_benchmark",
            run_datetime="20250124_120000",
            num_processes=8,
            parameters=default_params,
            override_parameters=override_parameters or {},
            system_info=cluster_info
        )
        return BenchmarkRun.from_data(data, mock_logger)

    @pytest.mark.parametrize("model", ["dlrm", "retinanet", "flux"])
    def test_new_model_recognized(self, mock_logger, model):
        """New models are recognized by the checker."""
        benchmark_run = self.create_benchmark_run(mock_logger, model)
        checker = TrainingRunRulesChecker(benchmark_run, logger=mock_logger)

        issue = checker.check_model_recognized()
        assert issue is None, f"Model {model} should be recognized"

    @pytest.mark.parametrize("model", ["dlrm", "retinanet", "flux"])
    def test_new_model_odirect_not_supported(self, mock_logger, model):
        """odirect is not supported for new models."""
        benchmark_run = self.create_benchmark_run(
            mock_logger,
            model,
            parameters={'reader': {'odirect': True}}
        )
        checker = TrainingRunRulesChecker(benchmark_run, logger=mock_logger)

        issue = checker.check_odirect_supported_model()
        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
        assert "odirect" in issue.message.lower()

    @pytest.mark.parametrize("model", ["dlrm", "retinanet", "flux"])
    def test_new_model_no_checkpoint_requirement(self, mock_logger, model):
        """New models don't require checkpoint workflow."""
        benchmark_run = self.create_benchmark_run(
            mock_logger,
            model,
            parameters={'workflow': {'checkpoint': False}}
        )
        checker = TrainingRunRulesChecker(benchmark_run, logger=mock_logger)

        # Should not return INVALID for missing checkpoint
        issue = checker.check_workflow_parameters()
        # Either None or not INVALID (UNET requires checkpoint, others don't)
        if issue is not None:
            assert issue.validation != PARAM_VALIDATION.INVALID

    def test_unrecognized_model_invalid(self, mock_logger):
        """Unrecognized model returns INVALID issue."""
        benchmark_run = self.create_benchmark_run(mock_logger, "unknown_model")
        checker = TrainingRunRulesChecker(benchmark_run, logger=mock_logger)

        issue = checker.check_model_recognized()
        assert issue is not None
        assert issue.validation == PARAM_VALIDATION.INVALID
        assert "unknown_model" in issue.message

    def test_check_model_recognized_in_check_methods(self, mock_logger):
        """Verify check_model_recognized is discovered and executed by run_checks."""
        benchmark_run = self.create_benchmark_run(mock_logger, "dlrm")
        checker = TrainingRunRulesChecker(benchmark_run, logger=mock_logger)

        # Verify method is in check_methods list
        method_names = [m.__name__ for m in checker.check_methods]
        assert 'check_model_recognized' in method_names, \
            f"check_model_recognized should be discovered, found: {method_names}"

        # Verify it gets called during run_checks (no exception)
        issues = checker.run_checks()
        assert isinstance(issues, list)


class TestCheckpointSubmissionRulesChecker:
    """Tests for CheckpointSubmissionRulesChecker class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    def create_checkpointing_run(self, mock_logger, num_reads, num_writes):
        """Helper to create checkpointing run with specified read/write counts."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            model="llama3-8b",
            command="run",
            run_datetime="20250111_143022",
            num_processes=8,
            parameters={
                "checkpoint": {
                    "num_checkpoints_read": num_reads,
                    "num_checkpoints_write": num_writes
                }
            },
            override_parameters={}
        )
        run = BenchmarkRun.from_data(data, mock_logger)
        run.category = PARAM_VALIDATION.CLOSED
        return run

    def test_check_num_runs_exactly_ten(self, mock_logger):
        """check_num_runs returns CLOSED when exactly 10 reads and 10 writes."""
        # Create 10 runs with 1 read and 1 write each
        runs = [self.create_checkpointing_run(mock_logger, 1, 1) for _ in range(10)]

        checker = CheckpointSubmissionRulesChecker(runs, logger=mock_logger)
        issues = checker.check_num_runs()

        # Should have CLOSED issues for reads, writes, and total
        closed_issues = [i for i in issues if i.validation == PARAM_VALIDATION.CLOSED]
        assert len(closed_issues) == 3

    def test_check_num_runs_insufficient_reads(self, mock_logger):
        """check_num_runs returns INVALID for insufficient reads."""
        # Create 5 runs with only reads (insufficient)
        runs = [self.create_checkpointing_run(mock_logger, 1, 2) for _ in range(5)]

        checker = CheckpointSubmissionRulesChecker(runs, logger=mock_logger)
        issues = checker.check_num_runs()

        invalid_issues = [i for i in issues if i.validation == PARAM_VALIDATION.INVALID]
        read_issue = [i for i in invalid_issues if "read" in i.message.lower()]
        assert len(read_issue) == 1

    def test_check_num_runs_insufficient_writes(self, mock_logger):
        """check_num_runs returns INVALID for insufficient writes."""
        # Create runs with insufficient writes
        runs = [self.create_checkpointing_run(mock_logger, 2, 1) for _ in range(5)]

        checker = CheckpointSubmissionRulesChecker(runs, logger=mock_logger)
        issues = checker.check_num_runs()

        invalid_issues = [i for i in issues if i.validation == PARAM_VALIDATION.INVALID]
        write_issue = [i for i in invalid_issues if "write" in i.message.lower()]
        assert len(write_issue) == 1

    def test_check_num_runs_single_run_all_operations(self, mock_logger):
        """check_num_runs handles single run with all operations."""
        # Single run with 10 reads and 10 writes
        run = self.create_checkpointing_run(mock_logger, 10, 10)

        checker = CheckpointSubmissionRulesChecker([run], logger=mock_logger)
        issues = checker.check_num_runs()

        # All should be CLOSED
        assert all(i.validation == PARAM_VALIDATION.CLOSED for i in issues)


class TestRulesCheckerInitialization:
    """Tests for rules checker initialization order.

    These tests verify that the initialization bug (benchmark_run not set
    before parent __init__) is fixed. The bug occurred because:
    1. RunRulesChecker.__init__ called super().__init__() first
    2. RulesChecker.__init__ discovered check methods using dir(self)
    3. These check methods reference self.benchmark_run
    4. But self.benchmark_run wasn't set until after super().__init__()

    The fix was to set self.benchmark_run BEFORE calling super().__init__().
    """

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def sample_training_run(self, mock_logger):
        """Create a sample training BenchmarkRun."""
        host_memory = HostMemoryInfo.from_total_mem_int(274877906944)  # 256 GB
        host = HostInfo(hostname='host1', memory=host_memory)
        cluster_info = ClusterInformation([host], mock_logger)

        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.training,
            model="unet3d",
            command="run",
            run_datetime="20250123_120000",
            num_processes=2,
            parameters={
                "dataset": {"num_files_train": 168, "record_length": 131072},
                "reader": {"read_threads": 4, "batch_size": 7},
                "workflow": {"train": True, "checkpoint": False}
            },
            override_parameters={},
            system_info=cluster_info
        )
        return BenchmarkRun.from_data(data, mock_logger)

    @pytest.fixture
    def sample_checkpointing_run(self, mock_logger):
        """Create a sample checkpointing BenchmarkRun."""
        data = BenchmarkRunData(
            benchmark_type=BENCHMARK_TYPES.checkpointing,
            model="llama3-8b",
            command="run",
            run_datetime="20250123_120000",
            num_processes=8,
            parameters={
                "checkpoint": {
                    "num_checkpoints_read": 1,
                    "num_checkpoints_write": 1
                }
            },
            override_parameters={}
        )
        return BenchmarkRun.from_data(data, mock_logger)

    def test_training_checker_init_sets_benchmark_run_first(self, mock_logger, sample_training_run):
        """TrainingRunRulesChecker should have benchmark_run accessible during init.

        This test would have failed before the fix because benchmark_run
        was set after super().__init__(), but the parent's __init__ tried
        to access check methods that use self.benchmark_run.
        """
        # This should not raise AttributeError
        checker = TrainingRunRulesChecker(sample_training_run, logger=mock_logger)

        # Verify benchmark_run is set
        assert checker.benchmark_run is sample_training_run
        assert checker.benchmark_run.model == "unet3d"

    def test_checkpointing_checker_init_sets_benchmark_run_first(self, mock_logger, sample_checkpointing_run):
        """CheckpointingRunRulesChecker should have benchmark_run accessible during init."""
        checker = CheckpointingRunRulesChecker(sample_checkpointing_run, logger=mock_logger)

        assert checker.benchmark_run is sample_checkpointing_run
        assert checker.benchmark_run.model == "llama3-8b"

    def test_checker_can_run_checks_after_init(self, mock_logger, sample_training_run):
        """Checker should be able to run all checks after initialization.

        This is a regression test - before the fix, run_checks() would fail
        because check methods couldn't access self.benchmark_run.
        """
        checker = TrainingRunRulesChecker(sample_training_run, logger=mock_logger)

        # This should not raise AttributeError: 'TrainingRunRulesChecker' object
        # has no attribute 'benchmark_run'
        issues = checker.run_checks()

        # Should return a list (may have issues due to small dataset)
        assert isinstance(issues, list)

    def test_checker_properties_work_after_init(self, mock_logger, sample_training_run):
        """Checker properties should work correctly after initialization."""
        checker = TrainingRunRulesChecker(sample_training_run, logger=mock_logger)

        # These properties delegate to benchmark_run
        assert checker.parameters is not None
        assert checker.override_parameters is not None
        assert checker.system_info is not None


class TestTrainingSubmissionRulesChecker:
    """Tests for TrainingSubmissionRulesChecker class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    def test_supported_models_includes_training_models(self, mock_logger):
        """TrainingSubmissionRulesChecker has correct supported models."""
        from mlpstorage.config import MODELS

        # Create empty checker to check class attribute
        checker = TrainingSubmissionRulesChecker([], logger=mock_logger)

        assert 'unet3d' in checker.supported_models
        assert 'resnet50' in checker.supported_models
        assert 'cosmoflow' in checker.supported_models
