# Testing Patterns

**Analysis Date:** 2026-01-23

## Test Framework

**Runner:**
- pytest 7.0+
- Config: `pyproject.toml` `[tool.pytest.ini_options]`

**Assertion Library:**
- pytest native assertions
- No additional assertion libraries

**Run Commands:**
```bash
pytest tests/unit -v                    # Run all unit tests
pytest tests/integration -v             # Run integration tests
pytest tests/unit/test_cli.py -v        # Run single test file
pytest tests/unit -v --cov=mlpstorage   # With coverage
pytest tests/unit -v --cov=mlpstorage --cov-report=xml  # Coverage with XML report
```

**Configuration from `pyproject.toml`:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

## Test File Organization

**Location:**
- Tests in separate `tests/` directory at project root
- Mirrors source structure with `unit/` and `integration/` subdirectories

**Naming:**
- Test files: `test_{module_name}.py`
- Test classes: `Test{FunctionName}` or `Test{ClassName}`
- Test functions: `test_{behavior_description}`

**Structure:**
```
tests/
├── __init__.py
├── conftest.py                          # Shared fixtures
├── fixtures/                            # Test utilities
│   ├── __init__.py
│   ├── mock_collector.py
│   ├── mock_executor.py
│   ├── mock_logger.py
│   └── sample_data.py
├── integration/
│   ├── __init__.py
│   ├── test_benchmark_flow.py
│   └── test_full_submission.py
└── unit/
    ├── __init__.py
    ├── test_benchmarks_base.py
    ├── test_cli.py
    ├── test_config.py
    ├── test_rules_calculations.py
    ├── test_rules_checkers.py
    ├── test_rules_dataclasses.py
    ├── test_rules_extractors.py
    ├── test_utils.py
    └── ...
```

## Test Structure

**Suite Organization:**
```python
"""
Tests for CLI argument parsing in mlpstorage.cli module.

Tests cover:
- Training command argument parsing
- Checkpointing command argument parsing
- VectorDB command argument parsing
- Argument validation
"""

class TestHelpMessages:
    """Tests for help message dictionary."""

    def test_help_messages_is_dict(self):
        """help_messages should be a dictionary."""
        assert isinstance(help_messages, dict)


class TestAddUniversalArguments:
    """Tests for add_universal_arguments function."""

    @pytest.fixture
    def parser(self):
        """Create a basic parser."""
        return argparse.ArgumentParser()

    def test_adds_results_dir_argument(self, parser):
        """Should add --results-dir argument."""
        add_universal_arguments(parser)
        args = parser.parse_args(['--results-dir', '/test/path'])
        assert args.results_dir == '/test/path'
```

**Patterns:**
- Group related tests in classes prefixed with `Test`
- Use class-level docstrings to explain test scope
- Use pytest fixtures for setup (prefer over `setUp`/`tearDown`)
- Test function names describe the expected behavior

## Mocking

**Framework:** `unittest.mock` (MagicMock, patch)

**Custom Mock Classes in `tests/fixtures/`:**

**MockLogger (`mock_logger.py`):**
```python
class MockLogger:
    LOG_LEVELS = [
        'debug', 'info', 'warning', 'error', 'critical',
        'status', 'verbose', 'verboser', 'ridiculous', 'result'
    ]

    def __init__(self):
        self.messages: Dict[str, List[str]] = {level: [] for level in self.LOG_LEVELS}
        self.call_count: Dict[str, int] = {level: 0 for level in self.LOG_LEVELS}

    def has_message(self, level: str, substring: str) -> bool:
        return any(substring in msg for msg in self.messages.get(level, []))

    def assert_logged(self, level: str, substring: str):
        if not self.has_message(level, substring):
            raise AssertionError(f"Expected '{substring}' in {level} messages.")
```

**MockCommandExecutor (`mock_executor.py`):**
```python
class MockCommandExecutor:
    def __init__(
        self,
        responses: Optional[Dict[str, Tuple[str, str, int]]] = None,
        default_response: Tuple[str, str, int] = ('', '', 0)
    ):
        self.responses = responses or {}
        self.executed_commands: List[str] = []

    def execute(self, command: str, **kwargs) -> Tuple[str, str, int]:
        self.executed_commands.append(command)
        for pattern, response in self.responses.items():
            if self._matches(pattern, command):
                return response
        return self.default_response

    def assert_command_executed(self, pattern: str):
        # Raises AssertionError if not found
```

**MockClusterCollector (`mock_collector.py`):**
```python
class MockClusterCollector(ClusterCollectorInterface):
    def __init__(self, mock_data=None, should_fail=False, fail_message="Mock collector failure"):
        self.mock_data = mock_data or self._default_data()
        self.should_fail = should_fail
        self.collect_calls: List[Dict[str, Any]] = []

    def collect(self, hosts: List[str], timeout: int = 60) -> CollectionResult:
        self.collect_calls.append({'hosts': hosts, 'timeout': timeout})
        if self.should_fail:
            raise RuntimeError(self.fail_message)
        return CollectionResult(success=True, data=self.mock_data, ...)

    def set_hosts(self, num_hosts: int, memory_gb: int = 256, cpu_cores: int = 64):
        # Configure mock data for testing
```

**What to Mock:**
- External command execution (subprocess calls)
- MPI/cluster operations
- File system operations in certain cases
- Logger for output capture

**What NOT to Mock:**
- Pure logic functions (test actual behavior)
- Data classes/dataclasses
- Configuration parsing (use temp files instead)

## Fixtures and Factories

**Shared Fixtures in `tests/conftest.py`:**
```python
@pytest.fixture
def mock_logger():
    """Create a mock logger that captures all log calls."""
    logger = MagicMock()
    for level in ['debug', 'info', 'warning', 'error', 'critical',
                  'status', 'verbose', 'verboser', 'ridiculous', 'result']:
        setattr(logger, level, MagicMock())
    return logger


@pytest.fixture
def mock_executor():
    """Create a MockCommandExecutor for testing without subprocess calls."""
    return MockCommandExecutor()


@pytest.fixture
def mock_collector():
    """Create a MockClusterCollector for testing without MPI."""
    return MockClusterCollector()


@pytest.fixture
def training_run_args(training_datasize_args) -> Namespace:
    """Args for training run command."""
    training_datasize_args.command = 'run'
    training_datasize_args.num_accelerators = 8
    return training_datasize_args


@pytest.fixture
def temp_result_dir(tmp_path, sample_training_parameters):
    """Create a temporary result directory with mock DLIO output files."""
    result_dir = tmp_path / "training" / "unet3d" / "run" / "20250111_143022"
    result_dir.mkdir(parents=True)
    # Create summary.json, metadata files, .hydra configs...
    return result_dir
```

**Factory Functions in `tests/fixtures/sample_data.py`:**
```python
def create_sample_cluster_info(
    num_hosts: int = 2,
    memory_gb_per_host: int = 256,
    cpu_cores_per_host: int = 64,
    logger: Optional[Any] = None
) -> Any:
    """Create a sample ClusterInformation object for testing."""


def create_sample_benchmark_args(
    benchmark_type: str = 'training',
    command: str = 'run',
    model: str = 'unet3d',
    **kwargs
) -> Namespace:
    """Create sample benchmark arguments for testing."""


def create_sample_benchmark_run_data(
    benchmark_type: str = 'training',
    model: str = 'unet3d',
    command: str = 'run',
    metrics: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """Create a sample BenchmarkRunData for testing."""
```

**Location:**
- Shared fixtures: `tests/conftest.py`
- Mock implementations: `tests/fixtures/*.py`
- Sample data constants: `tests/fixtures/sample_data.py`

## Coverage

**Requirements:** Not enforced (no minimum threshold)

**Configuration from `pyproject.toml`:**
```toml
[tool.coverage.run]
source = ["mlpstorage"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

**View Coverage:**
```bash
pytest tests/unit -v --cov=mlpstorage --cov-report=html
# Open htmlcov/index.html
```

## Test Types

**Unit Tests (`tests/unit/`):**
- Test individual functions and classes in isolation
- Mock external dependencies
- Fast execution, no I/O
- Examples: `test_utils.py`, `test_config.py`, `test_cli.py`

**Integration Tests (`tests/integration/`):**
- Test component interactions
- May use real file system (via `tmp_path`)
- May skip if dependencies not available
- Examples: `test_benchmark_flow.py`, `test_full_submission.py`

```python
def test_training_benchmark_what_if_mode(self, training_args, mock_setup, tmp_path):
    """Training benchmark in what-if mode generates command but doesn't execute."""
    training_args.what_if = True
    training_args.results_dir = str(tmp_path)

    try:
        from mlpstorage.benchmarks.dlio import TrainingBenchmark
    except ImportError:
        pytest.skip("DLIO dependencies not available")

    # Test proceeds with mocked components
```

## Common Patterns

**Async Testing:**
- Not heavily used in this codebase
- CommandExecutor uses threading but tests are synchronous

**Error Testing:**
```python
def test_invalid_llm_model_exits(self):
    """Should exit for invalid LLM model."""
    args = argparse.Namespace(
        program='checkpointing',
        model='invalid-model',
        num_checkpoints_read=5,
        num_checkpoints_write=5
    )
    with pytest.raises(SystemExit):
        validate_args(args)


def test_insufficient_slots_raises_error(self, mock_logger):
    """Insufficient configured slots raises ValueError."""
    with pytest.raises(ValueError, match="not sufficient"):
        generate_mpi_prefix_cmd(
            mpi_cmd=MPIRUN,
            hosts=['host1:2', 'host2:2'],  # Only 4 slots
            num_processes=8,  # Need 8 processes
            ...
        )
```

**Parametrized Tests:**
```python
# Not heavily used but supported
@pytest.mark.parametrize("datetime_str,expected", [
    ("20250111_143022", True),
    ("invalid", False),
    ("", False),
])
def test_is_valid_datetime_format(self, datetime_str, expected):
    assert is_valid_datetime_format(datetime_str) == expected
```

**Skipping Tests:**
```python
try:
    from mlpstorage.benchmarks.dlio import TrainingBenchmark
except ImportError:
    pytest.skip("DLIO dependencies not available")
```

**Using Temporary Directories:**
```python
def test_writes_metadata_file(self, benchmark):
    """Should write metadata to JSON file."""
    benchmark.write_metadata()
    assert os.path.exists(benchmark.metadata_file_path)


@pytest.fixture
def benchmark(self, tmp_path):
    args = Namespace(results_dir=str(tmp_path), ...)
    with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
        mock_gen.return_value = str(tmp_path / "output")
        os.makedirs(tmp_path / "output", exist_ok=True)
        return ConcreteBenchmark(args)
```

**Testing with Mock Patches:**
```python
def test_creates_output_directory(self, basic_args, tmp_path):
    """Should create output directory."""
    basic_args.results_dir = str(tmp_path / "results")

    with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen:
        output_dir = str(tmp_path / "results" / "output")
        mock_gen.return_value = output_dir

        benchmark = ConcreteBenchmark(basic_args)

    assert os.path.exists(output_dir)
```

**Creating Concrete Implementations for Abstract Classes:**
```python
class ConcreteBenchmark(Benchmark):
    """Concrete implementation of abstract Benchmark for testing."""

    BENCHMARK_TYPE = BENCHMARK_TYPES.training

    def _run(self):
        """Concrete implementation of abstract _run method."""
        return 0
```

---

*Testing analysis: 2026-01-23*
