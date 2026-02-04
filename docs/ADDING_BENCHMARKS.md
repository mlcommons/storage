# Adding New Benchmarks to MLPerf Storage

This guide explains how to add a new benchmark type to the MLPerf Storage
framework. Follow these steps to integrate a new benchmark that works with
the existing CLI, validation, and reporting infrastructure.

## Prerequisites

Before adding a new benchmark, ensure you understand:

1. The [Architecture](ARCHITECTURE.md) of MLPerf Storage
2. The benchmark you want to add and its execution requirements
3. What parameters need to be validated for CLOSED/OPEN submissions

## Step-by-Step Guide

### Step 1: Define Benchmark Type

Add your benchmark type to `mlpstorage/config.py`:

```python
# In mlpstorage/config.py

class BENCHMARK_TYPES(enum.Enum):
    training = "training"
    checkpointing = "checkpointing"
    vector_database = "vector_database"
    my_new_benchmark = "my_new_benchmark"  # Add your benchmark
```

### Step 2: Create Benchmark Class

Create a new file or add to an existing file in `mlpstorage/benchmarks/`:

```python
# mlpstorage/benchmarks/my_benchmark.py

from mlpstorage.benchmarks.base import Benchmark
from mlpstorage.config import BENCHMARK_TYPES


class MyNewBenchmark(Benchmark):
    """My new benchmark implementation.

    This benchmark measures [describe what it measures].
    """

    BENCHMARK_TYPE = BENCHMARK_TYPES.my_new_benchmark

    def __init__(self, args, logger=None, run_datetime=None, run_number=0,
                 cluster_collector=None, validator=None):
        super().__init__(args, logger, run_datetime, run_number,
                        cluster_collector, validator)

        # Initialize benchmark-specific attributes
        self.model = args.model
        self.data_dir = getattr(args, 'data_dir', None)

        # Load configuration
        self._load_config()

        # Collect cluster information if needed
        self.cluster_information = self._collect_cluster_information()

    def _load_config(self):
        """Load benchmark-specific configuration."""
        # Load YAML config, merge with CLI overrides, etc.
        pass

    def _run(self) -> int:
        """Execute the benchmark.

        Returns:
            Exit code (0 for success).
        """
        # 1. Verify parameters
        self.verify_benchmark()

        # 2. Generate the command
        cmd = self._generate_command()

        # 3. Execute the command
        self.logger.status(f"Running {self.BENCHMARK_TYPE.value} benchmark...")
        stdout, stderr, return_code = self._execute_command(
            cmd,
            output_file_prefix=f"{self.BENCHMARK_TYPE.value}_{self.run_datetime}",
            print_stdout=True,
            print_stderr=True
        )

        # 4. Process results (if needed)
        if return_code == 0:
            self._process_results()

        # 5. Write metadata
        self.write_metadata()

        return return_code

    def _generate_command(self) -> str:
        """Generate the benchmark execution command.

        Returns:
            Command string ready for execution.
        """
        cmd_parts = [
            "my_benchmark_tool",
            f"--model {self.model}",
        ]

        if self.data_dir:
            cmd_parts.append(f"--data-dir {self.data_dir}")

        return " ".join(cmd_parts)

    def _process_results(self):
        """Process and store benchmark results."""
        # Parse output files, extract metrics, etc.
        pass
```

### Step 3: Add CLI Arguments

Create argument builder in `mlpstorage/cli/`:

```python
# mlpstorage/cli/my_benchmark_args.py

from mlpstorage.cli.common_args import (
    HELP_MESSAGES,
    add_universal_arguments,
    add_mpi_arguments,
)


def add_my_benchmark_arguments(parser):
    """Add argument subparsers for my_new_benchmark program."""
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add subcommands
    run_parser = subparsers.add_parser("run", help="Run the benchmark")
    datasize_parser = subparsers.add_parser("datasize", help="Calculate data size")

    # Add common arguments to run command
    for _parser in [run_parser]:
        _parser.add_argument(
            '--model', '-m',
            required=True,
            help="Model to benchmark"
        )
        _parser.add_argument(
            '--data-dir', '-dd',
            type=str,
            help="Data directory"
        )
        _parser.add_argument(
            '--num-processes', '-np',
            type=int,
            default=1,
            help="Number of processes"
        )

        # Add MPI arguments if benchmark uses MPI
        add_mpi_arguments(_parser)

    # Add universal arguments to all subcommands
    for _parser in [run_parser, datasize_parser]:
        add_universal_arguments(_parser)
```

Update `mlpstorage/cli/__init__.py`:

```python
# Add to exports
from mlpstorage.cli.my_benchmark_args import add_my_benchmark_arguments

__all__ = [
    # ... existing exports ...
    'add_my_benchmark_arguments',
]
```

### Step 4: Register the Benchmark

Update `mlpstorage/benchmarks/__init__.py`:

```python
from mlpstorage.benchmarks.my_benchmark import MyNewBenchmark
from mlpstorage.registry import BenchmarkRegistry
from mlpstorage.cli import (
    PROGRAM_DESCRIPTIONS,
    add_my_benchmark_arguments,
)


def register_benchmarks():
    """Register all benchmark types with the BenchmarkRegistry."""
    # ... existing registrations ...

    BenchmarkRegistry.register(
        name='my_new_benchmark',
        benchmark_class=MyNewBenchmark,
        cli_builder=add_my_benchmark_arguments,
        description="Run my new benchmark",
        help_text="My new benchmark options"
    )
```

### Step 5: Update CLI Parser

Add the new benchmark to `mlpstorage/cli_parser.py`:

```python
def parse_arguments():
    # ... existing code ...

    # Add new benchmark parser
    my_benchmark_parsers = sub_programs.add_parser(
        "my_new_benchmark",
        description=PROGRAM_DESCRIPTIONS.get('my_new_benchmark', ''),
        help="My new benchmark options"
    )

    # ... update sub_programs_map ...
    sub_programs_map['my_new_benchmark'] = my_benchmark_parsers

    # Add arguments
    add_my_benchmark_arguments(my_benchmark_parsers)
```

### Step 6: Add Validation Rules (Optional)

If your benchmark needs CLOSED/OPEN validation, create rules:

```python
# mlpstorage/rules/run_checkers/my_benchmark.py

from mlpstorage.rules.base import RulesChecker
from mlpstorage.rules.issues import Issue
from mlpstorage.config import PARAM_VALIDATION


class MyBenchmarkRunChecker(RulesChecker):
    """Validation rules for individual my_new_benchmark runs."""

    def check(self, run_data):
        """Check if run meets CLOSED/OPEN requirements.

        Args:
            run_data: BenchmarkRunData instance.

        Returns:
            PARAM_VALIDATION enum value.
        """
        issues = []

        # Check required parameters
        if not run_data.parameters.get('required_param'):
            issues.append(Issue(
                severity='error',
                message='required_param is missing',
                allowed_in_open=False
            ))

        # Check numeric constraints
        num_processes = run_data.num_processes
        if num_processes < 4:
            issues.append(Issue(
                severity='warning',
                message=f'num_processes ({num_processes}) < 4',
                allowed_in_open=True
            ))

        self.issues = issues
        return self._determine_validation_state()

    def _determine_validation_state(self):
        """Determine validation state based on issues."""
        if any(not i.allowed_in_open for i in self.issues):
            return PARAM_VALIDATION.INVALID
        elif any(i.severity == 'warning' for i in self.issues):
            return PARAM_VALIDATION.OPEN
        return PARAM_VALIDATION.CLOSED
```

Register the checker in `mlpstorage/rules/verifier.py`:

```python
from mlpstorage.rules.run_checkers.my_benchmark import MyBenchmarkRunChecker

class BenchmarkVerifier:
    def _get_run_checker(self):
        checkers = {
            BENCHMARK_TYPES.training: TrainingRunChecker,
            BENCHMARK_TYPES.checkpointing: CheckpointingRunChecker,
            BENCHMARK_TYPES.my_new_benchmark: MyBenchmarkRunChecker,  # Add
        }
        return checkers.get(self.benchmark_type)
```

### Step 7: Update Main Entry Point

Update `mlpstorage/main.py` to handle the new benchmark:

```python
def main():
    # ... existing code ...

    if args.program == "my_new_benchmark":
        from mlpstorage.benchmarks.my_benchmark import MyNewBenchmark
        benchmark = MyNewBenchmark(args, logger=logger, run_datetime=run_datetime)
        result = benchmark.run()
        sys.exit(result)
```

### Step 8: Add Tests

Create tests for your benchmark:

```python
# tests/unit/test_my_benchmark.py

import pytest
from argparse import Namespace
from unittest.mock import MagicMock, patch

from mlpstorage.benchmarks.my_benchmark import MyNewBenchmark
from mlpstorage.config import BENCHMARK_TYPES


class TestMyNewBenchmark:
    """Tests for MyNewBenchmark."""

    @pytest.fixture
    def benchmark_args(self):
        """Create test arguments."""
        return Namespace(
            debug=False,
            verbose=False,
            what_if=True,
            model='test_model',
            data_dir='/data/test',
            results_dir='/tmp/results',
            closed=False,
            allow_invalid_params=True,
            stream_log_level='INFO',
        )

    def test_benchmark_type(self):
        """Benchmark has correct type."""
        assert MyNewBenchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.my_new_benchmark

    def test_generate_command(self, benchmark_args):
        """Test command generation."""
        with patch('mlpstorage.benchmarks.base.ClusterInformation'):
            benchmark = MyNewBenchmark(benchmark_args, logger=MagicMock())
            cmd = benchmark._generate_command()

            assert 'my_benchmark_tool' in cmd
            assert '--model test_model' in cmd

    def test_run_what_if_mode(self, benchmark_args):
        """Test run in what-if mode."""
        benchmark_args.what_if = True

        with patch('mlpstorage.benchmarks.base.ClusterInformation'):
            benchmark = MyNewBenchmark(benchmark_args, logger=MagicMock())
            result = benchmark.run()

            assert result == 0
```

## Configuration Files

If your benchmark needs workload configurations, add them to:

```
configs/
└── workloads/
    └── my_benchmark/
        ├── default.yaml
        ├── model_a.yaml
        └── model_b.yaml
```

Example configuration:

```yaml
# configs/workloads/my_benchmark/model_a.yaml
model:
  name: model_a
  batch_size: 32

dataset:
  format: npz
  num_files: 1000

workflow:
  iterations: 100
```

## Checklist

Before submitting your benchmark addition:

- [ ] Benchmark class created and inherits from `Benchmark`
- [ ] `BENCHMARK_TYPE` defined in `config.py`
- [ ] CLI arguments added to `cli/` package
- [ ] Benchmark registered in `benchmarks/__init__.py`
- [ ] CLI parser updated in `cli_parser.py`
- [ ] Main entry point updated in `main.py`
- [ ] Validation rules added (if needed)
- [ ] Unit tests created
- [ ] Integration tests created
- [ ] Configuration files added (if needed)
- [ ] Documentation updated

## Common Patterns

### Handling MPI Execution

```python
def _run(self):
    if self.args.exec_type == EXEC_TYPE.MPI:
        prefix = generate_mpi_prefix_cmd(
            self.args.mpi_bin,
            self.args.hosts,
            self.args.num_processes,
            self.args.oversubscribe,
            self.args.allow_run_as_root,
            self.args.mpi_params,
            self.logger
        )
        cmd = f"{prefix} {self._generate_command()}"
    else:
        cmd = self._generate_command()

    return self._execute_command(cmd)
```

### Loading YAML Configuration

```python
def _load_config(self):
    from mlpstorage.utils import read_config_from_file, update_nested_dict

    # Load base config
    config_path = f"workloads/my_benchmark/{self.model}.yaml"
    self.base_config = read_config_from_file(config_path)

    # Apply CLI overrides
    if self.args.params:
        overrides = self._parse_params(self.args.params)
        self.combined_params = update_nested_dict(self.base_config, overrides)
    else:
        self.combined_params = self.base_config
```

### Writing Custom Metadata

```python
@property
def metadata(self):
    base_metadata = super().metadata

    # Add benchmark-specific metadata
    base_metadata['my_specific_field'] = self.some_value
    base_metadata['metrics'] = self._collected_metrics

    return base_metadata
```

## Troubleshooting

### Common Issues

1. **Benchmark not found**: Ensure registration in `benchmarks/__init__.py`
2. **CLI arguments not recognized**: Check `cli_parser.py` and argument builder
3. **Validation always fails**: Check rule registration in `verifier.py`
4. **MPI errors**: Verify `exec_type` handling and MPI prefix generation

### Debugging Tips

- Use `--debug` flag for verbose logging
- Use `--what-if` to test command generation without execution
- Check metadata files for configuration issues
- Run with `--allow-invalid-params` to bypass validation during development
