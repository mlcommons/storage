# Defining rules checks in rules.py

Short Version: Add new checks by adding methods to `TrainingRulesChecker` and `CheckpointingRulesChecker` in `rules.py` that start with `check_*` and return an `Issue` object or a list of `Issue` objects.

## RulesCheckers
In rules.py, there is a class per workload for checking rules:
- `TrainingRulesChecker`
- `CheckpointingRulesChecker`

These classes are subclasses of `RulesChecker`. The parent class has the following attributes:
- `self.benchmark_run`, a `BenchmarkRun` object.
- `self.issues`. a list of `Issue` objects

When a `RulesChecker` instance is run, all methods starting with `check_*` will run. Each `check_*` method is expected to operate on `self.benchmark_run` and return an `Issue` object or a list of `Issue` objects.

## Issues
`Issue` defines the results of a rules check and may be a "Non-Issue" or an issue that has verified rules compatibility.

```python
@dataclass
class Issue:
    validation: PARAM_VALIDATION
    message: str
    parameter: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None
```

Here's an example of `Issue` creation that results in 3 types of issues, OPEN, CLOSED, and INVALID:

```python
 def check_allowed_params(self) -> Optional[Issue]:
    """
    This method will verify that the only parameters that were set were the allowed parameters.
    Allowed for closed:
      - dataset.num_files_train
      - dataset.num_subfolders_train
      -
    :return:
    """
    closed_allowed_params = ['dataset.num_files_train', 'dataset.num_subfolders_train', 'dataset.data_folder',
                             'reader.read_threads', 'reader.computation_threads', 'reader.transfer_size',
                             'reader.prefetch_size', 'checkpoint.checkpoint_folder',
                             'storage.storage_type', 'storage.storage_root']
    open_allowed_params = ['framework', 'dataset.format', 'dataset.num_samples_per_file', 'reader.data_loader']
    issues = []
    for param, value in self.benchmark_run.override_parameters.items():
        self.logger.debug(f"Processing override parameter: {param} = {value}")
        if param in closed_allowed_params:
            issues.append(Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=f"Closed parameter override allowed: {param} = {value}",
                parameter="Overrode Parameters",
                actual=value
            ))
        elif param in open_allowed_params:
            issues.append(Issue(
                validation=PARAM_VALIDATION.OPEN,
                message=f"Open parameter override allowed: {param} = {value}",
                parameter="Overrode Parameters",
                actual=value
            ))
        else:
            issues.append(Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Disallowed parameter override: {param} = {value}",
                parameter="Overrode Parameters",
                expected="None",
                actual=value
            ))
    return issues
```

## BenchmarkRun
A `BenchmarkRun` object maps the logs of a completed run and the instance of an executing run to the same object. This allows checkers to run before a test is executed and as part of the submission checker after tests have executed.

The relevant attributes on a `BenchmarkRun`:
```python
class BenchmarkRun:
    """
    Represents a benchmark run with all parameters and system information.
    Can be constructed either from a benchmark instance or from result files.
    """
    def __init__(self, benchmark_result=None, benchmark_instance=None, logger=None):
        self.logger = logger
        
        # These will be set when the result or instance are processed
        self.benchmark_type = None
        self.model = None
        self.command = None
        self.num_processes = None
        self.parameters = dict()
        self.override_parameters = dict()
        self.system_info = None
        self.metrics = {}
        self._run_id = None
        self.run_datetime = None
        self.result_root_dir = None

        self.benchmark_result = benchmark_result
        self.benchmark_instance = benchmark_instance

        if benchmark_instance:
            self._process_benchmark_instance(benchmark_instance)
            self.post_execution = False
        elif benchmark_result:
            self._process_benchmark_result(benchmark_result)
            self.post_execution = True

        self._run_id = RunID(program=self.benchmark_type.name, command=self.command,  model=self.model,
                            run_datetime=self.run_datetime)
```

`.benchmark_type` is the enum:
```python
class BENCHMARK_TYPES(enum.Enum):
    training = "training"
    vector_database = "vector_database"
    checkpointing = "checkpointing"
```

- `.parameters` (Dict)
  - the parameters as were run and include params from the config files and from the CLI.
- `.override_parameters` (Dict) 
  - the parameters that overrode the config file and were set by the user.
- `.system_info`  (ClusterInformation)
  - a `ClusterInformation` object with information on the number of clients and amount of memory per client.
- `.metrics` (Dict)
  - DLIO metrics from the test

`self.benchmark_result` and `self.benchmark_instance` are the associated objects when deeper inspection needs to be run with the context of pre- vs post- execution.
