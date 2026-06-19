# mlpstorage(1) — MLPerf Storage Benchmark Suite

## CURRRENT STATUS

**This version is not the final version** - there will be at least a few more changes, but it is accurate for the current version of the `mlpstorage` command and repo contents.  Please execute a `git pull` periodically to get the latest updates.

## NAME

**mlpstorage** — orchestrate the MLPerf Storage benchmark suite: training, checkpointing, vector-database, and KV-cache I/O workloads, plus submission packaging and validation.

## SYNOPSIS

```
mlpstorage <mode> <benchmark> [<model|index>] <command> [<storage>] [OPTIONS]
mlpstorage reports reportgen [OPTIONS]
mlpstorage history (show|rerun) [OPTIONS]
mlpstorage lockfile (generate|verify) [OPTIONS]
mlpstorage validate <submission-dir> [OPTIONS]
mlpstorage rules-coverage [OPTIONS]
mlpstorage version
```

Where:

- `<mode>` is `closed`, `open`, or `whatif`
- `<benchmark>` is `training`, `checkpointing`, `vectordb`, or `kvcache`
- `<model|index>` is required by `training` (e.g. `unet3d`), `checkpointing` (e.g. `llama3-70b`), and `vectordb` (e.g. `DISKANN`); `kvcache` takes no model positional
- `<command>` is `datasize`, `datagen`, `run`, or `configview` (subset depending on benchmark)
- `<storage>` is `file` or `object` — required by `datagen`, `run`, and `configview` for the benchmarks that touch storage

## DESCRIPTION

`mlpstorage` is the official command-line driver for the MLPerf Storage benchmark suite. It characterizes the performance of storage systems under realistic machine-learning workloads and produces results in a structured layout ready for MLCommons submission.

The suite currently includes four benchmarks:

- **Training** — DLIO-based emulation of accelerator-driven training I/O for `unet3d` (closed/open) and `retinanet` (closed/open). Additional models (`cosmoflow`, `resnet50`, `dlrm`, `flux`) are exposed under `whatif` for planning.
- **Checkpointing** — DLIO-driven checkpoint write and read at LLM scale: `llama3-8b`, `llama3-70b`, `llama3-405b`, and `llama3-1t`.
- **VectorDB** — Vector-database search and ingest, currently targeting Milvus across `DISKANN`, `HNSW`, and `AISAQ` index types (with `IVF_FLAT`, `IVF_SQ8`, and `FLAT` available in open/whatif).
- **KV-Cache** — LLM inference KV-cache tiering across GPU, CPU, and NVMe, with simulated multi-tenant user load.

`mlpstorage` handles cluster collection, MPI orchestration, dataset sizing, dataset generation, benchmark execution with time-series host metrics, result aggregation, history tracking, and end-to-end submission validation.

### Relationship to DLIO

The training and checkpointing benchmarks delegate the actual I/O workload to **DLIO** (Deep Learning I/O), which `mlpstorage` invokes as a subprocess. `mlpstorage` selects a YAML workload template from `configs/dlio/workload/`, merges it with CLI arguments and any `--params` overrides, executes DLIO under MPI, then collects, organizes, and validates the output. VectorDB and KV-Cache do not use DLIO.

### Submission Workflow

A submission is a packaged directory that mirrors the `closed/` and/or `open/` hierarchy under a submitter name, containing:

- `code/` — frozen snapshot of the `mlpstorage` repository used to produce the results (MD5-verified against a reference checksum in closed)
- `systems/<system-name>.{yaml,pdf}` — machine-readable system description plus the human-readable companion
- `results/<system-name>/<benchmark>/<model>/...` — per-run output trees populated by `mlpstorage`

The typical end-to-end flow is:

1. Run `datasize` to learn how much storage the workload requires.
2. Run `datagen` to materialize the dataset on the target storage.
3. Run `run` six times for training (1 warmup + 5 measured) or as required by the benchmark.
4. Write the system description.
5. Run `mlpstorage validate` against the submission root.
6. Submit the resulting package to MLCommons.

## DESIGN PHILOSOPHY: CORRECT BY CONSTRUCTION

The CLI is structured so that an invocation that parses is, as far as is statically expressible, a legal invocation for the chosen submission mode. The argument tree enforces submission rules rather than relying on a post-hoc checker to find them.

Mechanisms used:

- **Mode as the outermost positional.** The very first token (`closed`, `open`, `whatif`) selects an entire subparser tree. Arguments that are illegal in closed (for example, arbitrary `--params` overrides on training, or non-canonical `--num-checkpoints-write` values on checkpointing, or `--config` on kvcache) are simply not registered on the closed parser. They cannot be supplied — argparse rejects them before any benchmark code runs.
- **Benchmark and model as positionals.** Only models valid for the chosen mode appear as subparsers. A user cannot type `mlpstorage closed training cosmoflow ...` because `cosmoflow` only exists under `whatif`.
- **Command as a positional.** `datasize`, `datagen`, `run`, and `configview` are distinct subparsers, so each command sees only the flags relevant to it. `datasize` does not accept storage-access flags; `datagen` and `run` do.
- **Storage protocol as a positional.** Commands that touch storage require `file` or `object` as a positional after the command name, making the access path explicit at the call site and visible in command history.
- **Mutually exclusive groups.** For example, VectorDB's `--runtime` and `--queries` are wired into an `add_mutually_exclusive_group()`, so only one can be supplied.
- **Pinned defaults in closed.** Closed kvcache pins `--gpu-mem-gb`, `--cpu-mem-gb`, `--duration`, `--trials`, `--seed`, `--rag-num-docs`, and several boolean knobs to their rules-mandated values, with no flag exposed to change them.
- **Post-parse validators.** What argparse cannot express (for example, "`--num-checkpoints-write` must be 10 or 0 in closed per Rules §4.7.1") is enforced by `validate_<benchmark>_arguments()` functions called immediately after parsing.
- **Environment validation.** Before a benchmark starts, `validate_benchmark_environment()` verifies SSH connectivity to client hosts, MPI availability, DLIO accessibility, and results-directory writability. `--skip-validation` disables this for debugging only.

The result is that a closed-mode command line is exactly the command line a closed-mode submission requires, and an attempt to deviate is rejected at the earliest possible moment.

## COMMAND STRUCTURE

```
mlpstorage
├── closed | open | whatif
│   ├── training
│   │   ├── unet3d | retinanet | (cosmoflow|resnet50|dlrm|flux in whatif)
│   │   │   ├── datasize
│   │   │   ├── datagen   (file|object)
│   │   │   ├── run       (file|object)
│   │   │   └── configview (file|object)
│   ├── checkpointing
│   │   ├── llama3-8b | llama3-70b | llama3-405b | llama3-1t
│   │   │   ├── datasize
│   │   │   ├── run       (file|object)
│   │   │   └── configview (file|object)
│   ├── vectordb
│   │   ├── DISKANN | HNSW | AISAQ | (IVF_FLAT|IVF_SQ8|FLAT in open/whatif)
│   │   │   ├── datasize
│   │   │   ├── datagen   (file|object)
│   │   │   └── run       (file|object)
│   └── kvcache
│       ├── datasize
│       └── run
├── reports reportgen
├── history (show | rerun)
├── lockfile (generate | verify)
├── validate <submission-dir>
├── rules-coverage
└── version
```

## DATA DIRECTORY (`--data-dir`)

The data directory is the on-storage workspace for generated datasets and checkpoints. It is read by `run` and written by `datagen`. Its layout is determined by the underlying DLIO workload template plus any `--params` overrides, but the canonical structure produced by the bundled templates is:

```
<data-dir>/
├── training/
│   └── <model>/                    e.g. unet3d, retinanet
│       └── <rank>/                 zero-padded process rank: 0000, 0001, ...
│           └── <data files>        .npz / .hdf5 / .tfrecord depending on model
└── checkpointing/
    └── <model>/                    e.g. llama3-70b
        └── <rank>/
            └── <checkpoint shards> .safetensors / .pt
```

The `data-dir` must live on the storage system under test. For closed training submissions, the generated dataset must total at least five times the client host memory (`--client-host-memory-in-gb`) to prevent the OS page cache from absorbing the workload; `datasize` exists specifically to compute and report this lower bound.

VectorDB does not use `--data-dir`; vectors are loaded directly into the database engine (Milvus) by `datagen`. KV-Cache does not use `--data-dir`; cache tiers reside in GPU/CPU memory and (optionally) `--cache-dir` on NVMe.

## RESULTS DIRECTORY (`--results-dir`)

The results directory accumulates every artifact produced by `mlpstorage` as each new invocation of `mlpstorage` executes. The default is `$MLPERF_RESULTS_DIR` if set, otherwise a temporary directory. Its layout is benchmark-specific but always organized so that it is a valid submission structure.  Unwanted results can simply be removed from the tree.

### Training results

```
<results-dir>/training/<model>/
├── datagen/
│   └── <YYYYMMDD_HHMMSS>/                directory bumps on collision
│       ├── training_datagen.stdout.log
│       ├── training_datagen.stderr.log
│       ├── *output.json
│       ├── *per_epoch_stats.json
│       ├── *summary.json
│       ├── dlio.log
│       └── dlio_config/{config,hydra,overrides}.yaml
└── run/
    ├── results.json                      aggregated across timestamped runs
    └── <YYYYMMDD_HHMMSS>/                one per run; closed requires 6
        ├── training_run.stdout.log
        ├── training_run.stderr.log
        ├── *output.json
        ├── *per_epoch_stats.json
        ├── *summary.json
        ├── dlio.log
        ├── dlio_config/{config,hydra,overrides}.yaml
        ├── training_<ts>_timeseries.json metrics; absent if --skip-timeseries
        └── training_<ts>_metadata.json   args, env, cluster info, status
```

### Checkpointing results

```
<results-dir>/checkpointing/<model>/
├── results.json
└── <YYYYMMDD_HHMMSS>/                    one for write phase, one for read
    ├── checkpointing_run.stdout.log
    ├── checkpointing_run.stderr.log
    ├── *output.json
    ├── *summary.json
    ├── dlio.log
    ├── dlio_config/{config,hydra,overrides}.yaml
    ├── checkpointing_<ts>_timeseries.json
    └── checkpointing_<ts>_metadata.json
```

### VectorDB results

```
<results-dir>/vector_database/<engine>/
├── datagen/<YYYYMMDD_HHMMSS>/
│   ├── stdout.log
│   ├── stderr.log
│   ├── summary.json
│   └── metadata.json
└── run/
    ├── results.json
    └── <YYYYMMDD_HHMMSS>/
        ├── simple_detailed.json          enhanced/sweep variants for --benchmark-mode sweep
        ├── stdout.log
        ├── stderr.log
        ├── summary.json
        ├── *timeseries.json
        └── *metadata.json
```

### KV-Cache results

```
<results-dir>/kv_cache/<YYYYMMDD_HHMMSS>/
├── results.json
├── option_1_results.json                 one per autoscaler option
├── option_2_results.json
├── option_3_results.json
├── kv_cache_<ts>_timeseries.json
├── kv_cache_<ts>_metadata.json
├── stdout.log
└── stderr.log
```

### Common artifacts

Every benchmark run writes:

- **`*_metadata.json`** — run timestamp, benchmark type, model, full command line, all CLI argument values, cluster information (collected by `cluster_collector.py` over MPI), MPI configuration, environment variables (credentials redacted), final status.
- **`*_timeseries.json`** — sampled host metrics (CPU, memory, disk I/O, network) collected at `--timeseries-interval` (default 10s) up to `--max-timeseries-samples` (default 3600). Single-host runs use a local collector; multi-host runs use SSH fan-out.
- **`stdout.log` / `stderr.log`** — streamed subprocess output captured by `CommandExecutor`.
- **`results.json`** — aggregated summary across all timestamped run directories, used by `reportgen`.
- **Command history** is appended to `<results-dir>/.history/` (consumed by `mlpstorage history`).

## VALIDATOR

`mlpstorage` ships a layered validation system whose ultimate authority is `Rules.md` in the repository root.

### Architecture

1. **CLI-level argument validators** (`mlpstorage_py/cli/*_args.py`).
   Functions named `validate_<benchmark>_arguments(args)` run immediately after argparse. They catch semantic constraints argparse cannot express, such as the closed-mode requirement that `--num-checkpoints-write` be either 10 or 0 (Rules §4.7.1).

2. **Environment validator** (`mlpstorage_py/dependency_check.py`).
   `validate_benchmark_environment()` is called before any benchmark instantiates. It checks DLIO binary availability, MPI launcher availability, SSH connectivity to every `--hosts` entry, and the writability of `--results-dir`. Bypass with `--skip-validation` for offline debugging.

3. **Run-rule checkers** (`mlpstorage_py/rules/run_checkers/`).
   Per-benchmark `RunRulesChecker` classes inspect the merged DLIO configuration before execution. They enforce rules such as:
   - `check_num_files_train()` — generated dataset has enough files to satisfy the 5× memory rule
   - `check_allowed_params()` — every `--params` override is in the closed allow-list or open allow-list as appropriate
   - `check_workflow_parameters()` — UNet3D requires `workflow.checkpoint=True`
   - `check_odirect_supported_model()` — `reader.odirect` is only valid for UNet3D
   - `check_model()` (checkpointing) — model is one of the four supported LLM sizes

4. **Submission checkers** (`mlpstorage_py/submission_checker/`).
   The `mlpstorage validate` command walks a submission directory and applies a battery of `@rule(rule_id=...)`-decorated checks organized by Rules.md section:
   - `DirectoryCheck` — Rules §2: required directories, code-tree MD5, system file presence
   - `TrainingCheck` — Rules §3: datasize report format, six-run cadence, allowed parameters
   - `CheckpointingCheck` — Rules §4: rank counts, write/read split, scaling
   - `VdbCheck` — Rules §5: vector-database compliance
   - `KVCacheCheck` — Rules §6: KV-cache compliance
   - `SystemYamlSchemaCheck` — JSON-schema validation of `systems/<name>.yaml`
   - `SubmissionStructureCheck` — top-level hierarchy and submitter naming

### Validation states

The `PARAM_VALIDATION` enum (`mlpstorage_py/config.py`) classifies each finding:

- **`CLOSED`** — passes closed-mode rules.
- **`OPEN`** — fails closed but is acceptable in an open submission.
- **`INVALID`** — fails regardless of mode.

The overall verdict for a parameter set is the most severe state encountered: any `INVALID` finding produces `INVALID`, any `OPEN` finding (without `INVALID`) produces `OPEN`, otherwise `CLOSED`.

### Invocation

Explicit validation of a submission package:

```
mlpstorage validate <submission-dir> [--submitters <names>] [--mlperf-version <ver>] \
                                     [--csv <out.csv>] [--skip-output-file]
```

Coverage audit of which Rules.md IDs have implementing checks:

```
mlpstorage rules-coverage [--rules-md <path>]
```

Run-rule checking happens implicitly via the per-benchmark `RunRulesChecker`. Environment validation happens automatically before every run unless `--skip-validation` is set.

## OPTIONS

The options below are grouped by scope. Flags that appear under multiple commands are documented once at their broadest scope and noted as such.

### Universal options (every command)

- **`--results-dir <path>`, `-rd <path>`**
  Root directory for all written artifacts. Required for any command that writes results. Defaults to `$MLPERF_RESULTS_DIR` if set, otherwise a system temp directory.

- **`--config-file <path>`, `-c <path>`**
  YAML file of argument overrides merged in *after* CLI parsing. Useful for keeping repeatable closed-submission knob settings in one place.

- **`--debug`**
  Verbose internal logging, full tracebacks on error.

- **`--verbose`**
  Increase user-facing log output without enabling internal debug.

- **`--stream-log-level <level>`**
  Threshold for log lines streamed live to the terminal (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Default `INFO`.

- **`--quiet`**
  Suppress the run-configuration summary table printed before execution.

- **`--dry-run`**
  Resolve the final configuration and print the command that would execute, then exit without running anything. Intended for sanity-checking command lines.

- **`--verify-lockfile <path>`**
  Validate installed Python packages against the supplied lockfile before executing the benchmark. Used to guarantee reproducibility against a frozen environment.

- **`--skip-validation`**
  Skip environment checks (MPI, SSH, DLIO). For debugging only; should never be used for a real submission.

### MPI options (training, checkpointing, kvcache)

- **`--mpi-bin <mpirun|mpiexec>`**
  Which MPI launcher to invoke. Default `mpirun`.

- **`--oversubscribe`**
  Permit MPI to allocate more ranks than physical cores. Useful for small test clusters.

- **`--allow-run-as-root`**
  Pass the corresponding flag through to MPI. Required in many container environments where the entrypoint runs as root.

- **`--mpi-btl <auto|vader|tcp>`**
  Byte-transport layer selection for single-host runs only. `auto` lets OpenMPI pick (works on most systems). `vader` forces POSIX shared memory (fastest, but may fail in containers or under root). `tcp` forces TCP loopback (universally compatible; recommended inside containers). No effect on multi-host runs.

- **`--mpi-params=<string>`**
  Pass-through string appended verbatim to the MPI launcher. Use the `=` form because the embedded flags begin with `-` and would otherwise confuse argparse: `--mpi-params="-genv FI_PROVIDER=tcp"`. May be supplied multiple times; values are concatenated.

### Training options

Required positionals: `<model>` then `<command>` and, for `datagen`/`run`/`configview`, the storage protocol `<file|object>`.

- **`--accelerator-type <type>`, `-at <type>`**
  Accelerator the workload should emulate (e.g. `h100`, `b200`, `mi355`). Determines per-accelerator access patterns and data rates. Required for `datasize`, `run`, `configview`.

- **`--num-accelerators <N>`, `-na <N>`**
  Number of simulated accelerators for `run`/`configview`. Ranks are distributed round-robin across `--hosts`.

- **`--max-accelerators <N>`, `-ma <N>`**
  Used by `datasize` to size a dataset capable of feeding up to N accelerators.

- **`--num-processes <N>`, `-np <N>`**
  Process count for `datagen`. Distributed round-robin across `--hosts`.

- **`--client-host-memory-in-gb <GB>`, `-cm <GB>`**
  RAM available on each client host. Closed submissions require a dataset ≥ 5× this value to defeat the page cache.

- **`--num-client-hosts <N>`, `-nc <N>`**
  Number of participating client hosts. Inferred from `--hosts` if omitted.

- **`--hosts <h1 h2 ...>`, `-s <h1,h2,...>`**
  Space- or comma-separated list of hostnames or IPs. Default `127.0.0.1`. The set of hosts is the universe of ranks for MPI dispatch.

- **`--exec-type <mpi|docker>`, `-et`**
  Execution backend. Default `mpi`. `docker` runs DLIO inside a container per host.

- **`--data-dir <path>`, `-dd <path>`**
  Filesystem location for generated data. Read by `run`, written by `datagen`.

- **`--dlio-bin-path <path>`, `-dp <path>`**
  Override the DLIO binary location. Default: alongside the `mlpstorage` binary.

- **`--params KEY=VALUE [KEY=VALUE ...]`, `-p`**
  Override arbitrary DLIO YAML parameters using dotted keys, e.g. `--params dataset.num_files_train=1500 reader.read_threads=8`. In closed mode only a published allow-list is accepted (see `rules/run_checkers/training.py`); open mode allows any DLIO parameter but each override is recorded for disclosure.

- **`--loops <N>`** *(open/whatif only)*
  Repeat the benchmark run N times. Default 1.

- **`--allow-invalid-params`, `-aip`** *(open/whatif only)*
  Bypass the parameter allow-list check. For experimentation; never appropriate for submission.

- **`--timeseries-interval <seconds>`** *(open/whatif only)*
  Sampling cadence for host metrics. Default 10.0. Lower values increase resolution and overhead.

- **`--skip-timeseries`** *(open/whatif only)*
  Disable host-metric collection entirely. Used when even minimal sampling perturbs the measurement.

- **`--max-timeseries-samples <N>`** *(open/whatif only)*
  Cap on retained samples per host (default 3600 = 10 hours at 10s).

### Checkpointing options

Required positionals: `<model>` (one of `llama3-8b`, `llama3-70b`, `llama3-405b`, `llama3-1t`) then `<command>` and, for `run`/`configview`, `<file|object>`.

- **`--model <name>`, `-m <name>`**
  LLM model to emulate. The selection fixes the tensor-parallel, pipeline-parallel, and data-parallel sizes and the per-rank checkpoint footprint.

- **`--client-host-memory-in-gb <GB>`, `-cm <GB>`**
  Client RAM, used as a sizing input and rule check.

- **`--num-processes <N>`, `-np <N>`**
  Number of accelerator ranks to emulate. Permitted values are model-specific (see `CHECKPOINT_RANKS_STRINGS` in `config.py`).

- **`--num-checkpoints-read <N>`, `-ncr <N>`**
  Number of checkpoint read iterations. Default 10.

- **`--num-checkpoints-write <N>`, `-ncw <N>`**
  Number of checkpoint write iterations. Default 10. In closed mode must be 10 or 0; supplying 0 lets the run cover only the read or only the write half, with the missing half supplied by a separate invocation (Rules §4.7.1).

- **`--checkpoint-folder <path>`, `-cf <path>`**
  Storage location for checkpoint files. Required for `run`.

- **`--hosts`, `--exec-type`, `--dlio-bin-path`**
  Same semantics as the training options of the same name.

- **`--loops`, `--allow-invalid-params`, `--params`** *(open/whatif only)*
  Same semantics as the training equivalents.

- **`--timeseries-interval`, `--skip-timeseries`, `--max-timeseries-samples`** *(open/whatif, run only)*
  Time-series collection knobs; same as training.

### VectorDB options

Required positionals: `<index_type>` then `<command>` and, for `datagen`/`run`, `<file|object>`.

- **`--vdb-engine <name>`**
  Vector-database engine identifier; recorded in the results path so multiple engines coexist in one `--results-dir`. Default `milvus`.

- **`--host <ip-or-name>`, `-s`**
  Database host. Default `127.0.0.1`.

- **`--port <int>`, `-p`**
  Database port. Default `19530`.

- **`--config <name-or-path>`**
  Named or file-path config for the VectorDB benchmark harness.

- **`--collection <name>`**
  Collection name to operate on inside the database.

Datasize options:

- **`--dimension <N>`**
  Vector dimensionality. Default 1536.

- **`--num-vectors <N>`**
  Number of vectors. Default 1,000,000.

- **`--index-type <type>`**
  Index used for the storage estimate. Closed accepts `DISKANN`, `HNSW`, `AISAQ`; open/whatif additionally accept `IVF_FLAT`, `IVF_SQ8`, `FLAT`.

- **`--num-shards <N>`**
  Collection shard count. Recommended one shard per million vectors. Default 1.

- **`--vector-dtype <type>`**
  Element type. Currently `FLOAT_VECTOR` only.

Datagen options (in addition to the datasize options where applicable):

- **`--distribution <uniform|normal|zipfian>`**
  Source distribution for synthetic vectors. Default `uniform`.

- **`--batch-size <N>`**
  Vectors per insertion call. Default 1000.

- **`--chunk-size <N>`**
  Vectors generated in memory per chunk. Default 10000.

- **`--force`**
  Drop and recreate the collection if it exists.

Run options:

- **`--num-query-processes <N>`**
  Parallel query workers. Default 1.

- **`--batch-size <N>`**
  Queries per call per worker. Default 1.

- **`--report-count <N>`**
  Batches between progress lines. Default 100.

- **`--benchmark-mode <timed|query_count|sweep>`**
  Selects the benchmark harness: `timed` and `query_count` use the simple bench; `sweep` uses the enhanced/parameter-sweep bench. Default `timed`.

- **`--vector-dim <N>`**
  Dimensionality used when generating query vectors. Default 1536.

- **`--search-limit <N>`**
  Top-K returned per query. Default 10.

- **`--search-ef <N>`**
  ANN `ef` search-time parameter. Default 200.

- **`--gt-collection <name>`**
  Ground-truth FLAT collection used for recall computation. Defaults to `<collection>_flat_gt`.

- **`--num-query-vectors <N>`**
  Number of deterministic query vectors generated for recall. Default 1000.

- **`--recall-k <N>`**
  K for recall@K. Defaults to `--search-limit`.

- **`--runtime <seconds>`** *(mutually exclusive with `--queries`)*
  Run for a fixed wall-clock duration.

- **`--queries <N>`** *(mutually exclusive with `--runtime`)*
  Run for a fixed total query count. In distributed mode this is the global count, split across MPI ranks.

Distributed VectorDB (datagen and run):

- **`--distributed`**
  Launch under MPI across one or more benchmark client hosts.

- **`--hosts <list>`**
  Benchmark client hosts. *Not* the database host; that is `--host`.

- **`--npernode <N>`, `--num-processes-per-client <N>`**
  Ranks per client host. Default 1.

- **`--mpi-impl <mpich|openmpi>`**
  MPI dialect for the orchestrator. Default `mpich`.

- **`--coordination <filesystem|mpi>`**
  Cross-rank coordination backend. `filesystem` uses the shared results directory with marker files; `mpi` uses `mpi4py` bcast/barrier/gather.

- **`--rank-output-dir <path>`**
  Node-local per-rank scratch directory used with `--coordination mpi`. Default `/tmp/mlps_vdb`.

- **`--seed <N>`**
  Base random seed; effective seed per rank is `seed + rank`. Default 42.

- **`--ready-timeout <seconds>`**
  Maximum time to wait for ranks to synchronize. Default 7200.

- **`--mpi-bin <mpirun|mpiexec>`**
  As elsewhere. Default `mpiexec` for VectorDB.

Open/whatif VectorDB extras:

- **`--loops`, `--allow-invalid-params`, `--params`**
  As for training.

- **`--metric-type <COSINE|L2|IP>`** *(datagen)*
  Search metric for index construction. Default `COSINE`.

- **`--max-degree <N>`, `--search-list-size <N>`** *(datagen)*
  DiskANN tuning. Defaults 16 and 200.

- **`--M <N>`, `--ef-construction <N>`** *(datagen)*
  HNSW tuning. Defaults 16 and 200.

- **`--inline-pq <N>`** *(datagen)*
  AISAQ `inline_pq` parameter. Default 16.

- **`--monitor-interval <seconds>`** *(datagen)*
  Index-build progress polling interval. Default 5.

- **`--compact`** *(datagen)*
  Compact the collection after load.

- **`--timeseries-interval`, `--skip-timeseries`, `--max-timeseries-samples`** *(run only)*
  As for training.

### KV-Cache options

KV-cache has no model positional; the model is selected with `--model` (open/whatif only — closed pins it internally).

Closed pins the following at fixed values and does not expose flags to change them: `--gpu-mem-gb=16.0`, `--cpu-mem-gb=32.0`, `--duration=60`, `--generation-mode=realistic`, `--performance-profile=throughput`, `--disable-multi-turn=False`, `--disable-prefix-caching=False`, `--enable-rag=True`, `--rag-num-docs=10`, `--enable-autoscaling=True`, `--autoscaler-mode=qos`, `--seed=42`, `--trials=3`, `--inter-option-delay=20`.

Common:

- **`--cache-dir <path>`**
  NVMe tier directory. If omitted, a subdirectory of `--results-dir` is used.

Run (all modes):

- **`--kvcache-bin-path <path>`**
  Override the location of the `kv-cache.py` script. Auto-detected by default.

- **`--npernode <N>`, `--num-processes-per-client <N>`**
  KV-cache instances per host. Default 1.

- **`--exec-type <mpi|docker>`, `-et`**
  Execution backend. Default `mpi`.

- **`--num-processes <N>`, `-np <N>`**
  Total MPI ranks for distributed execution.

- **`--hosts <list>`, `-s`**
  Client hosts. Default `127.0.0.1`.

Run (open/whatif only):

- **`--model <name>`, `-m <name>`**
  One of `tiny-1b`, `mistral-7b`, `llama2-7b`, `llama3.1-8b` (default), `llama3.1-70b-instruct`.

- **`--num-users <N>`, `-nu <N>`**
  Concurrent simulated users. Default 100.

- **`--gpu-mem-gb <GB>`, `--cpu-mem-gb <GB>`**
  Sizes of the GPU and CPU cache tiers.

- **`--duration <seconds>`, `-d <seconds>`**
  Wall-clock duration per option.

- **`--generation-mode <none|fast|realistic>`**
  Token-generation simulation fidelity.

- **`--performance-profile <latency|throughput>`**
  Pass/fail criteria emphasis.

- **`--disable-multi-turn`**
  Force single-turn conversations.

- **`--disable-prefix-caching`**
  Turn off the prefix-cache optimization.

- **`--enable-rag` / `--rag-num-docs <N>`**
  Enable retrieval-augmented generation and set the per-query document count.

- **`--enable-autoscaling` / `--autoscaler-mode <qos|predictive>`**
  Enable the autoscaler and pick its mode.

- **`--seed <N>`, `--trials <N>`, `--inter-option-delay <seconds>`**
  Randomization and pacing controls.

- **`--config <path>`**
  Path to a `kv-cache` YAML config. Not valid in closed.

- **`--loops`, `--allow-invalid-params`, `--params`, `--timeseries-interval`, `--skip-timeseries`, `--max-timeseries-samples`**
  As for training.

### Reports

```
mlpstorage reports reportgen [--output-dir <path>] --results-dir <path>
```

- **`--output-dir <path>`**
  Destination for the generated submission report. Defaults to `--results-dir`.

- **`--results-dir <path>`, `-rd <path>`** (required)
  Results tree to summarize.

### History

```
mlpstorage history show  [-n <N>] [-i <ID>] --results-dir <path>
mlpstorage history rerun <ID>             --results-dir <path>
```

- **`show`**
  - **`--limit <N>`, `-n <N>`** — only the last N entries.
  - **`--id <N>`, `-i <N>`** — only the entry with this ID.
- **`rerun`**
  - **`<rerun_id>`** (positional, required) — ID of the historical command to re-execute.
- **`--results-dir`, `-rd`** (required) — points at the results tree whose history to consult; history is kept under `<results-dir>/.history/`.

### Lockfile

```
mlpstorage lockfile generate [-o <path>] [--extra <group>]... [--hashes]
                             [--python-version <ver>] [--pyproject <path>] [--all]
                             --results-dir <path>
mlpstorage lockfile verify   [-l <path>] [--skip <pkg>]... [--allow-missing] [--strict]
                             --results-dir <path>
```

Generate options:

- **`-o, --output <path>`** — output lockfile path. Default `requirements.txt`.
- **`--extra <group>`** — include an optional dependency group; repeatable (`--extra test --extra full`).
- **`--hashes`** — embed SHA-256 hashes (slower but more secure).
- **`--python-version <ver>`** — target Python version.
- **`--pyproject <path>`** — path to `pyproject.toml`. Default `pyproject.toml`.
- **`--all`** — generate both the base `requirements.txt` and a full `requirements-full.txt`.

Verify options:

- **`-l, --lockfile <path>`** — lockfile to check against. Default `requirements.txt`.
- **`--skip <pkg>`** — package name to ignore; repeatable.
- **`--allow-missing`** — tolerate packages absent from the environment.
- **`--strict`** — fail on any difference; default is fail only on version mismatch.

### Validate

```
mlpstorage validate <submission-dir> [--submitters <list>] [--mlperf-version <ver>]
                                     [--csv <path>] [--skip-output-file]
                                     [--reference-checksum <md5>]
```

- **`<submission-dir>`** (positional, required) — root of a submission package containing `closed/<submitter>` and/or `open/<submitter>` trees.
- **`--submitters <list>`** — comma-separated subset of submitters to check; default is every submitter found under the input directory.
- **`--mlperf-version <ver>`** — spec version the submission claims to conform to. Default is derived from this `mlpstorage` package's `major.minor`.
- **`--csv <path>`** — destination for the aggregate summary CSV. Default `summary.csv` in the current directory.
- **`--skip-output-file`** — do not emit per-submission log files alongside the CSV.
- **`--reference-checksum <md5>`** — override the bundled `REFERENCE_CHECKSUMS` used for the `code/` tree MD5 check.

Exit status: `0` if all submissions pass, `1` if any rule violation is detected.

### Rules-coverage

```
mlpstorage rules-coverage [--rules-md <path>]
```

- **`--rules-md <path>`** — `Rules.md` to audit. Default is the project-root copy.

Reports which Rules.md IDs are referenced by `@rule(rule_id=...)`-decorated checks and which are missing implementation. Intended for maintainers extending the submission checker.

## ENVIRONMENT

- **`MLPERF_RESULTS_DIR`** — default value for `--results-dir` when the flag is not supplied.
- **`MLPERF_DATA_DIR`** — fallback value for `--data-dir` for some commands.
- **`MPI_RUN_BIN`** — overrides the path used when invoking `mpirun`.

## EXIT STATUS

- `0` — success.
- non-zero — argument validation failed, an environment check failed, a benchmark subprocess returned non-zero, or `validate` found a rule violation.

## EXAMPLES

Size, generate, and run UNet3D in closed mode against a POSIX storage target:

```
mlpstorage closed training unet3d datasize \
    --accelerator-type b200 --max-accelerators 8 \
    --client-host-memory-in-gb 512 --results-dir /mnt/results

mlpstorage closed training unet3d datagen file \
    --num-processes 16 --data-dir /mnt/dataset \
    --client-host-memory-in-gb 512 --results-dir /mnt/results

mlpstorage closed training unet3d run file \
    --accelerator-type b200 --num-accelerators 8 \
    --client-host-memory-in-gb 512 \
    --data-dir /mnt/dataset --results-dir /mnt/results
```

Closed checkpointing for Llama 3 70B against object storage:

```
mlpstorage closed checkpointing llama3-70b run object \
    --num-processes 64 --client-host-memory-in-gb 1024 \
    --checkpoint-folder s3://bucket/checkpoints \
    --hosts host1,host2,host3,host4 \
    --results-dir /mnt/results
```

Open-mode VectorDB sweep against a remote Milvus:

```
mlpstorage open vectordb DISKANN run file \
    --host milvus.lab --port 19530 --collection bench_1m \
    --benchmark-mode sweep --runtime 600 \
    --num-query-processes 8 --results-dir /mnt/results
```

Validate a prepared submission directory:

```
mlpstorage validate /submissions/acme \
    --csv /submissions/acme.summary.csv
```

## FILES

- `<repo>/configs/dlio/workload/*.yaml` — bundled DLIO workload templates for training and checkpointing.
- `<repo>/Rules.md` — authoritative submission rules.
- `<results-dir>/.history/` — command history consumed by `mlpstorage history`.
- `<results-dir>/<benchmark>/...` — per-run output trees as documented under RESULTS DIRECTORY.
- `<submission-dir>/<mode>/<submitter>/{code,systems,results}/` — submission package layout consumed by `mlpstorage validate`.

## SEE ALSO

- `Rules.md` — definitive rule reference.
- `Submission_guidelines.md` — packaging and submission process.
- `README.md` — installation and quickstart.
- `DEVELOPMENT.md` — contributor documentation.
- DLIO — Deep Learning I/O benchmark (upstream workload engine).
