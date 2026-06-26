# mlpstorage(1) — MLPerf Storage Benchmark Suite

## CURRRENT STATUS

**This version is not the final version** - there will be at least a few more changes, but it is accurate for the current version of the `mlpstorage` command and repo contents.  Please execute a `git pull` periodically to get the latest updates.

## NAME

**mlpstorage** — orchestrate the MLPerf Storage benchmark suite: training, checkpointing, vector-database, and KV-cache I/O workloads, plus submission packaging and validation.

## SYNOPSIS

```
mlpstorage init <orgname> <path>
mlpstorage <mode> <benchmark> [<model|index>] <command> [<storage>] --systemname <name> [OPTIONS]
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
- `<orgname>` is the submitter / organization name pinned to the results-dir by `mlpstorage init`; `[A-Za-z0-9._-]+`, case-sensitive
- `<name>` (for `--systemname`) is the per-run system-under-test identifier; required on every emitting subcommand (`run`, `datagen`, `configview`, `reportgen`, `history`), and may be supplied via the `MLPERF_SYSTEMNAME` environment variable

Before any emitting subcommand can run, the `<results-dir>` must be initialized with `mlpstorage init`. The single bootstrap command `mlpstorage init <orgname> <path>` writes a `mlperf-results.yaml` sentinel that pins orgname to the directory; every later non-init command reads it as authoritative.

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
- **Orgname pinned to the results-dir by `mlpstorage init`.** There is no `--orgname` flag on any benchmark subcommand and no `MLPERF_ORGNAME` environment variable consulted by non-init commands. The results-dir is initialized exactly once with `mlpstorage init <orgname> <path>`, which atomically writes a `mlperf-results.yaml` sentinel. Every later command reads the sentinel; emitting subcommands invoked against an un-initialized results-dir refuse with a `ConfigurationError` directing the submitter to run `init` first. Re-initializing the same directory with the same orgname is idempotent; supplying a different orgname raises `DoubleInitError` rather than silently overwriting.
- **Systemname is per-run.** The `--systemname`/`-sn` flag (defaulting to `$MLPERF_SYSTEMNAME` when set) is required on every emitting subcommand. Because each run names its own system, the same results-dir can host runs from multiple system-under-test configurations without cross-contamination.
- **Mutually exclusive groups.** For example, VectorDB's `--runtime` and `--queries` are wired into an `add_mutually_exclusive_group()`, so only one can be supplied.
- **Pinned defaults in closed.** Closed kvcache pins `--gpu-mem-gb`, `--cpu-mem-gb`, `--duration`, `--trials`, `--seed`, `--rag-num-docs`, and several boolean knobs to their rules-mandated values, with no flag exposed to change them. Closed training/checkpointing/vectordb pin `--loops=1`, an empty `--params`, and `--allow-invalid-params=False` as internal defaults (the flags themselves are not registered on the closed parsers).
- **Post-parse validators.** What argparse cannot express (for example, "`--num-checkpoints-write` must be 10 or 0 in closed per Rules §4.7.1") is enforced by `validate_<benchmark>_arguments()` functions called immediately after parsing.
- **Environment validation.** Before a benchmark starts, `validate_benchmark_environment()` verifies SSH connectivity to client hosts, MPI availability, DLIO accessibility, and results-directory writability. `--skip-validation` disables this for debugging only.
- **Pre-execution capacity gates.** Before a benchmark spawns DLIO or any other workload, `_pre_execution_gate()` runs the CAP-01 disk-space check and (on multi-host runs) the CAP-02 shared-filesystem probe. Failures raise `FileSystemError` with a four-field message; there is no flag to bypass.

The result is that a closed-mode command line is exactly the command line a closed-mode submission requires, and an attempt to deviate is rejected at the earliest possible moment.

## COMMAND STRUCTURE

```
mlpstorage
├── init <orgname> <path>
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

## ORGNAME PINNING AND SYSTEMNAME RESOLUTION

### `mlpstorage init` and the `mlperf-results.yaml` sentinel

A results-dir becomes usable for emitting subcommands only after `mlpstorage init <orgname> <path>` succeeds. `init` is the *only* command that takes `<orgname>` on the command line and the *only* command that creates the results-dir if the parent directory exists. It atomically writes:

```yaml
# <path>/mlperf-results.yaml
mlperf_results_version: 1
orgname: <orgname>
initialized_at: <ISO-8601 UTC timestamp>
initialized_by: mlpstorage <version>
```

The orgname must match `[A-Za-z0-9._-]+` (Rules §2.1.5 submitter naming) and the comparison is case-sensitive — `Acme` and `acme` are two different organizations and writing one then re-running `init` with the other raises `DoubleInitError`. Re-running `init` with the same orgname is a no-op and returns success, so init is safe to script.

### Bypass set

The subcommands `init`, `version`, `lockfile`, and `rules-coverage` do not consult the sentinel — they have no need for an orgname. Every other top-level subcommand (`closed`, `open`, `whatif`, `validate`, `reports`, `history`) reads the sentinel via the orgname-resolution gate in `mlpstorage_py/main.py`. If the gate fires against an un-initialized results-dir it raises:

```
ConfigurationError: results-dir `<path>` has not been initialized.
Run `mlpstorage init <orgname> <path>` first.
```

### Systemname resolution

`--systemname <name>` / `-sn <name>` is required on every emitting subcommand (`run`, `datagen`, `configview`, `reportgen`, `history rerun`, etc.). Resolution priority is:

1. The CLI flag if supplied.
2. The `MLPERF_SYSTEMNAME` environment variable.
3. Otherwise empty string (which fails the required-on-emitting-commands check, surfacing as a parser error).

Because systemname is per-run, the same results-dir can host runs from many different systems-under-test. The canonical results path includes both `<orgname>` (from sentinel) and `<systemname>` (from CLI/env) so cross-system results never collide.

### Canonical results path

After init, every artifact-emitting command writes under:

```
<results-dir>/<mode>/<orgname>/results/<systemname>/<benchmark-specific tail>
```

The tail by benchmark (per `mlpstorage_py/rules/utils.py`):

| Benchmark      | Tail                                                                 |
|----------------|----------------------------------------------------------------------|
| training       | `training/<model>/<command>/<YYYYMMDD_HHMMSS>/`                      |
| checkpointing  | `checkpointing/<model>/<YYYYMMDD_HHMMSS>/` *(no `<command>` segment)*|
| vectordb       | `vector_database/<engine>/<index>/<command>/<YYYYMMDD_HHMMSS>/`      |
| kvcache        | `kv_cache/<model>/<command>/<YYYYMMDD_HHMMSS>/`                      |

Checkpointing intentionally omits the `<command>` segment to preserve the layout that pre-refactor submission tooling already accepts.

## SYSTEM DESCRIPTION (`systemname.yaml`)

`mlpstorage` auto-generates a partial system description at:

```
<results-dir>/<mode>/<orgname>/systems/<systemname>.yaml
```

one file per mode (closed, open, and whatif each own their own). The file is written *only* by the `run` command — never by `datagen`, `configview`, `datasize`, or any non-benchmark subcommand. (The client fleet that generates data is allowed to differ from the fleet that measures performance; per-mode separation prevents mode-specific environment-variable filtering and similar collector outputs from being mistaken for hardware drift.)

### Contents

The auto-collector emits a `system_under_test.clients[]` list keyed by quantity-grouped fingerprint:

```yaml
system_under_test:
  clients:
    - friendly_description: ""           # submitter to fill (SER-02 blank)
      quantity: <N>                       # auto-derived
      chassis:
        model_name: ""                    # DMI product name, blank when unparseable
        cpu_model: ""                     # /proc/cpuinfo model name
        cpu_qty: 0                        # socket count
        cpu_cores: 0
        memory_capacity: 0                # GiB, rounded
      networking: []                      # grouped by (type, speed, state)
      sysctl: []                          # allowlist snapshot
      environment: []                     # allowlist + redaction
      drives: []                          # grouped by (vendor, model, interface, capacity)
      operating_system:
        name: ""                          # os-release NAME
        version: ""                       # os-release VERSION_ID
```

Top-level blocks `solution`, `deployment`, `product_nodes`, `product_switches`, `total_rack_units`, and `rack_power_supplies` are intentionally omitted: the submission schema validator will fail on their absence, which is the intended UX prompting the submitter to fill them in.

Any single collection failure (missing file, parse error, missing tool) yields the empty string for that data point and never fails the benchmark — the universal collection-failure rule. Per-mode environment-variable allowlists and per-sysctl-name allowlists keep the fingerprint stable against ephemeral process-launch artifacts.

### Drift detection lifecycle

When `run` executes against a results-dir that already contains a `systemname.yaml`:

1. The file is parsed via `parse_on_disk_systemname_yaml` (`yaml.safe_load` only; no arbitrary object construction).
2. The current MPI fleet is collected and projected through the same emit pipeline.
3. `diff_node_dict_lists` compares both views.
4. If the diff is empty, the file is left untouched (LIFE-04 no-touch contract — the submitter's hand-fills are sacred).
5. If the diff has real entries, `SystemDriftError E404` is raised before DLIO launches; the error message renders a unified-diff with `--- on-disk` / `+++ in-memory` headers and a `Remediation:` block listing two options (rename + `--systemname <new>` or `rm <path>` and re-run).
6. If the on-disk YAML cannot be parsed, `SystemDescriptionParseError E104` is raised with the line/column the parser objected to.

### Hand-fill affordance (HANDFILL-01)

Seven fingerprint scalar positions are *soft-pair eligible*: `chassis.cpu_model`, `chassis.cpu_qty`, `chassis.cpu_cores`, `chassis.memory_capacity`, `chassis.model_name`, `operating_system.name`, `operating_system.version`. When the collector cannot resolve one of these (for example, `chassis.model_name` on a generic Linux host without parseable DMI strings) the field is written as `""`. A submitter is permitted to hand-edit that `""` to the correct value. On the next run, the diff layer's two-phase soft-pair pre-pass recognises the stanza as the same client (the four signature positions `networking_sig`, `sysctl_sig`, `environment_sig`, `drives_sig` must still match exactly per D-61) and the leaf-level Pitfall 3(a) SER-02 rule silently keeps the submitter's value. No drift is raised; the on-disk file stays unchanged.

If the collector *later* learns a value for a field the submitter did **not** hand-fill (e.g. a kernel upgrade exposes DMI), an INFO log is emitted: `collector resolved chassis.model_name='Dell Latitude 7420' (was "" on disk; on-disk file unchanged per LIFE-04 — manually update the YAML if you want to lock this value)`. No `DiffEntry`, no drift. This INFO log is scoped to the seven fingerprint scalar paths only; reverse-direction changes on non-fingerprint leaves continue to surface as drift.

Real hardware drift — a non-empty in-memory value that disagrees with a non-empty on-disk value at any leaf — still raises `SystemDriftError E404`. The hand-fill affordance is strictly empty-side adopt-on-empty; it never silences a non-empty disagreement.

### LIFE-04 no-touch contract

After the first successful write, `mlpstorage` will never again modify `systemname.yaml`. Subsequent runs either accept it (diff empty) or refuse to run (drift detected); they do not re-write it. Submitters can edit the YAML freely between runs as long as their edits do not contradict what the collector now sees.

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

The results directory accumulates every artifact produced by `mlpstorage` as each new invocation of `mlpstorage` executes. The default is `$MLPERF_RESULTS_DIR` if set, otherwise it must be supplied explicitly. Its layout follows the canonical Rules.md §2.1 shape from the moment `mlpstorage init` creates the sentinel:

```
<results-dir>/
├── mlperf-results.yaml                   sentinel written by `mlpstorage init` (LAY-02)
├── <mode>/                               closed | open | whatif (one or more)
│   └── <orgname>/                        from sentinel; same for every run
│       ├── systems/
│       │   └── <systemname>.yaml         auto-generated on first `run`; see SYSTEM DESCRIPTION
│       └── results/
│           └── <systemname>/             from --systemname / MLPERF_SYSTEMNAME
│               └── <benchmark-specific tail>
```

Every `run` adds a timestamped directory under its benchmark-specific tail; unwanted results can simply be removed from the tree (history records remain in `.history/`).

### Training results

```
<results-dir>/<mode>/<orgname>/results/<systemname>/training/<model>/
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
<results-dir>/<mode>/<orgname>/results/<systemname>/checkpointing/<model>/
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

Checkpointing intentionally omits the `<command>` segment under `<systemname>/checkpointing/<model>/` to preserve the layout that downstream submission tooling already accepts.

### VectorDB results

```
<results-dir>/<mode>/<orgname>/results/<systemname>/vector_database/<engine>/<index>/
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
<results-dir>/<mode>/<orgname>/results/<systemname>/kv_cache/<model>/run/<YYYYMMDD_HHMMSS>/
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

3. **Pre-execution capacity gates** (`mlpstorage_py/benchmarks/base.py::_pre_execution_gate`).
   After cluster collection and before the workload subprocess is spawned, every benchmark runs two checks. There is no flag to bypass either gate.

   - **CAP-01 — Disk-space gate.** Reads the destination filesystem via `os.statvfs(...)`, compares `available_bytes` against the benchmark's `required_bytes_for_capacity_gate` (computed per-subclass: training and checkpointing project the workload size from CLI arguments; vectordb returns `None` for the remote-engine escape hatch; kvcache projects from cache-tier sizes). On shortfall, raises `FileSystemError(code=FS_DISK_FULL)` with a four-field message:
     ```
     CAP-01: insufficient disk space at <destination_path>
       available_bytes: <int>
       required_bytes:  <int>
       deficit:         <int>
     ```
     Training's datagen path degrades gracefully (HARDEN-01): if `cluster_information` is unavailable (e.g. on a single-host dev machine without `mpi4py`/`psutil`), the gate logs a deferral notice and becomes a no-op. Checkpointing, vectordb, and kvcache use pure arg-derived math and never degrade.

   - **CAP-02 — Shared-filesystem probe.** On multi-host runs, launches `SHARED_FS_PROBE_SCRIPT` under `mpirun --tag-output` and stat's the run-uuid sentinel file from every rank. If the set of `(st_dev, st_ino)` pairs has cardinality > 1, the destination is not a shared filesystem and the probe raises `FileSystemError(code=FS_INVALID_STRUCTURE)` with a per-host/per-rank breakdown and the hint *"typically means one or more hosts have a local-disk path where a shared mount was expected."* The rank-0 result transports back via `__CAP02_RESULT_BEGIN__`/`__CAP02_RESULT_END__` stdout markers (HARDEN-02), and the `[rank,jobid]<channel>:` prefix emitted by OpenMPI 4.x is stripped via `_strip_tag_output_prefix` before JSON decode (HARDEN-04). Single-host runs skip the probe silently.

4. **Run-rule checkers** (`mlpstorage_py/rules/run_checkers/`).
   Per-benchmark `RunRulesChecker` classes inspect the merged DLIO configuration before execution. They enforce rules such as:
   - `check_num_files_train()` — generated dataset has enough files to satisfy the 5× memory rule
   - `check_allowed_params()` — every `--params` override is in the closed allow-list or open allow-list as appropriate
   - `check_workflow_parameters()` — UNet3D requires `workflow.checkpoint=True`
   - `check_odirect_supported_model()` — `reader.odirect` is only valid for UNet3D
   - `check_model()` (checkpointing) — model is one of the four supported LLM sizes

5. **Submission checkers** (`mlpstorage_py/submission_checker/`).
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

### Init options

```
mlpstorage init <orgname> <path>
```

- **`<orgname>`** (positional, required)
  Submitter / organization name to pin to the results-dir. Must match `[A-Za-z0-9._-]+` (Rules §2.1.5). Comparison is case-sensitive.

- **`<path>`** (positional, required)
  Filesystem path of the results-dir to initialize. Parent directory must exist; `<path>` is created if absent. Refuses to initialize a non-empty directory unless it already holds a `mlperf-results.yaml` sentinel whose orgname matches (idempotent re-init).

The `init` subcommand takes no flags — universal flags such as `--results-dir`, `--systemname`, `--debug`, etc. are not registered on the init parser, because the results-dir is the second positional and the sentinel does not yet exist.

### Universal options (every non-init command)

- **`--results-dir <path>`, `-rd <path>`**
  Root directory for all written artifacts. Required for any command that writes results. Defaults to `$MLPERF_RESULTS_DIR` if set. Must already be initialized with `mlpstorage init`; commands that consult the orgname-resolution gate refuse to run otherwise.

- **`--systemname <name>`, `-sn <name>`**
  System-under-test identifier for the current run. Required on every emitting subcommand (`run`, `datagen`, `configview`, `reportgen`, `history rerun`). Defaults to `$MLPERF_SYSTEMNAME`. Each mode (closed/open/whatif) owns its own `<systemname>.yaml` under the per-mode `systems/` directory, so the same name across modes is fine.

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

- **`MLPERF_RESULTS_DIR`** — default value for `--results-dir` when the flag is not supplied. The path must still have been initialized with `mlpstorage init`.
- **`MLPERF_SYSTEMNAME`** — default value for `--systemname` / `-sn` when the flag is not supplied. Emitting subcommands require systemname to be set via flag or env; an empty value is rejected at parse time.
- **`MLPERF_DATA_DIR`** — fallback value for `--data-dir` for some commands.
- **`MPI_RUN_BIN`** — overrides the path used when invoking `mpirun`.

There is intentionally **no `MLPERF_ORGNAME` environment variable** and no `--orgname` flag on benchmark subcommands. Orgname is sourced exclusively from the `mlperf-results.yaml` sentinel written by `mlpstorage init`.

## EXIT STATUS

- `0` — success.
- non-zero — argument validation failed, an environment check failed, a pre-execution capacity gate raised, a system-description error raised, a benchmark subprocess returned non-zero, or `validate` found a rule violation.

## ERROR CODES

A subset of the structured error codes a submitter may encounter at the CLI:

| Code  | Class                          | Raised by                                                                                                       |
|-------|--------------------------------|-----------------------------------------------------------------------------------------------------------------|
| E101  | `ConfigurationError`           | An emitting subcommand was run against a results-dir that has not been initialized with `mlpstorage init`.       |
| E104  | `SystemDescriptionParseError`  | On-disk `<systemname>.yaml` is malformed (yaml.YAMLError or missing `system_under_test.clients`).               |
| E201  | `ConfigurationError`           | Required configuration value missing or malformed (e.g. missing `--systemname` on an emitting subcommand).      |
| E404  | `SystemDriftError`             | The recomputed system description does not match the on-disk YAML; the error renders a unified-diff with remediation hints. |
| `FS_DISK_FULL`         | `FileSystemError` | CAP-01: destination filesystem free bytes < `required_bytes_for_capacity_gate`.                                  |
| `FS_INVALID_STRUCTURE` | `FileSystemError` | CAP-02: shared-FS probe found per-host `(st_dev, st_ino)` cardinality > 1, or per-rank stat failures.            |
| `DoubleInitError`      | `ConfigurationError` | `mlpstorage init` invoked against a results-dir already initialized under a different orgname.                |

## EXAMPLES

Initialize a fresh results-dir for organization "Acme":

```
mlpstorage init Acme /mnt/results
```

Size, generate, and run UNet3D in closed mode against a POSIX storage target:

```
export MLPERF_SYSTEMNAME=acme-prod-v1

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

The first `run` will auto-write `/mnt/results/closed/Acme/systems/acme-prod-v1.yaml`. Subsequent runs in the same mode/orgname/systemname diff against this file; rename + `--systemname <new>` (or remove the file) to start fresh.

Closed checkpointing for Llama 3 70B against object storage:

```
mlpstorage closed checkpointing llama3-70b run object \
    --num-processes 64 --client-host-memory-in-gb 1024 \
    --checkpoint-folder s3://bucket/checkpoints \
    --hosts host1,host2,host3,host4 \
    --systemname acme-prod-v1 \
    --results-dir /mnt/results
```

Open-mode VectorDB sweep against a remote Milvus:

```
mlpstorage open vectordb DISKANN run file \
    --host milvus.lab --port 19530 --collection bench_1m \
    --benchmark-mode sweep --runtime 600 \
    --num-query-processes 8 \
    --systemname acme-vdb-lab \
    --results-dir /mnt/results
```

Validate a prepared submission directory:

```
mlpstorage validate /submissions/acme \
    --csv /submissions/acme.summary.csv
```

## FILES

- `<repo>/configs/dlio/workload/*.yaml` — bundled DLIO workload templates for training and checkpointing.
- `<repo>/Rules.md` — authoritative submission rules.
- `<results-dir>/mlperf-results.yaml` — sentinel written by `mlpstorage init`; pins orgname to the results-dir.
- `<results-dir>/<mode>/<orgname>/systems/<systemname>.yaml` — auto-generated partial system description; one per mode; see SYSTEM DESCRIPTION.
- `<results-dir>/<mode>/<orgname>/results/<systemname>/...` — per-run output trees as documented under RESULTS DIRECTORY.
- `<results-dir>/.history/` — command history consumed by `mlpstorage history`.
- `<submission-dir>/<mode>/<submitter>/{code,systems,results}/` — submission package layout consumed by `mlpstorage validate`.

## SEE ALSO

- `Rules.md` — definitive rule reference.
- `Submission_guidelines.md` — packaging and submission process.
- `README.md` — installation and quickstart.
- `DEVELOPMENT.md` — contributor documentation.
- DLIO — Deep Learning I/O benchmark (upstream workload engine).
