# MLPerf Storage Benchmark Suite
MLPerf® Storage is a benchmark suite to characterize the performance of storage systems that support machine learning workloads.

- [Overview](#overview)
- [Submission Rules](#submission-rules)
  - [Validating a submission package](#validating-a-submission-package)
- [Normalizing Factors For Comparisons](#normalizing-factors-for-comparisons)
- [Usage](#usage)
  - [Prerequisite](#prerequisite)
  - [Installation](#installation)
  - [CLI Structure](#cli-structure)
  - [Storage Backend Selection (file | object)](#storage-backend-selection-file--object)
  - [Workload Categories](#workload-categories)
  - [Utility Commands](#utility-commands)
- [Advanced Usage](#advanced-usage)
  - [Object Storage Quick-start](#object-storage-quick-start)
  - [YAML Config-file Overrides (--config-file)](#yaml-config-file-overrides---config-file)
  - [Parameter Overrides (--params)](#parameter-overrides---params)
  - [Lockfile Validation on Benchmark Runs](#lockfile-validation-on-benchmark-runs)
  - [History and Replay Workflow](#history-and-replay-workflow)
  - [Report Generation (reports reportgen)](#report-generation-reports-reportgen)
  - [CI Integration and Exit Codes](#ci-integration-and-exit-codes)
- [Theory of Operations](#theory-of-operations)
  - [General Rules](#general-rules)
  - [CLOSED: virtually all changes are disallowed](#closed-virtually-all-changes-are-disallowed)
  - [OPEN: changes are allowed but must be disclosed](#open-changes-are-allowed-but-must-be-disclosed)
  - [System Description YAML - Structured Description](#system-description-yaml---structured-description)
  - [System Description PDF - Graphical and Prose Text](#system-description-pdf---graphical-and-prose-text)


## Overview

Two README files cover the full project in detail — read both before diving into the
code or running benchmarks:

| Document | What it covers |
|----------|----------------|
| **[docs/README.md](docs/README.md)** | Complete project overview: all four benchmark workloads, document reference, object storage library guides, and quick-link index to every test script |
| **[tests/README.md](tests/README.md)** | Everything needed to run tests: environment setup, unit tests, integration tests, object-store performance scripts, and how pytest is configured |

Additional quick links:

| Document | What it covers |
|----------|----------------|
| **[docs/OBJECT_STORAGE_GUIDE.md](docs/OBJECT_STORAGE_GUIDE.md)** | All settings required to run against S3-compatible storage with `--object` — `.env` setup, env vars, URI schemes, multi-endpoint |
| **[tests/object-store/bench-results-retinanet-20260425.md](tests/object-store/bench-results-retinanet-20260425.md)** | April 25, 2026 benchmark results: RetinaNet write_threads sweep on s3-ultra (loopback) |

The top-level sections below give the official MLCommons parameter reference and
are retained for submission compliance.

## Submission Rules

MLPerf™ Storage Benchmark submission rules are described in the
[Rules.md](https://github.com/mlcommons/storage/blob/main/Rules.md) file.
If you have questions, please contact the [Storage WG chairs](https://mlcommons.org/en/groups/research-storage/).

### Validating a submission package

Submitters and MLPerf reviewers can run the bundled submission checker to verify a
submission package against [`Rules.md`](https://github.com/mlcommons/storage/blob/main/Rules.md)
before submission:

```bash
mlpstorage validate <submission-dir>
```

#### Expected directory layout

`<submission-dir>` is the top-level submission directory — the parent of the
per-division (`closed/`, `open/`) trees. The validator walks the standard
MLPerf Storage submission hierarchy:

```
<submission-dir>/
├── closed/
│   └── <submitter>/
│       ├── code/                       # MD5-checked against REFERENCE_CHECKSUMS
│       ├── systems/<system>.yaml       # structured system description
│       ├── systems/<system>.pdf        # prose/graphical description
│       └── results/<system>/<mode>/<benchmark>/...
└── open/
    └── <submitter>/...
```

#### Behavior

The validator reports every rule violation it finds — each diagnostic carries
the `Rules.md` rule ID so the offending requirement can be looked up directly —
and continues checking the rest of the hierarchy rather than aborting on the
first failure. Exit codes:

- `0` — submission is clean.
- `1` — at least one rule violation was reported.

Per-submission output files (one log per `<division>/<submitter>` pair) are
written next to the CSV summary by default; pass `--skip-output-file` to keep
only the rolled-up `summary.csv`.

#### Options

| Option | Default | Purpose |
|---|---|---|
| `--submitters Acme,BetaCo` | all submitters under the input dir | Comma-separated allowlist of submitter directory names. Empty/whitespace tokens are stripped. |
| `--mlperf-version VERSION` | derived from package `major.minor` (currently `v3.0`) | Pin the spec version the submission claims to conform to. Supported: `v2.0`, `v3.0`. |
| `--csv PATH` | `summary.csv` in CWD | Path to write the summary CSV. |
| `--skip-output-file` | off | Suppress the per-submission text output file. |
| `--reference-checksum MD5` | bundled `REFERENCE_CHECKSUMS` table | Override the expected MD5 for the `code/` tree (use when validating against a code variant that legitimately differs from the reference). |

#### Examples

```bash
# Validate everything under ./submissions/
mlpstorage validate ./submissions

# Validate only Acme's closed and open packages, pin spec to v3.0,
# write CSV to a build dir, suppress per-submission text reports.
mlpstorage validate ./submissions \
  --submitters Acme \
  --mlperf-version v3.0 \
  --csv ./build/acme-summary.csv \
  --skip-output-file

# CI gate: fail the job on any rule violation.
mlpstorage validate ./submissions || exit 1
```

#### Self-validation: `rules-coverage`

A companion command audits the validator itself, reconciling every
`Rules.md` §2/§3/§4 rule ID against the live `@rule`-decorated check methods
to catch silent drift between the spec and the implementation:

```bash
mlpstorage rules-coverage
```

Use `--rules-md PATH` to point at an alternate `Rules.md` location. Exit code
`0` means every live ID has a check binding; `1` means at least one ID is
unmapped.

## Normalizing Factors For Comparisons

To compare the performance of two storage solutions that have very different architectures,
we must have a divisor that is independent of the storage system's architecture but is also present for all architectures.

### Rack Units Requirements (Mandatory)

If the system requires the physical deployment of dedicated hardware, ie: is not a cloud-based deployment or a hyperconverged deployment,
the SystemDescription.yaml will include the total number of rack units (RU's) that will be consumed by the storage system under test,
including any supporting gear that is required for the configuration being tested.
That supporting gear could include, for example, network switches for a "backend" or private network that is required for the storage system to operate.
The rack units measure does not need to include any of the gear that connects the storage system to the ``host nodes``.

This will show GB/s/RU or IOPs/RU.

### Power Requirements (Mandatory)

If the system requires the customer provisioning of power (for example, systems intended to be deployed in on-premises data centers or in co-located data centers)
the SystemDescription.yaml will include all hardware devices required to operate the storage system.
Shared network equipment also used for client network communication and optional storage management systems do not need to be included.

This will show GB/s/KW or IOPs/KW.

## Usage
For an overview of how this benchmark suite is used by submitters to compare the performance of storage systems supporting an AI cluster, see the MLPerf® Storage Benchmark submission rules here: [doc](https://github.com/mlcommons/storage/blob/main/Submission_guidelines.md). 

### Prerequisite

The installation and the configuration steps described in this README are validated against clients running Ubuntu 24.04 server with python 3.12.3. The benchmark script has to be run only in one participating client host(any) which internally calls `mpirun` to launch the distributed workloads across multiple client hosts. The launcher client host also participates in the distributed training process.

Following prerequisites must be satisfied

1. Pick one host to act as the launcher client host. Passwordless ssh must be setup from the launcher client host to all other participating client hosts.  `ssh-copy-id` is a useful tool.
2. The code and data location(discussed in further sections) must be exactly same in every client host including the launcher host. This is because, the same benchmark command is automatically triggered in every participating client host during the distributed training process.

#### Running as root
When the launcher client is root (common in container or bare-metal benchmark setups), `mpirun` will refuse to launch unless `--allow-run-as-root` is passed to every `mlpstorage` sub-command that triggers MPI (any `datasize`, `datagen`, or `run` leaf under `training`, `checkpointing`, or `kvcache`). Add it explicitly:

```bash
mlpstorage closed training unet3d datagen file \
  --hosts 127.0.0.1 --num-processes 8 --data-dir unet3d_data \
  --allow-run-as-root --param dataset.num_files_train=42000
```

### Installation 
**The following installation steps must be run on every client host that will participate in running the benchmarks.**

#### uv (Required)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project manager that handles virtual environment creation, dependency resolution, and Python version management automatically — no manual `venv` or `pip` steps required. It will install into your virutal environment exactly the versions of supporting libraries and tools that the benchmark has been tested with.

**Install uv** (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the MPI runtime (still required for distributed execution):

```bash
sudo apt install libopenmpi-dev openmpi-common
```

Clone the repo:

```bash
git clone https://github.com/mlcommons/storage.git
cd storage
```

Verify the installation:

```bash
mlpstorage --help
```

The `mlpstorage` script executes `uv run` every time you invoke the benchmark, keeping your virtual environment up to date.
`uv` creates a `.venv` virtual environment and installs all dependencies — including DLIO benchmark — automatically based upon the contents of the the `uv.lock` file.

> **Note:** `uv` installs the CPU-only version of PyTorch.
> GPU-accelerated training or checkpointing workloads are not supported, there is no need to have GPUs in your benchmark test gear, they will not be used.

The benchmark simulation will be performed through the [dlio_benchmark](https://github.com/mlcommons/DLIO_local_changes) code, a benchmark suite for emulating I/O patterns for deep learning workloads. The DLIO configuration of each workload is specified through a yaml file. You can see the configs of all MLPerf Storage workloads in the `configs` folder. 

#### Testing the Installation

See **[tests/README.md](tests/README.md)** for the complete test guide — environment
setup, unit tests (no infrastructure required), integration tests, and object-store
performance scripts for all three supported object storage libraries.

- **StreamingCheckpointing Demo**: Run `./tests/checkpointing/demo_checkpoint_methods.sh` to see:
  - dgen-py integration (155× faster data generation)
  - StreamingCheckpointing (192× memory reduction)
  - Comparison of old vs new checkpoint methods

- **Backend Validation**: Test multi-library support:
  ```bash
  python tests/checkpointing/test_streaming_backends.py --backends s3dlio minio
  ```

- **Unit tests** (no infrastructure required):
  ```bash
  pytest tests/unit/
  ```

### CLI Structure

The `mlpstorage` CLI uses a positional sub-command tree. Every benchmark
invocation begins with a **submission mode**, which fixes the rules surface
the rest of the command tree is checked against:

| Mode | What it means |
|---|---|
| `closed` | Submission mode where almost no tuning is allowed. Use this when generating results you intend to submit in the CLOSED division. |
| `open` | Submission mode that permits documented tuning and customization. Use this for OPEN-division submissions. |
| `whatif` | Non-submission exploration mode. Exposes models and parameters that are not eligible for either division — useful for sizing experiments, regression testing, and "what if I changed X" investigations. Results are never valid for submission. |

After the mode comes the benchmark family, then the workload (model or
algorithm), then the command (`datasize`, `datagen`, `run`, `configview`),
then — for commands that touch storage — the storage selector
(`file` or `object`):

```
mlpstorage <closed|open|whatif> <training|checkpointing|vectordb|kvcache>
           <model|algorithm> <datasize|datagen|run|configview> [file|object]
           [OPTIONS]
```

Top-level utility commands live as siblings of the modes:

```
mlpstorage (reports|history|lockfile|version) [subcommand] [OPTIONS]
mlpstorage validate <submission-dir> [OPTIONS]
mlpstorage rules-coverage [--rules-md PATH]
```

The full command tree, including every leaf's option set, is printed by:

```bash
mlpstorage --help_all
```

Context-sensitive help is available at every level — typing
`mlpstorage closed training` (or any other incomplete path) prints the next
valid tokens and exits. Append `--help` at any level for a SYNOPSIS plus the
next-level menu.

Top-level overview:

```bash
$ mlpstorage --help
usage: mlpstorage [-h] [--version]
                  {closed,open,whatif,reports,history,lockfile,version,
                   validate,rules-coverage} ...

Script to launch the MLPerf Storage benchmark

positional arguments:
  closed          Closed submission mode (comparable, no tuning)
  open            Open submission mode (tuning allowed, must be disclosed)
  whatif          Exploration mode — not submittable
  reports         Generate a report from benchmark results
  history         Display / replay benchmark history
  lockfile        Generate and verify package lockfiles
  version         Show installed package version and exit
  validate        Validate a submission package against Rules.md
  rules-coverage  Audit which Rules.md IDs are covered by check methods

optional arguments:
  -h, --help      show this help message and exit
  --help_all      print the full command tree and exit
  --version       show program's version number and exit
```

#### End-to-end example

A typical CLOSED-division UNet3D run on a POSIX backend with two H100s:

```bash
# 1. Size the dataset for the target cluster
mlpstorage closed training unet3d datasize \
  --num-accelerators 2 --accelerator-type h100 \
  --client-host-memory-in-gb 64

# 2. Generate the synthetic dataset
mlpstorage closed training unet3d datagen file \
  --num-processes 4 \
  --data-dir /databases/mlps-v3.0/data/ \
  --results-dir /databases/mlps-v3.0/results

# 3. Inspect the final merged configuration (no run)
mlpstorage closed training unet3d configview file \
  --num-accelerators 2 --accelerator-type h100 \
  --client-host-memory-in-gb 64 \
  --data-dir /databases/mlps-v3.0/data/

# 4. Run the benchmark
mlpstorage closed training unet3d run file \
  --num-accelerators 2 --accelerator-type h100 \
  --client-host-memory-in-gb 64 \
  --data-dir /databases/mlps-v3.0/data/ \
  --results-dir /databases/mlps-v3.0/results
```

The same sequence on an S3-compatible object backend swaps `file` → `object`
on every storage-touching command and supplies the endpoint/bucket env vars
described in [docs/OBJECT_STORAGE_GUIDE.md](docs/OBJECT_STORAGE_GUIDE.md).

### Storage Backend Selection (file | object)

The `training`, `checkpointing`, and `vectordb` workloads require you to
declare the storage backend under test as a **positional** immediately after
the leaf command. Exactly one of the following is required:

| Token | Backend | When to use |
|---|---|---|
| `file` | POSIX/parallel filesystem (local, NFS, Lustre, GPFS, WekaFS, etc.) accessed via the data / checkpoint-folder path. | Block storage, file storage, parallel filesystem submissions. |
| `object` | S3-compatible object store, accessed via one of the three supported object-store libraries (see [docs/OBJECT_STORAGE_GUIDE.md](docs/OBJECT_STORAGE_GUIDE.md) and `tests/README.md`). | Object-storage submissions. Requires endpoint/bucket env vars. |

The selector is absent on commands that don't touch storage:
- `training/<model>/datasize` — sizing only.
- All `kvcache` leaves — `kvcache` does not take a storage selector.

### Workload Categories

#### Training
Emulates training I/O for `unet3d` and `retinanet` in `closed`/`open`, and
adds `cosmoflow`, `resnet50`, `dlrm`, and `flux` in `whatif`. Commands:
`datasize`, `datagen`, `run`, `configview`.

See [training/README.md](training/README.md) for more details.

#### Checkpointing
Emulates checkpoint write/read for the Llama3 LLM at four scales: `llama3-8b`,
`llama3-70b`, `llama3-405b`, `llama3-1t`. Commands: `datasize`, `run`,
`configview`.

See [checkpointing/README.md](checkpointing/README.md) for more details.

#### VectorDB
Emulates a vector database used in an LLM RAG pipeline (Milvus). Algorithms:
`DISKANN`, `HNSW`, `AISAQ` in `closed`, plus `IVF_FLAT`, `IVF_SQ8`, `FLAT` in
`open`/`whatif`. Commands: `datasize`, `datagen`, `run`.

See [vdb_benchmark/README.md](vdb_benchmark/README.md) for more details.

#### KVCache
Emulates a context cache used by an LLM. In `closed`, kvcache takes no model
positional. In `open`/`whatif`, choose from `tiny-1b`, `mistral-7b`,
`llama2-7b`, `llama3.1-8b`, `llama3.1-70b-instruct`. Commands: `datasize`,
`run`.

See [kv_cache_benchmark/README.md](kv_cache_benchmark/README.md) for more details.

### Utility Commands

These siblings of the benchmark modes are not gated by `closed`/`open`/`whatif`:

| Command | Purpose |
|---|---|
| `mlpstorage reports reportgen ...` | Roll up results from a results directory into the submission-format report. |
| `mlpstorage history show [--limit N \| --id ID]` | Show previously executed `mlpstorage` invocations from the history file. |
| `mlpstorage history rerun <id>` | Re-run a recorded invocation by history ID — handy for repeating an exact run after iterating on the storage system. |
| `mlpstorage lockfile generate` | Produce a reproducible Python dependency lockfile from `pyproject.toml`. Used by submitters who must publish the exact dependency set they tested with. |
| `mlpstorage lockfile verify` | Verify the currently installed environment matches a lockfile. Benchmark `run` commands can also gate on this with `--verify-lockfile PATH`. |
| `mlpstorage version` | Print the installed `mlpstorage` package version. |
| `mlpstorage validate <submission-dir>` | Run the Rules.md submission checker — see [Validating a submission package](#validating-a-submission-package). |
| `mlpstorage rules-coverage` | Self-audit: every Rules.md rule ID is reconciled against the live `@rule`-decorated check methods. |

## Advanced Usage

### Object Storage Quick-start

Object-storage submissions use the `object` positional in place of `file` on
any storage-touching leaf. The endpoint, region, bucket, and credentials are
read from environment variables — see
[docs/OBJECT_STORAGE_GUIDE.md](docs/OBJECT_STORAGE_GUIDE.md) for the full
matrix across the three supported object-store libraries (`s3dlio`,
`s3torchconnector`, `minio`).

Minimum env vars for an S3-compatible endpoint (set in a `.env` file or your
shell):

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_ENDPOINT_URL=https://my-object-endpoint:9000
export AWS_REGION=us-east-1
export MLPS_S3_BUCKET=mlperf-storage-bench
```

With those exported, the UNet3D end-to-end example becomes:

```bash
mlpstorage closed training unet3d datagen object \
  --num-processes 4 \
  --data-dir s3://mlperf-storage-bench/unet3d/data/ \
  --results-dir /databases/mlps-v3.0/results

mlpstorage closed training unet3d run object \
  --num-accelerators 2 --accelerator-type h100 \
  --client-host-memory-in-gb 64 \
  --data-dir s3://mlperf-storage-bench/unet3d/data/ \
  --results-dir /databases/mlps-v3.0/results
```

Per-backend library selection, multi-endpoint setups, and TLS-cert overrides
are documented in [docs/OBJECT_STORAGE_GUIDE.md](docs/OBJECT_STORAGE_GUIDE.md).

### YAML Config-file Overrides (`--config-file`)

Every benchmark leaf accepts `--config-file PATH` (short form: `-c PATH`).
The YAML file is loaded **after** the CLI arguments are parsed and overrides
matching keys in the parsed namespace. Precedence is:

```
CLI args  >  YAML config file  >  argparse defaults  >  environment variables
```

Unknown keys produce a warning and are ignored; explicit `null` values are
skipped so a YAML file cannot silently null out a CLI-supplied flag.

Example `unet3d-h100.yaml`:

```yaml
# Cluster shape
num_accelerators: 8
accelerator_type: h100
client_host_memory_in_gb: 128
hosts:
  - 10.0.0.11
  - 10.0.0.12
  - 10.0.0.13
  - 10.0.0.14

# Common paths
data_dir: /mnt/mlps/unet3d/data
results_dir: /mnt/mlps/results

# DLIO parameter overrides (see "Parameter Overrides" below)
params:
  dataset.num_files_train: 42000
  reader.read_threads: 8
```

Invocation:

```bash
mlpstorage closed training unet3d run file -c unet3d-h100.yaml
```

The YAML form for `params` accepts either a dict (shown above, recommended)
or a list of `key=value` strings.

### Parameter Overrides (`--params`)

`--params` (short form `-p`) passes DLIO-level parameter overrides directly
into the workload configuration. Keys use **dotted notation** matching the
DLIO YAML schema; values are flattened into the final DLIO config at run
time.

Syntax — multiple `key=value` pairs after a single `--params`:

```bash
mlpstorage closed training unet3d run file \
  --num-accelerators 2 --accelerator-type h100 \
  --client-host-memory-in-gb 64 \
  --data-dir /databases/mlps-v3.0/data/ \
  --params dataset.num_files_train=42000 reader.read_threads=8
```

You can also repeat the flag — both forms are merged:

```bash
... --params dataset.num_files_train=42000 \
    --params reader.read_threads=8
```

In CLOSED submissions only a vetted subset of `params` is allowed; the CLI
rejects others unless you pass `--allow-invalid-params` (which **invalidates
the run for CLOSED submission**). OPEN and `whatif` permit the full DLIO
parameter surface, but every override used must be disclosed in the OPEN
submission's `system.yaml` / PDF.

### Lockfile Validation on Benchmark Runs

Submitters can pin their dependency set with `lockfile generate`, then gate
benchmark execution on a clean environment with `--verify-lockfile`:

```bash
# One-time: produce the lockfile that travels with the submission
mlpstorage lockfile generate --output requirements.txt

# (Optional) include extras
mlpstorage lockfile generate --extra full --output requirements-full.txt

# On every host before running, generate base + full in one shot
mlpstorage lockfile generate --all

# Gate a benchmark run on the installed environment matching the lockfile
mlpstorage closed training unet3d run file \
  --verify-lockfile ./requirements.txt \
  --num-accelerators 2 --accelerator-type h100 \
  --client-host-memory-in-gb 64 \
  --data-dir /databases/mlps-v3.0/data/ \
  --results-dir /databases/mlps-v3.0/results
```

If the verification fails, the run is aborted before any DLIO process is
spawned and the CLI prints the exact `pip install -r` / `uv pip sync` command
needed to bring the environment back into compliance. To run anyway, drop
the flag — the validation is opt-in per invocation.

### History and Replay Workflow

Every `mlpstorage` invocation (except `history` itself) is recorded to the
history file (`$HISTFILE`). This is intended for two workflows:

1. **Audit trail** — list what was run during a submission cycle:

   ```bash
   mlpstorage history show              # all entries
   mlpstorage history show --limit 10   # most recent 10
   mlpstorage history show --id 42      # one specific entry
   ```

2. **Exact replay** — reproduce a prior run after iterating on the storage
   system:

   ```bash
   # 1. Find the ID of the run you want to repeat
   mlpstorage history show --limit 20

   # 2. Re-run it verbatim with all its original arguments
   mlpstorage history rerun 42
   ```

   `rerun` reconstructs the original argv and re-dispatches it. Logging-level
   flags (`--debug`, `--verbose`, `--stream-log-level`) honored at the
   command line are re-applied before dispatch so an interactive `rerun` can
   override the original log level.

### Report Generation (`reports reportgen`)

After one or more benchmark `run` commands have populated a results directory,
roll them up into the submission-ready report layout with:

```bash
mlpstorage reports reportgen \
  --results-dir /databases/mlps-v3.0/results \
  --output-dir ./submission-build/results
```

What `reportgen` consumes:

- A results directory that contains one or more `<mode>/<benchmark>/<model>/<datetime>/`
  trees (the layout `mlpstorage <mode> <benchmark> <model> run` writes to).
- Each leaf must contain the per-run `summary.json`, the workload's
  `metadata.json`, and the DLIO output subtree.

What `reportgen` emits into `--output-dir` (or `--results-dir` if
`--output-dir` is omitted):

- `results.json` — machine-readable rollup of every discovered run.
- `results.csv` — flat CSV for spreadsheet / Pandas import.
- The directory tree expected by `mlpstorage validate` —
  `<division>/<submitter>/results/<system>/<mode>/<benchmark>/...` — so the
  output can be dropped directly into a submission package.

The recommended end-to-end submission workflow:

```bash
# 1. Run all required benchmark configurations
mlpstorage closed training unet3d run file ...
mlpstorage closed checkpointing llama3-8b run file ...
# ...etc.

# 2. Roll up results into the submission layout
mlpstorage reports reportgen \
  --results-dir /databases/mlps-v3.0/results \
  --output-dir ./submission-build

# 3. Hand-author the systems/<system>.yaml and systems/<system>.pdf files
#    (see "System Description YAML" and "System Description PDF" below).

# 4. Validate the assembled package against Rules.md before submission
mlpstorage validate ./submission-build
```

### CI Integration and Exit Codes

`mlpstorage` uses a small, stable set of integer exit codes so its
sub-commands can be gated from CI pipelines. The full set is defined in
`mlpstorage_py/config.py::EXIT_CODE`:

| Code | Name | Meaning |
|---|---|---|
| `0` | `SUCCESS` | Command completed cleanly. For `validate` / `rules-coverage` this also means "no violations". |
| `1` | `GENERAL_ERROR` | Catch-all failure not classified below. |
| `2` | `INVALID_ARGUMENTS` | argparse rejection or post-parse validation failure. |
| `3` | `FILE_NOT_FOUND` | A required file or directory was missing. |
| `4` | `PERMISSION_DENIED` | Filesystem / process permission denied. |
| `5` | `CONFIGURATION_ERROR` | Recognized arguments but an invalid combination or unsupported value (e.g. unknown benchmark). |
| `6` | `FAILURE` | Benchmark, lockfile verify, or submission validation reported violations or failed to run to completion. |
| `7` | `TIMEOUT` | Operation exceeded its time budget. |
| `8` | `INTERRUPTED` | Caught `SIGINT` / `SIGTERM` — process exited cleanly mid-run. |

A minimal pre-submission CI gate:

```bash
set -euo pipefail

# Step 1 — environment must match the published lockfile
mlpstorage lockfile verify --lockfile ./requirements.txt

# Step 2 — the validator implementation must cover every live Rules.md ID
mlpstorage rules-coverage

# Step 3 — the assembled submission must be Rules.md-clean
mlpstorage validate ./submission-build \
  --csv ./submission-build/summary.csv

# Step 4 — sanity-rerun the most recent benchmark from history,
#          fail the job if it no longer reproduces.
mlpstorage history rerun "$(mlpstorage history show --limit 1 | awk '{print $1}')"
```

With `set -e`, the pipeline aborts on the first non-zero exit. To keep
running and aggregate failures (useful for nightly health-checks), capture
each step's exit code explicitly:

```bash
mlpstorage validate ./submission-build || validate_rc=$?
mlpstorage rules-coverage              || coverage_rc=$?
exit $(( ${validate_rc:-0} | ${coverage_rc:-0} ))
```

## Theory of Operations

MLPerf™ Storage is a benchmark suite to characterize the performance of storage systems that support machine learning workloads.

This benchmark attempts to balance two goals. First, we aim for **comparability** between benchmark submissions to enable decision making by the AI/ML Community. Second, we aim for **flexibility** to enable experimentation and to show off unique storage system features that will benefit the AI/ML Community. To that end we have defined two classes of submissions: CLOSED and OPEN. 

The MLPerf name and logo are trademarks of the MLCommons® Association ("MLCommons"). In order to refer to a result using the MLPerf name, the result must conform to the letter and spirit of the rules specified in this document. MLCommons reserves the right to solely determine if a use of its name or logos is acceptable.

This version of the benchmark does not include offline or online data pre-processing. We are aware that data pre-processing is an important part of the ML data pipeline and we will include it in a future version of the benchmark.

### General Rules
 
The following apply to all results submitted for this benchmark.

Benchmarking should be conducted to measure the framework and storage system performance as fairly as possible. Ethics and reputation matter.

- **Available Systems**. To be called an ``available system`` all components of the system must be publicly available. If any components of the system are not available at the time of the benchmark results submission, those components must be included in an ``available system`` submission that is submitted in the next round of MLPerf Storage benchmark submissions.  Otherwise, the results for that submission may be retracted from the MLCommons results dashboard.
- **RDI Systems**. If you are measuring the performance of an experimental framework or system, you must make the system and framework you use available upon demand for replication by MLCommons. This class of systems will be called RDI (research, development, internal). 

The data generator in DLIO uses a fixed random seed that must not be changed, to ensure that all submissions are working with the same dataset. Random number generators may be seeded from the following sources:
- Clock
- System source of randomness, e.g. /dev/random or /dev/urandom
- Another random number generator initialized with an allowed seed
Random number generators may be initialized repeatedly in multiple processes or threads. For a single run, the same seed may be shared across multiple processes or threads.

The storage system must not be informed of the random seed or the source of randomness.  This is intended to disallow submissions where the storage systen can predict the access pattern of the data samples.

Public results should be rounded normally, to two decimal places.

For all workloads stable storage must be used, but there are some differences in the specifics.

Results that cannot be replicated are not valid results. Replicated results should be within 5% within 5 tries.

Each of the benchmarks described in this document have a requirement for multiple runs. This is to ensure consistency of operation of the system under test as well as ensure statistical significance of the measurements.

Unless otherwise noted, the multiple runs for a workload need to be run consecutively. To ensure this requirement is met, the time between runs (from the stop time of one run and the start time to the next run) needs to be less than the time to execute a single run. This is to discourage cherry-picking of results which is expressly forbidden and against the spirit of the rules.

### CLOSED: virtually all changes are disallowed
CLOSED represents a level playing field where all results are **comparable** across submissions. CLOSED explicitly forfeits flexibility in order to enable easy comparability. 

In order to accomplish that, most of the optimizations and customizations to the AI/ML algorithms and framework that might typically be applied during benchmarking or even during production use must be disallowed.  Optimizations and customizations to the storage system are allowed in CLOSED.

For CLOSED submissions of this benchmark, the MLPerf Storage codebase takes the place of the AI/ML algorithms and framework, and therefore cannot be changed. The sole exception to this rule is if the submitter decides to apply the code change identified in PR#299 of the DLIO repo in github, the resulting codebase will be considered "unchanged" for the purposes of this rule. 

### OPEN: changes are allowed but must be disclosed

OPEN allows more **flexibility** to tune and change both the benchmark and the storage system configuration to show off new approaches or new features that will benefit the AI/ML Community. OPEN explicitly forfeits comparability to allow showcasing innovation.

The essence of OPEN division results is that for a given benchmark area, they are “best case” results if optimizations and customizations are allowed.  The submitter has the opportunity to show the performance of the storage system if an arbitrary, but documented, set of changes are made to the data storage environment or algorithms.

Changes to DLIO itself are allowed in OPEN division submissions.  Any changes to DLIO code or command line options must be disclosed. 

While changes to DLIO are allowed, changing the workload itself is not.  Ie: how the workload is processed can be changed, but those changes cannot fundamentally change the purpose and result of the training.  For example, changing the workload imposed upon storage by a ResNet-50 training task into 3D-Unet training task is not allowed.

### System Description YAML - Structured Description

The purpose of the system description is to provide sufficient detail on the storage system under test, and the ``host nodes`` running the test, plus the network connecting them, to enable full reproduction of the benchmark results by a third party. 

Each submission must contain a ``<system-name>.yaml`` file and a ``<system-name>.pdf`` file.  If you submit more than one benchmark result, each submission must have a unique ``<system-name>.yaml`` file and a ``<system-name>.pdf`` file that documents the system under test and the environment that generated that result, including any configuration options in effect.

The system description yaml is a hybrid human-readable and machine-readable description of the total system under test. It contains fields for the System overall, the Nodes that make up the solution (clients and storage), as well as Power information of the nodes.

An example can be found [HERE](https://github.com/mlcommons/storage/blob/main/system_configuration.yaml)

### System Description PDF - Graphical and Prose Text

The goal of the pdf is to complement the YAML file, providing additional detail on the system to enable full reproduction by a third party. We encourage submitters to add details that are more easily captured by diagrams and text description, rather than a YAML.

This file is should include everything that a third party would need in order to recreate the results in the submission, including product model numbers or hardware config details, unit counts of drives and/or components, system and network topologies, software used with version numbers, and any non-default configuration options used by any of the above.
