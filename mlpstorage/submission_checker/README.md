# MLPerf Storage Submission Checker

Validates MLPerf Storage V2.0 benchmark submission packages against the rules defined in [Rules.md](Rules.md).

## Overview

The checker traverses a submission directory hierarchy, parses log files and metadata, runs a battery of structural and parameter checks, and exports a CSV summary of results.

### Submission Directory Structure

```
<root>/
├── closed/
│   └── <submitter>/
│       ├── code/
│       ├── systems/
│       │   ├── <system-name>.yaml
│       │   └── <system-name>.pdf
│       └── results/
│           └── <system-name>/
│               ├── training/
│               │   ├── unet3d/
│               │   ├── resnet50/
│               │   └── cosmoflow/
│               └── checkpointing/
│                   ├── llama3-8b/
│                   ├── llama3-70b/
│                   ├── llama3-405b/
│                   └── llama3-1t/
└── open/
    └── <submitter>/   (same structure as closed/)
```

Each training workload contains `datagen/` and `run/` phase directories. Each contains timestamp directories named `YYYYMMDD_HHmmss` with log files and a `dlio_config/` subdirectory.

## Usage

```bash
python -m mlpstorage.submission_checker.main \
    --input /path/to/submission \
    --version v3.0 \
    --csv results.csv
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input` | Yes | — | Root of the submission directory |
| `--version` | No | `v5.1` | MLPerf version (`v2.0` or `v3.0`) |
| `--submitters` | No | all | Comma-separated list of submitters to validate |
| `--csv` | No | `summary.csv` | Output CSV file path |
| `--skip-output-file` | No | `False` | Skip checking for `*output.json` files |

## Architecture

```
submission_checker/
├── main.py                        # Entry point and check orchestration
├── loader.py                      # Traverses filesystem, yields SubmissionLogs
├── constants.py                   # Version-specific file patterns and configs
├── results.py                     # Exports validation results to CSV
├── utils.py                       # Filesystem helpers and regex utilities
├── configuration/
│   └── configuration.py           # Version-aware configuration class
├── checks/
│   ├── base.py                    # BaseCheck abstract class
│   ├── directory_checks.py        # File/directory structure validation
│   ├── training_checks.py         # Training workload parameter validation
│   ├── checkpointing_checks.py    # Checkpointing workload validation
│   ├── vdb_checks.py              # Vector DB validation (not yet implemented)
│   └── kv_cache_checks.py         # KV Cache validation (not yet implemented)
└── parsers/
    ├── json_parser.py             # JSON log file parser
    └── yaml_parser.py             # YAML config file parser
```

## Rules Coverage

The following tables map each rule from [Rules.md](Rules.md) to its implementation status.

### Section 2 — Directory Structure

| Rule ID | Name | Status | Implementation |
|---|---|---|---|
| 2.1 | `submitterRootDirectory` | ❌ Not checked | — |
| 2.2 | `topLevelSubdirectories` | ❌ Not checked | — |
| 2.3 | `openMatchesClosed` | ❌ Not checked | — |
| 2.4 | `closedSubmitterDirectory` | ❌ Not checked | — |
| 2.5 | `requiredSubdirectories` | ❌ Not checked | — |
| 2.6 | `codeDirectoryContents` | ❌ Not checked | — |
| 2.7 | `systemsDirectoryFiles` | ❌ Not checked | — |
| 2.8 | `resultsDirectorySystems` | ❌ Not checked | — |
| 2.9 | `identicalSystemConfig` | ❌ Not checked | — |
| 2.10 | `workloadCategories` | ❌ Not checked | — |
| 2.11 | `trainingWorkloads` | ❌ Not checked | — |
| 2.12 | `trainingPhases` | ❌ Not checked | — |
| 2.13 | `datagenTimestamp` | ❌ Not checked | — |
| 2.14 | `datagenFiles` | ✅ Checked | `DirectoryCheck.datagen_files_check` |
| 2.15 | `datagenDlioConfig` | ✅ Checked | `DirectoryCheck.datagen_dlio_config_check` |
| 2.16 | `runResultsJson` | ✅ Checked | `DirectoryCheck.run_results_json_check` |
| 2.17 | `runTimestamps` | ✅ Checked | `DirectoryCheck.run_files_timestamp_check` |
| 2.18 | `runTimestampGap` | ✅ Checked | `DirectoryCheck.run_duration_valid_check` |
| 2.19 | `runFiles` | ✅ Checked | `DirectoryCheck.run_files_check` |
| 2.20 | `runDlioConfig` | ✅ Checked | `DirectoryCheck.run_dlio_config_check` |
| 2.21 | `checkpointingWorkloads` | ❌ Not checked | — |
| 2.22 | `checkpointingResultsJson` | ✅ Checked | `DirectoryCheck.checkpointing_results_json_check` |
| 2.23 | `checkpointingTimestamps` | ✅ Checked | `DirectoryCheck.checkpointing_timestamps_check` |
| 2.24 | `checkpointingTimestampGap` | ✅ Checked | `DirectoryCheck.checkpointing_timestamp_gap_check` |
| 2.25 | `checkpointingFiles` | ✅ Checked | `DirectoryCheck.checkpointing_files_check` |
| 2.26 | `checkpointingDlioConfig` | ✅ Checked | `DirectoryCheck.checkpointing_dlio_config_check` |

### Section 3 — Training Workload Validation

| Rule ID | Name | Status | Notes |
|---|---|---|---|
| 3.1.1 | `verifyDatasizeUsage` | ⚠️ Partial | Checks dataset params are present, not explicit datasize CLI usage |
| 3.1.2 | `recalculateDatasetSize` | ✅ Checked | `TrainingCheck.recalculate_dataset_size` |
| 3.2.1 | `datagenMinimumSize` | ✅ Checked | `TrainingCheck.datagen_minimum_size` |
| 3.3.1 | `runDataMatchesDatasize` | ⚠️ Partial | Checks bounds against dataset constants, not exact recalculated value |
| 3.3.2 | `acceleratorUtilizationCheck` | ✅ Checked | `TrainingCheck.accelerator_utilization_check` |
| 3.3.3 | `singleHostSimulatedAccelerators` | ⚠️ Warning only | `TrainingCheck.single_host_simulated_accelerators` — does not fail the run |
| 3.3.4 | `singleHostClientLimit` | ❌ Not active | Method exists but is not registered in `init_checks` |
| 3.3.5 | `distributedDataAccessibility` | ❌ Not checked | Spec notes this may be removed |
| 3.3.6 | `identicalAcceleratorsPerNode` | ✅ Checked | `TrainingCheck.identical_accelerators_per_node` |
| 3.3.7 | `nodeCapabilityConsistency` | ❌ Not checked | Spec notes this may be removed |
| 3.3.8 | `closedSubmissionChecksum` | 🔲 TODO | Skeleton exists, MD5 not implemented |
| 3.3.9 | `closedSubmissionParameters` | ✅ Checked | `TrainingCheck.closed_submission_parameters` |
| 3.3.10 | `openSubmissionParameters` | ✅ Checked | `TrainingCheck.open_submission_parameters` |
| 3.3.11 | `mlpstoragePathArgs` | ✅ Checked | `TrainingCheck.mlpstorage_path_args` |
| 3.3.12 | `mlpstorageFilesystemCheck` | 🔲 TODO | Skeleton exists, `df`-based check not implemented |

### Section 4 — Checkpointing Workload Validation

| Rule ID | Name | Status | Notes |
|---|---|---|---|
| 4.1.1 | `checkpointDataSizeRatio` | ⚠️ Warning only | `CheckpointingCheck.checkpoint_data_size_ratio` — does not fail the run |
| 4.1.2 | `fsyncVerification` | ✅ Checked | `CheckpointingCheck.fsync_verification` |
| 4.1.3 | `modelConfigurationReq` | ✅ Checked | `CheckpointingCheck.model_configuration_req` |
| 4.1.4 | `closedMpiProcesses` | ⚠️ Bug | `CheckpointingCheck.closed_mpi_processes` — `model_key` used before assignment in subset mode |
| 4.1.5 | `closedAcceleratorsPerHost` | ✅ Checked | `CheckpointingCheck.closed_accelerators_per_host` |
| 4.1.6 | `aggregateAcceleratorMemory` | ✅ Checked | `CheckpointingCheck.aggregate_accelerator_memory` — H100 memory hardcoded at 80 GB |
| 4.1.7 | `closedCheckpointParameters` | ✅ Checked | `CheckpointingCheck.closed_checkpoint_parameters` |
| 4.1.8 | `openSubmissionScaling` | ❌ Not checked | — |
| 4.1.9 | `checkpointPathArgs` | ✅ Checked | `CheckpointingCheck.checkpoint_path_args` |
| 4.1.10 | `checkpointFilesystemCheck` | ❌ Not checked | — |
| 4.1.11 | `subsetRunValidation` | ✅ Checked | `CheckpointingCheck.subset_run_validation` |
| 4.2.1 | `cacheFlushValidation` | ❌ Not checked | — |
| 4.2.2 | `totalTestDuration` | ❌ Not checked | — |
| 4.2.3 | `remappingTimeReporting` | ❌ Not checked | — |
| 4.2.4 | `simultaneousRwSupport` | ❌ Not checked | — |

### Sections 5–6 — VDB and KV Cache

All VDB (5.1–5.6) and KV Cache (6.1–6.6) rules are **not yet implemented**. Stub classes exist in `vdb_checks.py` and `kv_cache_checks.py`.

## Known Issues

- **`loader.py:120`** — `metadata_path` from the datagen loop is reused for all run timestamps instead of being recalculated per run directory.
- **`constants.py:22`** — Regex typo in v3.0 `DATAGEN_REQUIRED_FILES`: `,*summary\.json$` should be `.*summary\.json$`.
- **`checkpointing_checks.py:152`** — `model_key` referenced before assignment in the subset-mode branch of `closed_mpi_processes`.
- **`training_checks.py`** — `single_host_client_limit` (rule 3.3.4) is implemented but not registered in `init_checks`, so it never runs.
- **`constants.py:44-48`** — `CHECKPOINT_REQUIRED_FILES` uses `training_run` log prefixes instead of `checkpointing_run`.
