# MLPerf™ Storage V2.0 Benchmark Validation Rules
——————————————————————————————————————————

# Table of Contents

* [1. Introduction](#1-introduction)
* [2. Core/Common Rules for All Submissions](#2-corecommon-rules-for-all-submissions)
    * [2.1. Core/Common POSIX API Rules](#21-corecommon-posix-api-rules)
    * [2.2. Core/Common Object API Rules](#22-corecommon-object-api-rules)
* [3. Validating the Training Workloads](#3-validating-the-training-workloads)
    * [3.1. Training Sizing Options](#31-training-sizing-options)
    * [3.2. Training Generation Options](#32-training-generation-options)
    * [3.3. Training Run Options](#33-training-run-options)
    * [3.4. Training Access Via POSIX API Options](#34-training-access-via-posix-api-options)
    * [3.5. Training Access Via Object API Options](#35-training-access-via-object-api-options)
    * [3.6. Training OPEN versus CLOSED Options](#36-training-open-versus-closed-options)
* [4. Validating the Checkpointing Workloads](#4-validating-the-checkpointing-workloads)
    * [4.1. Checkpointing Sizing Options](#41-checkpointing-sizing-options)
    * 
    * [4.2. Checkpointing Generation Options](#42-checkpointing-generation-options)
    * [4.3. Checkpointing Run Options](#43-checkpointing-run-options)
    * [4.4. Checkpointing Access Via POSIX API Options](#44-checkpointing-access-via-posix-api-options)
    * [4.5. Checkpointing Access Via Object API Options](#45-checkpointing-access-via-object-api-options)
    * [4.6. Checkpointing OPEN versus CLOSED Options](#46-checkpointing-open-versus-closed-options)
    * [4.7. Storage System Must Be Simultaneously R/W or Remappable](#47-storage-system-must-be-simultaneously-rw-or-remappable)
* [5. Validating the VDB Workloads](#5-validating-the-vdb-workloads)
    * [5.1. VDB Sizing Options](#51-vdb-sizing-options)
    * [5.2. VDB Generation Options](#52-vdb-generation-options)
    * [5.3. VDB Run Options](#53-vdb-run-options)
    * [5.4. VDB Access Via POSIX API Options](#54-vdb-access-via-posix-api-options)
    * [5.5. VDB Access Via Object API Options](#55-vdb-access-via-object-api-options)
    * [5.6. VDB OPEN versus CLOSED Options](#56-vdb-open-versus-closed-options)
* [6. Validating the KVCache Options](#6-validating-the-kvcache-options)
    * [6.1. KVCache Sizing Options](#61-kvcache-sizing-options)
    * [6.2. KVCache Generation Options](#62-kvcache-generation-options)
    * [6.3. KVCache Run Options](#63-kvcache-run-options)
    * [6.4. KVCache Access Via POSIX API Options](#64-kvcache-access-via-posix-api-options)
    * [6.5. KVCache Access Via Object API Options](#65-kvcache-access-via-object-api-options)
    * [6.6. KVCache OPEN versus CLOSED Options](#66-kvcache-open-versus-closed-options)
# 1. Introduction

This document defines a valid MLPerf™ Storage *submission package*: the directory hierarchy a submitter delivers for review, together with the contents of the files in it.  A package is valid if and only if it satisfies every rule below; a package that violates any rule is INVALID.  The rules constrain both the structure of the package (which directories and files exist, and what they are named) and the values recorded in it (the options each benchmark was run with, their consistency with one another, and their agreement with the sanctioned tables in this document).

A submission package is produced by the `mlpstorage` command (1.1).  Results obtained by invoking the underlying benchmarks (e.g. DLIO) directly do not form a valid package.

**Scope of this document.** Rules.md defines a valid submission package and nothing else.  Every numbered rule is a declarative statement that can be decided by examining the package alone -- its directories and files, their contents, and the relationships between them -- together with the vocabulary, tables and file schemas needed to state such rules.  A rule that defines a published figure (e.g. 3.3.8, 4.3.6, 6.3.4) states the figure as a function of the package's contents.  Three kinds of text do not belong here: (1) rationale (why a rule is what it is) and explanation of how a value in the package is derived by the benchmark or the tool, which live in `RulesCommentary.md`, keyed by rule number; (2) what the `mlpstorage` command does at run time (checks, capture, warnings, defaults, invocation examples), which lives in `ManPage.md`; (3) editorial notes, open questions and references to issues or pull requests, which belong in the issue tracker.  Older sections of this document predate this statement and are being brought into line section by section; new and amended rules must follow it, with any commentary added to `RulesCommentary.md` under the same rule number.

1.1. **mlpstorageGeneratesHierarchy** -- Every file in a submission package is written by the `mlpstorage` command, with three exceptions that a submitter authors by hand: the per-system description files in each "systems" directory (2.1.7), the Markdown files 2.1.7 permits beside them, and the dot-prefixed entries 2.1.2 permits at the top level.  (Commentary: `RulesCommentary.md` §1.1.)

1.2. **storageBackendEnvVarDeclaration** -- Every storage-backend environment variable (the Storage-backend table under ENVIRONMENT in `ManPage.md`) that was set during a timed run must be declared in that run's configuration file.  A run during which an undeclared storage-backend variable was set is valid only under "open".

For cloud-specific credential and endpoint env vars consumed by s3dlio for Azure or GCS backends, see s3dlio's [Environment_Variables.md](https://github.com/mlcommons/s3dlio/blob/main/docs/Environment_Variables.md).

# 2. Core/Common Rules for All Submissions

## 2.1. Core/Common POSIX API Rules

2.1.1. **submitterRootDirectory** --  The submission structure must start from a single directory whose name is the name of the submitter.  This can be any string, but a blank or any other character in that string that cannot be part of a POSIX filename should be replaced 1-for-1 with a dash character.

2.1.2. **topLevelSubdirectories** --  Within the top-level directory of the submission structure there must be a directory named "closed" and/or one named "open", plus the code-image pool directory "code-images" (§2.1.6), and nothing more, with two exceptions: dot-prefixed entries (whose names begin with "."), such as version-control metadata (".git/", ".gitignore") and CI configuration (".github/"), are permitted; and a per-organization pool root (a directory named for a submitter that carries a ".mlps-image-pool" marker file, §2.1.6) is permitted.  These names are case-sensitive.  (Commentary: `RulesCommentary.md` §2.1.2.)

2.1.3. **openMatchesClosed** --  Whichever of the "open" and "closed" hierarchies are present must be constructed using the same rules described in the sections below.  The two hierarchies are individually optional: a submitter may submit to only "closed", only "open", or both, and there is no requirement that a submitter present in one hierarchy also be present in the other.

2.1.4. **closedSubmitterDirectory** --  Within the "closed" directory, each submitter's contribution lives in a directory whose name is the submitter's name (subject to 2.1.1).  Reviewers may run the submission checker against either a single submitter's pre-merge package (in which case the "closed" directory contains exactly one submitter directory, whose name matches the top-level submitter directory) or a merged tree containing multiple submitters' packages (in which case the "closed" directory contains one directory per participating submitter and the top-level directory is named for the merged set rather than any one submitter).  The same convention applies to the "open" directory per 2.1.3.

2.1.5. **requiredSubdirectories** -- The required subdirectories at the submitter level differ between CLOSED and OPEN submissions:

2.1.5.a. **requiredSubdirectoriesClosed** -- Within a CLOSED submitter directory, there must be exactly two directories: "results" and "systems".  These names are case-sensitive.  A "submission.yaml" manifest file (see "Provenance stamps and the submission manifest" below) sits beside them; this rule governs directories only.  Code images are not stored at the submitter level; every run leaf inside `results/` names its image in the tree-level pool (§2.1.6).

2.1.5.b. **requiredSubdirectoriesOpen** -- Within an OPEN submitter directory, there must be exactly two directories: "results" and "systems".  These names are case-sensitive.  The same "submission.yaml" manifest sits beside them.  OPEN runs use the same tree-level pool and per-leaf pointer as CLOSED runs (§2.1.6); nothing distinguishes the two divisions below the submitter directory except the rules that apply to the results.

See §2.1.6 and §2.1.27.

2.1.6. **codeDirectoryContents** -- Every run leaf (each *timestamp directory* under a "datasize", "datagen" or "run" *phase directory*, in every division) contains a ".mlps-code-image" file holding one line of the form "md5-tree-v2:<hash>", where <hash> is the *tree hash* (defined below) of the MLPerf Storage source tree that produced the leaf.  The package holds a *code-image pool*: the top-level "code-images" directory (§2.1.2) carrying a ".mlps-image-pool" marker file and, for every distinct <hash> that a leaf points at, a directory "code-<hash8>" where <hash8> is the first eight characters of <hash>.  Each "code-<hash8>" directory is a captured copy of that source tree (omitting the entries the tree hash excludes) accompanied by a top-level ".code-hash.json" file whose "hash" equals <hash>.  The pool is one per package, shared by every organization and by the "closed" and "open" hierarchies.  A per-organization pool -- "code-<hash8>" directories under a top-level directory named for the submitter and carrying its own ".mlps-image-pool" marker (the layout of the v3.0 release) -- is also valid; a leaf's pointer resolves against "code-images" first and the leaf's own organization pool second, and must resolve to an image.  (Commentary: `RulesCommentary.md` §2.1.6.)

The *tree hash* of a source tree is the MD5 digest, as 32 lowercase hex characters, of the concatenation, over the tree's files sorted by their POSIX-style relative path (UTF-8 byte order), of each file's relative path followed by its content -- excluding any directory named ".git", ".idea", ".vscode", ".claude", ".agent", ".agents", ".roo", ".planning", ".gsd-tmp", "__pycache__", ".pytest_cache", ".venv", "node_modules", "build", "dist", ".tox", "test" or "tests" (at any depth), any directory whose name ends in ".egg-info", any file named ".code-hash.json", ".DS_Store" or "Thumbs.db" or matching "*.pyc" or "*.pyo", and every symbolic link.

The ".code-hash.json" schema is:
- "hash": the tree hash of the captured copy (32-character lowercase hex).
- "algorithm": the string "md5-tree-v2".
- "captured_at": ISO-8601 UTC timestamp of the capture (e.g., "2026-06-16T15:42:11Z").
- "mlpstorage_version": the `mlpstorage` package version at capture time.
- "git_sha": full 40-character SHA of HEAD at capture, or null if unavailable.

For every "code-<hash8>" directory in a pool, the tree hash recomputed over its contents must equal the recorded "hash", and the recorded "hash" must begin with <hash8>.  For CLOSED submissions the recorded "hash" must additionally equal the reference digest of the sanctioned release (§3.6.1).

**Provenance stamps and the submission manifest** -- A run leaf (a `datasize`, `datagen` or `run` leaf, in any division) whose `*_metadata.json` declares `"provenance_file"` carries the `provenance.json` sidecar it names, and an organization directory may carry a `submission.yaml` manifest at `<mode>/<submitter_org>/submission.yaml`.  The two files record which rules edition, tool version and layout produced each leaf, which DLIO revision and client storage library ran, and which workload the leaf executed.

Three independent "versions" appear in both files:
- `rules_edition` (e.g. "3.0") -- the edition of this document a result was produced under.  It bumps only when a workload's numbers or validity change, or a workload is added or retired; it is the comparability key across rounds.  It is NOT the package version: every 3.0.x tool release implements edition "3.0".
- The tool version (`tool.version`, PEP 440, e.g. "3.0.46") -- provenance only.
- `layout_version` (an integer; 3 for trees with these files, 2 or 1 for earlier trees) -- the results-dir and leaf format; only readers key on it.

The `provenance.json` schema (`"schema": "mlps-run-provenance/1"`) is:
- "rules_edition", "layout_version" as above.
- "tool": {"name", "version", "git_sha", "code_image"} -- `code_image` repeats the leaf's `.mlps-code-image` pointer so the stamp reads whole without the pool.
- "dlio": {"version", "source", "commit"} -- the installed DLIO distribution and its git commit.
- "storage_library": {"name", "version"} -- the client library for `object` runs (`s3dlio`); `{"name": "none"}` for `file` runs.
- "core_config": {"algorithm": "core-config-v1", "allowlist", "hash", "keys"} -- a SHA-256 (first 16 hex digits) over the canonical JSON of the workload-defining DLIO parameters, keeping only the keys in the named allowlist (`mlpstorage_py/rules/core_config_keys.yaml`, e.g. "checkpointing@1").  Site tunables (the "Tunable Parameters for CLOSED" tables, paths, thread counts, listing and storage knobs) are excluded; "keys" lists exactly what was hashed, and the value is reproducible from the leaf alone.  Every family has an allowlist: `training@1` and `checkpointing@1` over the resolved DLIO parameters; `kv_cache@1` over the three Table KVCache-1 Options as emitted to `kv-cache.py`, the 6.3.2.1 seed and config, and the global flags forwarded; `vector_database@1` over the backend, index family, dataset scale (`num_vectors` x `dimension`) and recall target (`recall_k`, `search_limit`).
- "stamped_at", "stamped_by", and "provenance": one tag per stamp from the closed vocabulary `runtime`, `tool-constant`, `package-metadata`, `direct-url`, `code-hash-json`, `uv-lock`, `declared`, `inferred`, `n/a`, `unknown` saying how the value was obtained.
A stamp that cannot be determined is the string "unknown" -- never null, never absent.

The `submission.yaml` manifest (`schema: mlps-submission-manifest/1`) is an inventory, not a result: the organization and division, the declared `rules_edition` and `layout_version`, the generating tool, the distinct tool versions and DLIO commits seen, every `systems/` description (with its PDF), every run leaf with its benchmark, model, command, emulated accelerator, system, stamped rules edition, core-config hash, code image and the path of its stamp, and every code image referenced.  It carries no metrics and no Public IDs.

A leaf without `provenance.json` (one that never declared the sidecar) has a *derived stamp*: rules edition "unknown", with the tool, DLIO and storage-library values taken from the pointed-to image's `.code-hash.json` and `uv.lock` and from the leaf's own metadata (tagged `code-hash-json` / `uv-lock` / `inferred`).  Such a leaf is valid as it stands.  The following rules apply:
- PROV-01 (`leafProvenance`): a `provenance.json` that is present must parse and must name the same code image as the leaf's `.mlps-code-image` pointer; a leaf whose metadata declares the sidecar (`"provenance_file"`) but lacks it fails.  A leaf that never declared one draws no finding.
- PROV-02 (`submissionManifest`): a `submission.yaml` that is present must parse, must list exactly the run leaves the tree holds (a stale manifest fails), and its `rules_edition` must match every stamped leaf.  A tree without manifests draws no finding.

**Run-leaf and results-table integrity** --
- LEAF-01 (`runLeafMetadata`): a `kv_cache` or `vector_database` `run` leaf must carry exactly one `<benchmark-type>_<timestamp>_metadata.json`; a leaf without the file, or with more than one, fails.
- RPT-01 (`rollupTableAgreement`): every minted Public ID in an organization's rollup table (`<mode>/<submitter_org>/results/results.json`) must appear in a workload-level `results.json` under that organization's `results/<system>/` tree, and every minted ID in a workload-level table must appear in the rollup.  Rows without a Public ID (the identity-only `datagen` / `datasize` tables) are ignored; an organization with no rollup draws no finding.

**Rules editions and comparability classes** -- A *rules edition* is the version of this document a result was produced under ("3.0" today; the value every `provenance.json` and `submission.yaml` declares).  An edition bumps only when a workload's numbers or validity change, or a workload is added or retired; tool releases never bump it.  Which editions exist, what each sanctioned (division, family, model, emulated accelerator), which DLIO revisions and client storage libraries its rows ran on, and which results repository holds them is recorded in the machine-readable table `mlpstorage_py/rules/editions.yaml` (`schema: mlps-rules-editions/1`), maintained by the working group.  Its `current_edition` must equal the edition every stamped leaf and manifest declares when written.  Each edition that is checkable under this document carries a `checker` block -- the files and folders every datagen, run and checkpoint leaf of that edition must contain, the minimum AU per training model (3.3.2), the Table 2 CLOSED process count and checkpoint size per model (4.6.1, 4.3.4), the Table 3 simulated accelerator memory (4.3.4) and the KVCache sequence locks (6.3.2.1) -- and a submission is judged under the edition its `submission.yaml` declares (a submission without a manifest: `current_edition`), with the values of that edition's `checker` block wherever a rule below cites one, and only the rules bound to that edition apply to it.  An edition listed without a `checker` block is not checkable under this document (EDN-04).

The same table declares the *comparability classes*: a class is the working group's assertion that runs of one (division, family, model, emulated accelerator) whose `core-config-v1` hash is one of the class's listed hashes executed the same workload in every edition the class lists.  Only rows in one class are comparable, within a round or across rounds; nothing is inferred.  Comparisons across divisions are never declared: a class belongs to exactly one division, and an OPEN run that happens to use a CLOSED configuration is not in the CLOSED class.  A class may list several hashes (the same workload spelled differently by different DLIO generations, e.g. v1.0 `dataset.record_length` versus v2.0 `dataset.record_length_bytes`) and several editions (e.g. llama3-8b checkpointing is one class across 2.0 and 3.0).  Checkpointing classes of editions 3.0 and earlier use `accelerator: any`; from edition 4.0 on checkpoint classes are per accelerator (b200 and mi355 checkpoint rows do not compare).  For editions 3.0 and earlier, DLIO revisions and storage libraries are not part of the class key; they are recorded per edition as accepted lists.  From edition 4.0 on the working group may bring the DLIO revision into the key.  kv_cache and vector_database classes use `accelerator: any` permanently (neither family emulates an accelerator); the edition-3.0 seed is the Table KVCache-1 sequence (`llama3.1-8b-A`) and Milvus DISKANN with recall@10 at 1M x 1536 (`milvus-diskann-A`, the shipped `configs/vectordbbench/default.yaml`) and at 10M x 1536 (`milvus-diskann-B`, the `10m.yaml` scale).  The class is never written into a leaf or manifest; it is a function of the stamp and the table.  In `results.csv`, a row's `Rules Edition` and `Comparability Class` columns are the edition and class that every one of its run leaves resolves to (an unstamped leaf takes its edition from its submission's `submission.yaml`); the class is blank when the row's leaves fall into different classes or only some of them classify.  The following rules apply:
- EDN-01 (`rulesEdition`): a `provenance.json` or `submission.yaml` declaring a rules edition the table does not know fails.
- EDN-02 (`comparabilityClass`): a stamped `run` leaf whose (edition, division, family, model, emulated accelerator, core-config hash) resolves to no class fails under `closed/` and is reported as information under `open/` and `whatif/`; datasize/datagen leaves and leaves whose stamp carries no hash are skipped.
- EDN-03 (`dlioRevision`): a stamped `run` leaf whose DLIO commit is not in its edition's accepted list draws a warning.
- EDN-04 (`checkableEdition`): a `submission.yaml` declaring a rules edition the table lists without a `checker` block fails, and the submission's workload checks are skipped.
Leaves whose stamp is derived (no `provenance.json`, edition "unknown") draw no finding under any of the four.

2.1.7. **systemsDirectoryFiles** --  The "systems" directory must contain two files for each "system name", a .yaml file and a .pdf file, and nothing more, with two exceptions: Markdown files (any "*.md", e.g. "README.md", "NOTES.md") are permitted alongside the per-system files so submitters may include supplementary documentation, and dot-prefixed entries (such as ".DS_Store" or ".gitkeep") are ignored.  Each of the .yaml/.pdf files must be named with the "system name" ("<system name>.yaml" and "<system name>.pdf").  These names are case-sensitive.

2.1.8. **resultsDirectorySystems** --  The "results" directory, whether it is within the "closed" or "open" hierarchy, must include one or more directories, each named with a *system name*.  A *system name* is any string that is a valid POSIX directory name; it identifies the set of results collected from one configuration of the system-under-test, and it must equal the base name of the .yaml and .pdf files in "systems" that describe that system (2.1.7).

2.1.9. **identicalSystemConfig** --  The configuration parameters and the hardware and software components of the system-under-test recorded for every test and run within one *system name* directory hierarchy must be identical.  Results obtained after any change to those parameters or components belong under a separate *system name*.  The *system names* are case-sensitive.

2.1.10. **workloadCategories** --  Within a *system name* directory in the "results" directory, there must be one or both of the following directories, and nothing else: "training", and/or "checkpointing".  These names are case-sensitive.

2.1.11. **trainingWorkloads** --  Within the "training" directory, there must be one or more of the following *workload directories*, and nothing else: "unet3d" and/or "retinanet".  These names are case-sensitive.

2.1.12. **trainingPhases** --  Within the *workload directories* in the "training" hierarchy, there must exist *phase directories* named "datasize", "datagen", and "run", and nothing else.  These names are case-sensitive.

2.1.13. **datagenTimestamp** --  Within the "datagen" *phase directory* within the "training" directory hierarchy, there must be exactly one *timestamp directory* named *YYYYMMDD_HHmmss" that represent a *timestamp* of when that part of the test run was completed.  Where Y's are replaced with the year the run was performed, M's are replaced with the month, D's with the day, H's with the hour (in 24-hour format), m's with the minute, and s's with the second.  The timestamps should be relative to the local timezone where the test was actually run.

2.1.14. **datagenFiles** --  Within the *timestamp directory* within the "datagen" *phase*, there must exist the following files: "training_datagen.stdout.log", "training_datagen.stderr.log" file, "*output.json, "*per_epoch_stats.json", "*summary.json", and "dlio.log", plus a subdirectory named "dlio_config".  These names are case-sensitive.

2.1.15. **datagenDlioConfig** --  The "dlio_config" subdirectory in each *timestamp directory*  must contain the following list of files, and nothing else: "config.yaml", "hydra.yaml", and "overrides.yaml".  These names are case-sensitive.

2.1.16. **runResultsJson** --  Within the "run" *phase directory* within the "training" directory hierarchy, there must be one "results.json" file.  This name is case-sensitive.

2.1.17. **runTimestamps** --  Within the "run" *phase directory* within the "training" directory hierarchy, there must also be exactly 6 subdirectories named *YYYYMMDD_HHmmss" that represent a *timestamp* of when that part of the test run was completed.  Where Y's are replaced with the year the run was performed, M's are replaced with the month, D's with the day, H's with the hour (in 24-hour format), m's with the minute, and s's with the second.  The timestamps should be relative to the local timezone where the test was actually run.  Note that the 1st of those 6 is the *warm up* run and will not be included in the reported performance; §3.3.8 defines how the remaining 5 are reduced to the published figures.

2.1.18. **runTimestampGap** --  The timestamp (the day and time) represented by the name of each *timestamp directory* must be separated by less than the duration of a single *timestamp directory* from its neighboring *timestamp directories*.  (Commentary: `RulesCommentary.md` §2.1.18.)

2.1.18a.  **runPickingSubsets** -- It is permissible to execute a large number of runs (all consecutive) and then delete the *timestamp directories* for all but 6 consecutive runs in the middle; the 6 that remain must satisfy **runTimestampGap** (2.1.18).

2.1.19. **runFiles** --  Within each *timestamp directory* within the "run" *phase*, there must exist the following files: "training_run.stdout.log", "training_run.stderr.log" file, "*output.json, "*per_epoch_stats.json", "*summary.json", and "dlio.log", plus a subdirectory named "dlio_config".  These names are case-sensitive.

2.1.20. **runDlioConfig** --  The "dlio_config" subdirectory in each *timestamp directory* must contain the following list of files, and nothing else: "config.yaml", "hydra.yaml", and "overrides.yaml".  These names are case-sensitive.

2.1.21. **checkpointingWorkloads** --  Within the "checkpointing" directory, there must be one or more of the following *workload directories*, and nothing else: "llama3-8b", "llama3-70b", "llama3-405b", and/or "llama3-1t".  These names are case-sensitive.

2.1.22. **checkpointingResultsJson** --  Within the *workload directories* within the "checkpointing" directory hierarchy, there must be one "results.json" file.  This name is case-sensitive.

2.1.23. **checkpointingTimestamps** --  Within the *workload directories* within the "checkpointing" directory hierarchy, there must be either one or two *timestamp directories* named *YYYYMMDD_HHmmss" that represent a *timestamp* of when that part of the test run was completed (one timestamp directory per invocation, per §4.7.1: a single combined invocation OR a write-phase invocation followed by a read-phase invocation).  Where Y's are replaced with the year the run was performed, M's are replaced with the month, D's with the day, H's with the hour (in 24-hour format), m's with the minute, and s's with the second.  The timestamps should be relative to the local timezone where the test was actually run.

2.1.24. **checkpointingTimestampGap** --  The timestamp (the day and time) represented by the name of each *timestamp directory* must be separated by less than the duration of a single *timestamp directory* from its neighboring *timestamp directories*.  (Commentary: `RulesCommentary.md` §2.1.18.)

2.1.24a.  **checkpointingPickingSubsets** -- It is permissible to execute a large set of runs (all consecutive) and then delete all but one run (with 10 checkpoint files in it) in the middle.

2.1.25. **checkpointingFiles** --  Within the *timestamp directories* within the "checkpointing" directory hierarchy, there must exist the following files: "checkpointing_run.stdout.log", "checkpointing_run.stderr.log" file, "*output.json, "*per_epoch_stats.json", "*summary.json", and "dlio.log", plus a subdirectory named "dlio_config".  These names are case-sensitive.

2.1.26. **checkpointingDlioConfig** --  The "dlio_config" subdirectory in each *timestamp directory* must contain the following list of files, and nothing else: "config.yaml", "hydra.yaml", and "overrides.yaml".  These names are case-sensitive.

2.1.27. **directoryDiagram** --  Pictorially, here is what this looks like:
```
root_folder (or any name you prefer)
├── code-images
│ 	├── .mlps-image-pool
│ 	├── code-<hash8>
│ 	└── ... (one directory per distinct captured source tree; see §2.1.6)
├── Closed
│ 	└──<submitter_org>
│	  	├── submission.yaml  (per-organization manifest; see "Provenance stamps and the submission manifest")
│	  	├── results
│	  	│	└──system-name-1
│	  	│	 	├── training
│	  	│	 	│	├── unet3d
│	  	│		│	│	├── datagen
│	  	│		│	│	│	└── YYYYMMDD_HHmmss
│	  	│		│	│	│		├── provenance.json  (in every YYYYMMDD_HHmmss leaf below; omitted from the rest of this diagram)
│	  	│		│	│	│		└── dlio_config
│	  	│		│	│	└── run
│	  	│		│	│		├──results.json
│	  	│		│	│		├── YYYYMMDD_HHmmss
│	  	│		│	│		│	└── dlio_config 
│	  	│		│	│		... (5x Runs per Emulated Accelerator Type)
│	  	│		│	│		└── YYYYMMDD_HHmmss
│	  	│		│	│			└── dlio_config
│	  	│	 	│	└── retinanet
│	  	│		│	 	├── datagen
│	  	│		│	 	│	└── YYYYMMDD_HHmmss
│	  	│		│	 	│		└── dlio_config
│	  	│		│	 	└── run
│	  	│		│			├──results.json
│	  	│		│	 		├── YYYYMMDD_HHmmss
│	  	│		│	 		│	└── dlio_config 
│	  	│		│	 		... (5x Runs per Emulated Accelerator Type)
│	  	│		│	 		└── YYYYMMDD_HHmmss
│	  	│		│	 			└── dlio_config
│	  	│	 	├── checkpointing
│	  	│	 	│	├── llama3-8b
│	  	│		│	│	├──results.json
│	  	│		│	│	├── YYYYMMDD_HHmmss
│	  	│		│	│	│	└── dlio_config 
│	  	│		│	│ 	... (10x Runs for Read and Write. May be combined in a single run)
│	  	│		│	│	└── YYYYMMDD_HHmmss
│	  	│		│	│		└── dlio_config
│	  	│	 	│	├── llama3-70b
│	  	│		│	│	├──results.json
│	  	│		│	│	├── YYYYMMDD_HHmmss
│	  	│		│	│	│	└── dlio_config 
│	  	│		│	│ 	... (10x Runs for Read and Write. May be combined in a single run)
│	  	│		│	│	└── YYYYMMDD_HHmmss
│	  	│		│	│		└── dlio_config
│	  	│	 	│	├── llama3-405b
│	  	│		│	│	├──results.json
│	  	│		│	│	├── YYYYMMDD_HHmmss
│	  	│		│	│	│	└── dlio_config 
│	  	│		│	│ 	... (10x Runs for Read and Write. May be combined in a single run)
│	  	│		│	│	└── YYYYMMDD_HHmmss
│	  	│		│	│		└── dlio_config
│	  	│	 	│	└── llama3-1t
│	  	│		│		├──results.json
│	  	│		│	 	├── YYYYMMDD_HHmmss
│	  	│		│	 	│	└── dlio_config 
│	  	│		│	 	... (10x Runs for Read and Write. May be combined in a single run)
│	  	│		│		└── YYYYMMDD_HHmmss
│	  	│		│	 		└── dlio_config
│	  	│	 	└── vector_database
|		|			├── AISAQ
│	  	│	 		|	├── datagen
│	  	│			|	│	└── YYYYMMDD_HHmmss
│	  	│			|	│		└── summary.json
│	  	│			|	└── run
│	  	│			|		├── YYYYMMDD_HHmmss
│	  	│			|		│	└── summary.json
│	  	│			|		... (5x Runs total)
│	  	│			|		└── YYYYMMDD_HHmmss
│	  	│			|			└── summary.json
|		|			├── DISKANN
│	  	│	 		|	├── datagen
│	  	│			|	│	└── YYYYMMDD_HHmmss
│	  	│			|	│		└── summary.json
│	  	│			|	└── run
│	  	│			|		├── YYYYMMDD_HHmmss
│	  	│			|		│	└── summary.json
│	  	│			|		... (5x Runs total)
│	  	│			|		└── YYYYMMDD_HHmmss
│	  	│			|			└── summary.json
|		|			└── HNSW
│	  	│	 			├── datagen
│	  	│				│	└── YYYYMMDD_HHmmss
│	  	│				│		└── summary.json
│	  	│				└── run
│	  	│					├── YYYYMMDD_HHmmss
│	  	│					│	└── summary.json
│	  	│					... (5x Runs total)
│	  	│					└── YYYYMMDD_HHmmss
│	  	│						└── summary.json
│	  	└── systems
│	  		├──system-name-1.yaml
│	  		├──system-name-1.pdf
│	  		├──system-name-2.yaml
│	  		└──system-name-2.pdf
│
└── Open
 	└──<submitter_org>
		├── results
		│	└──system-name-1
		│	 	├── training
		│	 	│	├── unet3d
		│		│	│	├── datagen
		│		│	│	│	└── YYYYMMDD_HHmmss
		│		│	│	│		└── dlio_config
		│		│	│	└── run
		│		│	|		├──results.json
		│		│	│		├── YYYYMMDD_HHmmss
		│		│	│		│	└── dlio_config 
		│		│	│		... (5x Runs per Emulated Accelerator Type)
		│		│	│		└── YYYYMMDD_HHmmss
		│		│	│			└── dlio_config
		│	 	│	└── retinanet
		│		│	 	├── datagen
		│		│	 	│	└── YYYYMMDD_HHmmss
		│		│	 	│		└── dlio_config
		│		│	 	└── run
		│		│			├──results.json
		│		│	 		├── YYYYMMDD_HHmmss
		│		│	 		│	└── dlio_config 
		│		│	 		... (5x Runs per Emulated Accelerator Type)
		│		│	 		└── YYYYMMDD_HHmmss
		│		│	 			└── dlio_config
	  	│	 	├── checkpointing
	  	│	 	│	├── llama3-8b
	  	│		│	│	├──results.json
	  	│		│	│	├── YYYYMMDD_HHmmss
	  	│		│	│	│	└── dlio_config 
	  	│		│	│ 	... (10x Runs for Read and Write. May be combined in a single run)
	  	│		│	│	└── YYYYMMDD_HHmmss
	  	│		│	│		└── dlio_config
	  	│	 	│	├── llama3-70b
	  	│		│	│	├──results.json
	  	│		│	│	├── YYYYMMDD_HHmmss
	  	│		│	│	│	└── dlio_config 
	  	│		│	│ 	... (10x Runs for Read and Write. May be combined in a single run)
	  	│		│	│	└── YYYYMMDD_HHmmss
	  	│		│	│		└── dlio_config
	  	│	 	│	├── llama3-405b
	  	│		│	│	├──results.json
	  	│		│	│	├── YYYYMMDD_HHmmss
	  	│		│	│	│	└── dlio_config 
	  	│		│	│ 	... (10x Runs for Read and Write. May be combined in a single run)
	  	│		│	│	└── YYYYMMDD_HHmmss
	  	│		│	│		└── dlio_config
	  	│	 	│	└── llama3-1t
	  	│		│		├──results.json
	  	│		│	 	├── YYYYMMDD_HHmmss
	  	│		│	 	│	└── dlio_config 
	  	│		│	 	... (10x Runs for Read and Write. May be combined in a single run)
	  	│		│		└── YYYYMMDD_HHmmss
	  	│		│	 		└── dlio_config
	  	│	 	└── vector_database
		|			├── AISAQ
	  	│	 		|	├── datagen
	  	│			|	│	└── YYYYMMDD_HHmmss
	  	│			|	│		└── summary.json
	  	│			|	└── run
	  	│			|		├── YYYYMMDD_HHmmss
	  	│			|		│	└── summary.json
	  	│			|		... (5x Runs total)
	  	│			|		└── YYYYMMDD_HHmmss
	  	│			|			└── summary.json
		|			├── DISKANN
	  	│	 		|	├── datagen
	  	│			|	│	└── YYYYMMDD_HHmmss
	  	│			|	│		└── summary.json
	  	│			|	└── run
	  	│			|		├── YYYYMMDD_HHmmss
	  	│			|		│	└── summary.json
	  	│			|		... (5x Runs total)
	  	│			|		└── YYYYMMDD_HHmmss
	  	│			|			└── summary.json
		|			└── HNSW
	  	│	 			├── datagen
	  	│				│	└── YYYYMMDD_HHmmss
	  	│				│		└── summary.json
	  	│				└── run
	  	│					├── YYYYMMDD_HHmmss
	  	│					│	└── summary.json
	  	│					... (5x Runs total)
	  	│					└── YYYYMMDD_HHmmss
	  	│						└── summary.json
		└── systems
			├──system-name-1.yaml
			├──system-name-1.pdf
			├──system-name-2.yaml
			└──system-name-2.pdf
```
2.29. **dlioLog** --  The *timestamp directory* has the same structure in every case, shown pictorially below.  "mlpstorage.log" (every message the `mlpstorage` command logged during that invocation) and "mlpstorage.errors.log" (its WARNING-and-above subset) may be present; their absence does not invalidate the directory:
```
└── YYYYMMDD_HHmmss
    ├── [training|checkpointing]_[datagen|run].stdout.log
    ├── [training|checkpointing]_[datagen|run].stderr.log
    ├── mlpstorage.log
    ├── mlpstorage.errors.log
    ├── *[output|per_epoch_stats|summary].json
    ├── dlio.log
    └── dlio_config
        ├── config.yaml
        ├── hydra.yaml
        └── overrides.yaml
```

## 2.2. Core/Common Object API Rules

# 3. Validating the Training Workloads

## 3.1. Training Sizing Options

3.1.1. **trainingVerifyDatasizeUsage** -- The *submission validator* must verify that the *datasize* option was used by finding the entry(s) in the log file showing its use.

3.1.2. **trainingRecalculateDatasetSize** -- The *submission validator* must recalculate the minimum dataset size by using the provided number of simulated accelerators and the sizes of all of the host node’s memory as reported in the logfiles as described below and fail the run if the size recorded in the run's logfile doesn't exactly match the recalculated value.
  * Calculate required minimum samples given number of steps per epoch (NB: `num_steps_per_epoch` is a minimum of 500):
     * `min_samples_steps_per_epoch = num_steps_per_epoch * batch_size * num_accelerators_across_all_nodes`
  * Calculate required minimum samples given host memory to eliminate client-side caching effects; (NB: HOST_MEMORY_MULTIPLIER = 5):
     * `min_samples_host_memory_across_all_nodes = number_of_hosts * memory_per_host_in_GB * HOST_MEMORY_MULTIPLIER * 1024 * 1024 * 1024 / record_length`
  * Ensure we meet both constraints:
     * `min_samples = max(min_samples_steps_per_epoch, min_samples_host_memory_across_all_nodes)`
  * Calculate minimum files to generate
     * `min_total_files= min_samples / num_samples_per_file`
     * `min_files_size = min_samples * record_length / 1024 / 1024 / 1024`
  * A minimum of `min_total_files` files are required which will consume `min_files_size` GB of storage.

## 3.2. Training Generation Options

3.2.1. **trainingDatagenMinimumSize** --  The amount of data generated during the *datagen* phase must be equal **or larger** -- than the amount of data calculated during the *datasize* phase or the run must be failed.

## 3.3. Training Run Options

3.3.1. **trainingRunDataMatchesDatasize** -- The amount of data the *run* phase is told to use must be exactly equal to the *datasize* value calculated earlier, but can be less than the value used in the *datagen* phase.  To express that, you can run the benchmark on a subset of that dataset by setting `num_files_train` or `num_files_eval` smaller than the number of files available in the dataset folder, but `num_subfolders_train` and `num_subfolders_eval` must be to be equal to the actual number of subfolders inside the dataset folder in order to generate valid results.  Within the *timestamp directory* in the "datasize" *phase directory*, there must exist a ``*_metadata.json`` file whose ``parameters.dataset`` block records the outputs of the datasize calculation --- `num_files_train`, `num_subfolders_train`, and `total_disk_bytes` --- and whose ``args`` block records the inputs that produced them --- `model`, `accelerator_type`, `max_accelerators`, `client_host_memory_in_gb`, and `data_dir`.  These are the values the *run* phase is cross-checked against.

3.3.2. **trainingAcceleratorUtilizationCheck** -- To pass a benchmark run, the AU (Accelerator Utilization) should be equal to or greater than the minimum value:
  * `total_compute_time = (records_per_file * total_files) / simulated_accelerators / batch_size * computation_time * epochs`
  * `AU = (total_compute_time/total_benchmark_running_time) * 100`
  * All the I/O operations from the first step are excluded from the AU calculation. The I/O operations that are excluded from the AU calculation are included in the samples/second reported by the benchmark, however.
  * The minimum for each model is the `metric.au` of its workload template and is recorded per rules edition in `mlpstorage_py/rules/editions.yaml` (edition 3.0: unet3d 90 %, retinanet 85 %).  The *submission validator* fails a run whose mean AU over its epochs is below the edition's minimum, as well as a run the benchmark itself judged below its template's threshold.

3.3.3. **trainingSingleHostSimulatedAccelerators** -- For single-host submissions, increase the number of simulated accelerators by changing the `--num-accelerators` parameter to the benchmark.sh script. Note that the benchmarking tool requires approximately 0.5GB of host memory per simulated accelerator.

3.3.4. **trainingSingleHostClientLimit** -- For single-host submissions, in both CLOSED and OPEN division results, the validator should fail the run if there is more than one client node used during that run.

3.3.5. **trainingDistributedDataAccessibility** -- For distributed Training submissions, all the data must be accessible to all the host nodes.  **_(not clear how to check this, so maybe remove?)_**

3.3.6. **trainingIdenticalAcceleratorsPerNode** -- For distributed Training submissions, the number of simulated accelerators in each host node must be identical.

3.3.7. **trainingNodeCapabilityConsistency** -- For distributed Training submissions, the *submission validation checker* should emit a warning (not fail the validation) if the physical nodes that run the benchmark code are widely enough different in their capability.  **_(not clear we should do this, so maybe remove?)_**

3.3.8. **trainingResultAggregation** -- The training figures published for a *workload directory* are functions of its 5 measured *timestamp directories* (§2.1.17): the *warm up* directory is excluded and no other directory is excluded.  `Read B/W (GiB/s)` in `results.csv` is the arithmetic mean, over those 5 directories, of each directory's `*summary.json` field `metric.train_io_mean_MB_per_second`, divided by 1024.  Each `train_mean_of_<name>` value in the "run" *phase directory*'s `results.json` is the arithmetic mean, over the 5 directories, of each directory's arithmetic mean of its per-epoch list `metric.train_<name>`.  The §3.3.2 AU minimum applies to each *timestamp directory* individually.  A *workload directory* in which any measured *timestamp directory* has an empty per-epoch metric list is INVALID; `Read B/W (GiB/s)` is blank when any measured *timestamp directory* lacks `metric.train_io_mean_MB_per_second`.  (Commentary: `RulesCommentary.md` §3.3.8.)


## 3.4. Training Access Via POSIX API Options

3.4.1. **trainingMlpstoragePathArgs** --  The arguments to `mlpstorage` that set the directory pathname where the dataset is stored and the directory where the output logfiles are stored must both be set and must be set to different values.

3.4.2. **trainingMlpstorageFilesystemCheck** --  The `mlpstorage` command should do a "df" command on the directory pathname where the dataset is stored and another one on the directory pathname where the output logfiles are stored and record those values in the logfile.  The *submission validator* should find those entries in the run's logfile and verify that they are different filesystems.  We don't want the submitter to, by acccident, place the logfiles onto the storage system under test since that would skew the results.

## 3.5. Training Access Via Object API Options

## 3.6. Training OPEN versus CLOSED Options

3.6.1. **trainingClosedSubmissionChecksum** -- For CLOSED submissions of this benchmark, the MLPerf Storage codebase must not be changed.  The *submission validation checker* enforces this with a layered check:

  (a) **Self-consistency check (always runs):** the validator recomputes the captured `code/` tree's MD5 (per the exclusion set documented in §2.1.6) and compares it against the recorded "hash" in `.code-hash.json`.  This detects post-capture tampering of the submission package itself.

  (b) **Upstream-identity check (CLOSED only):** the validator additionally compares the captured tree's MD5 against a pinned digest from `REFERENCE_CHECKSUMS` (or a value supplied via the `--reference-checksum` CLI flag).  When no pinned digest is configured, the upstream-identity check is skipped with a single warning per run; the self-consistency check (a) still runs and can still fail.  The pinned digest, when present, must be computed against the same exclusion set as the runtime capture (currently dotfiles, dotdirs, `test/`, `tests/`, `__pycache__/`, `.egg-info/`, `*.pyc`, and `.code-hash.json` itself).

3.6.2. **trainingClosedSubmissionParameters** -- For CLOSED submissions of this benchmark, only a small number of parameters can be modified, and those parameters are listed in the table below.  Any other parameters being modified must generate a message and fail the validation.

**Table: Training Workload Tunable Parameters for CLOSED**

| Parameter                    | Description                                                                                                                         | Default  |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|----------|
| *Dataset parameters*         |                                                                                                                                     |          |
| dataset.num_files_train      | Number of files for the training set                                                                                                | --       |
| dataset.num_subfolders_train | Number of subfolders that the training set is stored                                                                                | 0        |
| dataset.data_folder          | The path where dataset is stored                                                                                                    | --       |
|                              |                                                                                                                                     |          |
| *Reader parameters*          |                                                                                                                                     |          |
| reader.read_threads          | Number of threads to load the data                                                                                                  | --       |
| reader.computation_threads   | Number of threads to preprocess the data (only for resnet)                                                                          | --       |
| reader.transfer_size         | An int64 scalar representing the number of bytes in the read buffer. (only supported for Tensorflow models -- Resnet and Cosmoflow) |          |
| reader.prefetch_size         | An int64 scalar representing the amount of prefetching done, with values of 0, 1, or 2.                                             |          |
| reader.odirect               | Enable ODIRECT mode for Unet3D Training                                                                                             | False    |
|                              |                                                                                                                                     |          |
| *Storage parameters*         |                                                                                                                                     |          |
| storage.storage_root         | The storage root directory                                                                                                          | ./       |
| storage.storage_type         | The storage type                                                                                                                    | local_fs |
| storage.storage_options.prefetch_window | Client-side s3dlio prefetch depth for object-storage runs. Same class of knob as `reader.read_threads`: tunes client I/O concurrency, not workload semantics (same bytes, same access pattern, same AU rule). Object-storage-only: on POSIX, prefetch flows through the reclaimable page cache and imposes no whole-record host-memory pressure; on object storage each prefetch slot pins a whole record until consumed, so the closed default may leave no valid operating point for large-record models. Uncapped, matching `reader.read_threads`. | s3dlio default |

3.6.3. **trainingOpenSubmissionParameters** -- For OPEN submissions of this benchmark, only a few additional parameters can be modified over those allowed in CLOSED, and those additional parameters are listed in the table below.  Any other parameters being modified must generate a message and fail the validation.

**Table: Training Workload Tunable Parameters for OPEN**

| Parameter                    | Description                                | Default                                                                               |
|------------------------------|--------------------------------------------|---------------------------------------------------------------------------------------|
| framework                    | The machine learning framework.            | 3D U-Net: PyTorch<br>ResNet-50: Tensorflow<br>Cosmoflow: Tensorflow                   |
|                              |                                            |                                                                                       |
| *Dataset parameters*         |                                            |                                                                                       |
| dataset.format               | Format of the dataset.                     | 3D U-Net: .npz<br>ResNet-50: .tfrecord<br>Cosmoflow: .tfrecord                        |
| dataset.num_samples_per_file |                                            | 3D U-Net: 1<br>ResNet-50: 1251<br>Cosmoflow: 1                                        |
|                              |                                            |                                                                                       |
| *Reader parameters*          |                                            |                                                                                       |
| reader.data_loader           | Supported options: Tensorflow or PyTorch.  | 3D U-Net: PyTorch<br>ResNet-50: Tensorflow<br>Cosmoflow: Tensorflow                   |

# 4. Validating the Checkpointing Workloads

## 4.1. Checkpointing Sizing Options

## 4.2. Checkpointing Generation Options

## 4.3. Checkpointing Run Options

4.3.1. **checkpointDataSizeRatio** -- The checkpoint data written per client node must be more than 3x the client node's memory capacity, otherwise the filesystem cache needs to be cleared between the write and read phases.

4.3.2. **checkpointFsyncVerification** -- We must verify that all the benchmark workload configuration files have been set to do an fsync call at the end of each of the 10 checkpoint writes.

4.3.3. **checkpointModelConfigurationReq** -- The benchmark must be run with one of the four model configuration detailed below.

4.3.4. **checkpointAggregateAcceleratorMemory** -- The aggregate simulated accelerator memory across all nodes must be sufficient to accommodate the model’s checkpoint size.  That is, the GB of memory associated with the chosen accelerator times the accelerator count must be equal to or greater than the total checkpoint size for that scale of checkpoint.  (see table 2)  The chosen accelerator is declared with `--accelerator-type` on `checkpointing run` and recorded in the run's metadata; the *submission validator* multiplies the accelerator's memory from Table 3 by the run's accelerator count and fails a run whose product is below the checkpoint size it wrote, whose metadata records no accelerator, or whose accelerator is not in Table 3.  The `mlpstorage` command applies the same check against the Table 2 checkpoint size before launching a run.

**Table 3 Simulated accelerator memory**

| Accelerator | Memory (GB) | Divisions        |
|-------------|-------------|------------------|
| b200        | 180         | CLOSED, OPEN     |
| mi355       | 288         | CLOSED, OPEN     |
| h100        | 80          | whatif only      |
| a100        | 80          | whatif only      |

**Table 2 LLM models**

| Model                  | 8B     | 70B    | 405B    | 1T     |
|------------------------|--------|--------|---------|--------|
| Hidden dimension       | 4096   | 8192   | 16384   | 25872  |
| FFN size               | 14336  | 28672  | 53248   | 98304  |
| num_attention_heads    | 32     | 128    | 128     | 192    |
| num_kv_heads           | 8      | 8      | 8       | 32     |
| Num layers             | 32     | 80     | 126     | 128    |
| Parallelism (TPxPPxDP) | 1×1×8  | 8×1x8  | 8×32×2  | 8×64×2 |
| Total Processes        | 8      | 64     | 512     | 1024   |
| ZeRO                   | 3      | 3      | 1       | 1      |
| Checkpoint size        | 105 GB | 912 GB | 5.29 TB | 18 TB  |
| Subset: 8-Process Size | 105 GB | Invalid | Invalid | Invalid |

*The "Invalid" entries are deliberate: subset mode is defined only for the 8B model (see rule 4.3.5).*

*Units: the "Checkpoint size" row is in binary units (GiB and TiB, 1024-based) as reported by the benchmark's `checkpoint_size_GB`, even though it is labeled GB/TB. The values are the ones the v3.0 round validated against and are kept as-is so v4.0 results remain comparable with v3.0.*

4.3.5. **checkpointSubsetRunValidation** --  The `mlpstorage` command must accept a parameter declaring the run a *subset* run and must record that declaration in the run's output log file. A *subset* run must use the "8B" model and a total of exactly 8 accelerators. The *submission validator* must flag an error for any *subset* run that uses any other model or any other accelerator count.

*Aside (not part of the rule): subset mode exists for storage architectures that centrally manage storage local to the client nodes, whose aggregate checkpoint bandwidth therefore scales linearly with node count. One 8-GPU node running the 8B workload demonstrates such an architecture's per-node bandwidth; the larger models measure storage where checkpoint data must reach a shared central store, so no subset form is defined for them.*

4.3.6. **checkpointResultAggregation** -- The checkpointing figures published for a *workload directory* are functions of its one or two *timestamp directories* (§2.1.23); none is excluded.  A *timestamp directory* whose recorded `checkpoint.num_checkpoints_write` is greater than 0 is a *write-phase directory*, and one whose recorded `checkpoint.num_checkpoints_read` is greater than 0 is a *read-phase directory*; a combined invocation is both.  In `results.csv`, `Write B/W (GiB/s)` and `Write Duration (secs)` are the arithmetic means, over the *write-phase directories*, of each directory's `*summary.json` fields `metric.save_checkpoint_io_mean_GB_per_second` and `metric.save_checkpoint_duration_mean_seconds`; `Read B/W (GiB/s)` and `Read Duration (secs)` are the arithmetic means, over the *read-phase directories*, of `metric.load_checkpoint_io_mean_GB_per_second` and `metric.load_checkpoint_duration_mean_seconds`.  The bandwidth fields are in binary GiB/s (see the units note under Table 2).  A column is blank when any directory of its phase lacks the field it is taken from.  (Commentary: `RulesCommentary.md` §4.3.6.)


## 4.4. Checkpointing Access Via POSIX API Options

4.4.1. **checkpointPathArgs** --  The arguments to `mlpstorage` that set the directory pathname where the checkpoints are written and read and the directory where the output logfiles are stored must both be set and must be set to different values.

4.4.2. **checkpointFilesystemCheck** --  The `mlpstorage` command should do a "df" command on the directory pathname where the checkpoints are written and read and another one on the directory pathname where the output logfiles are stored and record those values in the logfile.  The *submission validator* should find those entries in the run's logfile and verify that they are different filesystems.  We don't want the submitter to, by acccident, place the logfiles onto the storage system under test since that would skew the results.

## 4.5. Checkpointing Access Via Object API Options

## 4.6. Checkpointing OPEN versus CLOSED Options

4.6.1. **checkpointClosedMpiProcesses** -- For CLOSED submissions, the number of MPI processes must be set to 8, 64, 512, and 1024 for the respective models.  (see table 2)  No reduced-process form exists for the 70B, 405B, or 1T models; the only permitted *subset* run (the "8B" model, rule 4.3.5) uses that model's required 8 processes.

4.6.2. **checkpointClosedAcceleratorsPerHost** -- For CLOSED submissions, submitters may adjust the number of simulated accelerators **per host**, as long as each host uses more than 4 simulated accelerators and the total number of simulated accelerators (the total number of processes) matches the requirement.  (see table 2)

4.6.3. **checkpointClosedCheckpointParameters** -- For CLOSED submissions of this benchmark, only a small number of parameters can be modified, and those parameters are listed in the table below.  Any other parameters being modified must generate a message and fail the validation.

**Table: Checkpoint Workload Tunable Parameters for CLOSED**

| Parameter                        | Description                                                 | Default               |
|----------------------------------|-------------------------------------------------------------|-----------------------|
| checkpoint.checkpoint_folder     | The storage directory for writing and reading checkpoints   | ./checkpoints/<model> |

4.6.4. **checkpointOpenSubmissionScaling** -- For OPEN submissions of this benchmark, the total number of processes may be increased in multiples of (TP×PP) to showcase the scalability of the storage solution.

**Table 3: Configuration parameters and their mutability in CLOSED and OPEN divisions**

| Parameter                          | Meaning                                      | Default value                                 | Changeable in CLOSED | Changeable in OPEN |
|------------------------------------|----------------------------------------------|-----------------------------------------------|----------------------|--------------------|
| --ppn hostname:slotcount           | Number of processes per node                 | N/A                                           | YES (minimal 4)      | YES (minimal 4)    |
| --num-processes                    | Total number of processes                    | Node local: 8<br>Global: the value in Table 1 | NO                   | YES                |
| --checkpoint-folder                | The folder to save the checkpoint data       | checkpoint/{workload}                         | YES                  | YES                |
| --num-checkpoints-write            | Number of write checkpoints                  | 10 (or 0**)                                   | Only 10 or 0**       | YES                |
| --num-checkpoints-read             | Number of read checkpoints                   | 10 (or 0**)                                   | Only 10 or 0**       | YES                |

**NOTE: In the ``--ppn`` syntax above, the ``slotcount`` value means the number of processes per node to run.**

**\*\* NOTE: In CLOSED submissions, ``--num-checkpoints-write`` and ``--num-checkpoints-read`` may be set to ``0`` only as part of the two-invocation failover workflow described in §4.7.1: one invocation runs the write phase with ``--num-checkpoints-read=0`` and the next runs the read phase with ``--num-checkpoints-write=0``. The default for both flags is 10 and the total work performed across both invocations must still be 10 writes followed by 10 reads.**

## 4.7. Storage System Must Be Simultaneously R/W or _Remappable_

4.7.1. **checkpointCacheFlushValidation** -- Checkpointing models the failure of a client node followed by another client picking up the last checkpoint file written by the failed node for the read phase. In every submission the write phase (10 checkpoint files written) runs first, followed by the read phase (10 checkpoint files read). When the storage system supports the client-to-client handoff transparently — i.e., the read phase can proceed immediately after the write phase without external orchestration — the write and read phases may be executed as a single combined invocation, and no gap check applies. Storage system architectures that require an external callout (e.g., a submitter-provided script) to complete the failover between the writing and reading clients must instead execute the write and read phases as two separate invocations, with the submitter's failover callout occurring between them. A common in-callout activity is clearing a client-side filesystem cache when the total checkpoint size written per client is less than 3× the client node's memory capacity (see ``checkpointing/README.md``); the callout is not limited to that activity. To ensure the callout is a lightweight programmatic step rather than a long-running manual procedure, the validator confirms that the read-phase invocation was launched — i.e., its ``mlpstorage`` process reached the entry point of ``main.py`` — within 30 seconds of the write-phase invocation ending. The gap is measured as ``read.invocation_start_time − write.summary.end_time``, where ``invocation_start_time`` is captured at ``mlpstorage`` process start (before framework startup, MPI spawn, and other unavoidable per-invocation overhead) and ``end_time`` is recorded in the write invocation's ``summary.json``. A negative gap indicates clock skew between the write and read nodes and is reported as such rather than as a causality violation.

4.7.2. **checkpointTotalTestDuration** -- The validator must verify that the total test duration starts from the timestamp of the first checkpoint written and ends at the ending timestamp of the last checkpoint read, notably including the "remapping" time.

4.7.3. **checkpointRemappingTimeReporting** -- For a _remapping_ solution, the time duration between the checkpoint being completed and the earliest time that that checkpoint could be read by a different host node must be reported in the `SystemDescription.yaml` file.

4.7.4. **checkpointSimultaneousRwSupport** -- The system_configuration.yaml document must list whether the solution support simultaneous reads and/or writes as such:
```
System:
  shared_capabilities:
    multi_host_support: True            # False is used for local storage
    simultaneous_write_support: False   # Are simultaneous writes by multiple hosts supported in the submitted configuration
    simultaneous_read__support: True    # Are simultaneous reads by multiple hosts supported in the submitted configuration
```

# 5. Validating the VDB Workloads

## 5.1. VDB Sizing Options

5.1.1. **vdbDatasetScale** -- The benchmark must be run against one of the defined dataset scales (collection vector counts) listed in the VDB scale table. The *submission validator* must read `num_vectors` and `dimension` from the run's `config.json`/`summary.json` and verify they match a defined scale; any other scale must generate a message and fail validation.

5.1.2. **vdbDimensionConsistency** -- The vector `dimension` recorded at `datagen` (load) time must equal the `dimension` used at `run` (query) time. The *submission validator* must compare the dimension in the load summary against the dimension in each run's `summary.json` and fail validation if they differ.

## 5.2. VDB Generation Options

5.2.1. **vdbCollectionPopulated** -- The number of vectors actually inserted (`inserted_vectors`) during load must equal the declared `num_vectors` for the chosen scale. The *submission validator* must read the load summary and fail validation on a shortfall.

5.2.2. **vdbIndexBuildCompleted** -- The collection must be fully indexed and (when configured) compacted before the query phase. The *submission validator* must confirm an index-build / compaction record is present in the load output and that the index type recorded at load time matches the index type used at run time.

## 5.3. VDB Run Options

5.3.1. **vdbRunCount** -- Within each `vector_database/<index_type>/run/` directory (where `<index_type>` is one of the UPPERCASE tokens `DISKANN`, `HNSW`, or `AISAQ`), there must be exactly five `<datetime>` timestamp directories, each containing a `summary.json`. The count rule applies to query runs only — `datagen` is governed by §5.2. (see §2.1.27 directory diagram.)

5.3.2. **vdbRecallReported** -- Each run's `summary.json` (or its rank-local `recall_stats.json`) must report a recall value computed outside the timed query loop. The *submission validator* must verify a recall field is present and that recall meets or exceeds the minimum recall target defined for the chosen scale/metric.

5.3.3. **vdbQueryCountMinimum** -- Each run must issue at least the minimum number of queries defined for the benchmark (in `query_count` mode via `--queries`, or the equivalent issued-query total in `timed` mode). The *submission validator* must read `throughput_qps` and `total_time_seconds` (or the issued-query count) and fail validation if the minimum is not met.

5.3.4. **vdbMetricsReported** -- Each run's `summary.json` must report `throughput_qps` and the latency percentile set (`mean_latency_ms`, `p95_latency_ms`, `p99_latency_ms`, `p999_latency_ms`). The *submission validator* must verify these fields exist and are populated.

5.3.5. **vdbGroundTruthIntegrity** -- Each run's recall must be measured against a complete, correctly built ground truth: the exact (brute-force) reference used to score recall must cover every vector in the queried dataset. A run whose ground truth failed to build, or that covers fewer than all of the dataset's vectors, does not yield a trustworthy recall measurement and is invalid. The *submission validator* must fail any such run.

## 5.4. VDB Access Via POSIX API Options

5.4.1. **vdbPathArgs** -- For file-API submissions, the vector database data storage path (`--storage-root`, config key `storage.storage_root`) and the output logfile/results directory (`--results-dir`) must both be set and must be different, so that logfiles are not written onto the storage system under test. For object-API submissions the vector database data does not live on a submitter-owned local path, so a storage root is not required and the storage backend is validated under §5.5.1 instead.

5.4.2. **vdbFilesystemCheck** -- The `mlpstorage` command should do a "df" command on the directory pathname where the vector database stores its data and another on the directory pathname where the output logfiles are stored, and record those values in the logfile. The *submission validator* must find those entries in the run's logfile and verify that they are different filesystems, so that logfiles are not accidentally placed on the storage system under test.

## 5.5. VDB Access Via Object API Options

5.5.1. **vdbObjectStorageBackend** -- For object-API submissions, the vector database must be backed by S3-compatible object storage and the submission must record the storage backend in the system description. The *submission validator* must confirm the recorded backend is consistent with the declared API.

## 5.6. VDB OPEN versus CLOSED Options

> **Index type token convention.** The index type is recorded, validated, and
> stored on disk using the uppercase token (`DISKANN`, `HNSW`, `AISAQ`) defined
> by `VDB_INDEX_TYPES_CLOSED` in `mlpstorage_py/config.py`. The same token is
> used by the CLI (`--index-type`), in `summary.json.index_type`, and as the
> index directory name in the §2.1 directory diagram.

5.6.1. **vdbClosedSubmissionChecksum** -- For CLOSED VDB submissions, the *submission validator* enforces the same layered code-image check defined in §3.6.1: self-consistency against `.code-hash.json` always, plus upstream-identity against `REFERENCE_CHECKSUMS` (or `--reference-checksum`) for CLOSED. See §2.1.6 for the `.code-hash.json` schema and exclusion set.

5.6.2. **vdbClosedDatabaseBackend** -- For CLOSED submissions, the vector database backend must be Milvus. The *submission validator* must read the `database.database` field from the run's `config.json`/`summary.json` and fail validation if any backend other than `milvus` is recorded.

5.6.3. **vdbClosedIndexTypes** -- For CLOSED submissions, the index type must be one of exactly three supported types: `DISKANN`, `HNSW`, or `AISAQ` (matching `VDB_INDEX_TYPES_CLOSED`). The *submission validator* must read the `index_type` field and the index directory name under "vector_database" and fail validation if any other index type (e.g. `IVF_FLAT`, `IVF_SQ8`, or `FLAT`) is recorded. Within these three index types, the submitter is free to choose the metric type and any index-specific build and search parameters (see 5.6.4).

5.6.4. **vdbClosedSubmissionParameters** -- For CLOSED submissions of this benchmark, the database backend is fixed to Milvus (see 5.6.2) and the index type is restricted to `DISKANN`, `HNSW`, or `AISAQ` (see 5.6.3), but the submitter may freely choose the metric type and all index-specific build/search parameters for those three index types, plus the load and run parameters listed in the table below. Any other parameter being modified, any unsupported index type, or any attempt to substitute a different database backend must generate a message and fail the validation.

**Table: VectorDB Tunable Parameters for CLOSED (Milvus backend; DISKANN / HNSW / AISAQ only)**

| Parameter                  | CLI flag             | Description                                                      | Default      |
|----------------------------|----------------------|------------------------------------------------------------------|--------------|
| *Database parameters*      |                      |                                                                  |              |
| database.database          | --                   | Backend database engine — **fixed to `milvus` for CLOSED**       | milvus       |
|                            |                      |                                                                  |              |
| *Index selection*          |                      | *(restricted to the three CLOSED index types)*                   |              |
| index.index_type           | `--index-type`       | Index family — **one of: `DISKANN`, `HNSW`, `AISAQ`**            | DISKANN      |
| index.metric_type          | `--metric-type`      | Distance metric (e.g. COSINE, L2, IP)                            | COSINE       |
|                            |                      |                                                                  |              |
| *DISKANN index parameters* |                      |                                                                  |              |
| index.max_degree           | `--max-degree`       | Max graph degree (DiskANN build)                                 | 64           |
| index.search_list_size     | `--search-list-size` | Search list size (DiskANN build)                                 | 200          |
| search.search_ef           | `--search-ef`        | DiskANN search-time list size (recall/throughput trade-off)      | --           |
|                            |                      |                                                                  |              |
| *HNSW index parameters*    |                      |                                                                  |              |
| index.M                    | `--max-degree`       | Max neighbors per node (HNSW build; shares the degree flag)      | 64           |
| index.ef_construction      | `--ef-construction`  | Construction-time candidate list size (HNSW build)               | 200          |
| search.search_ef           | `--search-ef`        | HNSW search-time `ef` (recall/throughput trade-off)              | --           |
|                            |                      |                                                                  |              |
| *AISAQ index parameters*   |                      |                                                                  |              |
| index.max_degree           | `--max-degree`       | Max graph degree (AISAQ build)                                   | 64           |
| index.search_list_size     | `--search-list-size` | Search list size (AISAQ build)                                   | 200          |
| index.inline_pq            | `--inline-pq`        | AISAQ inline product-quantization parameter (perf vs scale)      | 16           |
| search.search_ef           | `--search-ef`        | AISAQ search-time list size (recall/throughput trade-off)        | --           |
|                            |                      |                                                                  |              |
| *Search / run parameters*  |                      |                                                                  |              |
| run.mode                   | `--mode`             | Benchmark mode: `timed` or `query_count`                         | timed        |
| run.num_query_processes    | `--num-query-processes` | Local Python query workers inside each rank                   | --           |
| run.batch_size             | `--batch-size`       | Query batch size                                                 | --           |
| run.report_count           | `--report-count`     | Reporting interval (queries between reports)                     | --           |
|                            |                      |                                                                  |              |
| *Dataset / load parameters*|                      |                                                                  |              |
| dataset.collection_name    | `--collection`       | Name of the collection populated and queried                     | --           |
| dataset.num_shards         | `--num-shards`       | Number of collection shards                                      | --           |
| dataset.chunk_size         | `--chunk-size`       | Vectors per load chunk                                           | --           |
| dataset.batch_size         | `--batch-size`       | Vectors per insert batch at load time                            | 1000         |
| dataset.vector_dtype       | `--vector-dtype`     | Vector data type (e.g. FLOAT_VECTOR)                             | FLOAT_VECTOR |
|                            |                      |                                                                  |              |
| *Storage parameters*       |                      |                                                                  |              |
| storage.storage_root       | `--storage-root`     | The storage root directory for VDB data                          | --           |
| storage.storage_type       | `--storage-type`     | The storage type (e.g. local_fs, s3)                            | local_fs     |

5.6.5. **vdbOpenSubmissionParameters** -- For OPEN submissions of this benchmark, the submitter may additionally run against vector database backends other than Milvus — including **Elasticsearch** and **pgvector** — in addition to everything already permitted in CLOSED. The *submission validator* must verify that the recorded `database.database` is one of the supported backends. OPEN submissions may use any index types, metrics, and parameters native to the chosen backend (including the full `VDB_INDEX_TYPES` set such as `IVF_FLAT`, `IVF_SQ8`, and `FLAT` on Milvus), but must still meet the recall target (5.3.2) and report the required metrics (5.3.4). Any parameter not listed here or in the CLOSED table, when modified, must generate a message and fail the validation.

**Table: VectorDB Additional Tunable Parameters for OPEN**

| Parameter                  | Description                                                                                                          | Default |
|----------------------------|---------------------------------------------------------------------------------------------------------------------|---------|
| *Database parameters*      |                                                                                                                     |         |
| database.database          | Backend database engine — OPEN permits alternative backends including `milvus`, `elasticsearch`, and `pgvector`     | milvus  |
| database.host              | Database endpoint host for the selected backend                                                                     | --      |
| database.port              | Database endpoint port for the selected backend                                                                     | --      |
|                            |                                                                                                                     |         |
| *Extended Milvus indexes*  | *(index types available on Milvus in OPEN beyond the three CLOSED types)*                                            |         |
| index.index_type           | Adds `IVF_FLAT`, `IVF_SQ8`, `FLAT` to the CLOSED `DISKANN` / `HNSW` / `AISAQ` set                                   | --      |
|                            |                                                                                                                     |         |
| *Backend-specific options* | *(any index types, metrics, and parameters native to a non-Milvus backend)*                                          |         |
| index.index_type           | Any index family supported by the chosen backend (e.g. HNSW on Elasticsearch; HNSW / IVFFlat on pgvector)           | --      |
| index.metric_type          | Any distance metric supported by the chosen backend                                                                 | --      |
| index.* (backend-native)   | Any backend-native build/search parameters (e.g. pgvector `lists` / `probes`; Elasticsearch `m` / `ef_construction` / `num_candidates`) | -- |

# 6. Validating the KVCache Options

## 6.1. KVCache Sizing Options

## 6.2. KVCache Generation Options

## 6.3. KVCache Run Options

The KVCache benchmark drives a fixed per-client workload (a fixed number of
simulated users at a fixed per-client I/O concurrency) and lets the submitter
add or remove client processes (MPI ranks) to load the storage system. A
submission is scored, per Option, on the **aggregate Storage Read Bandwidth**
(higher is better), the **aggregate device read- and write-latency P95** (lower
is better), and the **aggregate Storage Throughput in tokens/s**.

A run is launched with `mlpstorage <closed|open|whatif> kvcache run [OPTIONS]`
(the division is the first positional, post-PR #412 modal CLI). The results
directory must first be initialised once with `mlpstorage init <orgname>
<results-dir>`, and every run requires `--systemname` (or the `MLPSTORAGE_SYSTEMNAME`
environment variable). The command executes **all three Options sequentially**,
each repeated `trials` times, by prefixing `mpirun` to `mlperf_wrapper.py`. The
Option's parameters are built by `mlpstorage` (`_build_option_kvcache_args`) and
forwarded through the wrapper to `kv-cache.py`; the wrapper itself encodes no
workload parameters and only writes each rank's results to an isolated
`rank_<N>/` directory.

*Enforcement status: the per-Option / per-rank CLOSED checks in this section are
normative requirements for the submission validator. The kvcache submission
checker is currently a stub, so today these are enforced by the run-time CLI
locks (6.3.2.1) and by manual review until the checker is implemented.*

### 6.3.1. Sanctioned workload Options

6.3.1.1. **kvcacheFixedWorkloadPerOption** -- A CLOSED submission runs the three sanctioned MLPerf v3.0 KVCache workload Options of Table KVCache-1 via `mlpstorage closed kvcache run`. The per-Option parameters `model`, `num-users` (per client), `duration`, `gpu-mem-gb`, `cpu-mem-gb`, `max-concurrent-allocs`, and `generation-mode` are **immutable** and are emitted verbatim from `WORKLOAD_PARAMS` in `mlpstorage_py/benchmarks/kvcache.py`; in CLOSED no user CLI flag can reach `kv-cache.py`. The *submission validator* must fail the run if any recorded value differs from Table KVCache-1.

**Table KVCache-1: MLPerf v3.0 KVCache CLOSED workload Options**

| Option | Name                | model                  | num-users (per client) | duration (s) | gpu-mem-gb | cpu-mem-gb | max-concurrent-allocs | generation-mode |
|--------|---------------------|------------------------|------------------------|--------------|------------|------------|-----------------------|-----------------|
| 1      | Max Storage Stress  | llama3.1-8b            | 200                    | 300          | 0          | 0          | 16                    | none            |
| 2      | Storage Throughput  | llama3.1-8b            | 100                    | 300          | 0          | 4          | 16                    | none            |
| 3      | Large Model (70B)   | llama3.1-70b-instruct  | 70                     | 300          | 0          | 0          | 4                     | none            |

  * `gpu-mem-gb = 0` removes the GPU cache tier so all KV traffic is forced onto the storage tier under test. In Options 1 and 3 `cpu-mem-gb = 0` forces every object to NVMe; Option 2 allows a small (4 GiB) CPU tier.
  * `generation-mode = none` removes the simulated per-token compute delay so 100% of the measured latency is storage I/O.
  * `num-users` is **per client** (per MPI rank); the aggregate offered load is `num_clients × num-users` (see 6.3.3).

### 6.3.2. CLOSED sequence locks

6.3.2.1. **kvcacheClosedSequenceLocks** -- For CLOSED submissions the sequence parameters are fixed and the benchmark hard-fails on any override: `--seed` = 42, `--trials` = 3 (scored repeats per Option), `--inter-option-delay` = 90 s, and `--config` is not permitted. The *submission validator* must confirm these values in the run metadata.

6.3.2.2. **kvcacheAutoscalingProhibited** -- For CLOSED submissions, `--enable-autoscaling` must not be set. (Rationale: the runtime autoscaler does not add or remove worker threads — the worker pool is fixed at run start — so it cannot serve as a fair scaling mechanism.)

### 6.3.3. Client scaling (scaling up and scaling down)

6.3.3.1. **kvcacheClientDefinition** -- A *client* is one MPI process (rank). The cluster rank layout is resolved from `--num-processes` (total ranks across the cluster), `--npernode` (ranks per host), and `--hosts`: if only `--num-processes` is given, `npernode = num-processes / len(hosts)` (must divide evenly); if only `--npernode` is given, `total_ranks = npernode × len(hosts)`; if both are given they must be consistent; if neither, one rank per host. Each client runs one independent `kv-cache.py` instance with the Option's fixed per-client `--num-users`.

6.3.3.2. **kvcacheScaleUpModel** -- Submitters scale the number of clients **up or down** through the `mpirun` rank/host count. Each client runs the **full** per-client `num-users` of the Option, so the aggregate offered load is `num_clients × num-users` — adding clients **increases** total load (and aggregate in-flight concurrency `num_clients × max-concurrent-allocs`). There is no per-client load reduction: a storage system demonstrates scalability by sustaining higher aggregate read bandwidth at an acceptable device P95 as clients are added, until it saturates. The per-client parameters in Table KVCache-1, including `max-concurrent-allocs`, are immutable regardless of client count.

6.3.3.3. **kvcachePerClientIsolation** -- Each client writes its results JSON to, and uses a cache subdirectory under, a path unique to that client: `<results>/.../option_<O>/trial_<T>/rank_<N>/kvcache_results_*.json` and `<cache-dir>/rank_<N>/` (provided by `mlperf_wrapper.py` via its `--rank-output-base`/`--rank-cache-base` arguments). The *submission validator* must fail any multi-client run in which two clients share an output file or cache directory.

6.3.3.4. **kvcacheSharedResultsDir** -- For multi-host runs, `--results-dir` must be on a filesystem visible at the same path on every host in `--hosts`, so the controller can aggregate every rank's result file. `mlpstorage` probes this before running and fails fast (issue #521) if the results directory is not shared.

### 6.3.4. Result aggregation

`mlpstorage` aggregates the per-rank result files for each Option into a single
`kvcache_run_summary.json` (`options[<O>]`), as follows.

6.3.4.1. **kvcacheAggregateBandwidth** -- Within each trial, the per-rank `cache_stats.tier_storage_read_bandwidth_gbps` (resp. write) are **summed** across ranks; across trials, the per-trial sums are reduced by **mean** (`fmean`). The result is `aggregated_read_bandwidth_gbps` / `aggregated_write_bandwidth_gbps`.

6.3.4.2. **kvcacheAggregateThroughput** -- The aggregate **Storage Throughput (tokens/s)** is computed the same way (sum across ranks within a trial, mean across trials) from `summary.storage_throughput_tokens_per_sec` (`aggregated_storage_throughput_tokens_per_sec`).

6.3.4.3. **kvcacheAggregateDeviceLatency** -- The aggregate **device read-latency P95** is the **maximum**, across all ranks and trials, of `cache_stats.storage_read_device_p95_ms` (`aggregated_device_read_p95_ms`); the aggregate **device write-latency P95** is the maximum of `cache_stats.storage_write_device_p95_ms` (`aggregated_device_write_p95_ms`). Two properties must be stated in the result:
  * This is the **max of per-rank P95s** (a conservative, worst-client bound), **not** a pooled population percentile. Each per-rank P95 is itself `np.percentile` over that client's per-read device samples.
  * "Device" latency is the storage-tier I/O span: for reads the `np.load()` span on a page-cache-dropped file, for writes the `fsync` span. It excludes the GPU/CPU tiers, the simulated generation delay, and (per PR #287) the redundant host copy; it is **storage-tier latency**, not raw hardware queue latency, because the read span still includes the one host copy intrinsic to `np.load`.

6.3.4.4. **kvcacheLatencyValidity** -- The device-latency score is meaningful only when storage reads actually occur. A rank that served its working set entirely from the CPU tier (`cache_stats.storage_entries == 0`) contributes `0.0` (the key is absent and defaults to `0.0`) and is recorded in `cpu_tier_ranks`. The *submission validator* must fail (or flag OPEN-only) any Option whose ranks are **all** CPU-tier. This cannot occur for Options 1 and 3 (`cpu-mem-gb = 0`).

6.3.4.5. **kvcacheHeadlineResult** -- The headline result for an Option is *(aggregate Storage Read Bandwidth, aggregate device read-latency P95, aggregate device write-latency P95, aggregate Storage Throughput)* reported with the client configuration `total_ranks` (= `num_processes`, or `npernode × hosts`).

### 6.3.5. Example invocations

`--cache-dir` must be on the storage system under test; `--results-dir` must be
on a **different** filesystem (6.4) and initialised once with `mlpstorage init`.

```
# one-time, per results directory
mlpstorage init <orgname> /results/kv

# CLOSED, 1 client: runs Options 1,2,3 × 3 trials
mlpstorage closed kvcache run --systemname <name> \
  --cache-dir /mnt/nvme/kvcache --results-dir /results/kv

# CLOSED, 4 clients on one host (4 ranks):
mlpstorage closed kvcache run --systemname <name> --npernode 4 \
  --cache-dir /mnt/nvme/kvcache --results-dir /results/kv

# CLOSED, 4 clients across 4 hosts (1 rank each):
mlpstorage closed kvcache run --systemname <name> --hosts node1 node2 node3 node4 \
  --cache-dir /mnt/nvme/kvcache --results-dir /results/kv

# CLOSED, 8 clients via total count (must divide evenly across hosts):
mlpstorage closed kvcache run --systemname <name> --hosts node1 node2 --num-processes 8 \
  --cache-dir /mnt/nvme/kvcache --results-dir /results/kv
```

NFS backing — `--cache-dir` points at the NFS client mount (present on every
client host); the storage-tier metrics are measured at the application's POSIX
boundary over NFS (see 6.4):

```
mlpstorage closed kvcache run --systemname <name> --hosts node1 node2 node3 node4 \
  --cache-dir /mnt/nfs_kv/kvcache --results-dir /results/kv
```

S3 / object backing — `kv-cache.py` has no native object backend, so object
storage is accessed through a POSIX-presenting gateway (s3fs, rclone, or
Mountpoint-S3) and `--cache-dir` points at that mount (see 6.5):

```
mlpstorage closed kvcache run --systemname <name> --hosts node1 node2 node3 node4 \
  --cache-dir /mnt/s3_kv/kvcache --results-dir /results/kv
```

## 6.4. KVCache Access Via POSIX API Options

6.4.1. **kvcachePosixCacheDir** -- `--cache-dir` (the NVMe/storage cache tier) and `--results-dir` (logs and result JSON) must both be set and must resolve to **different** filesystems, so that result/log I/O does not perturb the storage system under test. (This is a submission requirement; unlike Training/VDB, the kvcache checker does not yet auto-verify it — the only kvcache filesystem probe is the multi-host shared-results-dir check of 6.3.3.4.)

6.4.2. **kvcachePosixIoModel** -- Over the POSIX API, `kv-cache.py` stores each KV object as a `.npy` file written with `np.save` + `fsync` and read back with `np.load` after a `POSIX_FADV_DONTNEED` page-cache drop. The storage-tier device latency (6.3.4.3) is therefore the cold-read `np.load` span and the `fsync` span at the POSIX boundary.

## 6.5. KVCache Access Via Object API Options

6.5.1. **kvcacheObjectViaGateway** -- `kv-cache.py` has no native object-storage backend; all storage access is POSIX file I/O against `--cache-dir`. An object-storage (S3) submission must therefore present the bucket as a POSIX filesystem via a gateway/mount (s3fs, rclone, or Mountpoint-S3) and point `--cache-dir` at it. NFS submissions point `--cache-dir` at the NFS client mount.

6.5.2. **kvcacheObjectLatencyScope** -- For object/gateway and NFS backings, the storage-tier device latency and bandwidth are measured at the application's POSIX boundary through the gateway/mount, not at a block device. The optional block-layer device tracer (`--enable-latency-tracing`, OPEN only) cannot observe this I/O — the cache directory's `st_dev` is synthetic — so it falls back to the VFS request sizes / NFS transport chunk. The submission must record the gateway/client software and configuration (e.g. s3fs/rclone version and mount options, NFS `vers`/`rsize`/`wsize`).

## 6.6. KVCache OPEN versus CLOSED Options

6.6.1. **kvcacheClosedImmutable** -- In CLOSED (`mlpstorage closed kvcache run`), only the following may vary: `--systemname`, `--cache-dir`, `--results-dir`, and the client topology (`--hosts`, `--npernode`, `--num-processes`, `--mpi-params`). Everything else is fixed: the three Options of Table KVCache-1 with their immutable per-Option parameters and the sequence locks of 6.3.2.1 (seed 42, trials 3, inter-option-delay 90 s, no `--config`). The *submission validator* must fail a CLOSED run that sets any other parameter.

6.6.2. **kvcacheOpenAllowances** -- OPEN submissions (`mlpstorage open kvcache run`) may, in addition, modify the workload to characterise it more broadly. In OPEN, user CLI flags supersede `WORKLOAD_PARAMS[option]` one key at a time (`max-concurrent-allocs` is not exposed and always comes from `WORKLOAD_PARAMS`). **Caveat:** the supersede cannot tell a user-set value from an argparse default, so a CLI default (`--gpu-mem-gb 16`, `--cpu-mem-gb 32`, `--model tiny-1b`) overrides the Option's value even when the submitter did not set it. An OPEN run that does not explicitly pass `--gpu-mem-gb 0 --cpu-mem-gb 0` (and a model) may keep the whole working set in the GPU/CPU tiers and never touch storage, producing zero storage bandwidth and device latency. OPEN submissions intended to stress storage must set the memory-tier sizes explicitly. OPEN allowances include:
  * a custom `--config <config.yaml>` (e.g. different `user_templates`, `qos_profiles`, `eviction`, or RAG settings);
  * **RAG** retrieval-augmented workloads (`--enable-rag`, `--rag-num-docs`);
  * **BurstGPT** trace-driven request arrivals (`--use-burst-trace`, `--burst-trace-path`);
  * **block-layer latency tracing** (`--enable-latency-tracing`), which captures device latency histograms and emits a distilled `fio` workload for independent replay;
  * changed `--seed`, `--trials`, `--inter-option-delay`, `--num-users`, `--duration`, `--generation-mode`, and the cache-tier sizes.
  * For more options / invocation examples for open submissions,please refer to the Design document:  https://github.com/mlcommons/storage/blob/main/kv_cache_benchmark/DESIGN.md and the proposal document [https://github.com/mlcommons/storage/blob/main/kv_cache_benchmark/docs/MLperf_v3_KV_cache_proposal.md][def]


6.6.3. **kvcacheOpenInvocationNote** -- The `mlpstorage kvcache run` path passes `--config`/`--seed` plus the per-option workload args to `mlperf_wrapper.py`, and consumes `--trials`/`--inter-option-delay` itself (the trial loop and inter-option delay). RAG, BurstGPT, and latency tracing reach `kv-cache.py` only when exposed by the OPEN CLI or via a `--config` file (RAG has a `rag:` config section); otherwise they are exercised by invoking `kv-cache.py` directly. An OPEN submission using these must record the exact invocation. Example (direct, single client, with latency tracing — requires `sudo` for `bpftrace`):

```
sudo python kv-cache.py --config config.yaml \
  --model llama3.1-8b --num-users 200 --duration 300 \
  --gpu-mem-gb 0 --cpu-mem-gb 0 --max-concurrent-allocs 16 \
  --generation-mode none --seed 42 --performance-profile throughput \
  --cache-dir /mnt/nvme/kvcache --enable-latency-tracing \
  --enable-rag --rag-num-docs 10 \
  --output /results/kv_open/run.json
# add --use-burst-trace --burst-trace-path BurstGPT/data/BurstGPT_1.csv for trace-driven arrivals
```


[def]: https://github.com/mlcommons/storage/blob/main/kv_cache_benchmark/docs/MLperf_v3_KV_cache_proposal.md
