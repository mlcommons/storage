# MLPerf™ Storage Rules Commentary

This file accompanies `Rules.md`.  `Rules.md` is a declarative definition of a
valid submission package: every numbered rule there is a statement that can be
decided by examining the package alone.  This file holds what such a
definition deliberately leaves out, keyed by the same rule numbers:

- **Why** -- the working-group reasoning behind a rule, and the alternatives it
  rejected.
- **How the value is produced** -- how the benchmark (DLIO, `kv-cache.py`,
  the VDB harness) or the `mlpstorage` tool arrives at a value that a rule
  refers to, when knowing that helps a reader interpret the rule.
- **Implementation** -- the code that implements the rule or produces the
  value, and the tests that pin its behaviour.  This is the hook the
  code-vs-rules drift checks key on.

What the `mlpstorage` command does at run time (checks, capture, warnings,
defaults, invocation examples) is documented in `ManPage.md`, not here.

Entries are added as rules are added or as older `Rules.md` text is migrated
out; a rule with no entry here needs none.  An entry never changes what a rule
means: if the commentary and the rule disagree, the rule is authoritative and
the commentary is wrong.

---

## 1.1 mlpstorageGeneratesHierarchy

### Why

The package is defined by its contents, but nearly all of those contents are
produced by one program.  Requiring that every file except the hand-authored
system descriptions be written by `mlpstorage` is what lets the rest of
Rules.md assume consistent naming, timestamps, provenance stamps and code
images: a submitter is not expected to create anything inside the hierarchy
by hand, and a package assembled by running DLIO or the other benchmarks
directly is not a submission even if it happens to look like one.

### How the value is produced

`mlpstorage` resolves the results directory (flag, environment variable, or
the default recorded by `mlpstorage init`) and creates or appends to the
files under it on every invocation; ManPage.md → RESULTS DIRECTORY and ORGNAME
PINNING describe the resolution and the layout it writes.

### Implementation

- `mlpstorage_py/benchmarks/base.py` -- result-directory creation and the
  metadata / provenance / code-image writes every run performs.
- `mlpstorage_py/cli_parser.py` (`_apply_results_dir_resolution`) and
  `mlpstorage_py/config.py` -- the results-dir resolver and sentinel.

---

## 1.3 runResultSubmission

### Why

Before this rule the document used "run" for a *timestamp directory*, for
the set of them under a workload, and for a whole submission, and the tool's
output did the same; a submitter reading "3 of 5 runs" could not tell which
was meant.  Three nouns at three levels remove the ambiguity: a *run* is what
one `mlpstorage ... run` invocation produces and what `mlpstorage runs`
manages by ID; a *result* is what one row of `results.csv` is computed from
(and therefore the unit that can be complete or short); a *submission* is
what one submitter uploads.  The accelerator is part of a result's identity
even though it is not a directory level: 2.1.17 requires exactly six
timestamp directories under one workload's "run" phase, so runs for two
accelerators cannot share a workload directory in a valid package, and
`results.csv` carries one row per accelerator.

The count of runs per complete result is data in the editions table rather
than a number in each rule so that a later edition can change one benchmark's
count (or add a benchmark) without rewording §2, §5 or §6, and so that the
validator's 2.1.17 / 5.3.1 checks and the tool's per-result readiness view
read the same value.  Checkpointing counts *phases*, not directories, because
4.7.1 accepts one combined invocation or two phase invocations as the same
complete result.

### How the value is produced

`mlpstorage` assigns every run a ledger ID when it reserves the leaf
(ManPage.md → `mlpstorage runs`).  The result a run belongs to is derived
from the leaf's path (division, system name, benchmark, workload) and from
the `accelerator` its `*_metadata.json` records.

### Implementation

- `mlpstorage_py/rules/editions.yaml` (`checker.runs_per_result`) and
  `mlpstorage_py/editions.py` -- the per-benchmark count and its validation
  (keys are exactly the edition's workload families; positive integers).
- `mlpstorage_py/submission_checker/checks/directory_checks.py` (2.1.17) and
  `vdb_checks.py` (5.3.1) -- the validator's counts, read through
  `Config.get_runs_per_result`.
- `mlpstorage_py/readiness.py` -- groups a results-dir's runs into results and
  scores each against the count.
- `tests/unit/test_readiness.py` -- pins the table value, the vocabulary and
  the grouping.

---

## 2.1.2 topLevelSubdirectories

### Why

Dot-prefixed entries are tolerated at the top level because merged reviewer
trees are distributed as git working trees, which carry `.git/`,
`.gitignore` and `.github/` alongside the submission proper.  A
per-organization pool root is tolerated because the v3.0 release placed each
organization's code images at `<root>/<orgname>/code-<hash8>/`, and the v3.0
submissions tree is preserved as published rather than rewritten.

### Implementation

- `mlpstorage_py/submission_checker/checks/submission_structure_checks.py`
  (STRUCT-02).

---

## 2.1.6 codeDirectoryContents

### Why

A result is only reproducible if the exact source that produced it travels
with it.  Content-addressing the captured tree by its hash means identical
source captured by different submitters, or by the same submitter across many
runs, coincides in one pool image; the per-leaf pointer preserves the
run-to-image link across that flat layout, so a merged reviewer tree holds one
`code-images/` directory and no duplication.  The per-organization pool of the
v3.0 release is accepted indefinitely for the same reason 2.1.2 tolerates its
root: that tree is frozen as published.

The provenance stamp and the submission manifest make a package
self-describing across rounds -- which rules edition, tool, layout, DLIO
revision and storage library produced each leaf -- so results from different
rounds can be compared by declared class rather than by inference.  Leaves
that predate the stamp are never rewritten; the derived stamp exists so the
frozen v3.0 tree validates exactly as it did before stamping existed.

The core-config hash excludes site tunables so that two runs of the same
workload hash alike wherever they ran; that is what makes it usable as the
comparability-class key.

LEAF-01 exists because nothing but the `*_metadata.json` file identifies a
`kv_cache` or `vector_database` run (training and checkpointing leaves
identify themselves through their DLIO configuration); a leaf without it is
one `reportgen` silently drops, publishing blank metric columns for the
workload.  RPT-01 exists for the same reason: rollup/leaf disagreement is
otherwise absorbed by `reportgen` and invisible at review.  Both make the
failure visible at validation time.  A LEAF-01 finding is repaired by
restoring the file; the measured data need not be regenerated.

EDN-03 is a warning rather than an error so review chairs can extend the
accepted-revision list in `editions.yaml` when a new DLIO commit turns up.
Checkpointing classes of editions 3.0 and earlier use `accelerator: any`
because those rounds recorded no emulated accelerator for checkpoint runs.  A
row whose leaves fall into different classes gets a blank class because such
a row is not comparable with anything.

### How the value is produced

**Tree hash.**  The tree is walked without following symlinks; excluded
directory names are pruned at any depth, `*.egg-info` directories are pruned
by suffix, excluded file names are matched against the basename, and symlinks
are skipped with a warning.  The surviving files are sorted by the UTF-8
bytes of their POSIX relative path and fed to one MD5 in that order, each as
its relative-path bytes followed by its full content.  The capture writes the
copy with the same exclusion predicate, so hashing the copy reproduces the
hash of the source.

**Capture and pool.**  Every `closed`/`open` `datasize`, `datagen` or `run`
hashes the running source tree first; if the pool already holds that hash the
image is reused, otherwise a new `code-<hash8>/` is captured beside the
existing ones (a changed tree is never a rejection -- source iteration is
supported).  A per-organization pool from a v3.0-layout tree is relocated
into `code-images/` the next time the tool captures into that tree.  ManPage.md
→ RESULTS DIRECTORY has the operator's view.

**Provenance stamp.**  Written right after `*_metadata.json`, which declares
it under `"provenance_file"`.  The DLIO git commit comes from the installed
package's PEP 610 `direct_url.json`; the storage library is the client
library the `object` run used.  The core-config hash is SHA-256 over the
canonical JSON of the allowlisted keys, truncated to 16 hex digits; a
`kv_cache` or `vector_database` leaf written before its family's block
existed in `*_metadata.json` is rebuilt at read time from what it recorded
(the wrapper command lines and `summary.json` / `config.json` /
`result_verdict.json`), stamping "unknown" when that is not enough.

**Derived stamp.**  `validate`, `reportgen` and `runs show` derive the stamp
of an unstamped leaf at read time from the pointed-to image's
`.code-hash.json` and `uv.lock` and from the leaf's own metadata; nothing is
written back.

**Submission manifest.**  `mlpstorage reports reportgen` writes
`submission.yaml` for every organization when run against a live results-dir
(one carrying the `mlperf-results.yaml` sentinel); a submissions or archive
tree is never touched.  A manifest that no longer matches the tree (PROV-02)
is refreshed by re-running `reportgen`.

**Editions and classes.**  `validate` reads the `checker` block of the
edition a submission declares and runs only the checks bound to that edition
(`@rule(..., since=, until=)`); a historical edition without a `checker`
block is checked by the tool of its own round.  `mlpstorage runs show` prints
a leaf's class; `reportgen` emits the `Rules Edition` and `Comparability
Class` columns.

### Implementation

- `mlpstorage_py/submission_checker/tools/code_checksum.py`
  (`compute_code_tree_md5`) and `mlpstorage_py/submission_checker/constants.py`
  (`MD5_EXCLUDE_PREFIXES`, `MD5_EXCLUDE_FILENAMES`, `REFERENCE_CHECKSUMS`).
- `mlpstorage_py/submission_checker/tools/code_image.py` -- capture, pool
  lookup and relocation, `.code-hash.json` (`_ALGORITHM`).
- `mlpstorage_py/provenance.py` -- the stamp, the derived stamp and the
  core-config hash; allowlists in `mlpstorage_py/rules/core_config_keys.yaml`.
- `mlpstorage_py/editions.py` / `mlpstorage_py/rules/editions.yaml` --
  editions, `checker` blocks and comparability classes.
- Checks (`mlpstorage_py/submission_checker/checks/`): STRUCT-06 and
  CHECK-01/02/03 (`directory_checks.py`, `pool_structure_checks.py`),
  PROV-01/02 (`provenance_checks.py`), LEAF-01 / RPT-01
  (`results_integrity_checks.py`), EDN-01..04 (`edition_checks.py`).
- Tests: `mlpstorage_py/tests/test_code_checksum.py`,
  `mlpstorage_py/tests/test_capture_or_verify_pool.py`,
  `tests/unit/test_leaf_provenance.py`, `tests/unit/test_rules_editions.py`,
  `tests/unit/test_edition_parameters.py`, `tests/unit/test_edition_values.py`.

---

## 2.1.18 runTimestampGap

### Why

The gap between consecutive *timestamp directories* is bounded so that a
reviewer can be sure no benchmark activity took place between them: the runs
that remain in the package are the runs that were executed back to back.
2.1.18a and 2.1.24a permit deleting all but a consecutive window from a longer
series precisely because this bound distinguishes that from cherry-picking
non-adjacent runs.  2.1.24 (checkpointing) is the same rule for the same
reason.

### Implementation

- `mlpstorage_py/submission_checker/checks/directory_checks.py` -- the
  timestamp-gap checks for 2.1.18 and 2.1.24.
- Tests: `tests/unit/test_directory_check_run_timestamps.py`,
  `tests/unit/test_run_timestamp_gap_none_guard.py`,
  `tests/unit/test_checkpointing_timestamp_gap_sentinel.py`.

---

## 3.1.2 trainingRecalculateDatasetSize

### Why

Two floors size the dataset, and the larger governs.  Five hundred steps per
epoch gives DLIO enough steps for the AU figure (3.3.2) to be a measurement of
the storage system rather than of start-up: the first step is excluded from AU
and a short epoch would be dominated by that exclusion.  Five times the total
client host memory defeats the client page cache: a dataset that fits in RAM
across the hosts is served from memory after the first epoch and the storage
system under test is not what is being measured.

The rule is a floor, not an exact match.  Earlier text said the size recorded
for the run must "exactly match" the recalculated value, but 3.2.1 permits a
larger generated dataset and 3.3.1 permits a run over a subset of it, and the
only thing ever enforced is that the run's file count reaches the minimum; the
rule now says so.  The ceil-minus-one tolerance that the v3.0 round carried
for datasets sized by tool versions older than 3.0.43 (which floored the file
count) was retired when the round closed.

### How the value is produced

`mlpstorage ... training <model> datasize` computes the same two floors from
`--max-accelerators`, `--client-host-memory-in-gb` times `--num-client-hosts`,
and the model's record length and samples per file, rounds up to whole files,
and records the outputs in the datasize leaf (3.3.1).  The validator recomputes
the floors from the run's own records.  `host_memory_GB` in the run's
`*summary.json` is DLIO's per-host list, and the validator sums it: on clusters
with several ranks per host DLIO's per-slot layout can double some slots and
zero others, and the sum is right either way where "hosts times memory per
host" is not.  `num_accelerators` in the summary is the total across hosts.
The validator's steps term is `max(500, steps the dataset yields per epoch)`
times batch size times accelerators; the second operand never exceeds the
dataset's own sample count, so it can never make a dataset fail, and the rule
states the 500 form that decides every case.

### Implementation

- `mlpstorage_py/submission_checker/checks/training_checks.py` --
  `recalculate_dataset_size`; `mlpstorage_py/submission_checker/dlio_summary_helpers.py`
  -- `cluster_total_host_memory_gb`.
- Runtime: `mlpstorage_py/rules/utils.py` -- `calculate_training_data_size`;
  `mlpstorage_py/config.py` -- `STEPS_PER_EPOCH`.
- Tests: `mlpstorage_py/tests/test_rule_3_1_2_exact_ceil_threshold.py`,
  `mlpstorage_py/tests/test_issue_669_host_memory_aggregation.py`.

---

## 3.3.1 trainingRunDataMatchesDatasize

### Why

One generated dataset may serve several run configurations: a sweep generates
once at the largest size and runs smaller configurations against it, so a run
may read fewer files than were generated.  It may never read fewer than the
datasize phase prescribed (that floor is what makes the run representative,
3.1.2) nor more than exist.  The subfolder counts must match the generated
tree because DLIO reconstructs file names from them rather than listing the
directory.  A run is paired with a datasize record by `data_dir` because one
results tree can hold datasize records for several datasets; an ambiguous
pairing is failed rather than guessed.  The manifest snapshot ties the run to
the datagen leaf inside the same package, so the run's data provenance can be
decided without access to the data directory, which the validator never reads.

### How the value is produced

`datasize` writes its inputs and outputs into its leaf's `training_<ts>_metadata.json`.
`datagen` leaves `.mlps-datagen-manifest.json` beside the generated splits;
`run` reads it, fits the run to the dataset, copies it into the run leaf as
`datagen-manifest.json` and names it under `"datagen_manifest_file"`
(ManPage.md → DATA DIRECTORY → "The datagen manifest").  Leaves that declare
neither `datagen_manifest_file` nor `provenance_file` (every leaf of the v3.0
round) are judged on the datasize, datagen and run leaves alone; a stamped leaf
that consumed no manifest draws a warning, not a failure.  A run whose summary
carries no `num_files_eval` draws a warning (models without an eval phase omit
it).

### Implementation

- `mlpstorage_py/submission_checker/checks/training_checks.py` --
  `run_data_matches_datasize` and `_check_manifest_snapshot`.  Finding tokens:
  `DATASIZE-MISSING`, `DATAGEN-MISSING`, `DATASIZE-MALFORMED`,
  `DATASIZE-REUSED`, `DATADIR-MISMATCH`, `DATASIZE-UNDERRUN`,
  `DATAGEN-OVERRUN`, `MANIFEST-MISSING`, `MANIFEST-OVERRIDDEN`,
  `MANIFEST-LINK`, `MANIFEST-COUNT`, `MANIFEST-OVERRUN` (errors);
  `EVAL-FIELD-MISSING`, `MANIFEST-ABSENT` (warnings).
- Runtime: `mlpstorage_py/rules/utils.py` (datasize), the datagen manifest
  writer and the run-time manifest check in `mlpstorage_py/benchmarks/`.
- Tests: `tests/unit/test_submission_checker_run_matches_datasize.py`.

---

## 3.3.2 trainingAcceleratorUtilizationCheck

### Why

A training result is a storage measurement only while the emulated
accelerators are kept busy; a run whose accelerators sit idle waiting for
data is measuring a bottleneck the benchmark is meant to exclude, so the AU
floor is what makes a run's bandwidth figure meaningful.  The minimum lives in
the edition table rather than only in DLIO's workload template because the
template travels with the captured source tree: recording the value per
edition means a template with a lowered threshold cannot pass, and since the
threshold key is in the `training@1` core-config allowlist such a template
also hashes outside every comparability class.

### How the value is produced

DLIO computes, per rank and per epoch:

- `total_compute_time = (records_per_file * total_files) / simulated_accelerators / batch_size * computation_time * epochs`
- `AU = (total_compute_time / total_benchmark_running_time) * 100`

All I/O of the first step is excluded from the AU calculation; the same I/O
is included in the samples-per-second figure.  The per-epoch AU is the mean
across ranks (see §3.3.8 for the reduction), and DLIO writes the mean over the
invocation's epochs as `metric.train_au_mean_percentage` together with its own
verdict, `metric.train_au_meet_expectation`, judged against the `metric.au` of
the workload template it ran.  The validator requires both: the verdict and
the edition's minimum.

### Implementation

- `mlpstorage_py/submission_checker/checks/training_checks.py` --
  `accelerator_utilization_check`; `mlpstorage_py/rules/editions.yaml` --
  `checker.training_au_thresholds`; `Config.get_training_au_threshold`.
- Tests: `tests/unit/test_edition_values.py`.
- Upstream: `dlio_benchmark/utils/statscounter.py`.

---

## 3.3.3 trainingSingleHostSimulatedAccelerators

### Why

The number of simulated accelerators is what loads the storage system.  A
single-host run simulating one or two is rarely representative of what the
system can deliver, so the validator draws the reviewer's attention to it
without failing the run: no rule sets a minimum count.  The threshold of four
is the value the advisory check was written with; the working group has not
set it by rule.

### How the value is produced

Each simulated accelerator is one DLIO rank.  Raising the count on one host
costs host memory (ManPage.md → Training options → `--num-accelerators`).

### Implementation

- `mlpstorage_py/submission_checker/checks/training_checks.py` --
  `single_host_simulated_accelerators` (one warning per distinct count across
  a workload's runs).
- Tests: `mlpstorage_py/tests/test_training_check_retrofit.py`.

---

## 3.3.5 trainingDistributedDataAccessibility

### Why

A distributed run measures the storage system only if every host reads from
it; a host with a local-disk path where the shared mount was expected would
measure its local disk and inflate the result.

### How the value is produced

Before a multi-host run launches its workload, the CAP-02 probe writes a
sentinel into the data directory from rank 0 and stats it from every rank;
differing `(st_dev, st_ino)` pairs fail the run before it starts (ManPage.md →
VALIDATOR → CAP-02).  A run leaf that holds a completed summary therefore
satisfies this rule by construction, and the validator emits one INFO line
per workload so tooling that greps by rule id sees the rule was visited.

### Implementation

- `mlpstorage_py/cluster_collector.py` -- `run_shared_fs_probe`;
  `mlpstorage_py/benchmarks/base.py` -- `_pre_execution_gate`.
- `mlpstorage_py/submission_checker/checks/training_checks.py` --
  `distributed_data_accessibility_check`.
- Tests: `mlpstorage_py/tests/test_training_check_retrofit.py`,
  `tests/unit/test_pre_execution_gate_results_dir.py`.

---

## 3.3.7 trainingNodeCapabilityConsistency

### Why

Client hosts of widely different capability make per-host throughput uneven
and the AU figure harder to interpret, so a reviewer should look at such a
cluster; it is not itself illegal, hence a warning rather than a failure.
The 1.5 ratio is the value the check was written with; the working group has
not set it by rule.

### How the value is produced

`mlpstorage` records the cluster collector's snapshot in each run leaf's
`*_metadata.json` under `cluster_information`: per host, total memory and CPU
core count, plus any `host_consistency_issues` the collector itself flagged.
The validator assesses one run per workload -- the cluster is the same across
the measured runs -- and compares the largest and smallest value of each
metric.

### Implementation

- `mlpstorage_py/submission_checker/checks/training_checks.py` --
  `node_capability_consistency_check`, `_NODE_CAPABILITY_DIVERGENCE_RATIO`.
- `mlpstorage_py/cluster_collector.py` -- the per-host snapshot.
- Tests: `mlpstorage_py/tests/test_training_check_retrofit.py`.

---

## 3.3.8 trainingResultAggregation

### Why

The published training figure is the arithmetic mean of the five measured
invocations, with the first (warm-up) invocation excluded and nothing else
excluded.  The mean is what the v3.0 results table was published with
(reportgen Phase 6, PR #707, 2026-07-07); no minimum, median or geometric mean
is used and no invocation is discarded as an outlier, so one slow invocation
lowers the published figure rather than being dropped.  Requiring every
measured invocation to carry the metric, and failing the workload when one
does not, prevents a partial mean over a subset from being published as if it
covered all five.

### How the value is produced

Within one invocation the benchmark records one value per epoch.  The
per-epoch AU is the mean across all ranks of each rank's AU (the formula under
§3.3.2 above), and the per-epoch throughput in samples/s is the sum across
all ranks (DLIO `statscounter.py`, `end_run`: `allreduce(...) / comm.size`
for AU, `allreduce(...)` for throughput).  DLIO then writes the arithmetic
mean over that invocation's epochs as `metric.train_au_mean_percentage` and,
multiplied by the sample size, `metric.train_io_mean_MB_per_second` in the
invocation's `*summary.json`.  That last field is MiB/s despite its name
(`samples/s × record_size / 1024 / 1024`, logged as "MiB/second"), which is
why the rule divides by 1024 to publish GiB/s.

Across invocations, `Read B/W (GiB/s)` is the mean of the five per-invocation
`train_io_mean_MB_per_second` values ÷ 1024.  Every list-valued metric DLIO
records per epoch (`train_au_percentage`, `train_throughput_samples_per_second`,
…) is reduced the same way -- mean over epochs within an invocation, then mean
over the five invocations -- and published in the run-phase `results.json`
under `train_mean_of_<name>` with the redundant `train_` prefix stripped from
the source key.

The 3.3.2 AU minimum is a per-invocation gate applied by the validator to each
leaf's `train_au_mean_percentage`; the five-invocation mean of AU is
informational.

### Implementation

- `mlpstorage_py/report_generator.py` -- `_aggregate_training` (the
  five-invocation mean; warm-up leaves are excluded by absolute path from the
  set `accumulate_results` records) and the `_INVALID_MSG_EMPTY_METRIC` gate
  in `accumulate_results` that marks the workload INVALID on an empty per-epoch
  list.
- `mlpstorage_py/submission_checker/checks/training_checks.py` -- the 3.3.2
  per-leaf AU check.
- Tests: `tests/unit/test_aggregation.py` (`TestTrainingAggregation`,
  `TestTrainingFinalTableColumns`).
- Upstream: `dlio_benchmark/utils/statscounter.py` (`end_run`,
  `compute_metrics_train`).

---

## 3.4.2 trainingMlpstorageFilesystemCheck

### Why

Logfiles written onto the storage system under test would add I/O to the
measurement and skew the result; requiring the two directories to be on
different filesystems, and requiring the package to carry the evidence, lets a
reviewer decide it without access to the hosts.  4.4.2 and 5.4.2 are the same
rule for checkpointing and VectorDB.

### How the value is produced

Before the workload launches, the CAP-03 probe on rank 0 hard-links a sentinel
from the data (or checkpoint) directory into the results directory: `EXDEV`
is the unambiguous "different filesystems" answer, so no `df` output is
parsed.  The structured result is written to `<run-leaf>/fs_separation.json`
whatever the outcome; a same-filesystem result fails the run before it starts
unless `--skip-fs-separation-gate` was given for a development run, in which
case the sidecar still records it and validation fails the leaf (ManPage.md →
VALIDATOR → CAP-03, and Common artifacts).  Leaves written before the sidecar
existed carry instead the `df` output the tool logged for both paths in
`training_run.stdout.log`; the validator matches each path against the
longest mount-point prefix in that listing.  Object-API runs have no
filesystem to compare, hence the exemption.

### Implementation

- `mlpstorage_py/benchmarks/fs_separation_probe.py` -- the probe and sidecar.
- `mlpstorage_py/submission_checker/checks/training_checks.py` --
  `mlpstorage_filesystem_check`; `checks/helpers.py` --
  `read_fs_separation_sidecar`, `_check_filesystem_separation`.
- Tests: `mlpstorage_py/tests/test_training_check_phase2.py`,
  `mlpstorage_py/tests/test_issue601_validator_reads_sidecar.py`,
  `mlpstorage_py/tests/test_issue601_fs_separation_probe.py`.

---

## 3.6.1 trainingClosedSubmissionChecksum

### Why

CLOSED results are comparable only if every submitter ran the same code.
Two layers establish that: the §2.1.6 self-consistency requirement (the
recomputed tree hash equals the recorded one) proves the captured image was
not altered after capture, and this rule proves the captured image *is* the
sanctioned release.  The first layer applies in every division; the second
only in CLOSED, where OPEN submitters are free to modify the code and
disclose it through the same code-image mechanism.

### How the value is produced

The reference digest is the tree hash of the release's source tree computed
with the §2.1.6 exclusions (test trees, caches and editor metadata are
excluded so the digest is stable across checkouts of the same release); the
same tool computes it:
`python -m mlpstorage_py.submission_checker.tools.compute_code_checksum <path>`.

### Implementation

- The self-consistency layer is live: CHECK-02 / STRUCT-06
  (`mlpstorage_py/submission_checker/checks/pool_structure_checks.py`,
  `submission_structure_checks.py`).
- The reference-digest comparison is not yet performed.  `REFERENCE_CHECKSUMS`
  in `mlpstorage_py/submission_checker/constants.py` carries no digest for any
  edition, the 3.6.1 and 5.6.1 check bodies return without a finding, and the
  `--reference-checksum` option of `mlpstorage validate` is parsed but not
  consulted.  The natural home for a published digest is the edition's
  `checker` block in `mlpstorage_py/rules/editions.yaml`, next to the other
  per-edition values.
- Tests: `mlpstorage_py/tests/test_config_reference_checksum.py` pins the
  empty table; `mlpstorage_py/tests/test_code_checksum.py` the hash.

---

## 4.3.1 checkpointDataSizeRatio

### Why

The read phase must be served by the storage system under test, not by the
client hosts' page cache.  When each host writes more than three times its
own memory during the write phase, the cache cannot hold the checkpoints and
the read phase necessarily reaches storage.  Below that ratio the read phase
may still be honest -- the submitter clears the cache in the failover callout
(4.7.1) -- so the rule is advisory: the validator draws the reviewer's
attention to the ratio without failing the run.  The 3x figure is the value
the advisory check was written with; the working group has not set it by
rule.

### How the value is produced

`metric.checkpoint_size_GB` is the checkpoint size summed across all ranks
(see §4.3.6, "How the value is produced"), so dividing by `num_hosts` gives
the bytes each host wrote.  `host_memory_GB` in the summary is DLIO's
per-host list; on clusters with several ranks per host the list is
positionally malformed, so only its sum divided by `num_hosts` is used.

### Implementation

- `mlpstorage_py/submission_checker/checks/checkpointing_checks.py` --
  `checkpoint_data_size_ratio` (warning; one per distinct size / memory /
  host-count condition, so a split write-read pair warns once).
- `mlpstorage_py/submission_checker/checks/helpers.py` --
  `per_host_memory_gb`.

---

## 4.3.4 checkpointAggregateAcceleratorMemory

### Why

The benchmark checkpoints a model that is sharded across the simulated
accelerators; the shards must fit in their memory or the run does not model
a real checkpoint of that model.  The rule is checked against the checkpoint
size the run actually wrote, so a run of the right model at too few
accelerators fails on its own evidence.

The Table 2 "Checkpoint size" row is in binary units although labeled
GB/TB, and its values are the ones the v3.0 round validated against.  They
are deliberately retained for later editions so results remain comparable
with v3.0; the 1T figure in particular is known to be a rounded, slightly
high value.

### How the value is produced

The accelerator is declared with `--accelerator-type` on `checkpointing run`
(ManPage.md → Checkpointing options) and recorded in the run's metadata as
`accelerator`, with `args.accelerator_type` as the older location.  Each
accelerator's memory, and each model's checkpoint size, is a per-edition
value in the `checker` block of `mlpstorage_py/rules/editions.yaml`; the
tool applies the same product test before launching a run, against the
Table 2 size, so a misconfigured run fails before DLIO starts.  Leaves
written before the flag existed carry no accelerator and cannot be verified;
the validator fails them rather than assuming one.

### Implementation

- `mlpstorage_py/submission_checker/checks/checkpointing_checks.py` --
  `aggregate_accelerator_memory`.
- `mlpstorage_py/rules/run_checkers/checkpointing.py` --
  `check_accelerator_memory` (the pre-launch gate).
- `mlpstorage_py/rules/editions.yaml` -- `checker.accelerator_memory_gb`,
  `checker.checkpoint_size_gb`.
- Tests: `tests/unit/test_rules_checkers.py`, `tests/unit/test_cli_parser.py`,
  `tests/unit/test_edition_values.py`.

---

## 4.3.5 checkpointSubsetRunValidation

### Why

Subset mode exists for storage architectures that centrally manage storage
local to the client nodes, whose aggregate checkpoint bandwidth therefore
scales linearly with node count.  One 8-GPU node running the 8B workload
demonstrates such an architecture's per-node bandwidth; the larger models
measure storage where checkpoint data must reach a shared central store, so
no subset form is defined for them.  An earlier wording of the rule omitted
the word "not" and read as permitting subset runs of the large models; the
rule now states the only legal form directly.

### How the value is produced

A submitter declares the claim with `--checkpoint-subset` on `checkpointing
run` (ManPage.md → Checkpointing options); the tool refuses the flag with any
other model or process count, and records it in the run's metadata under
`args.checkpoint_subset`.  Independently, the tool sets DLIO's
`checkpoint.mode` to "subset" for any run whose process count is below the
model's full count (Table 2), which engages DLIO's partial-checkpoint
mechanics; that override lands in `override_parameters`.  Either signal makes
a CLOSED run a subset run under this rule, because the 8B claim run is
execution-identical to a full 8B run and the flag is its only trace, while a
downscaled run of a larger model carries the override and no flag.

### Implementation

- `mlpstorage_py/submission_checker/checks/checkpointing_checks.py` --
  `subset_run_validation` (CLOSED only; an OPEN run below the full count is
  governed by 4.6.4).
- `mlpstorage_py/rules/run_checkers/checkpointing.py` --
  `check_subset_mode` (the pre-launch gate on the flag).
- `mlpstorage_py/benchmarks/dlio.py` -- `add_checkpoint_params` (the
  `checkpoint.mode` override).
- Tests: `tests/unit/test_checkpoint_capacity_gate_subset.py`.

---

## 4.3.6 checkpointResultAggregation

### Why

A checkpointing workload has one or two invocations (Rules.md 2.1.23 and
4.7.1): a single combined write-then-read invocation, or a write-phase
invocation followed by a read-phase invocation with the submitter's failover
callout between them.  The rule selects each column's source by what an
invocation was configured to do (`checkpoint.num_checkpoints_write` /
`num_checkpoints_read`) rather than by position, so both topologies are
covered by one definition: a combined invocation supplies all four columns, a
split supplies the write columns from the first invocation and the read
columns from the second.  There is no warm-up invocation for checkpointing;
the write phase self-warms the read phase.

The published bandwidth is the **mean of the ten per-checkpoint rates**, not
the total bytes written divided by the total elapsed time.  The two differ
whenever checkpoint durations vary (the aggregate rate weights long
checkpoints more heavily).  The mean of rates is what DLIO reports and what
the v3.0 table was published with; it is retained for comparability.

### How the value is produced

For each of the ten checkpoints written (respectively read) in an invocation,
DLIO records the checkpoint's wall-clock duration in seconds and its rate
`checkpoint_size / duration` (DLIO `statscounter.py`, `end_save_ckpt` /
`end_load_ckpt`).  `checkpoint_size` is the checkpoint size summed across all
ranks via `allreduce` (`base_checkpointing.py`), in binary GiB, so each rate
is the aggregate bandwidth across all processes in GiB/s -- and the Table 2
"Checkpoint size" row is in the same binary units despite its GB/TB label.
At the end of the run DLIO writes the arithmetic mean over the ten as
`metric.save_checkpoint_io_mean_GB_per_second` and
`metric.save_checkpoint_duration_mean_seconds` (respectively `load_*`) in the
invocation's `*summary.json`.  The rule's columns are those fields, averaged
over the invocations of the matching phase; with one invocation per phase,
which is every CLOSED submission, the column is the field itself.

A leaf that predates the recording of `num_checkpoints_{write,read}` in its
parameters cannot be classified by phase; reportgen then takes a mean over
whichever invocations carry the field, rather than blanking rows from older
trees.

### Implementation

- `mlpstorage_py/report_generator.py` -- `_aggregate_checkpointing`, in
  particular `_directional_mean` (phase classification from the recorded
  counts, strict blank on a missing field within the producing phase, the
  present-only fallback for unclassifiable legacy leaves).
- `mlpstorage_py/submission_checker/checks/helpers.py` --
  `_pair_checkpoint_runs`, the write/read pairing used by 4.7.1 / 4.7.2.
- Tests: `tests/unit/test_aggregation.py` (`TestCheckpointingAggregation`,
  `TestCheckpointingFinalTableColumns`, including the split-invocation and
  two-combined-invocation cases).
- Upstream: `dlio_benchmark/utils/statscounter.py` (`end_run`,
  `end_save_ckpt`, `end_load_ckpt`),
  `dlio_benchmark/checkpointing/base_checkpointing.py` (`checkpoint_size`).

---

## 4.4.2 checkpointFilesystemCheck

### Why

The same rule as 3.4.2 for checkpointing: output logfiles written onto the
storage system under test would add I/O to the measurement and skew the
result.  See §3.4.2 for the reasoning and the probe.

### How the value is produced

As for 3.4.2, with the checkpoint directory (`args.checkpoint_folder`) in
the role of the data directory and `checkpointing_run.stdout.log` carrying
the pre-sidecar `df` listing.

### Implementation

- `mlpstorage_py/submission_checker/checks/checkpointing_checks.py` --
  `checkpoint_filesystem_check`; `checks/helpers.py` --
  `read_fs_separation_sidecar`, `_check_filesystem_separation`.
- Tests: `mlpstorage_py/tests/test_issue601_validator_reads_sidecar.py`.

---

## 4.7.1 checkpointCacheFlushValidation

### Why

Checkpointing models the failure of a client node followed by another client
picking up the last checkpoint file written by the failed node for the read
phase.  When the storage system supports the client-to-client handoff
transparently -- the read phase can proceed immediately after the write phase
without external orchestration -- the two phases may run as a single combined
invocation, and no gap applies.  Storage architectures that need an external
callout (a submitter-provided script, say) to complete the failover between
the writing and reading clients run the phases as two invocations with the
callout between them.  A common in-callout activity is clearing a client-side
filesystem cache when the checkpoint data written per host is less than three
times the host's memory (4.3.1); the callout is not limited to that.

The 30-second bound keeps the callout a lightweight programmatic step rather
than a long-running manual procedure, without charging the submitter for
per-invocation framework overhead they cannot avoid: the gap is measured
between the moment the write invocation released its nodes and the moment
the read invocation's process started, not between the two timed sections.
A negative gap cannot be a real ordering violation (the structural part of
the rule already requires the read phase to start after the write phase
ends); it means the two invocations' hosts disagree about the time, which a
reviewer should see but which does not invalidate the run.

The structural part of the rule (one combined or exactly two split
invocations, 10 and 10) is what makes the published write and read figures
(4.3.6) comparable across submissions.  OPEN submissions may vary the counts
(Table 4), so only the gap applies to them.

### How the value is produced

`invocation_start_time` is captured at the first import of `mlpstorage`'s
entry module, before Python imports, MPI spawn and the pre-execution gates,
and written to the metadata of every run.  `invocation_end_time` is captured
at the top of the metadata write, after the post-benchmark cluster collection
has returned -- the latest point at which the value can still land in the
file and the closest to the moment the write nodes are released.  Leaves
written before either field existed fall back as the rule states: the latest
post-benchmark `collection_timestamp` is the moment the last node finished
the final cluster collection, a close proxy for node release; the summary
`end_time` and `start_time` bound the timed section and charge framework
startup or teardown to the gap.  The validator logs every pair's gap and
which origins it used, whatever the verdict.

### Implementation

- `mlpstorage_py/_invocation.py` -- the two bookends.
- `mlpstorage_py/benchmarks/base.py` -- `write_metadata` (records both
  fields).
- `mlpstorage_py/submission_checker/checks/checkpointing_checks.py` --
  `cache_flush_validation` (the gap) and `checkpoint_invocation_structure`
  (the one-or-two-invocation shape, CLOSED only); `checks/helpers.py` --
  `_pair_checkpoint_runs`, `_latest_final_collection_timestamp`.
- Tests: `tests/unit/test_aggregation.py` (`§4.7.1` cases),
  `mlpstorage_py/tests/test_checkpointing_check_phase2.py`.

---

## 4.7.3 checkpointRemappingTimeReporting

### Why

A solution that cannot serve a checkpoint to a second host the moment the
first host finishes writing it has a remapping delay, and that delay is part
of the recovery time the benchmark models (4.7.2).  The rule asks for the
figure in the system description and checks it two ways: for consistency
with the simultaneous-access declarations (a solution supporting
simultaneous reads and writes by multiple hosts has nothing to remap), and
against the interval the submission actually shows between the write and
read invocations, which cannot honestly be much shorter than the declared
delay.  The one-half tolerance is the value the check was written with; the
working group has not set it by rule.

### How the value is produced

The consistency test is part of the system-description schema, so it is
reported when the description is loaded; the interval test uses the same
write-end / read-start summary timestamps as 4.7.2.

### Implementation

- `mlpstorage_py/submission_checker/checks/system_yaml_schema_checks.py` --
  the `capabilities` cross-field validator (consistency).
- `mlpstorage_py/submission_checker/checks/checkpointing_checks.py` --
  `remapping_time_reporting` (the interval test).
