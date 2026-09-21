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
per-epoch AU is the mean across all ranks of each rank's AU (the formula in
Rules.md 3.3.2), and the per-epoch throughput in samples/s is the sum across
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
