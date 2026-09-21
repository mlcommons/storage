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
