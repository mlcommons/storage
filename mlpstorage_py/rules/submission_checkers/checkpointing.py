"""
Checkpointing benchmark submission rules checker.

Validates checkpointing benchmark submissions (multiple runs).
"""

from datetime import datetime
from typing import Optional, List

from mlpstorage_py.config import BENCHMARK_TYPES, LLM_MODELS, PARAM_VALIDATION
from mlpstorage_py.rules.issues import Issue
from mlpstorage_py.rules.submission_checkers.base import MultiRunRulesChecker


# Maximum allowed pause between the write-phase end and the read-phase start
# in a two-invocation CLOSED submission (Rules.md §4.7.1).
MAX_INTER_PHASE_GAP_SECONDS = 30


def _parse_summary_timestamp(value):
    """Parse a DLIO summary 'start'/'end' timestamp into a datetime, or None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        try:
            return datetime.strptime(value, "%Y%m%d_%H%M%S")
        except (TypeError, ValueError):
            return None


class CheckpointSubmissionRulesChecker(MultiRunRulesChecker):
    """Rules checker for checkpointing benchmark submissions."""

    supported_models = LLM_MODELS
    REQUIRED_WRITES = 10
    REQUIRED_READS = 10

    def _submission_invocations(self):
        """Return checkpointing runs that count as submission invocations.

        Issue #791: only ``command == 'run'`` invocations perform checkpoint
        reads/writes and count toward the Rules.md §4.7.1 invocation
        structure. Auxiliary commands like ``datasize`` are preflight
        helpers — they emit a results directory but carry the CLI defaults
        for ``--num-checkpoints-read``/``--num-checkpoints-write`` (both
        10) in their parameters, which would otherwise double-count into
        ``check_num_runs`` and inflate the invocation count past the
        1-or-2 bound.
        """
        return [
            run for run in self.benchmark_runs
            if run.benchmark_type == BENCHMARK_TYPES.checkpointing
            and run.command == 'run'
        ]

    def check_num_runs(self) -> List[Issue]:
        """
        Require 10 total writes and 10 total reads for checkpointing benchmarks.

        It's possible for a submitter to have:
        - A single run with all checkpoints
        - Two runs that separate reads and writes
        - Individual runs for each read and write operation
        """
        issues = []
        num_writes = num_reads = 0

        submission_runs = self._submission_invocations()
        if not submission_runs:
            return issues

        for run in submission_runs:
            checkpoint_params = run.parameters.get('checkpoint', {})
            num_writes += checkpoint_params.get('num_checkpoints_write', 0)
            num_reads += checkpoint_params.get('num_checkpoints_read', 0)

        # Check reads
        if num_reads != self.REQUIRED_READS:
            issues.append(Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Expected {self.REQUIRED_READS} total read operations, but found {num_reads}",
                parameter="checkpoint.num_checkpoints_read",
                expected=self.REQUIRED_READS,
                actual=num_reads
            ))
        else:
            issues.append(Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=f"Found expected {self.REQUIRED_READS} total read operations",
                parameter="checkpoint.num_checkpoints_read",
                expected=self.REQUIRED_READS,
                actual=num_reads
            ))

        # Check writes
        if num_writes != self.REQUIRED_WRITES:
            issues.append(Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Expected {self.REQUIRED_WRITES} total write operations, but found {num_writes}",
                parameter="checkpoint.num_checkpoints_write",
                expected=self.REQUIRED_WRITES,
                actual=num_writes
            ))
        else:
            issues.append(Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=f"Found expected {self.REQUIRED_WRITES} total write operations",
                parameter="checkpoint.num_checkpoints_write",
                expected=self.REQUIRED_WRITES,
                actual=num_writes
            ))

        # Combined check
        if num_writes == self.REQUIRED_WRITES and num_reads == self.REQUIRED_READS:
            issues.append(Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=f"Found expected {self.REQUIRED_READS} total read and write operations",
                parameter="checkpoint.num_checkpoints_read",
                expected=self.REQUIRED_READS,
                actual=self.REQUIRED_READS,
            ))

        return issues

    def check_invocation_structure(self) -> List[Issue]:
        """Enforce the CLOSED-mode invocation pattern from Rules.md §4.7.1.

        A CLOSED checkpointing submission must consist of either:
          - a single invocation with --num-checkpoints-write=10 and
            --num-checkpoints-read=10; or
          - two invocations: the first with 10 writes / 0 reads, followed
            within MAX_INTER_PHASE_GAP_SECONDS by a second invocation with
            0 writes / 10 reads (the gap covers the cache flush).

        Any other arrangement (e.g. ten 1-write runs or overlapping phases)
        is INVALID for CLOSED. Special-build relaxation: a >30s gap between
        the write and read phases is reported as a warning-severity issue
        rather than INVALID.
        """
        issues = []

        checkpoint_runs = self._submission_invocations()
        if not checkpoint_runs:
            return issues

        # Only enforce the structural pattern for CLOSED submissions; OPEN
        # may freely choose any non-negative read/write counts.
        categories = {run.category for run in checkpoint_runs}
        if categories != {PARAM_VALIDATION.CLOSED}:
            return issues

        def _wr(run):
            params = run.parameters.get('checkpoint', {})
            return (
                params.get('num_checkpoints_write', 0),
                params.get('num_checkpoints_read', 0),
            )

        if len(checkpoint_runs) == 1:
            writes, reads = _wr(checkpoint_runs[0])
            if writes == self.REQUIRED_WRITES and reads == self.REQUIRED_READS:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.CLOSED,
                    message=(
                        "Single-invocation CLOSED submission with "
                        f"{self.REQUIRED_WRITES} writes and {self.REQUIRED_READS} reads"
                    ),
                    parameter="checkpoint.invocation_structure",
                    expected="single run with 10 writes and 10 reads",
                    actual=f"writes={writes}, reads={reads}",
                ))
            else:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=(
                        "Single-invocation CLOSED submission must use 10 writes "
                        f"and 10 reads (got writes={writes}, reads={reads})."
                    ),
                    parameter="checkpoint.invocation_structure",
                    expected="single run with 10 writes and 10 reads",
                    actual=f"writes={writes}, reads={reads}",
                ))
            return issues

        if len(checkpoint_runs) == 2:
            ordered = sorted(
                checkpoint_runs,
                key=lambda r: _parse_summary_timestamp(r.run_datetime) or datetime.min,
            )
            first_w, first_r = _wr(ordered[0])
            second_w, second_r = _wr(ordered[1])

            if (first_w, first_r) != (self.REQUIRED_WRITES, 0) or \
               (second_w, second_r) != (0, self.REQUIRED_READS):
                issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=(
                        "Two-invocation CLOSED submission must run the write phase "
                        "(writes=10, reads=0) followed by the read phase "
                        f"(writes=0, reads=10). Got first=(writes={first_w}, reads={first_r}), "
                        f"second=(writes={second_w}, reads={second_r})."
                    ),
                    parameter="checkpoint.invocation_structure",
                    expected="run 1: writes=10/reads=0; run 2: writes=0/reads=10",
                    actual=(
                        f"run 1: writes={first_w}/reads={first_r}; "
                        f"run 2: writes={second_w}/reads={second_r}"
                    ),
                ))
                return issues

            # Prefer the mlpstorage invocation bookends
            # (metadata.invocation_end_time / invocation_start_time). These
            # exclude write-side post-benchmark cluster collection and
            # read-side framework startup, and are the authoritative origins
            # for the §4.7.1 cache-flush gap. This mirrors the origin selection
            # in the submission checker's 2.1.24 / 4.7.1 checks (storage#714 /
            # #782); measuring from the summary fields charges that
            # startup/collection overhead against the quiet window and produces
            # spurious >30s violations.
            #
            # Write side, in priority order (matches
            # checkpointing_checks.cache_flush_validation):
            #   1. invocation_end_time — #787+ bookend (excludes teardown).
            #   2. final_collection_time — latest post-benchmark
            #      collection_timestamp; node-release proxy for pre-#787
            #      results dirs (which are eligible for v3.0 without
            #      regeneration and carry no invocation_end_time).
            #   3. summary end — oldest fallback (charges teardown to the gap).
            write_end = (
                _parse_summary_timestamp(ordered[0].invocation_end_time)
                or _parse_summary_timestamp(ordered[0].final_collection_time)
                or _parse_summary_timestamp(ordered[0].end_datetime)
            )
            read_start = (
                _parse_summary_timestamp(ordered[1].invocation_start_time)
                or _parse_summary_timestamp(ordered[1].run_datetime)
            )
            if write_end is None or read_start is None:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=(
                        "Two-invocation CLOSED submission is missing parseable "
                        "start/end timestamps for the write or read phase, so the "
                        "inter-phase cache-flush gap (≤30s) cannot be verified."
                    ),
                    parameter="checkpoint.invocation_structure",
                    expected="parseable write-phase end and read-phase start",
                    actual=(
                        f"write_end={(ordered[0].invocation_end_time or ordered[0].final_collection_time or ordered[0].end_datetime)!r}, "
                        f"read_start={(ordered[1].invocation_start_time or ordered[1].run_datetime)!r}"
                    ),
                ))
                return issues

            gap_seconds = (read_start - write_end).total_seconds()
            if gap_seconds < 0:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=(
                        f"Read phase started {-gap_seconds:.1f}s before the write "
                        "phase ended; the two phases must not overlap."
                    ),
                    parameter="checkpoint.invocation_structure",
                    expected="read-phase start after write-phase end",
                    actual=f"gap={gap_seconds:.1f}s",
                ))
                return issues

            if gap_seconds > MAX_INTER_PHASE_GAP_SECONDS:
                # Special-build relaxation: a gap breach is reported as a
                # warning-severity issue rather than INVALID, so it never
                # invalidates the submission.
                issues.append(Issue(
                    validation=PARAM_VALIDATION.CLOSED,
                    message=(
                        f"Gap between write-phase end and read-phase start is "
                        f"{gap_seconds:.1f}s, exceeding the {MAX_INTER_PHASE_GAP_SECONDS}s "
                        "maximum required by Rules.md §4.7.1 (accepted with a "
                        "warning in this build)."
                    ),
                    parameter="checkpoint.invocation_structure",
                    expected=f"≤ {MAX_INTER_PHASE_GAP_SECONDS}s",
                    actual=f"{gap_seconds:.1f}s",
                    severity="warning",
                ))
                return issues

            issues.append(Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=(
                    "Two-invocation CLOSED submission: write phase (10/0) followed by "
                    f"read phase (0/10) with a {gap_seconds:.1f}s inter-phase gap."
                ),
                parameter="checkpoint.invocation_structure",
                expected="write 10/0 then read 0/10 within 30s",
                actual=f"gap={gap_seconds:.1f}s",
            ))
            return issues

        # 0 runs handled above; here len(checkpoint_runs) >= 3 (or == 0 already returned)
        issues.append(Issue(
            validation=PARAM_VALIDATION.INVALID,
            message=(
                f"CLOSED checkpointing submission must consist of 1 or 2 invocations "
                f"(got {len(checkpoint_runs)}). See Rules.md §4.7.1."
            ),
            parameter="checkpoint.invocation_structure",
            expected="1 or 2 invocations",
            actual=str(len(checkpoint_runs)),
        ))
        return issues
