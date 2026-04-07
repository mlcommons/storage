"""
Checkpointing benchmark submission rules checker.

Validates checkpointing benchmark submissions (multiple runs).
"""

from typing import Optional, List

from mlpstorage.config import BENCHMARK_TYPES, LLM_MODELS, PARAM_VALIDATION
from mlpstorage.rules.issues import Issue
from mlpstorage.rules.submission_checkers.base import MultiRunRulesChecker


class CheckpointSubmissionRulesChecker(MultiRunRulesChecker):
    """Rules checker for checkpointing benchmark submissions."""

    supported_models = LLM_MODELS
    REQUIRED_WRITES = 10
    REQUIRED_READS = 10

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

        for run in self.benchmark_runs:
            if run.benchmark_type == BENCHMARK_TYPES.checkpointing:
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
