"""
Training benchmark submission rules checker.

Validates training benchmark submissions (multiple runs).
"""

from typing import Optional

from mlpstorage_py.config import BENCHMARK_TYPES, PARAM_VALIDATION
from mlpstorage_py.editions import current_edition
from mlpstorage_py.rules.issues import Issue
from mlpstorage_py.rules.submission_checkers.base import MultiRunRulesChecker


class TrainingSubmissionRulesChecker(MultiRunRulesChecker):
    """Rules checker for training benchmark submissions."""

    # Rules.md 2.1.11 trainingWorkloads — closed/open submissions accept
    # only the training workloads the current rules edition sanctions
    # (editions.yaml `workloads:`; {unet3d, retinanet} in 3.0).
    supported_models = current_edition().models("training", "closed")
    REQUIRED_RUNS = 5

    def _submission_invocations(self):
        """Return training runs that count as submission invocations.

        Only ``command == 'run'`` invocations are scored runs. Auxiliary
        commands (``datagen``, ``datasize``, ``configview``) emit a results
        directory too, and reportgen's workload pass now verifies every
        group at submission level (Issue #865, ``BenchmarkVerifier(...,
        mode="multi")``) — without this filter a lone ``datagen`` group
        would be marked INVALID for "requires 5 runs", the #717 shape.
        Mirrors ``CheckpointSubmissionRulesChecker._submission_invocations``
        (#791).
        """
        return [
            run for run in self.benchmark_runs
            if run.benchmark_type == BENCHMARK_TYPES.training
            and run.command == 'run'
        ]

    def check_num_runs(self) -> Optional[Issue]:
        """
        Require 5 runs for training benchmark closed submission.

        Counts ``run`` invocations only; a group with none (e.g. a
        ``datagen``-only group) has no run-count rule to apply.
        """
        submission_runs = self._submission_invocations()
        if not submission_runs:
            return None

        num_runs = len(submission_runs)
        if num_runs < self.REQUIRED_RUNS:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Training submission requires {self.REQUIRED_RUNS} runs",
                parameter="num_runs",
                expected=self.REQUIRED_RUNS,
                actual=num_runs
            )

        return Issue(
            validation=PARAM_VALIDATION.CLOSED,
            message=f"Training submission has required {self.REQUIRED_RUNS} runs",
            parameter="num_runs",
            expected=self.REQUIRED_RUNS,
            actual=num_runs
        )
