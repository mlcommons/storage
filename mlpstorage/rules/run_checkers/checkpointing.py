"""
Checkpointing benchmark run rules checker.

Validates checkpointing benchmark parameters for individual runs.
"""

from typing import Optional

from mlpstorage.config import BENCHMARK_TYPES, PARAM_VALIDATION, LLM_MODELS
from mlpstorage.rules.issues import Issue
from mlpstorage.rules.run_checkers.base import RunRulesChecker


class CheckpointingRunRulesChecker(RunRulesChecker):
    """Rules checker for checkpointing benchmarks."""

    supported_models = LLM_MODELS

    def check_benchmark_type(self) -> Optional[Issue]:
        """Verify this is a checkpointing benchmark."""
        if self.benchmark_run.benchmark_type != BENCHMARK_TYPES.checkpointing:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid benchmark type: {self.benchmark_run.benchmark_type}",
                parameter="benchmark_type",
                expected=BENCHMARK_TYPES.checkpointing,
                actual=self.benchmark_run.benchmark_type
            )
        return None

    def check_model(self) -> Optional[Issue]:
        """Verify model is a valid LLM model."""
        model = self.benchmark_run.model
        valid_models = [m.value for m in self.supported_models]

        if model not in valid_models:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid model for checkpointing benchmark",
                parameter="model",
                expected=valid_models,
                actual=model
            )
        return None
