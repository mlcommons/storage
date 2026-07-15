"""
Checkpointing benchmark run rules checker.

Validates checkpointing benchmark parameters for individual runs.
"""

from typing import Optional

from mlpstorage_py.config import (
    BENCHMARK_TYPES,
    LLM_ALLOWED_VALUES,
    LLM_MODELS,
    LLM_SUBSET_PROCS,
    PARAM_VALIDATION,
)
from mlpstorage_py.rules.issues import Issue
from mlpstorage_py.rules.run_checkers.base import RunRulesChecker


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
        valid_models = list(self.supported_models)

        if model not in valid_models:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid model for checkpointing benchmark",
                parameter="model",
                expected=valid_models,
                actual=model
            )
        return None

    def check_num_processes(self) -> Optional[Issue]:
        """Fail-fast gate on --num-processes against Rules.md 4.6.1 / 4.6.4.

        Per Rules.md §4.6.1 (Table 2), a CLOSED checkpointing submission for a
        given LLM model must use *exactly* one of two process counts:
          * ``LLM_SUBSET_PROCS`` (8) — the subset-run form (§4.3.5); or
          * ``ClosedGPUs`` — the full-run form (8 / 64 / 512 / 1024 for
            llama3-8b / -70b / -405b / -1t).

        OPEN submissions (§4.6.4) may use any positive multiple of
        ``GPUpDP`` (TP*PP).

        Any other value — e.g. the ``51 hosts × 8 ranks = 408`` configuration
        that surfaced in mlcommons/storage#792 — is neither a subset run nor
        a full CLOSED run, and (for 405b) is not a multiple of TP*PP either,
        so it cannot qualify for OPEN. Prior to this check the misconfiguration
        was silently accepted at run time and only surfaced at
        ``mlpstorage validate`` as the misleading "subset run requires exactly
        8 accelerators, got 408" (auto-labeled by ``add_checkpoint_params`` in
        benchmarks/dlio.py setting ``checkpoint.mode="subset"`` for any
        downscaled run). Failing fast here saves the hours of wasted DLIO run
        time between the misconfigured launch and the eventual validate call.

        Silent-passes when model is not one of the four recognized LLMs;
        ``check_model`` above owns that surface.
        """
        model = self.benchmark_run.model
        if model not in LLM_ALLOWED_VALUES:
            return None

        num_processes = self.benchmark_run.num_processes
        _min_procs, _zero_level, gpu_per_dp, closed_gpus = LLM_ALLOWED_VALUES[model]

        if num_processes == closed_gpus or num_processes == LLM_SUBSET_PROCS:
            return None

        if num_processes > 0 and num_processes % gpu_per_dp == 0:
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message=(
                    f"num_processes={num_processes} is not a valid CLOSED "
                    f"configuration for {model}: Rules.md 4.6.1 requires "
                    f"exactly {LLM_SUBSET_PROCS} (subset run) or "
                    f"{closed_gpus} (full run). This value is a multiple of "
                    f"TP*PP ({gpu_per_dp}), so it qualifies for OPEN only — "
                    f"re-run with the 'open' positional argument."
                ),
                parameter="num_processes",
                expected=f"{LLM_SUBSET_PROCS} or {closed_gpus}",
                actual=num_processes,
            )

        return Issue(
            validation=PARAM_VALIDATION.INVALID,
            message=(
                f"num_processes={num_processes} is not a valid checkpointing "
                f"configuration for {model}. Rules.md 4.6.1 requires "
                f"exactly {LLM_SUBSET_PROCS} (subset run per §4.3.5) or "
                f"{closed_gpus} (full CLOSED run) processes; §4.6.4 allows "
                f"any positive multiple of TP*PP ({gpu_per_dp}) for OPEN. "
                f"For a {model} run distributed across N hosts, choose an "
                f"hosts × ranks-per-host product that lands on one of these "
                f"values."
            ),
            parameter="num_processes",
            expected=f"{LLM_SUBSET_PROCS} or {closed_gpus} (CLOSED); "
                     f"positive multiple of {gpu_per_dp} (OPEN)",
            actual=num_processes,
        )
