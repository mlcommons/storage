"""
Checkpointing benchmark run rules checker.

Validates checkpointing benchmark parameters for individual runs.
"""

from typing import Optional

from mlpstorage_py.config import (
    BENCHMARK_TYPES,
    LLAMA3_8B,
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

        Per Rules.md §4.6.1 (Table 2), a CLOSED checkpointing submission must
        use *exactly* ``ClosedGPUs`` processes for its model — 8 / 64 / 512 /
        1024 for llama3-8b / -70b / -405b / -1t, respectively.

        OPEN submissions (§4.6.4) may use any positive multiple of
        ``GPUpDP`` (TP*PP).

        History: the original #792 version of this gate also accepted
        ``LLM_SUBSET_PROCS`` (8) as a CLOSED form for *every* model, reading
        §4.3.5 as defining an 8-process "subset run" of the large models.
        That reading traces to a missing "not" in the published §4.3.5 text
        (mlcommons/storage#841): subset mode is defined *only* for the 8B
        model — one 8-GPU node, marking a local-NVMe linear-scale-out claim —
        and the large models have no subset form at all. For llama3-8b the
        full count is 8, so nothing changes there; a large-model run at 8
        processes is now OPEN-eligible when 8 is a TP*PP multiple (70b) and
        INVALID otherwise (405b, 1t).

        Any other value — e.g. the ``51 hosts × 8 ranks = 408`` configuration
        that surfaced in mlcommons/storage#792 — fails fast here, saving the
        hours of wasted DLIO run time between a misconfigured launch and the
        eventual validate call.

        Silent-passes when model is not one of the four recognized LLMs;
        ``check_model`` above owns that surface.
        """
        model = self.benchmark_run.model
        if model not in LLM_ALLOWED_VALUES:
            return None

        num_processes = self.benchmark_run.num_processes
        _min_procs, _zero_level, gpu_per_dp, closed_gpus = LLM_ALLOWED_VALUES[model]

        if num_processes == closed_gpus:
            return None

        if num_processes > 0 and num_processes % gpu_per_dp == 0:
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message=(
                    f"num_processes={num_processes} is not a valid CLOSED "
                    f"configuration for {model}: Rules.md 4.6.1 requires "
                    f"exactly {closed_gpus} processes (no subset form exists "
                    f"for this model — Rules.md 4.3.5, mlcommons/storage#841). "
                    f"This value is a multiple of TP*PP ({gpu_per_dp}), so it "
                    f"qualifies for OPEN only — re-run with the 'open' "
                    f"positional argument."
                ),
                parameter="num_processes",
                expected=closed_gpus,
                actual=num_processes,
            )

        return Issue(
            validation=PARAM_VALIDATION.INVALID,
            message=(
                f"num_processes={num_processes} is not a valid checkpointing "
                f"configuration for {model}. Rules.md 4.6.1 requires exactly "
                f"{closed_gpus} processes for a CLOSED run, and no subset "
                f"form exists for this model (Rules.md 4.3.5, "
                f"mlcommons/storage#841); §4.6.4 allows any positive multiple "
                f"of TP*PP ({gpu_per_dp}) for OPEN. For a {model} run "
                f"distributed across N hosts, choose an hosts × "
                f"ranks-per-host product that lands on one of these values."
            ),
            parameter="num_processes",
            expected=f"{closed_gpus} (CLOSED); "
                     f"positive multiple of {gpu_per_dp} (OPEN)",
            actual=num_processes,
        )

    def check_subset_mode(self) -> Optional[Issue]:
        """Gate the explicit ``--checkpoint-subset`` declaration (Rules.md 4.3.5).

        Subset mode is the 8B single-node claim: a storage solution that
        centrally manages node-local NVMe checkpoints one 8-GPU node and
        claims linear scale-out of the measured bandwidth. It is defined for
        no other model — the larger workloads exist to measure architectures
        where checkpoint data must reach shared central storage (see
        mlcommons/storage#841 for the missing "not" that read as permitting
        large-model subset runs).

        The flag with any model but llama3-8b, or with a process count other
        than ``LLM_SUBSET_PROCS`` (8), is an illegal combination and aborts
        before DLIO launches. Absent the flag this check has no surface:
        downscaled-run classification belongs to ``check_num_processes``.
        """
        if not self.benchmark_run.run_args.get('checkpoint_subset'):
            return None

        model = self.benchmark_run.model
        if model != LLAMA3_8B:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=(
                    f"--checkpoint-subset is not a legal combination with "
                    f"model {model}: Rules.md 4.3.5 defines subset mode only "
                    f"for the 8B model (one 8-GPU node claiming linear "
                    f"scale-out). Run {model} at its full process count "
                    f"instead."
                ),
                parameter="checkpoint_subset",
                expected=LLAMA3_8B,
                actual=model,
            )

        num_processes = self.benchmark_run.num_processes
        if num_processes != LLM_SUBSET_PROCS:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=(
                    f"--checkpoint-subset requires exactly "
                    f"{LLM_SUBSET_PROCS} processes (Rules.md 4.3.5: one "
                    f"8-GPU node), got {num_processes}."
                ),
                parameter="num_processes",
                expected=LLM_SUBSET_PROCS,
                actual=num_processes,
            )

        return None
