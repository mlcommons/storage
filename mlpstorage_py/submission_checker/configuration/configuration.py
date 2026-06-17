import os
import yaml

from ..constants import *


class Config:
    def __init__(self, version, submitters, skip_output_file=False, reference_checksum_override=None):
        self.version = version
        self.submitters = submitters
        self.skip_output_file = skip_output_file
        self.reference_checksum_override = reference_checksum_override
        self._parallelism_cache: dict[str, tuple[int, int]] = {}  # lazy-load cache for get_model_parallelism
        
    def check_submitter(self, submitter):
        if self.submitters is None:
            return True
        return submitter in self.submitters
    
    def get_datagen_required_files(self):
        return DATAGEN_REQUIRED_FILES[self.version]
    
    def get_run_required_files(self):
        return RUN_REQUIRED_FILES[self.version]
    
    def get_checkpoint_required_files(self):
        return CHECKPOINT_REQUIRED_FILES[self.version]
    
    def get_datagen_required_folders(self):
        return DATAGEN_REQUIRED_FOLDERS[self.version]
    
    def get_run_required_folders(self):
        return RUN_REQUIRED_FOLDERS[self.version]
    
    def get_checkpoint_required_folders(self):
        return CHECKPOINT_REQUIRED_FOLDERS[self.version]
    
    def get_num_train_files(self, model):
        # .get returns None for unknown model names — the caller decides
        # whether the per-model lookup is critical (skip with a diagnostic)
        # or expected to always resolve. Non-conforming workload directory
        # names (e.g. "unet3d_a100", "cosmoflow-20N-6PPN-A100") flagged by
        # 2.1.11 trainingWorkloads would otherwise crash this dict lookup.
        return NUM_DATASET_TRAIN_FILES.get(model)

    def get_num_eval_files(self, model):
        # See get_num_train_files: .get over [] for None-on-miss semantics.
        return NUM_DATASET_EVAL_FILES.get(model)

    def get_checkpoint_file(self, model):
        # See get_num_train_files: .get over [] for None-on-miss semantics.
        return CHECKPOINT_FILE_MAP.get(model)

    def get_reference_checksum(self, cli_override=None):
        """Resolve the reference MD5 for the current version.

        Precedence: cli_override > self.reference_checksum_override >
        REFERENCE_CHECKSUMS[self.version] > None.

        ``None`` means "not pinned" — the caller emits a warning via
        ``warn_violation`` and treats the check as passing (D-12).

        Args:
            cli_override: Optional hex MD5 string passed from the CLI
                ``--reference-checksum`` flag. Takes highest precedence.

        Returns:
            str | None: The resolved reference checksum, or None if not pinned.
        """
        if cli_override is not None:
            return cli_override
        if self.reference_checksum_override is not None:
            return self.reference_checksum_override
        return REFERENCE_CHECKSUMS.get(self.version)

    def get_model_parallelism(self, model_size: str) -> tuple[int, int]:
        """Return (tensor_parallelism, pipeline_parallelism) for the given model size.

        Lazy-loads configs/dlio/workload/llama3_{model_size}.yaml on first access
        and caches per key. model_size must be one of '8b', '70b', '405b', '1t'
        (lowercase, per D-C2).

        Args:
            model_size: Model size key (e.g., '8b', '70b', '405b', '1t').

        Returns:
            (tp, pp) tuple of ints (per Phase 2 D-C1).

        Raises:
            FileNotFoundError: if the workload YAML does not exist.
            KeyError: if the YAML does not contain model.parallelism.tensor/pipeline.
        """
        if model_size in self._parallelism_cache:
            return self._parallelism_cache[model_size]
        yaml_filename = f"llama3_{model_size}.yaml"
        config_dir = os.path.join(
            os.path.dirname(__file__),  # configuration/
            os.pardir,                  # submission_checker/
            os.pardir,                  # mlpstorage_py/
            os.pardir,                  # repo root
            "configs", "dlio", "workload",
        )
        yaml_path = os.path.normpath(os.path.join(config_dir, yaml_filename))
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        tp = int(data["model"]["parallelism"]["tensor"])
        pp = int(data["model"]["parallelism"]["pipeline"])
        result = (tp, pp)
        self._parallelism_cache[model_size] = result
        return result

    def get_closed_mpi_processes(self, model_size: str) -> int:
        """Return the required CLOSED total MPI process count for the given model.

        Source of truth: CLOSED_MPI_PROCESSES constant (Rules.md Table 2).
        DP is not in the DLIO workload YAMLs; this constant encodes TP*PP*DP.

        Args:
            model_size: Model size key (e.g., '8b', '70b', '405b', '1t').

        Returns:
            int — required total process count for CLOSED (8 / 64 / 512 / 1024).

        Raises:
            KeyError: if model_size is not one of the four recognized keys.
        """
        return CLOSED_MPI_PROCESSES[model_size]