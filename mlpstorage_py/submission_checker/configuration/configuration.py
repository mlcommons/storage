import os
import yaml

from mlpstorage_py.editions import load_editions

from ..constants import *


class Config:
    """Checker configuration: the tree-wide options plus the rules edition
    whose parameters (``mlpstorage_py/rules/editions.yaml`` ``checker:`` --
    required leaf contents, AU minimums, Table 2 process counts, Table 3
    accelerator memory) and sanctioned workloads (``workloads:``) the checks
    read.

    ``edition=None`` is the current edition. ``main.run`` builds one Config
    tree-wide for the pre-loop checks and then, per submission, one for the
    edition its ``submission.yaml`` declares (``for_edition``). Construction
    is strict: an edition the table does not know, or lists without checker
    parameters, raises ``UncheckableEditionError``.
    """

    def __init__(self, edition=None, submitters=None, skip_output_file=False):
        table = load_editions()
        self.edition = table.current_edition if edition is None else str(edition)
        self.checker = table.require_checkable(self.edition)
        self.edition_entry = table.edition(self.edition)
        self.submitters = submitters
        self.skip_output_file = skip_output_file
        self._parallelism_cache: dict[str, tuple[int, int]] = {}  # lazy-load cache for get_model_parallelism

    def for_edition(self, edition):
        """A Config for another rules edition with the same tree-wide options."""
        if str(edition) == self.edition:
            return self
        return Config(edition=edition, submitters=self.submitters,
                      skip_output_file=self.skip_output_file)

    def check_submitter(self, submitter):
        if self.submitters is None:
            return True
        return submitter in self.submitters

    def get_datagen_required_files(self):
        return self.checker.datagen_required_files

    def get_run_required_files(self):
        return self.checker.run_required_files

    def get_checkpoint_required_files(self):
        return self.checker.checkpoint_required_files

    def get_datagen_required_folders(self):
        return self.checker.datagen_required_folders

    def get_run_required_folders(self):
        return self.checker.run_required_folders

    def get_checkpoint_required_folders(self):
        return self.checker.checkpoint_required_folders

    def get_accelerator_memory_gb(self):
        """Rules.md Table 3 for this edition: accelerator name -> memory in GB (4.3.4)."""
        return self.checker.accelerator_memory_gb

    def get_runs_per_result(self, family):
        """Rules.md 1.3: the runs one complete result of ``family`` holds in
        this edition (2.1.17 training leaves, 4.7.1 checkpointing phases,
        5.3.1 vector_database leaves, kv_cache sequence runs). ``KeyError``
        for a family the edition does not sanction."""
        return self.checker.runs_per_result[family]

    def get_training_au_threshold(self, model):
        """Rules.md 3.3.2 minimum mean AU for ``model`` in this edition, as a
        fraction, or ``None`` when the edition lists no minimum for it."""
        return self.checker.training_au_thresholds.get(model)

    def models(self, family, division=None):
        """The models this edition sanctions for ``family`` (in one division, or
        any when ``None``), from the table's ``workloads:``."""
        return self.edition_entry.models(family, division)

    # Issue #608: get_num_train_files / get_num_eval_files were deleted —
    # they only ever returned values from the NUM_DATASET_*_FILES placeholder
    # dicts (`# TODO: Ask for correct values`) that are also gone. Rule 3.3.1
    # now reads per-submission datasize metadata instead.

    def get_checkpoint_file(self, model):
        # See get_num_train_files: .get over [] for None-on-miss semantics.
        return CHECKPOINT_FILE_MAP.get(model)

    # NOTE: get_reference_checksum() was removed in Phase 8 (D-88).
    # Per-image mlpstorage_version lookup (D-86) via REFERENCE_CHECKSUMS is now
    # the only path — see CHECK-05 in Plan 08-02. REFERENCE_CHECKSUMS constant
    # is retained in constants.py for that lookup.

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

        Source of truth: this edition's ``checker.closed_mpi_processes`` in the
        rules editions table (Rules.md Table 2 "Total Processes", TP*PP*DP; DP
        is not in the DLIO workload YAMLs).

        Args:
            model_size: Model size key (e.g., '8b', '70b', '405b', '1t'), or
                the full model name (``llama3-70b``).

        Returns:
            int — required total process count for CLOSED (8 / 64 / 512 / 1024 in 3.0).

        Raises:
            KeyError: if the edition lists no such model.
        """
        counts = self.checker.closed_mpi_processes
        if model_size in counts:
            return counts[model_size]
        return counts[f"llama3-{model_size}"]