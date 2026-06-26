"""
Training benchmark run rules checker.

Validates training benchmark parameters for individual runs.
"""

from typing import Optional, List

from mlpstorage_py.config import BENCHMARK_TYPES, PARAM_VALIDATION, UNET, DLRM, RETINANET, FLUX, MODELS
from mlpstorage_py.rules.issues import Issue
from mlpstorage_py.rules.run_checkers.base import RunRulesChecker
from mlpstorage_py.rules.utils import calculate_training_data_size


class TrainingRunRulesChecker(RunRulesChecker):
    """Rules checker for training benchmarks."""

    # Parameters allowed for CLOSED submission
    CLOSED_ALLOWED_PARAMS = [
        'dataset.num_files_train',
        'dataset.num_subfolders_train',
        'dataset.data_folder',
        'reader.read_threads',
        'reader.computation_threads',
        'reader.transfer_size',
        'reader.odirect',
        'reader.prefetch_size',
        'checkpoint.checkpoint_folder',
        'storage.storage_type',
        'storage.storage_root',
    ]

    # Parameters allowed for OPEN submission (but not CLOSED)
    OPEN_ALLOWED_PARAMS = [
        'framework',
        'dataset.format',
        'dataset.num_samples_per_file',
        'reader.data_loader',
    ]

    # Parameters that the mlpstorage tool injects into the DLIO config without
    # the user typing --params for them.  They show up in override_parameters
    # alongside genuine user overrides because both share the same dotted-key
    # surface, so check_allowed_params() must filter them out before deciding
    # CLOSED-vs-OPEN-vs-INVALID — otherwise a closed run is marked INVALID for
    # params the tool itself set (mlcommons/storage#494).
    #
    # When adding a new tool-side setter in mlpstorage_py/benchmarks/dlio.py
    # (or anywhere that writes to self.params_dict outside of args.params),
    # add the dotted-key here too.
    TOOL_INJECTED_PARAMS = frozenset({
        # _apply_skip_listing_params (#483)
        'dataset.skip_listing',
        'dataset.listing_validation_interval',
        # add_datadir_param — derived from --data-dir + model
        'dataset.data_folder',
        # _apply_object_storage_params — derived from --object + BUCKET / env
        'storage.storage_type',
        'storage.storage_root',
        'storage.storage_options.storage_library',
        'storage.storage_options.uri_scheme',
        'storage.s3_force_path_style',
    })

    def check_benchmark_type(self) -> Optional[Issue]:
        """Verify this is a training benchmark."""
        if self.benchmark_run.benchmark_type != BENCHMARK_TYPES.training:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid benchmark type: {self.benchmark_run.benchmark_type}",
                parameter="benchmark_type",
                expected=BENCHMARK_TYPES.training,
                actual=self.benchmark_run.benchmark_type
            )
        return None

    def check_model_recognized(self) -> Optional[Issue]:
        """Verify the model is a recognized training model."""
        if self.benchmark_run.model not in MODELS:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Unrecognized model: {self.benchmark_run.model}",
                parameter="model",
                expected=f"One of: {', '.join(MODELS)}",
                actual=self.benchmark_run.model
            )
        return None

    def check_num_files_train(self) -> Optional[Issue]:
        """Check if the number of training files meets the minimum requirement."""
        if 'dataset' not in self.benchmark_run.parameters:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="Missing dataset parameters",
                parameter="dataset"
            )

        dataset_params = self.benchmark_run.parameters['dataset']
        if 'num_files_train' not in dataset_params:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="Missing num_files_train parameter",
                parameter="dataset.num_files_train"
            )

        configured_num_files = int(dataset_params['num_files_train'])
        reader_params = self.benchmark_run.parameters.get('reader', {})

        try:
            required_num_files, _, _ = calculate_training_data_size(
                None,
                self.benchmark_run.system_info,
                dataset_params,
                reader_params,
                self.logger,
                self.benchmark_run.num_processes
            )
        except ValueError as e:
            # Loaded-from-disk runs (reportgen) may lack cluster_information,
            # so the 5×memory rule cannot be evaluated. Skip the check rather
            # than crashing the entire verification (which previously marked
            # every run INVALID via an AttributeError caught by the verifier
            # framework). (#503)
            if "cluster_information" in str(e):
                self.logger.warning(
                    f"Skipping check_num_files_train: {e}. "
                    f"The check requires live cluster info; this run was "
                    f"loaded from on-disk metadata that does not preserve it."
                )
                return None
            raise

        if configured_num_files < required_num_files:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Insufficient number of training files",
                parameter="dataset.num_files_train",
                expected=f">= {required_num_files}",
                actual=configured_num_files
            )

        return None

    def check_allowed_params(self) -> List[Issue]:
        """
        Verify that only allowed parameters were overridden.

        Returns list of issues describing which parameters are allowed
        for CLOSED, OPEN, or are invalid.  Tool-injected params (see
        TOOL_INJECTED_PARAMS) are reported under a separate "Tool-injected
        parameter" message so reviewers can audit them, but never count as
        user overrides for the CLOSED-vs-OPEN-vs-INVALID gate.
        """
        issues = []
        for param, value in self.benchmark_run.override_parameters.items():
            if param.startswith("workflow"):
                # Workflow parameters are handled separately
                continue

            self.logger.debug(f"Processing override parameter: {param} = {value}")

            if param in self.TOOL_INJECTED_PARAMS:
                # Tool-managed knob (skip_listing, object-storage backend,
                # auto-resolved data_folder, etc).  Surface it for audit but
                # don't subject it to the user-override allow-list.
                issues.append(Issue(
                    validation=PARAM_VALIDATION.CLOSED,
                    message=f"Tool-injected parameter: {param} = {value}",
                    parameter="Tool-Injected Parameters",
                    actual=value
                ))
            elif param in self.CLOSED_ALLOWED_PARAMS:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.CLOSED,
                    message=f"Closed parameter override allowed: {param} = {value}",
                    parameter="Overrode Parameters",
                    actual=value
                ))
            elif param in self.OPEN_ALLOWED_PARAMS:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.OPEN,
                    message=f"Open parameter override allowed: {param} = {value}",
                    parameter="Overrode Parameters",
                    actual=value
                ))
            else:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Disallowed parameter override: {param} = {value}",
                    parameter="Overrode Parameters",
                    expected="None",
                    actual=value
                ))

        return issues

    def check_workflow_parameters(self) -> Optional[Issue]:
        """Check if workflow parameters are valid for the model."""
        workflow_params = self.benchmark_run.parameters.get('workflow', {})

        for param, value in workflow_params.items():
            if self.benchmark_run.model == UNET and self.benchmark_run.command == "run_benchmark":
                # Unet3d training requires checkpoint workflow = True
                if param == "checkpoint":
                    if value == True:
                        return Issue(
                            validation=PARAM_VALIDATION.CLOSED,
                            message="Unet3D training requires executing a checkpoint",
                            parameter="workflow.checkpoint",
                            expected="True",
                            actual=value
                        )
                    elif value == False:
                        return Issue(
                            validation=PARAM_VALIDATION.INVALID,
                            message="Unet3D training requires executing a checkpoint. "
                                    "The parameter 'workflow.checkpoint' is set to False",
                            parameter="workflow.checkpoint",
                            expected="True",
                            actual=value
                        )

        return None

    def check_odirect_supported_model(self) -> Optional[Issue]:
        """Check if reader.odirect is only used with supported models."""
        odirect = self.benchmark_run.parameters.get('reader', {}).get('odirect')
        # odirect is only supported for UNet3D
        odirect_supported_models = [UNET]
        if odirect and self.benchmark_run.model not in odirect_supported_models:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"The reader.odirect option is only supported for {', '.join(odirect_supported_models)}",
                parameter="reader.odirect",
                expected="False",
                actual=odirect
            )
        return None

    def check_checkpoint_files_in_code(self) -> Optional[Issue]:
        """Placeholder for checkpoint files validation."""
        pass

    def check_num_epochs(self) -> Optional[Issue]:
        """Placeholder for epoch count validation."""
        pass

    def check_inter_test_times(self) -> Optional[Issue]:
        """Placeholder for inter-test timing validation."""
        pass

    def check_file_system_caching(self) -> Optional[Issue]:
        """Placeholder for file system caching validation."""
        pass
