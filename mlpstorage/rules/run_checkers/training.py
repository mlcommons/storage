"""
Training benchmark run rules checker.

Validates training benchmark parameters for individual runs.
"""

from typing import Optional, List

from mlpstorage.config import BENCHMARK_TYPES, PARAM_VALIDATION, UNET
from mlpstorage.rules.issues import Issue
from mlpstorage.rules.run_checkers.base import RunRulesChecker
from mlpstorage.rules.utils import calculate_training_data_size


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

        required_num_files, _, _ = calculate_training_data_size(
            None,
            self.benchmark_run.system_info,
            dataset_params,
            reader_params,
            self.logger,
            self.benchmark_run.num_processes
        )

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
        for CLOSED, OPEN, or are invalid.
        """
        issues = []
        for param, value in self.benchmark_run.override_parameters.items():
            if param.startswith("workflow"):
                # Workflow parameters are handled separately
                continue

            self.logger.debug(f"Processing override parameter: {param} = {value}")

            if param in self.CLOSED_ALLOWED_PARAMS:
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
        if self.benchmark_run.model != UNET and odirect:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="The reader.odirect option is only supported for Unet3d model",
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
