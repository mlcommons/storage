from .config import MODELS, PARAM_VALIDATION, MAX_READ_THREADS_TRAINING, LLM_MODELS


def validate_dlio_parameter(model, param, value):
    """
    This function applies rules to the allowed changeable dlio parameters.
    """
    if model in MODELS:
        # Allowed to change data_folder and number of files to train depending on memory requirements
        if param.startswith('dataset'):
            left, right = param.split('.')
            if right in ('data_folder', 'num_files_train'):
                # TODO: Add check of min num_files for given memory config
                return PARAM_VALIDATION.CLOSED

        # Allowed to set number of read threads
        if param.startswith('reader'):
            left, right = param.split('.')
            if right == "read_threads":
                if 0 < int(value) < MAX_READ_THREADS_TRAINING:
                    return PARAM_VALIDATION.CLOSED

    elif model in LLM_MODELS:
        # TODO: Define params that can be modified in closed
        pass

    return PARAM_VALIDATION.INVALID
