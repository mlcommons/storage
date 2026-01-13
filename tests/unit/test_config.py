"""
Tests for mlpstorage.config module.

Tests cover:
- Environment variable handling (check_env)
- Datetime string generation
- Enum values and constants
"""

import os
import pytest

from mlpstorage.config import (
    check_env,
    get_datetime_string,
    BENCHMARK_TYPES,
    PARAM_VALIDATION,
    EXIT_CODE,
    MODELS,
    ACCELERATORS,
    LLM_MODELS,
    LLM_ALLOWED_VALUES,
    LLM_SIZE_BY_RANK,
    EXEC_TYPE,
)


class TestCheckEnv:
    """Tests for check_env function."""

    def test_returns_default_when_env_not_set(self, clean_env):
        """check_env returns default value when env var is not set."""
        result = check_env('NONEXISTENT_VAR', 'default_value')
        assert result == 'default_value'

    def test_returns_env_value_when_set(self, clean_env):
        """check_env returns environment value when set."""
        clean_env.setenv('TEST_VAR', 'env_value')
        result = check_env('TEST_VAR', 'default_value')
        assert result == 'env_value'

    def test_converts_string_true_to_boolean(self, clean_env):
        """check_env converts string 'true' to boolean True."""
        clean_env.setenv('TEST_BOOL', 'true')
        result = check_env('TEST_BOOL', False)
        assert result is True

    def test_converts_string_false_to_boolean(self, clean_env):
        """check_env converts string 'false' to boolean False."""
        clean_env.setenv('TEST_BOOL', 'false')
        result = check_env('TEST_BOOL', True)
        assert result is False

    def test_converts_string_True_to_boolean(self, clean_env):
        """check_env converts string 'True' to boolean True."""
        clean_env.setenv('TEST_BOOL', 'True')
        result = check_env('TEST_BOOL', False)
        assert result is True

    def test_preserves_non_boolean_strings(self, clean_env):
        """check_env preserves non-boolean string values."""
        clean_env.setenv('TEST_STR', 'some_string')
        result = check_env('TEST_STR', 'default')
        assert result == 'some_string'

    def test_default_type_preserved_for_boolean(self, clean_env):
        """When default is boolean and env not set, return boolean."""
        result = check_env('NONEXISTENT', True)
        assert result is True
        assert isinstance(result, bool)


class TestGetDatetimeString:
    """Tests for get_datetime_string function."""

    def test_returns_string(self):
        """get_datetime_string returns a string."""
        result = get_datetime_string()
        assert isinstance(result, str)

    def test_format_is_correct(self):
        """get_datetime_string returns correct format YYYYmmdd_HHMMSS."""
        result = get_datetime_string()
        # Should be 15 characters: 8 for date + 1 underscore + 6 for time
        assert len(result) == 15
        assert result[8] == '_'
        # Date part should be numeric
        assert result[:8].isdigit()
        # Time part should be numeric
        assert result[9:].isdigit()


class TestBenchmarkTypesEnum:
    """Tests for BENCHMARK_TYPES enum."""

    def test_training_exists(self):
        """BENCHMARK_TYPES has training member."""
        assert hasattr(BENCHMARK_TYPES, 'training')

    def test_checkpointing_exists(self):
        """BENCHMARK_TYPES has checkpointing member."""
        assert hasattr(BENCHMARK_TYPES, 'checkpointing')

    def test_vector_database_exists(self):
        """BENCHMARK_TYPES has vector_database member."""
        assert hasattr(BENCHMARK_TYPES, 'vector_database')

    def test_training_value(self):
        """BENCHMARK_TYPES.training has correct value."""
        assert BENCHMARK_TYPES.training.value == 'training'

    def test_checkpointing_value(self):
        """BENCHMARK_TYPES.checkpointing has correct value."""
        assert BENCHMARK_TYPES.checkpointing.value == 'checkpointing'


class TestParamValidationEnum:
    """Tests for PARAM_VALIDATION enum."""

    def test_closed_exists(self):
        """PARAM_VALIDATION has CLOSED member."""
        assert hasattr(PARAM_VALIDATION, 'CLOSED')

    def test_open_exists(self):
        """PARAM_VALIDATION has OPEN member."""
        assert hasattr(PARAM_VALIDATION, 'OPEN')

    def test_invalid_exists(self):
        """PARAM_VALIDATION has INVALID member."""
        assert hasattr(PARAM_VALIDATION, 'INVALID')

    def test_closed_value(self):
        """PARAM_VALIDATION.CLOSED has correct value."""
        assert PARAM_VALIDATION.CLOSED.value == 'closed'

    def test_open_value(self):
        """PARAM_VALIDATION.OPEN has correct value."""
        assert PARAM_VALIDATION.OPEN.value == 'open'

    def test_invalid_value(self):
        """PARAM_VALIDATION.INVALID has correct value."""
        assert PARAM_VALIDATION.INVALID.value == 'invalid'


class TestExitCodeEnum:
    """Tests for EXIT_CODE enum."""

    def test_success_is_zero(self):
        """EXIT_CODE.SUCCESS is 0."""
        assert EXIT_CODE.SUCCESS == 0

    def test_general_error_is_one(self):
        """EXIT_CODE.GENERAL_ERROR is 1."""
        assert EXIT_CODE.GENERAL_ERROR == 1

    def test_invalid_arguments_is_two(self):
        """EXIT_CODE.INVALID_ARGUMENTS is 2."""
        assert EXIT_CODE.INVALID_ARGUMENTS == 2

    def test_file_not_found_is_three(self):
        """EXIT_CODE.FILE_NOT_FOUND is 3."""
        assert EXIT_CODE.FILE_NOT_FOUND == 3

    def test_all_codes_are_unique(self):
        """All EXIT_CODE values are unique."""
        values = [code.value for code in EXIT_CODE]
        assert len(values) == len(set(values))


class TestModelsConstant:
    """Tests for MODELS constant."""

    def test_is_list(self):
        """MODELS is a list."""
        assert isinstance(MODELS, list)

    def test_contains_unet3d(self):
        """MODELS contains unet3d."""
        assert 'unet3d' in MODELS

    def test_contains_resnet50(self):
        """MODELS contains resnet50."""
        assert 'resnet50' in MODELS

    def test_contains_cosmoflow(self):
        """MODELS contains cosmoflow."""
        assert 'cosmoflow' in MODELS


class TestAcceleratorsConstant:
    """Tests for ACCELERATORS constant."""

    def test_is_list(self):
        """ACCELERATORS is a list."""
        assert isinstance(ACCELERATORS, list)

    def test_contains_h100(self):
        """ACCELERATORS contains h100."""
        assert 'h100' in ACCELERATORS

    def test_contains_a100(self):
        """ACCELERATORS contains a100."""
        assert 'a100' in ACCELERATORS


class TestLLMModelsConstant:
    """Tests for LLM_MODELS constant."""

    def test_is_list(self):
        """LLM_MODELS is a list."""
        assert isinstance(LLM_MODELS, list)

    def test_contains_llama3_8b(self):
        """LLM_MODELS contains llama3-8b."""
        assert 'llama3-8b' in LLM_MODELS

    def test_contains_llama3_70b(self):
        """LLM_MODELS contains llama3-70b."""
        assert 'llama3-70b' in LLM_MODELS

    def test_contains_llama3_405b(self):
        """LLM_MODELS contains llama3-405b."""
        assert 'llama3-405b' in LLM_MODELS

    def test_contains_llama3_1t(self):
        """LLM_MODELS contains llama3-1t."""
        assert 'llama3-1t' in LLM_MODELS


class TestLLMAllowedValues:
    """Tests for LLM_ALLOWED_VALUES constant."""

    def test_is_dict(self):
        """LLM_ALLOWED_VALUES is a dict."""
        assert isinstance(LLM_ALLOWED_VALUES, dict)

    def test_all_llm_models_have_entries(self):
        """All LLM_MODELS have entries in LLM_ALLOWED_VALUES."""
        for model in LLM_MODELS:
            assert model in LLM_ALLOWED_VALUES, f"Missing entry for {model}"

    def test_entries_are_tuples_of_four(self):
        """Each entry is a tuple of 4 values."""
        for model, values in LLM_ALLOWED_VALUES.items():
            assert isinstance(values, tuple), f"{model} entry is not a tuple"
            assert len(values) == 4, f"{model} entry does not have 4 values"

    def test_llama3_8b_values(self):
        """llama3-8b has expected structure."""
        values = LLM_ALLOWED_VALUES.get('llama3-8b')
        assert values is not None
        min_procs, zero_level, gpu_per_dp, closed_gpus = values
        assert isinstance(min_procs, int)
        assert isinstance(zero_level, int)


class TestLLMSizeByRank:
    """Tests for LLM_SIZE_BY_RANK constant."""

    def test_is_dict(self):
        """LLM_SIZE_BY_RANK is a dict."""
        assert isinstance(LLM_SIZE_BY_RANK, dict)

    def test_all_llm_models_have_entries(self):
        """All LLM_MODELS have entries in LLM_SIZE_BY_RANK."""
        for model in LLM_MODELS:
            assert model in LLM_SIZE_BY_RANK, f"Missing entry for {model}"

    def test_entries_are_tuples_of_two(self):
        """Each entry is a tuple of 2 values (model_gb, optimizer_gb)."""
        for model, values in LLM_SIZE_BY_RANK.items():
            assert isinstance(values, tuple), f"{model} entry is not a tuple"
            assert len(values) == 2, f"{model} entry does not have 2 values"

    def test_values_are_numeric(self):
        """All size values are numeric."""
        for model, (model_gb, optimizer_gb) in LLM_SIZE_BY_RANK.items():
            assert isinstance(model_gb, (int, float)), f"{model} model_gb is not numeric"
            assert isinstance(optimizer_gb, (int, float)), f"{model} optimizer_gb is not numeric"


class TestExecTypeEnum:
    """Tests for EXEC_TYPE enum."""

    def test_mpi_exists(self):
        """EXEC_TYPE has MPI member."""
        assert hasattr(EXEC_TYPE, 'MPI')

    def test_docker_exists(self):
        """EXEC_TYPE has DOCKER member."""
        assert hasattr(EXEC_TYPE, 'DOCKER')

    def test_mpi_value(self):
        """EXEC_TYPE.MPI has correct value."""
        assert EXEC_TYPE.MPI.value == 'mpi'
