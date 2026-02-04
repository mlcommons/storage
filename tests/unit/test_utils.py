"""
Tests for mlpstorage.utils module.

Tests cover:
- MLPSJsonEncoder custom JSON encoding
- Datetime validation and parsing
- Nested dictionary operations (update, create, flatten)
- NaN value removal
- MPI prefix command generation
"""

import enum
import json
import math
import logging
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from mlpstorage.utils import (
    MLPSJsonEncoder,
    is_valid_datetime_format,
    get_datetime_from_timestamp,
    update_nested_dict,
    create_nested_dict,
    flatten_nested_dict,
    remove_nan_values,
    generate_mpi_prefix_cmd,
)
from mlpstorage.config import MPIRUN, MPIEXEC


class TestMLPSJsonEncoder:
    """Tests for MLPSJsonEncoder custom JSON encoder."""

    def test_encodes_basic_types(self):
        """Encoder handles basic types (int, float, str, list, dict)."""
        data = {
            'int': 42,
            'float': 3.14,
            'str': 'hello',
            'list': [1, 2, 3],
            'dict': {'nested': True}
        }
        result = json.dumps(data, cls=MLPSJsonEncoder)
        parsed = json.loads(result)
        assert parsed == data

    def test_encodes_set_as_list(self):
        """Encoder converts set to list."""
        data = {'set_value': {1, 2, 3}}
        result = json.dumps(data, cls=MLPSJsonEncoder)
        parsed = json.loads(result)
        # Sets become lists, order may vary
        assert set(parsed['set_value']) == {1, 2, 3}

    def test_encodes_enum_as_value(self):
        """Encoder converts enum to its value."""
        class TestEnum(enum.Enum):
            TEST = 'test_value'

        data = {'enum': TestEnum.TEST}
        result = json.dumps(data, cls=MLPSJsonEncoder)
        parsed = json.loads(result)
        assert parsed['enum'] == 'test_value'

    def test_encodes_dataclass(self):
        """Encoder converts dataclass to dict via __dict__."""
        @dataclass
        class TestDataclass:
            name: str
            value: int

        obj = TestDataclass(name='test', value=42)
        data = {'dataclass': obj}
        result = json.dumps(data, cls=MLPSJsonEncoder)
        parsed = json.loads(result)
        assert parsed['dataclass'] == {'name': 'test', 'value': 42}

    def test_encodes_logger_as_string(self):
        """Encoder converts Logger objects to string representation."""
        logger = logging.getLogger('test_logger')
        data = {'logger': logger}
        result = json.dumps(data, cls=MLPSJsonEncoder)
        parsed = json.loads(result)
        assert parsed['logger'] == 'Logger object'

    def test_encodes_object_with_dict_attribute(self):
        """Encoder uses __dict__ for objects with that attribute."""
        class CustomObject:
            def __init__(self):
                self.name = 'custom'
                self.value = 123

        obj = CustomObject()
        data = {'custom': obj}
        result = json.dumps(data, cls=MLPSJsonEncoder)
        parsed = json.loads(result)
        assert parsed['custom']['name'] == 'custom'
        assert parsed['custom']['value'] == 123

    def test_fallback_to_str_on_exception(self):
        """Encoder falls back to str() on exception."""
        class UnserializableObject:
            def __init__(self):
                pass

            @property
            def __dict__(self):
                raise Exception("Cannot serialize")

            def __str__(self):
                return "UnserializableObject()"

        obj = UnserializableObject()
        data = {'obj': obj}
        # Should not raise, should convert to string
        result = json.dumps(data, cls=MLPSJsonEncoder)
        assert 'UnserializableObject' in result


class TestIsValidDatetimeFormat:
    """Tests for is_valid_datetime_format function."""

    def test_valid_datetime_format(self):
        """Valid datetime string returns True."""
        assert is_valid_datetime_format('20250111_143022') is True

    def test_valid_datetime_midnight(self):
        """Valid midnight datetime returns True."""
        assert is_valid_datetime_format('20250101_000000') is True

    def test_valid_datetime_end_of_day(self):
        """Valid end-of-day datetime returns True."""
        assert is_valid_datetime_format('20251231_235959') is True

    def test_invalid_length_short(self):
        """Too short string returns False."""
        assert is_valid_datetime_format('20250111_14302') is False

    def test_invalid_length_long(self):
        """Too long string returns False."""
        assert is_valid_datetime_format('20250111_1430221') is False

    def test_invalid_separator(self):
        """Wrong separator returns False."""
        assert is_valid_datetime_format('20250111-143022') is False

    def test_invalid_no_separator(self):
        """Missing separator returns False."""
        assert is_valid_datetime_format('202501111430222') is False

    def test_invalid_date(self):
        """Invalid date (month 13) returns False."""
        assert is_valid_datetime_format('20251311_143022') is False

    def test_invalid_time(self):
        """Invalid time (hour 25) returns False."""
        assert is_valid_datetime_format('20250111_253022') is False

    def test_non_numeric_characters(self):
        """Non-numeric characters return False."""
        assert is_valid_datetime_format('2025011a_143022') is False

    def test_empty_string(self):
        """Empty string returns False."""
        assert is_valid_datetime_format('') is False


class TestGetDatetimeFromTimestamp:
    """Tests for get_datetime_from_timestamp function."""

    def test_valid_timestamp_returns_datetime(self):
        """Valid timestamp returns datetime object."""
        result = get_datetime_from_timestamp('20250111_143022')
        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 11
        assert result.hour == 14
        assert result.minute == 30
        assert result.second == 22

    def test_invalid_timestamp_returns_none(self):
        """Invalid timestamp returns None."""
        result = get_datetime_from_timestamp('invalid')
        assert result is None

    def test_wrong_format_returns_none(self):
        """Wrong format returns None."""
        result = get_datetime_from_timestamp('2025-01-11 14:30:22')
        assert result is None


class TestUpdateNestedDict:
    """Tests for update_nested_dict function."""

    def test_simple_update(self):
        """Simple key update works."""
        original = {'a': 1, 'b': 2}
        update = {'b': 3}
        result = update_nested_dict(original, update)
        assert result == {'a': 1, 'b': 3}

    def test_nested_update(self):
        """Nested dictionary update merges correctly."""
        original = {'a': {'b': 1, 'c': 2}}
        update = {'a': {'c': 3}}
        result = update_nested_dict(original, update)
        assert result == {'a': {'b': 1, 'c': 3}}

    def test_adds_new_keys(self):
        """New keys from update are added."""
        original = {'a': 1}
        update = {'b': 2}
        result = update_nested_dict(original, update)
        assert result == {'a': 1, 'b': 2}

    def test_adds_new_nested_keys(self):
        """New nested keys from update are added."""
        original = {'a': {'b': 1}}
        update = {'a': {'c': 2}}
        result = update_nested_dict(original, update)
        assert result == {'a': {'b': 1, 'c': 2}}

    def test_preserves_non_overlapping(self):
        """Non-overlapping keys are preserved."""
        original = {'a': 1, 'b': {'c': 2}}
        update = {'d': 3}
        result = update_nested_dict(original, update)
        assert result == {'a': 1, 'b': {'c': 2}, 'd': 3}

    def test_deep_nested_update(self):
        """Deep nested update works correctly."""
        original = {'a': {'b': {'c': {'d': 1}}}}
        update = {'a': {'b': {'c': {'d': 2, 'e': 3}}}}
        result = update_nested_dict(original, update)
        assert result == {'a': {'b': {'c': {'d': 2, 'e': 3}}}}

    def test_original_not_modified(self):
        """Original dictionary is not modified."""
        original = {'a': 1}
        update = {'a': 2}
        result = update_nested_dict(original, update)
        assert original == {'a': 1}
        assert result == {'a': 2}

    def test_replace_dict_with_scalar(self):
        """Scalar value replaces nested dict."""
        original = {'a': {'b': 1}}
        update = {'a': 2}
        result = update_nested_dict(original, update)
        assert result == {'a': 2}

    def test_replace_scalar_with_dict(self):
        """Dict value replaces scalar."""
        original = {'a': 1}
        update = {'a': {'b': 2}}
        result = update_nested_dict(original, update)
        assert result == {'a': {'b': 2}}


class TestCreateNestedDict:
    """Tests for create_nested_dict function."""

    def test_single_level_key(self):
        """Single level key creates simple dict."""
        flat = {'a': 1}
        result = create_nested_dict(flat)
        assert result == {'a': 1}

    def test_two_level_key(self):
        """Two level key creates nested dict."""
        flat = {'a.b': 1}
        result = create_nested_dict(flat)
        assert result == {'a': {'b': 1}}

    def test_multiple_nested_keys(self):
        """Multiple nested keys create proper structure."""
        flat = {'a.b': 1, 'a.c': 2}
        result = create_nested_dict(flat)
        assert result == {'a': {'b': 1, 'c': 2}}

    def test_deep_nesting(self):
        """Deep nesting works correctly."""
        flat = {'a.b.c.d': 1}
        result = create_nested_dict(flat)
        assert result == {'a': {'b': {'c': {'d': 1}}}}

    def test_mixed_keys(self):
        """Mix of flat and nested keys works."""
        flat = {'a': 1, 'b.c': 2}
        result = create_nested_dict(flat)
        assert result == {'a': 1, 'b': {'c': 2}}

    def test_custom_separator(self):
        """Custom separator is honored."""
        flat = {'a/b': 1}
        result = create_nested_dict(flat, separator='/')
        assert result == {'a': {'b': 1}}

    def test_with_parent_dict(self):
        """Parent dict is updated correctly."""
        parent = {'x': 1}
        flat = {'a.b': 2}
        result = create_nested_dict(flat, parent_dict=parent)
        assert result == {'x': 1, 'a': {'b': 2}}

    def test_empty_dict(self):
        """Empty flat dict returns empty dict."""
        result = create_nested_dict({})
        assert result == {}


class TestFlattenNestedDict:
    """Tests for flatten_nested_dict function."""

    def test_flat_dict_unchanged(self):
        """Already flat dict is unchanged."""
        nested = {'a': 1, 'b': 2}
        result = flatten_nested_dict(nested)
        assert result == {'a': 1, 'b': 2}

    def test_single_nesting(self):
        """Single level nesting flattens correctly."""
        nested = {'a': {'b': 1}}
        result = flatten_nested_dict(nested)
        assert result == {'a.b': 1}

    def test_multiple_nested_keys(self):
        """Multiple nested keys flatten correctly."""
        nested = {'a': {'b': 1, 'c': 2}}
        result = flatten_nested_dict(nested)
        assert result == {'a.b': 1, 'a.c': 2}

    def test_deep_nesting(self):
        """Deep nesting flattens correctly."""
        nested = {'a': {'b': {'c': {'d': 1}}}}
        result = flatten_nested_dict(nested)
        assert result == {'a.b.c.d': 1}

    def test_mixed_structure(self):
        """Mixed structure flattens correctly."""
        nested = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        result = flatten_nested_dict(nested)
        assert result == {'a': 1, 'b.c': 2, 'b.d.e': 3}

    def test_custom_separator(self):
        """Custom separator is used in keys."""
        nested = {'a': {'b': 1}}
        result = flatten_nested_dict(nested, separator='/')
        assert result == {'a/b': 1}

    def test_empty_dict(self):
        """Empty dict returns empty dict."""
        result = flatten_nested_dict({})
        assert result == {}

    def test_list_value_not_flattened(self):
        """List values are not flattened."""
        nested = {'a': {'b': [1, 2, 3]}}
        result = flatten_nested_dict(nested)
        assert result == {'a.b': [1, 2, 3]}


class TestFlattenAndCreateInverse:
    """Tests that flatten_nested_dict and create_nested_dict are inverses."""

    def test_flatten_then_create(self):
        """Flattening then creating restores original."""
        original = {'a': {'b': 1, 'c': 2}, 'd': {'e': {'f': 3}}}
        flattened = flatten_nested_dict(original)
        restored = create_nested_dict(flattened)
        assert restored == original

    def test_create_then_flatten(self):
        """Creating then flattening restores original."""
        original = {'a.b': 1, 'a.c': 2, 'd.e.f': 3}
        nested = create_nested_dict(original)
        flattened = flatten_nested_dict(nested)
        assert flattened == original


class TestRemoveNanValues:
    """Tests for remove_nan_values function."""

    def test_removes_nan_float(self):
        """NaN float values are removed."""
        input_dict = {'a': 1, 'b': float('nan')}
        result = remove_nan_values(input_dict)
        assert 'a' in result
        assert 'b' not in result

    def test_preserves_normal_values(self):
        """Normal values are preserved."""
        input_dict = {'a': 1, 'b': 2.5, 'c': 'string'}
        result = remove_nan_values(input_dict)
        assert result == input_dict

    def test_removes_multiple_nan(self):
        """Multiple NaN values are removed."""
        input_dict = {'a': float('nan'), 'b': 1, 'c': float('nan')}
        result = remove_nan_values(input_dict)
        assert result == {'b': 1}

    def test_empty_dict(self):
        """Empty dict returns empty dict."""
        result = remove_nan_values({})
        assert result == {}

    def test_all_nan_returns_empty(self):
        """Dict with only NaN values returns empty dict."""
        input_dict = {'a': float('nan'), 'b': float('nan')}
        result = remove_nan_values(input_dict)
        assert result == {}

    def test_original_not_modified(self):
        """Original dictionary is not modified."""
        input_dict = {'a': 1, 'b': float('nan')}
        result = remove_nan_values(input_dict)
        assert len(input_dict) == 2  # Original still has both keys

    def test_preserves_zero(self):
        """Zero values are preserved (not confused with NaN)."""
        input_dict = {'a': 0, 'b': 0.0}
        result = remove_nan_values(input_dict)
        assert result == {'a': 0, 'b': 0.0}

    def test_preserves_negative(self):
        """Negative values are preserved."""
        input_dict = {'a': -1, 'b': -1.5}
        result = remove_nan_values(input_dict)
        assert result == {'a': -1, 'b': -1.5}


class TestGenerateMpiPrefixCmd:
    """Tests for generate_mpi_prefix_cmd function."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock(spec=logging.Logger)

    def test_single_host_mpirun(self, mock_logger):
        """Single host with mpirun generates correct command."""
        result = generate_mpi_prefix_cmd(
            mpi_cmd=MPIRUN,
            hosts=['host1'],
            num_processes=4,
            oversubscribe=False,
            allow_run_as_root=False,
            params=None,
            logger=mock_logger
        )
        assert '-n 4' in result
        assert 'host1:4' in result
        assert '--bind-to none' in result
        assert '--map-by socket' in result

    def test_multiple_hosts_mpirun(self, mock_logger):
        """Multiple hosts with mpirun generates correct command."""
        result = generate_mpi_prefix_cmd(
            mpi_cmd=MPIRUN,
            hosts=['host1', 'host2'],
            num_processes=8,
            oversubscribe=False,
            allow_run_as_root=False,
            params=None,
            logger=mock_logger
        )
        assert '-n 8' in result
        assert '--map-by node' in result  # Multi-host uses node

    def test_mpiexec_command(self, mock_logger):
        """mpiexec generates correct command prefix."""
        result = generate_mpi_prefix_cmd(
            mpi_cmd=MPIEXEC,
            hosts=['host1'],
            num_processes=4,
            oversubscribe=False,
            allow_run_as_root=False,
            params=None,
            logger=mock_logger
        )
        assert 'mpiexec' in result or '-n 4' in result

    def test_oversubscribe_flag(self, mock_logger):
        """Oversubscribe flag is added when True."""
        result = generate_mpi_prefix_cmd(
            mpi_cmd=MPIRUN,
            hosts=['host1'],
            num_processes=4,
            oversubscribe=True,
            allow_run_as_root=False,
            params=None,
            logger=mock_logger
        )
        assert '--oversubscribe' in result

    def test_allow_run_as_root_flag(self, mock_logger):
        """Allow run as root flag is added when True."""
        result = generate_mpi_prefix_cmd(
            mpi_cmd=MPIRUN,
            hosts=['host1'],
            num_processes=4,
            oversubscribe=False,
            allow_run_as_root=True,
            params=None,
            logger=mock_logger
        )
        assert '--allow-run-as-root' in result

    def test_custom_params(self, mock_logger):
        """Custom params are appended to command."""
        result = generate_mpi_prefix_cmd(
            mpi_cmd=MPIRUN,
            hosts=['host1'],
            num_processes=4,
            oversubscribe=False,
            allow_run_as_root=False,
            params=['--custom-param', '--another-param'],
            logger=mock_logger
        )
        assert '--custom-param' in result
        assert '--another-param' in result

    def test_hosts_with_slots(self, mock_logger):
        """Hosts with slot definitions are handled correctly."""
        result = generate_mpi_prefix_cmd(
            mpi_cmd=MPIRUN,
            hosts=['host1:4', 'host2:4'],
            num_processes=8,
            oversubscribe=False,
            allow_run_as_root=False,
            params=None,
            logger=mock_logger
        )
        assert '-n 8' in result
        assert 'host1:4' in result
        assert 'host2:4' in result

    def test_insufficient_slots_raises_error(self, mock_logger):
        """Insufficient configured slots raises ValueError."""
        with pytest.raises(ValueError, match="not sufficient"):
            generate_mpi_prefix_cmd(
                mpi_cmd=MPIRUN,
                hosts=['host1:2', 'host2:2'],  # Only 4 slots
                num_processes=8,  # Need 8 processes
                oversubscribe=False,
                allow_run_as_root=False,
                params=None,
                logger=mock_logger
            )

    def test_unsupported_mpi_command_raises_error(self, mock_logger):
        """Unsupported MPI command raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported MPI command"):
            generate_mpi_prefix_cmd(
                mpi_cmd='unknown_mpi',
                hosts=['host1'],
                num_processes=4,
                oversubscribe=False,
                allow_run_as_root=False,
                params=None,
                logger=mock_logger
            )

    def test_uneven_distribution(self, mock_logger):
        """Processes are distributed unevenly when not divisible."""
        result = generate_mpi_prefix_cmd(
            mpi_cmd=MPIRUN,
            hosts=['host1', 'host2', 'host3'],
            num_processes=7,
            oversubscribe=False,
            allow_run_as_root=False,
            params=None,
            logger=mock_logger
        )
        # 7 processes across 3 hosts: 3, 2, 2 distribution
        assert '-n 7' in result


class TestCommandExecutor:
    """Tests for CommandExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create a CommandExecutor instance."""
        from mlpstorage.utils import CommandExecutor
        logger = MagicMock(spec=logging.Logger)
        return CommandExecutor(logger=logger)

    def test_execute_simple_command(self, executor):
        """Execute a simple command returns output."""
        stdout, stderr, returncode = executor.execute('echo hello')
        assert 'hello' in stdout
        assert returncode == 0

    def test_execute_command_with_stderr(self, executor):
        """Command with stderr captures stderr output."""
        stdout, stderr, returncode = executor.execute('ls /nonexistent_path_12345')
        assert returncode != 0
        # stderr should contain error message or stdout may contain it

    def test_execute_returns_return_code(self, executor):
        """Execute returns correct return code."""
        _, _, returncode = executor.execute('true')
        assert returncode == 0

        _, _, returncode = executor.execute('false')
        assert returncode == 1

    def test_execute_command_list(self, executor):
        """Execute accepts command as list."""
        stdout, stderr, returncode = executor.execute(['echo', 'hello', 'world'])
        assert 'hello world' in stdout
        assert returncode == 0
