"""
Comprehensive Tests for the MLPerf Storage CLI parser.
Validates structural boundaries, subcommand availability, value constraints,
and explicit verification that 'closed' defaults match 'open' defaults.
"""

import sys
import pytest
import argparse
from unittest.mock import patch
from mlpstorage_py.cli_parser import parse_arguments
from mlpstorage_py.config import EXIT_CODE

# =====================================================================
# 1. Open vs. Closed Equivalence Tests
# =====================================================================

def test_kvcache_open_closed_defaults_match():
    """
    Verify that all hardcoded defaults in KVCache closed mode exactly
    match the argparse defaults provided in open mode.
    """
    base_args = ['kvcache', 'run', '--file']
    
    with patch('sys.argv', ['mlpstorage', 'closed'] + base_args):
        args_closed = parse_arguments()
        
    with patch('sys.argv', ['mlpstorage', 'open'] + base_args):
        args_open = parse_arguments()

    # Model and Users
    assert args_closed.model == args_open.model
    assert args_closed.num_users == args_open.num_users
    
    # Cache Tier Memory
    assert args_closed.gpu_mem_gb == args_open.gpu_mem_gb == 16.0
    assert args_closed.cpu_mem_gb == args_open.cpu_mem_gb == 32.0
    
    # Run configuration
    assert args_closed.duration == args_open.duration
    assert args_closed.generation_mode == args_open.generation_mode == 'realistic'
    assert args_closed.performance_profile == args_open.performance_profile == 'latency'
    
    # Optional Features
    assert args_closed.disable_multi_turn == args_open.disable_multi_turn == False
    assert args_closed.enable_rag == args_open.enable_rag == True
    assert args_closed.autoscaler_mode == args_open.autoscaler_mode == 'qos'
    
    # Universal Universal/Common arguments defaults
    assert args_closed.loops == args_open.loops == 1


def test_checkpointing_open_closed_defaults_match():
    """
    Verify that Checkpointing 'closed' hardcoded defaults (like read/write checkpoints)
    match the 'open' default values.
    """
    # Note: Checkpointing requires some mandatory arguments to pass validation
    base_args = ['checkpointing', 'run', '-cm', '1024', '-m', 'llama3.1-8b', '-np', '2', '-cf', '/tmp', '--file']
    
    with patch('sys.argv', ['mlpstorage', 'closed'] + base_args):
        args_closed = parse_arguments()
        
    with patch('sys.argv', ['mlpstorage', 'open'] + base_args):
        args_open = parse_arguments()

    assert args_closed.num_checkpoints_read == args_open.num_checkpoints_read == 10
    assert args_closed.num_checkpoints_write == args_open.num_checkpoints_write == 10


def test_training_params_strictness():
    """
    Verify that 'params' behaves correctly across modes:
    Closed mode forces it to an empty string. Open mode leaves it None if unpassed.
    """
    base_args = ['training', 'run', '-cm', '1024', '-m', 'unet3d', '-g', 'v100', '-na', '2', '--file']
    
    with patch('sys.argv', ['mlpstorage', 'closed'] + base_args):
        args_closed = parse_arguments()
        assert args_closed.params == '' # Closed mode strictness
        
    with patch('sys.argv', ['mlpstorage', 'open'] + base_args):
        args_open = parse_arguments()
        assert args_open.params is None # Open mode default (unspecified)


# =====================================================================
# 2. Structural & Subcommand Combinations (Positive Cases)
# =====================================================================

@pytest.mark.parametrize("cmd_list, expected_program, expected_command", [
    # Training combinations
    (['training', 'datasize', '-cm', '1024', '-m', 'unet3d', '-g', 'v100', '-ma', '4', '--file'], 'training', 'datasize'),
    (['training', 'datagen', '-m', 'unet3d', '-np', '4', '--file'], 'training', 'datagen'),
    (['training', 'configview', '-na', '4', '--file'], 'training', 'configview'),
    
    # Checkpointing combinations
    (['checkpointing', 'datasize', '-cm', '1024', '-m', 'llama3.1-8b', '--file'], 'checkpointing', 'datasize'),
    
    # KVCache combinations
    (['kvcache', 'datasize', '--file'], 'kvcache', 'datasize'),
    
    # Utilities
    (['reports', 'reportgen', '--file'], 'reports', 'reportgen'),
    (['history', 'show', '--file'], 'history', 'show'),
    (['history', 'rerun', '123', '--file'], 'history', 'rerun'),
    (['lockfile', 'verify', '--file'], 'lockfile', 'verify'),
])
def test_all_program_subcommand_combinations(cmd_list, expected_program, expected_command):
    """
    Parametrized test to ensure all major benchmarks and utility subcommands
    can successfully parse their minimum required arguments in open mode.
    """
    test_args = ['mlpstorage', 'open'] + cmd_list
    with patch('sys.argv', test_args):
        args = parse_arguments()
        assert args.open is True
        assert args.program == expected_program
        if expected_command:
            # Handle utilities that map dest="command" or "lockfile_command"
            cmd_val = getattr(args, 'command', getattr(args, 'lockfile_command', None))
            assert cmd_val == expected_command


# =====================================================================
# 3. Protocol Parsing & Data Access Protocol Mapping
# =====================================================================

def test_data_access_protocol_consolidation_file():
    """Test that --file is correctly consolidated into the data_access_protocol field."""
    test_args = ['mlpstorage', 'open', 'kvcache', 'run', '--file']
    with patch('sys.argv', test_args):
        args = parse_arguments()
        assert args.data_access_protocol == 'file'
        assert not hasattr(args, 'file')   # Should be deleted
        assert not hasattr(args, 'object') # Should be deleted

def test_data_access_protocol_consolidation_object():
    """Test that --object is correctly consolidated into the data_access_protocol field."""
    # Training supports object storage. We use training datagen for a simple required args profile.
    test_args = ['mlpstorage', 'open', 'training', 'datagen', '-m', 'unet3d', '-np', '4', '--object', 's3']
    with patch('sys.argv', test_args):
        args = parse_arguments()
        assert args.data_access_protocol == 's3'


# =====================================================================
# 4. Negative Validation Tests (Value constraints and illegal inputs)
# =====================================================================

def test_kvcache_rejects_object_storage():
    """KVCache custom validation should reject object storage outright."""
    test_args = ['mlpstorage', 'open', 'kvcache', 'run', '--object', 's3']
    with patch('sys.argv', test_args):
        with pytest.raises(SystemExit) as exc_info:
            parse_arguments()
        assert exc_info.value.code == EXIT_CODE.INVALID_ARGUMENTS


def test_checkpointing_rejects_negative_checkpoints():
    """Checkpointing custom validation should reject negative checkpoint counts."""
    test_args = [
        'mlpstorage', 'open', 'checkpointing', 'run', 
        '-cm', '1024', '-m', 'llama3.1-8b', '-np', '2', '-cf', '/tmp', '--file',
        '--num-checkpoints-read', '-5'
    ]
    with patch('sys.argv', test_args):
        with pytest.raises(SystemExit) as exc_info:
            parse_arguments()
        assert exc_info.value.code == EXIT_CODE.INVALID_ARGUMENTS


def test_closed_mode_rejects_open_only_args():
    """
    Open-mode arguments (like --timeseries-interval or --allow-invalid-params)
    should trigger an argparse unrecognized argument error if passed in closed mode.
    """
    test_args = [
        'mlpstorage', 'closed', 'kvcache', 'run', '--file',
        '--allow-invalid-params'  # Not added to parser in closed mode
    ]
    with patch('sys.argv', test_args):
        with pytest.raises(SystemExit) as exc_info:
            parse_arguments()
        # argparse default exit code for unrecognized arguments is 2
        assert exc_info.value.code != 0


def test_missing_required_argument_triggers_exit():
    """Omitting a required argument (like -m / --model for training) should fail."""
    test_args = ['mlpstorage', 'open', 'training', 'run', '-cm', '1024', '-g', 'v100', '-na', '2', '--file']
    # Missing '-m'
    with patch('sys.argv', test_args):
        with pytest.raises(SystemExit) as exc_info:
            parse_arguments()
        assert exc_info.value.code != 0
