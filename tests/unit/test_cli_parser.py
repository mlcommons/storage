"""
Comprehensive Tests for the MLPerf Storage CLI parser.
Validates structural boundaries, subcommand availability, value constraints,
YAML overrides, post-parse argument updates, and 'closed' vs 'open' parity.
"""

import sys
import pytest
import argparse
from unittest.mock import patch, mock_open
from mlpstorage_py.cli_parser import parse_arguments, update_args, apply_yaml_config_overrides
from mlpstorage_py.config import EXIT_CODE

# =====================================================================
# 1. Open vs. Closed Equivalence & Constraints Tests
# =====================================================================

class TestOpenClosedEquivalence:
    
    def test_kvcache_open_closed_defaults_match(self):
        """Verify hardcoded defaults in KVCache closed mode exactly match argparse open mode defaults."""
        base_args = ['kvcache', 'run', '-rd', '/tmp', '-m', 'llama3.1-8b', '-nu', '100']
        
        with patch('sys.argv', ['mlpstorage', 'closed'] + base_args):
            args_closed = parse_arguments()
            
        with patch('sys.argv', ['mlpstorage', 'open'] + base_args):
            args_open = parse_arguments()

        # Check attributes forced by closed mode
        assert args_closed.model == args_open.model == 'llama3.1-8b'
        assert args_closed.gpu_mem_gb == args_open.gpu_mem_gb == 16.0
        assert args_closed.duration == args_open.duration == 60
        assert args_closed.loops == args_open.loops == 1
        assert args_closed.disable_multi_turn == args_open.disable_multi_turn == False

    def test_checkpointing_open_closed_defaults_match(self):
        """Verify Checkpointing 'closed' forces read/write checkpoint counts to match 'open' defaults."""
        base_args = ['checkpointing', 'run', '-cm', '1024', '-m', 'llama3-8b', '-np', '2', '-cf', '/tmp/ckpt', '-rd', '/tmp', '--file']
        
        with patch('sys.argv', ['mlpstorage', 'closed'] + base_args):
            args_closed = parse_arguments()
            
        with patch('sys.argv', ['mlpstorage', 'open'] + base_args):
            args_open = parse_arguments()

        assert args_closed.num_checkpoints_read == args_open.num_checkpoints_read == 10
        assert args_closed.num_checkpoints_write == args_open.num_checkpoints_write == 10

    def test_closed_mode_strips_open_args(self):
        """Open-mode arguments should trigger an unrecognized argument error if passed in closed mode."""
        test_args = ['mlpstorage', 'closed', 'kvcache', 'run', '-rd', '/tmp', '--allow-invalid-params']
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code != 0


# =====================================================================
# 2. Structural & Subcommand Combinations (Positive Cases)
# =====================================================================

class TestCLIStructureAndCombinations:

    @pytest.mark.parametrize("test_name, cmd_list, expected_program, expected_command", [
        # Training (Run requires -rd)
        ("01", ['training', 'run', '-cm', '1024', '-m', 'retinanet', '-g', 'b200', '-na', '4', '-rd', '/tmp', '--file'], 'training', 'run'),
        ("02", ['training', 'datasize', '-cm', '1024', '-m', 'flux', '-g', 'b200', '-ma', '4'], 'training', 'datasize'),
        ("03", ['training', 'datagen', '-m', 'dlrm', '-np', '4', '--file',  '-rd', '/tmp'], 'training', 'datagen'),
        ("04", ['training', 'configview', '-na', '4',  '-rd', '/tmp'], 'training', 'configview'),
        
        # Checkpointing (Note: datasize lacks universal args due to how it's built in checkpointing_args.py)
        ("05", ['checkpointing', 'run', '-cm', '1024', '-m', 'llama3-8b', '-np', '4', '-cf', '/tmp/ckpt', '-rd', '/tmp', '--file'], 'checkpointing', 'run'),
        ("06", ['checkpointing', 'datasize', '-cm', '1024', '-m', 'llama3-8b'], 'checkpointing', 'datasize'),
        
        # KVCache
        ("07", ['kvcache', 'run', '-rd', '/tmp', '-m', 'llama3.1-8b', '-nu', '100'], 'kvcache', 'run'),
        ("08", ['kvcache', 'datasize', '-m', 'llama3.1-8b', '-nu', '100'], 'kvcache', 'datasize'),
        
        # VectorDB
        ("09", ['vectordb', 'run', '-rd', '/tmp', '--file'], 'vectordb', 'run'),
        ("10", ['vectordb', 'datagen', '--file',  '-rd', '/tmp'], 'vectordb', 'datagen'),
        ("11", ['vectordb', 'datasize'], 'vectordb', 'datasize'),
        
        # Utilities (All require -rd)
        ("12", ['reports', 'reportgen', '-rd', '/tmp', '--file'], 'reports', 'reportgen'),
        ("13", ['history', 'show', '-rd', '/tmp', '--file'], 'history', 'show'),
        ("14", ['lockfile', 'generate', '-rd', '/tmp', '--file'], 'lockfile', 'generate'),
        ("15", ['lockfile', 'verify', '-rd', '/tmp', '--file'], 'lockfile', 'verify'),
    ])
    def test_all_program_subcommand_combinations(self, test_name, cmd_list, expected_program, expected_command):
        """Ensure all benchmarks and subcommands can parse their minimum required arguments."""
        test_args = ['mlpstorage', 'closed'] + cmd_list
        with patch('sys.argv', test_args):
            args = parse_arguments()
            assert test_name != None and args.closed is True
            assert test_name != None and args.program == expected_program
            cmd_val = getattr(args, 'command', getattr(args, 'lockfile_command', None))
            assert test_name != None and cmd_val == expected_command

    def test_missing_required_results_dir(self):
        """Omitting -rd when req_results=True (e.g., training run) should fail."""
        test_args = ['mlpstorage', 'closed', 'training', 'run', '-cm', '1024', '-m', 'flux', '-g', 'b200', '-na', '4', '--file']
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code != 0

    def test_data_access_protocol_consolidation(self):
        """Test that --file/--object are correctly popped and consolidated into data_access_protocol."""
        test_args = ['mlpstorage', 'closed', 'training', 'datagen', '-m', 'dlrm', '-np', '4', '--object', 's3', '-rd', '/tmp']
        with patch('sys.argv', test_args):
            args = parse_arguments()
            assert args.data_access_protocol == 's3'
            assert not hasattr(args, 'file')
            assert not hasattr(args, 'object')


# =====================================================================
# 3. Validation Rules
# =====================================================================

class TestCustomValidation:

    def test_kvcache_rejects_object_storage(self):
        """KVCache validate_args should reject object storage."""
        test_args = ['mlpstorage', 'closed', 'kvcache', 'run', '-rd', '/tmp', '--object', 's3']
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code == EXIT_CODE.INVALID_ARGUMENTS

    def test_checkpointing_rejects_negative_checkpoints(self):
        """Checkpointing validate_args should reject negative checkpoint counts."""
        test_args = [
            'mlpstorage', 'closed', 'checkpointing', 'run', 
            '-cm', '1024', '-m', 'llama3-8b', '-np', '2', '-cf', '/tmp/ckpt', '-rd', '/tmp', '--file',
            '--num-checkpoints-read', '-5'
        ]
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code == EXIT_CODE.INVALID_ARGUMENTS


# =====================================================================
# 4. Post-Parse Configuration (update_args & YAML)
# =====================================================================

class TestUpdateArgsAndConfig:

    def test_update_args_normalizes_hosts(self):
        """update_args should handle messy host strings and normalize them to a clean list."""
        args = argparse.Namespace(hosts="host1, host2   host3,host4")
        update_args(args)
        assert args.hosts == ['host1', 'host2', 'host3', 'host4']
        assert args.num_client_hosts == 4

    def test_update_args_empty_hosts_fails(self):
        """update_args should exit if host normalization results in an empty list."""
        args = argparse.Namespace(hosts=" , , ")
        with pytest.raises(SystemExit) as exc_info:
            update_args(args)
        assert exc_info.value.code == EXIT_CODE.INVALID_ARGUMENTS

    def test_update_args_process_nomenclature_mapping(self):
        """update_args should unify 'num_accelerators' or 'max_accelerators' into 'num_processes'."""
        args = argparse.Namespace(num_accelerators=8)
        update_args(args)
        assert args.num_processes == 8

    def test_update_args_flattens_params(self):
        """update_args should flatten lists of lists for params/mpi_params resulting from multiple append actions."""
        args = argparse.Namespace(params=[['key=val1'], ['key=val2']])
        update_args(args)
        assert args.params == ['key=val1', 'key=val2']

    def test_yaml_config_overrides(self):
        """apply_yaml_config_overrides should update namespace attributes safely."""
        mock_yaml_content = """
        duration: 999
        hosts: "node1,node2"
        params:
            batch_size: 32
        """
        # Create a namespace simulating a parsed output
        initial_args = argparse.Namespace(
            config_file="dummy.yaml",
            duration=100,
            hosts=['default_host'],
            params=None
        )
        
        with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
            updated_args = apply_yaml_config_overrides(initial_args)
            
        assert updated_args.duration == 999
        assert updated_args.hosts == ['node1', 'node2'] # Special yaml handling for hosts
