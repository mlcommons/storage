"""
Tests for CLI argument parsing in mlpstorage.cli module.

Tests cover:
- Training command argument parsing
- Checkpointing command argument parsing
- VectorDB command argument parsing
- Reports command argument parsing
- History command argument parsing
- Argument validation
- YAML config file overrides
"""

import argparse
import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from mlpstorage.cli import (
    add_training_arguments,
    add_checkpointing_arguments,
    add_vectordb_arguments,
    add_reports_arguments,
    add_history_arguments,
    add_universal_arguments,
    add_mpi_group,
    validate_args,
    update_args,
    apply_yaml_config_overrides,
    help_messages,
    prog_descriptions,
)
from mlpstorage.config import MODELS, ACCELERATORS, LLM_MODELS, EXEC_TYPE


class TestHelpMessages:
    """Tests for help message dictionary."""

    def test_help_messages_is_dict(self):
        """help_messages should be a dictionary."""
        assert isinstance(help_messages, dict)

    def test_help_messages_has_required_keys(self):
        """help_messages should have required keys."""
        required_keys = ['model', 'accelerator_type', 'results_dir', 'params']
        for key in required_keys:
            assert key in help_messages

    def test_prog_descriptions_has_benchmark_types(self):
        """prog_descriptions should have all benchmark types."""
        assert 'training' in prog_descriptions
        assert 'checkpointing' in prog_descriptions
        assert 'vectordb' in prog_descriptions


class TestAddUniversalArguments:
    """Tests for add_universal_arguments function."""

    @pytest.fixture
    def parser(self):
        """Create a basic parser."""
        return argparse.ArgumentParser()

    def test_adds_results_dir_argument(self, parser):
        """Should add --results-dir argument."""
        add_universal_arguments(parser)
        args = parser.parse_args(['--results-dir', '/test/path'])
        assert args.results_dir == '/test/path'

    def test_adds_loops_argument(self, parser):
        """Should add --loops argument."""
        add_universal_arguments(parser)
        args = parser.parse_args(['--loops', '5'])
        assert args.loops == 5

    def test_adds_debug_argument(self, parser):
        """Should add --debug argument."""
        add_universal_arguments(parser)
        args = parser.parse_args(['--debug'])
        assert args.debug is True

    def test_adds_verbose_argument(self, parser):
        """Should add --verbose argument."""
        add_universal_arguments(parser)
        args = parser.parse_args(['--verbose'])
        assert args.verbose is True

    def test_adds_what_if_argument(self, parser):
        """Should add --what-if argument."""
        add_universal_arguments(parser)
        args = parser.parse_args(['--what-if'])
        assert args.what_if is True

    def test_closed_open_mutually_exclusive(self, parser):
        """--closed and --open should be mutually exclusive."""
        add_universal_arguments(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(['--closed', '--open'])

    def test_adds_allow_invalid_params(self, parser):
        """Should add --allow-invalid-params argument."""
        add_universal_arguments(parser)
        args = parser.parse_args(['--allow-invalid-params'])
        assert args.allow_invalid_params is True

    def test_adds_config_file_argument(self, parser):
        """Should add --config-file argument."""
        add_universal_arguments(parser)
        args = parser.parse_args(['--config-file', '/path/to/config.yaml'])
        assert args.config_file == '/path/to/config.yaml'


class TestAddMpiGroup:
    """Tests for add_mpi_group function."""

    @pytest.fixture
    def parser(self):
        """Create a basic parser."""
        return argparse.ArgumentParser()

    def test_adds_mpi_bin_argument(self, parser):
        """Should add --mpi-bin argument."""
        add_mpi_group(parser)
        args = parser.parse_args(['--mpi-bin', 'mpirun'])
        assert args.mpi_bin == 'mpirun'

    def test_adds_oversubscribe_argument(self, parser):
        """Should add --oversubscribe argument."""
        add_mpi_group(parser)
        args = parser.parse_args(['--oversubscribe'])
        assert args.oversubscribe is True

    def test_adds_allow_run_as_root_argument(self, parser):
        """Should add --allow-run-as-root argument."""
        add_mpi_group(parser)
        args = parser.parse_args(['--allow-run-as-root'])
        assert args.allow_run_as_root is True

    def test_adds_mpi_params_argument(self, parser):
        """Should add --mpi-params argument."""
        add_mpi_group(parser)
        args = parser.parse_args(['--mpi-params', 'param1', 'param2'])
        assert args.mpi_params == [['param1', 'param2']]


class TestAddTrainingArguments:
    """Tests for add_training_arguments function."""

    @pytest.fixture
    def parser(self):
        """Create a parser with training subcommands."""
        parser = argparse.ArgumentParser()
        add_training_arguments(parser)
        return parser

    def test_datasize_subcommand_exists(self, parser):
        """Training should have datasize subcommand."""
        args = parser.parse_args([
            'datasize',
            '--model', 'unet3d',
            '--max-accelerators', '8',
            '--accelerator-type', 'h100',
            '--client-host-memory-in-gb', '128'
        ])
        assert args.command == 'datasize'
        assert args.model == 'unet3d'
        assert args.max_accelerators == 8

    def test_datagen_subcommand_exists(self, parser):
        """Training should have datagen subcommand."""
        args = parser.parse_args([
            'datagen',
            '--model', 'resnet50',
            '--num-processes', '16',
            '--data-dir', '/data'
        ])
        assert args.command == 'datagen'
        assert args.model == 'resnet50'
        assert args.num_processes == 16

    def test_run_subcommand_exists(self, parser):
        """Training should have run subcommand."""
        args = parser.parse_args([
            'run',
            '--model', 'cosmoflow',
            '--num-accelerators', '4',
            '--accelerator-type', 'a100',
            '--client-host-memory-in-gb', '256'
        ])
        assert args.command == 'run'
        assert args.model == 'cosmoflow'
        assert args.num_accelerators == 4

    def test_configview_subcommand_exists(self, parser):
        """Training should have configview subcommand."""
        # Note: configview only has --num-accelerators, not --model
        args = parser.parse_args([
            'configview',
            '--num-accelerators', '8'
        ])
        assert args.command == 'configview'
        assert args.num_accelerators == 8

    def test_hosts_argument(self, parser):
        """Should accept --hosts argument."""
        args = parser.parse_args([
            'run',
            '--model', 'unet3d',
            '--num-accelerators', '8',
            '--accelerator-type', 'h100',
            '--client-host-memory-in-gb', '128',
            '--hosts', 'host1', 'host2'
        ])
        assert args.hosts == ['host1', 'host2']

    def test_params_argument(self, parser):
        """Should accept --params argument."""
        args = parser.parse_args([
            'run',
            '--model', 'unet3d',
            '--num-accelerators', '8',
            '--accelerator-type', 'h100',
            '--client-host-memory-in-gb', '128',
            '--params', 'key1=val1', 'key2=val2'
        ])
        assert args.params == [['key1=val1', 'key2=val2']]


class TestAddCheckpointingArguments:
    """Tests for add_checkpointing_arguments function."""

    @pytest.fixture
    def parser(self):
        """Create a parser with checkpointing subcommands."""
        parser = argparse.ArgumentParser()
        add_checkpointing_arguments(parser)
        return parser

    def test_datasize_subcommand_exists(self, parser):
        """Checkpointing should have datasize subcommand."""
        args = parser.parse_args([
            'datasize',
            '--model', 'llama3-8b',
            '--num-processes', '8',
            '--client-host-memory-in-gb', '512',
            '--checkpoint-folder', '/ckpt'
        ])
        assert args.command == 'datasize'
        assert args.model == 'llama3-8b'

    def test_run_subcommand_exists(self, parser):
        """Checkpointing should have run subcommand."""
        args = parser.parse_args([
            'run',
            '--model', 'llama3-70b',
            '--num-processes', '64',
            '--client-host-memory-in-gb', '1024',
            '--checkpoint-folder', '/ckpt'
        ])
        assert args.command == 'run'
        assert args.model == 'llama3-70b'
        assert args.num_processes == 64

    def test_num_checkpoints_read_argument(self, parser):
        """Should accept --num-checkpoints-read argument."""
        args = parser.parse_args([
            'run',
            '--model', 'llama3-8b',
            '--num-processes', '8',
            '--client-host-memory-in-gb', '512',
            '--checkpoint-folder', '/ckpt',
            '--num-checkpoints-read', '5'
        ])
        assert args.num_checkpoints_read == 5

    def test_num_checkpoints_write_argument(self, parser):
        """Should accept --num-checkpoints-write argument."""
        args = parser.parse_args([
            'run',
            '--model', 'llama3-8b',
            '--num-processes', '8',
            '--client-host-memory-in-gb', '512',
            '--checkpoint-folder', '/ckpt',
            '--num-checkpoints-write', '3'
        ])
        assert args.num_checkpoints_write == 3


class TestAddVectordbArguments:
    """Tests for add_vectordb_arguments function."""

    @pytest.fixture
    def parser(self):
        """Create a parser with vectordb subcommands."""
        parser = argparse.ArgumentParser()
        add_vectordb_arguments(parser)
        return parser

    def test_datagen_subcommand_exists(self, parser):
        """VectorDB should have datagen subcommand."""
        args = parser.parse_args(['datagen'])
        assert args.command == 'datagen'

    def test_run_search_subcommand_exists(self, parser):
        """VectorDB should have run-search subcommand."""
        args = parser.parse_args(['run-search'])
        assert args.command == 'run-search'

    def test_datagen_dimension_argument(self, parser):
        """Datagen should accept --dimension argument."""
        args = parser.parse_args(['datagen', '--dimension', '768'])
        assert args.dimension == 768

    def test_datagen_num_vectors_argument(self, parser):
        """Datagen should accept --num-vectors argument."""
        args = parser.parse_args(['datagen', '--num-vectors', '100000'])
        assert args.num_vectors == 100000

    def test_run_search_batch_size_argument(self, parser):
        """Run-search should accept --batch-size argument."""
        args = parser.parse_args(['run-search', '--batch-size', '32'])
        assert args.batch_size == 32


class TestAddReportsArguments:
    """Tests for add_reports_arguments function."""

    @pytest.fixture
    def parser(self):
        """Create a parser with reports subcommands."""
        parser = argparse.ArgumentParser()
        add_reports_arguments(parser)
        return parser

    def test_reportgen_subcommand_exists(self, parser):
        """Reports should have reportgen subcommand."""
        args = parser.parse_args(['reportgen'])
        assert args.command == 'reportgen'

    def test_output_dir_argument(self, parser):
        """Reportgen should accept --output-dir argument."""
        args = parser.parse_args(['reportgen', '--output-dir', '/output'])
        assert args.output_dir == '/output'


class TestAddHistoryArguments:
    """Tests for add_history_arguments function."""

    @pytest.fixture
    def parser(self):
        """Create a parser with history subcommands."""
        parser = argparse.ArgumentParser()
        add_history_arguments(parser)
        return parser

    def test_show_subcommand_exists(self, parser):
        """History should have show subcommand."""
        args = parser.parse_args(['show'])
        assert args.command == 'show'

    def test_show_limit_argument(self, parser):
        """Show should accept --limit argument."""
        args = parser.parse_args(['show', '--limit', '10'])
        assert args.limit == 10

    def test_show_id_argument(self, parser):
        """Show should accept --id argument."""
        args = parser.parse_args(['show', '--id', '5'])
        assert args.id == 5

    def test_rerun_subcommand_exists(self, parser):
        """History should have rerun subcommand."""
        args = parser.parse_args(['rerun', '42'])
        assert args.command == 'rerun'
        assert args.rerun_id == 42


class TestValidateArgs:
    """Tests for validate_args function."""

    def test_valid_checkpointing_args(self):
        """Should not raise for valid checkpointing args."""
        args = argparse.Namespace(
            program='checkpointing',
            model='llama3-8b',
            num_checkpoints_read=5,
            num_checkpoints_write=5
        )
        # Should not raise
        validate_args(args)

    def test_invalid_llm_model_exits(self):
        """Should exit for invalid LLM model."""
        args = argparse.Namespace(
            program='checkpointing',
            model='invalid-model',
            num_checkpoints_read=5,
            num_checkpoints_write=5
        )
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_negative_checkpoints_read_exits(self):
        """Should exit for negative num_checkpoints_read."""
        args = argparse.Namespace(
            program='checkpointing',
            model='llama3-8b',
            num_checkpoints_read=-1,
            num_checkpoints_write=5
        )
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_negative_checkpoints_write_exits(self):
        """Should exit for negative num_checkpoints_write."""
        args = argparse.Namespace(
            program='checkpointing',
            model='llama3-8b',
            num_checkpoints_read=5,
            num_checkpoints_write=-1
        )
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_training_args_pass_validation(self):
        """Training args should pass validation."""
        args = argparse.Namespace(
            program='training',
            command='run',
            model='unet3d'
        )
        # Should not raise
        validate_args(args)


class TestUpdateArgs:
    """Tests for update_args function."""

    def test_sets_num_processes_from_num_accelerators(self):
        """Should set num_processes from num_accelerators."""
        args = argparse.Namespace(num_accelerators=16, params=None, mpi_params=None)
        update_args(args)
        assert args.num_processes == 16

    def test_sets_num_processes_from_max_accelerators(self):
        """Should set num_processes from max_accelerators."""
        args = argparse.Namespace(max_accelerators=32, params=None, mpi_params=None)
        update_args(args)
        assert args.num_processes == 32

    def test_flattens_params_list(self):
        """Should flatten nested params list."""
        args = argparse.Namespace(
            params=[['key1=val1', 'key2=val2'], ['key3=val3']],
            mpi_params=None
        )
        update_args(args)
        assert args.params == ['key1=val1', 'key2=val2', 'key3=val3']

    def test_flattens_mpi_params_list(self):
        """Should flatten nested mpi_params list."""
        args = argparse.Namespace(
            params=None,
            mpi_params=[['--bind-to', 'core'], ['--map-by', 'socket']]
        )
        update_args(args)
        assert args.mpi_params == ['--bind-to', 'core', '--map-by', 'socket']

    def test_splits_comma_separated_hosts(self):
        """Should split comma-separated hosts string."""
        args = argparse.Namespace(
            hosts=['host1,host2,host3'],
            params=None,
            mpi_params=None
        )
        update_args(args)
        assert args.hosts == ['host1', 'host2', 'host3']

    def test_sets_num_client_hosts_from_hosts(self):
        """Should set num_client_hosts from hosts length."""
        args = argparse.Namespace(
            hosts=['host1', 'host2', 'host3'],
            params=None,
            mpi_params=None
        )
        update_args(args)
        assert args.num_client_hosts == 3

    def test_sets_default_runtime_for_vectordb(self):
        """Should set default runtime for vectordb when not specified."""
        args = argparse.Namespace(
            runtime=None,
            queries=None,
            params=None,
            mpi_params=None
        )
        update_args(args)
        assert args.runtime is not None


class TestApplyYamlConfigOverrides:
    """Tests for apply_yaml_config_overrides function."""

    def test_applies_simple_overrides(self, tmp_path):
        """Should apply simple overrides from YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("debug: true\nloops: 5")

        args = argparse.Namespace(
            config_file=str(config_file),
            debug=False,
            loops=1
        )
        result = apply_yaml_config_overrides(args)
        assert result.debug is True
        assert result.loops == 5

    def test_skips_unknown_params(self, tmp_path, capsys):
        """Should skip unknown parameters with warning."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("unknown_param: value\ndebug: true")

        args = argparse.Namespace(
            config_file=str(config_file),
            debug=False
        )
        result = apply_yaml_config_overrides(args)
        captured = capsys.readouterr()
        assert "unknown parameter" in captured.out.lower()
        assert result.debug is True

    def test_handles_empty_config(self, tmp_path, capsys):
        """Should handle empty config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        args = argparse.Namespace(
            config_file=str(config_file),
            debug=False
        )
        result = apply_yaml_config_overrides(args)
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()
        assert result.debug is False

    def test_converts_hosts_string_to_list(self, tmp_path):
        """Should convert comma-separated hosts to list."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("hosts: 'host1,host2,host3'")

        args = argparse.Namespace(
            config_file=str(config_file),
            hosts=['localhost']
        )
        result = apply_yaml_config_overrides(args)
        assert result.hosts == ['host1', 'host2', 'host3']

    def test_converts_params_dict_to_list(self, tmp_path):
        """Should convert params dict to list of key=value strings."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("params:\n  key1: val1\n  key2: val2")

        args = argparse.Namespace(
            config_file=str(config_file),
            params=[]
        )
        result = apply_yaml_config_overrides(args)
        assert 'key1=val1' in result.params
        assert 'key2=val2' in result.params

    def test_exits_on_file_not_found(self, tmp_path):
        """Should exit when config file not found."""
        args = argparse.Namespace(
            config_file='/nonexistent/config.yaml'
        )
        with pytest.raises(SystemExit):
            apply_yaml_config_overrides(args)

    def test_exits_on_invalid_yaml(self, tmp_path):
        """Should exit on invalid YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: syntax: :")

        args = argparse.Namespace(
            config_file=str(config_file)
        )
        with pytest.raises(SystemExit):
            apply_yaml_config_overrides(args)

    def test_skips_none_values(self, tmp_path):
        """Should skip None values in YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("debug: null\nloops: 5")

        args = argparse.Namespace(
            config_file=str(config_file),
            debug=True,
            loops=1
        )
        result = apply_yaml_config_overrides(args)
        assert result.debug is True  # Should not be overwritten
        assert result.loops == 5
