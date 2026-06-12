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

# Import argument builders from cli package
from mlpstorage_py.cli import (
    add_training_arguments,
    add_checkpointing_arguments,
    add_vectordb_arguments,
    add_reports_arguments,
    add_history_arguments,
    add_universal_arguments,
    add_mpi_arguments,
    HELP_MESSAGES,
    PROGRAM_DESCRIPTIONS,
)
# Import parser functions from cli_parser module
from mlpstorage_py.cli_parser import (
    validate_args,
    update_args,
    apply_yaml_config_overrides,
    help_messages,
    prog_descriptions,
)
from mlpstorage_py.config import MODELS, ACCELERATORS, LLM_MODELS, EXEC_TYPE


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
        add_universal_arguments(parser, req_results=False)
        args = parser.parse_args(['--results-dir', '/test/path'])
        assert args.results_dir == '/test/path'

    def test_adds_loops_argument(self, parser):
        """Should add --loops argument when called directly on a parser."""
        # Note: --loops is now an open-gated arg added by benchmark builders,
        # not by add_universal_arguments. add_universal_arguments only adds
        # results-dir, debug, verbose, dry-run, config-file, skip-validation etc.
        # This test verifies the function accepts req_results and adds core args.
        add_universal_arguments(parser, req_results=False)
        args = parser.parse_args([])
        assert hasattr(args, 'results_dir')  # always added

    def test_adds_debug_argument(self, parser):
        """Should add --debug argument."""
        add_universal_arguments(parser, req_results=False)
        args = parser.parse_args(['--debug'])
        assert args.debug is True

    def test_adds_verbose_argument(self, parser):
        """Should add --verbose argument."""
        add_universal_arguments(parser, req_results=False)
        args = parser.parse_args(['--verbose'])
        assert args.verbose is True

    def test_adds_dry_run_argument(self, parser):
        """Should add --dry-run argument (replaces old --what-if)."""
        add_universal_arguments(parser, req_results=False)
        args = parser.parse_args(['--dry-run'])
        assert args.dry_run is True

    def test_adds_allow_invalid_params(self, parser):
        """Should add --allow-invalid-params argument."""
        # Note: --allow-invalid-params is an open-gated arg added by benchmark
        # builders, not by add_universal_arguments. Verify that add_universal_arguments
        # does not include it (closed mode shouldn't expose it via universal args).
        add_universal_arguments(parser, req_results=False)
        # The parser should succeed with no args (allow_invalid_params is not universal)
        args = parser.parse_args([])
        assert not hasattr(args, 'allow_invalid_params') or args.allow_invalid_params is False

    def test_adds_config_file_argument(self, parser):
        """Should add --config-file argument."""
        add_universal_arguments(parser, req_results=False)
        args = parser.parse_args(['--config-file', '/path/to/config.yaml'])
        assert args.config_file == '/path/to/config.yaml'

    def test_results_dir_required_when_req_results_true(self, parser):
        """Should make --results-dir required when req_results=True."""
        add_universal_arguments(parser, req_results=True)
        with pytest.raises(SystemExit):
            parser.parse_args([])  # Missing required --results-dir

    def test_results_dir_optional_when_req_results_false(self, parser):
        """Should make --results-dir optional when req_results=False."""
        add_universal_arguments(parser, req_results=False)
        # Should not raise even without --results-dir
        args = parser.parse_args([])
        assert hasattr(args, 'results_dir')


class TestAddMpiArguments:
    """Tests for add_mpi_arguments function."""

    @pytest.fixture
    def parser(self):
        """Create a basic parser."""
        return argparse.ArgumentParser()

    def test_adds_mpi_bin_argument(self, parser):
        """Should add --mpi-bin argument."""
        add_mpi_arguments(parser)
        args = parser.parse_args(['--mpi-bin', 'mpirun'])
        assert args.mpi_bin == 'mpirun'

    def test_adds_oversubscribe_argument(self, parser):
        """Should add --oversubscribe argument."""
        add_mpi_arguments(parser)
        args = parser.parse_args(['--oversubscribe'])
        assert args.oversubscribe is True

    def test_adds_allow_run_as_root_argument(self, parser):
        """Should add --allow-run-as-root argument."""
        add_mpi_arguments(parser)
        args = parser.parse_args(['--allow-run-as-root'])
        assert args.allow_run_as_root is True

    def test_adds_mpi_params_argument(self, parser):
        """Should add --mpi-params argument that accepts a single string.

        MPI flags begin with '-', so --mpi-params now takes one string value
        (use the '=' form) rather than nargs='+', which rejected dash-led
        values with "expected at least one argument" (issue #422).
        """
        add_mpi_arguments(parser)
        args = parser.parse_args(['--mpi-params=-genv FI_PROVIDER=tcp'])
        assert args.mpi_params == ['-genv FI_PROVIDER=tcp']

    def test_mpi_params_appends_multiple(self, parser):
        """Multiple --mpi-params should accumulate via action='append'."""
        add_mpi_arguments(parser)
        args = parser.parse_args(
            ['--mpi-params=-x FOO=1', '--mpi-params=-genv BAR=2']
        )
        assert args.mpi_params == ['-x FOO=1', '-genv BAR=2']


class TestAddTrainingArguments:
    """Tests for add_training_arguments function."""

    @pytest.fixture
    def parser(self):
        """Create a parser with training subcommands in open mode."""
        parser = argparse.ArgumentParser()
        add_training_arguments(parser, 'open')
        return parser

    def test_datasize_subcommand_exists(self, parser):
        """Training should have datasize subcommand."""
        args = parser.parse_args([
            'unet3d',
            'datasize',
            '--max-accelerators', '8',
            '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '128'
        ])
        assert args.command == 'datasize'
        assert args.model == 'unet3d'
        assert args.max_accelerators == 8

    def test_datagen_subcommand_exists(self, parser):
        """Training should have datagen subcommand."""
        args = parser.parse_args([
            'unet3d',
            'datagen',
            '--num-processes', '16',
            '--data-dir', '/data',
            'file'
        ])
        assert args.command == 'datagen'
        assert args.model == 'unet3d'
        assert args.num_processes == 16

    def test_run_subcommand_exists(self, parser):
        """Training should have run subcommand."""
        args = parser.parse_args([
            'unet3d',
            'run',
            '--num-accelerators', '4',
            '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '256',
            '--results-dir', '/tmp',
            'file'
        ])
        assert args.command == 'run'
        assert args.model == 'unet3d'
        assert args.num_accelerators == 4

    def test_configview_subcommand_exists(self, parser):
        """Training should have configview subcommand."""
        args = parser.parse_args([
            'unet3d',
            'configview',
            '--num-accelerators', '8',
            '--client-host-memory-in-gb', '64',
            '--accelerator-type', 'b200',
            '--results-dir', '/tmp',
            'file'
        ])
        assert args.command == 'configview'
        assert args.num_accelerators == 8

    def test_hosts_argument(self, parser):
        """Should accept --hosts argument."""
        args = parser.parse_args([
            'unet3d',
            'run',
            '--num-accelerators', '8',
            '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '128',
            '--hosts', 'host1', 'host2',
            '--results-dir', '/tmp',
            'file'
        ])
        assert args.hosts == ['host1', 'host2']

    def test_params_argument(self, parser):
        """Should accept --params argument in open mode.

        Note: 'file' positional must appear before --params because
        nargs='+' would otherwise greedily consume 'file' as a param value.
        """
        args = parser.parse_args([
            'unet3d',
            'run',
            '--num-accelerators', '8',
            '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '128',
            '--results-dir', '/tmp',
            'file',
            '--params', 'key1=val1', 'key2=val2',
        ])
        assert args.params == [['key1=val1', 'key2=val2']]

    def test_model_is_positional(self):
        """Training model should be a positional argument, not a flag."""
        parser = argparse.ArgumentParser()
        add_training_arguments(parser, 'closed')
        # Positional: model comes before subcommand
        args = parser.parse_args([
            'unet3d',
            'run',
            '--num-accelerators', '1',
            '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64',
            '--results-dir', '/tmp',
            'file'
        ])
        assert args.model == 'unet3d'

    def test_closed_mode_no_loops(self):
        """Closed mode should not expose --loops flag."""
        parser = argparse.ArgumentParser()
        add_training_arguments(parser, 'closed')
        with pytest.raises(SystemExit):
            parser.parse_args([
                'unet3d', 'run',
                '--num-accelerators', '1', '--accelerator-type', 'b200',
                '--client-host-memory-in-gb', '64', '--results-dir', '/tmp', 'file',
                '--loops', '3'
            ])

    def test_open_mode_exposes_loops(self):
        """Open mode should expose --loops flag."""
        parser = argparse.ArgumentParser()
        add_training_arguments(parser, 'open')
        args = parser.parse_args([
            'unet3d', 'run',
            '--num-accelerators', '1', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64', '--results-dir', '/tmp', 'file',
            '--loops', '3'
        ])
        assert args.loops == 3

    def test_closed_mode_namespace_has_open_defaults(self):
        """Closed-mode parse must supply loops/allow_invalid_params via set_defaults.

        `params` is now registered as a real flag in core training args (it is
        CLOSED-allowed for the parameters listed in CLOSED_ALLOWED_PARAMS — see
        issue #433), so its default is None from the flag, not '' from
        set_defaults.
        """
        parser = argparse.ArgumentParser()
        add_training_arguments(parser, 'closed')
        args = parser.parse_args([
            'unet3d', 'run',
            '--num-accelerators', '1', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64', '--results-dir', '/tmp', 'file',
        ])
        assert args.loops == 1
        assert args.params is None
        assert args.allow_invalid_params is False

    def test_closed_mode_accepts_params(self):
        """Closed mode must accept --params for CLOSED_ALLOWED_PARAMS (regression for #433)."""
        parser = argparse.ArgumentParser()
        add_training_arguments(parser, 'closed')
        args = parser.parse_args([
            'unet3d', 'datagen',
            '--num-processes', '8', '--results-dir', '/tmp', 'file',
            '--params', 'dataset.num_files_train=1000', 'dataset.num_subfolders_train=10',
        ])
        flattened = [kv for batch in (args.params or []) for kv in batch]
        assert 'dataset.num_files_train=1000' in flattened
        assert 'dataset.num_subfolders_train=10' in flattened


class TestAddCheckpointingArguments:
    """Tests for add_checkpointing_arguments function."""

    @pytest.fixture
    def parser(self):
        """Create a parser with checkpointing subcommands in open mode."""
        parser = argparse.ArgumentParser()
        add_checkpointing_arguments(parser, 'open')
        return parser

    def test_datasize_subcommand_exists(self, parser):
        """Checkpointing should have datasize subcommand."""
        args = parser.parse_args([
            'datasize',
            '--model', 'llama3-8b',
            '--num-processes', '8',
            '--client-host-memory-in-gb', '512',
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
            '--checkpoint-folder', '/ckpt',
            '--results-dir', '/tmp',
            'file'
        ])
        assert args.command == 'run'
        assert args.model == 'llama3-70b'
        assert args.num_processes == 64

    def test_num_checkpoints_read_argument(self, parser):
        """Should accept --num-checkpoints-read argument in open mode."""
        args = parser.parse_args([
            'run',
            '--model', 'llama3-8b',
            '--num-processes', '8',
            '--client-host-memory-in-gb', '512',
            '--checkpoint-folder', '/ckpt',
            '--results-dir', '/tmp',
            '--num-checkpoints-read', '5',
            'file'
        ])
        assert args.num_checkpoints_read == 5

    def test_num_checkpoints_write_argument(self, parser):
        """Should accept --num-checkpoints-write argument in open mode."""
        args = parser.parse_args([
            'run',
            '--model', 'llama3-8b',
            '--num-processes', '8',
            '--client-host-memory-in-gb', '512',
            '--checkpoint-folder', '/ckpt',
            '--results-dir', '/tmp',
            '--num-checkpoints-write', '3',
            'file'
        ])
        assert args.num_checkpoints_write == 3

    def test_open_mode_accepts_loops(self, parser):
        """Open mode should expose --loops for checkpointing run."""
        args = parser.parse_args([
            'run', '--model', 'llama3-8b', '--num-processes', '8',
            '--client-host-memory-in-gb', '512', '--checkpoint-folder', '/ckpt',
            '--results-dir', '/tmp', 'file', '--loops', '5',
        ])
        assert args.loops == 5

    def test_open_mode_accepts_params(self, parser):
        """Open mode should expose --params for checkpointing run."""
        args = parser.parse_args([
            'run', '--model', 'llama3-8b', '--num-processes', '8',
            '--client-host-memory-in-gb', '512', '--checkpoint-folder', '/ckpt',
            '--results-dir', '/tmp', 'file', '--params', 'k=v',
        ])
        assert args.params == [['k=v']]

    def test_open_mode_accepts_num_checkpoints_read(self, parser):
        """Open mode should expose --num-checkpoints-read."""
        args = parser.parse_args([
            'run', '--model', 'llama3-8b', '--num-processes', '8',
            '--client-host-memory-in-gb', '512', '--checkpoint-folder', '/ckpt',
            '--results-dir', '/tmp', '--num-checkpoints-read', '20', 'file',
        ])
        assert args.num_checkpoints_read == 20


class TestAddCheckpointingArgumentsClosed:
    """Tests for add_checkpointing_arguments in closed mode."""

    RUN_ARGS = [
        'run', '--model', 'llama3-8b', '--num-processes', '8',
        '--client-host-memory-in-gb', '512', '--checkpoint-folder', '/ckpt',
        '--results-dir', '/tmp', 'file',
    ]

    @pytest.fixture
    def parser(self):
        p = argparse.ArgumentParser()
        add_checkpointing_arguments(p, 'closed')
        return p

    def test_closed_mode_rejects_loops(self, parser):
        """Closed checkpointing must reject --loops."""
        with pytest.raises(SystemExit):
            parser.parse_args(self.RUN_ARGS + ['--loops', '3'])

    def test_closed_mode_rejects_num_checkpoints_read(self, parser):
        """Closed checkpointing must reject --num-checkpoints-read."""
        with pytest.raises(SystemExit):
            parser.parse_args(self.RUN_ARGS + ['--num-checkpoints-read', '20'])

    def test_closed_mode_rejects_num_checkpoints_write(self, parser):
        """Closed checkpointing must reject --num-checkpoints-write."""
        with pytest.raises(SystemExit):
            parser.parse_args(self.RUN_ARGS + ['--num-checkpoints-write', '20'])

    def test_closed_mode_namespace_has_open_defaults(self, parser):
        """Closed-mode parse must supply all open-gated attrs via set_defaults."""
        args = parser.parse_args(self.RUN_ARGS)
        assert args.loops == 1
        assert args.params == ''
        assert args.allow_invalid_params is False
        assert args.num_checkpoints_read == 10
        assert args.num_checkpoints_write == 10


class TestAddVectordbArguments:
    """Tests for add_vectordb_arguments function."""

    @pytest.fixture
    def parser(self):
        """Create a parser with vectordb subcommands in open mode."""
        parser = argparse.ArgumentParser()
        add_vectordb_arguments(parser, 'open')
        return parser

    def test_datagen_subcommand_exists(self, parser):
        """VectorDB should have datagen subcommand."""
        args = parser.parse_args(['datagen', '--results-dir', '/tmp', 'file'])
        assert args.command == 'datagen'

    def test_run_subcommand_exists(self, parser):
        """VectorDB should have run subcommand."""
        args = parser.parse_args(['run', '--results-dir', '/tmp', 'file'])
        assert args.command == 'run'

    def test_datagen_dimension_argument(self, parser):
        """Datagen should accept --dimension argument."""
        args = parser.parse_args(['datagen', '--dimension', '768', '--results-dir', '/tmp', 'file'])
        assert args.dimension == 768

    def test_datagen_num_vectors_argument(self, parser):
        """Datagen should accept --num-vectors argument."""
        args = parser.parse_args(['datagen', '--num-vectors', '100000', '--results-dir', '/tmp', 'file'])
        assert args.num_vectors == 100000

    def test_run_batch_size_argument(self, parser):
        """Run should accept --batch-size argument."""
        args = parser.parse_args(['run', '--batch-size', '32', '--results-dir', '/tmp', 'file'])
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
        args = parser.parse_args(['reportgen', '--results-dir', '/tmp'])
        assert args.command == 'reportgen'

    def test_output_dir_argument(self, parser):
        """Reportgen should accept --output-dir argument."""
        args = parser.parse_args(['reportgen', '--results-dir', '/tmp', '--output-dir', '/output'])
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
        args = parser.parse_args(['show', '--results-dir', '/tmp'])
        assert args.command == 'show'

    def test_show_limit_argument(self, parser):
        """Show should accept --limit argument."""
        args = parser.parse_args(['show', '--results-dir', '/tmp', '--limit', '10'])
        assert args.limit == 10

    def test_show_id_argument(self, parser):
        """Show should accept --id argument."""
        args = parser.parse_args(['show', '--results-dir', '/tmp', '--id', '5'])
        assert args.id == 5

    def test_rerun_subcommand_exists(self, parser):
        """History should have rerun subcommand."""
        args = parser.parse_args(['rerun', '42', '--results-dir', '/tmp'])
        assert args.command == 'rerun'
        assert args.rerun_id == 42


class TestValidateArgs:
    """Tests for validate_args function."""

    def test_valid_checkpointing_args(self):
        """Should not raise for valid checkpointing args."""
        args = argparse.Namespace(
            benchmark='checkpointing',
            model='llama3-8b',
            num_checkpoints_read=5,
            num_checkpoints_write=5
        )
        # Should not raise
        validate_args(args)

    def test_invalid_llm_model_exits(self):
        """Should exit for invalid LLM model."""
        args = argparse.Namespace(
            benchmark='checkpointing',
            model='invalid-model',
            num_checkpoints_read=5,
            num_checkpoints_write=5
        )
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_negative_checkpoints_read_exits(self):
        """Should exit for negative num_checkpoints_read."""
        args = argparse.Namespace(
            benchmark='checkpointing',
            model='llama3-8b',
            num_checkpoints_read=-1,
            num_checkpoints_write=5
        )
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_negative_checkpoints_write_exits(self):
        """Should exit for negative num_checkpoints_write."""
        args = argparse.Namespace(
            benchmark='checkpointing',
            model='llama3-8b',
            num_checkpoints_read=5,
            num_checkpoints_write=-1
        )
        with pytest.raises(SystemExit):
            validate_args(args)

    def test_training_args_pass_validation(self):
        """Training args should pass validation."""
        args = argparse.Namespace(
            benchmark='training',
            command='run',
            model='unet3d'
        )
        # Should not raise
        validate_args(args)

    def test_utility_args_no_benchmark(self):
        """Utility commands (no args.benchmark) should pass validate_args without error."""
        # history/reports/lockfile have mode but no benchmark
        args = argparse.Namespace(mode='history')
        # Should not raise — no benchmark means nothing to validate
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

    def test_tokenizes_mpi_params_string(self):
        """Should shlex-split each --mpi-params string into a flat token list."""
        args = argparse.Namespace(
            params=None,
            mpi_params=['-genv FI_PROVIDER=tcp', '--bind-to core']
        )
        update_args(args)
        assert args.mpi_params == [
            '-genv', 'FI_PROVIDER=tcp', '--bind-to', 'core'
        ]

    def test_mpi_params_honors_quoting(self):
        """shlex tokenization must keep inner-quoted values as one token."""
        args = argparse.Namespace(
            params=None,
            mpi_params=['-x LD_PRELOAD="/opt/my lib/foo.so"']
        )
        update_args(args)
        assert args.mpi_params == ['-x', 'LD_PRELOAD=/opt/my lib/foo.so']

    def test_flattens_mpi_params_list(self):
        """Legacy nested-list shape (old nargs='+') is still tolerated."""
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

    # -------------------------------------------------------------------
    # Regression tests for https://github.com/mlcommons/storage/issues/322
    #
    # These exercise every form of `--hosts` the CLI can plausibly receive,
    # including the forms that used to silently produce a single "host"
    # containing whitespace and then crash `ssh`.
    # -------------------------------------------------------------------

    def test_hosts_space_separated_list_unchanged(self):
        """`--hosts h1 h2 h3` -> argparse nargs='+' gives a clean list; pass through."""
        args = argparse.Namespace(
            hosts=['host1', 'host2', 'host3'],
            params=None,
            mpi_params=None,
        )
        update_args(args)
        assert args.hosts == ['host1', 'host2', 'host3']

    def test_hosts_single_quoted_space_separated_string(self):
        """`--hosts 'h1 h2 h3'` -> one token with spaces must be split (issue #322 Sample 3)."""
        args = argparse.Namespace(
            hosts=['srt017-e0 srt018-e0'],
            params=None,
            mpi_params=None,
        )
        update_args(args)
        assert args.hosts == ['srt017-e0', 'srt018-e0']

    def test_hosts_equals_quoted_space_separated(self):
        """`--hosts='h1 h2 h3'` -> same single-token-with-spaces case (issue #322 Sample 3)."""
        args = argparse.Namespace(
            hosts=['host-a host-b host-c'],
            params=None,
            mpi_params=None,
        )
        update_args(args)
        assert args.hosts == ['host-a', 'host-b', 'host-c']

    def test_hosts_mixed_comma_and_space(self):
        """Accept mixed comma/space separators in a single token."""
        args = argparse.Namespace(
            hosts=['host1, host2,host3  host4'],
            params=None,
            mpi_params=None,
        )
        update_args(args)
        assert args.hosts == ['host1', 'host2', 'host3', 'host4']

    def test_hosts_mixed_list_and_internal_split(self):
        """A list where some entries need splitting and others don't."""
        args = argparse.Namespace(
            hosts=['h1', 'h2 h3', 'h4,h5'],
            params=None,
            mpi_params=None,
        )
        update_args(args)
        assert args.hosts == ['h1', 'h2', 'h3', 'h4', 'h5']

    def test_hosts_preserves_slots_suffix(self):
        """`host:N` slot notation must survive the split."""
        args = argparse.Namespace(
            hosts=['host1:2 host2:4'],
            params=None,
            mpi_params=None,
        )
        update_args(args)
        assert args.hosts == ['host1:2', 'host2:4']

    def test_hosts_strips_stray_whitespace_and_empty_tokens(self):
        """Multiple spaces and leading/trailing whitespace don't produce empty entries."""
        args = argparse.Namespace(
            hosts=['   host1   host2 ,,, host3  '],
            params=None,
            mpi_params=None,
        )
        update_args(args)
        assert args.hosts == ['host1', 'host2', 'host3']

    def test_hosts_empty_after_parsing_exits(self):
        """An input that normalizes to zero tokens is a user error; exit cleanly."""
        args = argparse.Namespace(
            hosts=['   ,,,  '],
            params=None,
            mpi_params=None,
        )
        with pytest.raises(SystemExit):
            update_args(args)

    def test_num_client_hosts_derived_when_none(self):
        """When argparse leaves num_client_hosts=None (user didn't pass it), derive from hosts."""
        args = argparse.Namespace(
            hosts=['h1', 'h2', 'h3'],
            num_client_hosts=None,
            params=None,
            mpi_params=None,
        )
        update_args(args)
        assert args.num_client_hosts == 3

    def test_num_client_hosts_respected_when_set(self):
        """An explicit --num-client-hosts must not be overwritten."""
        args = argparse.Namespace(
            hosts=['h1', 'h2'],
            num_client_hosts=5,
            params=None,
            mpi_params=None,
        )
        update_args(args)
        assert args.num_client_hosts == 5

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

    def test_num_client_hosts_zero_is_preserved(self):
        """Regression: --num-client-hosts 0 must not be re-derived from len(hosts)."""
        args = argparse.Namespace(hosts=['h1', 'h2', 'h3'], num_client_hosts=0)
        update_args(args)
        assert args.num_client_hosts == 0

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

class TestParseArgumentsStoragePositional:
    """Regression tests for storage type as positional argument.

    After the CLI refactor, 'file' and 'object' are positional arguments
    (not --file/--object flags). This tests that:
    - Benchmark commands that require storage type parse 'file'/'object' as positionals
    - Utility commands (reports, history, lockfile) do not need storage type
    - args.data_access_protocol is set correctly; no args.file or args.object remain
    """

    @staticmethod
    def _run(monkeypatch, argv):
        """Invoke parse_arguments() with a synthetic sys.argv."""
        from mlpstorage_py.cli_parser import parse_arguments
        monkeypatch.setattr(sys, "argv", argv)
        return parse_arguments()

    # --- non-benchmark subcommands: must not require storage type ---

    def test_reportgen_does_not_need_storage_positional(self, monkeypatch, tmp_path):
        """Regression test: `reports reportgen` must parse cleanly without storage positional."""
        args = self._run(
            monkeypatch,
            ["mlpstorage", "reports", "reportgen", "--results-dir", str(tmp_path)],
        )
        assert args.mode == "reports"
        assert args.command == "reportgen"
        assert not hasattr(args, "file")
        assert not hasattr(args, "object")

    def test_history_does_not_need_storage_positional(self, monkeypatch, tmp_path):
        """`history show` must parse cleanly (no storage positional)."""
        args = self._run(monkeypatch, ["mlpstorage", "history", "show",
                                        "--results-dir", str(tmp_path)])
        assert args.mode == "history"
        assert args.command == "show"
        assert not hasattr(args, "file")
        assert not hasattr(args, "object")

    def test_lockfile_does_not_need_storage_positional(self, monkeypatch, tmp_path):
        """`lockfile generate` must parse cleanly (no storage positional)."""
        args = self._run(monkeypatch, ["mlpstorage", "lockfile", "generate",
                                        "--results-dir", str(tmp_path)])
        assert args.mode == "lockfile"
        assert not hasattr(args, "file")
        assert not hasattr(args, "object")

    # --- benchmark subcommands: 'file'/'object' as positional ---

    def test_training_run_file_positional(self, monkeypatch, tmp_path):
        """`training run <args> file` should set data_access_protocol='file'."""
        args = self._run(
            monkeypatch,
            [
                "mlpstorage", "closed", "training", "unet3d", "run",
                "--num-accelerators", "1",
                "--accelerator-type", "b200",
                "--client-host-memory-in-gb", "64",
                "--data-dir", str(tmp_path / "data"),
                "--results-dir", str(tmp_path / "results"),
                "file",
            ],
        )
        assert args.data_access_protocol == "file"
        assert not hasattr(args, "file")
        assert not hasattr(args, "object")

    def test_training_run_object_positional(self, monkeypatch, tmp_path):
        """`training run <args> object` should set data_access_protocol='object'."""
        args = self._run(
            monkeypatch,
            [
                "mlpstorage", "closed", "training", "unet3d", "run",
                "--num-accelerators", "1",
                "--accelerator-type", "b200",
                "--client-host-memory-in-gb", "64",
                "--data-dir", str(tmp_path / "data"),
                "--results-dir", str(tmp_path / "results"),
                "object",
            ],
        )
        assert args.data_access_protocol == "object"
        assert not hasattr(args, "file")
        assert not hasattr(args, "object")
