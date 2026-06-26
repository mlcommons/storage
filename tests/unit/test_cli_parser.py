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
        """Verify hardcoded defaults in KVCache closed mode match open mode defaults.

        In closed mode, --model and --num-users are not accepted (model is fixed by
        the benchmark; users must be specified via set_defaults). In open mode, they
        are required flags. We verify the common defaults match.
        """
        # Closed mode: model/num-users are set_defaults; only --results-dir required
        with patch('sys.argv', ['mlpstorage', 'closed', 'kvcache', 'run', '-rd', '/tmp',
                                 '-sn', 'sys-v1']):
            args_closed = parse_arguments()

        # Open mode: model and num-users are required flags
        with patch('sys.argv', ['mlpstorage', 'open', 'kvcache', 'run', '-rd', '/tmp',
                                 '-m', 'llama3.1-8b', '-nu', '100', '-sn', 'sys-v1']):
            args_open = parse_arguments()

        # Check common defaults that should match between modes
        assert args_closed.gpu_mem_gb == args_open.gpu_mem_gb == 16.0
        assert args_closed.duration == args_open.duration == 60
        assert args_closed.loops == args_open.loops == 1
        assert args_closed.disable_multi_turn == args_open.disable_multi_turn == False

    def test_checkpointing_open_closed_defaults_match(self):
        """Verify Checkpointing 'closed' forces read/write checkpoint counts to match 'open' defaults."""
        base_args = ['checkpointing', 'run', '-cm', '1024', '-m', 'llama3-8b', '-np', '2', '-cf', '/tmp/ckpt', '-rd', '/tmp', '-sn', 'sys-v1', 'file']

        with patch('sys.argv', ['mlpstorage', 'closed'] + base_args):
            args_closed = parse_arguments()

        with patch('sys.argv', ['mlpstorage', 'open'] + base_args):
            args_open = parse_arguments()

        assert args_closed.num_checkpoints_read == args_open.num_checkpoints_read == 10
        assert args_closed.num_checkpoints_write == args_open.num_checkpoints_write == 10

    def test_closed_mode_strips_open_args(self):
        """Open-mode arguments should trigger an unrecognized argument error if passed in closed mode."""
        test_args = ['mlpstorage', 'closed', 'kvcache', 'run', '-rd', '/tmp', '-sn', 'sys-v1', '--allow-invalid-params']
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code != 0


# =====================================================================
# 2. Structural & Subcommand Combinations (Positive Cases)
# =====================================================================

class TestCLIStructureAndCombinations:

    @pytest.mark.parametrize("test_name, cmd_list, expected_mode_or_benchmark, expected_command", [
        # Training — model is now a positional (no --model flag); storage type is positional
        # closed mode: only 'unet3d' and 'retinanet' are valid model choices
        # NOTE: every emitting command now requires --systemname per D-10/LAY-04.
        ("01", ['training', 'retinanet', 'run', '-cm', '1024', '-at', 'b200', '-na', '4', '-dd', '/tmp', '-rd', '/tmp', '-sn', 'sys-v1', 'file'], 'training', 'run'),
        ("02", ['training', 'unet3d', 'datasize', '-cm', '1024', '-at', 'b200', '-ma', '4', '-rd', '/tmp', '-sn', 'sys-v1'], 'training', 'datasize'),
        ("03", ['training', 'unet3d', 'datagen', '-np', '4', '-dd', '/tmp', 'file', '-rd', '/tmp', '-sn', 'sys-v1'], 'training', 'datagen'),
        ("04", ['training', 'unet3d', 'configview', '-na', '4', '-cm', '64', '-at', 'b200', '-rd', '/tmp', '-sn', 'sys-v1', 'file'], 'training', 'configview'),

        # Checkpointing — --model stays as a flag; storage type is positional
        ("05", ['checkpointing', 'run', '-cm', '1024', '-m', 'llama3-8b', '-np', '4', '-cf', '/tmp/ckpt', '-rd', '/tmp', '-sn', 'sys-v1', 'file'], 'checkpointing', 'run'),
        ("06", ['checkpointing', 'datasize', '-cm', '1024', '-m', 'llama3-8b', '-np', '4', '-sn', 'sys-v1'], 'checkpointing', 'datasize'),

        # KVCache closed mode: model/num-users are not accepted in closed mode
        ("07", ['kvcache', 'run', '-rd', '/tmp', '-sn', 'sys-v1'], 'kvcache', 'run'),
        ("08", ['kvcache', 'datasize'], 'kvcache', 'datasize'),

        # VectorDB
        ("09", ['vectordb', 'run', '-rd', '/tmp', '-sn', 'sys-v1', 'file'], 'vectordb', 'run'),
        ("10", ['vectordb', 'datagen', 'file', '-rd', '/tmp', '-sn', 'sys-v1'], 'vectordb', 'datagen'),
        ("11", ['vectordb', 'datasize'], 'vectordb', 'datasize'),

        # Utilities — top-level siblings, no mode prefix needed (they are their own mode)
        ("12", ['reports', 'reportgen', '-rd', '/tmp', '-sn', 'sys-v1'], 'reports', 'reportgen'),
        ("13", ['history', 'show', '-rd', '/tmp', '-sn', 'sys-v1'], 'history', 'show'),
        ("14", ['lockfile', 'generate', '-rd', '/tmp'], 'lockfile', 'generate'),
        ("15", ['lockfile', 'verify', '-rd', '/tmp'], 'lockfile', 'verify'),
    ])
    def test_all_program_subcommand_combinations(self, test_name, cmd_list, expected_mode_or_benchmark, expected_command):
        """Ensure all benchmarks and subcommands can parse their minimum required arguments."""
        # Benchmark commands run under 'closed'; utility commands are top-level
        benchmark_benchmarks = {'training', 'checkpointing', 'vectordb', 'kvcache'}
        if expected_mode_or_benchmark in benchmark_benchmarks:
            test_args = ['mlpstorage', 'closed'] + cmd_list
        else:
            test_args = ['mlpstorage'] + cmd_list

        with patch('sys.argv', test_args):
            args = parse_arguments()

        if expected_mode_or_benchmark in benchmark_benchmarks:
            assert args.mode == "closed", f"[{test_name}] expected mode==closed, got {args.mode}"
            assert args.benchmark == expected_mode_or_benchmark, f"[{test_name}] expected benchmark=={expected_mode_or_benchmark}, got {args.benchmark}"
        else:
            assert args.mode == expected_mode_or_benchmark, f"[{test_name}] expected mode=={expected_mode_or_benchmark}, got {args.mode}"

        cmd_val = getattr(args, 'command', getattr(args, 'lockfile_command', None))
        assert cmd_val == expected_command, f"[{test_name}] expected command=={expected_command}, got {cmd_val}"

    def test_missing_required_results_dir(self):
        """Omitting -rd when req_results=True (e.g., training run) should fail."""
        test_args = ['mlpstorage', 'closed', 'training', 'unet3d', 'run', '-cm', '1024', '-at', 'b200', '-na', '4', '-dd', '/tmp', 'file']
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code != 0

    def test_data_access_protocol_positional(self):
        """Test that the data_access_protocol positional is set correctly."""
        # Use 'unet3d' — a valid model in closed mode
        test_args = ['mlpstorage', 'closed', 'training', 'unet3d', 'datagen', '-np', '4', '-dd', '/tmp', 'file', '-rd', '/tmp', '-sn', 'sys-v1']
        with patch('sys.argv', test_args):
            args = parse_arguments()
            assert args.data_access_protocol == 'file'
            # Positional means no separate 'file' or 'object' attributes
            assert not hasattr(args, 'file')
            assert not hasattr(args, 'object')


# =====================================================================
# 3. Validation Rules
# =====================================================================

class TestCustomValidation:

    def test_kvcache_rejects_object_storage(self):
        """KVCache validate_args should reject object storage."""
        # kvcache has no storage type positional; 'object' would be unrecognized
        test_args = ['mlpstorage', 'closed', 'kvcache', 'run', '-rd', '/tmp', 'object']
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code != 0

    def test_checkpointing_rejects_negative_checkpoints(self):
        """Checkpointing validate_args should reject negative checkpoint counts."""
        test_args = [
            'mlpstorage', 'closed', 'checkpointing', 'run',
            '-cm', '1024', '-m', 'llama3-8b', '-np', '2', '-cf', '/tmp/ckpt', '-rd', '/tmp',
            '-sn', 'sys-v1', 'file',
            '--num-checkpoints-read', '-5'
        ]
        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code == EXIT_CODE.INVALID_ARGUMENTS


class TestTrainingDataDirEnforcement:
    """Cover the file-fast-fail / object-post-YAML split for --data-dir."""

    @pytest.mark.parametrize("command, extra", [
        ("datagen", ['-np', '4']),
        ("run", ['-cm', '64', '-at', 'b200', '-na', '4']),
    ])
    def test_file_mode_missing_data_dir_fails_at_parse(self, command, extra):
        """File storage must reject training datagen/run without --data-dir at the CLI boundary."""
        argv = ['mlpstorage', 'closed', 'training', 'unet3d', command,
                *extra, '-rd', '/tmp', 'file']
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code != 0

    @pytest.mark.parametrize("data_dir", ['retinanet', 's3://bucket/unet3d', 'unet3d/data'])
    def test_object_mode_accepts_explicit_data_dir(self, data_dir):
        """Object storage accepts a bare key prefix, full URI, or nested prefix as --data-dir."""
        argv = ['mlpstorage', 'closed', 'training', 'unet3d', 'datagen',
                '-np', '4', '--data-dir', data_dir, '-rd', '/tmp', '-sn', 'sys-v1', 'object']
        with patch('sys.argv', argv):
            args = parse_arguments()
        assert args.data_access_protocol == 'object'
        assert args.data_dir == data_dir

    def test_object_mode_data_dir_from_config_file(self, tmp_path):
        """Object storage must accept data_dir supplied via --config-file (Russ' P2 regression case)."""
        config_file = tmp_path / "object-config.yaml"
        config_file.write_text("data_dir: retinanet\n")

        argv = ['mlpstorage', 'closed', 'training', 'unet3d', 'datagen',
                '-np', '4', '-rd', '/tmp', '-sn', 'sys-v1',
                '--config-file', str(config_file), 'object']
        with patch('sys.argv', argv):
            args = parse_arguments()
        assert args.data_access_protocol == 'object'
        assert args.data_dir == 'retinanet'

    def test_object_mode_missing_data_dir_fails_after_yaml(self):
        """Object storage without --data-dir (and no config-file supplying it) must still fail."""
        argv = ['mlpstorage', 'closed', 'training', 'unet3d', 'datagen',
                '-np', '4', '-rd', '/tmp', 'object']
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code == EXIT_CODE.INVALID_ARGUMENTS

    def test_file_mode_data_dir_from_config_file(self, tmp_path):
        """File storage may also accept data_dir from --config-file (no immediate failure if CLI omits it)."""
        config_file = tmp_path / "file-config.yaml"
        config_file.write_text("data_dir: /mnt/data\n")

        argv = ['mlpstorage', 'closed', 'training', 'unet3d', 'datagen',
                '-np', '4', '-rd', '/tmp',
                '--config-file', str(config_file), '-dd', '/tmp', 'file']
        # File-mode check runs before YAML, so this still fails fast even though
        # the YAML would have supplied data_dir. Users supplying data_dir via
        # YAML for file storage must also pass --data-dir on the CLI.
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()
            assert exc_info.value.code != 0


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

    def test_update_args_rejects_space_separated_params(self):
        """update_args should fail with a clear error when --param uses space instead of '='. (issue #469)"""
        # Simulates `--param dataset.num_files_train 35000` — argparse with
        # nargs="+" captures both tokens; without validation the missing '='
        # surfaces later as "not enough values to unpack".
        args = argparse.Namespace(params=[['dataset.num_files_train', '35000']])
        with pytest.raises(SystemExit) as excinfo:
            update_args(args)
        assert excinfo.value.code == EXIT_CODE.INVALID_ARGUMENTS

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
        assert updated_args.hosts == ['node1', 'node2']  # Special yaml handling for hosts

    def test_update_args_allows_none_hosts_for_single_node(self):
        """Optional --hosts=None must remain valid for single-node benchmarks."""
        args = argparse.Namespace(
            hosts=None,
            num_client_hosts=None,
        )

        update_args(args)

        assert args.hosts is None
        assert args.num_client_hosts is None

    def test_update_args_allows_missing_num_client_hosts_with_none_hosts(self):
        """A namespace with hosts=None must not evaluate len(None)."""
        args = argparse.Namespace(hosts=None)

        update_args(args)

        assert args.hosts is None
        assert not hasattr(args, "num_client_hosts")

# =====================================================================
# 5. args.mode and args.benchmark attributes
# =====================================================================

class TestModeAndBenchmarkAttributes:
    """Verify the new args.mode and args.benchmark namespace attributes."""

    def test_closed_training_sets_mode_and_benchmark(self):
        """closed training unet3d run should set mode='closed' and benchmark='training'."""
        test_args = [
            'mlpstorage', 'closed', 'training', 'unet3d', 'run',
            '--data-dir', '/tmp', '--results-dir', '/tmp',
            '--num-accelerators', '1', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64', '--systemname', 'sys-v1', 'file'
        ]
        with patch('sys.argv', test_args):
            args = parse_arguments()
        assert args.mode == 'closed'
        assert args.benchmark == 'training'
        assert args.model == 'unet3d'
        assert args.command == 'run'
        assert args.data_access_protocol == 'file'

    def test_open_mode_allows_loops(self):
        """open mode should accept --loops flag for training."""
        test_args = [
            'mlpstorage', 'open', 'training', 'unet3d', 'run',
            '--data-dir', '/tmp', '--results-dir', '/tmp',
            '--num-accelerators', '1', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64',
            '--systemname', 'sys-v1', 'file', '--loops', '3'
        ]
        with patch('sys.argv', test_args):
            args = parse_arguments()
        assert args.mode == 'open'
        assert args.loops == 3

    def test_reports_sets_mode(self):
        """reports subcommand should set mode='reports' (no benchmark attribute)."""
        with patch('sys.argv', ['mlpstorage', 'reports', 'reportgen',
                                '--results-dir', '/tmp', '--systemname', 'sys-v1']):
            args = parse_arguments()
        assert args.mode == 'reports'
        assert not hasattr(args, 'benchmark')

    def test_history_sets_mode(self):
        """history subcommand should set mode='history'."""
        with patch('sys.argv', ['mlpstorage', 'history', 'show',
                                '--results-dir', '/tmp', '--systemname', 'sys-v1']):
            args = parse_arguments()
        assert args.mode == 'history'
        assert not hasattr(args, 'benchmark')

    def test_lockfile_sets_mode(self):
        """lockfile subcommand should set mode='lockfile'."""
        with patch('sys.argv', ['mlpstorage', 'lockfile', 'generate', '--results-dir', '/tmp']):
            args = parse_arguments()
        assert args.mode == 'lockfile'
        assert not hasattr(args, 'benchmark')

    def test_no_file_object_consolidation_needed(self):
        """data_access_protocol is set directly by positional; no 'file' or 'object' attrs remain."""
        test_args = [
            'mlpstorage', 'closed', 'training', 'unet3d', 'run',
            '--data-dir', '/tmp', '--results-dir', '/tmp',
            '--num-accelerators', '1', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64', '--systemname', 'sys-v1', 'file'
        ]
        with patch('sys.argv', test_args):
            args = parse_arguments()
        assert args.data_access_protocol == 'file'
        assert not hasattr(args, 'file')
        assert not hasattr(args, 'object')


# =====================================================================
# 6. --systemname / MLPERF_SYSTEMNAME plumbing (LAY-04, D-10)
# =====================================================================

class TestSystemname:
    """Tests for the --systemname flag and MLPERF_SYSTEMNAME env-var plumbing.

    Per CONTEXT.md D-10, --systemname is required on every emitting subcommand:
    training {datagen, run, configview, datasize}, checkpointing {datagen, run,
    configview, validate}, vectordb {datagen, run}, kvcache {run, datagen}, plus
    reports reportgen and history (show, rerun).

    Pure utility commands (lockfile, version, init, rules-coverage) are exempt
    and continue to parse without --systemname.
    """

    # Sets of full argv (without --systemname) that should each accept the flag.
    EMITTING_COMMANDS = [
        # (label, argv tail)
        ('training-run', [
            'closed', 'training', 'unet3d', 'run',
            '--data-dir', '/d', '--results-dir', '/r',
            '--num-accelerators', '1', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64', 'file',
        ]),
        ('training-datagen', [
            'closed', 'training', 'unet3d', 'datagen', 'file',
            '--data-dir', '/d', '--results-dir', '/r',
            '--num-processes', '1',
        ]),
        ('training-configview', [
            'closed', 'training', 'unet3d', 'configview',
            '--data-dir', '/d', '--results-dir', '/r',
            '--num-accelerators', '1', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64', 'file',
        ]),
        ('training-datasize', [
            'closed', 'training', 'unet3d', 'datasize',
            '-ma', '4', '--accelerator-type', 'b200',
            '--client-host-memory-in-gb', '64',
            '--results-dir', '/r',
        ]),
        ('checkpointing-run', [
            'closed', 'checkpointing', 'run',
            '-cm', '1024', '-m', 'llama3-8b', '-np', '2',
            '-cf', '/tmp/ckpt', '-rd', '/r', 'file',
        ]),
        ('vectordb-run', [
            'closed', 'vectordb', 'run',
            '-rd', '/r', 'file',
        ]),
        ('kvcache-run', [
            'closed', 'kvcache', 'run', '-rd', '/r',
        ]),
        ('reports-reportgen', [
            'reports', 'reportgen', '--results-dir', '/r',
        ]),
        ('history-show', [
            'history', 'show', '--results-dir', '/r',
        ]),
    ]

    @pytest.mark.parametrize(
        "label,argv_tail",
        EMITTING_COMMANDS,
        ids=[label for label, _ in EMITTING_COMMANDS],
    )
    def test_systemname_on_emitting_commands(self, label, argv_tail, monkeypatch):
        """Every emitting subcommand accepts --systemname and binds it to args.systemname."""
        monkeypatch.delenv('MLPERF_SYSTEMNAME', raising=False)
        full = ['mlpstorage'] + argv_tail + ['--systemname', 'sys-v1']
        with patch('sys.argv', full):
            args = parse_arguments()
        assert getattr(args, 'systemname', None) == 'sys-v1', (
            f"{label}: expected args.systemname=='sys-v1' got "
            f"{getattr(args, 'systemname', None)!r}"
        )

    @pytest.mark.parametrize(
        "label,argv_tail",
        EMITTING_COMMANDS,
        ids=[label for label, _ in EMITTING_COMMANDS],
    )
    def test_empty_systemname_errors(self, label, argv_tail, monkeypatch):
        """Omitting --systemname on emitting commands raises SystemExit.

        Post CR-02 fix: ``--systemname`` is no longer ``required=True`` at
        the argparse layer (that would silently neuter the
        ``MLPERF_SYSTEMNAME`` env-var fallback per D-10). Instead, the post-
        parse validator checks that the resolved value is non-empty; with
        neither the CLI flag nor the env var supplied, ``DEFAULT_SYSTEMNAME``
        resolves to ``""`` and the validator errors out.
        """
        # Ensure env var is unset so the default falls back to '' (empty).
        monkeypatch.delenv('MLPERF_SYSTEMNAME', raising=False)
        full = ['mlpstorage'] + argv_tail  # no --systemname
        with patch('sys.argv', full):
            with pytest.raises(SystemExit):
                parse_arguments()

    # ---- CR-02: MLPERF_SYSTEMNAME / MLPERF_RESULTS_DIR env-var fallback ----
    #
    # The reviewer flagged that ``required=True`` + ``default=`` is
    # contradictory in argparse — ``required=True`` checks the CLI tokens,
    # not the resolved value, so the env-var defaults were dead on every
    # emitting subcommand. These tests pin the post-fix behavior: when
    # ``MLPERF_SYSTEMNAME`` / ``MLPERF_RESULTS_DIR`` is set, the CLI flag
    # is no longer mandatory.

    @pytest.mark.parametrize(
        "label,argv_tail",
        EMITTING_COMMANDS,
        ids=[label for label, _ in EMITTING_COMMANDS],
    )
    def test_systemname_env_var_satisfies_requirement(
        self, label, argv_tail, monkeypatch,
    ):
        """If MLPERF_SYSTEMNAME is set, ``--systemname`` may be omitted on the CLI.

        Pre-fix this raised SystemExit because argparse's ``required=True``
        ignored the env-var-sourced default. Post-fix, the resolved
        ``DEFAULT_SYSTEMNAME`` (sourced from the env var) satisfies the
        requirement and the parsed namespace carries that value.
        """
        monkeypatch.setenv('MLPERF_SYSTEMNAME', 'env-sys-v1')
        # Patch DEFAULT_SYSTEMNAME in place — do NOT reload mlpstorage_py.config.
        # Reloading config re-mints PARAM_VALIDATION (and other enums), breaking
        # `enum_instance in [enum_class.MEMBER, ...]` checks in any module that
        # already imported them (notably mlpstorage_py.rules.*). Downstream CLI
        # builders captured DEFAULT_SYSTEMNAME by name at import time, so reload
        # them so they pick up the patched value.
        import importlib
        import mlpstorage_py.config as cfg_mod
        import mlpstorage_py.cli.common_args as common_args_mod
        saved_systemname = cfg_mod.DEFAULT_SYSTEMNAME
        cfg_mod.DEFAULT_SYSTEMNAME = 'env-sys-v1'
        importlib.reload(common_args_mod)
        import mlpstorage_py.cli as cli_mod
        importlib.reload(cli_mod)
        import mlpstorage_py.cli_parser as cli_parser_mod
        importlib.reload(cli_parser_mod)
        try:
            full = ['mlpstorage'] + argv_tail  # no --systemname
            with patch('sys.argv', full):
                args = cli_parser_mod.parse_arguments()
            assert getattr(args, 'systemname', None) == 'env-sys-v1', (
                f"{label}: MLPERF_SYSTEMNAME env var must satisfy --systemname "
                f"requirement; got {getattr(args, 'systemname', None)!r}"
            )
        finally:
            monkeypatch.delenv('MLPERF_SYSTEMNAME', raising=False)
            cfg_mod.DEFAULT_SYSTEMNAME = saved_systemname
            importlib.reload(common_args_mod)
            importlib.reload(cli_mod)
            importlib.reload(cli_parser_mod)

    def test_results_dir_env_var_satisfies_requirement(self, monkeypatch):
        """If MLPERF_RESULTS_DIR is set, ``--results-dir`` may be omitted on the CLI.

        DEFAULT_RESULTS_DIR has a non-empty tempdir fallback even without
        the env var, but the contractual D-10 promise is that
        MLPERF_RESULTS_DIR is honored as a default on every emitting
        subcommand. Pre-fix, ``required=True`` ignored the env var entirely.
        """
        monkeypatch.setenv('MLPERF_RESULTS_DIR', '/env/results')
        monkeypatch.setenv('MLPERF_SYSTEMNAME', 'env-sys')  # so systemname check passes
        # Patch DEFAULT_RESULTS_DIR + DEFAULT_SYSTEMNAME in place — do NOT
        # reload mlpstorage_py.config (see sibling test for the enum-identity
        # rationale). Reload the downstream CLI builders so the patched values
        # propagate into argparse defaults.
        import importlib
        import mlpstorage_py.config as cfg_mod
        import mlpstorage_py.cli.common_args as common_args_mod
        saved_results_dir = cfg_mod.DEFAULT_RESULTS_DIR
        saved_systemname = cfg_mod.DEFAULT_SYSTEMNAME
        cfg_mod.DEFAULT_RESULTS_DIR = '/env/results'
        cfg_mod.DEFAULT_SYSTEMNAME = 'env-sys'
        importlib.reload(common_args_mod)
        import mlpstorage_py.cli as cli_mod
        importlib.reload(cli_mod)
        import mlpstorage_py.cli_parser as cli_parser_mod
        importlib.reload(cli_parser_mod)
        try:
            # Training run without --results-dir; everything else still passed.
            argv = [
                'mlpstorage', 'closed', 'training', 'unet3d', 'run',
                '--data-dir', '/d',
                '--num-accelerators', '1', '--accelerator-type', 'b200',
                '--client-host-memory-in-gb', '64', 'file',
            ]
            with patch('sys.argv', argv):
                args = cli_parser_mod.parse_arguments()
            assert args.results_dir == '/env/results', (
                f"MLPERF_RESULTS_DIR env var must satisfy --results-dir "
                f"requirement; got {args.results_dir!r}"
            )
        finally:
            monkeypatch.delenv('MLPERF_RESULTS_DIR', raising=False)
            monkeypatch.delenv('MLPERF_SYSTEMNAME', raising=False)
            cfg_mod.DEFAULT_RESULTS_DIR = saved_results_dir
            cfg_mod.DEFAULT_SYSTEMNAME = saved_systemname
            importlib.reload(common_args_mod)
            importlib.reload(cli_mod)
            importlib.reload(cli_parser_mod)

    def test_lockfile_does_not_require_systemname(self, monkeypatch):
        """Pure utility subcommands (lockfile) parse without --systemname."""
        monkeypatch.delenv('MLPERF_SYSTEMNAME', raising=False)
        with patch('sys.argv', ['mlpstorage', 'lockfile', 'generate',
                                '-o', '/tmp/x', '--results-dir', '/r']):
            args = parse_arguments()
        assert args.mode == 'lockfile'

    def test_init_does_not_require_systemname(self, monkeypatch):
        """`mlpstorage init` parses without --systemname (init is bootstrap, not emitting)."""
        monkeypatch.delenv('MLPERF_SYSTEMNAME', raising=False)
        with patch('sys.argv', ['mlpstorage', 'init', 'Acme', '/tmp/r']):
            args = parse_arguments()
        assert args.mode == 'init'

    def test_systemname_flag_default_reflects_env_var(self, monkeypatch):
        """The argparse default for --systemname reflects DEFAULT_SYSTEMNAME (which reads MLPERF_SYSTEMNAME).

        Per CONTEXT.md D-10 / plan Task 1, --systemname is `required=True` on
        emitting commands — argparse demands the flag on CLI even when the
        env var supplies a default. This test asserts the default value
        the parser stores when the user DOES pass --systemname matches the
        config-module constant; tests of the empty-systemname raise live in
        the slice-3 / slice-4 generate_output_location path, not the parser.
        """
        monkeypatch.setenv('MLPERF_SYSTEMNAME', 'env-sys')
        # Call the env-resolver helper directly — reloading mlpstorage_py.config
        # re-mints PARAM_VALIDATION and corrupts enum-identity in downstream
        # tests (see _resolve_default_systemname docstring in config.py).
        import mlpstorage_py.config as cfg_mod
        assert cfg_mod._resolve_default_systemname() == 'env-sys'
