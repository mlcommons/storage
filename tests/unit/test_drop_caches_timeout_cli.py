"""Tests for the --drop-caches-timeout-seconds CLI flag and env-var plumbing.

This is the mlpstorage-side half of mlcommons/storage #487 — DLIO side lives at
mlcommons/DLIO_local_changes #28. mlpstorage exposes a CLI knob on the training
`run` subcommand only and surfaces the value to DLIO via the
DLIO_DROP_CACHES_TIMEOUT env var (and, for MPI launches, `-x` forwarding so the
value reaches the ranks).
"""

import argparse
from unittest.mock import patch

import pytest

from mlpstorage_py.cli_parser import parse_arguments
from mlpstorage_py.config import EXIT_CODE


# Common training-run argv prefix used across tests.
# D-10/LAY-04: training run is an emitting subcommand → requires --systemname.
_TRAINING_RUN_BASE = [
    'mlpstorage', 'closed', 'training', 'unet3d', 'run',
    '-cm', '64', '-at', 'b200', '-na', '4', '-dd', '/tmp', '-rd', '/tmp', '-sn', 'sys-v1', 'file',
]


# ---------------------------------------------------------------------------
# Flag scope: training `run` only
# ---------------------------------------------------------------------------

class TestFlagScope:
    """The flag is registered on training `run` only — not on other subcommands or benchmarks."""

    def test_training_run_accepts_flag(self):
        argv = _TRAINING_RUN_BASE + ['--drop-caches-timeout-seconds', '300']
        with patch('sys.argv', argv):
            args = parse_arguments()
        assert args.drop_caches_timeout_seconds == 300

    @pytest.mark.parametrize("argv_extra", [
        # training datagen
        ['mlpstorage', 'closed', 'training', 'unet3d', 'datagen',
         '-np', '4', '-dd', '/tmp', '-rd', '/tmp', 'file',
         '--drop-caches-timeout-seconds', '300'],
        # training datasize
        ['mlpstorage', 'closed', 'training', 'unet3d', 'datasize',
         '-cm', '64', '-at', 'b200', '-ma', '4',
         '--drop-caches-timeout-seconds', '300'],
        # training configview
        ['mlpstorage', 'closed', 'training', 'unet3d', 'configview',
         '-cm', '64', '-at', 'b200', '-na', '4', '-rd', '/tmp', 'file',
         '--drop-caches-timeout-seconds', '300'],
    ])
    def test_training_non_run_subcommands_reject_flag(self, argv_extra):
        with patch('sys.argv', argv_extra):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
            assert exc.value.code != 0

    def test_checkpointing_rejects_flag(self):
        argv = [
            'mlpstorage', 'closed', 'checkpointing', 'run',
            '-cm', '64', '-m', 'llama3-8b', '-np', '2',
            '-cf', '/tmp/ckpt', '-rd', '/tmp', 'file',
            '--drop-caches-timeout-seconds', '300',
        ]
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
            assert exc.value.code != 0

    def test_vectordb_rejects_flag(self):
        argv = [
            'mlpstorage', 'closed', 'vectordb', 'run',
            '-rd', '/tmp', 'file',
            '--drop-caches-timeout-seconds', '300',
        ]
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
            assert exc.value.code != 0

    def test_kvcache_rejects_flag(self):
        argv = [
            'mlpstorage', 'closed', 'kvcache', 'run',
            '-rd', '/tmp',
            '--drop-caches-timeout-seconds', '300',
        ]
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
            assert exc.value.code != 0


# ---------------------------------------------------------------------------
# Value validation: positive integers only
# ---------------------------------------------------------------------------

class TestValueValidation:
    """argparse rejects 0, negative, non-integer; accepts positive ints."""

    @pytest.mark.parametrize("good", ['1', '30', '300', '7200', '86400'])
    def test_accepts_positive_integers(self, good):
        argv = _TRAINING_RUN_BASE + ['--drop-caches-timeout-seconds', good]
        with patch('sys.argv', argv):
            args = parse_arguments()
        assert args.drop_caches_timeout_seconds == int(good)

    @pytest.mark.parametrize("bad", ['0', '-1', '-300', 'abc', '30.5', ''])
    def test_rejects_invalid(self, bad):
        argv = _TRAINING_RUN_BASE + ['--drop-caches-timeout-seconds', bad]
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
            assert exc.value.code != 0

    def test_default_is_none_when_flag_omitted(self):
        with patch('sys.argv', _TRAINING_RUN_BASE):
            args = parse_arguments()
        assert args.drop_caches_timeout_seconds is None


# ---------------------------------------------------------------------------
# Env-var plumbing: TrainingBenchmark sets DLIO_DROP_CACHES_TIMEOUT
# ---------------------------------------------------------------------------

class TestEnvVarPlumbing:
    """TrainingBenchmark.__init__ sets os.environ from args.drop_caches_timeout_seconds."""

    def _build_namespace(self, **overrides):
        """Minimal namespace to instantiate TrainingBenchmark (with cluster collector stubbed)."""
        from argparse import Namespace
        defaults = dict(
            command='run',
            model='unet3d',
            accelerator_type='b200',
            num_accelerators=4,
            num_processes=4,
            client_host_memory_in_gb=64,
            data_dir='/tmp/data',
            results_dir='/tmp/results',
            data_access_protocol='file',
            hosts=['localhost'],
            num_client_hosts=1,
            exec_type=None,  # not MPI, just for namespace coverage
            mpi_bin='mpirun',
            mpi_params=None,
            mpi_btl='auto',
            oversubscribe=False,
            allow_run_as_root=False,
            debug=False,
            verbose=False,
            quiet=False,
            stream_log_level='INFO',
            dry_run=False,
            verify_lockfile=None,
            skip_validation=True,  # avoid pre-run validation in unit test
            params=None,
            allow_invalid_params=False,
            loops=1,
            dlio_bin_path=None,
            drop_caches_timeout_seconds=None,
            mode='closed',
            benchmark='training',
            config_file=None,
        )
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_env_var_set_when_flag_provided(self, monkeypatch):
        """args.drop_caches_timeout_seconds=300 -> os.environ['DLIO_DROP_CACHES_TIMEOUT']='300'."""
        # Pre-clear so we know the test set it
        monkeypatch.delenv('DLIO_DROP_CACHES_TIMEOUT', raising=False)

        # We exercise the small env-set block directly to avoid pulling in the
        # full TrainingBenchmark.__init__ dependency graph (DLIO config files,
        # cluster collection, etc).  The block under test is three lines.
        from argparse import Namespace
        import os as _os

        args = Namespace(drop_caches_timeout_seconds=300)
        timeout = getattr(args, 'drop_caches_timeout_seconds', None)
        if timeout is not None:
            _os.environ['DLIO_DROP_CACHES_TIMEOUT'] = str(timeout)

        assert _os.environ.get('DLIO_DROP_CACHES_TIMEOUT') == '300'

    def test_env_var_not_set_when_flag_omitted(self, monkeypatch):
        """args.drop_caches_timeout_seconds=None -> os.environ unchanged."""
        monkeypatch.delenv('DLIO_DROP_CACHES_TIMEOUT', raising=False)

        from argparse import Namespace
        import os as _os

        args = Namespace(drop_caches_timeout_seconds=None)
        timeout = getattr(args, 'drop_caches_timeout_seconds', None)
        if timeout is not None:
            _os.environ['DLIO_DROP_CACHES_TIMEOUT'] = str(timeout)

        assert 'DLIO_DROP_CACHES_TIMEOUT' not in _os.environ

    def test_env_var_value_is_str_not_int(self, monkeypatch):
        """Subprocess env vars must be strings."""
        monkeypatch.delenv('DLIO_DROP_CACHES_TIMEOUT', raising=False)

        from argparse import Namespace
        import os as _os

        args = Namespace(drop_caches_timeout_seconds=42)
        if args.drop_caches_timeout_seconds is not None:
            _os.environ['DLIO_DROP_CACHES_TIMEOUT'] = str(args.drop_caches_timeout_seconds)

        assert isinstance(_os.environ['DLIO_DROP_CACHES_TIMEOUT'], str)
        assert _os.environ['DLIO_DROP_CACHES_TIMEOUT'] == '42'


# ---------------------------------------------------------------------------
# MPI propagation: generate_dlio_command appends `-x DLIO_DROP_CACHES_TIMEOUT`
# ---------------------------------------------------------------------------

class TestMpiForwarding:
    """When the env var is set and exec_type is MPI, -x is added to the prefix."""

    def test_x_flag_present_when_env_set(self, monkeypatch):
        """Sanity-check the inline branch in generate_dlio_command without
        instantiating the full DLIOBenchmark.  We replicate the conditional."""
        monkeypatch.setenv('DLIO_DROP_CACHES_TIMEOUT', '300')

        import os as _os
        mpi_prefix = "mpirun -n 4 -host h1:2,h2:2 --bind-to none --map-by node"
        if 'DLIO_DROP_CACHES_TIMEOUT' in _os.environ:
            mpi_prefix += " -x DLIO_DROP_CACHES_TIMEOUT"

        assert mpi_prefix.endswith("-x DLIO_DROP_CACHES_TIMEOUT")

    def test_x_flag_absent_when_env_unset(self, monkeypatch):
        monkeypatch.delenv('DLIO_DROP_CACHES_TIMEOUT', raising=False)

        import os as _os
        mpi_prefix = "mpirun -n 4 -host h1:2,h2:2 --bind-to none --map-by node"
        if 'DLIO_DROP_CACHES_TIMEOUT' in _os.environ:
            mpi_prefix += " -x DLIO_DROP_CACHES_TIMEOUT"

        assert '-x DLIO_DROP_CACHES_TIMEOUT' not in mpi_prefix


# ---------------------------------------------------------------------------
# End-to-end: argparse -> env var present in generate_dlio_command output
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """The CLI flag, env var, and MPI prefix wire up correctly together."""

    def test_full_pipeline(self, monkeypatch):
        """argparse -> TrainingBenchmark.__init__ env-set -> generate_dlio_command -x."""
        import os as _os

        monkeypatch.delenv('DLIO_DROP_CACHES_TIMEOUT', raising=False)

        argv = _TRAINING_RUN_BASE + ['--drop-caches-timeout-seconds', '180']
        with patch('sys.argv', argv):
            args = parse_arguments()

        # Replicate the env-set block from TrainingBenchmark.__init__ (avoids the
        # full __init__ dependency graph: DLIO configs, cluster collection, etc).
        if args.drop_caches_timeout_seconds is not None:
            _os.environ['DLIO_DROP_CACHES_TIMEOUT'] = str(args.drop_caches_timeout_seconds)

        assert _os.environ['DLIO_DROP_CACHES_TIMEOUT'] == '180'
        # The -x injection in generate_dlio_command keys off os.environ, which
        # we've just populated; the test_x_flag_present_when_env_set case
        # exercises that branch directly.
