"""
Tests for three-mode parser structure and open-gated arg isolation (TEST-01, TEST-02, TEST-03, TEST-04).
Reviewed 2026-06-08 (04-REVIEWS.md).
"""

import sys
import pytest
from unittest.mock import patch

from mlpstorage_py.cli_parser import parse_arguments
from mlpstorage_py.config import (
    MODELS_CLOSED,
    MODELS_OPEN,
    MODELS,
    ACCELERATORS_CLOSED,
    ACCELERATORS,
    LLM_MODELS,
    KVCACHE_MODELS,
)


# =====================================================================
# TEST-01: Parser Mode Structure
# =====================================================================

class TestParserModeStructure:
    """Verify that the three-branch parser sets mode and benchmark attributes correctly."""

    def test_whatif_mode_sets_mode_attr(self):
        """whatif mode must set args.mode='whatif' and args.benchmark='training'."""
        with patch('sys.argv', ['mlpstorage', 'whatif', 'training', 'cosmoflow',
                                 'datasize', '-cm', '64', '-at', 'h100', '-ma', '4']):
            args = parse_arguments()
        assert args.mode == 'whatif'
        assert args.benchmark == 'training'

    @pytest.mark.parametrize('mode, model, accel', [
        ('closed', 'unet3d', 'b200'),
        ('open',   'unet3d', 'b200'),
        ('whatif', 'cosmoflow', 'h100'),
    ])
    def test_all_three_modes_set_benchmark(self, mode, model, accel):
        """All three modes must produce args.benchmark == 'training'."""
        with patch('sys.argv', ['mlpstorage', mode, 'training', model,
                                 'datasize', '-cm', '64', '-at', accel, '-ma', '4']):
            args = parse_arguments()
        assert args.benchmark == 'training'

    @pytest.mark.parametrize('benchmark, extra_args', [
        ('training',      ['unet3d', 'datasize', '-cm', '64', '-at', 'b200', '-ma', '4']),
        ('checkpointing', ['datasize', '-cm', '64', '-m', 'llama3-8b', '-np', '2']),
        ('vectordb',      ['datasize']),
        ('kvcache',       ['run', '-rd', '/tmp']),
    ])
    def test_all_benchmarks_reachable_in_closed(self, benchmark, extra_args):
        """All four benchmark types must be reachable under 'closed' mode without error."""
        with patch('sys.argv', ['mlpstorage', 'closed', benchmark] + extra_args):
            args = parse_arguments()
        assert args.benchmark == benchmark


# =====================================================================
# TEST-02: Open-Gated Arg Exclusion
# =====================================================================

class TestOpenGatedArgExclusion:
    """Verify that open-gated args are rejected in closed mode and accepted in open/whatif."""

    # ------------------------------------------------------------------
    # Negative cases — closed mode must reject open-gated arguments
    # ------------------------------------------------------------------

    def test_closed_training_rejects_loops(self):
        """closed training run must reject --loops with a non-zero SystemExit."""
        base = ['mlpstorage', 'closed', 'training', 'unet3d', 'run',
                '-cm', '64', '-at', 'b200', '-na', '4', '-rd', '/tmp', 'file']
        with patch('sys.argv', base + ['--loops', '2']):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code != 0

    def test_closed_training_accepts_params(self):
        """closed training must accept --params for CLOSED_ALLOWED_PARAMS (issue #433).

        The rules checker (TrainingRunRulesChecker.CLOSED_ALLOWED_PARAMS) explicitly
        permits dotted-key overrides like dataset.num_files_train and
        dataset.num_subfolders_train in closed submissions, so the parser must
        register --params in closed mode too. Per-key validity is a rules-checker
        concern, not an argparse concern.
        """
        base = ['mlpstorage', 'closed', 'training', 'unet3d', 'run',
                '-cm', '64', '-at', 'b200', '-na', '4', '-rd', '/tmp', 'file']
        with patch('sys.argv', base + ['--params', 'dataset.num_files_train=1000']):
            args = parse_arguments()
        flattened = [kv for batch in (args.params or []) for kv in batch]
        assert 'dataset.num_files_train=1000' in flattened

    def test_closed_training_rejects_timeseries_interval(self):
        """closed training run must reject --timeseries-interval with a non-zero SystemExit."""
        base = ['mlpstorage', 'closed', 'training', 'unet3d', 'run',
                '-cm', '64', '-at', 'b200', '-na', '4', '-rd', '/tmp', 'file']
        with patch('sys.argv', base + ['--timeseries-interval', '5']):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code != 0

    def test_closed_training_rejects_allow_invalid_params(self):
        """closed training run must reject --allow-invalid-params with a non-zero SystemExit."""
        base = ['mlpstorage', 'closed', 'training', 'unet3d', 'run',
                '-cm', '64', '-at', 'b200', '-na', '4', '-rd', '/tmp', 'file']
        with patch('sys.argv', base + ['--allow-invalid-params']):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code != 0

    def test_closed_checkpointing_rejects_loops(self):
        """closed checkpointing run must reject --loops with a non-zero SystemExit."""
        argv = ['mlpstorage', 'closed', 'checkpointing', 'run',
                '-cm', '64', '-m', 'llama3-8b', '-np', '2',
                '-cf', '/tmp', '-rd', '/tmp', 'file', '--loops', '2']
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code != 0

    def test_closed_vectordb_rejects_loops(self):
        """closed vectordb run must reject --loops with a non-zero SystemExit."""
        argv = ['mlpstorage', 'closed', 'vectordb', 'run',
                '-rd', '/tmp', 'file', '--loops', '2']
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code != 0

    def test_closed_kvcache_rejects_loops(self):
        """closed kvcache run must reject --loops (set_defaults; flag absent from parser)."""
        argv = ['mlpstorage', 'closed', 'kvcache', 'run',
                '-rd', '/tmp', '--loops', '2']
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code != 0

    # ------------------------------------------------------------------
    # Positive cases — open/whatif mode must accept open-gated arguments
    # ------------------------------------------------------------------

    def test_open_training_accepts_loops(self):
        """open training run must accept --loops and set args.loops correctly."""
        argv = ['mlpstorage', 'open', 'training', 'unet3d', 'run',
                '-cm', '64', '-at', 'b200', '-na', '4', '-rd', '/tmp', 'file',
                '--loops', '3']
        with patch('sys.argv', argv):
            args = parse_arguments()
        assert args.loops == 3

    def test_open_training_accepts_params(self):
        """open training run must accept --params and set args.params to a non-None value."""
        argv = ['mlpstorage', 'open', 'training', 'unet3d', 'run',
                '-cm', '64', '-at', 'b200', '-na', '4', '-rd', '/tmp', 'file',
                '--params', 'key=val']
        with patch('sys.argv', argv):
            args = parse_arguments()
        assert args.params is not None

    def test_open_training_accepts_timeseries_interval(self):
        """open training run must accept --timeseries-interval and set args.timeseries_interval."""
        argv = ['mlpstorage', 'open', 'training', 'unet3d', 'run',
                '-cm', '64', '-at', 'b200', '-na', '4', '-rd', '/tmp', 'file',
                '--timeseries-interval', '5.0']
        with patch('sys.argv', argv):
            args = parse_arguments()
        assert args.timeseries_interval == 5.0

    def test_whatif_training_accepts_loops(self):
        """whatif training run must accept --loops and set args.loops correctly."""
        argv = ['mlpstorage', 'whatif', 'training', 'unet3d', 'run',
                '-cm', '64', '-at', 'h100', '-na', '4', '-rd', '/tmp', 'file',
                '--loops', '2']
        with patch('sys.argv', argv):
            args = parse_arguments()
        assert args.loops == 2

    # ------------------------------------------------------------------
    # Regression: every subcommand must expose `args.loops` on its namespace.
    # main.py:326 does `for i in range(getattr(args, 'loops', 1))` after the
    # fix for #444, but the parsers should still guarantee the attribute so
    # the run_summary output and any future caller doesn't see None. Issue
    # #444 was caused by kvcache datasize in open/whatif lacking this.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize('mode, benchmark, sub_argv', [
        # kvcache — the direct repro for #444
        ('closed',  'kvcache',       ['datasize']),
        ('open',    'kvcache',       ['datasize']),
        ('whatif',  'kvcache',       ['datasize']),
        # training — non-run subcommands have no --loops flag
        ('closed',  'training',      ['unet3d', 'datasize', '-cm', '64', '-at', 'b200', '-ma', '4']),
        ('open',    'training',      ['unet3d', 'datasize', '-cm', '64', '-at', 'b200', '-ma', '4']),
        ('whatif',  'training',      ['unet3d', 'datasize', '-cm', '64', '-at', 'h100', '-ma', '4']),
        ('closed',  'training',      ['unet3d', 'datagen', '-np', '4', 'file']),
        # checkpointing — datasize / configview
        ('closed',  'checkpointing', ['datasize', '-cm', '64', '-m', 'llama3-8b', '-np', '2']),
        ('open',    'checkpointing', ['datasize', '-cm', '64', '-m', 'llama3-8b', '-np', '2']),
        # vectordb — datasize / datagen
        ('closed',  'vectordb',      ['datasize']),
        ('open',    'vectordb',      ['datasize']),
        ('whatif',  'vectordb',      ['datasize']),
    ])
    def test_non_run_subcommands_expose_loops_attr(self, mode, benchmark, sub_argv):
        """Every benchmark subcommand must populate args.loops so main.py can drive the run loop.

        Regression for #444: `mlpstorage whatif kvcache datasize` crashed with
        AttributeError because the run loop in main.py reads args.loops, but
        --loops was only registered on the run subparser.
        """
        argv = ['mlpstorage', mode, benchmark] + sub_argv
        with patch('sys.argv', argv):
            args = parse_arguments()
        assert hasattr(args, 'loops'), (
            f"{mode} {benchmark} {sub_argv[0]} did not set args.loops on the namespace"
        )
        assert args.loops == 1

    # ------------------------------------------------------------------
    # Help-text hiding test (MEDIUM concern from 04-REVIEWS.md)
    # ------------------------------------------------------------------

    def test_closed_training_leaf_help_hides_open_args(self, capsys):
        """closed training run file --help must NOT show open-only flags in stdout.

        --params was moved into the core training args in #433 (it is CLOSED-allowed
        per CLOSED_ALLOWED_PARAMS) and should be visible in closed help. The truly
        open-only flags (--loops, --allow-invalid-params, --timeseries-interval) must
        still be hidden.
        """
        with patch('sys.argv', ['mlpstorage', 'closed', 'training', 'unet3d', 'run', 'file', '--help']):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert '--loops' not in out
        assert '--allow-invalid-params' not in out
        assert '--timeseries-interval' not in out
        # --params is now exposed in closed mode and should appear in help.
        assert '--params' in out


# =====================================================================
# TEST-03: Model and Accelerator Restrictions
# =====================================================================

class TestModelAcceleratorRestrictions:
    """Verify model and accelerator allow-lists per mode."""

    # ------------------------------------------------------------------
    # Training model allow-list — closed only accepts MODELS_CLOSED
    # ------------------------------------------------------------------

    @pytest.mark.parametrize('model', ['cosmoflow', 'resnet50', 'dlrm', 'flux'])
    def test_closed_training_rejects_open_only_models(self, model):
        """closed training must reject models not in MODELS_CLOSED."""
        argv = ['mlpstorage', 'closed', 'training', model,
                'datasize', '-cm', '64', '-at', 'b200', '-ma', '4']
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code != 0

    @pytest.mark.parametrize('model', ['unet3d', 'retinanet'])
    def test_closed_training_accepts_closed_models(self, model):
        """closed training must accept all models in MODELS_CLOSED."""
        argv = ['mlpstorage', 'closed', 'training', model,
                'datasize', '-cm', '64', '-at', 'b200', '-ma', '4']
        with patch('sys.argv', argv):
            args = parse_arguments()
        assert args.model == model

    def test_open_training_rejects_cosmoflow(self):
        """open training must reject cosmoflow (MODELS_OPEN == MODELS_CLOSED)."""
        argv = ['mlpstorage', 'open', 'training', 'cosmoflow',
                'datasize', '-cm', '64', '-at', 'b200', '-ma', '4']
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code != 0

    @pytest.mark.parametrize('model', MODELS)
    def test_whatif_training_accepts_all_models(self, model):
        """whatif training must accept all models in MODELS (6 total)."""
        argv = ['mlpstorage', 'whatif', 'training', model,
                'datasize', '-cm', '64', '-at', 'h100', '-ma', '4']
        with patch('sys.argv', argv):
            args = parse_arguments()
        assert args.model == model

    # ------------------------------------------------------------------
    # Accelerator allow-list — closed only accepts ACCELERATORS_CLOSED
    # ------------------------------------------------------------------

    @pytest.mark.parametrize('accel', ['h100', 'a100'])
    def test_closed_training_rejects_open_accelerators(self, accel):
        """closed training run must reject accelerators not in ACCELERATORS_CLOSED."""
        argv = ['mlpstorage', 'closed', 'training', 'unet3d', 'run',
                '-cm', '64', '-at', accel, '-na', '4', '-rd', '/tmp', 'file']
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code != 0

    @pytest.mark.parametrize('accel', ['b200', 'mi355'])
    def test_closed_training_accepts_closed_accelerators(self, accel):
        """closed training run must accept all accelerators in ACCELERATORS_CLOSED."""
        argv = ['mlpstorage', 'closed', 'training', 'unet3d', 'run',
                '-cm', '64', '-at', accel, '-na', '4', '-rd', '/tmp', 'file']
        with patch('sys.argv', argv):
            args = parse_arguments()
        assert args.accelerator_type == accel

    @pytest.mark.parametrize('accel', ACCELERATORS)
    def test_whatif_training_accepts_all_accelerators(self, accel):
        """whatif training run must accept all accelerators in ACCELERATORS (4 total)."""
        argv = ['mlpstorage', 'whatif', 'training', 'unet3d', 'run',
                '-cm', '64', '-at', accel, '-na', '4', '-rd', '/tmp', 'file']
        with patch('sys.argv', argv):
            args = parse_arguments()
        assert args.accelerator_type == accel

    # ------------------------------------------------------------------
    # KVCache positional grammar — closed has no model positional
    # ------------------------------------------------------------------

    def test_closed_kvcache_run_no_model_needed(self):
        """closed kvcache run must parse successfully without any model argument."""
        with patch('sys.argv', ['mlpstorage', 'closed', 'kvcache', 'run', '-rd', '/tmp']):
            args = parse_arguments()
        # The parsed result must not require model — assert no exception was raised
        assert args.benchmark == 'kvcache'
        assert args.command == 'run'

    def test_closed_kvcache_rejects_model_flag(self):
        """closed kvcache run must reject --model (flag absent from parser)."""
        argv = ['mlpstorage', 'closed', 'kvcache', 'run', '-rd', '/tmp',
                '--model', 'llama3.1-8b']
        with patch('sys.argv', argv):
            with pytest.raises(SystemExit) as exc:
                parse_arguments()
        assert exc.value.code != 0

    def test_open_kvcache_accepts_model_flag(self):
        """open kvcache run must accept --model and set args.model correctly."""
        argv = ['mlpstorage', 'open', 'kvcache', 'run', '-rd', '/tmp',
                '--model', 'llama3.1-8b', '-nu', '100']
        with patch('sys.argv', argv):
            args = parse_arguments()
        assert args.model == 'llama3.1-8b'


# =====================================================================
# TEST-04: Version Subcommand Dispatch
# =====================================================================

class TestVersionDispatch:
    """Verify that the version subcommand dispatches correctly via parse_arguments()."""

    def test_version_subcommand_sets_mode(self):
        """mlpstorage version must set args.mode == 'version' without SystemExit."""
        with patch('sys.argv', ['mlpstorage', 'version']):
            args = parse_arguments()
        assert args.mode == 'version'

    def test_version_string_is_non_empty(self):
        """VERSION constant must be a non-empty string that is not 'unknown'."""
        from mlpstorage_py import VERSION
        assert isinstance(VERSION, str)
        assert len(VERSION) > 0
        assert VERSION != 'unknown'

    def test_version_dispatch_does_not_raise(self):
        """mlpstorage version must parse cleanly; result must have mode='version' and no benchmark."""
        with patch('sys.argv', ['mlpstorage', 'version']):
            result = parse_arguments()
        assert result.mode == 'version'
        assert not hasattr(result, 'benchmark')
