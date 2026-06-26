"""Unit tests for kv_cache_benchmark/mlperf_wrapper.py.

Tests cover:
- get_rank(): env var reading, OMPI precedence over PMI, invalid value fallback
- BASE_SEED constant
- main(): rank dir creation, effective seed, output flag, forwarded args
  pass-through, per-rank overrides come last (argparse-style override)
"""

import importlib.util
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

WRAPPER_PATH = Path(__file__).parent.parent.parent / 'kv_cache_benchmark' / 'mlperf_wrapper.py'


@pytest.fixture
def wrapper_module():
    """Load mlperf_wrapper fresh per test (function scope prevents monkeypatch leakage)."""
    spec = importlib.util.spec_from_file_location('mlperf_wrapper', WRAPPER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGetRank:
    """Tests for get_rank() env-var detection."""

    def test_returns_zero_no_env_vars(self, wrapper_module, monkeypatch):
        monkeypatch.delenv('OMPI_COMM_WORLD_RANK', raising=False)
        monkeypatch.delenv('PMI_RANK', raising=False)
        assert wrapper_module.get_rank() == 0

    def test_reads_ompi_rank(self, wrapper_module, monkeypatch):
        monkeypatch.delenv('PMI_RANK', raising=False)
        monkeypatch.setenv('OMPI_COMM_WORLD_RANK', '3')
        assert wrapper_module.get_rank() == 3

    def test_pmi_rank_fallback(self, wrapper_module, monkeypatch):
        monkeypatch.delenv('OMPI_COMM_WORLD_RANK', raising=False)
        monkeypatch.setenv('PMI_RANK', '5')
        assert wrapper_module.get_rank() == 5

    def test_ompi_takes_precedence_over_pmi(self, wrapper_module, monkeypatch):
        monkeypatch.setenv('OMPI_COMM_WORLD_RANK', '1')
        monkeypatch.setenv('PMI_RANK', '9')
        assert wrapper_module.get_rank() == 1

    def test_invalid_ompi_falls_back_to_zero(self, wrapper_module, monkeypatch):
        monkeypatch.setenv('OMPI_COMM_WORLD_RANK', 'not_a_number')
        monkeypatch.delenv('PMI_RANK', raising=False)
        assert wrapper_module.get_rank() == 0


class TestBaseConstants:
    """Tests for module-level constants."""

    def test_base_seed_is_42(self, wrapper_module):
        assert wrapper_module.BASE_SEED == 42

    def test_wrapper_does_not_define_workload_params(self, wrapper_module):
        """WORKLOAD_PARAMS lives in mlpstorage_py.benchmarks.kvcache now;
        the wrapper must remain a dumb rank-aware launcher."""
        assert not hasattr(wrapper_module, 'WORKLOAD_PARAMS')


class TestMain:
    """Tests for main() — invokes subprocess and exits."""

    def _run_main(self, wrapper_module, monkeypatch, tmp_path, extra_argv=None,
                  ompi_rank=None, start_delay=0, end_delay=0):
        """Helper: sets up env, patches argv+subprocess, calls main(), captures cmd."""
        monkeypatch.delenv('OMPI_COMM_WORLD_RANK', raising=False)
        monkeypatch.delenv('PMI_RANK', raising=False)
        if ompi_rank is not None:
            monkeypatch.setenv('OMPI_COMM_WORLD_RANK', str(ompi_rank))

        output_dir = tmp_path / 'output'
        cache_dir = tmp_path / 'cache'
        argv = [
            'mlperf_wrapper.py',
            '--rank-output-base', str(output_dir),
            '--rank-cache-base', str(cache_dir),
            '--seed-base', '42',
            '--start-delay', str(start_delay),
            '--end-delay', str(end_delay),
        ]
        if extra_argv:
            argv.extend(extra_argv)

        captured = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            return MagicMock(returncode=0)

        with patch('sys.argv', argv), \
             patch.object(wrapper_module.subprocess, 'run', side_effect=fake_run), \
             patch.object(wrapper_module.time, 'sleep'):
            with pytest.raises(SystemExit) as exc_info:
                wrapper_module.main()

        assert exc_info.value.code == 0
        return captured[0], output_dir, cache_dir

    def test_creates_rank_output_dir(self, wrapper_module, monkeypatch, tmp_path):
        _, output_dir, _ = self._run_main(wrapper_module, monkeypatch, tmp_path)
        assert (output_dir / 'rank_0').is_dir()

    def test_creates_rank_cache_dir(self, wrapper_module, monkeypatch, tmp_path):
        _, _, cache_dir = self._run_main(wrapper_module, monkeypatch, tmp_path)
        assert (cache_dir / 'rank_0').is_dir()

    def test_effective_seed_is_base_plus_rank(self, wrapper_module, monkeypatch, tmp_path):
        cmd, _, _ = self._run_main(
            wrapper_module, monkeypatch, tmp_path,
            extra_argv=['--seed-base', '10'],
            ompi_rank=2,
        )
        seed_idx = cmd.index('--seed')
        assert cmd[seed_idx + 1] == '12'

    def test_output_flag_points_to_rank_dir(self, wrapper_module, monkeypatch, tmp_path):
        cmd, _, _ = self._run_main(wrapper_module, monkeypatch, tmp_path)
        output_idx = cmd.index('--output')
        output_val = cmd[output_idx + 1]
        assert 'rank_0' in output_val
        assert 'kvcache_results_' in output_val

    def test_cache_dir_points_to_rank_cache_dir(self, wrapper_module, monkeypatch, tmp_path):
        cmd, _, cache_dir = self._run_main(wrapper_module, monkeypatch, tmp_path)
        cache_idx = cmd.index('--cache-dir')
        cache_val = cmd[cache_idx + 1]
        assert str(cache_dir / 'rank_0') == cache_val

    def test_forwarded_args_passed_through_verbatim(self, wrapper_module, monkeypatch, tmp_path):
        """Wrapper forwards unrecognized args to kv-cache.py without modification."""
        cmd, _, _ = self._run_main(
            wrapper_module, monkeypatch, tmp_path,
            extra_argv=['--model', 'llama3.1-8b', '--num-users', '200',
                        '--config', '/tmp/cfg.yaml'],
        )
        assert '--model' in cmd
        assert cmd[cmd.index('--model') + 1] == 'llama3.1-8b'
        assert '--num-users' in cmd
        assert cmd[cmd.index('--num-users') + 1] == '200'
        assert '--config' in cmd
        assert cmd[cmd.index('--config') + 1] == '/tmp/cfg.yaml'

    def test_per_rank_overrides_come_last(self, wrapper_module, monkeypatch, tmp_path):
        """Per-rank --seed/--output/--cache-dir are appended after forwarded
        args so argparse store-action picks them over any duplicates."""
        cmd, _, _ = self._run_main(
            wrapper_module, monkeypatch, tmp_path,
            extra_argv=['--seed', '999', '--cache-dir', '/forwarded/cache'],
        )
        # Last --seed wins; wrapper's effective seed (42 + rank 0 = 42)
        # must appear AFTER the forwarded --seed.
        seed_positions = [i for i, x in enumerate(cmd) if x == '--seed']
        assert len(seed_positions) == 2
        assert cmd[seed_positions[-1] + 1] == '42'

        cache_positions = [i for i, x in enumerate(cmd) if x == '--cache-dir']
        assert len(cache_positions) == 2
        assert '/forwarded/cache' not in cmd[cache_positions[-1] + 1]

    def test_no_config_default_injected(self, wrapper_module, monkeypatch, tmp_path):
        """Wrapper no longer falls back to an adjacent config.yaml — caller owns the path."""
        cmd, _, _ = self._run_main(wrapper_module, monkeypatch, tmp_path)
        assert '--config' not in cmd
