"""Unit tests for kv_cache_benchmark/mlperf_wrapper.py.

Tests cover:
- get_rank(): env var reading, OMPI precedence over PMI, invalid value fallback
- BASE_SEED and WORKLOAD_PARAMS constants
- main(): rank dir creation, effective seed, output flag, option params, config path
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


class TestOptionParams:
    """Tests for WORKLOAD_PARAMS fixed MLPerf values."""

    def test_option_keys_are_1_2_3(self, wrapper_module):
        assert set(wrapper_module.WORKLOAD_PARAMS.keys()) == {1, 2, 3}

    def test_option1_model_is_8b(self, wrapper_module):
        assert wrapper_module.WORKLOAD_PARAMS[1]['model'] == 'llama3.1-8b'

    def test_option3_model_is_70b(self, wrapper_module):
        assert wrapper_module.WORKLOAD_PARAMS[3]['model'] == 'llama3.1-70b-instruct'

    def test_generation_mode_always_none(self, wrapper_module):
        for opt in [1, 2, 3]:
            assert wrapper_module.WORKLOAD_PARAMS[opt]['generation-mode'] == 'none'

    def test_option2_cpu_mem_gb_is_4(self, wrapper_module):
        assert wrapper_module.WORKLOAD_PARAMS[2]['cpu-mem-gb'] == 4


class TestMain:
    """Tests for main() — invokes subprocess and exits."""

    def _run_main(self, wrapper_module, monkeypatch, tmp_path, extra_argv=None,
                  ompi_rank=None):
        """Helper: sets up env, patches argv+subprocess, calls main(), captures cmd."""
        monkeypatch.delenv('OMPI_COMM_WORLD_RANK', raising=False)
        monkeypatch.delenv('PMI_RANK', raising=False)
        if ompi_rank is not None:
            monkeypatch.setenv('OMPI_COMM_WORLD_RANK', str(ompi_rank))

        output_dir = tmp_path / 'output'
        cache_dir = tmp_path / 'cache'
        argv = [
            'mlperf_wrapper.py',
            '--option', '1',
            '--seed', '42',
            '--base-output-dir', str(output_dir),
            '--cache-dir', str(cache_dir),
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
            extra_argv=['--seed', '10'],
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

    def test_workload_params_injected_for_option1(self, wrapper_module, monkeypatch, tmp_path):
        cmd, _, _ = self._run_main(wrapper_module, monkeypatch, tmp_path)
        assert '--model' in cmd
        model_idx = cmd.index('--model')
        assert cmd[model_idx + 1] == 'llama3.1-8b'

    def test_config_default_is_adjacent_config_yaml(self, wrapper_module, monkeypatch, tmp_path):
        cmd, _, _ = self._run_main(wrapper_module, monkeypatch, tmp_path)
        assert any('config.yaml' in arg for arg in cmd)

    def test_config_override_accepted(self, wrapper_module, monkeypatch, tmp_path):
        cmd, _, _ = self._run_main(
            wrapper_module, monkeypatch, tmp_path,
            extra_argv=['--config', '/custom/config.yaml'],
        )
        assert '/custom/config.yaml' in cmd
