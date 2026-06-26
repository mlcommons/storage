"""
Tests for KVCacheBenchmark class in mlpstorage.benchmarks.kvcache module.

Tests cover:
- Cluster information collection for distributed runs
- _interruptible_sleep: what-if skip, chunked sleep, Ctrl-C propagation
- _aggregate_option_results: bandwidth sum, P95 max, partial failure, CPU-tier flag
- _write_run_summary: output path, JSON schema, MLPSJsonEncoder usage
- _execute_run: MLPerf sequence, CLOSED enforcement, option/trial loop
"""

import sys
import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, call
from argparse import Namespace

# Stub out optional heavy deps so benchmark imports succeed without the full
# ML stack. Use importlib.util.find_spec — checking sys.modules alone would
# install a MagicMock for a perfectly importable module that just hasn't been
# imported yet, which then poisons later test collections (e.g. test_parquet_reader).
import importlib.util as _ilu
for _dep in ('pyarrow', 'pyarrow.ipc', 'psutil'):
    if _ilu.find_spec(_dep) is None and _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

from mlpstorage_py.config import BENCHMARK_TYPES, EXEC_TYPE


class TestKVCacheClusterCollection:
    """Tests for cluster information collection in KVCacheBenchmark."""

    @pytest.fixture
    def basic_args(self, tmp_path):
        """Create basic args for KV cache benchmark."""
        return Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            mode='closed',
            orgname='Acme',
            systemname='sys-v1',
            results_dir=str(tmp_path),
            model='llama3.1-8b',
            command='run',
            num_users=100,
            duration=60,
            gpu_mem_gb=16.0,
            cpu_mem_gb=32.0,
            cache_dir=None,
            generation_mode='realistic',
            performance_profile='latency',
            kvcache_bin_path=None,
            disable_multi_turn=False,
            disable_prefix_caching=False,
            enable_rag=False,
            enable_autoscaling=False,
            seed=None,
            exec_type=None,
            hosts=None,
            num_processes=None,
            mpi_bin='mpirun',
            oversubscribe=False,
            allow_run_as_root=False,
            mpi_params=None,
        )

    def test_cluster_collection_called_for_run_command(self, basic_args, tmp_path):
        """Should collect cluster information for run command."""
        basic_args.command = 'run'

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = MagicMock()

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

        mock_cluster.assert_called_once()
        assert hasattr(benchmark, 'cluster_information')

    def test_cluster_collection_not_called_for_datasize_command(self, basic_args, tmp_path):
        """Should NOT collect cluster information for datasize command."""
        basic_args.command = 'datasize'

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

        mock_cluster.assert_not_called()

class TestKVCacheNumProcessesStorage:
    """Tests for num_processes storage in KVCacheBenchmark."""

    @pytest.fixture
    def basic_args(self, tmp_path):
        """Create basic args for KV cache benchmark."""
        return Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            mode='closed',
            orgname='Acme',
            systemname='sys-v1',
            results_dir=str(tmp_path),
            model='llama3.1-8b',
            command='run',
            num_users=100,
            duration=60,
            gpu_mem_gb=16.0,
            cpu_mem_gb=32.0,
            cache_dir=None,
            generation_mode='realistic',
            performance_profile='latency',
            kvcache_bin_path=None,
            disable_multi_turn=False,
            disable_prefix_caching=False,
            enable_rag=False,
            enable_autoscaling=False,
            seed=None,
            exec_type=None,
            hosts=None,
            num_processes=8,
            mpi_bin='mpirun',
            oversubscribe=False,
            allow_run_as_root=False,
            mpi_params=None,
        )

    def test_num_processes_stored_from_args(self, basic_args, tmp_path):
        """Should store num_processes from args."""
        basic_args.num_processes = 16

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

        assert benchmark.num_processes == 16

    def test_num_processes_none_when_not_provided(self, basic_args, tmp_path):
        """Should be None when num_processes not in args."""
        del basic_args.num_processes  # Remove attribute

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            benchmark = KVCacheBenchmark(basic_args, run_datetime="20250115_120000")

        assert benchmark.num_processes is None


class TestKVCacheMetadata:
    """Test metadata structure for history integration."""

    @pytest.fixture
    def base_args(self, tmp_path):
        """Create base args for KV cache benchmark metadata tests."""
        return Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            mode='closed',
            orgname='Acme',
            systemname='sys-v1',
            results_dir=str(tmp_path),
            model='llama3.1-8b',
            command='run',
            num_users=100,
            duration=60,
            gpu_mem_gb=16.0,
            cpu_mem_gb=32.0,
            cache_dir=None,
            generation_mode='realistic',
            performance_profile='latency',
            kvcache_bin_path=None,
            disable_multi_turn=False,
            disable_prefix_caching=False,
            enable_rag=False,
            enable_autoscaling=False,
            seed=None,
            exec_type=None,
            hosts=None,
            num_processes=None,
            mpi_bin='mpirun',
            oversubscribe=False,
            allow_run_as_root=False,
            mpi_params=None,
        )

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        logger = MagicMock()
        logger.status = MagicMock()
        logger.info = MagicMock()
        logger.debug = MagicMock()
        logger.warning = MagicMock()
        logger.verboser = MagicMock()
        logger.verbose = MagicMock()
        return logger

    def test_metadata_has_required_fields(self, base_args, mock_logger, tmp_path):
        """Verify metadata includes fields required by history module."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            bm = KVCacheBenchmark(base_args, logger=mock_logger, run_datetime="20250124_120000")
            meta = bm.metadata

        # Required by history module
        assert 'benchmark_type' in meta
        assert 'model' in meta
        assert 'command' in meta
        assert 'run_datetime' in meta
        assert 'result_dir' in meta

    def test_metadata_includes_kvcache_specific_fields(self, base_args, mock_logger, tmp_path):
        """Verify KV cache specific metadata fields."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            bm = KVCacheBenchmark(base_args, logger=mock_logger, run_datetime="20250124_120000")
            meta = bm.metadata

        assert 'kvcache_model' in meta
        assert 'num_users' in meta
        assert 'duration' in meta
        assert 'gpu_mem_gb' in meta
        assert 'cpu_mem_gb' in meta
        assert 'generation_mode' in meta
        assert 'performance_profile' in meta

    def test_metadata_parameters_populated_for_run_checker(self, base_args, mock_logger, tmp_path):
        """Issue #537: KVCacheRunRulesChecker reads workload config from
        metadata['parameters']. KVCache has no DLIO combined_params so the base
        class falls back to {}, and reportgen then classifies every run INVALID
        with 'Missing model parameter'. The kvcache metadata must populate
        ['parameters'] with the workload config keys the checker consults."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            bm = KVCacheBenchmark(base_args, logger=mock_logger, run_datetime="20250124_120000")
            meta = bm.metadata

        params = meta.get('parameters')
        assert isinstance(params, dict) and params, \
            "metadata['parameters'] must be a non-empty dict so reportgen can read workload config"

        # Keys the KVCacheRunRulesChecker reads via self.benchmark_run.parameters.get(...)
        assert params.get('model') == 'llama3.1-8b'
        assert params.get('num_users') == 100
        assert params.get('duration') == 60
        assert params.get('gpu_mem_gb') == 16.0
        assert params.get('cpu_mem_gb') == 32.0
        assert params.get('generation_mode') == 'realistic'
        assert params.get('performance_profile') == 'latency'

    def test_metadata_includes_distributed_info(self, base_args, mock_logger, tmp_path):
        """Verify metadata includes distributed execution info."""
        base_args.exec_type = EXEC_TYPE.MPI
        base_args.hosts = ['host1', 'host2']
        base_args.num_processes = 4

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            bm = KVCacheBenchmark(base_args, logger=mock_logger, run_datetime="20250124_120000")
            meta = bm.metadata

        assert 'num_processes' in meta
        assert meta['num_processes'] == 4
        assert 'hosts' in meta
        assert meta['hosts'] == ['host1', 'host2']
        assert 'exec_type' in meta

    def test_metadata_model_consistency(self, base_args, mock_logger, tmp_path):
        """Verify 'model' field matches 'kvcache_model' for history compatibility."""
        base_args.model = 'llama3.1-70b-instruct'

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            bm = KVCacheBenchmark(base_args, logger=mock_logger, run_datetime="20250124_120000")
            meta = bm.metadata

        assert meta['model'] == 'llama3.1-70b-instruct'
        assert meta['kvcache_model'] == 'llama3.1-70b-instruct'

    def test_metadata_closed_mode_defaults_model(self, base_args, mock_logger, tmp_path):
        """In closed mode the CLI does not expose --model (see
        _add_kvcache_model_arguments — only added for open/whatif). The
        benchmark must default args.model from KVCACHE_MODEL_DEFAULT before
        the base class writes metadata, otherwise the on-disk model field
        would be None and workload grouping would mis-bucket the run."""
        from mlpstorage_py.config import KVCACHE_MODEL_DEFAULT

        delattr(base_args, 'model')

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            bm = KVCacheBenchmark(base_args, logger=mock_logger, run_datetime="20250124_120000")
            meta = bm.metadata

        assert meta['model'] == KVCACHE_MODEL_DEFAULT
        assert meta['kvcache_model'] == KVCACHE_MODEL_DEFAULT
        # args was mutated to carry the default forward
        assert base_args.model == KVCACHE_MODEL_DEFAULT

    def test_metadata_without_distributed_info(self, base_args, mock_logger, tmp_path):
        """Verify metadata works correctly without distributed execution info."""
        # exec_type, hosts, num_processes are None by default in base_args

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information') as mock_cluster:
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            mock_cluster.return_value = None

            from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
            bm = KVCacheBenchmark(base_args, logger=mock_logger, run_datetime="20250124_120000")
            meta = bm.metadata

        # num_processes should be included but can be None
        assert 'num_processes' in meta
        assert meta['num_processes'] is None
        # hosts and exec_type should not be in metadata when not set
        assert 'hosts' not in meta
        assert 'exec_type' not in meta


# ---------------------------------------------------------------------------
# Helper fixture shared by AGG tests
# ---------------------------------------------------------------------------

def _make_run_benchmark(tmp_path, what_if=False):
    """Instantiate KVCacheBenchmark for command='run' with mocked deps."""
    args = Namespace(
        debug=False,
        verbose=False,
        what_if=what_if,
        stream_log_level='INFO',
        # mode='open' here so the strict CLOSED-mode override checks in
        # KVCacheBenchmark._execute_run (seed/trials/inter-option-delay)
        # don't fire — these tests deliberately override those args.
        # TestClosedEnforcement sets mode='closed' on bm.args explicitly.
        mode='open',
        orgname='Acme',
        systemname='sys-v1',
        results_dir=str(tmp_path),
        command='run',
        npernode=2,
        seed=42,
        cache_dir='/tmp/kv',
        trials=3,
        inter_option_delay=20,
        kvcache_bin_path=None,
        config=None,
        hosts=['localhost'],
        mpi_bin='mpirun',
        oversubscribe=False,
        allow_run_as_root=False,
        mpi_params=None,
        mpi_btl='auto',
        model='llama3.1-8b',
        num_users=100,
        duration=60,
        gpu_mem_gb=0,
        cpu_mem_gb=0,
        generation_mode='none',
        performance_profile='latency',
        num_processes=None,
        exec_type=None,
        closed=False,
        open=False,
    )
    output_dir = str(tmp_path / 'run_output')
    with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
         patch('mlpstorage_py.benchmarks.kvcache.KVCacheBenchmark._collect_cluster_information',
               return_value=None):
        mock_gen.return_value = output_dir
        from mlpstorage_py.benchmarks.kvcache import KVCacheBenchmark
        bm = KVCacheBenchmark(args, run_datetime='20260523_120000')
    bm.write_cluster_info = MagicMock()
    return bm


class TestInterruptibleSleep:
    """Tests for KVCacheBenchmark._interruptible_sleep."""

    def test_returns_immediately_with_zero_seconds(self, tmp_path):
        """Sleep of 0 seconds returns immediately."""
        bm = _make_run_benchmark(tmp_path)
        # Must not raise, must complete quickly
        bm._interruptible_sleep(0)

    def test_skips_sleep_in_what_if_mode(self, tmp_path):
        """_interruptible_sleep must return immediately when what_if=True."""
        bm = _make_run_benchmark(tmp_path, what_if=True)
        with patch('time.sleep') as mock_sleep:
            bm._interruptible_sleep(20)
        mock_sleep.assert_not_called()

    def test_calls_time_sleep_in_1s_chunks(self, tmp_path):
        """_interruptible_sleep(3) calls time.sleep(1) three times."""
        bm = _make_run_benchmark(tmp_path, what_if=False)
        with patch('time.sleep') as mock_sleep:
            bm._interruptible_sleep(3)
        assert mock_sleep.call_count == 3
        mock_sleep.assert_called_with(1)

    def test_propagates_keyboard_interrupt(self, tmp_path):
        """KeyboardInterrupt raised inside sleep should propagate out."""
        bm = _make_run_benchmark(tmp_path, what_if=False)
        with patch('time.sleep', side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                bm._interruptible_sleep(5)


class TestAggregateOptionResults:
    """Tests for KVCacheBenchmark._aggregate_option_results."""

    def _make_rank_file(self, rank_dir, bw, p95, storage_entries=100,
                        write_bw=0.0, avg_throughput=0.0, storage_throughput=0.0):
        """Write a synthetic rank output JSON file."""
        rank_dir.mkdir(parents=True, exist_ok=True)
        data = {
            'summary': {
                'cache_stats': {
                    'tier_storage_read_bandwidth_gbps': bw,
                    'tier_storage_write_bandwidth_gbps': write_bw,
                    'storage_entries': storage_entries,
                },
                'storage_io_latency_ms': {'p95': p95},
                'avg_throughput_tokens_per_sec': avg_throughput,
                'storage_throughput_tokens_per_sec': storage_throughput,
            }
        }
        (rank_dir / 'kvcache_results_20260523_120000.json').write_text(json.dumps(data))

    def test_sums_bandwidth_across_ranks(self, tmp_path):
        """aggregated_read_bandwidth_gbps == sum of all rank values."""
        bm = _make_run_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        self._make_rank_file(trial_dir / 'rank_0', bw=1.5, p95=10.0)
        self._make_rank_file(trial_dir / 'rank_1', bw=2.5, p95=15.0)

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=2)

        assert result['aggregated_read_bandwidth_gbps'] == pytest.approx(4.0)

    def test_takes_max_p95_latency_across_ranks(self, tmp_path):
        """aggregated_p95_latency_ms == max of all rank p95 values."""
        bm = _make_run_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        self._make_rank_file(trial_dir / 'rank_0', bw=1.5, p95=10.0)
        self._make_rank_file(trial_dir / 'rank_1', bw=2.5, p95=15.0)

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=2)

        assert result['aggregated_p95_latency_ms'] == pytest.approx(15.0)

    def test_no_partial_failure_when_all_files_present(self, tmp_path):
        """partial_failure is False when all rank files exist."""
        bm = _make_run_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        self._make_rank_file(trial_dir / 'rank_0', bw=1.0, p95=5.0)
        self._make_rank_file(trial_dir / 'rank_1', bw=2.0, p95=8.0)

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=2)

        assert result['partial_failure'] is False
        assert result['missing_files'] == []

    def test_partial_failure_when_rank_file_missing(self, tmp_path):
        """partial_failure is True when a rank directory has no result file."""
        bm = _make_run_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        # Only rank_0 present; rank_1 is missing
        self._make_rank_file(trial_dir / 'rank_0', bw=2.0, p95=8.0)
        # rank_1 directory exists but has no json file
        (trial_dir / 'rank_1').mkdir(parents=True, exist_ok=True)

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=2)

        assert result['partial_failure'] is True
        assert len(result['missing_files']) == 1

    def test_cpu_tier_ranks_populated_when_storage_entries_zero(self, tmp_path):
        """cpu_tier_ranks is populated and bandwidth included when storage_entries==0."""
        bm = _make_run_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        self._make_rank_file(trial_dir / 'rank_0', bw=0.0, p95=5.0, storage_entries=0)
        self._make_rank_file(trial_dir / 'rank_1', bw=0.0, p95=5.0, storage_entries=0)

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=2)

        # AGG-04: 0 bandwidth is included, not a failure
        assert result['aggregated_read_bandwidth_gbps'] == pytest.approx(0.0)
        assert result['partial_failure'] is False
        assert len(result['cpu_tier_ranks']) == 2

    def test_cpu_tier_log_message_contains_required_text(self, tmp_path):
        """Logger must log 'working set served from CPU tier' for storage_entries==0."""
        bm = _make_run_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        self._make_rank_file(trial_dir / 'rank_0', bw=0.0, p95=5.0, storage_entries=0)

        log_messages = []
        original_info = bm.logger.info
        def capture_info(msg, *a, **kw):
            log_messages.append(str(msg))
            if callable(original_info):
                try:
                    original_info(msg, *a, **kw)
                except Exception:
                    pass
        bm.logger.info = capture_info

        bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=1)

        assert any('working set served from CPU tier' in m for m in log_messages), \
            f"Expected 'working set served from CPU tier' in log messages, got: {log_messages}"

    def test_result_structure_has_required_keys(self, tmp_path):
        """Return dict must contain all AGG-06 required keys."""
        bm = _make_run_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        self._make_rank_file(trial_dir / 'rank_0', bw=1.0, p95=5.0)

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=1)

        required_keys = {
            'option',
            'aggregated_read_bandwidth_gbps', 'aggregated_write_bandwidth_gbps',
            'aggregated_avg_throughput_tokens_per_sec', 'aggregated_storage_throughput_tokens_per_sec',
            'aggregated_p95_latency_ms',
            'rank_count', 'trial_count', 'partial_failure', 'missing_files', 'cpu_tier_ranks',
        }
        assert required_keys.issubset(set(result.keys()))

    def test_aggregates_across_multiple_trials(self, tmp_path):
        """Aggregation spans multiple trial directories."""
        bm = _make_run_benchmark(tmp_path)
        trial_dirs = []
        for t in range(2):
            trial_dir = tmp_path / f'trial_{t}'
            self._make_rank_file(trial_dir / 'rank_0', bw=1.0, p95=10.0)
            trial_dirs.append(str(trial_dir))

        result = bm._aggregate_option_results(1, trial_dirs, expected_rank_count=1)

        # 2 trials × 1 rank × 1.0 GBps each → fmean([1.0, 1.0]) = 1.0
        assert result['aggregated_read_bandwidth_gbps'] == pytest.approx(1.0)
        assert result['trial_count'] == 2

    def test_uses_glob_not_constructed_filename(self, tmp_path):
        """Discovery must use glob so clock-drift timestamps are tolerated."""
        bm = _make_run_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        rank_dir = trial_dir / 'rank_0'
        rank_dir.mkdir(parents=True)
        # Write file with a different timestamp than run_datetime
        data = {
            'summary': {
                'cache_stats': {'tier_storage_read_bandwidth_gbps': 3.0, 'storage_entries': 50},
                'storage_io_latency_ms': {'p95': 7.0},
            }
        }
        (rank_dir / 'kvcache_results_20260523_130055.json').write_text(json.dumps(data))

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=1)

        assert result['aggregated_read_bandwidth_gbps'] == pytest.approx(3.0)
        assert result['partial_failure'] is False

    def test_none_p95_when_no_successful_reads(self, tmp_path):
        """aggregated_p95_latency_ms is 0.0 when all rank files are missing."""
        bm = _make_run_benchmark(tmp_path)
        trial_dir = tmp_path / 'trial_0'
        # Both rank dirs exist but have no json files
        (trial_dir / 'rank_0').mkdir(parents=True)
        (trial_dir / 'rank_1').mkdir(parents=True)

        result = bm._aggregate_option_results(1, [str(trial_dir)], expected_rank_count=2)

        # Empty trial contributes 0.0; fmean([0.0]) = 0.0
        assert result['aggregated_p95_latency_ms'] == pytest.approx(0.0)
        assert result['partial_failure'] is True


class TestWriteRunSummary:
    """Tests for KVCacheBenchmark._write_run_summary."""

    def _option_result(self, option=1, bw=3.0, p95=12.0, partial=False):
        return {
            'option': option,
            'aggregated_read_bandwidth_gbps': bw,
            'aggregated_write_bandwidth_gbps': 0.0,
            'aggregated_avg_throughput_tokens_per_sec': 0.0,
            'aggregated_storage_throughput_tokens_per_sec': 0.0,
            'aggregated_p95_latency_ms': p95,
            'rank_count': 2,
            'trial_count': 1,
            'partial_failure': partial,
            'missing_files': [],
            'cpu_tier_ranks': [],
        }

    def test_writes_file_to_run_result_output(self, tmp_path):
        """Summary JSON is written to run_result_output with correct filename."""
        bm = _make_run_benchmark(tmp_path)
        output_dir = str(tmp_path / 'summary_out')
        os.makedirs(output_dir, exist_ok=True)
        bm.run_result_output = output_dir

        option_results = {1: self._option_result()}
        bm._write_run_summary(option_results, npernode=2, host_count=1, total_ranks=2, trials=3)

        expected = Path(output_dir) / 'kvcache_run_summary_20260523_120000.json'
        assert expected.exists(), f"Expected summary at {expected}"

    def test_schema_version_is_1_0(self, tmp_path):
        """Written JSON must have schema_version='1.0'."""
        bm = _make_run_benchmark(tmp_path)
        output_dir = str(tmp_path / 'summary_out')
        os.makedirs(output_dir, exist_ok=True)
        bm.run_result_output = output_dir

        bm._write_run_summary({1: self._option_result()}, npernode=2, host_count=1, total_ranks=2, trials=3)

        with open(Path(output_dir) / 'kvcache_run_summary_20260523_120000.json') as f:
            data = json.load(f)
        assert data['schema_version'] == '1.0'

    def test_summary_includes_required_keys(self, tmp_path):
        """JSON must contain all AGG-06 top-level keys."""
        bm = _make_run_benchmark(tmp_path)
        output_dir = str(tmp_path / 'summary_out')
        os.makedirs(output_dir, exist_ok=True)
        bm.run_result_output = output_dir

        bm._write_run_summary({1: self._option_result()}, npernode=2, host_count=1, total_ranks=2, trials=3)

        with open(Path(output_dir) / 'kvcache_run_summary_20260523_120000.json') as f:
            data = json.load(f)

        required = {'schema_version', 'run_datetime', 'npernode', 'host_count',
                    'total_ranks', 'trials_per_option', 'options', 'partial_failure'}
        assert required.issubset(set(data.keys()))

    def test_partial_failure_true_when_any_option_fails(self, tmp_path):
        """Top-level partial_failure is True when any option has partial_failure=True."""
        bm = _make_run_benchmark(tmp_path)
        output_dir = str(tmp_path / 'summary_out')
        os.makedirs(output_dir, exist_ok=True)
        bm.run_result_output = output_dir

        option_results = {
            1: self._option_result(partial=False),
            2: self._option_result(option=2, partial=True),
            3: self._option_result(option=3, partial=False),
        }
        bm._write_run_summary(option_results, npernode=2, host_count=1, total_ranks=2, trials=3)

        with open(Path(output_dir) / 'kvcache_run_summary_20260523_120000.json') as f:
            data = json.load(f)
        assert data['partial_failure'] is True

    def test_partial_failure_false_when_no_option_fails(self, tmp_path):
        """Top-level partial_failure is False when no option has partial_failure=True."""
        bm = _make_run_benchmark(tmp_path)
        output_dir = str(tmp_path / 'summary_out')
        os.makedirs(output_dir, exist_ok=True)
        bm.run_result_output = output_dir

        option_results = {1: self._option_result(partial=False)}
        bm._write_run_summary(option_results, npernode=2, host_count=1, total_ranks=2, trials=3)

        with open(Path(output_dir) / 'kvcache_run_summary_20260523_120000.json') as f:
            data = json.load(f)
        assert data['partial_failure'] is False

    def test_options_key_contains_per_option_data(self, tmp_path):
        """The 'options' key must hold the per-option result dict."""
        bm = _make_run_benchmark(tmp_path)
        output_dir = str(tmp_path / 'summary_out')
        os.makedirs(output_dir, exist_ok=True)
        bm.run_result_output = output_dir

        option_results = {1: self._option_result(bw=5.0)}
        bm._write_run_summary(option_results, npernode=2, host_count=1, total_ranks=2, trials=3)

        with open(Path(output_dir) / 'kvcache_run_summary_20260523_120000.json') as f:
            data = json.load(f)
        # JSON keys are strings after serialization
        options = data['options']
        assert '1' in options or 1 in options

    def test_write_run_summary_does_not_raise_with_float_values(self, tmp_path):
        """MLPSJsonEncoder must serialize standard Python floats without raising (G5)."""
        bm = _make_run_benchmark(tmp_path)
        output_dir = str(tmp_path / 'summary_g5')
        os.makedirs(output_dir, exist_ok=True)
        bm.run_result_output = output_dir

        option_result = {
            'option': 1,
            'aggregated_read_bandwidth_gbps': float(3.5),
            'aggregated_write_bandwidth_gbps': float(0.0),
            'aggregated_avg_throughput_tokens_per_sec': float(0.0),
            'aggregated_storage_throughput_tokens_per_sec': float(0.0),
            'aggregated_p95_latency_ms': float(12.0),
            'rank_count': 2,
            'trial_count': 1,
            'partial_failure': False,
            'missing_files': [],
            'cpu_tier_ranks': [],
        }
        bm._write_run_summary({1: option_result}, npernode=2, host_count=1, total_ranks=2, trials=3)
        summary_files = list(Path(output_dir).glob('kvcache_run_summary_*.json'))
        assert len(summary_files) == 1


class TestExecuteRun:
    """Tests for KVCacheBenchmark._execute_run and command_method_map wiring.

    Covers:
    - command_method_map contains 'run' key mapping to _execute_run (DIST-01)
    - _execute_run returns 0
    - _execute_command called 3 times per run (once per option) with trials=1 (DIST-02, DIST-04)
    - mpirun command contains '--mca orte_abort_on_non_zero_status 0' (DIST-08)
    - mpirun command contains '--npernode N' (DIST-03)
    - wrapper receives --seed-base, --rank-output-base, --rank-cache-base (DIST-07)
    - per-option/trial dirs created with correct naming (option_{N}/trial_{T}/)
    - _interruptible_sleep called 2 times (after options 1 and 2; not after 3) (DIST-05)
    - _aggregate_option_results called 3 times when what_if=False
    - _aggregate_option_results NOT called when what_if=True (DIST-06)
    - _write_run_summary called once when what_if=False
    - _write_run_summary NOT called when what_if=True
    - write_metadata called regardless of what_if
    - CLOSED enforcement: hard-fails on illegal seed/trials/inter-option-delay/config overrides
    """

    @pytest.fixture
    def bm(self, tmp_path):
        """Benchmark instance for validate command."""
        return _make_run_benchmark(tmp_path, what_if=False)

    @pytest.fixture
    def bm_whatif(self, tmp_path):
        """Benchmark instance for validate --what-if."""
        return _make_run_benchmark(tmp_path, what_if=True)

    @pytest.fixture
    def fake_agg_result(self):
        """Return value for _aggregate_option_results mock."""
        return {
            'option': 1,
            'aggregated_read_bandwidth_gbps': 0.0,
            'aggregated_write_bandwidth_gbps': 0.0,
            'aggregated_avg_throughput_tokens_per_sec': 0.0,
            'aggregated_storage_throughput_tokens_per_sec': 0.0,
            'aggregated_p95_latency_ms': 0.0,
            'rank_count': 2,
            'trial_count': 1,
            'partial_failure': False,
            'missing_files': [],
            'cpu_tier_ranks': [],
        }

    def test_run_in_command_method_map(self, bm):
        """'run' key must exist in command_method_map and map to _execute_run."""
        assert 'run' in bm.command_method_map
        assert bm.command_method_map['run'] == bm._execute_run

    def test_execute_run_returns_zero(self, bm, fake_agg_result):
        """_execute_run must return 0 on success."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            rc = bm._execute_run()
        assert rc == 0

    def test_execute_command_called_3_times_for_3_options(self, bm, fake_agg_result):
        """_execute_command must be called once per option (3x) with trials=1."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        assert len(executed_cmds) == 3, f"Expected 3 _execute_command calls, got {len(executed_cmds)}"

    def test_mpirun_contains_mca_orte_flag(self, bm, fake_agg_result):
        """mpirun command must contain '--mca orte_abort_on_non_zero_status 0' (DIST-08)."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        assert executed_cmds, "No commands were executed"
        cmd0 = executed_cmds[0]
        assert '--mca orte_abort_on_non_zero_status 0' in cmd0, \
            f"Missing --mca orte_abort_on_non_zero_status 0 in: {cmd0}"

    def test_mpirun_passes_through_user_mpi_params(self, bm, fake_agg_result):
        """User --mpi-params must reach mpirun, ordered before the mandatory
        --mca orte_abort_on_non_zero_status 0 flag so OpenMPI's last-wins
        resolution keeps the abort-suppression authoritative (#520)."""
        # Shape matches what cli_parser.py emits post-shlex.split.
        bm.args.mpi_params = ['-genv', 'PMI_VERSION=2', '--mca', 'btl', 'tcp,self']
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        cmd0 = executed_cmds[0]
        assert '-genv PMI_VERSION=2' in cmd0, \
            f"Missing user --mpi-params in: {cmd0}"
        assert '--mca btl tcp,self' in cmd0, \
            f"Missing user --mpi-params in: {cmd0}"
        # Mandatory --mca must follow user params (OpenMPI last-wins).
        user_idx = cmd0.index('--mca btl tcp,self')
        mandatory_idx = cmd0.index('--mca orte_abort_on_non_zero_status 0')
        assert mandatory_idx > user_idx, \
            f"Mandatory --mca must come after user --mpi-params in: {cmd0}"

    def test_mpirun_contains_npernode(self, bm, fake_agg_result):
        """mpirun command must contain '--npernode 2' when npernode=2 (DIST-03)."""
        bm.args.npernode = 2
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        assert '--npernode 2' in executed_cmds[0], \
            f"Missing --npernode 2 in: {executed_cmds[0]}"

    def test_wrapper_receives_rank_bases_and_seed_base(self, bm, fake_agg_result):
        """Wrapper command must include --rank-output-base, --rank-cache-base, --seed-base (DIST-07).

        The wrapper API was reshaped for #498/#500: the wrapper no longer takes
        --option and no longer encodes WORKLOAD_PARAMS. Per-option kv-cache.py
        args are emitted by mlpstorage_py.benchmarks.kvcache and pass through
        the wrapper via parse_known_args."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        bm.args.seed = 42
        bm.args.cache_dir = '/tmp/kv'
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        cmd0 = executed_cmds[0]
        assert '--seed-base 42' in cmd0, f"Missing --seed-base 42 in: {cmd0}"
        assert '--rank-output-base' in cmd0, f"Missing --rank-output-base in: {cmd0}"
        assert '--rank-cache-base /tmp/kv' in cmd0, f"Missing --rank-cache-base in: {cmd0}"
        # The legacy --option flag is no longer part of the wrapper API.
        assert '--option ' not in cmd0, f"Stale --option flag in: {cmd0}"

    def test_per_option_trial_dirs_created(self, bm, fake_agg_result, tmp_path):
        """option_{N}/trial_{T}/ directories must be created."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        # At least one option/trial dir must exist beneath run_result_output
        run_out = Path(bm.run_result_output)
        option1_trial0 = run_out / 'option_1' / 'trial_0'
        assert option1_trial0.exists(), f"Expected {option1_trial0} to exist"

    def test_option_trial_dirs_in_command_path(self, bm, fake_agg_result):
        """Command must reference option_N/trial_T subdirectory in --rank-output-base."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        cmd0 = executed_cmds[0]
        assert 'option_1' in cmd0, f"Missing option_1 in: {cmd0}"
        assert 'trial_0' in cmd0, f"Missing trial_0 in: {cmd0}"

    def test_interruptible_sleep_called_2_times_not_3(self, bm, fake_agg_result):
        """_interruptible_sleep called after options 1 and 2 but NOT after option 3."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 5
        sleep_calls = []
        def fake_sleep(seconds):
            sleep_calls.append(seconds)
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep', side_effect=fake_sleep), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        assert len(sleep_calls) == 2, f"Expected 2 sleep calls, got {len(sleep_calls)}"
        assert all(s == 5 for s in sleep_calls), f"Expected delay=5 for all, got {sleep_calls}"

    def test_aggregate_called_3_times_when_not_what_if(self, bm, fake_agg_result):
        """_aggregate_option_results called once per option (3x) when what_if=False."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result) as mock_agg, \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        assert mock_agg.call_count == 3, f"Expected 3 aggregate calls, got {mock_agg.call_count}"

    def test_aggregate_not_called_when_what_if(self, bm_whatif):
        """_aggregate_option_results must NOT be called when what_if=True."""
        bm_whatif.args.trials = 1
        bm_whatif.args.inter_option_delay = 0
        with patch.object(bm_whatif, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm_whatif, '_interruptible_sleep'), \
             patch.object(bm_whatif, '_aggregate_option_results') as mock_agg, \
             patch.object(bm_whatif, '_write_run_summary') as mock_ws, \
             patch.object(bm_whatif, 'write_metadata'):
            bm_whatif._execute_run()
        assert mock_agg.call_count == 0, f"Expected 0 aggregate calls in what-if, got {mock_agg.call_count}"

    def test_write_summary_called_once_when_not_what_if(self, bm, fake_agg_result):
        """_write_run_summary called once after all options when what_if=False."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary') as mock_ws, \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        assert mock_ws.call_count == 1, f"Expected 1 summary write, got {mock_ws.call_count}"

    def test_write_summary_not_called_when_what_if(self, bm_whatif):
        """_write_run_summary must NOT be called when what_if=True."""
        bm_whatif.args.trials = 1
        bm_whatif.args.inter_option_delay = 0
        with patch.object(bm_whatif, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm_whatif, '_interruptible_sleep'), \
             patch.object(bm_whatif, '_aggregate_option_results'), \
             patch.object(bm_whatif, '_write_run_summary') as mock_ws, \
             patch.object(bm_whatif, 'write_metadata'):
            bm_whatif._execute_run()
        assert mock_ws.call_count == 0, f"Expected 0 summary writes in what-if, got {mock_ws.call_count}"

    def test_write_metadata_called_regardless_of_what_if(self, bm_whatif):
        """write_metadata must be called even in what-if mode."""
        bm_whatif.args.trials = 1
        bm_whatif.args.inter_option_delay = 0
        with patch.object(bm_whatif, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm_whatif, '_interruptible_sleep'), \
             patch.object(bm_whatif, '_aggregate_option_results'), \
             patch.object(bm_whatif, '_write_run_summary'), \
             patch.object(bm_whatif, 'write_metadata') as mock_meta:
            bm_whatif._execute_run()
        assert mock_meta.call_count == 1, "write_metadata must be called even in what-if"

    def test_multiple_trials_per_option(self, bm, fake_agg_result):
        """With trials=3, _execute_command called 9 times (3 options × 3 trials)."""
        bm.args.trials = 3
        bm.args.inter_option_delay = 0
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        assert len(executed_cmds) == 9, f"Expected 9 commands (3 options × 3 trials), got {len(executed_cmds)}"

    def test_execute_command_targets_mlperf_wrapper_not_kvcache(self, bm, fake_agg_result):
        """Command must reference mlperf_wrapper.py, not kv-cache.py (G2)."""
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        executed_cmds = []
        def fake_execute(cmd, **kwargs):
            executed_cmds.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        cmd0 = executed_cmds[0]
        assert 'mlperf_wrapper.py' in cmd0, f"Expected mlperf_wrapper.py in cmd, got: {cmd0}"
        assert 'kv-cache.py' not in cmd0, f"Must not reference kv-cache.py directly: {cmd0}"

    def test_aggregate_receives_correct_trial_dirs_for_two_trials(self, bm, fake_agg_result):
        """_aggregate_option_results must receive trial_dirs for all trials (G3)."""
        bm.args.trials = 2
        bm.args.inter_option_delay = 0
        agg_calls = []
        def fake_agg(option, trial_dirs, expected_rank_count):
            agg_calls.append((option, list(trial_dirs)))
            return fake_agg_result
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', side_effect=fake_agg), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        # agg_calls[0] is for option 1
        option_1_dirs = agg_calls[0][1]
        assert len(option_1_dirs) == 2
        assert any('trial_0' in str(d) for d in option_1_dirs)
        assert any('trial_1' in str(d) for d in option_1_dirs)

    def test_localhost_fallback_when_hosts_is_none(self, bm, fake_agg_result):
        """When hosts is None, _execute_run must complete without raising (G4)."""
        bm.args.hosts = None
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            rc = bm._execute_run()
        assert rc == 0


class TestClosedEnforcement:
    """Tests for CLOSED submission enforcement in _execute_run."""

    @pytest.fixture
    def bm(self, tmp_path):
        bm = _make_run_benchmark(tmp_path)
        bm.args.mode = 'closed'
        return bm

    def test_closed_seed_non_42_returns_1(self, bm):
        """CLOSED: --seed != 42 must hard-fail with return code 1."""
        bm.args.seed = 99
        rc = bm._execute_run()
        assert rc == 1

    def test_closed_seed_42_is_allowed(self, bm, tmp_path):
        """CLOSED: --seed 42 (the mandated value) must not fail."""
        bm.args.seed = 42
        # Keep trials=3 and inter_option_delay=20 (CLOSED mandated values from fixture)
        _agg = {
            'option': 1, 'aggregated_read_bandwidth_gbps': 0.0,
            'aggregated_write_bandwidth_gbps': 0.0,
            'aggregated_avg_throughput_tokens_per_sec': 0.0,
            'aggregated_storage_throughput_tokens_per_sec': 0.0,
            'aggregated_p95_latency_ms': 0.0,
            'rank_count': 1, 'trial_count': 3,
            'partial_failure': False, 'missing_files': [], 'cpu_tier_ranks': [],
        }
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=_agg), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            rc = bm._execute_run()
        assert rc == 0

    def test_closed_seed_none_uses_default_42(self, bm, tmp_path):
        """CLOSED: seed=None (not set by user) must not fail (default 42 applies)."""
        bm.args.seed = None
        # Keep trials=3 and inter_option_delay=20 (CLOSED mandated values from fixture)
        _agg = {
            'option': 1, 'aggregated_read_bandwidth_gbps': 0.0,
            'aggregated_write_bandwidth_gbps': 0.0,
            'aggregated_avg_throughput_tokens_per_sec': 0.0,
            'aggregated_storage_throughput_tokens_per_sec': 0.0,
            'aggregated_p95_latency_ms': 0.0,
            'rank_count': 1, 'trial_count': 3,
            'partial_failure': False, 'missing_files': [], 'cpu_tier_ranks': [],
        }
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=_agg), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            rc = bm._execute_run()
        assert rc == 0

    def test_closed_trials_non_3_returns_1(self, bm):
        """CLOSED: --trials != 3 must hard-fail with return code 1."""
        bm.args.trials = 5
        rc = bm._execute_run()
        assert rc == 1

    def test_closed_inter_option_delay_non_20_returns_1(self, bm):
        """CLOSED: --inter-option-delay != 20 must hard-fail with return code 1."""
        bm.args.inter_option_delay = 10
        rc = bm._execute_run()
        assert rc == 1

    def test_closed_config_set_returns_1(self, bm):
        """CLOSED: --config set to any value must hard-fail with return code 1."""
        bm.args.config = '/path/to/config.yaml'
        rc = bm._execute_run()
        assert rc == 1

    def test_open_seed_override_allowed(self, tmp_path):
        """OPEN: custom --seed must be accepted (no enforcement)."""
        bm = _make_run_benchmark(tmp_path)
        bm.args.closed = False
        bm.args.seed = 99
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        with patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value={
                 'option': 1, 'aggregated_read_bandwidth_gbps': 0.0,
                 'aggregated_write_bandwidth_gbps': 0.0,
                 'aggregated_avg_throughput_tokens_per_sec': 0.0,
                 'aggregated_storage_throughput_tokens_per_sec': 0.0,
                 'aggregated_p95_latency_ms': 0.0,
                 'rank_count': 1, 'trial_count': 1,
                 'partial_failure': False, 'missing_files': [], 'cpu_tier_ranks': [],
             }), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            rc = bm._execute_run()
        assert rc == 0


class TestWorkloadParamsConstant:
    """WORKLOAD_PARAMS lives in mlpstorage_py.benchmarks.kvcache — the single
    source of truth for per-option MLPerf v3.0 workloads. Previously it lived
    in kv_cache_benchmark/mlperf_wrapper.py; centralizing it here is what
    closes issues #498 and #500."""

    def test_options_are_1_2_3(self):
        from mlpstorage_py.benchmarks.kvcache import WORKLOAD_PARAMS
        assert set(WORKLOAD_PARAMS.keys()) == {1, 2, 3}

    def test_option1_model_is_8b(self):
        from mlpstorage_py.benchmarks.kvcache import WORKLOAD_PARAMS
        assert WORKLOAD_PARAMS[1]['model'] == 'llama3.1-8b'

    def test_option3_model_is_70b(self):
        from mlpstorage_py.benchmarks.kvcache import WORKLOAD_PARAMS
        assert WORKLOAD_PARAMS[3]['model'] == 'llama3.1-70b-instruct'

    def test_generation_mode_always_none(self):
        from mlpstorage_py.benchmarks.kvcache import WORKLOAD_PARAMS
        for opt in (1, 2, 3):
            assert WORKLOAD_PARAMS[opt]['generation-mode'] == 'none'


class TestBuildOptionKvcacheArgs:
    """Per-option kv-cache.py CLI args returned by _build_option_kvcache_args.

    Verifies CLOSED uses the mandated WORKLOAD_PARAMS verbatim and OPEN lets
    user-set CLI flags supersede the per-option defaults — the behavior the
    old in-wrapper squashing prevented (issues #498 and #500)."""

    def test_closed_emits_workload_params_verbatim_for_option1(self, tmp_path):
        from mlpstorage_py.benchmarks.kvcache import WORKLOAD_PARAMS
        bm = _make_run_benchmark(tmp_path)
        # Even if args carry non-default values, CLOSED ignores them.
        bm.args.model = 'llama3.1-70b-instruct'
        bm.args.num_users = 999
        out = bm._build_option_kvcache_args(1, is_closed=True)
        assert '--model' in out and out[out.index('--model') + 1] == WORKLOAD_PARAMS[1]['model']
        assert '--num-users' in out and out[out.index('--num-users') + 1] == str(WORKLOAD_PARAMS[1]['num-users'])

    def test_closed_emits_option3_70b_model(self, tmp_path):
        bm = _make_run_benchmark(tmp_path)
        out = bm._build_option_kvcache_args(3, is_closed=True)
        assert out[out.index('--model') + 1] == 'llama3.1-70b-instruct'

    def test_open_user_args_override_defaults(self, tmp_path):
        """Issue #498: in OPEN, user CLI args must reach kv-cache.py."""
        bm = _make_run_benchmark(tmp_path)
        bm.args.model = 'llama3.1-70b-instruct'
        bm.args.num_users = 42
        bm.args.duration = 600
        bm.args.gpu_mem_gb = 8
        bm.args.cpu_mem_gb = 16
        bm.args.generation_mode = 'realistic'
        out = bm._build_option_kvcache_args(1, is_closed=False)
        assert out[out.index('--model') + 1] == 'llama3.1-70b-instruct'
        assert out[out.index('--num-users') + 1] == '42'
        assert out[out.index('--duration') + 1] == '600'
        assert out[out.index('--gpu-mem-gb') + 1] == '8'
        assert out[out.index('--cpu-mem-gb') + 1] == '16'
        assert out[out.index('--generation-mode') + 1] == 'realistic'

    def test_open_falls_back_to_workload_params_when_args_missing(self, tmp_path):
        """In OPEN, attributes the user did not set come from WORKLOAD_PARAMS."""
        from mlpstorage_py.benchmarks.kvcache import WORKLOAD_PARAMS
        bm = _make_run_benchmark(tmp_path)
        for attr in ('model', 'num_users', 'duration', 'gpu_mem_gb',
                     'cpu_mem_gb', 'generation_mode'):
            if hasattr(bm.args, attr):
                delattr(bm.args, attr)
        out = bm._build_option_kvcache_args(2, is_closed=False)
        assert out[out.index('--model') + 1] == WORKLOAD_PARAMS[2]['model']
        assert out[out.index('--num-users') + 1] == str(WORKLOAD_PARAMS[2]['num-users'])
        assert out[out.index('--cpu-mem-gb') + 1] == str(WORKLOAD_PARAMS[2]['cpu-mem-gb'])

    def test_max_concurrent_allocs_always_from_workload_params(self, tmp_path):
        """max-concurrent-allocs is not user-exposed even in OPEN — it
        always tracks the per-option WORKLOAD_PARAMS value."""
        from mlpstorage_py.benchmarks.kvcache import WORKLOAD_PARAMS
        bm = _make_run_benchmark(tmp_path)
        out_open = bm._build_option_kvcache_args(3, is_closed=False)
        out_closed = bm._build_option_kvcache_args(3, is_closed=True)
        expected = str(WORKLOAD_PARAMS[3]['max-concurrent-allocs'])
        assert out_open[out_open.index('--max-concurrent-allocs') + 1] == expected
        assert out_closed[out_closed.index('--max-concurrent-allocs') + 1] == expected


class TestWrapperCommandForwardsPerOptionArgs:
    """End-to-end check that the wrapper invocation built by _execute_run
    contains the per-option model/num-users/etc., so kv-cache.py receives
    them downstream of mlperf_wrapper.py."""

    def _capture_first_cmd(self, bm, fake_agg_result):
        executed = []
        def fake_execute(cmd, **kwargs):
            executed.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        return executed[0]

    def _fake_agg(self):
        return {
            'option': 1, 'aggregated_read_bandwidth_gbps': 0.0,
            'aggregated_write_bandwidth_gbps': 0.0,
            'aggregated_avg_throughput_tokens_per_sec': 0.0,
            'aggregated_storage_throughput_tokens_per_sec': 0.0,
            'aggregated_p95_latency_ms': 0.0,
            'rank_count': 1, 'trial_count': 1,
            'partial_failure': False, 'missing_files': [], 'cpu_tier_ranks': [],
        }

    def test_closed_option1_wrapper_cmd_contains_8b_model(self, tmp_path):
        bm = _make_run_benchmark(tmp_path)
        bm.args.mode = 'closed'
        # CLOSED requires trials=3 and inter_option_delay=20 (the fixture's
        # values); overriding them would trigger CLOSED enforcement and
        # short-circuit before any command is built.
        cmd0 = self._capture_first_cmd(bm, self._fake_agg())
        assert '--model llama3.1-8b' in cmd0
        assert '--num-users 200' in cmd0
        assert '--max-concurrent-allocs 16' in cmd0

    def test_open_user_model_appears_in_wrapper_cmd(self, tmp_path):
        """Issue #498: user --model must reach the wrapper-built command."""
        bm = _make_run_benchmark(tmp_path)
        bm.args.mode = 'open'
        bm.args.model = 'llama3.1-70b-instruct'
        bm.args.num_users = 33
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        cmd0 = self._capture_first_cmd(bm, self._fake_agg())
        assert '--model llama3.1-70b-instruct' in cmd0
        assert '--num-users 33' in cmd0

    def test_wrapper_cmd_contains_config_path(self, tmp_path):
        """mlpstorage now owns the config path; it must be passed to the wrapper."""
        bm = _make_run_benchmark(tmp_path)
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        cmd0 = self._capture_first_cmd(bm, self._fake_agg())
        assert '--config' in cmd0

    def _capture_all_cmds(self, bm, fake_agg_result):
        """Capture every per-option/per-trial wrapper command built by _execute_run."""
        executed = []
        def fake_execute(cmd, **kwargs):
            executed.append(cmd)
            return ('', '', 0)
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=fake_agg_result), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        return executed

    def test_open_option3_defaults_to_70b_when_user_did_not_override(self, tmp_path):
        """OPEN with no --model override: option 3 must still receive the
        MLPerf-mandated llama70b model from WORKLOAD_PARAMS[3]. Closes the
        gap between _build_option_kvcache_args (covered) and the actual
        wrapper command (was uncovered)."""
        bm = _make_run_benchmark(tmp_path)
        bm.args.mode = 'open'
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        # Remove the fixture's model/num-users defaults so the fallback path
        # in _build_option_kvcache_args is exercised — this mirrors the case
        # where the OPEN user invoked the run without those flags.
        for attr in ('model', 'num_users'):
            if hasattr(bm.args, attr):
                delattr(bm.args, attr)
        cmds = self._capture_all_cmds(bm, self._fake_agg())
        assert len(cmds) == 3, f"Expected 3 commands (one per option), got {len(cmds)}"
        # cmds[0] is option 1, cmds[1] is option 2, cmds[2] is option 3
        assert '--model llama3.1-8b' in cmds[0]
        assert '--num-users 200' in cmds[0]
        assert '--model llama3.1-8b' in cmds[1]
        assert '--num-users 100' in cmds[1]
        assert '--model llama3.1-70b-instruct' in cmds[2]
        assert '--num-users 70' in cmds[2]

    def test_open_user_config_path_supersedes_wrapper_default(self, tmp_path):
        """OPEN: when the user provides --config /custom/path.yaml, mlpstorage
        forwards that exact path to the wrapper instead of the kv-cache.py
        adjacent default. (CLOSED rejects --config — covered separately.)"""
        bm = _make_run_benchmark(tmp_path)
        bm.args.mode = 'open'
        bm.args.config = '/custom/path/user.yaml'
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        cmd0 = self._capture_first_cmd(bm, self._fake_agg())
        assert '--config /custom/path/user.yaml' in cmd0, \
            f"User config path missing from cmd: {cmd0}"
        # And the wrapper-adjacent default must NOT also be present.
        assert 'kv_cache_benchmark/config.yaml' not in cmd0, \
            f"Default config path leaked into cmd alongside user override: {cmd0}"


class TestResolveRankLayout:
    """Tests for KVCacheBenchmark._resolve_rank_layout (issue #500).

    --num-processes was previously a dead flag — registered on the kvcache
    distributed-execution group but never read by _execute_run. Total ranks
    were computed solely from npernode * len(hosts), so a user passing
    --num-processes 2 with the default --npernode 1 still got one rank per
    host. This class covers the new resolver semantics:

      - --num-processes is total cluster ranks (matches the flag's help
        text and DLIO's --num-accelerators convention).
      - --npernode is ranks per host.
      - When only one is set, the other is derived.
      - When both are set, they must be consistent.
      - --num-processes must divide evenly across the host list.
    """

    def test_only_npernode_uses_today_behavior(self, tmp_path):
        bm = _make_run_benchmark(tmp_path)
        bm.args.num_processes = None
        bm.args.npernode = 4
        npn, total = bm._resolve_rank_layout(['h1', 'h2'])
        assert npn == 4
        assert total == 8

    def test_only_num_processes_single_host(self, tmp_path):
        """The exact scenario dslik reproduced in issue #500."""
        bm = _make_run_benchmark(tmp_path)
        bm.args.num_processes = 2
        bm.args.npernode = 1  # the CLI default — treated as 'not explicitly set'
        npn, total = bm._resolve_rank_layout(['localhost'])
        assert total == 2, "num_processes=2 with one host must produce 2 total ranks"
        assert npn == 2

    def test_only_num_processes_multi_host_even_divide(self, tmp_path):
        bm = _make_run_benchmark(tmp_path)
        bm.args.num_processes = 8
        bm.args.npernode = 1
        npn, total = bm._resolve_rank_layout(['h1', 'h2', 'h3', 'h4'])
        assert total == 8
        assert npn == 2

    def test_num_processes_not_divisible_by_hosts_fails(self, tmp_path):
        bm = _make_run_benchmark(tmp_path)
        bm.args.num_processes = 5
        bm.args.npernode = 1
        npn, total = bm._resolve_rank_layout(['h1', 'h2'])
        assert (npn, total) == (None, None)

    def test_both_set_consistent(self, tmp_path):
        bm = _make_run_benchmark(tmp_path)
        bm.args.num_processes = 6
        bm.args.npernode = 3
        npn, total = bm._resolve_rank_layout(['h1', 'h2'])
        assert npn == 3
        assert total == 6

    def test_both_set_inconsistent_fails(self, tmp_path):
        bm = _make_run_benchmark(tmp_path)
        bm.args.num_processes = 5
        bm.args.npernode = 3
        npn, total = bm._resolve_rank_layout(['h1', 'h2'])
        assert (npn, total) == (None, None)

    def test_neither_set_defaults_one_per_host(self, tmp_path):
        bm = _make_run_benchmark(tmp_path)
        bm.args.num_processes = None
        bm.args.npernode = None
        npn, total = bm._resolve_rank_layout(['h1', 'h2', 'h3'])
        assert npn == 1
        assert total == 3

    def test_negative_num_processes_fails(self, tmp_path):
        bm = _make_run_benchmark(tmp_path)
        bm.args.num_processes = -1
        bm.args.npernode = 1
        npn, total = bm._resolve_rank_layout(['h1'])
        assert (npn, total) == (None, None)


class TestExecuteRunHonorsNumProcesses:
    """End-to-end check that _execute_run actually uses --num-processes.

    Repro of issue #500: with --num-processes 2 on a single host, the
    wrapper-launch mpirun command must request 2 ranks (not 1)."""

    def _capture_first_cmd(self, bm):
        executed = []
        def fake_execute(cmd, **kwargs):
            executed.append(cmd)
            return ('', '', 0)
        agg = {
            'option': 1, 'aggregated_read_bandwidth_gbps': 0.0,
            'aggregated_write_bandwidth_gbps': 0.0,
            'aggregated_avg_throughput_tokens_per_sec': 0.0,
            'aggregated_storage_throughput_tokens_per_sec': 0.0,
            'aggregated_p95_latency_ms': 0.0,
            'rank_count': 1, 'trial_count': 1,
            'partial_failure': False, 'missing_files': [], 'cpu_tier_ranks': [],
        }
        with patch.object(bm, '_execute_command', side_effect=fake_execute), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=agg), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        return executed[0] if executed else None

    def test_num_processes_2_single_host_launches_2_ranks(self, tmp_path):
        bm = _make_run_benchmark(tmp_path)
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        bm.args.num_processes = 2
        bm.args.hosts = ['localhost']
        cmd = self._capture_first_cmd(bm)
        assert cmd is not None, "Expected at least one wrapper command"
        assert '-n 2 ' in cmd, f"Expected '-n 2' in mpirun cmd, got: {cmd}"
        assert '--npernode 2' in cmd, f"Expected '--npernode 2' in mpirun cmd, got: {cmd}"

    def test_num_processes_inconsistent_fails_run(self, tmp_path):
        bm = _make_run_benchmark(tmp_path)
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        bm.args.num_processes = 5  # not divisible by 2 hosts
        bm.args.hosts = ['h1', 'h2']
        with patch.object(bm, '_execute_command') as mock_exec, \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results'), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            rc = bm._execute_run()
        assert rc == 1
        mock_exec.assert_not_called()


class TestProbeResultsDirShared:
    """Tests for KVCacheBenchmark._probe_results_dir_shared.

    Issue #521: kvcache must fail fast when --results-dir is not visible at
    the same path on every host in --hosts. Otherwise rank result files
    written on remote nodes are invisible to the controller's aggregation.
    """

    def test_localhost_only_is_noop(self, tmp_path):
        """All-localhost runs cannot exhibit the bug; probe must not spawn mpi."""
        bm = _make_run_benchmark(tmp_path)
        with patch.object(bm, '_execute_command') as mock_exec:
            bm._probe_results_dir_shared(['localhost'])
        mock_exec.assert_not_called()

    def test_single_host_is_noop(self, tmp_path):
        """A single-host run (even non-localhost) cannot scatter results."""
        bm = _make_run_benchmark(tmp_path)
        with patch.object(bm, '_execute_command') as mock_exec:
            bm._probe_results_dir_shared(['h1'])
        mock_exec.assert_not_called()

    def test_passes_when_all_hosts_write_sentinel(self, tmp_path):
        """Shared FS: each host writes its sentinel; probe returns cleanly."""
        bm = _make_run_benchmark(tmp_path)

        def fake_execute(cmd, **kwargs):
            # Simulate every host successfully landing its sentinel in the
            # probe dir (which is what a shared FS would produce).
            probe_dir = Path(bm.run_result_output) / '.fs_probe'
            assert probe_dir.exists(), "probe dir should be pre-created"
            # Extract probe_id from the embedded inline script in the command.
            import re
            m = re.search(r"'([0-9a-f]{12})__rank'", cmd)
            assert m is not None, f"probe_id not found in cmd: {cmd}"
            probe_id = m.group(1)
            for rank, host in enumerate(['h1', 'h2', 'h3']):
                marker = probe_dir / f"{probe_id}__rank{rank}__{host}.ok"
                marker.write_text(host)
            return ('', '', 0)

        with patch.object(bm, '_execute_command', side_effect=fake_execute):
            bm._probe_results_dir_shared(['h1', 'h2', 'h3'])

    def test_raises_when_remote_hosts_miss_sentinel(self, tmp_path):
        """Non-shared FS: only the controller's sentinel is visible; fail."""
        bm = _make_run_benchmark(tmp_path)

        def fake_execute(cmd, **kwargs):
            import re
            probe_dir = Path(bm.run_result_output) / '.fs_probe'
            m = re.search(r"'([0-9a-f]{12})__rank'", cmd)
            probe_id = m.group(1)
            # Only one host out of three landed a sentinel — the other two
            # wrote to their own local FS (invisible to the controller).
            (probe_dir / f"{probe_id}__rank0__h1.ok").write_text('h1')
            return ('', '', 0)

        with patch.object(bm, '_execute_command', side_effect=fake_execute):
            with pytest.raises(RuntimeError) as excinfo:
                bm._probe_results_dir_shared(['h1', 'h2', 'h3'])
        msg = str(excinfo.value)
        assert 'not visible on every host' in msg
        assert 'shared' in msg.lower() or 'NFS' in msg
        # Hostnames the user passed must surface in the diagnostic.
        assert 'h2' in msg and 'h3' in msg

    def test_strips_slot_suffixes_before_probing(self, tmp_path):
        """--hosts 'h1:4 h2:4' should probe 2 unique hosts, not 8."""
        bm = _make_run_benchmark(tmp_path)
        captured = {}

        def fake_execute(cmd, **kwargs):
            captured['cmd'] = cmd
            # Land sentinels for both unique hosts.
            import re
            probe_dir = Path(bm.run_result_output) / '.fs_probe'
            m = re.search(r"'([0-9a-f]{12})__rank'", cmd)
            probe_id = m.group(1)
            (probe_dir / f"{probe_id}__rank0__h1.ok").write_text('h1')
            (probe_dir / f"{probe_id}__rank1__h2.ok").write_text('h2')
            return ('', '', 0)

        with patch.object(bm, '_execute_command', side_effect=fake_execute):
            bm._probe_results_dir_shared(['h1:4', 'h2:4'])

        # 1 rank per unique host, with :1 slot pinning in the host arg.
        assert '-n 2 ' in captured['cmd']
        assert '-host h1:1,h2:1' in captured['cmd']

    def test_execute_run_invokes_probe_for_multi_host(self, tmp_path):
        """_execute_run should call the probe when --hosts has remote entries."""
        bm = _make_run_benchmark(tmp_path)
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        bm.args.num_processes = 2
        bm.args.npernode = 1  # consistent with num_processes=2 across 2 hosts
        bm.args.hosts = ['h1', 'h2']
        agg = {
            'option': 1, 'aggregated_read_bandwidth_gbps': 0.0,
            'aggregated_write_bandwidth_gbps': 0.0,
            'aggregated_avg_throughput_tokens_per_sec': 0.0,
            'aggregated_storage_throughput_tokens_per_sec': 0.0,
            'aggregated_p95_latency_ms': 0.0,
            'rank_count': 2, 'trial_count': 1,
            'partial_failure': False, 'missing_files': [], 'cpu_tier_ranks': [],
        }
        with patch.object(bm, '_probe_results_dir_shared') as mock_probe, \
             patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results', return_value=agg), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        mock_probe.assert_called_once_with(['h1', 'h2'])

    def test_execute_run_skips_probe_in_what_if_mode(self, tmp_path):
        """--what-if must not spawn mpirun probes (no execution at all)."""
        bm = _make_run_benchmark(tmp_path, what_if=True)
        bm.args.trials = 1
        bm.args.inter_option_delay = 0
        bm.args.num_processes = 2
        bm.args.npernode = 1
        bm.args.hosts = ['h1', 'h2']
        with patch.object(bm, '_probe_results_dir_shared') as mock_probe, \
             patch.object(bm, '_execute_command', return_value=('', '', 0)), \
             patch.object(bm, '_interruptible_sleep'), \
             patch.object(bm, '_aggregate_option_results'), \
             patch.object(bm, '_write_run_summary'), \
             patch.object(bm, 'write_metadata'):
            bm._execute_run()
        mock_probe.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 02 / Plan 02-05 — non-DLIO regression: assert the shared
# Benchmark.run() systemname.yaml write hook fires for KVCacheBenchmark.
# ---------------------------------------------------------------------------


class TestKVCacheSystemnameYamlHook:
    """KVCacheBenchmark inherits Benchmark.run() — the LIFE-01 hook must
    fire for it just as for DLIO-based benchmarks. If a future refactor
    overrides run() on the subclass and accidentally drops the hook, these
    tests catch the regression.

    The shared fixture `_make_run_benchmark` uses `mode='open'` (see its
    docstring: the strict CLOSED-mode override checks in _execute_run would
    otherwise fire on tests that deliberately override those args). We
    therefore assert the file lands at `<tmp>/open/Acme/systems/sys-v1.yaml`.
    """

    def _install_cluster_info_mock(self, bm):
        """Mock _collect_cluster_start so it populates self._cluster_info_start
        with a one-host MagicMock fleet — the production write hook reads
        host_info_list from this attribute."""
        from mlpstorage_py.rules.models import HostCPUInfo, HostInfo, HostMemoryInfo
        from mlpstorage_py.cluster_collector import HostSystemInfo

        host = HostInfo(
            hostname='h0',
            cpu=HostCPUInfo(
                model='Intel(R) Xeon Platinum 8480+',
                num_cores=56, num_logical_cores=112, num_sockets=2,
                architecture='x86_64',
            ),
            memory=HostMemoryInfo(total=274_877_906_944),
            system=HostSystemInfo(
                hostname='h0',
                os_release={'NAME': 'Rocky Linux', 'VERSION_ID': '9.5'},
            ),
        )
        cluster_info_mock = MagicMock(host_info_list=[host])

        def _side_effect():
            bm._cluster_info_start = cluster_info_mock
            bm._collection_method = 'mpi'

        bm._collect_cluster_start = MagicMock(side_effect=_side_effect)

    def _mock_remaining_lifecycle(self, bm):
        bm._validate_environment = MagicMock()
        bm._start_timeseries_collection = MagicMock()
        bm._stop_timeseries_collection = MagicMock()
        bm._collect_cluster_end = MagicMock()
        bm.write_timeseries_data = MagicMock()
        bm._run = MagicMock(return_value=0)

    def test_kvcache_run_writes_systemname_yaml(self, tmp_path):
        """KVCacheBenchmark.run() must write systemname.yaml at the canonical
        path (Phase 02 LIFE-01, regression coverage for the shared base hook
        on non-DLIO benchmarks)."""
        bm = _make_run_benchmark(tmp_path)
        # Fixture uses mode='open', orgname='Acme', systemname='sys-v1' —
        # see _make_run_benchmark docstring.
        self._install_cluster_info_mock(bm)
        self._mock_remaining_lifecycle(bm)

        rc = bm.run()
        assert rc == 0

        target = tmp_path / 'open' / 'Acme' / 'systems' / 'sys-v1.yaml'
        assert target.exists(), (
            f"KVCacheBenchmark.run() should have written systemname.yaml at "
            f"{target}; this is the LIFE-01 non-DLIO regression coverage."
        )
