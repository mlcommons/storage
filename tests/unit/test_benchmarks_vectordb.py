"""
Tests for VectorDBBenchmark class in mlpstorage.benchmarks.vectordbbench module.

Tests cover:
- Command method map structure
- Metadata generation for history integration
- Command-specific metadata fields
- VectorDB index normalization and command generation
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from argparse import Namespace


class TestVectorDBCommandMap:
    """Tests for VectorDBBenchmark command routing."""

    @pytest.fixture
    def basic_args(self, tmp_path):
        """Create basic args for VectorDB benchmark."""
        return Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            command='run',
            config='default',
            host='127.0.0.1',
            port=19530,
            collection=None,
            category=None,
            num_query_processes=1,
            batch_size=1,
            runtime=60,
            queries=None,
            report_count=100,
        )

    def test_run_command_in_map(self, basic_args, tmp_path):
        """Command map should contain 'run' key."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(basic_args)

            assert 'run' in bm.command_method_map
            assert 'run-search' not in bm.command_method_map

    def test_datagen_command_in_map(self, basic_args, tmp_path):
        """Command map should contain 'datagen' key."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(basic_args)

            assert 'datagen' in bm.command_method_map

    def test_command_map_has_correct_methods(self, basic_args, tmp_path):
        """Command map should map to correct methods."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(basic_args)

            assert bm.command_method_map['run'] == bm.execute_run
            assert bm.command_method_map['datagen'] == bm.execute_datagen


class TestVectorDBMetadata:
    """Test metadata structure for history integration."""

    @pytest.fixture
    def run_args(self, tmp_path):
        """Create args for VectorDB run command."""
        return Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            command='run',
            config='10m',
            vdb_engine='milvus',
            vdb_index='DISKANN',
            host='192.168.1.100',
            port=19531,
            collection='test_collection',
            category=None,
            num_query_processes=4,
            batch_size=10,
            runtime=120,
            queries=None,
            report_count=100,
        )

    @pytest.fixture
    def datagen_args(self, tmp_path):
        """Create args for VectorDB datagen command."""
        return Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            command='datagen',
            config='default',
            vdb_engine='milvus',
            vdb_index='HNSW',
            index_type=None,
            host='127.0.0.1',
            port=19530,
            collection='gen_collection',
            category=None,
            dimension=768,
            num_vectors=5000000,
            num_shards=2,
            vector_dtype='FLOAT_VECTOR',
            distribution='normal',
            batch_size=1000,
            chunk_size=10000,
            force=True,
        )

    def test_metadata_has_required_fields(self, run_args, tmp_path):
        """Verify metadata includes fields required by history module."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(run_args)
            meta = bm.metadata

        # Required by history module
        assert 'benchmark_type' in meta
        assert 'model' in meta  # Engine (vdb_engine), recorded so accumulated
                                # results from multiple engines stay separate.
        assert 'command' in meta
        assert 'run_datetime' in meta
        assert 'result_dir' in meta

    def test_metadata_includes_vectordb_specific_fields(self, run_args, tmp_path):
        """Verify VectorDB specific metadata fields."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(run_args)
            meta = bm.metadata

        assert meta['vectordb_config'] == '10m'
        assert meta['vdb_engine'] == 'milvus'
        assert meta['vdb_index'] == 'DISKANN'
        assert meta['host'] == '192.168.1.100'
        assert meta['port'] == 19531
        assert meta['collection'] == 'test_collection'

    def test_metadata_model_is_engine_config_preserved_separately(
        self, run_args, tmp_path
    ):
        """'model' records the vdb_engine (so workload grouping treats engines
        as distinct workloads), while 'vectordb_config' preserves the config
        name for history/replay."""
        run_args.config = '10m'
        run_args.vdb_engine = 'milvus'

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(run_args)
            meta = bm.metadata

        assert meta['model'] == 'milvus_DISKANN'
        assert meta['vdb_engine'] == 'milvus'
        assert meta['vdb_index'] == 'DISKANN'
        assert meta['vectordb_config'] == '10m'

    def test_metadata_run_command_fields(self, run_args, tmp_path):
        """Verify run-specific metadata fields."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(run_args)
            meta = bm.metadata

        assert 'num_query_processes' in meta
        assert meta['num_query_processes'] == 4
        assert 'batch_size' in meta
        assert meta['batch_size'] == 10
        assert 'runtime' in meta
        assert meta['runtime'] == 120

    def test_metadata_datagen_command_fields(self, datagen_args, tmp_path):
        """Verify datagen-specific metadata fields."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(datagen_args)
            meta = bm.metadata

        assert 'dimension' in meta
        assert meta['dimension'] == 768
        assert 'num_vectors' in meta
        assert meta['num_vectors'] == 5000000
        assert 'num_shards' in meta
        assert meta['num_shards'] == 2
        assert 'vector_dtype' in meta
        assert meta['vector_dtype'] == 'FLOAT_VECTOR'
        assert 'distribution' in meta
        assert meta['distribution'] == 'normal'
        assert meta['vdb_engine'] == 'milvus'
        assert meta['vdb_index'] == 'HNSW'
        assert meta['index_type'] == 'HNSW'
        assert meta['model'] == 'milvus_HNSW'

    def test_metadata_connection_info(self, run_args, tmp_path):
        """Verify host/port connection info in metadata."""
        run_args.host = '10.0.0.50'
        run_args.port = 9999

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(run_args)
            meta = bm.metadata

        assert meta['host'] == '10.0.0.50'
        assert meta['port'] == 9999

    def test_metadata_run_no_datagen_fields(self, run_args, tmp_path):
        """Verify run command metadata does not include datagen fields."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(run_args)
            meta = bm.metadata

        # Datagen-specific fields should not be in run metadata
        assert 'dimension' not in meta
        assert 'num_vectors' not in meta
        assert 'num_shards' not in meta
        assert 'vector_dtype' not in meta
        assert 'distribution' not in meta

    def test_metadata_datagen_no_run_fields(self, datagen_args, tmp_path):
        """Verify datagen command metadata does not include run-specific fields."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(datagen_args)
            meta = bm.metadata

        # Run-specific fields should not be in datagen metadata.
        # Note: 'runtime' exists in base metadata with a different meaning
        # (execution time) so we check for VectorDB run-specific fields.
        # batch_size IS recorded for datagen (multi-host VDB reproducibility),
        # but its semantics differ from run's batch_size.
        assert 'num_query_processes' not in meta
        assert 'queries' not in meta
        assert 'benchmark_mode' not in meta
        assert 'search_limit' not in meta


class TestVectorDBBenchmarkType:
    """Tests for VectorDB benchmark type configuration."""

    @pytest.fixture
    def basic_args(self, tmp_path):
        """Create basic args for VectorDB benchmark."""
        return Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            command='run',
            config='default',
            host='127.0.0.1',
            port=19530,
            collection=None,
            category=None,
            num_query_processes=1,
            batch_size=1,
            runtime=60,
            queries=None,
            report_count=100,
        )

    def test_benchmark_type_is_vector_database(self, basic_args, tmp_path):
        """VectorDBBenchmark should have correct BENCHMARK_TYPE."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            from mlpstorage_py.config import BENCHMARK_TYPES

            assert VectorDBBenchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.vector_database

    def test_metadata_benchmark_type(self, basic_args, tmp_path):
        """Metadata should include correct benchmark_type."""
        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(basic_args)
            meta = bm.metadata

        assert meta['benchmark_type'] == 'vector_database'


class TestVectorDBConfigHandling:
    """Tests for VectorDB config handling."""

    @pytest.fixture
    def basic_args(self, tmp_path):
        """Create basic args for VectorDB benchmark."""
        return Namespace(
            debug=False,
            verbose=False,
            what_if=False,
            stream_log_level='INFO',
            results_dir=str(tmp_path),
            command='run',
            config='custom_config',
            host='127.0.0.1',
            port=19530,
            collection=None,
            category=None,
            num_query_processes=1,
            batch_size=1,
            runtime=60,
            queries=None,
            report_count=100,
        )

    def test_config_name_from_args(self, basic_args, tmp_path):
        """Should use config name from args."""
        basic_args.config = 'my_custom_config'

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(basic_args)

        assert bm.config_name == 'my_custom_config'

    def test_default_config_name(self, basic_args, tmp_path):
        """Should default to 'default' if config not specified."""
        basic_args.config = None

        with patch('mlpstorage_py.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage_py.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'), \
             patch('mlpstorage_py.benchmarks.vectordbbench.VectorDBBenchmark._validate_vdb_dependencies'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir

            from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(basic_args)

        assert bm.config_name == 'default'


class TestVectorDBIndexResolution:
    """Tests benchmark-side normalization for API/direct construction."""

    @pytest.mark.parametrize("command", ["datasize", "datagen"])
    def test_default_index_populates_both_names(self, command):
        from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
        from mlpstorage_py.config import VDB_INDEX_DEFAULT

        args = Namespace(command=command, vdb_index=None, index_type=None)

        resolved = VectorDBBenchmark._resolve_vdb_index_arguments(args)

        assert resolved == VDB_INDEX_DEFAULT
        assert args.vdb_index == VDB_INDEX_DEFAULT
        assert args.index_type == VDB_INDEX_DEFAULT

    @pytest.mark.parametrize("command", ["datasize", "datagen"])
    def test_vdb_index_populates_index_type(self, command):
        from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark

        args = Namespace(command=command, vdb_index='HNSW', index_type=None)

        resolved = VectorDBBenchmark._resolve_vdb_index_arguments(args)

        assert resolved == 'HNSW'
        assert args.vdb_index == args.index_type == 'HNSW'

    @pytest.mark.parametrize("command", ["datasize", "datagen"])
    def test_legacy_index_type_populates_vdb_index(self, command):
        from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark

        args = Namespace(command=command, vdb_index=None, index_type='AISAQ')

        resolved = VectorDBBenchmark._resolve_vdb_index_arguments(args)

        assert resolved == 'AISAQ'
        assert args.vdb_index == args.index_type == 'AISAQ'

    @pytest.mark.parametrize("command", ["datasize", "datagen"])
    def test_conflicting_index_names_fail_before_result_dir_creation(
        self, command
    ):
        from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark

        args = Namespace(
            command=command,
            vdb_index='DISKANN',
            index_type='HNSW',
        )

        with pytest.raises(ValueError, match='must match'):
            VectorDBBenchmark._resolve_vdb_index_arguments(args)

    def test_run_defaults_index_without_creating_index_type(self):
        from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark
        from mlpstorage_py.config import VDB_INDEX_DEFAULT

        args = Namespace(command='run', vdb_index=None)

        resolved = VectorDBBenchmark._resolve_vdb_index_arguments(args)

        assert resolved == VDB_INDEX_DEFAULT
        assert args.vdb_index == VDB_INDEX_DEFAULT
        assert not hasattr(args, 'index_type')


class TestVectorDBEffectiveIndexUse:
    """Tests that datasize and datagen use the normalized Milvus index."""

    @staticmethod
    def _bare_benchmark(args):
        from mlpstorage_py.benchmarks.vectordbbench import VectorDBBenchmark

        benchmark = object.__new__(VectorDBBenchmark)
        benchmark.args = args
        benchmark.command = args.command
        benchmark.logger = MagicMock()
        benchmark.write_metadata = MagicMock()
        return benchmark

    @staticmethod
    def _datagen_args(**overrides):
        values = {
            'command': 'datagen',
            'vdb_index': 'HNSW',
            'index_type': None,
            'host': '127.0.0.1',
            'port': 19530,
            'dimension': 768,
            'num_shards': 1,
            'vector_dtype': 'FLOAT_VECTOR',
            'num_vectors': 1000,
            'distribution': 'uniform',
            'batch_size': 100,
            'chunk_size': 500,
            'metric_type': None,
            'max_degree': None,
            'search_list_size': None,
            'M': None,
            'ef_construction': None,
            'inline_pq': None,
            'monitor_interval': None,
            'compact': False,
            'force': False,
            'ready_timeout': 7200,
            'coordination': 'filesystem',
            'rank_output_dir': '/tmp/mlps_vdb',
            'seed': 42,
            'what_if': True,
        }
        values.update(overrides)
        return Namespace(**values)

    def test_datasize_uses_vdb_index_when_index_type_is_omitted(self):
        args = Namespace(
            command='datasize',
            vdb_index='HNSW',
            index_type=None,
            dimension=128,
            num_vectors=1000,
            num_shards=1,
        )
        benchmark = self._bare_benchmark(args)

        rc = benchmark.execute_datasize()

        assert rc == 0
        assert args.index_type == 'HNSW'
        assert args.vdb_index == 'HNSW'
        assert any(
            'Index type: HNSW' in call.args[0]
            for call in benchmark.logger.result.call_args_list
        )
        benchmark.write_metadata.assert_called_once_with()

    def test_single_node_datagen_passes_effective_index_to_load_vdb(self):
        args = self._datagen_args()
        benchmark = self._bare_benchmark(args)
        benchmark._collection_name = MagicMock(return_value='test_collection')
        benchmark.build_command = MagicMock(return_value='uv run load-vdb')
        benchmark._execute_command = MagicMock(return_value=('', '', 0))

        rc = benchmark._execute_datagen_single_node()

        assert rc == 0
        script_name, additional_params = benchmark.build_command.call_args.args
        assert script_name == 'load-vdb'
        assert additional_params['index-type'] == 'HNSW'
        assert args.index_type == args.vdb_index == 'HNSW'
        benchmark._execute_command.assert_called_once()
        benchmark.write_metadata.assert_called_once_with()

    def test_distributed_datagen_passes_effective_index_to_wrapper(
        self, tmp_path
    ):
        args = self._datagen_args()
        benchmark = self._bare_benchmark(args)
        benchmark.run_result_output = str(tmp_path / 'run')
        benchmark.config_file = '/tmp/default.yaml'
        benchmark._base_output_dir = MagicMock(
            return_value=str(tmp_path / 'run' / 'vectordb' / 'load')
        )
        benchmark._mpi_world_size = MagicMock(return_value=2)
        benchmark._mpi_prefix = MagicMock(return_value='mpiexec -n 2')
        benchmark._get_uv_prefix = MagicMock(return_value='uv run ')
        benchmark._coordination_backend = MagicMock(return_value='filesystem')
        benchmark._rank_output_dir = MagicMock(return_value='/tmp/mlps_vdb')
        benchmark._run_id = MagicMock(return_value='20250111_160000')
        benchmark._collection_name = MagicMock(return_value='test_collection')
        benchmark._execute_command = MagicMock(return_value=('', '', 0))
        benchmark._run_aggregate = MagicMock(return_value=0)

        rc = benchmark._execute_datagen_distributed()

        assert rc == 0
        command = benchmark._execute_command.call_args.args[0]
        assert '--index-type HNSW' in command
        assert args.index_type == args.vdb_index == 'HNSW'
        benchmark._run_aggregate.assert_not_called()
        benchmark.write_metadata.assert_called_once_with()

    def test_effective_index_rejects_conflicting_values(self):
        args = Namespace(
            command='datagen',
            vdb_index='DISKANN',
            index_type='HNSW',
        )
        benchmark = self._bare_benchmark(args)

        with pytest.raises(ValueError, match='must match'):
            benchmark._effective_index_type()
