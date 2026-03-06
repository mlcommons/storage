"""
Tests for VectorDBBenchmark class in mlpstorage.benchmarks.vectordbbench module.

Tests cover:
- Command method map structure
- Metadata generation for history integration
- Command-specific metadata fields
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
        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(basic_args)

            assert 'run' in bm.command_method_map
            assert 'run-search' not in bm.command_method_map

    def test_datagen_command_in_map(self, basic_args, tmp_path):
        """Command map should contain 'datagen' key."""
        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(basic_args)

            assert 'datagen' in bm.command_method_map

    def test_command_map_has_correct_methods(self, basic_args, tmp_path):
        """Command map should map to correct methods."""
        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
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
        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(run_args)
            meta = bm.metadata

        # Required by history module
        assert 'benchmark_type' in meta
        assert 'model' in meta  # Uses config_name
        assert 'command' in meta
        assert 'run_datetime' in meta
        assert 'result_dir' in meta

    def test_metadata_includes_vectordb_specific_fields(self, run_args, tmp_path):
        """Verify VectorDB specific metadata fields."""
        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(run_args)
            meta = bm.metadata

        assert 'vectordb_config' in meta
        assert 'host' in meta
        assert 'port' in meta
        assert 'collection' in meta

    def test_metadata_model_uses_config_name(self, run_args, tmp_path):
        """Verify 'model' field uses config_name for history compatibility."""
        run_args.config = '10m'

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(run_args)
            meta = bm.metadata

        assert meta['model'] == '10m'
        assert meta['vectordb_config'] == '10m'

    def test_metadata_run_command_fields(self, run_args, tmp_path):
        """Verify run-specific metadata fields."""
        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
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
        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
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

    def test_metadata_connection_info(self, run_args, tmp_path):
        """Verify host/port connection info in metadata."""
        run_args.host = '10.0.0.50'
        run_args.port = 9999

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(run_args)
            meta = bm.metadata

        assert meta['host'] == '10.0.0.50'
        assert meta['port'] == 9999

    def test_metadata_run_no_datagen_fields(self, run_args, tmp_path):
        """Verify run command metadata does not include datagen fields."""
        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
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
        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(datagen_args)
            meta = bm.metadata

        # Run-specific fields should not be in datagen metadata
        # Note: 'runtime' exists in base metadata with a different meaning (execution time)
        # so we check for num_query_processes and queries which are VectorDB run-specific
        assert 'num_query_processes' not in meta
        assert 'queries' not in meta
        assert 'batch_size' not in meta  # datagen uses different batch_size semantics


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
        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
            from mlpstorage.config import BENCHMARK_TYPES

            assert VectorDBBenchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.vector_database

    def test_metadata_benchmark_type(self, basic_args, tmp_path):
        """Metadata should include correct benchmark_type."""
        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
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

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(basic_args)

        assert bm.config_name == 'my_custom_config'

    def test_default_config_name(self, basic_args, tmp_path):
        """Should default to 'default' if config not specified."""
        basic_args.config = None

        with patch('mlpstorage.benchmarks.base.generate_output_location') as mock_gen, \
             patch('mlpstorage.benchmarks.vectordbbench.read_config_from_file', return_value={}), \
             patch('mlpstorage.benchmarks.vectordbbench.VectorDBBenchmark.verify_benchmark'):
            output_dir = str(tmp_path / "output")
            mock_gen.return_value = output_dir
            os.makedirs(output_dir, exist_ok=True)

            from mlpstorage.benchmarks.vectordbbench import VectorDBBenchmark
            bm = VectorDBBenchmark(basic_args)

        assert bm.config_name == 'default'
