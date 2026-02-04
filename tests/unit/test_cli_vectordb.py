"""
Tests for VectorDB benchmark CLI argument parsing.

Tests cover:
- VectorDB subcommand structure (run, datagen)
- Common arguments (host, port, collection, config)
- Datagen-specific arguments (dimension, num_vectors, distribution, etc.)
- Run-specific arguments (num_query_processes, batch_size, runtime, queries)
"""

import argparse
import pytest

from mlpstorage.cli.vectordb_args import add_vectordb_arguments
from mlpstorage.config import VECTOR_DTYPES, DISTRIBUTIONS


class TestVectorDBSubcommands:
    """Tests for VectorDB subcommand structure."""

    @pytest.fixture
    def parser(self):
        """Create a parser with vectordb subcommands."""
        parser = argparse.ArgumentParser()
        add_vectordb_arguments(parser)
        return parser

    def test_run_subcommand_exists(self, parser):
        """VectorDB should have run subcommand (not run-search)."""
        args = parser.parse_args(['run'])
        assert args.command == 'run'

    def test_datagen_subcommand_exists(self, parser):
        """VectorDB should have datagen subcommand."""
        args = parser.parse_args(['datagen'])
        assert args.command == 'datagen'

    def test_run_search_does_not_exist(self, parser):
        """run-search should NOT be a valid subcommand."""
        with pytest.raises(SystemExit):
            parser.parse_args(['run-search'])


class TestVectorDBCommonArguments:
    """Tests for arguments common to both run and datagen."""

    @pytest.fixture
    def parser(self):
        """Create a parser with vectordb subcommands."""
        parser = argparse.ArgumentParser()
        add_vectordb_arguments(parser)
        return parser

    def test_host_argument_default(self, parser):
        """Host should default to 127.0.0.1."""
        args = parser.parse_args(['run'])
        assert args.host == '127.0.0.1'

    def test_host_argument_custom(self, parser):
        """Should accept custom host."""
        args = parser.parse_args(['run', '--host', '192.168.1.100'])
        assert args.host == '192.168.1.100'

    def test_host_short_flag(self, parser):
        """Should accept -s shorthand for host."""
        args = parser.parse_args(['run', '-s', 'milvus.local'])
        assert args.host == 'milvus.local'

    def test_port_argument_default(self, parser):
        """Port should default to 19530."""
        args = parser.parse_args(['run'])
        assert args.port == 19530

    def test_port_argument_custom(self, parser):
        """Should accept custom port."""
        args = parser.parse_args(['run', '--port', '19531'])
        assert args.port == 19531

    def test_port_short_flag(self, parser):
        """Should accept -p shorthand for port."""
        args = parser.parse_args(['run', '-p', '8080'])
        assert args.port == 8080

    def test_config_argument(self, parser):
        """Should accept --config argument."""
        args = parser.parse_args(['run', '--config', '10m'])
        assert args.config == '10m'

    def test_collection_argument(self, parser):
        """Should accept --collection argument."""
        args = parser.parse_args(['run', '--collection', 'my_collection'])
        assert args.collection == 'my_collection'

    def test_common_args_work_for_datagen(self, parser):
        """Common arguments should also work for datagen command."""
        args = parser.parse_args([
            'datagen',
            '--host', '10.0.0.1',
            '--port', '19531',
            '--config', 'custom_config',
            '--collection', 'test_collection'
        ])
        assert args.host == '10.0.0.1'
        assert args.port == 19531
        assert args.config == 'custom_config'
        assert args.collection == 'test_collection'


class TestVectorDBDatagenArguments:
    """Tests for datagen-specific arguments."""

    @pytest.fixture
    def parser(self):
        """Create a parser with vectordb subcommands."""
        parser = argparse.ArgumentParser()
        add_vectordb_arguments(parser)
        return parser

    def test_dimension_argument_default(self, parser):
        """Dimension should default to 1536."""
        args = parser.parse_args(['datagen'])
        assert args.dimension == 1536

    def test_dimension_argument_custom(self, parser):
        """Should accept custom dimension."""
        args = parser.parse_args(['datagen', '--dimension', '768'])
        assert args.dimension == 768

    def test_num_shards_argument_default(self, parser):
        """num_shards should default to 1."""
        args = parser.parse_args(['datagen'])
        assert args.num_shards == 1

    def test_num_shards_argument_custom(self, parser):
        """Should accept custom num_shards."""
        args = parser.parse_args(['datagen', '--num-shards', '4'])
        assert args.num_shards == 4

    def test_vector_dtype_argument_default(self, parser):
        """vector_dtype should default to FLOAT_VECTOR."""
        args = parser.parse_args(['datagen'])
        assert args.vector_dtype == 'FLOAT_VECTOR'

    def test_vector_dtype_argument_choices(self, parser):
        """Should accept valid vector dtype choices."""
        for dtype in VECTOR_DTYPES:
            args = parser.parse_args(['datagen', '--vector-dtype', dtype])
            assert args.vector_dtype == dtype

    def test_vector_dtype_invalid_choice(self, parser):
        """Should reject invalid vector dtype."""
        with pytest.raises(SystemExit):
            parser.parse_args(['datagen', '--vector-dtype', 'INVALID_TYPE'])

    def test_num_vectors_argument_default(self, parser):
        """num_vectors should default to 1_000_000."""
        args = parser.parse_args(['datagen'])
        assert args.num_vectors == 1_000_000

    def test_num_vectors_argument_custom(self, parser):
        """Should accept custom num_vectors."""
        args = parser.parse_args(['datagen', '--num-vectors', '5000000'])
        assert args.num_vectors == 5000000

    def test_distribution_argument_default(self, parser):
        """distribution should default to uniform."""
        args = parser.parse_args(['datagen'])
        assert args.distribution == 'uniform'

    def test_distribution_argument_choices(self, parser):
        """Should accept valid distribution choices."""
        for dist in DISTRIBUTIONS:
            args = parser.parse_args(['datagen', '--distribution', dist])
            assert args.distribution == dist

    def test_distribution_invalid_choice(self, parser):
        """Should reject invalid distribution."""
        with pytest.raises(SystemExit):
            parser.parse_args(['datagen', '--distribution', 'invalid_dist'])

    def test_batch_size_argument_default(self, parser):
        """batch_size should default to 1000."""
        args = parser.parse_args(['datagen'])
        assert args.batch_size == 1_000

    def test_batch_size_argument_custom(self, parser):
        """Should accept custom batch_size."""
        args = parser.parse_args(['datagen', '--batch-size', '500'])
        assert args.batch_size == 500

    def test_chunk_size_argument_default(self, parser):
        """chunk_size should default to 10000."""
        args = parser.parse_args(['datagen'])
        assert args.chunk_size == 10_000

    def test_chunk_size_argument_custom(self, parser):
        """Should accept custom chunk_size."""
        args = parser.parse_args(['datagen', '--chunk-size', '50000'])
        assert args.chunk_size == 50000

    def test_force_argument(self, parser):
        """Should accept --force flag."""
        args = parser.parse_args(['datagen', '--force'])
        assert args.force is True

    def test_force_argument_default(self, parser):
        """force should default to False."""
        args = parser.parse_args(['datagen'])
        assert args.force is False


class TestVectorDBRunArguments:
    """Tests for run-specific arguments."""

    @pytest.fixture
    def parser(self):
        """Create a parser with vectordb subcommands."""
        parser = argparse.ArgumentParser()
        add_vectordb_arguments(parser)
        return parser

    def test_num_query_processes_argument_default(self, parser):
        """num_query_processes should default to 1."""
        args = parser.parse_args(['run'])
        assert args.num_query_processes == 1

    def test_num_query_processes_argument_custom(self, parser):
        """Should accept custom num_query_processes."""
        args = parser.parse_args(['run', '--num-query-processes', '8'])
        assert args.num_query_processes == 8

    def test_batch_size_argument_default(self, parser):
        """batch_size should default to 1."""
        args = parser.parse_args(['run'])
        assert args.batch_size == 1

    def test_batch_size_argument_custom(self, parser):
        """Should accept custom batch_size."""
        args = parser.parse_args(['run', '--batch-size', '100'])
        assert args.batch_size == 100

    def test_report_count_argument_default(self, parser):
        """report_count should default to 100."""
        args = parser.parse_args(['run'])
        assert args.report_count == 100

    def test_report_count_argument_custom(self, parser):
        """Should accept custom report_count."""
        args = parser.parse_args(['run', '--report-count', '500'])
        assert args.report_count == 500

    def test_runtime_argument(self, parser):
        """Should accept --runtime argument."""
        args = parser.parse_args(['run', '--runtime', '120'])
        assert args.runtime == 120

    def test_queries_argument(self, parser):
        """Should accept --queries argument."""
        args = parser.parse_args(['run', '--queries', '10000'])
        assert args.queries == 10000

    def test_runtime_and_queries_mutually_exclusive(self, parser):
        """runtime and queries should be mutually exclusive."""
        with pytest.raises(SystemExit):
            parser.parse_args(['run', '--runtime', '60', '--queries', '1000'])

    def test_runtime_not_required(self, parser):
        """Neither runtime nor queries is required."""
        args = parser.parse_args(['run'])
        assert args.runtime is None
        assert args.queries is None


class TestVectorDBDatagenNoRunArgs:
    """Tests verifying datagen doesn't have run-specific args."""

    @pytest.fixture
    def parser(self):
        """Create a parser with vectordb subcommands."""
        parser = argparse.ArgumentParser()
        add_vectordb_arguments(parser)
        return parser

    def test_datagen_has_dimension(self, parser):
        """datagen should have dimension argument."""
        args = parser.parse_args(['datagen', '--dimension', '512'])
        assert args.dimension == 512

    def test_run_no_dimension(self, parser):
        """run should not have dimension argument."""
        args = parser.parse_args(['run'])
        assert not hasattr(args, 'dimension')

    def test_run_no_num_vectors(self, parser):
        """run should not have num_vectors argument."""
        args = parser.parse_args(['run'])
        assert not hasattr(args, 'num_vectors')

    def test_run_no_force(self, parser):
        """run should not have force argument."""
        args = parser.parse_args(['run'])
        assert not hasattr(args, 'force')

    def test_run_no_chunk_size(self, parser):
        """run should not have chunk_size argument."""
        args = parser.parse_args(['run'])
        assert not hasattr(args, 'chunk_size')

    def test_run_no_num_shards(self, parser):
        """run should not have num_shards argument."""
        args = parser.parse_args(['run'])
        assert not hasattr(args, 'num_shards')


class TestVectorDBRunNoDatagenArgs:
    """Tests verifying run doesn't have datagen-specific args."""

    @pytest.fixture
    def parser(self):
        """Create a parser with vectordb subcommands."""
        parser = argparse.ArgumentParser()
        add_vectordb_arguments(parser)
        return parser

    def test_datagen_no_num_query_processes(self, parser):
        """datagen should not have num_query_processes argument."""
        args = parser.parse_args(['datagen'])
        assert not hasattr(args, 'num_query_processes')

    def test_datagen_no_runtime(self, parser):
        """datagen should not have runtime argument."""
        args = parser.parse_args(['datagen'])
        assert not hasattr(args, 'runtime')

    def test_datagen_no_queries(self, parser):
        """datagen should not have queries argument."""
        args = parser.parse_args(['datagen'])
        assert not hasattr(args, 'queries')

    def test_datagen_no_report_count(self, parser):
        """datagen should not have report_count argument."""
        args = parser.parse_args(['datagen'])
        assert not hasattr(args, 'report_count')


class TestVectorDBFullCommandParsing:
    """Tests for complete command parsing with multiple arguments."""

    @pytest.fixture
    def parser(self):
        """Create a parser with vectordb subcommands."""
        parser = argparse.ArgumentParser()
        add_vectordb_arguments(parser)
        return parser

    def test_datagen_full_command(self, parser):
        """Should parse a complete datagen command."""
        args = parser.parse_args([
            'datagen',
            '--host', '192.168.1.100',
            '--port', '19531',
            '--config', 'custom_config',
            '--collection', 'test_collection',
            '--dimension', '768',
            '--num-shards', '4',
            '--vector-dtype', 'FLOAT_VECTOR',
            '--num-vectors', '5000000',
            '--distribution', 'normal',
            '--batch-size', '500',
            '--chunk-size', '50000',
            '--force'
        ])
        assert args.command == 'datagen'
        assert args.host == '192.168.1.100'
        assert args.port == 19531
        assert args.config == 'custom_config'
        assert args.collection == 'test_collection'
        assert args.dimension == 768
        assert args.num_shards == 4
        assert args.vector_dtype == 'FLOAT_VECTOR'
        assert args.num_vectors == 5000000
        assert args.distribution == 'normal'
        assert args.batch_size == 500
        assert args.chunk_size == 50000
        assert args.force is True

    def test_run_full_command_with_runtime(self, parser):
        """Should parse a complete run command with runtime."""
        args = parser.parse_args([
            'run',
            '--host', '10.0.0.50',
            '--port', '9999',
            '--config', '10m',
            '--collection', 'benchmark_collection',
            '--num-query-processes', '16',
            '--batch-size', '32',
            '--report-count', '1000',
            '--runtime', '300'
        ])
        assert args.command == 'run'
        assert args.host == '10.0.0.50'
        assert args.port == 9999
        assert args.config == '10m'
        assert args.collection == 'benchmark_collection'
        assert args.num_query_processes == 16
        assert args.batch_size == 32
        assert args.report_count == 1000
        assert args.runtime == 300
        assert args.queries is None

    def test_run_full_command_with_queries(self, parser):
        """Should parse a complete run command with queries."""
        args = parser.parse_args([
            'run',
            '--host', '10.0.0.50',
            '--port', '9999',
            '--num-query-processes', '8',
            '--batch-size', '10',
            '--queries', '50000'
        ])
        assert args.command == 'run'
        assert args.queries == 50000
        assert args.runtime is None
