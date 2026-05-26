"""
Tests for KV Cache benchmark CLI argument parsing.

Tests cover:
- KV Cache model and cache configuration arguments
- Run command arguments
- Distributed execution arguments (--hosts, --exec-type, --num-processes)
- MPI arguments (--mpi-bin, --oversubscribe, --allow-run-as-root, --mpi-params)
- Verification that datasize command doesn't have distributed args
"""

import argparse
import pytest

from mlpstorage_py.cli.kvcache_args import add_kvcache_arguments
from mlpstorage_py.config import EXEC_TYPE, KVCACHE_MODELS


class TestKVCacheSubcommands:
    """Tests for KV cache subcommand structure."""

    @pytest.fixture
    def parser(self):
        """Create a parser with kvcache subcommands."""
        parser = argparse.ArgumentParser()
        add_kvcache_arguments(parser)
        return parser

    def test_run_subcommand_exists(self, parser):
        """KV cache should have run subcommand."""
        args = parser.parse_args(['run'])
        assert args.command == 'run'

    def test_datasize_subcommand_exists(self, parser):
        """KV cache should have datasize subcommand."""
        args = parser.parse_args(['datasize'])
        assert args.command == 'datasize'


class TestKVCacheModelArguments:
    """Tests for KV cache model configuration arguments."""

    @pytest.fixture
    def parser(self):
        """Create a parser with kvcache subcommands."""
        parser = argparse.ArgumentParser()
        add_kvcache_arguments(parser)
        return parser

    def test_model_argument_default(self, parser):
        """Model should default to llama3.1-8b."""
        args = parser.parse_args(['run'])
        assert args.model == 'llama3.1-8b'

    def test_model_argument_choices(self, parser):
        """Model should accept valid choices."""
        for model in KVCACHE_MODELS:
            args = parser.parse_args(['run', '--model', model])
            assert args.model == model

    def test_num_users_argument(self, parser):
        """Should accept --num-users argument."""
        args = parser.parse_args(['run', '--num-users', '200'])
        assert args.num_users == 200

    def test_num_users_default(self, parser):
        """num_users should default to 100."""
        args = parser.parse_args(['run'])
        assert args.num_users == 100


class TestKVCacheCacheArguments:
    """Tests for KV cache tier configuration arguments."""

    @pytest.fixture
    def parser(self):
        """Create a parser with kvcache subcommands."""
        parser = argparse.ArgumentParser()
        add_kvcache_arguments(parser)
        return parser

    def test_gpu_mem_gb_argument(self, parser):
        """Should accept --gpu-mem-gb argument."""
        args = parser.parse_args(['run', '--gpu-mem-gb', '80.0'])
        assert args.gpu_mem_gb == 80.0

    def test_cpu_mem_gb_argument(self, parser):
        """Should accept --cpu-mem-gb argument."""
        args = parser.parse_args(['run', '--cpu-mem-gb', '256.0'])
        assert args.cpu_mem_gb == 256.0

    def test_cache_dir_argument(self, parser):
        """Should accept --cache-dir argument."""
        args = parser.parse_args(['run', '--cache-dir', '/nvme/cache'])
        assert args.cache_dir == '/nvme/cache'


class TestKVCacheRunArguments:
    """Tests for KV cache run-specific arguments."""

    @pytest.fixture
    def parser(self):
        """Create a parser with kvcache subcommands."""
        parser = argparse.ArgumentParser()
        add_kvcache_arguments(parser)
        return parser

    def test_duration_argument(self, parser):
        """Should accept --duration argument."""
        args = parser.parse_args(['run', '--duration', '300'])
        assert args.duration == 300

    def test_generation_mode_argument(self, parser):
        """Should accept --generation-mode argument."""
        for mode in ['none', 'fast', 'realistic']:
            args = parser.parse_args(['run', '--generation-mode', mode])
            assert args.generation_mode == mode

    def test_performance_profile_argument(self, parser):
        """Should accept --performance-profile argument."""
        for profile in ['latency', 'throughput']:
            args = parser.parse_args(['run', '--performance-profile', profile])
            assert args.performance_profile == profile

    def test_seed_argument(self, parser):
        """Should accept --seed argument."""
        args = parser.parse_args(['run', '--seed', '42'])
        assert args.seed == 42


class TestKVCacheDistributedArguments:
    """Tests for KV cache distributed execution arguments."""

    @pytest.fixture
    def parser(self):
        """Create a parser with kvcache subcommands."""
        parser = argparse.ArgumentParser()
        add_kvcache_arguments(parser)
        return parser

    def test_hosts_argument(self, parser):
        """Run should accept --hosts argument."""
        args = parser.parse_args(['run', '--hosts', 'host1', 'host2', 'host3'])
        assert args.hosts == ['host1', 'host2', 'host3']

    def test_hosts_short_flag(self, parser):
        """Run should accept -s shorthand for --hosts."""
        args = parser.parse_args(['run', '-s', 'node1', 'node2'])
        assert args.hosts == ['node1', 'node2']

    def test_hosts_default(self, parser):
        """Hosts should default to localhost."""
        args = parser.parse_args(['run'])
        assert args.hosts == ['127.0.0.1']

    def test_exec_type_argument_mpi(self, parser):
        """Run should accept --exec-type mpi."""
        args = parser.parse_args(['run', '--exec-type', 'mpi'])
        assert args.exec_type == EXEC_TYPE.MPI

    def test_exec_type_argument_docker(self, parser):
        """Run should accept --exec-type docker."""
        args = parser.parse_args(['run', '--exec-type', 'docker'])
        assert args.exec_type == EXEC_TYPE.DOCKER

    def test_exec_type_default(self, parser):
        """exec_type should default to MPI."""
        args = parser.parse_args(['run'])
        assert args.exec_type == EXEC_TYPE.MPI

    def test_exec_type_short_flag(self, parser):
        """Run should accept -et shorthand for --exec-type."""
        args = parser.parse_args(['run', '-et', 'mpi'])
        assert args.exec_type == EXEC_TYPE.MPI

    def test_num_processes_argument(self, parser):
        """Run should accept --num-processes argument."""
        args = parser.parse_args(['run', '--num-processes', '16'])
        assert args.num_processes == 16

    def test_num_processes_short_flag(self, parser):
        """Run should accept -np shorthand for --num-processes."""
        args = parser.parse_args(['run', '-np', '8'])
        assert args.num_processes == 8


class TestKVCacheMPIArguments:
    """Tests for KV cache MPI-related arguments."""

    @pytest.fixture
    def parser(self):
        """Create a parser with kvcache subcommands."""
        parser = argparse.ArgumentParser()
        add_kvcache_arguments(parser)
        return parser

    def test_mpi_bin_argument(self, parser):
        """Run should accept --mpi-bin argument."""
        args = parser.parse_args(['run', '--mpi-bin', 'mpirun'])
        assert args.mpi_bin == 'mpirun'

    def test_mpi_bin_mpiexec(self, parser):
        """Run should accept --mpi-bin mpiexec."""
        args = parser.parse_args(['run', '--mpi-bin', 'mpiexec'])
        assert args.mpi_bin == 'mpiexec'

    def test_oversubscribe_argument(self, parser):
        """Run should accept --oversubscribe argument."""
        args = parser.parse_args(['run', '--oversubscribe'])
        assert args.oversubscribe is True

    def test_allow_run_as_root_argument(self, parser):
        """Run should accept --allow-run-as-root argument."""
        args = parser.parse_args(['run', '--allow-run-as-root'])
        assert args.allow_run_as_root is True

    def test_mpi_params_argument(self, parser):
        """Run should accept --mpi-params argument."""
        args = parser.parse_args(['run', '--mpi-params', 'param1', 'param2'])
        assert args.mpi_params == [['param1', 'param2']]


class TestKVCacheDatasizeNoDistributedArgs:
    """Tests verifying datasize command doesn't have distributed execution args."""

    @pytest.fixture
    def parser(self):
        """Create a parser with kvcache subcommands."""
        parser = argparse.ArgumentParser()
        add_kvcache_arguments(parser)
        return parser

    def test_datasize_no_hosts_argument(self, parser):
        """Datasize should not have --hosts argument."""
        # Parse datasize without --hosts - should work
        args = parser.parse_args(['datasize'])
        # hosts should not be in the namespace
        assert not hasattr(args, 'hosts')

    def test_datasize_no_exec_type_argument(self, parser):
        """Datasize should not have --exec-type argument."""
        args = parser.parse_args(['datasize'])
        assert not hasattr(args, 'exec_type')

    def test_datasize_no_num_processes_argument(self, parser):
        """Datasize should not have --num-processes argument."""
        args = parser.parse_args(['datasize'])
        assert not hasattr(args, 'num_processes')

    def test_datasize_no_mpi_bin_argument(self, parser):
        """Datasize should not have --mpi-bin argument."""
        args = parser.parse_args(['datasize'])
        assert not hasattr(args, 'mpi_bin')

    def test_datasize_basic_args_work(self, parser):
        """Datasize should work with basic model and cache args."""
        args = parser.parse_args([
            'datasize',
            '--model', 'llama3.1-70b-instruct',
            '--num-users', '500',
            '--gpu-mem-gb', '80',
            '--cpu-mem-gb', '256'
        ])
        assert args.command == 'datasize'
        assert args.model == 'llama3.1-70b-instruct'
        assert args.num_users == 500
        assert args.gpu_mem_gb == 80.0
        assert args.cpu_mem_gb == 256.0


class TestKVCacheOptionalFeatures:
    """Tests for KV cache optional feature arguments."""

    @pytest.fixture
    def parser(self):
        """Create a parser with kvcache subcommands."""
        parser = argparse.ArgumentParser()
        add_kvcache_arguments(parser)
        return parser

    def test_disable_multi_turn_argument(self, parser):
        """Run should accept --disable-multi-turn argument."""
        args = parser.parse_args(['run', '--disable-multi-turn'])
        assert args.disable_multi_turn is True

    def test_disable_prefix_caching_argument(self, parser):
        """Run should accept --disable-prefix-caching argument."""
        args = parser.parse_args(['run', '--disable-prefix-caching'])
        assert args.disable_prefix_caching is True

    def test_enable_rag_argument(self, parser):
        """Run should accept --enable-rag argument."""
        args = parser.parse_args(['run', '--enable-rag'])
        assert args.enable_rag is True

    def test_rag_num_docs_argument(self, parser):
        """Run should accept --rag-num-docs argument."""
        args = parser.parse_args(['run', '--rag-num-docs', '20'])
        assert args.rag_num_docs == 20

    def test_enable_autoscaling_argument(self, parser):
        """Run should accept --enable-autoscaling argument."""
        args = parser.parse_args(['run', '--enable-autoscaling'])
        assert args.enable_autoscaling is True

    def test_autoscaler_mode_argument(self, parser):
        """Run should accept --autoscaler-mode argument."""
        for mode in ['qos', 'predictive']:
            args = parser.parse_args(['run', '--autoscaler-mode', mode])
            assert args.autoscaler_mode == mode


class TestKVCacheRunMLPerfArguments:
    """Tests for MLPerf sequence arguments now on the run subcommand."""

    @pytest.fixture
    def parser(self):
        parser = argparse.ArgumentParser()
        add_kvcache_arguments(parser)
        return parser

    def test_validate_subcommand_no_longer_exists(self, parser):
        """The validate subcommand was merged into run; parsing it must fail."""
        with pytest.raises(SystemExit):
            parser.parse_args(['validate', '--cache-dir', '/tmp/kv'])

    def test_npernode_default_is_1(self, parser):
        args = parser.parse_args(['run'])
        assert args.npernode == 1

    def test_npernode_accepts_value(self, parser):
        args = parser.parse_args(['run', '--npernode', '4'])
        assert args.npernode == 4

    def test_npernode_long_form_accepted(self, parser):
        args = parser.parse_args(['run', '--num-processes-per-client', '2'])
        assert args.npernode == 2

    def test_seed_default_is_none(self, parser):
        """seed defaults to None so CLOSED enforcement can detect explicit setting."""
        args = parser.parse_args(['run'])
        assert args.seed is None

    def test_seed_accepts_value(self, parser):
        args = parser.parse_args(['run', '--seed', '42'])
        assert args.seed == 42

    def test_trials_default_is_none(self, parser):
        """trials defaults to None so CLOSED enforcement can detect explicit setting."""
        args = parser.parse_args(['run'])
        assert args.trials is None

    def test_trials_accepts_value(self, parser):
        args = parser.parse_args(['run', '--trials', '5'])
        assert args.trials == 5

    def test_inter_option_delay_default_is_none(self, parser):
        """inter_option_delay defaults to None so CLOSED enforcement can detect explicit setting."""
        args = parser.parse_args(['run'])
        assert args.inter_option_delay is None

    def test_inter_option_delay_accepts_value(self, parser):
        args = parser.parse_args(['run', '--inter-option-delay', '5'])
        assert args.inter_option_delay == 5

    def test_config_default_is_none(self, parser):
        args = parser.parse_args(['run'])
        assert args.config is None

    def test_config_argument_accepted(self, parser):
        args = parser.parse_args(['run', '--config', '/path/to/config.yaml'])
        assert args.config == '/path/to/config.yaml'

    def test_kvcache_bin_path_accepted(self, parser):
        args = parser.parse_args(['run', '--kvcache-bin-path', '/opt/kv-cache.py'])
        assert args.kvcache_bin_path == '/opt/kv-cache.py'
