"""
KV Cache Benchmark for MLPerf Storage.

This module provides the KVCacheBenchmark class that wraps the kv-cache.py
benchmark script for integration into the mlpstorage_py framework.

The KV Cache benchmark simulates storage system performance for Large Language
Model (LLM) Key-Value cache offloading, including:
- Multi-tier cache (GPU → CPU → NVMe)
- Phase-aware processing (prefill/decode phases)
- Multi-tenant inference environment simulation
- Adaptive autoscaling capabilities

Classes:
    KVCacheBenchmark: Benchmark implementation for KV cache workloads.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

from mlpstorage_py.benchmarks.base import Benchmark
from mlpstorage_py.config import (
    BENCHMARK_TYPES,
    KVCACHE_DEFAULT_DURATION,
)
from mlpstorage_py.interfaces import BenchmarkCommand
from mlpstorage_py.utils import generate_mpi_prefix_cmd, MLPSJsonEncoder


class KVCacheBenchmark(Benchmark):
    """KV Cache benchmark for LLM inference storage.

    This benchmark tests storage system performance for KV cache offloading
    in large language model inference workloads. It wraps the kv-cache.py
    script from the kv_cache_benchmark directory.

    Key Features:
    - Multi-tier caching (GPU → CPU → NVMe)
    - Phase-aware processing (prefill vs decode phases)
    - Multi-user simulation
    - Support for various LLM model configurations
    - Autoscaling capabilities

    Attributes:
        BENCHMARK_TYPE: Enum identifying this as a KV cache benchmark.
        KVCACHE_SCRIPT: Name of the kv-cache.py script to execute.

    Example:
        benchmark = KVCacheBenchmark(args, logger=logger)
        result = benchmark.run()
    """

    BENCHMARK_TYPE = BENCHMARK_TYPES.kv_cache
    KVCACHE_SCRIPT = "kv-cache.py"

    def __init__(self, args, logger=None, run_datetime=None, run_number=0,
                 cluster_collector=None, validator=None):
        """Initialize the KV Cache benchmark.

        Args:
            args: Parsed command-line arguments containing:
                - model: KV cache model configuration to use
                - num_users: Number of concurrent users to simulate
                - duration: Benchmark duration in seconds
                - gpu_mem_gb: GPU memory for cache tier (GiB)
                - cpu_mem_gb: CPU memory for cache tier (GiB)
                - cache_dir: Directory for NVMe cache tier
                - generation_mode: Token generation simulation mode
                - performance_profile: Pass/fail criteria profile
            logger: Logger instance for output.
            run_datetime: Datetime string for the run.
            run_number: Run number for loop execution.
            cluster_collector: Optional cluster collector for DI.
            validator: Optional validator for DI.
        """
        super().__init__(args, logger, run_datetime, run_number,
                         cluster_collector, validator)

        # Store num_processes for MPI execution
        self.num_processes = getattr(args, 'num_processes', None)

        # Collect cluster information for distributed runs
        if getattr(args, 'command', '') == 'run':
            self.cluster_information = self._collect_cluster_information()

        # Command handler mapping
        self.command_method_map = {
            "run": self._execute_run,
            "datasize": self._execute_datasize,
        }

        # Store key parameters
        self.model = getattr(args, 'model', 'llama3.1-8b')
        self.num_users = getattr(args, 'num_users', 100)
        self.duration = getattr(args, 'duration', KVCACHE_DEFAULT_DURATION)

        # Cache configuration
        self.gpu_mem_gb = getattr(args, 'gpu_mem_gb', 16.0)
        self.cpu_mem_gb = getattr(args, 'cpu_mem_gb', 32.0)
        self.cache_dir = getattr(args, 'cache_dir', None)

        # Benchmark configuration
        self.generation_mode = getattr(args, 'generation_mode', 'realistic')
        self.performance_profile = getattr(args, 'performance_profile', 'latency')

        # Find the kv-cache.py script
        self.kvcache_bin_path = self._find_kvcache_script()

    def _find_kvcache_script(self) -> str:
        """Locate the kv-cache.py script.

        Searches for the script in:
        1. Custom path from --kvcache-bin-path argument
        2. kv_cache_benchmark directory relative to project root
        3. Current working directory

        Returns:
            Absolute path to kv-cache.py script.

        Raises:
            FileNotFoundError: If script cannot be found.
        """
        # Check for custom path first
        custom_path = getattr(self.args, 'kvcache_bin_path', None)
        if custom_path and os.path.isfile(custom_path):
            return os.path.abspath(custom_path)

        # Look in kv_cache_benchmark directory
        project_root = Path(__file__).parent.parent.parent
        kvcache_dir = project_root / "kv_cache_benchmark"
        kvcache_script = kvcache_dir / self.KVCACHE_SCRIPT

        if kvcache_script.exists():
            return str(kvcache_script)

        # Check current directory
        local_script = Path(self.KVCACHE_SCRIPT)
        if local_script.exists():
            return str(local_script.absolute())

        self.logger.warning(
            f"KV Cache script not found. Expected at: {kvcache_script}"
        )
        return self.KVCACHE_SCRIPT  # Return name, let execution fail with clear error

    def _get_supported_commands(self) -> List[BenchmarkCommand]:
        """Return supported commands for KV Cache benchmark."""
        return [BenchmarkCommand.RUN, BenchmarkCommand.DATASIZE]

    def _run(self) -> int:
        """Execute the benchmark based on the command.

        Routes to the appropriate command handler based on args.command.

        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        command = getattr(self.args, 'command', 'run')
        handler = self.command_method_map.get(command)

        if handler:
            return handler()
        else:
            self.logger.error(f"Unknown command: {command}")
            return 1

    def _execute_run(self) -> int:
        """Execute the MLPerf v3.0 KV cache benchmark sequence across all three options.

        Runs options 1, 2, and 3 sequentially via mpirun targeting mlperf_wrapper.py.
        Each option runs `trials` times with `inter_option_delay` seconds between options.

        In CLOSED submissions, seed, trials, inter-option-delay, and --config are fixed
        to their mandated values; the run hard-fails if the user attempts to override them.

        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        is_closed = getattr(self.args, 'closed', False)

        # Enforce CLOSED submission restrictions — hard fail on illegal overrides
        seed_arg = getattr(self.args, 'seed', None)
        if is_closed and seed_arg is not None and seed_arg != 42:
            self.logger.error(
                f"--seed cannot be changed in a CLOSED submission (must be 42, got {seed_arg})"
            )
            return 1

        trials_arg = getattr(self.args, 'trials', None)
        if is_closed and trials_arg is not None and trials_arg != 3:
            self.logger.error(
                f"--trials cannot be changed in a CLOSED submission (must be 3, got {trials_arg})"
            )
            return 1

        inter_option_delay_arg = getattr(self.args, 'inter_option_delay', None)
        if is_closed and inter_option_delay_arg is not None and inter_option_delay_arg != 20:
            self.logger.error(
                f"--inter-option-delay cannot be changed in a CLOSED submission "
                f"(must be 20, got {inter_option_delay_arg})"
            )
            return 1

        config_arg = getattr(self.args, 'config', None)
        if is_closed and config_arg is not None:
            self.logger.error("--config is not valid in a CLOSED submission")
            return 1

        # Resolve effective values, applying mandated defaults
        seed = seed_arg if seed_arg is not None else 42
        trials = trials_arg if trials_arg is not None else 3
        inter_option_delay = inter_option_delay_arg if inter_option_delay_arg is not None else 20
        config = config_arg

        hosts = getattr(self.args, 'hosts', None) or ['localhost']
        npernode = getattr(self.args, 'npernode', 1)
        total_ranks = npernode * len(hosts)
        cache_dir = (
            getattr(self.args, 'cache_dir', None)
            or str(Path(self.run_result_output) / 'kvcache_cache')
        )

        wrapper_path = Path(self.kvcache_bin_path).parent / 'mlperf_wrapper.py'

        mpi_prefix = generate_mpi_prefix_cmd(
            mpi_cmd=getattr(self.args, 'mpi_bin', 'mpirun'),
            hosts=hosts,
            num_processes=total_ranks,
            oversubscribe=getattr(self.args, 'oversubscribe', False),
            allow_run_as_root=getattr(self.args, 'allow_run_as_root', False),
            params=['--mca orte_abort_on_non_zero_status 0'],
            logger=self.logger,
            processes_per_node=npernode,
        )

        option_results = {}
        for option in [1, 2, 3]:
            trial_dirs = []

            for trial in range(trials):
                option_trial_dir = (
                    Path(self.run_result_output) / f"option_{option}" / f"trial_{trial}"
                )
                option_trial_dir.mkdir(parents=True, exist_ok=True)

                wrapper_cmd = (
                    f"{mpi_prefix} {sys.executable} {wrapper_path}"
                    f" --option {option}"
                    f" --seed {seed}"
                    f" --base-output-dir {option_trial_dir}"
                    f" --cache-dir {cache_dir}"
                )
                if config:
                    wrapper_cmd += f" --config {config}"

                self.logger.status(f"Running option {option} trial {trial + 1}/{trials}...")
                self._execute_command(
                    wrapper_cmd,
                    output_file_prefix=f"kvcache_opt{option}_trial{trial}_{self.run_datetime}",
                    print_stdout=True,
                    print_stderr=True,
                )
                trial_dirs.append(str(option_trial_dir))

            if not getattr(self.args, 'what_if', False):
                option_results[option] = self._aggregate_option_results(
                    option, trial_dirs, total_ranks
                )
            else:
                self.logger.info(f"what-if: skipping aggregation for option {option}")

            if option < 3:
                self._interruptible_sleep(inter_option_delay)

        if not getattr(self.args, 'what_if', False):
            self._write_run_summary(option_results, npernode, len(hosts), total_ranks, trials)

        self.write_metadata()
        self.write_cluster_info()
        return 0

    def _execute_datasize(self) -> int:
        """Calculate memory requirements for KV cache.

        Provides estimates for GPU, CPU, and NVMe cache tiers based
        on model configuration and number of users.

        Returns:
            Exit code (0 for success).
        """
        self.logger.status("Calculating KV Cache memory requirements...")

        # Import model configs from kv-cache.py or use estimates
        model_cache_estimates = {
            'tiny-1b': {'per_token_bytes': 768, 'typical_sequence': 2048},
            'mistral-7b': {'per_token_bytes': 4096, 'typical_sequence': 4096},
            'llama2-7b': {'per_token_bytes': 8192, 'typical_sequence': 4096},
            'llama3.1-8b': {'per_token_bytes': 4096, 'typical_sequence': 8192},
            'llama3.1-70b-instruct': {'per_token_bytes': 16384, 'typical_sequence': 8192},
        }

        model_info = model_cache_estimates.get(self.model, {
            'per_token_bytes': 4096,
            'typical_sequence': 4096
        })

        per_token = model_info['per_token_bytes']
        seq_len = model_info['typical_sequence']

        # Calculate per-user cache size
        cache_per_user_mb = (per_token * seq_len) / (1024 * 1024)
        total_cache_mb = cache_per_user_mb * self.num_users

        self.logger.info(f"\nKV Cache Size Estimates for {self.model}:")
        self.logger.info(f"  Per-token cache: {per_token} bytes")
        self.logger.info(f"  Typical sequence length: {seq_len} tokens")
        self.logger.info(f"  Per-user cache estimate: {cache_per_user_mb:.2f}MiB")
        self.logger.info(f"  Total for {self.num_users} users: {total_cache_mb:.2f}MiB")
        self.logger.info(f"\nRecommended tier sizes:")
        self.logger.info(f"  GPU memory: {max(self.gpu_mem_gb, total_cache_mb/1024 * 0.2):.1f}GiB")
        self.logger.info(f"  CPU memory: {max(self.cpu_mem_gb, total_cache_mb/1024 * 0.5):.1f}GiB")
        self.logger.info(f"  NVMe storage: {total_cache_mb/1024 * 2:.1f}GiB (2x for headroom)")

        return 0

    def _interruptible_sleep(self, seconds: int) -> None:
        """Sleep in 1-second chunks, interruptible by Ctrl-C. Skipped in what-if mode."""
        if getattr(self.args, 'what_if', False):
            return
        for _ in range(seconds):
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Inter-option sleep interrupted by user.")
                raise

    def _aggregate_option_results(
        self,
        option: int,
        trial_dirs: list,
        expected_rank_count: int,
    ) -> dict:
        """Aggregate per-rank JSON results for one option across all trials.

        Sums read/write bandwidth and token throughput across all rank files.
        Takes max storage_io_latency_ms.p95. Records missing files without
        crashing and sets partial_failure. When storage_entries == 0, logs
        that the working set was served from the CPU tier.
        """
        all_read_bw = []
        all_write_bw = []
        all_avg_throughput = []
        all_storage_throughput = []
        all_p95_latency = []
        missing_files = []
        cpu_tier_flags = []

        for trial_dir in trial_dirs:
            for rank_idx in range(expected_rank_count):
                rank_dir = Path(trial_dir) / f"rank_{rank_idx}"
                result_file = next(rank_dir.glob('kvcache_results_*.json'), None)
                if result_file is None:
                    missing_files.append(str(rank_dir))
                    self.logger.warning(f"No result file in {rank_dir}")
                    continue
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    summary = data.get('summary', {})
                    cache_stats = summary.get('cache_stats', {})
                    if cache_stats.get('storage_entries', None) == 0:
                        self.logger.info(
                            f"Rank {rank_idx} trial {trial_dir}: working set served from CPU tier"
                        )
                        cpu_tier_flags.append(str(result_file))
                    # Include all values regardless (0 is correct for CPU-tier)
                    all_read_bw.append(cache_stats.get('tier_storage_read_bandwidth_gbps', 0.0))
                    all_write_bw.append(cache_stats.get('tier_storage_write_bandwidth_gbps', 0.0))
                    all_avg_throughput.append(summary.get('avg_throughput_tokens_per_sec', 0.0))
                    all_storage_throughput.append(summary.get('storage_throughput_tokens_per_sec', 0.0))
                    all_p95_latency.append(summary.get('storage_io_latency_ms', {}).get('p95', 0.0))
                except Exception as e:
                    self.logger.warning(f"Failed to parse {result_file}: {e}")
                    missing_files.append(str(result_file))

        return {
            'option': option,
            'aggregated_read_bandwidth_gbps': sum(all_read_bw),
            'aggregated_write_bandwidth_gbps': sum(all_write_bw),
            'aggregated_avg_throughput_tokens_per_sec': sum(all_avg_throughput),
            'aggregated_storage_throughput_tokens_per_sec': sum(all_storage_throughput),
            'aggregated_p95_latency_ms': max(all_p95_latency) if all_p95_latency else None,
            'rank_count': expected_rank_count,
            'trial_count': len(trial_dirs),
            'partial_failure': len(missing_files) > 0,
            'missing_files': missing_files,
            'cpu_tier_ranks': cpu_tier_flags,
        }

    def _write_run_summary(
        self,
        option_results: dict,
        npernode: int,
        host_count: int,
        total_ranks: int,
        trials: int,
    ) -> None:
        """Write aggregated run summary JSON to run_result_output."""
        summary = {
            'schema_version': '1.0',
            'run_datetime': self.run_datetime,
            'npernode': npernode,
            'host_count': host_count,
            'total_ranks': total_ranks,
            'trials_per_option': trials,
            'options': option_results,
            'partial_failure': any(
                r.get('partial_failure', False) for r in option_results.values()
            ),
        }
        summary_filename = f"kvcache_run_summary_{self.run_datetime}.json"
        summary_path = Path(self.run_result_output) / summary_filename
        with open(summary_path, 'w+') as fd:
            json.dump(summary, fd, indent=2, cls=MLPSJsonEncoder)
        self.logger.status(f"Run summary written to: {summary_path}")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Generate metadata for the KV cache benchmark run.

        Returns:
            Dictionary containing benchmark metadata.
        """
        base_metadata = super().metadata

        # Add KV cache specific metadata
        base_metadata.update({
            'kvcache_model': self.model,
            'model': self.model,  # Add for consistency with other benchmarks
            'num_users': self.num_users,
            'duration': self.duration,
            'gpu_mem_gb': self.gpu_mem_gb,
            'cpu_mem_gb': self.cpu_mem_gb,
            'cache_dir': self.cache_dir,
            'generation_mode': self.generation_mode,
            'performance_profile': self.performance_profile,
            'num_processes': self.num_processes,  # Include for distributed runs
        })

        # Add execution info for distributed runs
        exec_type = getattr(self.args, 'exec_type', None)
        if exec_type:
            base_metadata['exec_type'] = exec_type.value if hasattr(exec_type, 'value') else str(exec_type)

        hosts = getattr(self.args, 'hosts', None)
        if hosts:
            base_metadata['hosts'] = hosts

        # Add metrics if available
        if hasattr(self, 'metrics'):
            base_metadata['kvcache_metrics'] = self.metrics

        return base_metadata

    def generate_command(self, command: str) -> str:
        """Generate the shell command to execute."""
        return ""
