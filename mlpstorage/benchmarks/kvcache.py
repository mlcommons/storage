"""
KV Cache Benchmark for MLPerf Storage.

This module provides the KVCacheBenchmark class that wraps the kv-cache.py
benchmark script for integration into the mlpstorage framework.

The KV Cache benchmark simulates storage system performance for Large Language
Model (LLM) Key-Value cache offloading, including:
- Multi-tier cache (GPU → CPU → NVMe)
- Phase-aware processing (prefill/decode phases)
- Multi-tenant inference environment simulation
- Adaptive autoscaling capabilities

Classes:
    KVCacheBenchmark: Benchmark implementation for KV cache workloads.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

from mlpstorage.benchmarks.base import Benchmark
from mlpstorage.config import (
    BENCHMARK_TYPES,
    EXEC_TYPE,
    KVCACHE_MODELS,
    KVCACHE_DEFAULT_DURATION,
    KVCACHE_PERFORMANCE_PROFILES,
    KVCACHE_GENERATION_MODES,
)
from mlpstorage.interfaces import BenchmarkCommand
from mlpstorage.utils import generate_mpi_prefix_cmd


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
                - gpu_mem_gb: GPU memory for cache tier (GB)
                - cpu_mem_gb: CPU memory for cache tier (GB)
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
        """Execute the KV cache benchmark run.

        Generates and executes the kv-cache.py command with configured
        parameters, then processes the results.

        Returns:
            Exit code from benchmark execution.
        """
        # Verify benchmark parameters if running for submission
        if hasattr(self.args, 'closed') and self.args.closed:
            self.verify_benchmark()

        # Build the command
        cmd = self._build_kvcache_command()

        self.logger.status(f"Running KV Cache benchmark for {self.duration}s...")
        self.logger.status(f"Model: {self.model}, Users: {self.num_users}")

        # Execute the command
        stdout, stderr, return_code = self._execute_command(
            cmd,
            output_file_prefix=f"kvcache_{self.run_datetime}",
            print_stdout=True,
            print_stderr=True
        )

        # Process results if successful
        if return_code == 0:
            self._process_results()

        # Write metadata
        self.write_metadata()
        self.write_cluster_info()

        return return_code

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
        self.logger.info(f"  Per-user cache estimate: {cache_per_user_mb:.2f} MB")
        self.logger.info(f"  Total for {self.num_users} users: {total_cache_mb:.2f} MB")
        self.logger.info(f"\nRecommended tier sizes:")
        self.logger.info(f"  GPU memory: {max(self.gpu_mem_gb, total_cache_mb/1024 * 0.2):.1f} GB")
        self.logger.info(f"  CPU memory: {max(self.cpu_mem_gb, total_cache_mb/1024 * 0.5):.1f} GB")
        self.logger.info(f"  NVMe storage: {total_cache_mb/1024 * 2:.1f} GB (2x for headroom)")

        return 0

    def _build_kvcache_command(self) -> str:
        """Build the kv-cache.py command with parameters.

        Constructs the full command line for executing kv-cache.py
        with all configured parameters.

        Returns:
            Command string ready for execution.
        """
        cmd_parts = [
            sys.executable,  # Use same Python interpreter
            self.kvcache_bin_path,
            f"--model {self.model}",
            f"--num-users {self.num_users}",
            f"--duration {self.duration}",
            f"--gpu-mem-gb {self.gpu_mem_gb}",
            f"--cpu-mem-gb {self.cpu_mem_gb}",
            f"--generation-mode {self.generation_mode}",
            f"--performance-profile {self.performance_profile}",
        ]

        # Add cache directory if specified
        if self.cache_dir:
            cmd_parts.append(f"--cache-dir {self.cache_dir}")
        else:
            # Use results directory for cache
            cache_path = os.path.join(self.run_result_output, "kvcache_data")
            cmd_parts.append(f"--cache-dir {cache_path}")

        # Output file
        output_file = os.path.join(
            self.run_result_output,
            f"kvcache_results_{self.run_datetime}.json"
        )
        cmd_parts.append(f"--output {output_file}")

        # Optional features based on args
        if getattr(self.args, 'disable_multi_turn', False):
            cmd_parts.append("--disable-multi-turn")

        if getattr(self.args, 'disable_prefix_caching', False):
            cmd_parts.append("--disable-prefix-caching")

        if getattr(self.args, 'enable_rag', False):
            cmd_parts.append("--enable-rag")
            rag_docs = getattr(self.args, 'rag_num_docs', 10)
            cmd_parts.append(f"--rag-num-docs {rag_docs}")

        if getattr(self.args, 'enable_autoscaling', False):
            cmd_parts.append("--enable-autoscaling")
            autoscaler_mode = getattr(self.args, 'autoscaler_mode', 'qos')
            cmd_parts.append(f"--autoscaler-mode {autoscaler_mode}")

        # Seed for reproducibility
        seed = getattr(self.args, 'seed', None)
        if seed is not None:
            cmd_parts.append(f"--seed {seed}")

        # Build the base command
        cmd = " ".join(cmd_parts)

        # Add MPI wrapper if distributed execution requested
        exec_type = getattr(self.args, 'exec_type', None)
        if exec_type == EXEC_TYPE.MPI:
            hosts = getattr(self.args, 'hosts', None)
            if hosts and len(hosts) > 0:
                # Default num_processes to number of hosts if not specified
                num_procs = self.num_processes or len(hosts)
                mpi_prefix = generate_mpi_prefix_cmd(
                    mpi_cmd=getattr(self.args, 'mpi_bin', 'mpirun'),
                    hosts=hosts,
                    num_processes=num_procs,
                    oversubscribe=getattr(self.args, 'oversubscribe', False),
                    allow_run_as_root=getattr(self.args, 'allow_run_as_root', False),
                    params=getattr(self.args, 'mpi_params', None),
                    logger=self.logger
                )
                cmd = f"{mpi_prefix} {cmd}"

        return cmd

    def _process_results(self):
        """Process KV cache benchmark results.

        Parses the output JSON file and extracts key metrics
        for storage in metadata.
        """
        import json

        output_file = os.path.join(
            self.run_result_output,
            f"kvcache_results_{self.run_datetime}.json"
        )

        if not os.path.exists(output_file):
            self.logger.warning(f"Results file not found: {output_file}")
            return

        try:
            with open(output_file, 'r') as f:
                results = json.load(f)

            # Extract key metrics
            self.metrics = {
                'overall_passed': results.get('overall_passed', False),
                'total_requests': results.get('total_requests', 0),
                'avg_latency_ms': results.get('avg_latency_ms', 0),
                'p99_latency_ms': results.get('p99_latency_ms', 0),
                'throughput_req_per_sec': results.get('throughput_req_per_sec', 0),
                'storage_throughput_mbps': results.get('storage_throughput_mbps', 0),
                'cache_hit_rate': results.get('cache_hit_rate', 0),
            }

            self.logger.status(f"Benchmark passed: {self.metrics['overall_passed']}")
            self.logger.info(f"Total requests: {self.metrics['total_requests']}")
            self.logger.info(f"Average latency: {self.metrics['avg_latency_ms']:.2f} ms")
            self.logger.info(f"P99 latency: {self.metrics['p99_latency_ms']:.2f} ms")
            self.logger.info(f"Throughput: {self.metrics['throughput_req_per_sec']:.2f} req/s")

        except Exception as e:
            self.logger.warning(f"Failed to process results: {e}")

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
            'num_users': self.num_users,
            'duration': self.duration,
            'gpu_mem_gb': self.gpu_mem_gb,
            'cpu_mem_gb': self.cpu_mem_gb,
            'cache_dir': self.cache_dir,
            'generation_mode': self.generation_mode,
            'performance_profile': self.performance_profile,
        })

        # Add metrics if available
        if hasattr(self, 'metrics'):
            base_metadata['kvcache_metrics'] = self.metrics

        return base_metadata

    def generate_command(self, command: str) -> str:
        """Generate the shell command to execute.

        Args:
            command: Command string ('run', 'datasize').

        Returns:
            Shell command string.
        """
        if command == 'run':
            return self._build_kvcache_command()
        return ""
