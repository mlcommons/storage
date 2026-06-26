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
import shlex
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List
from statistics import fmean

from mlpstorage_py.benchmarks.base import Benchmark
from mlpstorage_py.cluster_collector import _is_localhost
from mlpstorage_py.config import (
    BENCHMARK_TYPES,
    KVCACHE_DEFAULT_DURATION,
    KVCACHE_MODEL_DEFAULT,
)
from mlpstorage_py.interfaces import BenchmarkCommand
from mlpstorage_py.utils import generate_mpi_prefix_cmd, MLPSJsonEncoder


# MLPerf v3.0 fixed per-option workload parameters. Single source of truth for
# what option N means at the kv-cache.py level. In CLOSED these are mandated;
# in OPEN they act as per-option defaults that user CLI flags supersede.
WORKLOAD_PARAMS = {
    1: {
        'model': 'llama3.1-8b',
        'num-users': 200,
        'duration': 300,
        'gpu-mem-gb': 0,
        'cpu-mem-gb': 0,
        'max-concurrent-allocs': 16,
        'generation-mode': 'none',
    },
    2: {
        'model': 'llama3.1-8b',
        'num-users': 100,
        'duration': 300,
        'gpu-mem-gb': 0,
        'cpu-mem-gb': 4,
        'max-concurrent-allocs': 16,
        'generation-mode': 'none',
    },
    3: {
        'model': 'llama3.1-70b-instruct',
        'num-users': 70,
        'duration': 300,
        'gpu-mem-gb': 0,
        'cpu-mem-gb': 0,
        'max-concurrent-allocs': 4,
        'generation-mode': 'none',
    },
}


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
        # Closed-mode kvcache CLI does not expose --model (the model is fixed),
        # so args.model may be absent. The model is required as a path
        # component (kv_cache/<model>/<command>/<datetime>/) and as a
        # workload-grouping key, so guarantee args.model is set with the
        # closed-mode default before the base class computes the output path.
        if getattr(args, "model", None) is None:
            args.model = KVCACHE_MODEL_DEFAULT
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

        # Store key parameters. args.model is guaranteed above.
        self.model = args.model
        self.num_users = getattr(args, 'num_users', 100)
        self.duration = getattr(args, 'duration', KVCACHE_DEFAULT_DURATION)

        # Cache configuration
        self.gpu_mem_gb = getattr(args, 'gpu_mem_gb', 16.0)
        self.cpu_mem_gb = getattr(args, 'cpu_mem_gb', 32.0)
        self.cache_dir = getattr(args, 'cache_dir', None)

        # Benchmark configuration
        self.generation_mode = getattr(args, 'generation_mode', 'realistic')
        self.performance_profile = getattr(args, 'performance_profile', 'throughput')

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
        is_closed = (getattr(self.args, 'mode', None) == 'closed')

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
        npernode, total_ranks = self._resolve_rank_layout(hosts)
        if total_ranks is None:
            return 1  # error already logged
        cache_dir = (
            getattr(self.args, 'cache_dir', None)
            or str(Path(self.run_result_output) / 'kvcache_cache')
        )

        wrapper_path = Path(self.kvcache_bin_path).parent / 'mlperf_wrapper.py'
        # Wrapper-adjacent config.yaml is the default; CLOSED forbids overriding.
        config_path = config or str(Path(self.kvcache_bin_path).parent / 'config.yaml')

        # User --mpi-params (already shlex-flattened by the central CLI parser)
        # are passed through first; the mandatory --mca is appended last so
        # that OpenMPI's last-wins resolution for repeated --mca keys keeps
        # the abort-suppression flag authoritative even if the user supplies a
        # conflicting value (kvcache expects per-rank non-zero exits).
        user_mpi_params = list(getattr(self.args, 'mpi_params', None) or [])
        mpi_params = user_mpi_params + ['--mca', 'orte_abort_on_non_zero_status', '0']

        mpi_prefix = generate_mpi_prefix_cmd(
            mpi_cmd=getattr(self.args, 'mpi_bin', 'mpirun'),
            hosts=hosts,
            num_processes=total_ranks,
            oversubscribe=getattr(self.args, 'oversubscribe', False),
            allow_run_as_root=getattr(self.args, 'allow_run_as_root', False),
            params=mpi_params,
            logger=self.logger,
            processes_per_node=npernode,
        )

        # Issue #521: rank result JSONs are written on the node where each rank
        # lands, but aggregation globs them locally on the controller. Without a
        # shared filesystem, remote-host results are invisible and the run
        # silently records partial_failure. Fail fast with an actionable error.
        if not getattr(self.args, 'what_if', False):
            self._probe_results_dir_shared(hosts)

        option_results = {}
        for option in [1, 2, 3]:
            option_kv_args = self._build_option_kvcache_args(option, is_closed)
            trial_dirs = []

            for trial in range(trials):
                option_trial_dir = (
                    Path(self.run_result_output) / f"option_{option}" / f"trial_{trial}"
                )
                option_trial_dir.mkdir(parents=True, exist_ok=True)

                wrapper_cmd = (
                    f"{mpi_prefix} {sys.executable} {wrapper_path}"
                    f" --rank-output-base {option_trial_dir}"
                    f" --rank-cache-base {cache_dir}"
                    f" --seed-base {seed}"
                    f" --config {config_path}"
                    f" {' '.join(option_kv_args)}"
                )

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

    # ------------------------------------------------------------------
    # CAP-01 capacity-gate hooks (Phase 5 / Plan 05-03)
    # ------------------------------------------------------------------

    # Inline model_cache_estimates table mirrored from _execute_datasize()
    # below. Per A6, CAP-01 routes through THIS internal table — NOT
    # config.py LLM_SIZE_BY_RANK / LLAMA3_* constants. The two tables track
    # different things (KV cache footprint vs full model weights).
    _MODEL_CACHE_ESTIMATES = {
        'tiny-1b': {'per_token_bytes': 768, 'typical_sequence': 2048},
        'mistral-7b': {'per_token_bytes': 4096, 'typical_sequence': 4096},
        'llama2-7b': {'per_token_bytes': 8192, 'typical_sequence': 4096},
        'llama3.1-8b': {'per_token_bytes': 4096, 'typical_sequence': 8192},
        'llama3.1-70b-instruct': {'per_token_bytes': 16384, 'typical_sequence': 8192},
    }
    _MODEL_CACHE_DEFAULT = {'per_token_bytes': 4096, 'typical_sequence': 4096}

    def required_bytes_for_capacity_gate(self) -> int:
        """Return KV-cache footprint in bytes (CAP-01).

        A6 lock: enforces the 1x FLOOR — ``int(total_cache_mb * 1024 *
        1024)``. The 2x NVMe-headroom recommendation at
        ``kvcache.py:_execute_datasize`` is a performance suggestion, not
        a benchmark-cannot-run threshold. Operators who want the 2x
        headroom should size their ``--data-dir`` to 2x manually; CAP-01
        enforces the minimum-to-not-fail.

        Routes through the inline ``_MODEL_CACHE_ESTIMATES`` table — NOT
        ``LLM_SIZE_BY_RANK`` from ``config.py`` (those constants describe
        model weights, not KV cache footprint).
        """
        model_info = self._MODEL_CACHE_ESTIMATES.get(self.model, self._MODEL_CACHE_DEFAULT)
        per_token = model_info['per_token_bytes']
        seq_len = model_info['typical_sequence']
        cache_per_user_mb = (per_token * seq_len) / (1024 * 1024)
        total_cache_mb = cache_per_user_mb * self.num_users
        return int(total_cache_mb * 1024 * 1024)

    def _capacity_gate_destination(self):
        """Return the NVMe KV-cache directory the benchmark writes to.

        Falls back to ``None`` (A8 escape hatch) if ``cache_dir`` is unset —
        the kvcache benchmark may run entirely in CPU/GPU memory tiers when
        the NVMe tier is unconfigured.
        """
        return self.cache_dir if self.cache_dir else None

    # ------------------------------------------------------------------
    # Per-option argv build + rank layout (from main)
    # ------------------------------------------------------------------

    def _build_option_kvcache_args(self, option: int, is_closed: bool) -> List[str]:
        """Return the kv-cache.py CLI args for this option.

        CLOSED: emits WORKLOAD_PARAMS[option] verbatim — MLPerf-mandated, no
        user input can reach kv-cache.py through this path because the CLOSED
        CLI does not expose the corresponding flags.

        OPEN: user-set flags supersede WORKLOAD_PARAMS[option] one key at a
        time. max-concurrent-allocs is not exposed by the OPEN CLI, so it
        always comes from WORKLOAD_PARAMS.
        """
        defaults = WORKLOAD_PARAMS[option]
        if is_closed:
            params = dict(defaults)
        else:
            params = {
                'model': getattr(self.args, 'model', None) or defaults['model'],
                'num-users': (
                    getattr(self.args, 'num_users', None)
                    if getattr(self.args, 'num_users', None) is not None
                    else defaults['num-users']
                ),
                'duration': (
                    getattr(self.args, 'duration', None)
                    if getattr(self.args, 'duration', None) is not None
                    else defaults['duration']
                ),
                'gpu-mem-gb': (
                    getattr(self.args, 'gpu_mem_gb', None)
                    if getattr(self.args, 'gpu_mem_gb', None) is not None
                    else defaults['gpu-mem-gb']
                ),
                'cpu-mem-gb': (
                    getattr(self.args, 'cpu_mem_gb', None)
                    if getattr(self.args, 'cpu_mem_gb', None) is not None
                    else defaults['cpu-mem-gb']
                ),
                'max-concurrent-allocs': defaults['max-concurrent-allocs'],
                'generation-mode': (
                    getattr(self.args, 'generation_mode', None) or defaults['generation-mode']
                ),
            }
        out = []
        for key, value in params.items():
            out.extend([f'--{key}', str(value)])
        return out

    def _resolve_rank_layout(self, hosts):
        """Resolve (npernode, total_ranks) from user-supplied --num-processes and
        --npernode, given the host list. Issue #500.

        Semantics:
          - `--num-processes` is total ranks across the cluster (matches the flag's
            existing help text and DLIO's `--num-accelerators` convention).
          - `--npernode` is ranks per host.
          - If only `--num-processes` is set, `npernode = num_processes // len(hosts)`;
            num_processes must divide evenly across hosts.
          - If only `--npernode` is set, `total_ranks = npernode * len(hosts)`
            (today's behavior — preserves backward compat for existing users).
          - If both are set, they must be consistent
            (`num_processes == npernode * len(hosts)`); otherwise the run fails.
          - If neither is set, defaults to one rank per host.

        Returns:
            (npernode, total_ranks) on success, or (None, None) after logging
            an error on inconsistent / non-divisible input.
        """
        host_count = len(hosts)
        np_arg = getattr(self.args, 'num_processes', None)
        npn_arg = getattr(self.args, 'npernode', None)

        if np_arg is not None and npn_arg is not None and npn_arg != 1:
            # Both explicitly set — require consistency. npn_arg defaults to 1
            # in the CLI builder, so we only treat npernode as "explicitly set"
            # when it diverges from the default to keep CLI default behavior
            # backward-compatible.
            if np_arg != npn_arg * host_count:
                self.logger.error(
                    f"--num-processes ({np_arg}) and --npernode ({npn_arg}) are "
                    f"inconsistent for {host_count} host(s): "
                    f"expected num_processes == npernode * len(hosts) "
                    f"({npn_arg * host_count}). Pass only one of the two, or "
                    f"set them to consistent values."
                )
                return None, None
            return npn_arg, np_arg

        if np_arg is not None:
            if np_arg <= 0:
                self.logger.error(f"--num-processes must be positive, got {np_arg}")
                return None, None
            if np_arg % host_count != 0:
                self.logger.error(
                    f"--num-processes ({np_arg}) must divide evenly across "
                    f"--hosts ({host_count} host(s)). Adjust --num-processes or "
                    f"the host list so num_processes %% len(hosts) == 0."
                )
                return None, None
            return np_arg // host_count, np_arg

        # np_arg is None: fall back to --npernode (default 1)
        npernode = npn_arg if npn_arg is not None else 1
        return npernode, npernode * host_count

    def _execute_datasize(self) -> int:
        """Calculate memory requirements for KV cache.

        Provides estimates for GPU, CPU, and NVMe cache tiers based
        on model configuration and number of users.

        Returns:
            Exit code (0 for success).
        """
        # CAP-01: enforce 1x floor BEFORE printing recommendations. The
        # 2x figure in the existing log lines below is a recommendation,
        # not a hard minimum; CAP-01 enforces the floor.
        self._pre_execution_gate()
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

    def _probe_results_dir_shared(self, hosts: List[str]) -> None:
        """Verify --results-dir is visible at the same path on every host.

        Issue #521: ``mlperf_wrapper.py`` writes ``kvcache_results_*.json``
        on whichever node each rank lands on, but
        ``_aggregate_option_results`` globs locally on the controller. Without
        a filesystem mounted at the same path on every host listed in
        ``--hosts``, the controller never sees the remote-host result files
        and the run silently records ``partial_failure``.

        We probe by asking each unique host (1 rank per host) to drop a
        sentinel file inside ``self.run_result_output``. If the FS is shared,
        the controller sees N sentinels; if not, it sees only the sentinel
        from the host(s) it shares storage with. We fail closed with a
        diagnostic that names the shared-FS requirement, so the user is not
        stuck debugging a partial-failure summary after the option loop has
        already run for ~15+ minutes.

        No-ops when every entry in ``hosts`` resolves to the local machine
        (single-host runs cannot exhibit the bug).
        """
        unique_hosts: List[str] = []
        seen = set()
        for raw in hosts or []:
            hostname = raw.split(':')[0] if ':' in raw else raw
            if hostname and hostname not in seen:
                seen.add(hostname)
                unique_hosts.append(hostname)

        if len(unique_hosts) <= 1:
            return
        if all(_is_localhost(h) for h in unique_hosts):
            return

        probe_id = uuid.uuid4().hex[:12]
        probe_dir = Path(self.run_result_output) / '.fs_probe'
        probe_dir.mkdir(parents=True, exist_ok=True)

        # Inline probe: each rank tags a sentinel with its hostname so we can
        # tell from the controller which hosts share the FS and which do not.
        # ``mkdir(parents=True, exist_ok=True)`` lets the rank succeed even
        # when --results-dir was never created on its node (it writes to its
        # own local filesystem; controller will not see those sentinels).
        inline = (
            "import os,socket,pathlib;"
            f"d=pathlib.Path({str(probe_dir)!r});"
            "d.mkdir(parents=True,exist_ok=True);"
            "r=os.environ.get('OMPI_COMM_WORLD_RANK',os.environ.get('PMI_RANK','x'));"
            "h=socket.gethostname();"
            f"(d/('{probe_id}__rank'+r+'__'+h+'.ok')).write_text(h)"
        )

        probe_hosts_arg = ",".join(f"{h}:1" for h in unique_hosts)
        mpi_bin = getattr(self.args, 'mpi_bin', 'mpirun')
        probe_prefix = (
            f"{mpi_bin} -n {len(unique_hosts)} -host {probe_hosts_arg} "
            f"--map-by node --bind-to none"
        )
        if getattr(self.args, 'allow_run_as_root', False):
            probe_prefix += " --allow-run-as-root"

        probe_cmd = f"{probe_prefix} {sys.executable} -c {shlex.quote(inline)}"

        self.logger.status(
            f"Probing --results-dir visibility across "
            f"{len(unique_hosts)} host(s)..."
        )
        self._execute_command(
            probe_cmd,
            output_file_prefix=f"kvcache_fs_probe_{self.run_datetime}",
            print_stdout=False,
            print_stderr=False,
        )

        found_hosts: set = set()
        for marker in probe_dir.glob(f"{probe_id}__rank*__*.ok"):
            try:
                found_hosts.add(marker.read_text().strip())
            except Exception:
                continue

        if len(found_hosts) >= len(unique_hosts):
            return

        raise RuntimeError(
            "kvcache --results-dir is not visible on every host listed in "
            f"--hosts. Probed {len(unique_hosts)} host(s) "
            f"({sorted(unique_hosts)}); only {len(found_hosts)} wrote a "
            f"sentinel into {self.run_result_output} "
            f"({sorted(found_hosts)}). The kvcache benchmark requires "
            "--results-dir to be on a filesystem mounted at the same path on "
            "every host in --hosts (e.g. NFS/Lustre/GPFS); otherwise rank "
            "result files written on remote nodes are invisible to the "
            "controller's aggregation step. Mount a shared filesystem and "
            "re-run, or run on a single host."
        )

    def _aggregate_option_results(
        self,
        option: int,
        trial_dirs: list,
        expected_rank_count: int,
    ) -> dict:
        """Aggregate per-rank JSON results for one option across all trials.

        Sums read/write bandwidth and token throughput across all rank files.
        Takes the mean of read/write bandwidth and token throughput across
        the trials. Takes max storage_io_latency_ms.p95 across all ranks and
        trials. Takes the max Records missing files without crashing and
        sets partial_failure. When storage_entries == 0, logs that the
        working set was served from the CPU tier.
        """
        all_read_bw = []
        all_write_bw = []
        all_avg_throughput = []
        all_storage_throughput = []
        all_p95_latency = []
        missing_files = []
        cpu_tier_flags = []
        for trial_dir in trial_dirs:
            trial_read_bw = []
            trial_write_bw = []
            trial_avg_throughput = []
            trial_storage_throughput = []
            trial_p95_latency = []
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
                    trial_read_bw.append(cache_stats.get('tier_storage_read_bandwidth_gbps', 0.0))
                    trial_write_bw.append(cache_stats.get('tier_storage_write_bandwidth_gbps', 0.0))
                    trial_avg_throughput.append(summary.get('avg_throughput_tokens_per_sec', 0.0))
                    trial_storage_throughput.append(summary.get('storage_throughput_tokens_per_sec', 0.0))
                    trial_p95_latency.append(summary.get('storage_io_latency_ms', {}).get('p95', 0.0))
                except Exception as e:
                    self.logger.warning(f"Failed to parse {result_file}: {e}")
                    missing_files.append(str(result_file))
            all_read_bw.append(sum(trial_read_bw))
            all_write_bw.append(sum(trial_write_bw))
            all_avg_throughput.append(sum(trial_avg_throughput))
            all_storage_throughput.append(sum(trial_storage_throughput))
            all_p95_latency.append(max(trial_p95_latency) if trial_p95_latency else 0.0)
        if missing_files:
            hosts = getattr(self.args, 'hosts', None) or []
            multi_host = any(not _is_localhost(h.split(':')[0]) for h in hosts)
            if multi_host:
                # Defense-in-depth — _probe_results_dir_shared should already
                # have failed the run before we get here. Surface the same
                # hint anyway in case the probe was skipped or missed an edge
                # case (e.g. partial mount on a subset of nodes).
                self.logger.warning(
                    f"Option {option}: {len(missing_files)} rank result "
                    "file(s) missing. In multi-host runs this typically "
                    "means --results-dir is not on a filesystem visible at "
                    "the same path on every host in --hosts (see issue #521)."
                )
        return {
            'option': option,
            'aggregated_read_bandwidth_gbps': fmean(all_read_bw) if all_read_bw else 0.0,
            'aggregated_write_bandwidth_gbps': fmean(all_write_bw) if all_write_bw else 0.0,
            'aggregated_avg_throughput_tokens_per_sec': fmean(all_avg_throughput) if all_avg_throughput else 0.0,
            'aggregated_storage_throughput_tokens_per_sec': fmean(all_storage_throughput) if all_storage_throughput else 0.0,
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

        # Issue #537: KVCacheRunRulesChecker and reportgen read workload config
        # from metadata['parameters']. KVCache has no DLIO combined_params, so
        # the base class fills parameters with {} and every closed run lands as
        # INVALID 'Missing model parameter'. Mirror the workload keys into
        # parameters so both live verification and reportgen see the real run
        # config instead of the run-checker's hard-coded defaults.
        base_metadata['parameters'] = {
            'model': self.model,
            'num_users': self.num_users,
            'duration': self.duration,
            'gpu_mem_gb': self.gpu_mem_gb,
            'cpu_mem_gb': self.cpu_mem_gb,
            'cache_dir': self.cache_dir,
            'generation_mode': self.generation_mode,
            'performance_profile': self.performance_profile,
        }

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
