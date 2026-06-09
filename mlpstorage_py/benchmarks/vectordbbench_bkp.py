"""VectorDB Benchmark for MLPerf Storage.

Single-node integration with the mlpstorage_py framework, plus optional
MPI-based multi-client orchestration for:

  * datagen / load phase
  * run --mode timed / query_count using simple_bench
  * run --mode sweep using enhanced_bench

The distributed path expects the vdb_benchmark package to provide:

  * vdb-mpi-wrapper
  * vdb-aggregate

as console scripts.

Important naming convention:

  * --host is the Milvus database endpoint.
  * --hosts is the list of benchmark client hosts for MPI ranks.
"""

import os
import shlex
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mlpstorage_py.benchmarks.base import Benchmark
from mlpstorage_py.config import BENCHMARK_TYPES, CONFIGS_ROOT_DIR
from mlpstorage_py.utils import generate_mpi_prefix_cmd, read_config_from_file


class VectorDBBenchmark(Benchmark):
    """VectorDB benchmark integration for mlpstorage_py."""

    VECTORDB_CONFIG_PATH = "vectordbbench"
    VDBBENCH_BIN = "vdbbench"
    BENCHMARK_TYPE = BENCHMARK_TYPES.vector_database

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self.command_method_map = {
            "datasize": self.execute_datasize,
            "datagen": self.execute_datagen,
            "run": self.execute_run,
        }

        self.command = args.command
        self.category = args.category if hasattr(args, "category") else None

        self.config_path = os.path.join(
            CONFIGS_ROOT_DIR,
            self.VECTORDB_CONFIG_PATH,
        )
        self.config_name = (
            args.config if hasattr(args, "config") and args.config else "default"
        )
        self.config_file = self._resolve_config_file(self.config_name)

        self.yaml_params = read_config_from_file(self.config_file)

        # Validate VDB-specific dependencies.
        # Keep the current behavior: skip for datasize and what-if.
        if not getattr(args, "what_if", False) and self.command != "datasize":
            self._validate_vdb_dependencies()

        self.verify_benchmark()
        self.logger.status("Instantiated the VectorDB Benchmark...")

    # -----------------------------------------------------------------
    # Dependency validation
    # -----------------------------------------------------------------

    def _validate_vdb_dependencies(self):
        """Check that local VectorDB dependencies are importable.

        Remote MPI ranks must have the same repo path and uv environment.
        """
        missing = []
        for pkg in ["pymilvus", "numpy", "tabulate"]:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)

        if missing:
            from mlpstorage_py.errors import DependencyError, ErrorCode

            raise DependencyError(
                f"Missing VDB dependencies: {', '.join(missing)}",
                suggestion=(
                    "Install with: uv sync --extra vectordb\n"
                    " or: uv pip install -e ./vdb_benchmark\n"
                    " or: pip install pymilvus numpy tabulate"
                ),
                code=ErrorCode.BENCHMARK_DEPENDENCY_MISSING,
            )

    # -----------------------------------------------------------------
    # Command dispatch
    # -----------------------------------------------------------------

    def _run(self) -> int:
        """Execute the appropriate command based on command_method_map."""
        if self.command not in self.command_method_map:
            self.logger.error(f"Unsupported command: {self.command}")
            sys.exit(1)

        self.logger.verboser(f"Executing command: {self.command}")
        result = self.command_method_map[self.command]()
        return int(result or 0)

    # -----------------------------------------------------------------
    # Config helpers
    # -----------------------------------------------------------------

    def _resolve_config_file(self, config_name_or_path: str) -> str:
        """Resolve a VectorDB config name or path to a YAML file path.

        Supported forms:
          * default
          * default.yaml
          * /absolute/path/to/config.yaml
          * relative/path/to/config.yaml
        """
        if not config_name_or_path:
            config_name_or_path = "default"

        candidates: List[str] = []

        if os.path.isabs(config_name_or_path):
            candidates.append(config_name_or_path)
        else:
            candidates.append(os.path.abspath(config_name_or_path))

            if config_name_or_path.endswith((".yaml", ".yml")):
                candidates.append(os.path.join(self.config_path, config_name_or_path))
            else:
                candidates.append(
                    os.path.join(self.config_path, f"{config_name_or_path}.yaml")
                )

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        # Preserve the prior behavior by returning the default expected config
        # path. read_config_from_file will raise a clear error if it is absent.
        if config_name_or_path.endswith((".yaml", ".yml")):
            return os.path.join(self.config_path, config_name_or_path)
        return os.path.join(self.config_path, f"{config_name_or_path}.yaml")

    def _yaml_get(self, *path: str, default: Any = None) -> Any:
        """Read a nested key from the loaded VectorDB YAML config."""
        current: Any = self.yaml_params
        for key in path:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current

    def _collection_name(self) -> str:
        """Return collection name from CLI first, then YAML."""
        collection = getattr(self.args, "collection", None)
        if collection:
            return collection

        for path in (
            ("dataset", "collection_name"),
            ("dataset", "collection"),
            ("collection_name",),
            ("collection",),
        ):
            value = self._yaml_get(*path)
            if value:
                return str(value)

        raise ValueError(
            "VectorDB collection name is required. "
            "Pass --collection or set dataset.collection_name in the config."
        )

    # -----------------------------------------------------------------
    # Shell helpers
    # -----------------------------------------------------------------

    def _get_uv_prefix(self) -> str:
        """Return the uv execution prefix used by subprocess commands."""
        return "uv run "

    @staticmethod
    def _quote(value: Any) -> str:
        return shlex.quote(str(value))

    @staticmethod
    def _option_name(param: str) -> str:
        """Normalize option names for command construction."""
        if param.startswith("--"):
            return param
        return f"--{param}"

    @classmethod
    def _append_cli_option(cls, parts: List[str], param: str, value: Any) -> None:
        """Append one CLI option to command parts.

        Rules:
          * None and False are omitted.
          * True becomes a flag.
          * lists/tuples repeat the option once per value.
          * all scalar values are shell-quoted.
        """
        if value is None or value is False:
            return

        option = cls._option_name(param)

        if value is True:
            parts.append(option)
            return

        if isinstance(value, (list, tuple)):
            for item in value:
                if item is not None:
                    parts.append(f"{option} {cls._quote(item)}")
            return

        parts.append(f"{option} {cls._quote(value)}")

    @staticmethod
    def _flatten_mpi_params(params: Optional[Iterable[Any]]) -> List[str]:
        """Flatten argparse action='append', nargs='+' MPI params."""
        if not params:
            return []

        flattened: List[str] = []
        for item in params:
            if isinstance(item, (list, tuple)):
                flattened.extend(str(x) for x in item)
            else:
                flattened.append(str(item))
        return flattened

    # -----------------------------------------------------------------
    # Base command builder
    # -----------------------------------------------------------------

    def build_command(
        self,
        script_name: str,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a single-node vdbbench command string.

        Args:
            script_name:
                Console script name, for example:
                  * load-vdb
                  * vdbbench
                  * enhanced-bench
            additional_params:
                Mapping of CLI option name without leading "--" to value.

        Returns:
            Complete shell command prefixed with "uv run".
        """
        os.makedirs(self.run_result_output, exist_ok=True)

        parts = [f"{self._get_uv_prefix()}{script_name}"]

        # All VDB scripts accept --config, --host, and --port.
        self._append_cli_option(parts, "config", self.config_file)

        if hasattr(self.args, "host") and self.args.host:
            self._append_cli_option(parts, "host", self.args.host)

        if hasattr(self.args, "port") and self.args.port:
            self._append_cli_option(parts, "port", self.args.port)

        if additional_params:
            for param, value in additional_params.items():
                self._append_cli_option(parts, param, value)

        return " ".join(parts)

    # -----------------------------------------------------------------
    # Distributed helpers
    # -----------------------------------------------------------------

    def _client_hosts(self) -> List[str]:
        """Return benchmark client hosts for MPI ranks.

        Accepts both:
          --hosts node01 node02
          --hosts=node01,node02
        """
        raw_hosts = getattr(self.args, "hosts", None)

        if not raw_hosts:
            return ["localhost"]

        hosts: List[str] = []
        for token in raw_hosts:
            for host in str(token).split(","):
                host = host.strip()
                if host:
                    hosts.append(host)

        return hosts or ["localhost"]

    def _is_distributed(self) -> bool:
        """Whether this VectorDB command should use MPI orchestration."""
        return bool(
            getattr(self.args, "distributed", False)
            or getattr(self.args, "hosts", None)
            or int(getattr(self.args, "npernode", 1) or 1) > 1
        )

    def _mpi_world_size(self) -> int:
        hosts = self._client_hosts()
        npernode = int(getattr(self.args, "npernode", 1) or 1)
        return len(hosts) * npernode

    @staticmethod
    def _strip_host_slots(host: str) -> str:
        """Strip optional slot notation from host:slots.

        For VectorDB, --npernode is the source of truth.
        """
        return host.split(":", 1)[0]

    def _mpi_prefix(self) -> str:
        """Build the MPI command prefix.

        MPICH is the default path for the VectorDB multi-node launcher.
        Open MPI reuses mlpstorage_py.utils.generate_mpi_prefix_cmd.
        """
        mpi_impl = getattr(self.args, "mpi_impl", "mpich")
        mpi_bin = getattr(self.args, "mpi_bin", None) or "mpiexec"

        hosts = self._client_hosts()
        npernode = int(getattr(self.args, "npernode", 1) or 1)
        world_size = self._mpi_world_size()
        mpi_params = self._flatten_mpi_params(getattr(self.args, "mpi_params", None))

        if mpi_impl == "mpich":
            # MPICH / Hydra style:
            #   mpiexec -n <world_size> -hosts node01,node02 -ppn <npernode>
            #
            # Site-specific Hydra args can still be supplied with --mpi-params.
            cmd = [
                mpi_bin,
                "-n",
                str(world_size),
                "-hosts",
                ",".join(self._strip_host_slots(h) for h in hosts),
            ]

            if npernode > 0:
                cmd.extend(["-ppn", str(npernode)])

            cmd.extend(mpi_params)
            return " ".join(self._quote(x) for x in cmd)

        if mpi_impl == "openmpi":
            return generate_mpi_prefix_cmd(
                mpi_cmd=mpi_bin,
                hosts=hosts,
                num_processes=world_size,
                oversubscribe=getattr(self.args, "oversubscribe", False),
                allow_run_as_root=getattr(self.args, "allow_run_as_root", False),
                params=mpi_params,
                logger=self.logger,
                mpi_btl=getattr(self.args, "mpi_btl", "auto"),
                processes_per_node=npernode,
            )

        raise ValueError(f"Unsupported VectorDB MPI implementation: {mpi_impl}")

    def _base_output_dir(self, phase: str) -> str:
        """Return the shared output directory for distributed VDB rank files."""
        return os.path.join(self.run_result_output, "vectordb", phase)

    def _run_aggregate(
        self,
        *,
        phase: str,
        base_output_dir: str,
        expected_ranks: int,
    ) -> int:
        """Run the post-MPI aggregation script."""
        cmd = (
            f"{self._get_uv_prefix()}vdb-aggregate "
            f"--phase {self._quote(phase)} "
            f"--base-output-dir {self._quote(base_output_dir)} "
            f"--expected-ranks {expected_ranks}"
        )

        self.logger.verbose(f"Aggregating distributed VectorDB {phase} results.")
        _, _, rc = self._execute_command(
            cmd,
            output_file_prefix=(
                f"{self.BENCHMARK_TYPE.value}_{self.args.command}_{phase}_aggregate"
            ),
        )
        return int(rc or 0)

    # -----------------------------------------------------------------
    # datasize
    # -----------------------------------------------------------------

    def execute_datasize(self) -> int:
        """Calculate storage requirements for the VDB dataset.

        This is pure math and does not require pymilvus.
        """
        dim = self.args.dimension
        num_vectors = self.args.num_vectors
        dtype_bytes = 4  # FLOAT_VECTOR = float32 = 4 bytes
        index_type = getattr(self.args, "index_type", "DISKANN")
        num_shards = getattr(self.args, "num_shards", 1)

        raw_bytes = num_vectors * dim * dtype_bytes

        # Approximate index overhead multipliers.
        overhead = {
            "DISKANN": 1.3,
            "HNSW": 1.5,
            "AISAQ": 0.15,
            "IVF_FLAT": 1.05,
            "IVF_SQ8": 0.40,
            "FLAT": 1.0,
        }.get(index_type, 1.3)

        total_bytes = raw_bytes * overhead * num_shards

        self.logger.result(f"Vectors: {num_vectors:,} x dim={dim} x 4B")
        self.logger.result(f"Raw data: {raw_bytes / 1e9:.2f} GB")
        self.logger.result(f"Index type: {index_type} ({overhead:.0%} overhead)")
        self.logger.result(f"Shards: {num_shards}")
        self.logger.result(f"Estimated total: {total_bytes / 1e9:.2f} GB")

        self.write_metadata()
        return 0

    # -----------------------------------------------------------------
    # datagen / load
    # -----------------------------------------------------------------

    def execute_datagen(self) -> int:
        """Execute VectorDB data generation / load."""
        if self._is_distributed():
            return self._execute_datagen_distributed()

        return self._execute_datagen_single_node()

    def _execute_datagen_single_node(self) -> int:
        """Execute existing single-node load-vdb path."""
        additional_params = {
            "collection-name": self._collection_name(),
            "dimension": self.args.dimension,
            "num-shards": self.args.num_shards,
            "vector-dtype": self.args.vector_dtype,
            "num-vectors": self.args.num_vectors,
            "distribution": self.args.distribution,
            "batch-size": self.args.batch_size,
            "chunk-size": self.args.chunk_size,
            "index-type": getattr(self.args, "index_type", None),
            "metric-type": getattr(self.args, "metric_type", None),
            "max-degree": getattr(self.args, "max_degree", None),
            "search-list-size": getattr(self.args, "search_list_size", None),
            "M": getattr(self.args, "M", None),
            "ef-construction": getattr(self.args, "ef_construction", None),
            "inline-pq": getattr(self.args, "inline_pq", None),
            "monitor-interval": getattr(self.args, "monitor_interval", None),
            "compact": getattr(self.args, "compact", False),
            "force": getattr(self.args, "force", False),
        }

        cmd = self.build_command("load-vdb", additional_params)

        self.logger.verbose("Executing single-node VectorDB data generation.")
        _, _, rc = self._execute_command(
            cmd,
            output_file_prefix=f"{self.BENCHMARK_TYPE.value}_{self.args.command}",
        )

        self.write_metadata()
        return int(rc or 0)

    def _execute_datagen_distributed(self) -> int:
        """Execute distributed VectorDB load through MPI wrapper."""
        base_output_dir = self._base_output_dir("load")
        world_size = self._mpi_world_size()
        os.makedirs(base_output_dir, exist_ok=True)

        wrapper_parts = [
            self._mpi_prefix(),
            f"{self._get_uv_prefix()}vdb-mpi-wrapper load",
        ]

        wrapper_params = {
            "base-output-dir": base_output_dir,
            "expected-ranks": world_size,
            "ready-timeout": getattr(self.args, "ready_timeout", 7200),
            "config": self.config_file,
            "host": self.args.host,
            "port": self.args.port,
            "collection-name": self._collection_name(),
            "dimension": self.args.dimension,
            "num-shards": self.args.num_shards,
            "vector-dtype": self.args.vector_dtype,
            "num-vectors": self.args.num_vectors,
            "distribution": self.args.distribution,
            "batch-size": self.args.batch_size,
            "chunk-size": self.args.chunk_size,
            "index-type": getattr(self.args, "index_type", None),
            "metric-type": getattr(self.args, "metric_type", None),
            "max-degree": getattr(self.args, "max_degree", None),
            "search-list-size": getattr(self.args, "search_list_size", None),
            "M": getattr(self.args, "M", None),
            "ef-construction": getattr(self.args, "ef_construction", None),
            "inline-pq": getattr(self.args, "inline_pq", None),
            "monitor-interval": getattr(self.args, "monitor_interval", None),
            "compact": getattr(self.args, "compact", False),
            "force": getattr(self.args, "force", False),
            "seed": getattr(self.args, "seed", 42),
        }

        for param, value in wrapper_params.items():
            self._append_cli_option(wrapper_parts, param, value)

        cmd = " ".join(wrapper_parts)

        self.logger.verbose(
            "Executing distributed VectorDB data generation "
            f"with {world_size} MPI rank(s)."
        )

        _, _, rc = self._execute_command(
            cmd,
            output_file_prefix=f"{self.BENCHMARK_TYPE.value}_{self.args.command}_mpi",
        )
        rc = int(rc or 0)

        # The wrapper's rank 0 should aggregate at the end of a successful run,
        # but run the aggregator from mlpstorage as well to make re-aggregation
        # deterministic and to handle partial outputs.
        if not getattr(self.args, "what_if", False):
            agg_rc = self._run_aggregate(
                phase="load",
                base_output_dir=base_output_dir,
                expected_ranks=world_size,
            )
            rc = rc or agg_rc

        self.write_metadata()
        return rc

    # -----------------------------------------------------------------
    # run
    # -----------------------------------------------------------------

    def execute_run(self) -> int:
        """Execute VectorDB benchmark run.

        --mode timed / query_count:
            simple_bench via vdbbench

        --mode sweep:
            enhanced_bench via enhanced-bench
        """
        if self._is_distributed():
            return self._execute_run_distributed()

        return self._execute_run_single_node()

    def _execute_run_single_node(self) -> int:
        """Execute existing single-node VectorDB run path."""
        mode = getattr(self.args, "mode", "timed")

        if mode == "sweep":
            script = "enhanced-bench"

            # Important: do NOT pass --batch-size to enhanced-bench sweep mode.
            # enhanced_bench treats any --batch-size as activation of its
            # runtime/query-count simple path.
            additional_params = {
                "collection": self._collection_name(),
                "processes": self.args.num_query_processes,
                "queries": self.args.queries,
                "k": getattr(
                    self.args,
                    "recall_k",
                    None,
                )
                or getattr(self.args, "search_limit", None),
                "seed": getattr(self.args, "seed", None),
                "out-dir": self.run_result_output,
                "sweep": True,
                "json-output": True,
            }
        else:
            script = "vdbbench"
            additional_params = {
                "collection-name": self._collection_name(),
                "processes": self.args.num_query_processes,
                "batch-size": self.args.batch_size,
                "runtime": self.args.runtime,
                "queries": self.args.queries,
                "report-count": self.args.report_count,
                "output-dir": self.run_result_output,
                "vector-dim": getattr(self.args, "vector_dim", None),
                "search-limit": getattr(self.args, "search_limit", None),
                "search-ef": getattr(self.args, "search_ef", None),
                "gt-collection": getattr(self.args, "gt_collection", None),
                "num-query-vectors": getattr(self.args, "num_query_vectors", None),
                "recall-k": getattr(self.args, "recall_k", None),
                "json-output": True,
            }

        cmd = self.build_command(script, additional_params)

        self.logger.verbose("Executing single-node VectorDB benchmark run.")
        _, _, rc = self._execute_command(
            cmd,
            output_file_prefix=f"{self.BENCHMARK_TYPE.value}_{self.args.command}",
        )

        self.write_metadata()
        return int(rc or 0)

    def _execute_run_distributed(self) -> int:
        """Execute distributed VectorDB run through MPI wrapper."""
        mode = getattr(self.args, "mode", "timed")
        phase = "enhanced" if mode == "sweep" else "simple"
        world_size = self._mpi_world_size()
        base_output_dir = self._base_output_dir(phase)
        os.makedirs(base_output_dir, exist_ok=True)

        wrapper_parts = [
            self._mpi_prefix(),
            f"{self._get_uv_prefix()}vdb-mpi-wrapper {phase}",
        ]

        wrapper_params = {
            "base-output-dir": base_output_dir,
            "expected-ranks": world_size,
            "ready-timeout": getattr(self.args, "ready_timeout", 7200),
        }

        # In query_count and sweep mode, --queries is interpreted as a global
        # query count and split by the rank-aware wrapper.
        if getattr(self.args, "queries", None):
            wrapper_params["queries"] = self.args.queries

        for param, value in wrapper_params.items():
            self._append_cli_option(wrapper_parts, param, value)

        # Separator: everything after "--" is passed by vdb-mpi-wrapper to the
        # underlying benchmark script.
        wrapper_parts.append("--")

        if phase == "simple":
            pass_through = self._simple_bench_pass_through_args()
        else:
            pass_through = self._enhanced_bench_pass_through_args()

        wrapper_parts.extend(self._quote(x) for x in pass_through)
        cmd = " ".join(wrapper_parts)

        self.logger.verbose(
            "Executing distributed VectorDB benchmark run "
            f"phase={phase}, mode={mode}, MPI ranks={world_size}."
        )

        _, _, rc = self._execute_command(
            cmd,
            output_file_prefix=f"{self.BENCHMARK_TYPE.value}_{self.args.command}_{phase}_mpi",
        )
        rc = int(rc or 0)

        if not getattr(self.args, "what_if", False):
            agg_rc = self._run_aggregate(
                phase=phase,
                base_output_dir=base_output_dir,
                expected_ranks=world_size,
            )
            rc = rc or agg_rc

        self.write_metadata()
        return rc

    def _simple_bench_pass_through_args(self) -> List[str]:
        """Arguments passed to vdbbench inside each MPI rank."""
        args: List[str] = [
            "--config",
            self.config_file,
            "--host",
            str(self.args.host),
            "--port",
            str(self.args.port),
            "--collection-name",
            self._collection_name(),
            "--processes",
            str(self.args.num_query_processes),
            "--batch-size",
            str(self.args.batch_size),
            "--report-count",
            str(self.args.report_count),
            "--vector-dim",
            str(getattr(self.args, "vector_dim", 1536)),
            "--search-limit",
            str(getattr(self.args, "search_limit", 10)),
            "--search-ef",
            str(getattr(self.args, "search_ef", 200)),
            "--num-query-vectors",
            str(getattr(self.args, "num_query_vectors", 1000)),
            "--json-output",
        ]

        if getattr(self.args, "runtime", None) is not None:
            args.extend(["--runtime", str(self.args.runtime)])

        # Do not add --queries here. The wrapper receives the global value and
        # appends each rank's local query count.
        if getattr(self.args, "gt_collection", None):
            args.extend(["--gt-collection", str(self.args.gt_collection)])

        if getattr(self.args, "recall_k", None):
            args.extend(["--recall-k", str(self.args.recall_k)])

        return args

    def _enhanced_bench_pass_through_args(self) -> List[str]:
        """Arguments passed to enhanced-bench inside each MPI rank."""
        # Important: no --batch-size here. enhanced_bench interprets
        # --batch-size as the runtime/query-count simple path.
        args: List[str] = [
            "--config",
            self.config_file,
            "--host",
            str(self.args.host),
            "--port",
            str(self.args.port),
            "--collection",
            self._collection_name(),
            "--processes",
            str(self.args.num_query_processes),
            "--out-dir",
            self._base_output_dir("enhanced_rank_output"),
            "--sweep",
            "--json-output",
        ]

        k_value = (
            getattr(self.args, "recall_k", None)
            or getattr(self.args, "search_limit", None)
            or 10
        )
        args.extend(["--k", str(k_value)])

        if getattr(self.args, "seed", None) is not None:
            args.extend(["--seed", str(self.args.seed)])

        if getattr(self.args, "gt_collection", None):
            args.extend(["--gt-collection", str(self.args.gt_collection)])

        # Do not add --queries here. The wrapper receives the global value and
        # appends each rank's local query count. If no global value was supplied,
        # enhanced-bench uses its own default.
        return args

    # -----------------------------------------------------------------
    # metadata
    # -----------------------------------------------------------------

    @property
    def metadata(self) -> Dict[str, Any]:
        """Generate metadata for the VectorDB benchmark run."""
        base_metadata = super().metadata

        base_metadata.update(
            {
                "vectordb_config": self.config_name,
                "vectordb_config_file": self.config_file,
                "model": self.config_name,
                "host": getattr(self.args, "host", "127.0.0.1"),
                "port": getattr(self.args, "port", 19530),
                "collection": getattr(self.args, "collection", None),
                "distributed": self._is_distributed()
                if self.command in ("datagen", "run")
                else False,
                "client_hosts": self._client_hosts()
                if self.command in ("datagen", "run")
                else None,
                "npernode": getattr(self.args, "npernode", None),
                "mpi_impl": getattr(self.args, "mpi_impl", None),
                "mpi_bin": getattr(self.args, "mpi_bin", None),
                "mpi_world_size": self._mpi_world_size()
                if self.command in ("datagen", "run")
                else None,
            }
        )

        if self.command == "datasize":
            base_metadata.update(
                {
                    "dimension": getattr(self.args, "dimension", None),
                    "num_vectors": getattr(self.args, "num_vectors", None),
                    "index_type": getattr(self.args, "index_type", None),
                    "num_shards": getattr(self.args, "num_shards", None),
                }
            )

        elif self.command == "datagen":
            base_metadata.update(
                {
                    "dimension": getattr(self.args, "dimension", None),
                    "num_vectors": getattr(self.args, "num_vectors", None),
                    "num_shards": getattr(self.args, "num_shards", None),
                    "vector_dtype": getattr(self.args, "vector_dtype", None),
                    "distribution": getattr(self.args, "distribution", None),
                    "batch_size": getattr(self.args, "batch_size", None),
                    "chunk_size": getattr(self.args, "chunk_size", None),
                    "index_type": getattr(self.args, "index_type", None),
                    "metric_type": getattr(self.args, "metric_type", None),
                    "seed": getattr(self.args, "seed", None),
                }
            )

        elif self.command == "run":
            base_metadata.update(
                {
                    "num_query_processes": getattr(
                        self.args,
                        "num_query_processes",
                        None,
                    ),
                    "batch_size": getattr(self.args, "batch_size", None),
                    "runtime": getattr(self.args, "runtime", None),
                    "queries": getattr(self.args, "queries", None),
                    "mode": getattr(self.args, "mode", "timed"),
                    "vector_dim": getattr(self.args, "vector_dim", None),
                    "search_limit": getattr(self.args, "search_limit", None),
                    "search_ef": getattr(self.args, "search_ef", None),
                    "recall_k": getattr(self.args, "recall_k", None),
                }
            )

        return base_metadata
