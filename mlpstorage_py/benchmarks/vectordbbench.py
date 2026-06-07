"""
VectorDB Benchmark for MLPerf Storage.
 
Single-node integration with the mlpstorage_py framework.
Wraps vdbbench console scripts (load-vdb, vdbbench, enhanced-bench).
 
Post PR #308, subprocess commands are prefixed with "uv run" by default
to match the locked-dependency execution model.
"""
 
import os
import sys
from typing import Dict, Any, List
 
from mlpstorage_py.benchmarks.base import Benchmark
from mlpstorage_py.config import CONFIGS_ROOT_DIR, BENCHMARK_TYPES
from mlpstorage_py.utils import read_config_from_file
 
 
class VectorDBBenchmark(Benchmark):
 
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
        self.category = args.category if hasattr(args, 'category') else None
        self.config_path = os.path.join(CONFIGS_ROOT_DIR, self.VECTORDB_CONFIG_PATH)
        self.config_name = args.config if hasattr(args, 'config') and args.config else "default"
        self.yaml_params = read_config_from_file(
            os.path.join(self.config_path, f"{self.config_name}.yaml")
        )
 
        # Validate VDB-specific dependencies (skip for dry-run / datasize)
        if not getattr(args, 'dry_run', False) and self.command != 'datasize':
            self._validate_vdb_dependencies()
 
        self.verify_benchmark()
        self.logger.status(f'Instantiated the VectorDB Benchmark...')
 
    # -----------------------------------------------------------------
    # Dependency validation
    # -----------------------------------------------------------------
 
    def _validate_vdb_dependencies(self):
        """Check that pymilvus and other VDB dependencies are importable.
 
        Raises DependencyError with install instructions if missing.
        """
        missing = []
        for pkg in ['pymilvus', 'numpy', 'tabulate']:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
 
        if missing:
            from mlpstorage_py.errors import DependencyError, ErrorCode
            raise DependencyError(
                f"Missing VDB dependencies: {', '.join(missing)}",
                suggestion=(
                    'Install with:  uv sync --extra vectordb\n'
                    '           or: pip install pymilvus numpy tabulate'
                ),
                code=ErrorCode.BENCHMARK_DEPENDENCY_MISSING
            )
 
    # -----------------------------------------------------------------
    # Command dispatch
    # -----------------------------------------------------------------
 
    def _run(self):
        """Execute the appropriate command based on the command_method_map."""
        if self.command in self.command_method_map:
            self.logger.verboser(f"Executing command: {self.command}")
            self.command_method_map[self.command]()
            return 0
        else:
            self.logger.error(f"Unsupported command: {self.command}")
            sys.exit(1)
 
    # -----------------------------------------------------------------
    # uv run prefix
    # -----------------------------------------------------------------
 
    def _get_uv_prefix(self) -> str:
        """Return 'uv run ' to match the PR #308 execution model.
 
        The ./mlpstorage bash wrapper already invokes mlpstorage_py via
        'uv run'. Sub-process calls to vdbbench scripts should also go
        through uv to use the same locked venv.
        """
        return "uv run "
 
    # -----------------------------------------------------------------
    # Command builder
    # -----------------------------------------------------------------
 
    def build_command(self, script_name, additional_params=None):
        """Build a command string for executing a vdbbench script.
 
        Args:
            script_name: Console script name (e.g., "vdbbench", "load-vdb",
                         "enhanced-bench").
            additional_params: Dict of param_name -> value to append.
 
        Returns:
            Complete command string prefixed with 'uv run'.
        """
        os.makedirs(self.run_result_output, exist_ok=True)
 
        config_file = os.path.join(self.config_path, f"{self.config_name}.yaml")
 
        cmd = f"{self._get_uv_prefix()}{script_name}"
        cmd += f" --config {config_file}"
 
        if script_name == "load-vdb":
            if self.args.force:
                cmd += " --force"
 
        if hasattr(self.args, 'host') and self.args.host:
            cmd += f" --host {self.args.host}"
        if hasattr(self.args, 'port') and self.args.port:
            cmd += f" --port {self.args.port}"
 
        if additional_params:
            for param, attr in additional_params.items():
                if attr is not None:
                    cmd += f" --{param} {attr}"
 
        return cmd
 
    # -----------------------------------------------------------------
    # datasize
    # -----------------------------------------------------------------
 
    def execute_datasize(self):
        """Calculate storage requirements for the VDB dataset.
 
        Estimates raw vector data size plus index overhead based on
        index type. Does NOT require pymilvus (pure math).
        """
        dim = self.args.dimension
        num_vectors = self.args.num_vectors
        dtype_bytes = 4  # FLOAT_VECTOR = float32 = 4 bytes
        index_type = getattr(self.args, 'index_type', 'DISKANN')
        num_shards = getattr(self.args, 'num_shards', 1)
 
        raw_bytes = num_vectors * dim * dtype_bytes
 
        # Approximate index overhead multipliers (measured empirically)
        overhead = {
            "DISKANN": 1.3,
            "HNSW": 1.5,
            "AISAQ": 0.15,
            "IVF_FLAT": 1.05,
            "IVF_SQ8": 0.40,
            "FLAT": 1.0,
        }.get(index_type, 1.3)
 
        total_bytes = raw_bytes * overhead * num_shards
 
        self.logger.result(f"Vectors:       {num_vectors:,} x dim={dim} x 4B")
        self.logger.result(f"Raw data:      {raw_bytes / 1e9:.2f} GB")
        self.logger.result(f"Index type:    {index_type} ({overhead:.0%} overhead)")
        self.logger.result(f"Shards:        {num_shards}")
        self.logger.result(f"Estimated total: {total_bytes / 1e9:.2f} GB")
 
    # -----------------------------------------------------------------
    # datagen
    # -----------------------------------------------------------------
 
    def execute_datagen(self):
        """Execute the data generation command using load-vdb."""
        additional_params = {
            "dimension": self.args.dimension,
            "num-shards": self.args.num_shards,
            "vector-dtype": self.args.vector_dtype,
            "num-vectors": self.args.num_vectors,
            "distribution": self.args.distribution,
            "batch-size": self.args.batch_size,
            "chunk-size": self.args.chunk_size,
        }
        cmd = self.build_command("load-vdb", additional_params)
 
        self.logger.verbose('Executing data generation.')
        self._execute_command(cmd)
        self.write_metadata()
 
    # -----------------------------------------------------------------
    # run
    # -----------------------------------------------------------------
 
    def execute_run(self):
        """Execute the benchmark run command.
 
        Dispatches to simple_bench (vdbbench) or enhanced_bench based
        on --mode:
          timed / query_count  ->  vdbbench   (simple_bench.py)
          sweep                ->  enhanced-bench (enhanced_bench.py)
        """
        mode = getattr(self.args, 'benchmark_mode', 'timed')

        if mode == 'sweep':
            script = "enhanced-bench"
        else:
            script = "vdbbench"
 
        additional_params = {
            "processes": self.args.num_query_processes,
            "batch-size": self.args.batch_size,
            "runtime": self.args.runtime,
            "queries": self.args.queries,
            "report-count": self.args.report_count,
            "output-dir": self.run_result_output,
        }
 
        cmd = self.build_command(script, additional_params)
        self.logger.verbose('Executing benchmark run.')
        self._execute_command(
            cmd,
            output_file_prefix=f"{self.BENCHMARK_TYPE.value}_{self.args.command}"
        )
        self.write_metadata()
 
    # -----------------------------------------------------------------
    # metadata
    # -----------------------------------------------------------------
 
    @property
    def metadata(self) -> Dict[str, Any]:
        """Generate metadata for the VectorDB benchmark run."""
        base_metadata = super().metadata
 
        base_metadata.update({
            'vectordb_config': self.config_name,
            'model': self.config_name,
            'host': getattr(self.args, 'host', '127.0.0.1'),
            'port': getattr(self.args, 'port', 19530),
            'collection': getattr(self.args, 'collection', None),
        })
 
        if self.command == 'datasize':
            base_metadata.update({
                'dimension': getattr(self.args, 'dimension', None),
                'num_vectors': getattr(self.args, 'num_vectors', None),
                'index_type': getattr(self.args, 'index_type', None),
                'num_shards': getattr(self.args, 'num_shards', None),
            })
        elif self.command == 'datagen':
            base_metadata.update({
                'dimension': getattr(self.args, 'dimension', None),
                'num_vectors': getattr(self.args, 'num_vectors', None),
                'num_shards': getattr(self.args, 'num_shards', None),
                'vector_dtype': getattr(self.args, 'vector_dtype', None),
                'distribution': getattr(self.args, 'distribution', None),
            })
        elif self.command == 'run':
            base_metadata.update({
                'num_query_processes': getattr(self.args, 'num_query_processes', None),
                'batch_size': getattr(self.args, 'batch_size', None),
                'runtime': getattr(self.args, 'runtime', None),
                'queries': getattr(self.args, 'queries', None),
                'benchmark_mode': getattr(self.args, 'benchmark_mode', 'timed'),
            })
 
        return base_metadata
