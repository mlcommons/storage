import os
import sys
from typing import Dict, Any

from mlpstorage.benchmarks.base import Benchmark
from mlpstorage.config import CONFIGS_ROOT_DIR, BENCHMARK_TYPES
from mlpstorage.utils import read_config_from_file


class VectorDBBenchmark(Benchmark):

    VECTORDB_CONFIG_PATH = "vectordbbench"
    VDBBENCH_BIN = "vdbbench"
    BENCHMARK_TYPE = BENCHMARK_TYPES.vector_database

    def __init__(self, args):
        super().__init__(args)
        self.command_method_map = {
            "datagen": self.execute_datagen,
            "run": self.execute_run,
        }

        self.command = args.command
        self.category = args.category if hasattr(args, 'category') else None
        self.config_path = os.path.join(CONFIGS_ROOT_DIR, self.VECTORDB_CONFIG_PATH)
        self.config_name = args.config if hasattr(args, 'config') and args.config else "default"
        self.yaml_params = read_config_from_file(os.path.join(self.config_path, f"{self.config_name}.yaml"))

        self.verify_benchmark()

        self.logger.status(f'Instantiated the VectorDB Benchmark...')

    def _run(self):
        """Execute the appropriate command based on the command_method_map"""
        if self.command in self.command_method_map:
            self.logger.verboser(f"Executing command: {self.command}")
            self.command_method_map[self.command]()
        else:
            self.logger.error(f"Unsupported command: {self.command}")
            sys.exit(1)

    def build_command(self, script_name, additional_params=None):
        """
        Build a command string for executing a script with appropriate parameters

        Args:
            script_name (str): Name of the script to execute (e.g., "load_vdb.py" or "simple_bench.py")
            additional_params (dict, optional): Additional parameters to add to the command

        Returns:
            str: The complete command string
        """
        # Ensure output directory exists
        os.makedirs(self.run_result_output, exist_ok=True)

        # Build the base command
        config_file = os.path.join(self.config_path, f"{self.config_name}.yaml")

        cmd = f"{script_name}"
        cmd += f" --config {config_file}"

        if script_name == "load-vdb":
            if self.args.force:
                cmd += " --force"

        # Add host and port if provided (common to both datagen and run)
        if hasattr(self.args, 'host') and self.args.host:
            cmd += f" --host {self.args.host}"
        if hasattr(self.args, 'port') and self.args.port:
            cmd += f" --port {self.args.port}"

        # Add any additional parameters
        if additional_params:
            for param, attr in additional_params.items():
                if attr:
                    cmd += f" --{param} {attr}"

        return cmd

    def execute_datagen(self):
        """Execute the data generation command using load_vdb.py"""
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

        self.logger.verbose(f'Executing data generation.')
        self._execute_command(cmd)
        # Write metadata for history tracking
        self.write_metadata()

    def execute_run(self):
        """Execute the benchmark run command using simple_bench.py"""
        # Define additional parameters specific to the run command
        additional_params = {
            "processes": self.args.num_query_processes,
            "batch-size": self.args.batch_size,
            "runtime": self.args.runtime,
            "queries": self.args.queries,
            "report-count": self.args.report_count,
            "output-dir": self.run_result_output,
        }

        cmd = self.build_command("vdbbench", additional_params)
        self.logger.verbose(f'Execuging benchmark run.')
        self._execute_command(cmd, output_file_prefix=f"{self.BENCHMARK_TYPE.value}_{self.args.command}")
        # Write metadata for history tracking
        self.write_metadata()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Generate metadata for the VectorDB benchmark run.

        Returns:
            Dictionary containing benchmark metadata compatible with
            history module and reporting tools.
        """
        base_metadata = super().metadata

        # Use config_name as 'model' equivalent for history compatibility
        # VectorDB doesn't have ML models, but config_name serves same purpose
        base_metadata.update({
            'vectordb_config': self.config_name,
            'model': self.config_name,  # For history module compatibility
            'host': getattr(self.args, 'host', '127.0.0.1'),
            'port': getattr(self.args, 'port', 19530),
            'collection': getattr(self.args, 'collection', None),
        })

        # Add command-specific parameters
        if self.command == 'datagen':
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
            })

        return base_metadata

