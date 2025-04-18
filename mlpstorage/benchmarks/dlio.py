import os
import pprint

from mlpstorage.benchmarks.base import Benchmark
from mlpstorage.config import CONFIGS_ROOT_DIR, BENCHMARK_TYPES, EXEC_TYPE, MPIRUN
from mlpstorage.rules import calculate_training_data_size
from mlpstorage.utils import read_config_from_file, create_nested_dict, update_nested_dict, ClusterInformation, \
    generate_mpi_prefix_cmd


class TrainingBenchmark(Benchmark):

    TRAINING_CONFIG_PATH = "dlio/training"
    BENCHMARK_TYPE = BENCHMARK_TYPES.training

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        # This allows each command to map to a specific wrapper method. When methods are created, replace the default
        # 'self.execute_command' with the command-specific method (like "self._datasize()")
        self.command_method_map = dict(
            datasize=self.datasize,
            datagen=self._execute_command,
            run=self._execute_command,
            configview=self._execute_command,
            reportgen=self._execute_command,
        )

        self.per_host_mem_kB = None
        self.total_mem_kB = None

        self.params_dict = dict() if not args.params else {k: v for k, v in (item.split("=") for item in args.params)}

        self.base_command_path = f"dlio_benchmark"

        config_suffix = "datagen" if args.command == "datagen" else args.accelerator_type
        self.config_file = f"{args.model}_{config_suffix}.yaml"
        self.config_name = f"{args.model}_{config_suffix}"
        self.config_path = os.path.join(CONFIGS_ROOT_DIR, self.TRAINING_CONFIG_PATH)

        self.run_result_output = self.generate_output_location()

        self.yaml_params = read_config_from_file(os.path.join(self.TRAINING_CONFIG_PATH, self.config_file))

        self.logger.info(f'nested: {create_nested_dict(self.params_dict)}')
        self.combined_params = update_nested_dict(self.yaml_params, create_nested_dict(self.params_dict))

        self.cluster_information = ClusterInformation(hosts=self.args.hosts, username=args.ssh_username,
                                                      debug=self.args.debug)

        self.verify_benchmark()

        self.logger.debug(f'yaml params: \n{pprint.pformat(self.yaml_params)}')
        self.logger.debug(f'combined params: \n{pprint.pformat(self.combined_params)}')
        self.logger.debug(f'Instance params: \n{pprint.pformat(self.__dict__)}')
        self.logger.status(f'Instantiated the Training Benchmark...')

    def generate_output_location(self):
        return self._generate_output_location()

    def run(self):
        self.command_method_map[self.args.command]()

    def generate_command(self):
        cmd = ""

        # Set the config file to use for params not passed via CLI
        if self.args.command in ["datagen", "run"]:
            cmd = f"{self.base_command_path}"
            cmd += f" --config-dir={self.config_path}"
            cmd += f" --config-name={self.config_name}"
        else:
            raise ValueError(f"Unsupported command: {self.args.command}")

        # Run directory for Hydra to output log files
        cmd += f" ++hydra.run.dir={self.run_result_output}"

        # Set the dataset directory and checkpoint directory
        if self.args.data_dir:
            cmd += f" ++workload.dataset.data_folder={self.args.data_dir}"
            cmd += f" ++workload.checkpoint.checkpoint_folder={self.args.data_dir}"

        # Configure the workflow depending on command
        if self.args.command == "datagen":
            cmd += " ++workload.workflow.generate_data=True ++workload.workflow.train=False"
        elif self.args.command == "run_benchmark":
            cmd += " ++workload.workflow.generate_data=False ++workload.workflow.train=True"

        # Training doesn't do checkpoints
        cmd += " ++workload.workflow.checkpoint=False"

        if self.params_dict:
            for key, value in self.params_dict.items():
                cmd += f" ++workload.{key}={value}"

        if self.args.exec_type == EXEC_TYPE.MPI:
            mpi_prefix = generate_mpi_prefix_cmd(MPIRUN, self.args.hosts, self.args.num_processes,
                                                 self.args.oversubscribe, self.args.allow_run_as_root)
            cmd = f"{mpi_prefix} {cmd}"

        return cmd

    def generate_datagen_benchmark_command(self, num_files_train, num_subfolders_train, num_samples_per_file):
        """
        This function will generate the command to use to call this program with the training & datagen parameters.
        """
        kv_map = {
            "num_files_train": num_files_train,
            "num_subfolders_train": num_subfolders_train,
            "num_samples_per_file": num_samples_per_file,
        }

        cmd = f"{sys.executable} {os.path.abspath(__file__)} training datagen"
        if self.args.hosts:
            cmd += f" --hosts={self.args.hosts}"
        cmd += f" --model={self.args.model}"
        cmd += f" --exec-type={self.args.exec_type}"
        if self.args.ssh_username:
            cmd += f" --ssh-username={self.args.ssh_username}"

        if self.params_dict:
            for key, value in self.params_dict.items():
                if key in kv_map.keys():
                    continue
                cmd += f" --{key}={value}"

        for key, value in kv_map.items():
            cmd += f" --param {key}={value}"

        # During datasize, this will be set to max_accelerators
        cmd += f" --num-processes={self.args.num_processes}"
        cmd += f" --results-dir={self.args.results_dir}"
        cmd += f" --data-dir=<INSERT_DATA_DIR>"

        return cmd

    def execute_command(self):
        cmd = self.generate_command()
        self._execute_command(cmd)

    def datasize(self):
        num_files_train, num_subfolders_train, num_samples_per_file = calculate_training_data_size(
            self.args, self.cluster_information, self.combined_params['dataset'], self.combined_params['reader'], logger
        )
        self.logger.result(f'Number of training files: {num_files_train}')
        self.logger.result(f'Number of training subfolders: {num_subfolders_train}')
        self.logger.result(f'Number of training samples per file: {num_samples_per_file}')

        cmd = self.generate_datagen_benchmark_command(num_files_train, num_subfolders_train, num_samples_per_file)
        self.logger.result(f'Run the following command to generate data: \n{cmd}')
        self.logger.warning(f'The parameter for --num-processes is the same as --max-accelerators. Adjust the value '
                       f'according to your system.')
