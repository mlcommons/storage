import abc
import os
import pprint
import sys

from mlpstorage.benchmarks.base import Benchmark
from mlpstorage.config import (CONFIGS_ROOT_DIR, BENCHMARK_TYPES, EXEC_TYPE, MPIRUN, MLPSTORAGE_BIN_NAME,
                               LLM_ALLOWED_VALUES, LLM_SUBSET_PROCS, EXIT_CODE, MODELS, HYDRA_OUTPUT_SUBDIR)
from mlpstorage.rules import calculate_training_data_size, HostInfo, HostMemoryInfo, HostCPUInfo, ClusterInformation
from mlpstorage.utils import (read_config_from_file, create_nested_dict, update_nested_dict, generate_mpi_prefix_cmd)


class DLIOBenchmark(Benchmark, abc.ABC):

    DLIO_CONFIG_PATH = "dlio"
    BENCHMARK_TYPE = None

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self._config_name = None
        self.base_command_path = f"dlio_benchmark"

        # This is the path that DLIO needs. The files are in this self.config_path/workload
        self.config_path = os.path.join(CONFIGS_ROOT_DIR, self.DLIO_CONFIG_PATH)

        self.per_host_mem_kB = None
        self.total_mem_kB = None

        self.cluster_information = self.accumulate_host_info(args)
        # self.cluster_information = dict()

    def accumulate_host_info(self, args):
        host_info_list = []
        per_host_mem = args.client_host_memory_in_gb
        for host in args.hosts:
            host_info = HostInfo(
                hostname=host,
                cpu=None,
                memory=HostMemoryInfo.from_total_mem_int(per_host_mem * 1024 * 1024 * 1024)
            )
            host_info_list.append(host_info)
        return ClusterInformation(host_info_list=host_info_list, logger=self.logger)

    @property
    def config_name(self):
        if self._config_name is None:
            self.logger.error("This subclass doesn't appropriately set config name. self.config_name should be set in __init__")
            raise ValueError("config_name not set")
        return self._config_name

    @config_name.setter
    def config_name(self, config_name):
        self._config_name = config_name

    def process_dlio_params(self, config_file):
        params_dict = dict() if not self.args.params else {k: v for k, v in (item.split("=") for item in self.args.params)}
        yaml_params = read_config_from_file(os.path.join(self.DLIO_CONFIG_PATH, "workload", config_file))
        combined_params = update_nested_dict(yaml_params, create_nested_dict(params_dict))

        self.logger.debug(f'yaml params: \n{pprint.pformat(yaml_params)}')
        self.logger.debug(f'combined params: \n{pprint.pformat(combined_params)}')
        self.logger.debug(f'Instance params: \n{pprint.pformat(self.__dict__)}')

        return params_dict, yaml_params, combined_params

    @abc.abstractmethod
    def _run(self):
        """
        This method needs to call execute_command method to run the benchmark
        :return:
        """
        raise NotImplementedError("Subclasses must implement this method")

    def execute_command(self):
        cmd = self.generate_dlio_command()
        self.logger.status(f'Running benchmark command:: {cmd}')
        output_file_prefix = f"{self.BENCHMARK_TYPE.value}"
        if hasattr(self.args, "command"):
            output_file_prefix += f"_{self.args.command}"

        self._execute_command(cmd, output_file_prefix=output_file_prefix)

    @abc.abstractmethod
    def add_workflow_to_cmd(self, cmd) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    def generate_dlio_command(self):
        self.logger.verboser(f'Generating DLIO command for benchmark {self.BENCHMARK_TYPE.value}')
        cmd = ""
        cmd = f"{self.base_command_path}"
        cmd += f" workload={self.config_name}"

        # Run directory for Hydra to output log files
        cmd += f" ++hydra.run.dir={self.run_result_output}"
        cmd += f" ++hydra.output_subdir={HYDRA_OUTPUT_SUBDIR}"

        cmd = self.add_workflow_to_cmd(cmd)

        if self.params_dict:
            for key, value in self.params_dict.items():
                cmd += f" ++workload.{key}={value}"

        cmd += f" --config-dir={self.config_path}"

        if self.args.exec_type == EXEC_TYPE.MPI:
            self.logger.debug(f'Generating MPI Command with binary "{self.args.mpi_bin}"')
            mpi_prefix = generate_mpi_prefix_cmd(self.args.mpi_bin, self.args.hosts, self.args.num_processes,
                                                 self.args.oversubscribe, self.args.allow_run_as_root, self.logger)
            cmd = f"{mpi_prefix} {cmd}"

        return cmd


class TrainingBenchmark(DLIOBenchmark):

    BENCHMARK_TYPE = BENCHMARK_TYPES.training

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        # This allows each command to map to a specific wrapper method. When methods are created, replace the default
        # 'self.execute_command' with the command-specific method (like "self._datasize()")
        self.command_method_map = dict(
            datasize=self.datasize,
            datagen=self.execute_command,
            run=self.execute_command,
            configview=self.execute_command,
            reportgen=self.execute_command)

        config_suffix = "datagen" if args.command == "datagen" else args.accelerator_type
        under_model = args.model.replace("-", "_")
        self.config_file = f"{under_model}_{config_suffix}.yaml"
        self.config_name = f"{under_model}_{config_suffix}"

        self.params_dict, self.yaml_params, self.combined_params = self.process_dlio_params(self.config_file)

        if self.args.command not in ("datagen", "datasize"):
            self.verify_benchmark()

        if self.args.command != "datasize":
            # The datasize command uses --data-dir and needs to generate a command that also calls --data-dir
            # The add_datadir_param would convert --data-dir to --dataset.data_folder which is invalid to
            # mlpstorage.
            self.add_datadir_param()
        self.logger.verboser(f'Instantiated the Training Benchmark...')

    def add_datadir_param(self):
        self.params_dict['dataset.data_folder'] = self.args.data_dir
        if not any([self.args.data_dir.endswith(m) for m in MODELS]):
            # Add the model to the data dir path and make sure it exists
            self.params_dict['dataset.data_folder'] = os.path.join(self.args.data_dir, self.args.model)
            if not os.path.exists(self.params_dict['dataset.data_folder']):
                self.logger.info(f'Creating data directory: {self.params_dict["dataset.data_folder"]}...')
                os.makedirs(self.params_dict['dataset.data_folder'])

        # Create the train, eval, test directories
        for folder in ["train", "valid", "test"]:
            folder_path = os.path.join(self.params_dict['dataset.data_folder'], folder)
            if not os.path.exists(folder_path):
                self.logger.info(f'Creating directory: {folder_path}...')
                os.makedirs(folder_path)

    def add_workflow_to_cmd(self, cmd) -> str:
        # Configure the workflow depending on command
        if self.args.command == "datagen":
            cmd += " ++workload.workflow.generate_data=True ++workload.workflow.train=False"
        elif self.args.command == "run_benchmark":
            cmd += " ++workload.workflow.generate_data=False ++workload.workflow.train=True"

        # Training doesn't do checkpoints
        cmd += " ++workload.workflow.checkpoint=False"
        return cmd

    def generate_datagen_benchmark_command(self, num_files_train, num_subfolders_train):
        """
        This function will generate the command to use to call this program with the training & datagen parameters.
        """
        kv_map = {
            "dataset.num_files_train": num_files_train,
            "dataset.num_subfolders_train": num_subfolders_train,
        }

        cmd = f"{MLPSTORAGE_BIN_NAME} training datagen"
        if self.args.hosts:
            cmd += f" --hosts={','.join(self.args.hosts)}"
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
            if value == 0:
                continue
            cmd += f" --param {key}={value}"

        # During datasize, this will be set to max_accelerators
        cmd += f" --num-processes={self.args.num_processes}"
        cmd += f" --results-dir={self.args.results_dir}"

        if self.args.data_dir:
            cmd += f" --data-dir={self.args.data_dir}"
        else:
            cmd += f" --data-dir=<INSERT_DATA_DIR>"

        return cmd


    def datasize(self):
        num_files_train, num_subfolders_train, total_disk_bytes = calculate_training_data_size(
            self.args, self.cluster_information, self.combined_params['dataset'], self.combined_params['reader'], self.logger
        )
        self.logger.result(f'Number of training files: {num_files_train}')
        self.logger.result(f'Number of training subfolders: {num_subfolders_train}')
        self.logger.result(f'Total disk space required for training: {total_disk_bytes / 1024**3:.2f} GB')

        if num_files_train > 10000:
            self.logger.warning(
                f'The number of files required may be excessive for some filesystems. You can use the num_subfolders_train parameter to shard the dataset. To keep near 10,000 files per folder use "{int(num_files_train / 10000)}x" subfolders by adding "--param dataset.num_subfolders_train={int(num_files_train / 10000)}"')

        cmd = self.generate_datagen_benchmark_command(num_files_train, num_subfolders_train)
        self.logger.result(f'Run the following command to generate data: \n{cmd}')
        self.logger.warning(f'The parameter for --num-processes is the same as --max-accelerators. Adjust the value '
                       f'according to your system.')

    def _run(self):
        try:
            self.command_method_map[self.args.command]()
        except Exception as e:
            self.logger.error(f'Error occurred while executing command: {str(e)}')
            return EXIT_CODE.FAILURE
        return EXIT_CODE.SUCCESS


class CheckpointingBenchmark(DLIOBenchmark):

    BENCHMARK_TYPE = BENCHMARK_TYPES.checkpointing

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self.config_name = f'{args.model.replace("-", "_")}'
        self.config_file = f'{self.config_name}.yaml'
        self.params_dict, self.yaml_params, self.combined_params = self.process_dlio_params(self.config_file)
        self.verify_benchmark()
        self.add_checkpoint_params()
        self.logger.status(f'Instantiated the Checkpointing Benchmark...')

    def add_checkpoint_params(self):
        min_procs, zero_level, GPUpDP, ClosedGPUs = LLM_ALLOWED_VALUES.get(self.args.model)
        data_parallelism = int(ClosedGPUs / GPUpDP)

        if self.args.num_processes == ClosedGPUs:
            pass
        elif self.args.num_processes < ClosedGPUs:
            self.params_dict['checkpoint.mode'] = "subset"
            self.params_dict['model.parallelism.data'] = data_parallelism
        elif self.args.num_processes > ClosedGPUs:
            self.params_dict['model.parallelism.data'] = self.args.num_processes // GPUpDP

        self.params_dict['checkpoint.num_checkpoints_read'] = self.args.num_checkpoints_read
        self.params_dict['checkpoint.num_checkpoints_write'] = self.args.num_checkpoints_write
        self.params_dict['checkpoint.checkpoint_folder'] = f"{self.args.data_dir}/{self.args.model}"


    def add_workflow_to_cmd(self, cmd) -> str:
        cmd += " ++workload.workflow.generate_data=False ++workload.workflow.train=False"
        cmd += " ++workload.workflow.checkpoint=True"
        return cmd

    def _run(self):
        try:
            self.execute_command()
        except Exception as e:
            return EXIT_CODE.FAILURE
        return EXIT_CODE.SUCCESS