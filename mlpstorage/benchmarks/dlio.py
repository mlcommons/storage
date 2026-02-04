import abc
import os
import os.path
import pprint
import sys

from mlpstorage.benchmarks.base import Benchmark
from mlpstorage.config import (CONFIGS_ROOT_DIR, BENCHMARK_TYPES, EXEC_TYPE, MPIRUN, MLPSTORAGE_BIN_NAME,
                               LLM_ALLOWED_VALUES, LLM_SUBSET_PROCS, EXIT_CODE, MODELS, HYDRA_OUTPUT_SUBDIR,
                               LLM_SIZE_BY_RANK)
from mlpstorage.dependency_check import validate_benchmark_dependencies
from mlpstorage.rules import calculate_training_data_size, HostInfo, HostMemoryInfo, HostCPUInfo, ClusterInformation
from mlpstorage.utils import (read_config_from_file, create_nested_dict, update_nested_dict, generate_mpi_prefix_cmd)


class DLIOBenchmark(Benchmark, abc.ABC):

    DLIO_CONFIG_PATH = "dlio"
    BENCHMARK_TYPE = None

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self._config_name = None
        self.base_command = "dlio_benchmark"
        if args.dlio_bin_path:
            self.base_path = args.dlio_bin_path
        else:
            self.base_path = os.path.dirname(sys.argv[0])
        self.base_command_path = os.path.join(self.base_path, self.base_command)

        # This is the path that DLIO needs. The files are in this self.config_path/workload
        self.config_path = os.path.join(CONFIGS_ROOT_DIR, self.DLIO_CONFIG_PATH)

        self.per_host_mem_kB = None
        self.total_mem_kB = None

        # Fail-fast dependency validation (skip for what-if mode)
        if not getattr(args, 'what_if', False):
            self._validate_dependencies(args)

        if args.command != "datagen":
            self.cluster_information = self.accumulate_host_info(args)

    def _validate_dependencies(self, args):
        """Validate required external dependencies before benchmark execution.

        Performs fail-fast checks for MPI and DLIO to provide clear error
        messages early rather than failing during benchmark execution.

        Args:
            args: Parsed command-line arguments.

        Raises:
            DependencyError: If required dependencies are not available.
        """
        requires_mpi = getattr(args, 'exec_type', None) == EXEC_TYPE.MPI
        mpi_bin = getattr(args, 'mpi_bin', 'mpirun')
        dlio_bin_path = getattr(args, 'dlio_bin_path', None)

        mpi_path, dlio_path = validate_benchmark_dependencies(
            requires_mpi=requires_mpi,
            requires_dlio=True,
            mpi_bin=mpi_bin,
            dlio_bin_path=dlio_bin_path,
            logger=self.logger
        )

        # Update base_command_path if DLIO was found in a different location
        if dlio_path:
            self.base_command_path = dlio_path

    def accumulate_host_info(self, args):
        """Collect cluster information from all hosts.

        This method first attempts to collect detailed system information via MPI.
        If MPI collection fails or is not available, it falls back to using the
        CLI argument `client_host_memory_in_gb` applied uniformly to all hosts.

        Args:
            args: Parsed command-line arguments.

        Returns:
            ClusterInformation instance with host details.
        """
        # Try MPI-based collection first
        cluster_info = self._collect_cluster_information()
        if cluster_info is not None:
            self.logger.verbose(
                f'Using MPI-collected cluster info: {cluster_info.num_hosts} hosts, '
                f'{cluster_info.total_memory_bytes / (1024**3):.1f} GB total memory'
            )
            return cluster_info

        # Fall back to CLI args-based collection
        self.logger.debug('Using CLI args for cluster info (MPI collection not available)')
        host_info_list = []
        per_host_mem = args.client_host_memory_in_gb
        for host in args.hosts:
            host_info = HostInfo(
                hostname=host,
                cpu=None,
                memory=HostMemoryInfo.from_total_mem_int(per_host_mem * 1024 * 1024 * 1024)
            )
            host_info_list.append(host_info)

        cluster_info = ClusterInformation(host_info_list=host_info_list, logger=self.logger)
        cluster_info.collection_method = "args"
        return cluster_info

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
                                                 self.args.oversubscribe, self.args.allow_run_as_root,
                                                 self.args.mpi_params, self.logger)
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
        # # Configure the workflow depending on command
        # if self.args.command == "datagen":
        #     cmd += " ++workload.workflow.generate_data=True ++workload.workflow.train=False"
        # elif self.args.command == "run_benchmark":
        #     cmd += " ++workload.workflow.generate_data=False ++workload.workflow.train=True"
        #
        # # Training doesn't do checkpoints
        # cmd += " ++workload.workflow.checkpoint=False"
        # We're now using the workflow defined in the yaml file only
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
        configured_data_parallelism = int(ClosedGPUs / GPUpDP)

        # We only need the param "model.parallelism.data" if we are not using default checkpoint_mode
        if self.args.num_processes < ClosedGPUs:
            self.params_dict['checkpoint.mode'] = "subset"
            self.params_dict['model.parallelism.data'] = configured_data_parallelism

        self.params_dict['checkpoint.num_checkpoints_read'] = self.args.num_checkpoints_read
        self.params_dict['checkpoint.num_checkpoints_write'] = self.args.num_checkpoints_write
        self.params_dict['checkpoint.checkpoint_folder'] = os.path.join(self.args.checkpoint_folder, self.args.model)


    def add_workflow_to_cmd(self, cmd) -> str:
        # cmd += " ++workload.workflow.generate_data=False ++workload.workflow.train=False"
        # cmd += " ++workload.workflow.checkpoint=True"
        # We're now using the workflow defined in the yaml file only
        return cmd

    def _run(self):
        try:
            if self.args.command == "run":
                self.execute_command()
            elif self.args.command == "datasize":
                self.datasize()
            else:
                self.logger.error(f'Invalid command: {self.args.command}')
                return EXIT_CODE.INVALID_ARGUMENTS
        except Exception as e:
            return EXIT_CODE.FAILURE
        return EXIT_CODE.SUCCESS

    def datasize(self):
        self.logger.verbose(f'Running datasize for {self.args.model}...')
        # Calculate the total writes per rank which equates to memory required per rank
        # If zero_level is 1, then rank 0 writes the entire model,
        # If zero_level is 3, then the model is sharded across all ranks
        min_procs, zero_level, GPUpDP, ClosedGPUs = LLM_ALLOWED_VALUES.get(self.args.model)
        model_gb, optimizer_gb = LLM_SIZE_BY_RANK.get(self.args.model)
        rank_gb = []

        self.logger.verbose(f'Model & optimizer size: {model_gb:.2f} GB, {optimizer_gb:.2f} GB')
        for rank in range(self.args.num_processes):
            rank_gb.append(0)
            if zero_level == 1:
                self.logger.debug("Optimizer is written by all ranks, but only the ranks on the first DP instance write the model")
                rank_gb[rank] = optimizer_gb / self.args.num_processes
                if rank < GPUpDP:
                    rank_gb[rank] += model_gb / GPUpDP
                    self.logger.debug(f'First DP: rank-{rank} write model: {rank_gb[rank]:.2f} GB')
            elif zero_level == 3:
                rank_gb[rank] = (model_gb + optimizer_gb) / self.args.num_processes
                self.logger.debug(f'Rank {rank} writes portion of model and optimizer: {rank_gb[rank]:.2f} GB')
            else:
                self.logger.error(f'Invalid zero_level: {zero_level}')
                raise ValueError("Invalid zero_level")

        rank_string = "\n\t".join(f"Rank {rank}: {rank_gb[rank]:.2f} GB" for rank in range(self.args.num_processes))

        self.logger.result(f'Total GB required per rank:\n\t{rank_string}')
        self.logger.result(f'Total GB required for all ranks: {sum(rank_gb):.2f} GB')


