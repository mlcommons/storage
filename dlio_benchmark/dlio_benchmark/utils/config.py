"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import importlib
import inspect
import hydra

import logging

from typing import Any, Dict, List, ClassVar, Union

from dlio_benchmark.common.constants import MODULE_CONFIG
from dlio_benchmark.common.enumerations import StorageType, StorageLibrary, FormatType, Shuffle, ReadType, FileAccess, Compression, \
    FrameworkType, \
    DataLoaderType, Profiler, DataLoaderSampler, CheckpointLocationType, CheckpointMechanismType, CheckpointModeType
from dlio_benchmark.utils.utility import DLIOMPI, get_trace_name, utcnow
from dlio_benchmark.utils.utility import Profile, PerfTrace, DFTRACER_ENABLE, DLIOLogger, OUTPUT_LEVEL, gen_random_tensor
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig
import math
import os
import numpy as np
from typing import Optional, Dict

dlp = Profile(MODULE_CONFIG)
@dataclass
class ConfigArguments:
    __instance = None

    # command line argument
    # Framework to use
    model: str = "default"
    framework: FrameworkType = FrameworkType.TENSORFLOW
    # Dataset format, such as PNG, JPEG
    format: FormatType = FormatType.TFRECORD
    # Shuffle type
    file_shuffle: Shuffle = Shuffle.OFF
    shuffle_size: int = 1024
    sample_shuffle: Shuffle = Shuffle.OFF
    read_type: ReadType = ReadType.ON_DEMAND
    file_access: FileAccess = FileAccess.MULTI
    # Set root as the current directory by default
    storage_root: str = "./"
    storage_type: StorageType = StorageType.LOCAL_FS
    storage_library: Optional[StorageLibrary] = None  # For S3: s3torchconnector, s3dlio, minio
    storage_options: Optional[Dict[str, str]] = None
    record_length: int = 64 * 1024
    record_length_stdev: int = 0
    record_length_resize: int = 0
    num_files_train: int = 8
    num_samples_per_file: int = 1
    batch_size: int = 1
    epochs: int = 1
    seed_change_epoch: bool = True
    generate_data: bool = False
    generate_only: bool = False
    log_level: int = OUTPUT_LEVEL
    data_folder: str = "./data/"
    output_folder: str = None
    metric_exclude_start_steps: int = 1
    metric_exclude_end_steps: int = 0
    checkpoint_folder: str = "./checkpoints/"
    log_file: str = "dlio.log"
    file_prefix: str = "img"
    keep_files: bool = True
    do_profiling: bool = False
    profiler: Profiler = Profiler.IOSTAT
    seed: int = 123
    data_gen_method: str = None  # 'dgen' (fast, zero-copy) or 'numpy' (legacy). Defaults to env DLIO_DATA_GEN or auto-detect
    do_checkpoint: bool = False
    do_train: bool = True
    checkpoint_after_epoch: int = 1
    epochs_between_checkpoints: int = 1
    steps_between_checkpoints: int = -1
    transfer_size: int = None
    read_threads: int = 1
    dont_use_mmap: bool = False
    computation_threads: int = 1
    computation_time: ClassVar[Dict[str, Any]] = {}
    preprocess_time: ClassVar[Dict[str, Any]] = {}
    prefetch_size: int = 2
    enable_chunking: bool = False
    chunk_size: int = 0
    compression: Compression = Compression.NONE
    compression_level: int = 4
    total_training_steps: int = -1
    do_eval: bool = False
    batch_size_eval: int = 1
    num_files_eval: int = 0
    generation_buffer_size: int = 2 * 1073741824  # 2 GB
    eval_time: ClassVar[Dict[str, Any]] = {}
    eval_after_epoch: int = 1
    epochs_between_evals: int = 1
    checkpoint_type: CheckpointLocationType = CheckpointLocationType.RANK_ZERO
    checkpoint_mechanism: CheckpointMechanismType = CheckpointMechanismType.NONE
    checkpoint_mode: CheckpointModeType = CheckpointModeType.DEFAULT
    model_datatype: str = "fp16"
    optimizer_datatype: str = "fp32"
    checkpoint_fsync: bool = False
    checkpoint_only: bool = False
    checkpoint_recovery_rank_shift: bool = False
    time_between_checkpoints: float = -1
    checkpoint_rank_sync: bool = False
    num_checkpoints_write: int = -1
    num_checkpoints_read: int = -1
    checkpoint_randomize_tensor: bool = True
    ksm_madv_mergeable_id: int = 12
    ksm_high_ram_trigger: float = 30.0
    ksm_low_ram_exit: float = 15
    ksm_await_time: int = 200
    ksm_present: bool = False
    model_size: int = 10240
    model_type: str = None
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    ffn_hidden_size: int = 8192
    zero_stage: int = 0
    optimization_groups: ClassVar[List[int]] = []
    num_layers: int = -1
    layer_parameters: ClassVar[List[int]] = []
    tensor_parallelism: int = 1
    pipeline_parallelism: int = 1
    data_parallelism: int = -1
    data_loader: DataLoaderType = DataLoaderType.TENSORFLOW.value
    num_subfolders_train: int = 0
    num_subfolders_eval: int = 0
    iostat_devices: ClassVar[List[str]] = []
    data_loader_classname = None
    checkpoint_mechanism_classname = None
    data_loader_sampler: DataLoaderSampler = None
    reader_classname: str = None
    multiprocessing_context: str = "fork"
    pin_memory: bool = True
    odirect: bool = False

    # derived fields
    required_samples: int = 1
    total_samples_eval: int = 1
    total_samples_train: int = 1
    file_list_eval: ClassVar[List[str]] = []
    file_list_train: ClassVar[List[str]] = []
    max_dimension: Union[int, List[int]] = 1
    storage = None
    dimension_stdev: float = 0.0
    dimension: Union[int, List[int]] = 1
    training_steps: int = 0
    eval_steps: int = 0
    samples_per_thread: int = 1
    au: float = 0.90
    file_map = None
    global_index_map = None
    data_loader_class = None
    reader_class = None
    checkpoint_mechanism_class = None
    ksm_init = False
    native_data_loader = False
    train_sample_index_sum = 1
    eval_sample_index_sum = 1

    #################################################
    # New API
    #################################################
    # dataset
    record_dims: ClassVar[List[int]] = []
    record_element_type: str = "uint8" # user provided

    # dataset -- derived
    record_element_bytes: int = 4
    record_element_dtype: ClassVar[np.dtype] = np.dtype("uint8")

    ## dataset: hdf5-only
    num_dset_per_record: int = 1
    chunk_dims: ClassVar[List[int]] = []
    max_shape: ClassVar[List[int]] = []

    ## reader
    transformed_record_dims: ClassVar[List[int]] = []
    transformed_record_element_type: str = "uint8" # user provided
    ## reader -- derived
    transformed_record_element_dtype: ClassVar[np.dtype] = np.dtype("uint8")

    # s3 defaults
    s3_region: str = "us-east-1"
    s3_force_path_style = False
    s3_max_attempts: int = 5

    def __init__(self):
        """ Virtually private constructor. """
        if ConfigArguments.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.comm_size = DLIOMPI.get_instance().size()
            self.my_rank = DLIOMPI.get_instance().rank()
            self.logger = DLIOLogger.get_instance()
            ConfigArguments.__instance = self

    def __setstate__(self, state):
        self.__dict__.update(state)
        DLIOLogger.reset()
        DLIOMPI.reset()  # in 'fork' case, clear parent's DLIOMPI
        DLIOMPI.get_instance().set_parent_values(self.my_rank, self.comm_size)
        ConfigArguments.__instance = self

    @staticmethod
    def get_instance():
        """ Static access method. """
        if ConfigArguments.__instance is None:
            ConfigArguments()
        return ConfigArguments.__instance

    def configure_dlio_logging(self, is_child=False):
        global DLIOLogger
        # with "multiprocessing_context=fork" the log file remains open in the child process
        if is_child and self.multiprocessing_context == "fork":
            return
        # Configure the logging library
        log_format_verbose = '[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]'
        log_format_simple = '[%(levelname)s] %(message)s'
        # Set logging format to be simple only when debug_level <= INFO
        log_format = log_format_simple
        if 'DLIO_LOG_LEVEL' in os.environ:
            log_level_str = os.environ["DLIO_LOG_LEVEL"]
        else:
            log_level_str = "warning"
        if log_level_str in ["info", "INFO"]:
            log_level = logging.INFO
        elif log_level_str in ["warning", "warn", "WARNING", "WARN"]:
            log_level = logging.WARNING
        elif log_level_str in ["error", "ERROR"]:
            log_level = logging.ERROR
        elif log_level_str in ["critical", "CRITICAL"]:
            log_level = logging.CRITICAL
        elif log_level_str in ["DEBUG", "debug"]:
            log_format = log_format_verbose
            log_level = logging.DEBUG
        logging.basicConfig(
            force = True,
            level=log_level,
            handlers=[
                logging.FileHandler(self.logfile_path, mode="a", encoding='utf-8'),
                logging.StreamHandler()
            ],
            format = log_format
            # logging's max timestamp resolution is msecs, we will pass in usecs in the message
        )

    def configure_dftracer(self, is_child=False, use_pid=False):
        # with "multiprocessing_context=fork" the profiler file remains open in the child process
        if is_child and self.multiprocessing_context == "fork":
            return
        # Configure the profiler
        if DFTRACER_ENABLE:
            dlp_trace = get_trace_name(self.output_folder, use_pid)
            if DLIOMPI.get_instance().rank() == 0:
                self.logger.output(f"{utcnow()} Profiling DLIO {dlp_trace}")
            return PerfTrace.initialize_log(logfile=dlp_trace,
                                                   data_dir=f"{os.path.abspath(self.data_folder)}:"
                                                            f"{self.data_folder}:./{self.data_folder}:"
                                                            f"{self.checkpoint_folder}:./{self.checkpoint_folder}:"
                                                            f"{os.path.abspath(self.checkpoint_folder)}",
                                                   process_id=self.my_rank)
        return None

    def finalize_dftracer(self, dlp_logger):
        if DFTRACER_ENABLE and dlp_logger:
            dlp_logger.finalize()

    @dlp.log
    def validate(self):
        """ validate whether the parameters are set correctly"""
        if (self.do_profiling == True) and (self.profiler == Profiler('darshan')):
            if ('LD_PRELOAD' not in os.environ or os.environ["LD_PRELOAD"].find("libdarshan") == -1):
                raise Exception("Please set darshan runtime library in LD_PRELOAD")
        if self.format is FormatType.TFRECORD and (self.data_loader is DataLoaderType.PYTORCH):
            raise Exception(f"{self.framework} support for tfrecord is not implemented for {self.data_loader}.")
        if (self.framework == FrameworkType.TENSORFLOW and self.data_loader == DataLoaderType.PYTORCH) or (
                self.framework == FrameworkType.PYTORCH and self.data_loader == DataLoaderType.TENSORFLOW):
            raise Exception("Imcompatible between framework and data_loader setup.")
        if len(self.file_list_train) != self.num_files_train:
            raise Exception(
                f"Expected {self.num_files_train} training files but {len(self.file_list_train)} found. Ensure data was generated correctly.")
        if len(self.file_list_eval) != self.num_files_eval:
            raise Exception(
                f"Expected {self.num_files_eval} evaluation files but {len(self.file_list_eval)} found. Ensure data was generated correctly.")
        if self.data_loader_classname is not None and self.data_loader_sampler is None:
            raise Exception(
                f"For custom data loaders workload.reader.data_loader_sampler needs to be defined as iter or index.")
        if self.read_threads > 1:
            import platform
            if platform.system() in ["Linux", "Windows"]:
                import psutil
                p = psutil.Process()
                cores_available = len(p.cpu_affinity())
                if cores_available < self.read_threads:
                    self.logger.warning(
                        f"Running DLIO with {self.read_threads} threads for I/O but core available {cores_available} "
                        f"are insufficient and can lead to lower performance.")
        if self.num_layers > 0 and self.num_layers < self.pipeline_parallelism:
            raise Exception(
                f"Expected model.num_layers {self.num_layers} should be larger than "
                f"model.parallelism.pipeline {self.pipeline_parallelism}.")
        if self.pipeline_parallelism > 1 and self.zero_stage == 3:
            raise Exception(f"ZeRO stage {self.zero_stage} is not compatible with pipeline parallelism.")
        if self.data_parallelism > 0 and self.checkpoint_mode == CheckpointModeType.DEFAULT:
            raise Exception(f"workload.parallelism.data should not be set in {self.checkpoint_mode} Checkpoint Mode; it will be determined internally.")
        if self.checkpoint_mode == CheckpointModeType.SUBSET:
            if self.data_parallelism <= 0:
                raise Exception("To perform subset Checkpointing, please set a target data parallelism: workload.parallelism.data.")
            elif self.data_parallelism * self.tensor_parallelism * self.pipeline_parallelism < self.comm_size:
                raise Exception(f"Comm size: {self.comm_size} is larger than 3D parallelism size: {self.data_parallelism * self.tensor_parallelism * self.pipeline_parallelism}")
        if self.checkpoint_mode == CheckpointModeType.DEFAULT:
            if self.comm_size % (self.pipeline_parallelism * self.tensor_parallelism) != 0:
                raise Exception(f"Number of processes {self.comm_size} is not a multiple of model parallelism size: {self.pipeline_parallelism * self.tensor_parallelism}")
        if self.num_checkpoints_write > 0:
            if self.num_checkpoints_read > self.num_checkpoints_write:
                raise Exception(f"Number of checkpoints to read {self.num_checkpoints_read} cannot be larger than number of checkpoints to write {self.num_checkpoints_write}")
        if self.ksm_present and self.checkpoint_randomize_tensor:
            raise Exception(f"checkpoint.ksm is {self.ksm_present} which requires checkpoint.randomize_tensor to be False")

        # HDF5 specific checks        
        if len(self.record_dims) > 0:
            if self.record_dims[0] % self.num_dset_per_record != 0:
                raise ValueError("hdf5.num_dset_per_record should be divisible by record_dims[0]")

        # Image specific checks
        if self.format in [FormatType.JPEG, FormatType.PNG]:
            if np.dtype(self.record_element_type) != np.uint8:
                # @ray: ensure compatibility with PIL fromarray (https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray)
                raise ValueError(f"{self.format} format requires record_element_type to be np.uint8, this should be automatically set. Please contact developers if this message appears.")
            if len(self.record_dims) > 2:
                raise ValueError(f"{self.format} format does not support more than 2 dimensions, but got {len(self.record_dims)} dimensions.")

        # check if both record_dims and record_length_stdev are set
        if len(self.record_dims) > 0 and self.record_length_stdev > 0:
            raise ValueError("Both record_dims and record_length_bytes_stdev are set. This is not supported. If you need stdev on your records, please specify record_length_bytes with record_length_bytes_stdev instead.")

        # S3 specific checks
        if self.storage_type == StorageType.S3 and self.framework == FrameworkType.PYTORCH:
            if self.format not in (FormatType.NPZ, FormatType.NPY):
                raise Exception(f"For S3 using PyTorch framework, only NPZ or NPY formats are supported. Got format {self.format}")

            # Also validate that s3torchconnector dependency is available
            try:
                from s3torchconnector._s3client import S3Client, S3ClientConfig
            except ImportError:
                raise Exception(
                    "The s3torchconnector package is required for S3 with PyTorch but is not installed. "
                    "Please install it before running the benchmark data generation or loading for S3."
                )

            if self.do_checkpoint == True:
                try:
                    from s3torchconnector import S3Checkpoint
                except ImportError:
                    raise Exception(
                        "The s3torchconnector package is required for S3 with PyTorch but is not installed. "
                        "Please install it before running the benchmark checkpointing for S3."
                    )
                if self.checkpoint_mechanism != CheckpointMechanismType.PT_S3_SAVE:
                    raise Exception(f"For S3 checkpointing using PyTorch framework, invalid mechanism type supported. Got mechanism type as {self.checkpoint_mechanism}")

            if self.format == FormatType.NPY:
                # Ensure the NPY S3 reader is used with s3
                try:
                    from dlio_benchmark.reader.npy_reader_s3 import NPYReaderS3
                except ImportError:
                    raise Exception(
                        "S3 with NPY requires dlio_benchmark.reader.npy_reader_s3.NPYReaderS3, "
                        "but it could not be imported. Ensure the module is available."
                    )
            elif self.format == FormatType.NPZ:
                # Ensure the NPZ S3 reader is used with s3
                try:
                    from dlio_benchmark.reader.npz_reader_s3 import NPZReaderS3
                except ImportError:
                    raise Exception(
                        "S3 with NPZ requires dlio_benchmark.reader.npz_reader_s3.NPZReaderS3, "
                        "but it could not be imported. Ensure the module is available."
                    )

            # Validate required credentials is set for s3 (from config)
            missing = []
            access_key_id = self.storage_options.get("access_key_id")
            if not access_key_id:
                missing.append("storage_options['access_key_id']")
            secret_access_key = self.storage_options.get("secret_access_key")
            if not secret_access_key:
                missing.append("storage_options['secret_access_key']")
            endpoint = self.storage_options.get("endpoint_url")
            if not endpoint:
                missing.append("storage_options['endpoint_url']")
            if missing:
                raise Exception(
                    "Missing required S3 credentials for s3torchconnector: " + ", ".join(missing)
                )


    @staticmethod
    def reset():
        ConfigArguments.__instance = None

    @dlp.log
    def derive_configurations(self, file_list_train=None, file_list_eval=None):
        # Initialize data generation method from config or environment
        if self.data_gen_method is None:
            self.data_gen_method = os.environ.get('DLIO_DATA_GEN', 'auto')
        
        # Log data generation method selection
        from dlio_benchmark.utils.utility import HAS_DGEN
        method = self.data_gen_method.lower()
        if method == 'numpy' or (method in ['auto', 'dgen'] and not HAS_DGEN):
            self.logger.output(f"{'='*80}")
            self.logger.output(f"Data Generation Method: NUMPY (Legacy)")
            self.logger.output(f"  Using NumPy random generation (155x slower than dgen-py)")
            if method == 'dgen':
                self.logger.output(f"  Note: dgen-py requested but not installed")
                self.logger.output(f"  Install with: pip install dgen-py")
            self.logger.output(f"  Set DLIO_DATA_GEN=dgen or dataset.data_gen_method=dgen for speedup")
            self.logger.output(f"{'='*80}")
        else:
            self.logger.output(f"{'='*80}")
            self.logger.output(f"Data Generation Method: DGEN (Optimized)")
            self.logger.output(f"  Using dgen-py with zero-copy BytesView (155x faster, 0MB overhead)")
            self.logger.output(f"  Set DLIO_DATA_GEN=numpy or dataset.data_gen_method=numpy for legacy mode")
            self.logger.output(f"{'='*80}")
        
        if self.checkpoint_mechanism == CheckpointMechanismType.NONE:
            if self.framework == FrameworkType.TENSORFLOW:
                self.checkpoint_mechanism = CheckpointMechanismType.TF_SAVE
            elif self.framework == FrameworkType.PYTORCH:
                if self.storage_type == StorageType.S3:
                    self.checkpoint_mechanism = CheckpointMechanismType.PT_S3_SAVE
                else:
                    self.checkpoint_mechanism = CheckpointMechanismType.PT_SAVE

        record_dims_length = len(self.record_dims)
        if record_dims_length > 0:
            self.dimension = self.record_dims
            self.dimension_stdev = self.record_length_stdev / 2.0 / self.record_length
            self.max_dimension = int(math.sqrt(self.record_length))
        else:
            self.dimension = int(math.sqrt(self.record_length))
            self.dimension_stdev = self.record_length_stdev / 2.0 / math.sqrt(self.record_length)
            self.max_dimension = self.dimension

        if self.record_length_resize > 0:
            self.max_dimension = int(math.sqrt(self.record_length_resize))

        if (file_list_train is not None and file_list_eval is not None):
            if self.transformed_record_dims is not None and len(self.transformed_record_dims) > 0:
                self.logger.output(f"Generating random tensor with shape {self.transformed_record_dims} and dtype {self.transformed_record_element_dtype}")
                rng = np.random.default_rng()
                self.resized_image = gen_random_tensor(shape=self.transformed_record_dims, dtype=self.transformed_record_element_dtype, rng=rng)
            else:
                self.resized_image = np.random.randint(255, size=(self.max_dimension, self.max_dimension), dtype=np.uint8)
            self.file_list_train = file_list_train
            self.file_list_eval = file_list_eval
            self.num_files_eval = len(file_list_eval)
            self.num_files_train = len(file_list_train)
            self.total_samples_train = self.num_samples_per_file * len(self.file_list_train)
            self.total_samples_eval = self.num_samples_per_file * len(self.file_list_eval)
            self.train_sample_index_sum = self.total_samples_train * (self.total_samples_train - 1) // 2
            self.eval_sample_index_sum = self.total_samples_eval * (self.total_samples_eval - 1) // 2
            self.required_samples = self.comm_size * self.batch_size
            if self.read_threads > 0:
                self.required_samples *= self.read_threads
            self.training_steps = int(math.ceil(self.total_samples_train / self.batch_size / self.comm_size))
            self.eval_steps = int(math.ceil(self.total_samples_eval / self.batch_size_eval / self.comm_size))
        if self.data_loader_sampler is None and self.data_loader_classname is None:
            if self.data_loader == DataLoaderType.TENSORFLOW:
                self.data_loader_sampler = DataLoaderSampler.ITERATIVE
            elif self.data_loader in [DataLoaderType.PYTORCH, DataLoaderType.DALI]:
                self.data_loader_sampler = DataLoaderSampler.INDEX
        if self.data_loader_classname is not None:
            from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
            classname = self.data_loader_classname.split(".")[-1]
            module = importlib.import_module(".".join(self.data_loader_classname.split(".")[:-1]))
            for class_name, obj in inspect.getmembers(module):
                if class_name == classname and issubclass(obj, BaseDataLoader):
                    if DLIOMPI.get_instance().rank() == 0:
                        self.logger.info(f"Discovered custom data loader {class_name}")
                    self.data_loader_class = obj
                    break
        if self.checkpoint_mechanism_classname is not None:
            from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
            classname = self.checkpoint_mechanism_classname.split(".")[-1]
            module = importlib.import_module(".".join(self.checkpoint_mechanism_classname.split(".")[:-1]))
            for class_name, obj in inspect.getmembers(module):
                if class_name == classname and issubclass(obj, BaseCheckpointing):
                    if DLIOMPI.get_instance().rank() == 0:
                        self.logger.info(f"Discovered custom checkpointing mechanism {class_name}")
                    self.checkpoint_mechanism_class = obj
                    break
        if self.reader_classname is not None:
            from dlio_benchmark.reader.reader_handler import FormatReader
            classname = self.reader_classname.split(".")[-1]
            module = importlib.import_module(".".join(self.reader_classname.split(".")[:-1]))
            for class_name, obj in inspect.getmembers(module):
                if class_name == classname and issubclass(obj, FormatReader):
                    if DLIOMPI.get_instance().rank() == 0:
                        self.logger.info(f"Discovered custom data reader {class_name}")
                    self.reader_class = obj
                    break
        self.train_file_map = {self.my_rank : {}}
        self.val_file_map = {self.my_rank : {}}
        self.train_global_index_map = {}
        self.val_global_index_map = {}
        self.native_data_loader = False
        self.ksm_init = self.ksm_present
        if self.data_loader == DataLoaderType.TENSORFLOW:
            if self.format == FormatType.TFRECORD:
                self.native_data_loader = True
        elif self.data_loader == DataLoaderType.NATIVE_DALI:
            if self.format in [FormatType.JPEG, FormatType.PNG, FormatType.NPY, FormatType.TFRECORD]:
                self.native_data_loader = True


        # dimension-based derivations

        if self.format in [FormatType.JPEG, FormatType.PNG]:
            if self.record_element_type != "uint8":
                # @ray: ensure compatibility with PIL fromarray (https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray)        
                # force uint8 on image dataset
                self.logger.warning(f"Image format {self.format} requires record_element_type to be np.uint8, but given {self.record_element_type}. Re-setting to np.uint8.")
                self.record_element_type = "uint8"

        # recalculate record_element_bytes if record_element_type is provided
        # to make them consistent
        self.record_element_dtype = np.dtype(self.record_element_type)
        self.record_element_bytes = self.record_element_dtype.itemsize

        # hdf5 specific derivations
        self.record_length = np.prod(self.record_dims) * self.record_element_bytes

        self.transformed_record_element_dtype = np.dtype(self.transformed_record_element_type)

    @dlp.log
    def build_sample_map_iter(self, file_list, total_samples, epoch_number):
        self.logger.debug(f"ranks {self.comm_size} threads {self.read_threads} tensors")
        
        num_files = len(file_list)
        samples_sum = 0
        process_thread_file_map = {}
        if num_files > 0:
            num_threads = 1
            if self.read_threads > 0 and self.data_loader is not DataLoaderType.DALI:
                num_threads = self.read_threads
            samples_per_proc = int(math.ceil(total_samples/self.comm_size)) 
            self.samples_per_thread = samples_per_proc // num_threads
            start_sample_index = samples_per_proc * self.my_rank
            end_sample_index = samples_per_proc * (self.my_rank + 1) - 1
            if end_sample_index > total_samples - 1:
                end_sample_index = total_samples - 1
            sample_list = np.arange(start_sample_index, end_sample_index + 1)
            self.logger.debug(f"{self.my_rank} {start_sample_index} {end_sample_index}")
            if self.sample_shuffle is not Shuffle.OFF:
                if self.seed_change_epoch:
                    np.random.seed(self.seed + epoch_number)
                else:
                    np.random.seed(self.seed)
                np.random.shuffle(sample_list)
            sample_index = 0
            if num_files > 0:
                files_per_rank = (num_files // self.comm_size) % num_files
                file_index = self.my_rank * files_per_rank
                for thread_index in range(num_threads):
                    process_thread_file_map[thread_index] = []
                for sample in sample_list:
                    samples_sum += sample
                    thread_index = (sample_index // self.samples_per_thread) % num_threads
                    abs_path = os.path.abspath(file_list[file_index])
                    process_thread_file_map[thread_index].append((sample,
                                                abs_path,
                                                sample_list[sample_index] % self.num_samples_per_file))
                    sample_index += 1
                    file_index = (sample_index // self.num_samples_per_file) % num_files
        return process_thread_file_map, samples_sum

    @dlp.log
    def get_global_map_index(self, file_list, total_samples, epoch_number):
        process_thread_file_map = {}
        num_files = len(file_list)
        start_sample = 0
        end_sample = 0
        samples_sum = 0
        if num_files > 0:
            end_sample = total_samples - 1
            samples_per_proc = int(math.ceil(total_samples/self.comm_size)) 
            start_sample = self.my_rank * samples_per_proc
            end_sample = (self.my_rank + 1) * samples_per_proc - 1
            if end_sample > total_samples - 1:
                end_sample = total_samples - 1
            self.logger.debug(f"my_rank: {self.my_rank}, start_sample: {start_sample}, end_sample: {end_sample}")
            sample_list = np.arange(start_sample, end_sample + 1)
            if self.sample_shuffle is not Shuffle.OFF:
                if self.seed_change_epoch:
                    np.random.seed(self.seed + epoch_number)
                else:
                    np.random.seed(self.seed)
                np.random.shuffle(sample_list)
            for sample_index in range(end_sample - start_sample + 1):
                global_sample_index = sample_list[sample_index]
                samples_sum += global_sample_index
                file_index = int(math.floor(global_sample_index/self.num_samples_per_file))
                if self.storage_type == StorageType.LOCAL_FS:
                    abs_path = os.path.abspath(file_list[file_index])
                else:
                    abs_path = file_list[file_index]
                sample_index = global_sample_index % self.num_samples_per_file
                process_thread_file_map[global_sample_index] = (abs_path, sample_index)
        return process_thread_file_map, samples_sum

    @dlp.log
    def reconfigure(self, epoch_number):
        if self.data_loader_sampler == DataLoaderSampler.ITERATIVE:
            if self.file_shuffle is not Shuffle.OFF:
                if self.seed_change_epoch:
                    np.random.seed(self.seed + epoch_number)
                else:
                    np.random.seed(self.seed)
                np.random.shuffle(self.file_list_train) 
                np.random.shuffle(self.file_list_eval)
        if self.data_loader_sampler == DataLoaderSampler.ITERATIVE:
            self.train_file_map, local_train_sample_sum = self.build_sample_map_iter(self.file_list_train, self.total_samples_train,
                                                             epoch_number)
            self.val_file_map, local_eval_sample_sum = self.build_sample_map_iter(self.file_list_eval, self.total_samples_eval, epoch_number)
        elif self.data_loader_sampler == DataLoaderSampler.INDEX:
            self.train_global_index_map, local_train_sample_sum = self.get_global_map_index(self.file_list_train, self.total_samples_train,
                                                             epoch_number)
            self.val_global_index_map, local_eval_sample_sum = self.get_global_map_index(self.file_list_eval, self.total_samples_eval,
                                                             epoch_number)
        global_train_sample_sum = DLIOMPI.get_instance().reduce(local_train_sample_sum)
        global_eval_sample_sum = DLIOMPI.get_instance().reduce(local_eval_sample_sum)        
        if self.my_rank == 0:
            self.logger.info(f"{utcnow()} Total number of samples: train {global_train_sample_sum}, eval {global_eval_sample_sum}")
            if self.train_sample_index_sum != global_train_sample_sum:
                raise Exception(f"Sharding of train samples are missing samples got {global_train_sample_sum} but expected {self.train_sample_index_sum}")
            
            if self.eval_sample_index_sum != global_eval_sample_sum:
                raise Exception(f"Sharding of eval samples are missing samples got {global_eval_sample_sum} but expected {self.eval_sample_index_sum}")

def GetConfig(args, key):
    keys = key.split(".")
    value = None
    if len(keys) > 0 and keys[0] == "framework":
        value = args.framework
    
    if len(keys) > 1 and keys[0] == "storage":
        if keys[1] == "storage_type":
            value = args.storage_type
        elif keys[1] == "storage_root":
            value = args.storage_root
        elif keys[1] == "storage_options" and len(keys) > 2:
            if args.storage_type == "s3":
                option_key = keys[2]
                if option_key in ["access_key_id", "secret_access_key", "endpoint_url", "region", "s3_force_path_style", "s3_max_attempts"]:
                    value = config["storage"].get("storage_options", {}).get(option_key)
    
    if len(keys) > 1 and keys[0] == "dataset":
        if keys[1] == "record_length_bytes":
            value = args.record_length
        elif keys[1] == "record_length_bytes_stdev":
            value = args.record_length_stdev
        elif keys[1] == "record_length_bytes_resize":
            value = args.record_length_resize
        elif keys[1] == "num_files_train":
            value = args.num_files_train
        elif keys[1] == "num_files_eval":
            value = args.num_files_eval
        elif keys[1] == "generation_buffer_size":
            value = args.generation_buffer_size
        elif keys[1] == "num_samples_per_file":
            value = args.num_samples_per_file
        elif keys[1] == "data_folder":
            value = args.data_folder
        elif keys[1] == "num_subfolders_train":
            value = args.num_subfolders_train
        elif keys[1] == "num_subfolders_eval":
            value = args.num_subfolders_eval
        elif keys[1] == "enable_chunking":
            value = args.enable_chunking
        elif keys[1] == "chunk_size":
            value = args.chunk_size
        elif keys[1] == "compression":
            value = args.compression
        elif keys[1] == "compression_level":
            value = args.compression_level
        elif keys[1] == "file_prefix":
            value = args.file_prefix
        elif keys[1] == "format":
            value = args.format
        elif keys[1] == "keep_files":
            value = args.keep_files

    # data reader
    reader = None
    if len(keys) > 1 and (keys[0] == "data_reader" or keys[0] == "reader"):
        if keys[1] == "dont_use_mmap":
            value = args.dont_use_mmap
        elif keys[1] == "reader_classname":
            value = args.reader_classname
        elif keys[1] == "multiprocessing_context":
            value = args.multiprocessing_context
        elif keys[1] == "data_loader":
            value = args.data_loader
        elif keys[1] == "data_loader_classname":
            value = args.data_loader_classname
        elif keys[1] == "data_loader_sampler":
            value = args.data_loader_sampler
        elif keys[1] == "read_threads":
            value = args.read_threads
        elif keys[1] == "computation_threads":
            value = args.computation_threads
        elif keys[1] == "batch_size":
            value = args.batch_size
        elif keys[1] == "batch_size_eval":
            value = args.batch_size_eval
        elif keys[1] == "prefetch_size":
            value = args.prefetch_size
        elif keys[1] == "file_shuffle":
            value = args.file_shuffle
        elif keys[1] == "file_access":
            value = args.file_access
        elif keys[1] == "shuffle_size":
            value = args.shuffle_size
        elif keys[1] == "sample_shuffle":
            value = args.sample_shuffle
        elif keys[1] == "read_type":
            value = args.read_type
        elif keys[1] == "transfer_size":
            value = args.transfer_size
        elif keys[1] == "preprocess_time":
            value = args.preprocess_time.get("mean", 0)
        elif keys[1] == "preprocess_time_stdev":
            value = args.preprocess_time.get("stdev", None)
        elif keys[1] == "pin_memory":
            value = args.pin_memory

    # training relevant setting
    if len(keys) > 1 and keys[0] == "train":
        if keys[1] == "epochs":
            value = args.epochs
        elif keys[1] == "total_training_steps":
            value = args.total_training_steps
        elif keys[1] == "seed_change_epoch":
            value = args.seed_change_epoch
        elif keys[1] == "computation_time":
            value = args.computation_time.get("mean", 0)
        elif keys[1] == "computation_time_stdev":
            value = args.computation_time.get("stdev", None)
        elif keys[1] == "seed":
            value = args.seed

    if len(keys) > 1 and keys[0] == "evaluation":
        if keys[1] == "eval_time":
            value = args.eval_time.get("mean", 0)
        elif keys[1] == "eval_time_stdev":
            value = args.eval_time.get("stdev", None)
        elif keys[1] == "eval_after_epoch":
            value = args.eval_after_epoch
        elif keys[1] == "epochs_between_evals":
            value = args.epochs_between_evals

    if len(keys) > 1 and keys[0] == "checkpoint":
        if keys[1] == "checkpoint_folder":
            value = args.checkpoint_folder
        elif keys[1] == "checkpoint_after_epoch":
            value = args.checkpoint_after_epoch
        elif keys[1] == "epochs_between_checkpoints":
            value = args.epochs_between_checkpoints
        elif keys[1] == "steps_between_checkpoints":
            value = args.steps_between_checkpoints
        elif keys[1] == "type":
            value = args.checkpoint_type
        elif keys[1] == 'mode':
            value = args.checkpoint_mode
        elif keys[1] == "checkpoint_mechanism_classname":
            value = args.checkpoint_mechanism_classname
        elif keys[1] == "fsync":
            value = args.checkpoint_fsync
        elif keys[1] == "time_between_checkpoints":
            value = args.time_between_checkpoints
        elif keys[1] == "num_checkpoints_write":
            value = args.num_checkpoints_write
        elif keys[1] == "num_checkpoints_read":
            value = args.num_checkpoints_read
        elif keys[1] == "checkpoint_rank_sync":
            value = args.checkpoint_rank_sync
        elif keys[1] == "recovery_rank_shift":  
            value = args.checkpoint_recovery_rank_shift

    if len(keys) > 1 and keys[0] == "model":
        if keys[1] == "name":
            value = args.model
        elif keys[1] == "type":
            value = args.model_type
        elif keys[1] == "model_size_bytes":
            value = args.model_size
        elif keys[1] == "optimization_groups":
            value = args.optimization_groups
        elif keys[1] == "num_layers":
            value = args.num_layers
        elif keys[1] == "layer_parameters":
            value = args.layer_parameters
        elif keys[1] == "model_datatype":
            value = args.model_datatype
        elif keys[1] == "optimizer_datatype":
            value = args.optimizer_datatype

        if len(keys) > 2 and keys[1] == "parallelism":
            if keys[2] == "tensor":
                value = args.tensor_parallelism
            elif keys[2] == "pipeline":
                value = args.pipeline_parallelism
            elif keys[2] == "data":
                value = args.data_parallelism
            elif keys[2] == "zero_stage":
                value = args.zero_stage

        if len(keys) > 2 and keys[1] == "transformer":
            if keys[2] == "vocab_size":
                value = args.vocab_size
            elif keys[2] == "hidden_size":
                value = args.hidden_size
            elif keys[2] == "ffn_hidden_size":
                value = args.ffn_hidden_size
            elif keys[2] == "num_attention_heads":
                value = args.num_attention_heads
            elif keys[2] == "num_kv_heads":
                value = args.num_kv_heads
            
    if len(keys) > 1 and keys[0] == "output":
        if keys[1] == "folder":
            value = args.output_folder
        elif keys[1] == "log_file":
            value = args.log_file
        elif keys[1] == "metric":
            if len(keys) > 2 and keys[2] == "exclude_start_steps":
                value = args.metric_exclude_start_steps
            elif len(keys) > 2 and keys[2] == "exclude_end_steps":
                value = args.metric_exclude_end_steps

    if len(keys) > 1 and keys[0] == "workflow":
        if keys[1] == "train":
            value = args.do_train
        elif keys[1] == "generate_data":
            value = args.generate_data
        elif keys[1] == "evaluation":
            value = args.do_eval
        elif keys[1] == "checkpoint":
            value = args.do_checkpoint
        elif keys[1] == "profiling":
            value = args.do_profiling

    if len(keys) > 0 and keys[0] == "profiling":
        if len(keys) > 1 and keys[1] == "profiler":
            value = args.profiler
        elif len(keys) > 1 and keys[1] == "iostat_devices":
            value = args.iostat_devices

    if len(keys) > 0 and keys[0] == "metric":
        if len(keys) > 1 and keys[1] == "au":
            value = args.au
    return str(value) if value is not None else None

def LoadConfig(args, config):
    '''
    Override the args by a system config (typically loaded from a YAML file)
    '''
    if 'framework' in config:
        args.framework = FrameworkType(config['framework'])

    if 'storage' in config:
        if 'storage_type' in config['storage']:
            args.storage_type = StorageType(config['storage']['storage_type'])
        if 'storage_library' in config['storage']:
            args.storage_library = StorageLibrary(config['storage']['storage_library'])
        if 'storage_root' in config['storage']:
            args.storage_root = config['storage']['storage_root']
        if 'storage_options' in config['storage']:
            args.storage_options = config['storage']['storage_options']

    # dataset related settings
    if 'dataset' in config:
        if 'record_length_bytes' in config['dataset']:
            args.record_length = config['dataset']['record_length_bytes']
        if 'record_length_bytes_stdev' in config['dataset']:
            args.record_length_stdev = config['dataset']['record_length_bytes_stdev']
        if 'record_length_bytes_resize' in config['dataset']:
            args.record_length_resize = config['dataset']['record_length_bytes_resize']
        if 'num_files_train' in config['dataset']:
            args.num_files_train = config['dataset']['num_files_train']
        if 'num_files_eval' in config['dataset']:
            args.num_files_eval = config['dataset']['num_files_eval']
        if 'generation_buffer_size' in config['dataset']:
            args.generation_buffer_size = config['dataset']['generation_buffer_size']
        if 'num_samples_per_file' in config['dataset']:
            args.num_samples_per_file = config['dataset']['num_samples_per_file']
        if 'data_folder' in config['dataset']:
            args.data_folder = config['dataset']['data_folder']
            args.data_folder = args.data_folder.rstrip('/')
        if 'num_subfolders_train' in config['dataset']:
            args.num_subfolders_train = config['dataset']['num_subfolders_train']
        if 'num_subfolders_eval' in config['dataset']:
            args.num_subfolders_eval = config['dataset']['num_subfolders_eval']
        if 'enable_chunking' in config['dataset']:
            args.enable_chunking = config['dataset']['enable_chunking']
        if 'chunk_size' in config['dataset']:
            args.chunk_size = config['dataset']['chunk_size']
        if 'compression' in config['dataset']:
            args.compression = config['dataset']['compression']
        if 'compression_level' in config['dataset']:
            args.compression_level = config['dataset']['compression_level']
        if 'file_prefix' in config['dataset']:
            args.file_prefix = config['dataset']['file_prefix']
        if 'format' in config['dataset']:
            args.format = FormatType(config['dataset']['format'])
        if 'data_gen_method' in config['dataset']:
            args.data_gen_method = config['dataset']['data_gen_method']
        if 'keep_files' in config['dataset']:
            args.keep_files = config['dataset']['keep_files']
        if 'record_element_bytes' in config['dataset']:
            args.record_element_bytes = config['dataset']['record_element_bytes']
        if 'record_element_type' in config['dataset']:
            args.record_element_type = config['dataset']['record_element_type']
        if 'record_dims' in config['dataset']:
            args.record_dims = list(config['dataset']['record_dims'])

        # hdf5 only config
        if 'hdf5' in config['dataset']:
            if 'chunk_dims' in config['dataset']['hdf5']:
                args.chunk_dims = tuple(config['dataset']['hdf5']['chunk_dims'])
            if 'num_dset_per_record' in config['dataset']['hdf5']:
                args.num_dset_per_record = config['dataset']['hdf5']['num_dset_per_record']
            if 'max_shape' in config['dataset']['hdf5']:
                args.max_shape = list(config['dataset']['hdf5']['max_shape'])

    # data reader
    reader = None
    if 'data_reader' in config:
        reader = config['data_reader']
    elif 'reader' in config:
        reader = config['reader']
    if reader is not None:
        if 'dont_use_mmap' in reader:
            args.dont_use_mmap = reader['dont_use_mmap']
        if 'reader_classname' in reader:
            args.reader_classname = reader['reader_classname']
        if 'multiprocessing_context' in reader:
            args.multiprocessing_context = reader['multiprocessing_context']
        if 'data_loader' in reader:
            args.data_loader = DataLoaderType(reader['data_loader'])
        if 'data_loader_classname' in reader:
            args.data_loader_classname = reader['data_loader_classname']
        if 'data_loader_sampler' in reader:
            args.data_loader_sampler = DataLoaderSampler(reader['data_loader_sampler'])
        if 'read_threads' in reader:
            args.read_threads = reader['read_threads']
        if 'computation_threads' in reader:
            args.computation_threads = reader['computation_threads']
        if 'batch_size' in reader:
            args.batch_size = reader['batch_size']
        if 'batch_size_eval' in reader:
            args.batch_size_eval = reader['batch_size_eval']
        if 'prefetch_size' in reader:
            args.prefetch_size = reader['prefetch_size']
        if 'file_shuffle' in reader:
            args.file_shuffle = reader['file_shuffle']
        if 'file_access' in reader:
            args.file_access = FileAccess(reader['file_access'])
        if 'shuffle_size' in reader:
            args.shuffle_size = reader['shuffle_size']
        if 'sample_shuffle' in reader:
            args.sample_shuffle = Shuffle(reader['sample_shuffle'])
        if 'read_type' in reader:
            args.read_type = reader['read_type']
        if 'transfer_size' in reader:
            args.transfer_size = reader['transfer_size']
        if 'odirect' in reader:
            args.odirect = reader['odirect']

        args.preprocess_time = {}
        if 'preprocess_time' in reader:
            preprocess_time = {}
            if isinstance(reader['preprocess_time'], dict):
                preprocess_time = reader['preprocess_time']
            elif isinstance(reader['preprocess_time'], (int, float)):
                preprocess_time["mean"] = reader['preprocess_time']
            elif isinstance(reader['preprocess_time'], DictConfig):
                preprocess_time = OmegaConf.to_container(reader['preprocess_time'])
            else:
                args.preprocess_time = reader['preprocess_time']
            args.preprocess_time = preprocess_time if preprocess_time is not None else {}
        if 'preprocess_time_stdev' in reader:
            args.preprocess_time["stdev"] = reader['preprocess_time_stdev']
        if 'pin_memory' in reader:
            args.pin_memory = reader['pin_memory']
        if 'transformed_record_dims' in reader:
            args.transformed_record_dims = list(reader['transformed_record_dims'])
        if 'transformed_record_element_type' in reader:
            args.transformed_record_element_type = reader['transformed_record_element_type']
        
        # Storage configuration (multi-protocol architecture)
        if 'storage_type' in reader:
            args.storage_type = StorageType(reader['storage_type'])
        if 'protocol' in reader:
            args.protocol = reader['protocol']
        if 'storage_library' in reader:
            args.storage_library = reader['storage_library']
        if 'storage_root' in reader:
            args.storage_root = reader['storage_root']
        if 'storage_options' in reader:
            args.storage_options = reader['storage_options']

    # training relevant setting
    if 'train' in config:
        if 'epochs' in config['train']:
            args.epochs = config['train']['epochs']
        if 'total_training_steps' in config['train']:
            args.total_training_steps = config['train']['total_training_steps']
        if 'seed_change_epoch' in config['train']:
            args.seed_change_epoch = config['train']['seed_change_epoch']
        args.computation_time = {}
        if 'computation_time' in config['train']:
            computation_time = {}
            if isinstance(config['train']['computation_time'], dict):
                computation_time = config['train']['computation_time']
            elif isinstance(config['train']['computation_time'], (int, float)):
                computation_time["mean"] = config['train']['computation_time']
            elif isinstance(config['train']['computation_time'], DictConfig):
                computation_time = OmegaConf.to_container(config['train']['computation_time'])
            else:
                args.computation_time = config['train']['computation_time']
            args.computation_time = computation_time if computation_time is not None else {}
        if 'computation_time_stdev' in config['train']:
            args.computation_time["stdev"] = config['train']['computation_time_stdev']
        if 'seed' in config['train']:
            args.seed = config['train']['seed']

    if 'evaluation' in config:
        args.eval_time = {}
        if 'eval_time' in config['evaluation']:
            eval_time = {}
            if isinstance(config['evaluation']['eval_time'], dict):
                eval_time = config['evaluation']['eval_time']
            elif isinstance(config['evaluation']['eval_time'], (int, float)):
                eval_time["mean"] = config['evaluation']['eval_time']
            elif isinstance(config['evaluation']['eval_time'], DictConfig):
                eval_time = OmegaConf.to_container(config['evaluation']['eval_time'])
            else:
                args.eval_time = config['evaluation']['eval_time']
            args.eval_time = eval_time if eval_time is not None else {}
                
        if 'eval_time_stdev' in config['evaluation']:
            args.eval_time["stdev"] = config['evaluation']['eval_time_stdev']
        if 'eval_after_epoch' in config['evaluation']:
            args.eval_after_epoch = config['evaluation']['eval_after_epoch']
        if 'epochs_between_evals' in config['evaluation']:
            args.epochs_between_evals = config['evaluation']['epochs_between_evals']

    if 'checkpoint' in config:
        if 'checkpoint_folder' in config['checkpoint']:
            args.checkpoint_folder = config['checkpoint']['checkpoint_folder']
            args.checkpoint_folder = args.checkpoint_folder.rstrip('/')
        if 'checkpoint_after_epoch' in config['checkpoint']:
            args.checkpoint_after_epoch = config['checkpoint']['checkpoint_after_epoch']
        if 'epochs_between_checkpoints' in config['checkpoint']:
            args.epochs_between_checkpoints = config['checkpoint']['epochs_between_checkpoints']
        if 'steps_between_checkpoints' in config['checkpoint']:
            args.steps_between_checkpoints = config['checkpoint']['steps_between_checkpoints']
        if 'type' in config['checkpoint']:
            args.checkpoint_type = CheckpointLocationType(config['checkpoint']['type'])
        if 'checkpoint_mechanism_classname' in config['checkpoint']:
            args.checkpoint_mechanism_classname = config['checkpoint']['checkpoint_mechanism_classname']
        if 'fsync' in config['checkpoint']:
            args.checkpoint_sync = config['checkpoint']['fsync']
        if 'time_between_checkpoints' in config['checkpoint']:
            args.time_between_checkpoints = config['checkpoint']['time_between_checkpoints']
        if 'num_checkpoints_write' in config['checkpoint']:
            args.num_checkpoints_write = config['checkpoint']['num_checkpoints_write']
        if 'num_checkpoints_read' in config['checkpoint']:
            args.num_checkpoints_read = config['checkpoint']['num_checkpoints_read']
        if 'recovery_rank_shift' in config['checkpoint']:
            args.checkpoint_recover_rank_shift = config['checkpoint']['recovery_rank_shift']
        if 'rank_sync' in config['checkpoint']:
            args.checkpoint_rank_sync = config['checkpoint']['rank_sync']
        if 'mode' in config['checkpoint']:
            args.checkpoint_mode = CheckpointModeType(config['checkpoint']['mode'])
        if 'randomize_tensor' in config['checkpoint']:
            args.checkpoint_randomize_tensor = config['checkpoint']['randomize_tensor']
        if 'ksm' in config['checkpoint']:
            args.ksm_present = True
            if 'madv_mergeable_id' in config['checkpoint']['ksm']:
                args.ksm_madv_mergeable_id = config['checkpoint']['ksm']['madv_mergeable_id']
            if 'high_ram_trigger' in config['checkpoint']['ksm']:
                args.ksm_high_ram_trigger = config['checkpoint']['ksm']['high_ram_trigger']
            if 'low_ram_exit' in config['checkpoint']['ksm']:
                args.ksm_low_ram_exit = config['checkpoint']['ksm']['low_ram_exit']
            if 'await_time' in config['checkpoint']['ksm']:
                args.ksm_await_time = config['checkpoint']['ksm']['await_time']

    if 'model' in config:
        if 'name' in config['model']:
            args.model = config['model']['name']
        if 'type' in config['model']:
            args.model_type = config['model']['type']
        if 'model_size_bytes' in config['model']:
            args.model_size = config['model']['model_size_bytes']
        if 'optimization_groups' in config['model']:
            args.optimization_groups = config['model']['optimization_groups']
        if 'num_layers' in config['model']:
            args.num_layers = config['model']['num_layers']
        if 'layer_parameters' in config['model']:
            args.layer_parameters = config['model']['layer_parameters']
        if 'model_datatype' in config['model']:
            args.model_datatype = config['model']['model_datatype']
        if 'optimizer_datatype' in config['model']:
            args.optimizer_datatype = config['model']['optimizer_datatype']

        if 'parallelism' in config['model']:
            if 'tensor' in config['model']['parallelism']:
                args.tensor_parallelism = config['model']['parallelism']['tensor']
            if 'pipeline' in config['model']['parallelism']:
                args.pipeline_parallelism = config['model']['parallelism']['pipeline']
            if 'data' in config['model']['parallelism']:
                args.data_parallelism = config['model']['parallelism']['data']
            if 'zero_stage' in config['model']['parallelism']:
                args.zero_stage = config['model']['parallelism']['zero_stage']

        if 'transformer' in config['model']:
            if 'vocab_size' in config['model']['transformer']:
                args.vocab_size = config['model']['transformer']['vocab_size']
            if 'hidden_size' in config['model']['transformer']:
                args.hidden_size = config['model']['transformer']['hidden_size']
            if 'ffn_hidden_size' in config['model']['transformer']:
                args.ffn_hidden_size = config['model']['transformer']['ffn_hidden_size']
            if 'num_attention_heads' in config['model']['transformer']:
                args.num_attention_heads = config['model']['transformer']['num_attention_heads']
            if 'num_kv_heads' in config['model']['transformer']:
                args.num_kv_heads = config['model']['transformer']['num_kv_heads']
            
    if 'output' in config:
        if 'folder' in config['output']:
            args.output_folder = config['output']['folder']
        if 'log_file' in config['output']:
            args.log_file = config['output']['log_file']
        if 'metric' in config['output']:
            if 'exclude_start_steps' in config['output']['metric']:
                args.metric_exclude_start_steps = int(config['output']['metric']['exclude_start_steps'])
            if 'exclude_end_steps' in config['output']['metric']:
                args.metric_exclude_end_steps = int(config['output']['metric']['exclude_end_steps'])

    if args.output_folder is None:
        try:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            args.output_folder = hydra_cfg['runtime']['output_dir']
        except:
            args.output_folder = 'output/'
    args.logfile_path = os.path.join(args.output_folder, args.log_file)

    if 'workflow' in config:
        if 'train' in config['workflow']:
            args.do_train = config['workflow']['train']
        if 'generate_data' in config['workflow']:
            args.generate_data = config['workflow']['generate_data']
        if 'evaluation' in config['workflow']:
            args.do_eval = config['workflow']['evaluation']
        if 'checkpoint' in config['workflow']:
            args.do_checkpoint = config['workflow']['checkpoint']
        if 'profiling' in config['workflow']:
            args.do_profiling = config['workflow']['profiling']
    
    if not args.do_train:
        if args.generate_data and (not args.do_checkpoint):
            args.generate_only = True
        if args.do_checkpoint:
            args.checkpoint_only = True

    if 'profiling' in config:
        if 'profiler' in config['profiling']:
            args.profiler = Profiler(config['profiling']['profiler'])
        if 'iostat_devices' in config['profiling']:
            args.iostat_devices = config['profiling']['iostat_devices']
            if isinstance(args.iostat_devices, str):
                args.iostat_devices = [args.iostat_devices]

    if 'metric' in config:
        if 'au' in config['metric']:
            args.au = config['metric']['au']
