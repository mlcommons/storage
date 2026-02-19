import datetime
import enum
import os
import pathlib
import tempfile


def check_env(setting, default_value=None):
    """
    This function checks the config, the default value, and the environment variables in the correct order for setting
    our constants. Lower position overrides a higher position
        - default_value
        - value_from_config
        - environment variable
    """
    value_from_environment = os.environ.get(setting)
    if type(value_from_environment) is str:
        if value_from_environment.lower() == 'true':
            value_from_environment = True
        elif value_from_environment.lower() == 'false':
            value_from_environment = False

    set_value = None
    if value_from_environment is not None:
        set_value = value_from_environment
    elif default_value is not None:
        set_value = default_value
    else:
        set_value = None

    return set_value


MLPS_DEBUG = check_env('MLPS_DEBUG', False)
HISTFILE = os.path.join(pathlib.Path.home(), "mlps_history")

def get_datetime_string():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Define constants:
DATETIME_STR = get_datetime_string()
CONFIGS_ROOT_DIR = os.path.join(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0], "configs")

MLPSTORAGE_BIN_NAME = "mlpstorage"

HYDRA_OUTPUT_SUBDIR = "dlio_config"

COSMOFLOW = "cosmoflow"
RESNET = "resnet50"
UNET = "unet3d"
DLRM = "dlrm"
RETINANET = "retinanet"
FLUX = "flux"
MODELS = [COSMOFLOW, RESNET, UNET, DLRM, RETINANET, FLUX]

H100 = "h100"
A100 = "a100"
ACCELERATORS = [H100, A100]

OPEN = "open"
CLOSED = "closed"
CATEGORIES = [OPEN, CLOSED]

LLAMA3_8B = "llama3-8b"
LLAMA3_70B = 'llama3-70b'
LLAMA3_405B = 'llama3-405b'
LLAMA3_1T = 'llama3-1t'
LLM_MODELS = [LLAMA3_70B, LLAMA3_405B, LLAMA3_1T, LLAMA3_8B]

LLM_SUBSET_PROCS = 8
# Defined as (MinProcs, ZeroLevel, GPU per Data Parallel Instance, Closed GPU Count)
LLM_ALLOWED_VALUES = {
    LLAMA3_1T: (LLM_SUBSET_PROCS, 1, 8*64, 8*64*2),     # 8*64*2 = 1,024 processes
    LLAMA3_405B: (LLM_SUBSET_PROCS, 1, 8*32, 8*32*2),   # 8*32*2 = 512 processes
    LLAMA3_70B: (LLM_SUBSET_PROCS, 3, 8, 8*8),          # 8*8*1 = 64 processes
    LLAMA3_8B: (LLM_SUBSET_PROCS, 3, 8, 8)              # 8*1*1 = 8 processes
}

# Defined as (Model GB, Optimizer GB)
# These need to be updated with actual values
LLM_SIZE_BY_RANK = {
    LLAMA3_1T: (2571, 15426),
    LLAMA3_405B: (755, 4533),
    LLAMA3_70B: (130, 781),
    LLAMA3_8B: (15, 90)
}

CHECKPOINT_RANKS_STRINGS = "\n    ".join(
    [f'{key}: CLOSED in [{value[0]} || {value[3]}], OPEN allows a multiple of {value[2]}' for key, value in LLM_ALLOWED_VALUES.items()])

LLM_MODELS_STRINGS = "\n    ".join(LLM_MODELS)

# KV Cache benchmark model configurations
KVCACHE_MODELS = [
    'tiny-1b',
    'mistral-7b',
    'llama2-7b',
    'llama3.1-8b',
    'llama3.1-70b-instruct',
]

# KV Cache performance profiles
KVCACHE_PERFORMANCE_PROFILES = ['latency', 'throughput']

# KV Cache generation modes
KVCACHE_GENERATION_MODES = ['none', 'fast', 'realistic']

# Default runtime for KV Cache benchmark (seconds)
KVCACHE_DEFAULT_DURATION = 60

MPIRUN = "mpirun"
MPIEXEC = "mpiexec"
MPI_CMDS = [MPIRUN, MPIEXEC]

STEPS_PER_EPOCH = 500
MOST_MEMORY_MULTIPLIER = 5
MAX_READ_THREADS_TRAINING = 32

DEFAULT_HOSTS = ["127.0.0.1",]

MPI_RUN_BIN = os.environ.get("MPI_RUN_BIN", MPIRUN)
MPI_EXEC_BIN = os.environ.get("MPI_EXEC_BIN", MPIEXEC)
ALLOW_RUN_AS_ROOT = True

MAX_NUM_FILES_TRAIN = 128*1024

DEFAULT_RESULTS_DIR = os.path.join(tempfile.gettempdir(), f"mlperf_storage_results")

import enum

class EXIT_CODE(enum.IntEnum):
    SUCCESS = 0
    GENERAL_ERROR = 1
    INVALID_ARGUMENTS = 2
    FILE_NOT_FOUND = 3
    PERMISSION_DENIED = 4
    CONFIGURATION_ERROR = 5
    FAILURE = 6
    TIMEOUT = 7
    # Add more as needed
    
    def __str__(self):
        return f"{self.name} ({self.value})"
class EXEC_TYPE(enum.Enum):
    MPI = "mpi"
    DOCKER = "docker"
    def __str__(self):
        return self.value


class PARAM_VALIDATION(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    INVALID = "invalid"


class BENCHMARK_TYPES(enum.Enum):
    training = "training"
    vector_database = "vector_database"
    checkpointing = "checkpointing"
    kv_cache = "kv_cache"

# Enum for supported search metric types of COSINE, L2, IP
SEARCH_METRICS = ["COSINE", "L2", "IP"]

# Supported Index Types is only DISKANN but more could be supported in the future
INDEX_TYPES = ["DISKANN"]

# Supported vector data types is currently only FLOAT_VECTOR but more could be supported in the future
VECTOR_DTYPES = ["FLOAT_VECTOR"]

# Supported distributions are currently uniform, normal, or zipfian
DISTRIBUTIONS = ["uniform", "normal", "zipfian"]

# Default runtime for vector database benchmarks if not defined
VECTORDB_DEFAULT_RUNTIME = 60