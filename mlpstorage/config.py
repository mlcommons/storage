import datetime
import enum
import os
import tempfile

# Define constants:
DATETIME_STR = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CONFIGS_ROOT_DIR = os.path.join(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0], "configs")

COSMOFLOW = "cosmoflow"
RESNET = "resnet50"
UNET = "unet3d"
MODELS = [COSMOFLOW, RESNET, UNET]

H100 = "h100"
A100 = "a100"
ACCELERATORS = [H100, A100]

OPEN = "open"
CLOSED = "closed"
CATEGORIES = [OPEN, CLOSED]

LLAMA3_70B = 'llama3-70b'
LLAMA3_405B = 'llama3-405b'
LLM_1620B = 'llm-1620b'
LLM_MODELS = [LLAMA3_70B, LLAMA3_405B, LLM_1620B]

LLM_MODEL_PARALLELISMS = {
    LLAMA3_70B: dict(tp=8, pp=4, min_ranks=4 * 8),
    LLAMA3_405B: dict(tp=8, pp=16, min_ranks=8 * 16),
    LLM_1620B: dict(tp=8, pp=64, min_ranks=64 * 8),
}
MIN_RANKS_STR = "; ".join(
    [f'{key} = {value["min_ranks"]} accelerators' for key, value in LLM_MODEL_PARALLELISMS.items()])

MPIRUN = "mpirun"
MPIEXEC = "mpiexec"
MPI_CMDS = [MPIRUN, MPIEXEC]

STEPS_PER_EPOCH = 500
MOST_MEMORY_MULTIPLIER = 5
MAX_READ_THREADS_TRAINING = 32

DEFAULT_HOSTS = ["127.0.0.1",]

MPI_RUN_BIN = os.environ.get("MPI_RUN_BIN", MPIRUN)
ALLOW_RUN_AS_ROOT = True

MAX_NUM_FILES_TRAIN = 128*1024

DEFAULT_RESULTS_DIR = os.path.join(tempfile.gettempdir(), f"mlperf_storage_results")

class EXEC_TYPE(enum.Enum):
    MPI = "mpi"
    DOCKER = "docker"


class PARAM_VALIDATION(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    INVALID = "invalid"


class BENCHMARK_TYPES(enum.Enum):
    training = "training"
    vector_database = "vector_database"
    checkpointing = "checkpointing"

# Enum for supported search metric types of COSINE, L2, IP
SEARCH_METRICS = ["COSINE", "L2", "IP"]

# Supported Index Types is only DISKANN but more could be supported in the future
INDEX_TYPES = ["DISKANN"]

# Supported vector data types is currently only FLOAT_VECTOR but more could be supported in the future
VECTOR_DTYPES = ["FLOAT_VECTOR"]

# Supported distributions are currently uniform, normal, or zipfian
DISTRIBUTIONS = ["uniform", "normal", "zipfian"]