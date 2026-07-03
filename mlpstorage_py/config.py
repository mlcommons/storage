import datetime
import enum
import os
import pathlib


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
MODELS_CLOSED = [UNET, RETINANET]
MODELS_OPEN   = [UNET, RETINANET]

H100 = "h100"
A100 = "a100"
B200 = "b200"
MI355 = "mi355"
ACCELERATORS = [H100, A100, B200, MI355]
ACCELERATORS_CLOSED = [B200, MI355]

OPEN = "open"
CLOSED = "closed"
CATEGORIES = [OPEN, CLOSED]

LLAMA3_8B = "llama3-8b"
LLAMA3_70B = 'llama3-70b'
LLAMA3_405B = 'llama3-405b'
LLAMA3_1T = 'llama3-1t'
LLM_MODELS = [LLAMA3_70B, LLAMA3_405B, LLAMA3_1T, LLAMA3_8B]
LLM_MODELS_CLOSED = LLM_MODELS

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
KVCACHE_MODEL_DEFAULT = 'llama3.1-8b'
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

# VDB Benchmark Configuration
VDB_INDEX_TYPES = ["DISKANN", "HNSW", "AISAQ", "IVF_FLAT", "IVF_SQ8", "FLAT"]
VDB_INDEX_TYPES_CLOSED = ["DISKANN", "HNSW", "AISAQ"]

VDB_ORCHESTRATION_MODES = ["ssh", "mpi"]
VDB_BENCHMARK_MODES = ["timed", "query_count", "sweep"]
# Vector-database engines. Only milvus is wired up today; the slot exists so
# accumulated results from multiple engines can coexist in one results-dir
# (path: vector_database/<engine>/<index>/<command>/<datetime>/).
VDB_ENGINES = ["milvus"]
VDB_ENGINE_DEFAULT = "milvus"
VDB_INDEX_DEFAULT = "DISKANN"

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

# -----------------------------------------------------------------------------
# MLPSTORAGE_* env-var-name string constants — SINGLE SOURCE OF TRUTH (D-10).
#
# Every mlpstorage-owned environment-variable NAME lives here as a string
# constant. Downstream modules (rules/utils.py, cli_parser.py, etc.) MUST
# import these names from this module rather than redefining them locally.
# Import direction is one-way (D-11): config.py MUST NOT import from
# mlpstorage_py.rules.*; doing so would create a cycle at interpreter
# startup. tests/unit/test_no_import_cycles.py locks this invariant.
# -----------------------------------------------------------------------------
MLPSTORAGE_ORGNAME_ENVVAR = "MLPSTORAGE_ORGNAME"
MLPSTORAGE_SYSTEMNAME_ENVVAR = "MLPSTORAGE_SYSTEMNAME"
MLPSTORAGE_RESULTS_DIR_ENVVAR = "MLPSTORAGE_RESULTS_DIR"
MLPSTORAGE_DATA_DIR_ENVVAR = "MLPSTORAGE_DATA_DIR"
MLPSTORAGE_CHECKPOINT_FOLDER_ENVVAR = "MLPSTORAGE_CHECKPOINT_FOLDER"

# _LEGACY_ENVVAR_MAP: new MLPSTORAGE_* env-var-name -> legacy MLPERF_*
# predecessor. Consumed by the parse-time migration-hint emitter (Plan 05-02
# D-04/D-05). MLPSTORAGE_CHECKPOINT_FOLDER_ENVVAR is intentionally absent —
# per D-08 no MLPERF_CHECKPOINT_FOLDER ever existed, so the migration hint
# never fires for that pair.
_LEGACY_ENVVAR_MAP = {
    MLPSTORAGE_SYSTEMNAME_ENVVAR: "MLPERF_SYSTEMNAME",
    MLPSTORAGE_RESULTS_DIR_ENVVAR: "MLPERF_RESULTS_DIR",
    MLPSTORAGE_DATA_DIR_ENVVAR: "MLPERF_DATA_DIR",
    MLPSTORAGE_ORGNAME_ENVVAR: "MLPERF_ORGNAME",
}


# -----------------------------------------------------------------------------
# ENV_FALLBACK_* module-level constants — resolved-at-import-time env-var
# reads for the four universal path/name arguments. Every resolver returns
# the empty string when its env var is unset (SC-3): never None, never a
# tempdir path. The universal-args layer (add_universal_arguments) decides
# required-vs-optional per subcommand; an empty fallback plus the parse-time
# loud-error gate (Plan 05-02) makes "no --flag and no env var" fail at
# parse time rather than silently producing malformed output paths (T-1-02).
#
# Resolvers are extracted from the module-level assignments so tests can
# exercise the env-driven branch without reloading this module — reload
# re-creates the PARAM_VALIDATION enum class, which then fails `in`-checks
# against any pre-imported copy held by other modules (notably the rules
# verifier). See LAY-04 for the same shape mirrored across all four.
# -----------------------------------------------------------------------------
def _resolve_env_fallback_results_dir() -> str:
    """Return MLPSTORAGE_RESULTS_DIR or empty string (never a tempdir path)."""
    return os.environ.get(MLPSTORAGE_RESULTS_DIR_ENVVAR, "")


def _resolve_env_fallback_systemname() -> str:
    """Return MLPSTORAGE_SYSTEMNAME or empty string."""
    return os.environ.get(MLPSTORAGE_SYSTEMNAME_ENVVAR, "")


def _resolve_env_fallback_data_dir() -> str:
    """Return MLPSTORAGE_DATA_DIR or empty string."""
    return os.environ.get(MLPSTORAGE_DATA_DIR_ENVVAR, "")


def _resolve_env_fallback_checkpoint_folder() -> str:
    """Return MLPSTORAGE_CHECKPOINT_FOLDER or empty string."""
    return os.environ.get(MLPSTORAGE_CHECKPOINT_FOLDER_ENVVAR, "")


ENV_FALLBACK_RESULTS_DIR = _resolve_env_fallback_results_dir()
ENV_FALLBACK_SYSTEMNAME = _resolve_env_fallback_systemname()
ENV_FALLBACK_DATA_DIR = _resolve_env_fallback_data_dir()
ENV_FALLBACK_CHECKPOINT_FOLDER = _resolve_env_fallback_checkpoint_folder()

import enum

class EXIT_CODE(enum.IntEnum):
    SUCCESS = 0
    GENERAL_ERROR = 1
    INVALID_ARGUMENTS = 2
    # CAP/VALR failure exit code (per 02-CONTEXT.md D-22). Aliased with INVALID_ARGUMENTS=2 for ergonomic naming at the typed-exception → exit mapping in main.py.
    CODE_IMAGE_ERROR = 2
    FILE_NOT_FOUND = 3
    PERMISSION_DENIED = 4
    CONFIGURATION_ERROR = 5
    FAILURE = 6
    TIMEOUT = 7
    INTERRUPTED = 8
    
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
