from .parsers.json_parser import JSONParser
from .parsers.yaml_parser import YamlParser

VERSIONS = ["v2.0", "v3.0"]
VALID_DIVISIONS = ["open", "closed"]

SYSTEM_PATH = {
    "v2.0": "{division}/{submitter}/systems/{system}.yaml",
    "v3.0": "{division}/{submitter}/systems/{system}.yaml",
    "default": "{division}/{submitter}/systems/{system}.yaml",
}

PARSER_MAP = {
    "System": YamlParser,
    "Summary": JSONParser,
    "Metadata": JSONParser,
    "default": JSONParser
}

DATAGEN_REQUIRED_FILES = {
    "v2.0": [r"training_datagen\.stdout.log", r"training_datagen.stderr\.log", r".*output\.json$", r".*per_epoch_stats\.json$", r".*summary\.json$", r"dlio\.log"],
    "v3.0": [r"training_datagen\.stdout.log", r"training_datagen.stderr\.log", r".*output\.json$", r".*per_epoch_stats\.json$", r",*summary\.json$", r"dlio\.log"],
    "default": [r"training_datagen\.stdout.log", r"training_datagen.stderr\.log", r".*output\.json$", r".*per_epoch_stats\.json$", r".*summary\.json$", r"dlio\.log"],
}

DATAGEN_REQUIRED_FOLDERS = {
    "v2.0": ["dlio_config"],
    "v3.0": ["dlio_config"],
    "default": ["dlio_config"],
}

RUN_REQUIRED_FILES = {
    "v2.0": [r"training_run\.stdout.log", r"training_run\.stderr.log", r".*output\.json", r".*per_epoch_stats\.json", r".*summary\.json", r"dlio\.log"],
    "v3.0": [r"training_run\.stdout.log", r"training_run\.stderr.log", r".*output\.json", r".*per_epoch_stats\.json", r".*summary\.json", r"dlio\.log"],
    "default": [r"training_run\.stdout.log", r"training_run\.stderr.log", r".*output\.json", r".*per_epoch_stats\.json", r".*summary\.json", r"dlio\.log"],
}

RUN_REQUIRED_FOLDERS = {
    "v2.0": ["dlio_config"],
    "v3.0": ["dlio_config"],
    "default": ["dlio_config"],
}

CHECKPOINT_REQUIRED_FILES = {
    "v2.0": [r"training_run\.stdout.log", r"training_run\.stderr.log", r".*output\.json", r".*per_epoch_stats\.json", r".*summary\.json", r"dlio\.log"],
    "v3.0": [r"training_run\.stdout.log", r"training_run\.stderr.log", r".*output\.json", r".*per_epoch_stats\.json", r".*summary\.json", r"dlio\.log"],
    "default": [r"training_run\.stdout.log", r"training_run\.stderr.log", r".*output\.json", r".*per_epoch_stats\.json", r".*summary\.json", r"dlio\.log"],
}

CHECKPOINT_REQUIRED_FOLDERS = {
    "v2.0": ["dlio_config"],
    "v3.0": ["dlio_config"],
    "default": ["dlio_config"],
}

# TODO: Ask for correct values
NUM_DATASET_TRAIN_FILES = {
    "cosmoflow": 524288,
    "resnet50": 10391,
    "unet3d": 14000
}

NUM_DATASET_EVAL_FILES = {
    "cosmoflow": 0,
    "resnet50": 0,
    "unet3d": 0
}

NUM_DATASET_TRAIN_FOLDERS = {
    "cosmoflow": 0,
    "resnet50": 0,
    "unet3d": 0
}

NUM_DATASET_EVAL_FOLDERS = {
    "cosmoflow": 0,
    "resnet50": 0,
    "unet3d": 0
}

CHECKPOINT_FILE_MAP = {
    "llama3-1t": "llama3_1t.yaml",
    "llama3-8b": "llama3_8b.yaml",
    "llama3-70b": "llama3_70b.yaml",
    "llama3-405b": "llama3_405b.yaml",
}