"""
Test that datagen and run workload configs have matching dataset parameters.

Addresses: https://github.com/mlcommons/storage/issues/319
"""
import os
import glob
import pytest
import yaml

CONFIGS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "dlio", "workload"
)

# Models that have datagen + accelerator-specific run configs
MODELS_WITH_DATAGEN = ["flux", "dlrm", "retinanet"]

# Dataset keys that MUST be identical across datagen and run configs
MATCHED_KEYS = [
    "num_samples_per_file",
    "num_files_train",
    "record_length",
    "record_length_bytes",
    "format",
    "compression",
]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_config_sets():
    """
    Yield (model, datagen_path, [(accel_name, run_path), ...]) tuples.
    """
    for model in MODELS_WITH_DATAGEN:
        datagen = os.path.join(CONFIGS_DIR, f"{model}_datagen.yaml")
        if not os.path.exists(datagen):
            continue
        run_files = sorted(
            glob.glob(os.path.join(CONFIGS_DIR, f"{model}_*.yaml"))
        )
        run_files = [
            f for f in run_files if "_datagen" not in os.path.basename(f)
        ]
        if run_files:
            yield model, datagen, run_files


@pytest.mark.parametrize(
    "model, datagen_path, run_paths",
    list(get_config_sets()),
    ids=[m for m, _, _ in get_config_sets()],
)
def test_datagen_matches_run_configs(model, datagen_path, run_paths):
    """
    For each model, every key in MATCHED_KEYS that is present in the datagen
    config must have the same value in every accelerator-specific run config.
    """
    datagen_cfg = load_yaml(datagen_path).get("dataset", {})

    for run_path in run_paths:
        run_cfg = load_yaml(run_path).get("dataset", {})
        run_name = os.path.basename(run_path)

        for key in MATCHED_KEYS:
            if key in datagen_cfg and key in run_cfg:
                assert datagen_cfg[key] == run_cfg[key], (
                    f"{model}: dataset.{key} mismatch — "
                    f"{os.path.basename(datagen_path)} has {datagen_cfg[key]} "
                    f"but {run_name} has {run_cfg[key]}"
                )
