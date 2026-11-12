"""Submission configuration is data, never Python object constructors."""
import json
from unittest.mock import Mock

import pytest

from mlpstorage_py.config import HYDRA_OUTPUT_SUBDIR
from mlpstorage_py.rules.models import BenchmarkResult, DLIOResultParser


@pytest.mark.parametrize("legacy", [False, True])
def test_result_yaml_loaders_reject_python_objects(tmp_path, legacy):
    config = tmp_path / HYDRA_OUTPUT_SUBDIR
    config.mkdir()
    marker = tmp_path / "must-not-exist"
    (config / "unsafe.yaml").write_text(
        "!!python/object/apply:os.mkdir [" + json.dumps(str(marker)) + "]\n"
    )
    (config / "config.yaml").write_text("workload:\n  dataset:\n    num_files_train: 4\n")
    (config / "overrides.yaml").write_text("- workload.reader.read_threads=2\n")
    logger = Mock()
    if legacy:
        result = BenchmarkResult(str(tmp_path), logger)
        configs = result.hydra_configs
    else:
        configs = DLIOResultParser(logger=logger)._load_hydra_configs(str(tmp_path))
    assert not marker.exists()
    assert "unsafe.yaml" not in configs
    assert configs["config.yaml"]["workload"]["dataset"]["num_files_train"] == 4
    assert configs["overrides.yaml"] == ["workload.reader.read_threads=2"]
    logger.error.assert_called()
