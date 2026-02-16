"""
   Copyright (c) 2022, UChicago Argonne, LLC
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

#!/usr/bin/env python
from hydra import initialize, initialize_config_dir, compose
from omegaconf import OmegaConf
import unittest
from datetime import datetime
import uuid
from io import BytesIO
import glob
from mpi4py import MPI
from tests.utils import TEST_TIMEOUT_SECONDS

comm = MPI.COMM_WORLD

import pytest
import time
import subprocess
import logging
import os
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI
import dlio_benchmark

from unittest.mock import patch
try:
    from s3torchconnector._s3client import MockS3Client
    from s3torchconnector import S3Checkpoint
except ImportError as e:
    MockS3Client = None
    S3Checkpoint = None
from urllib.parse import urlparse

config_dir=os.path.dirname(dlio_benchmark.__file__)+"/configs/"

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("dlio_benchmark_test.log", mode="a", encoding='utf-8'),
        logging.StreamHandler()
    ], format='[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]'
    # logging's max timestamp resolution is msecs, we will pass in usecs in the message
)

from dlio_benchmark.main import DLIOBenchmark, set_dftracer_initialize, set_dftracer_finalize

def finalize():
    # DLIOMPI.get_instance().finalize()
    pass

def clean_s3(mock_client, bucket: str, prefixes: list[str]) -> None:
    comm.Barrier()
    if comm.rank == 0:
        for prefix in prefixes:
            keys = mock_client.list_objects(bucket, prefix)
            for key in keys:
                mock_client.remove_object(key)
    comm.Barrier()

def get_s3_prefixes_from_uri(uri: str, subdirs=("train", "valid")):
    parsed = urlparse(uri)
    base_prefix = parsed.path.lstrip("/")
    return [f"{base_prefix}/{subdir}" for subdir in subdirs]

def run_benchmark(cfg, verify=True):
    comm.Barrier()
    t0 = time.time()
    ConfigArguments.reset()
    benchmark = DLIOBenchmark(cfg["workload"])
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    t1 = time.time()
    if (comm.rank==0):
        logging.info("Time for the benchmark: %.10f" %(t1-t0))
        if (verify):
            assert(len(glob.glob(benchmark.output_folder+"./*_output.json"))==benchmark.comm_size)    
    return benchmark

class SafeMockS3Client:
    def __init__(self, storage):
        self.storage = storage

    def get_object(self, bucket, key, start=None, end=None):
        if key.startswith("s3://"):
            key = key[len("s3://"):]
            key = key.split("/", 1)[1]
        elif key.startswith(bucket + "/"):
            key = key[len(bucket) + 1:]
        data = self.storage.get(key, b"")
        if start is not None and end is not None:
            return BytesIO(data[start:end+1])
        return BytesIO(data)

    def put_object(self, bucket, key, storage_class=None):
        if key.startswith("s3://"):
            key = key[len("s3://"):]
            key = key.split("/", 1)[1]
        return MockS3Writer(key, self.storage)

    def list_objects(self, bucket, prefix="", delimiter=None, max_keys=None):
        parsed = urlparse(prefix)
        if parsed.scheme == 's3':
            prefix = parsed.path.lstrip('/')
        keys = [k for k in self.storage.keys() if k.startswith(prefix)]
        if max_keys is not None:
            keys = keys[:max_keys]
        stripped_keys = [k[len(prefix):].lstrip("/") if k.startswith(prefix) else k for k in keys]
        return [MockListObjectsResult([MockObjectInfo(k) for k in stripped_keys])]

class MockS3Writer:
    def __init__(self, key, storage):
        self.key = key
        self.storage = storage
        self.buffer = bytearray()
        self._closed = False

    def __enter__(self):
        # return the object used as 'writer' in the with-block
        return self

    def __exit__(self, exc_type, exc, tb):
        # Emulate a flush before close
        self.flush()
        # Always close; optionally handle exceptions if needed
        self.close()
        # Return False to propagate exceptions, True to suppress.
        return False

    def write(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.buffer.extend(data)

    def flush(self):
        # No-op for mock
        pass

    def close(self):
        if not self._closed:
            self.storage[self.key] = bytes(self.buffer)
            self._closed = True

class MockObjectInfo:
    def __init__(self, key):
        self.key = key

class MockListObjectsResult:
    def __init__(self, object_info_list):
        self.object_info = object_info_list

@pytest.fixture
def setup_test_env():
    DLIOMPI.get_instance().initialize()
    if comm.rank == 0:
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        storage_root = f"s3-test-bucket-{now}-{str(uuid.uuid4())}"
        storage_type = "s3"
    else:
        storage_root = None
        storage_type = None
        mock_client = None

    storage_root = comm.bcast(storage_root, root=0)
    storage_type = comm.bcast(storage_type, root=0)

    # Only rank 0 initializes the mock storage
    if comm.rank == 0:
        # Shared in-memory mock storage
        mock_storage = {}

        # Create mock client
        mock_client = MockS3Client(region="us-east-1", bucket=storage_root)
        mock_client.storage = mock_storage

        # Simulate bucket existence
        mock_client.add_object("init.txt", b"bucket initialized")
        mock_storage = mock_client.storage
    else:
        mock_storage = None
        mock_client = MockS3Client(region="us-east-1", bucket=storage_root)

    # Broadcast the mock_storage dictionary to all ranks
    mock_storage = comm.bcast(mock_storage, root=0)
    mock_client.storage = mock_storage

    # Patch internal client builder to return the same mock
    mock_client._client_builder = lambda: mock_client._mock_client

    # Patch put_object and get_object to simulate S3 behavior
    def mock_put_object(bucket, key, storage_class=None):
        if key.startswith("s3://"):
            key = key[len("s3://"):]
            key = key.split("/", 1)[1]
        return MockS3Writer(key, mock_storage)

    def mock_get_object(bucket, key, start=None, end=None):
        if key.startswith("s3://"):
            key = key[len("s3://"):]
            key = key.split("/", 1)[1]
        elif key.startswith(bucket + "/"):
            key = key[len(bucket) + 1:]  # removes bucket name if it's prepended manually

        data = mock_storage.get(key, b"")
        if start is not None and end is not None:
            return BytesIO(data[start:end+1])
        return BytesIO(data)

    def mock_list_objects(bucket, prefix="", delimiter=None, max_keys=None):
        # Just use prefix directly, no need to strip bucket name
        parsed = urlparse(prefix)
        if parsed.scheme == 's3':
            prefix = parsed.path.lstrip('/')
        keys = [k for k in mock_storage.keys() if k.startswith(prefix)]
        if max_keys is not None:
            keys = keys[:max_keys]

        # Strip the prefix from each key
        stripped_keys = [k[len(prefix):].lstrip("/") if k.startswith(prefix) else k for k in keys]

        if parsed.scheme == 's3':
            # Wrap keys in the expected structure
            object_info_list = [MockObjectInfo(k) for k in stripped_keys]
            return [MockListObjectsResult(object_info_list)]

        return stripped_keys

    mock_client.put_object = mock_put_object
    mock_client.get_object = mock_get_object
    mock_client.list_objects = mock_list_objects

    s3_overrides = [
        f"++workload.storage.storage_type={storage_type}",
        f"++workload.storage.storage_root={storage_root}",
        f"++workload.dataset.data_folder=s3://{storage_root}",
        "++workload.storage.storage_options.access_key_id=test-access-key",
        "++workload.storage.storage_options.secret_access_key=test-secret-key",
        "++workload.storage.storage_options.endpoint_url=https://localhost:9000",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0"
    ]

    comm.Barrier()
    yield storage_root, storage_type, mock_client, s3_overrides
    comm.Barrier()

@pytest.fixture
def patch_s3_checkpoint(setup_test_env):
    storage_root, storage_type, mock_client, s3_overrides = setup_test_env
    s3_overrides += [f"++workload.checkpoint.checkpoint_folder=s3://{storage_root}/checkpoints"]

    def mock_init(self, region=None, endpoint=None, s3client_config=None):
        self.region = region
        self.endpoint = endpoint
        self.s3client_config = s3client_config
        self._client = mock_client

    with patch("dlio_benchmark.checkpointing.pytorch_s3_checkpointing.S3Checkpoint.__init__", new=mock_init):
        yield setup_test_env  # yield the full tuple so tests can still use all values

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("fmt, framework", [("npy", "pytorch"), ("npz", "pytorch")])
def test_s3_gen_data(setup_test_env, fmt, framework) -> None:
    storage_root, storage_type, mock_client, s3_overrides = setup_test_env

    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        if (comm.rank == 0):
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO test for generating {fmt} dataset")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config', overrides=s3_overrides + [f'++workload.framework={framework}',
                                                           f'++workload.reader.data_loader={framework}',
                                                           '++workload.workflow.train=False',
                                                           '++workload.workflow.generate_data=True',
                                                           f"++workload.dataset.format={fmt}", 
                                                           "++workload.dataset.num_files_train=8", 
                                                           "++workload.dataset.num_files_eval=8"])
            benchmark = run_benchmark(cfg, verify=False)

            # Extract bucket and prefix from data_folder
            fmt = cfg.workload.dataset.format
            bucket_name = cfg.workload.storage.storage_root

            # Filter keys based on actual prefix
            train_keys = [k for k in mock_client.list_objects(bucket_name, "train/") if k.endswith(f".{fmt}")]
            valid_keys = [k for k in mock_client.list_objects(bucket_name, "valid/") if k.endswith(f".{fmt}")]
            assert len(train_keys) == cfg.workload.dataset.num_files_train
            assert len(valid_keys) == cfg.workload.dataset.num_files_eval
        
            # Clean up mock S3 after test
            clean_s3(mock_client, bucket_name, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_s3_subset(setup_test_env) -> None:
    storage_root, storage_type, mock_client, s3_overrides = setup_test_env
    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        if comm.rank == 0:
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO training test for subset")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            set_dftracer_finalize(False)
            # Generate data
            cfg = compose(config_name='config', overrides=s3_overrides + [
                '++workload.workflow.train=False',
                '++workload.workflow.generate_data=True'])
            benchmark = run_benchmark(cfg, verify=False)

            # Train on subset
            set_dftracer_initialize(False)
            cfg = compose(config_name='config', overrides=s3_overrides + [
                '++workload.workflow.train=True',
                '++workload.workflow.generate_data=False',
                '++workload.dataset.num_files_train=8',
                '++workload.train.computation_time=0.01'])
            benchmark = run_benchmark(cfg, verify=True)
            bucket_name = cfg.workload.storage.storage_root

        # Clean up mock S3
        clean_s3(mock_client, bucket_name, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_s3_eval(setup_test_env) -> None:
    storage_root, storage_type, mock_client, s3_overrides = setup_test_env
    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        if (comm.rank == 0):
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO test for evaluation")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config',
                          overrides=s3_overrides + ['++workload.workflow.train=True', \
                                     '++workload.workflow.generate_data=True', \
                                     'workload.train.computation_time=0.01', \
                                     'workload.evaluation.eval_time=0.005', \
                                     '++workload.train.epochs=4', 
                                     '++workload.workflow.evaluation=True'])
            benchmark = run_benchmark(cfg)
            bucket_name = cfg.workload.storage.storage_root
            # Clean up mock S3 after test
            clean_s3(mock_client, bucket_name, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("framework, nt", [("pytorch", 0), ("pytorch", 1), ("pytorch", 2)])
def test_s3_multi_threads(setup_test_env, framework, nt) -> None:
    storage_root, storage_type, mock_client, s3_overrides = setup_test_env
    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        if (comm.rank == 0):
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO test for generating multithreading read_threads={nt} {framework} framework")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config', overrides=s3_overrides + ['++workload.workflow.train=True',
                                                           '++workload.workflow.generate_data=True',
                                                           f"++workload.framework={framework}",
                                                           f"++workload.reader.data_loader={framework}",
                                                           f"++workload.reader.read_threads={nt}",
                                                           'workload.train.computation_time=0.01',
                                                           'workload.evaluation.eval_time=0.005',
                                                           '++workload.train.epochs=1',
                                                           '++workload.dataset.num_files_train=8',
                                                           '++workload.dataset.num_files_eval=8'])
            benchmark = run_benchmark(cfg)
            bucket_name = cfg.workload.storage.storage_root
        # Clean up mock S3 after test
        clean_s3(mock_client, bucket_name, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("nt, context", [(0, None), (1, "fork"), (2, "spawn"), (2, "forkserver")])
def test_s3_pytorch_multiprocessing_context(setup_test_env, nt, context, monkeypatch) -> None:
    if nt == 2 and context in ("spawn", "forkserver"):
        pytest.skip("Skipping multiprocessing test with mock client under spawn/forkserver due to patching limitations.")

    storage_root, storage_type, mock_client, s3_overrides = setup_test_env

    # Create a multiprocessing-safe mock client for this test only
    mock_storage = mock_client.storage if hasattr(mock_client, "storage") else {}
    safe_mock_client = SafeMockS3Client(mock_storage)

    # Patch globally using monkeypatch
    monkeypatch.setattr("s3torchconnector._s3client._s3client.S3Client", lambda *args, **kwargs: safe_mock_client)
    monkeypatch.setattr("dlio_benchmark.storage.s3_torch_storage.S3Client", lambda *args, **kwargs: safe_mock_client)

    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for pytorch multiprocessing_context={context} read_threads={nt}")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=s3_overrides + ['++workload.workflow.train=True',
                                                       '++workload.workflow.generate_data=True',
                                                       f"++workload.framework=pytorch",
                                                       f"++workload.reader.data_loader=pytorch",
                                                       f"++workload.reader.read_threads={nt}",
                                                       f"++workload.reader.multiprocessing_context={context}",
                                                       'workload.train.computation_time=0.01',
                                                       'workload.evaluation.eval_time=0.005',
                                                       '++workload.train.epochs=1',
                                                       '++workload.dataset.num_files_train=8',
                                                       '++workload.dataset.num_files_eval=8'])
        benchmark = run_benchmark(cfg)
        bucket_name = cfg.workload.storage.storage_root
    # Clean up mock S3 after test
    clean_s3(mock_client, bucket_name, ["train/", "valid/"])
    finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("fmt, framework, dataloader, is_even", [
                                            ("npz", "pytorch", "pytorch", True),
                                            ("npz", "pytorch", "pytorch", False),
                                            ("npy", "pytorch", "pytorch", True),
                                            ("npy", "pytorch", "pytorch", False),
                                            ])
def test_s3_train(setup_test_env, fmt, framework, dataloader, is_even) -> None:
    storage_root, storage_type, mock_client, s3_overrides = setup_test_env
    if is_even:
        num_files = 16
    else:
        num_files = 17
    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        if comm.rank == 0:
            logging.info("")
            logging.info("=" * 80)
            logging.info(f" DLIO training test: Generating data for {fmt} format")
            logging.info("=" * 80)
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config', overrides=s3_overrides + ['++workload.workflow.train=True',
                                                           '++workload.workflow.generate_data=True',
                                                           f"++workload.framework={framework}", \
                                                           f"++workload.reader.data_loader={dataloader}", \
                                                           f"++workload.dataset.format={fmt}",
                                                           'workload.train.computation_time=0.01', \
                                                           'workload.evaluation.eval_time=0.005', \
                                                           '++workload.train.epochs=1', \
                                                           f'++workload.dataset.num_files_train={num_files}', \
                                                           '++workload.reader.read_threads=1'])
            benchmark = run_benchmark(cfg)
            bucket_name = cfg.workload.storage.storage_root
        # Clean up mock S3 after test
        clean_s3(mock_client, bucket_name, ["train/", "valid/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("framework, model_size, optimizers, num_layers, layer_params, zero_stage, randomize", [
                                                                                         ("pytorch", 1024, [1024, 128], 2, [16], 0, True),
                                                                                         ("pytorch", 1024, [1024, 128], 2, [16], 3, True),
                                                                                         ("pytorch", 1024, [128], 1, [16], 0, True),
                                                                                         ("pytorch", 1024, [1024, 128], 2, [16], 0, False),
                                                                                         ("pytorch", 1024, [1024, 128], 2, [16], 3, False),
                                                                                         ("pytorch", 1024, [128], 1, [16], 0, False)])
def test_s3_checkpoint_epoch(patch_s3_checkpoint, framework, model_size, optimizers, num_layers, layer_params, zero_stage, randomize) -> None:
    storage_root, storage_type, mock_client, s3_overrides = patch_s3_checkpoint
    if comm.rank == 0:
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for checkpointing at the end of epochs")
        logging.info("=" * 80)
    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            epochs = 8
            epoch_per_ckp = 2
            cfg = compose(config_name='config',
                          overrides=s3_overrides + [f'++workload.framework={framework}',
                                     f'++workload.reader.data_loader={framework}',
                                     '++workload.workflow.train=True',
                                     '++workload.workflow.generate_data=True',
                                     f'++workload.checkpoint.randomize_tensor={randomize}',
                                     '++workload.train.computation_time=0.01',
                                     '++workload.evaluation.eval_time=0.005',
                                     f'++workload.train.epochs={epochs}', '++workload.workflow.checkpoint=True',
                                     f'++workload.checkpoint.epochs_between_checkpoints={epoch_per_ckp}',
                                     f'++workload.model.model_size={model_size}',
                                     f'++workload.model.optimization_groups={optimizers}',
                                     f'++workload.model.num_layers={num_layers}',
                                     f'++workload.model.parallelism.zero_stage={zero_stage}',
                                     f'++workload.model.layer_parameters={layer_params}',
                                     f'++workload.model.parallelism.tensor={comm.size}'])
            #comm.Barrier()
            benchmark = run_benchmark(cfg)
            bucket_name = cfg.workload.storage.storage_root
            # Filter keys based on actual prefix
            load_bin = mock_client.list_objects(bucket_name, "checkpoints/")
            n = 0
            if len(layer_params) > 0:
                n = num_layers
            nranks = comm.size
            num_model_files = 1
            num_optimizer_files = 1
            # We are setting num_layer_files to be one because pipeline parallelism is not used.
            num_layer_files = 1
            files_per_checkpoint = (num_model_files + num_optimizer_files + num_layer_files) * nranks
            if framework == "pytorch":
                num_check_files = epochs / epoch_per_ckp * files_per_checkpoint
                assert (len(load_bin) == num_check_files), f"files produced are {len(load_bin)} {num_check_files} {load_bin}"
            #comm.Barrier()
            clean_s3(mock_client, bucket_name, ["checkpoints/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_s3_checkpoint_step(patch_s3_checkpoint) -> None:
    storage_root, storage_type, mock_client, s3_overrides = patch_s3_checkpoint
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for checkpointing at the end of steps")
        logging.info("=" * 80)
    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config',
                          overrides=s3_overrides + ['++workload.workflow.train=True', \
                                     '++workload.workflow.generate_data=True', \
                                     '++workload.train.computation_time=0.01', \
                                     '++workload.evaluation.eval_time=0.005', \
                                     '++workload.train.epochs=8', '++workload.workflow.checkpoint=True', \
                                     '++workload.checkpoint.steps_between_checkpoints=2'])
            comm.Barrier()
            benchmark = run_benchmark(cfg)
            bucket_name = cfg.workload.storage.storage_root
            dataset = cfg['workload']['dataset']
            nstep = dataset.num_files_train * dataset.num_samples_per_file // cfg['workload']['reader'].batch_size // benchmark.comm_size
            ncheckpoints = nstep // 2 * 8
            load_bin = mock_client.list_objects(bucket_name, "checkpoints/")
            assert (len(load_bin) == ncheckpoints)
            clean_s3(mock_client, bucket_name, ["checkpoints/"])
        finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_s3_checkpoint_ksm_config(patch_s3_checkpoint) -> None:
    """
    Tests the loading and derivation of KSM configuration parameters
    based on the presence and content of the checkpoint.ksm subsection.
    """
    storage_root, storage_type, mock_client, s3_overrides = patch_s3_checkpoint
    if comm.rank == 0:
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for KSM checkpoint configuration loading")
        logging.info("=" * 80)

    # --- Test Case 1: KSM enabled with defaults ---
    # KSM is enabled just by adding the 'ksm: {}' section in overrides
    logging.info("Testing KSM enabled with defaults...")
    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config',
                          overrides=s3_overrides + [
                              '++workload.workflow.checkpoint=True',
                              '++workload.checkpoint.ksm={}',
                              '++workload.workflow.generate_data=False',
                              '++workload.workflow.train=False',
                              '++workload.checkpoint.num_checkpoints_write=1',
                              '++workload.checkpoint.num_checkpoints_read=1',
                              '++workload.checkpoint.randomize_tensor=False',
                          ])
            ConfigArguments.reset()
            # Pass only the workload part of the config
            benchmark = DLIOBenchmark(cfg['workload'])
            # initialize() loads and derives the config
            benchmark.initialize()
            bucket_name = cfg.workload.storage.storage_root

            # Get the loaded arguments instance
            args = ConfigArguments.get_instance()

            # --- Assertions for Case 1 ---
            # Check derived ksm_init flag
            assert args.ksm_init is True, "[Test Case 1 Failed] ksm_init should be True when ksm section is present"
            # Check default KSM parameter values loaded into flat args attributes
            assert args.ksm_madv_mergeable_id == 12, f"[Test Case 1 Failed] Expected default madv_mergeable_id 12, got {args.ksm_madv_mergeable_id}"
            assert args.ksm_high_ram_trigger == 30.0, f"[Test Case 1 Failed] Expected default high_ram_trigger 30.0, got {args.ksm_high_ram_trigger}"
            assert args.ksm_low_ram_exit == 15.0, f"[Test Case 1 Failed] Expected default low_ram_exit 15.0, got {args.ksm_low_ram_exit}"
            assert args.ksm_await_time == 200, f"[Test Case 1 Failed] Expected default await_time 200, got {args.ksm_await_time}"
            logging.info("[Test Case 1 Passed]")

    # --- Test Case 2: KSM enabled with overrides ---
    logging.info("Testing KSM enabled with overrides...")
    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config',
                          overrides=s3_overrides + [
                              '++workload.workflow.checkpoint=True',
                              '++workload.checkpoint.ksm.high_ram_trigger=25.5',
                              '++workload.checkpoint.ksm.await_time=100',
                              '++workload.workflow.generate_data=False',
                              '++workload.workflow.train=False',
                              '++workload.checkpoint.num_checkpoints_write=1',
                              '++workload.checkpoint.num_checkpoints_read=1',
                              '++workload.checkpoint.randomize_tensor=False'
                          ])
            ConfigArguments.reset()
            benchmark = DLIOBenchmark(cfg['workload'])
            benchmark.initialize()

            args = ConfigArguments.get_instance()

            # --- Assertions for Case 2 ---
            # Check derived ksm_init flag
            assert args.ksm_init is True, "[Test Case 2 Failed] ksm_init should be True"
            # Check overridden values
            assert args.ksm_high_ram_trigger == 25.5, f"[Test Case 2 Failed] Expected overridden high_ram_trigger 25.5, got {args.ksm_high_ram_trigger}"
            assert args.ksm_await_time == 100, f"[Test Case 2 Failed] Expected overridden await_time 100, got {args.ksm_await_time}"
            # Check defaults for non-overridden values
            assert args.ksm_madv_mergeable_id == 12, f"[Test Case 2 Failed] Expected default madv_mergeable_id 12, got {args.ksm_madv_mergeable_id}"
            assert args.ksm_low_ram_exit == 15.0, f"[Test Case 2 Failed] Expected default low_ram_exit 15.0, got {args.ksm_low_ram_exit}"
            logging.info("[Test Case 2 Passed]")

    # --- Test Case 3: KSM disabled (section omitted) ---
    logging.info("Testing KSM disabled (section omitted)...")
    with patch("dlio_benchmark.storage.s3_torch_storage.S3Client", return_value=mock_client):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='config',
                          overrides=s3_overrides + [
                              '++workload.workflow.checkpoint=True',
                              '++workload.workflow.generate_data=False',
                              '++workload.workflow.train=False',
                              '++workload.checkpoint.num_checkpoints_write=1',
                              '++workload.checkpoint.num_checkpoints_read=1',
                              '++workload.checkpoint.randomize_tensor=False'
                          ])
            ConfigArguments.reset()
            benchmark = DLIOBenchmark(cfg['workload'])
            benchmark.initialize()

            args = ConfigArguments.get_instance()

            # --- Assertions for Case 3 ---
            assert args.ksm_init is False, "[Test Case 3 Failed] ksm_init should be False when ksm section is omitted"
            assert args.ksm_madv_mergeable_id == 12, f"[Test Case 3 Failed] Expected default madv_mergeable_id 12, got {args.ksm_madv_mergeable_id}"
            assert args.ksm_high_ram_trigger == 30.0, f"[Test Case 3 Failed] Expected default high_ram_trigger 30.0, got {args.ksm_high_ram_trigger}"
            assert args.ksm_low_ram_exit == 15.0, f"[Test Case 3 Failed] Expected default low_ram_exit 15.0, got {args.ksm_low_ram_exit}"
            assert args.ksm_await_time == 200, f"[Test Case 3 Failed] Expected default await_time 200, got {args.ksm_await_time}"
            logging.info("[Test Case 3 Passed]")

    clean_s3(mock_client, bucket_name, ["checkpoints/"])
    finalize()

if __name__ == '__main__':
    unittest.main()
