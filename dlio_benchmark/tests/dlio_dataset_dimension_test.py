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
import uuid
import pytest
import logging
import os
import glob
from datetime import datetime

import numpy as np

import dlio_benchmark

from tests.utils import delete_folder, run_mpi_benchmark, NUM_PROCS, TEST_TIMEOUT_SECONDS

DTYPES = ["float32", "int8", "float16"]
DIMENSIONS = [2, 3, 4]


config_dir = os.path.dirname(dlio_benchmark.__file__) + "/configs/"

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(
            "dlio_dataset_dimension_test.log", mode="a", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
    format="[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]",
    # logging's max timestamp resolution is msecs, we will pass in usecs in the message
)

def generate_dlio_param(framework, storage_root, fmt, num_data, num_epochs=2):
    return [
        f"++workload.framework={framework}",
        f"++workload.reader.data_loader={framework}",
        "++workload.workflow.generate_data=True",
        f"++workload.output.folder={storage_root}",
        f"++workload.dataset.data_folder={storage_root}/data",
        f"++workload.dataset.num_files_train={num_data}",
        "++workload.dataset.num_files_eval=0",
        f"++workload.dataset.format={fmt}",
        "++workload.workflow.generate_data=True",
        f"++workload.dataset.num_files_train={num_data}",
        "++workload.dataset.num_files_eval=0",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0",
        "++workload.workflow.evaluate=False",
        "++workload.workflow.train=True",
        f"++workload.train.epochs={num_epochs}",
    ]

def generate_random_shape(dim):
    """Generate a random shape with the given dimensions (deterministic per test run)."""
    shape = [np.random.randint(1, 10) for _ in range(dim)]
    return shape

@pytest.fixture
def setup_test_env():
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    storage_root = os.path.join("outputs", f"{now}-{str(uuid.uuid4())}")

    if os.path.exists(storage_root):
        delete_folder(storage_root)
    os.makedirs(storage_root, exist_ok=True)

    yield storage_root

    delete_folder(storage_root)


def check_h5(path):
    import h5py

    f = h5py.File(path, "r")
    keys = list(f.keys())
    keys.remove("labels")
    variable = keys[-1]
    return f[variable].shape, f[variable].dtype, len(keys)


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("dtype, dim", [
    (dtype, dim)
    for dtype in DTYPES
    for dim in DIMENSIONS
])
def test_dim_based_hdf5_gen_data(setup_test_env, dtype, dim) -> None:
    fmt = "hdf5"
    framework = "pytorch"
    num_dset_per_record = 3
    shape_per_dataset = (1, *generate_random_shape(dim))
    shape = (num_dset_per_record * shape_per_dataset[0], *shape_per_dataset[1:])
    num_data_pp = 8
    total_data = num_data_pp * NUM_PROCS
    storage_root = setup_test_env

    overrides = [
        f"++workload.dataset.record_dims={list(shape)}",
        f"++workload.dataset.record_element_type={dtype}",
        f"++workload.dataset.hdf5.num_dset_per_record={num_dset_per_record}",
    ] + generate_dlio_param(framework=framework,
                            storage_root=storage_root,
                            fmt=fmt,
                            num_data=total_data)

    # Run benchmark in subprocess
    run_mpi_benchmark(overrides, num_procs=NUM_PROCS)

    paths = glob.glob(os.path.join(storage_root, "data", "train", "*.hdf5"))
    assert len(paths) > 0

    chosen_path = paths[0]
    gen_shape, gen_dtype, gen_num_ds = check_h5(chosen_path)

    print(f"Generated shape: {gen_shape}")
    print(f"Generated dtype: {gen_dtype}")
    print(f"Number of datasets: {gen_num_ds}")

    assert shape_per_dataset == gen_shape
    assert dtype == gen_dtype
    assert num_dset_per_record == gen_num_ds

def check_image(path):
    from PIL import Image

    img = Image.open(path)
    return img.size, img.format


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("fmt, dtype, dim", [
    (fmt, dtype, dim)
    for fmt in ["png", "jpeg"]
    for dtype in DTYPES
    for dim in DIMENSIONS
])
def test_dim_based_image_gen_data(setup_test_env, dtype, fmt, dim) -> None:
    framework = "pytorch"
    shape = generate_random_shape(dim)
    num_data_pp = 8
    total_data = num_data_pp * NUM_PROCS
    storage_root = setup_test_env

    if dim > 2:
        # @ray: check if dimension provided by user > 3
        # this will throw exception because we only support 2D shape for image
        print("Checking assertion when dimension > 2")

        overrides = [
            f"++workload.dataset.record_element_type={dtype}",
            f"++workload.dataset.record_dims={list(shape)}",
        ] + generate_dlio_param(framework=framework,
                                storage_root=storage_root,
                                fmt=fmt,
                                num_data=total_data)

        # Run benchmark expecting it to fail
        result = run_mpi_benchmark(overrides, num_procs=NUM_PROCS, expect_failure=True)
        assert result.returncode != 0, "Expected benchmark to fail for dim > 2"
        expected_error = f"{fmt} format does not support more than 2 dimensions, but got {dim} dimensions."
        assert expected_error in result.stderr, f"Expected error message not found in stderr: {result.stderr}"
    else:
        overrides = [
            f"++workload.dataset.record_element_type={dtype}",
            f"++workload.dataset.record_dims={list(shape)}",
        ] + generate_dlio_param(framework=framework,
                                storage_root=storage_root,
                                fmt=fmt,
                                num_data=total_data)

        # Run benchmark in subprocess
        run_mpi_benchmark(overrides, num_procs=NUM_PROCS)

        # @ray: we auto convert other dtype to uint8.
        # this is to ensure compatibility with PIL fromarray
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray)
        paths = glob.glob(os.path.join(storage_root, "data", "train", f"*.{fmt}"))
        assert len(paths) > 0

        chosen_path = paths[0]
        gen_shape, gen_format = check_image(chosen_path)

        print(f"Generated width: {gen_shape[0]}")
        print(f"Generated height: {gen_shape[1]}")
        print(f"Generated format: {gen_format}")

        assert len(shape) == 2
        height, width = shape
        assert (width, height) == gen_shape
        assert fmt == gen_format.lower()

def check_np(path, fmt):
    if fmt == "npy":
        data = np.load(path)
        return data.shape, data.dtype
    elif fmt == "npz":
        data = np.load(path)
        return data["x"].shape, data["x"].dtype
    else:
        raise ValueError(f"Unsupported format: {fmt}")

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("fmt, dtype, dim", [
    (fmt, dtype, dim)
    for fmt in ["npz", "npy"]
    for dtype in DTYPES
    for dim in DIMENSIONS
])
def test_dim_based_np_gen_data(setup_test_env, fmt, dtype, dim) -> None:
    framework = "pytorch"
    num_samples_per_file = 1
    shape = generate_random_shape(dim)
    num_data_pp = 8
    total_data = num_data_pp * NUM_PROCS
    final_shape = (*shape, num_samples_per_file)
    storage_root = setup_test_env

    overrides = [
        f"++workload.dataset.num_samples_per_file={num_samples_per_file}",
        f"++workload.dataset.record_element_type={dtype}",
        f"++workload.dataset.record_dims={list(shape)}",
    ] + generate_dlio_param(framework=framework,
                            storage_root=storage_root,
                            fmt=fmt,
                            num_data=total_data)

    # Run benchmark in subprocess
    run_mpi_benchmark(overrides, num_procs=NUM_PROCS)

    paths = glob.glob(os.path.join(storage_root, "data", "train", f"*.{fmt}"))
    assert len(paths) > 0

    chosen_path = paths[0]
    gen_shape, gen_format = check_np(chosen_path, fmt=fmt)

    print(f"Generated shape: {gen_shape}")
    print(f"Generated format: {gen_format}")

    assert final_shape == gen_shape
    assert np.dtype(dtype) == gen_format
    assert np.dtype(dtype).itemsize == gen_format.itemsize

def check_tfrecord(paths):
    import tensorflow as tf
    dataset = tf.data.TFRecordDataset(paths)

    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
    }

    for data in dataset.take(1):
        parsed = tf.io.parse_example(data, features)
        record_length_bytes = (
            tf.strings.length(parsed["image"], unit="BYTE").numpy().item()
        )        
        return record_length_bytes

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("dtype, dim", [
    (dtype, dim)
    for dtype in DTYPES
    for dim in DIMENSIONS
])
def test_dim_based_tfrecord_gen_data(setup_test_env, dtype, dim) -> None:
    framework = "tensorflow"
    fmt = "tfrecord"
    shape = generate_random_shape(dim)
    storage_root = setup_test_env
    num_data_pp = 8
    total_data = num_data_pp * NUM_PROCS

    overrides = [
        f"++workload.dataset.record_element_type={dtype}",
        f"++workload.dataset.record_dims={list(shape)}",
    ] + generate_dlio_param(framework=framework,
                            storage_root=storage_root,
                            fmt=fmt,
                            num_data=total_data)

    # Run benchmark in subprocess
    run_mpi_benchmark(overrides, num_procs=NUM_PROCS)

    train_data_dir = os.path.join(storage_root, "data", "train")
    paths = glob.glob(os.path.join(train_data_dir, "*.tfrecord"))
    assert len(paths) > 0

    gen_bytes = check_tfrecord(paths)

    print(f"Generated bytes: {gen_bytes}")

    assert np.prod(shape) * np.dtype(dtype).itemsize == gen_bytes

# @ray: this code is taken from dlio_benchmark/reader/indexed_binary_reader.py
# if that file is changed this code may need to be updated
def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a

# @ray: this code is taken from dlio_benchmark/reader/indexed_binary_reader.py
# if that file is changed this code may need to be updated
def index_file_path_off(prefix_path):
    return prefix_path + '.off.idx'

# @ray: this code is taken from dlio_benchmark/reader/indexed_binary_reader.py
# if that file is changed this code may need to be updated
def index_file_path_size(prefix_path):
    return prefix_path + '.sz.idx'

# @ray: this code is taken from dlio_benchmark/reader/indexed_binary_reader.py
# if that file is changed this code may need to be updated
def get_indexed_metadata(path, num_samples_per_file):
    offset_file = index_file_path_off(path)
    sz_file = index_file_path_size(path)
    with open(offset_file, 'rb') as f:
        offsets = read_longs(f, num_samples_per_file)
    with open(sz_file, 'rb') as f:
        sizes = read_longs(f, num_samples_per_file)
    return offsets, sizes

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("dtype, num_samples_per_file, dim", [
    (dtype, num_samples_per_file, dim)
    for dtype in DTYPES
    for num_samples_per_file in [1, 2, 3]  # even and odd
    for dim in DIMENSIONS
])
def test_dim_based_indexed_gen_data(setup_test_env, dtype, num_samples_per_file, dim) -> None:
    framework = "pytorch"
    fmt = "indexed_binary"
    shape = generate_random_shape(dim)
    storage_root = setup_test_env
    num_data_pp = 8
    total_data = num_data_pp * NUM_PROCS

    overrides = [
        f"++workload.dataset.num_samples_per_file={num_samples_per_file}",
        f"++workload.dataset.record_element_type={dtype}",
        f"++workload.dataset.record_dims={list(shape)}",
    ] + generate_dlio_param(framework=framework,
                            storage_root=storage_root,
                            fmt=fmt,
                            num_data=total_data)

    # Run benchmark in subprocess
    run_mpi_benchmark(overrides, num_procs=NUM_PROCS)

    train_data_dir = os.path.join(storage_root, "data", "train")
    paths = glob.glob(os.path.join(train_data_dir, "*.indexed_binary"))
    assert len(paths) > 0

    chosen_path = paths[0]
    offsets, sizes = get_indexed_metadata(chosen_path, num_samples_per_file)

    assert len(offsets) == num_samples_per_file
    assert len(sizes) == num_samples_per_file

    print(f"Dimensions: {shape}")
    print(f"Generated offsets: {offsets}")
    print(f"Generated sizes: {sizes}")

    sample_size = np.prod(shape) * np.dtype(dtype).itemsize
    sample_size = sample_size.item()

    with open(chosen_path, "rb") as f:
        for i in range(len(offsets)):
            f.seek(offsets[i])
            data = f.read(sizes[i])
            assert len(data) == sizes[i]
            print(f"Read data of size {len(data)}")
            assert len(data) == sample_size, f"Sample size mismatch: {len(data)} != {sample_size}"


def check_csv(path):
    import pandas as pd
    df = pd.read_csv(path, compression="infer", header=None)
    return len(df.iloc[0])

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("dtype, dim", [
    (dtype, dim)
    for dtype in DTYPES
    for dim in DIMENSIONS
])
def test_dim_based_csv(setup_test_env, dtype, dim) -> None:
    framework = "pytorch"
    fmt = "csv"
    shape = generate_random_shape(dim)
    storage_root = setup_test_env
    num_data_pp = 8
    total_data = num_data_pp * NUM_PROCS

    overrides = [
        f"++workload.dataset.record_element_type={dtype}",
        f"++workload.dataset.record_dims={list(shape)}",
    ] + generate_dlio_param(framework=framework,
                            storage_root=storage_root,
                            fmt=fmt,
                            num_data=total_data)

    # Run benchmark in subprocess
    run_mpi_benchmark(overrides, num_procs=NUM_PROCS)

    train_data_dir = os.path.join(storage_root, "data", "train")
    paths = glob.glob(os.path.join(train_data_dir, "*.csv"))
    assert len(paths) > 0

    chosen_path = paths[0]

    expected_rows = np.prod(shape).item()
    print(f"Total rows from shape ({shape}): {expected_rows}")

    num_rows = check_csv(chosen_path)
    assert num_rows == expected_rows


def _run_transformed_sample_worker(storage_root, dtype, transformed_dtype, dim, shape, transformed_sample):
    """Worker function to run in spawned subprocess - needs to import everything locally."""
    import os
    import numpy as np
    import torch
    from mpi4py import MPI
    from hydra import initialize_config_dir, compose
    from dlio_benchmark.main import DLIOBenchmark
    from dlio_benchmark.utils.config import ConfigArguments
    from dlio_benchmark.utils.utility import DLIOMPI
    from dlio_benchmark.common.enumerations import DatasetType
    import dlio_benchmark

    comm = MPI.COMM_WORLD
    config_dir = os.path.dirname(dlio_benchmark.__file__) + "/configs/"

    DLIOMPI.get_instance().initialize()

    torch_to_numpy_dtype_map = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
        torch.complex64: np.complex64,
        torch.complex128: np.complex128,
    }

    framework = "pytorch"
    fmt = "hdf5"
    num_data_pp = 8
    num_data = num_data_pp * comm.size
    bbatch = None

    def generate_dlio_param(framework, storage_root, fmt, num_data, num_epochs=2):
        return [
            f"++workload.framework={framework}",
            f"++workload.reader.data_loader={framework}",
            "++workload.workflow.generate_data=True",
            f"++workload.output.folder={storage_root}",
            f"++workload.dataset.data_folder={storage_root}/data",
            f"++workload.dataset.num_files_train={num_data}",
            "++workload.dataset.num_files_eval=0",
            f"++workload.dataset.format={fmt}",
            "++workload.workflow.generate_data=True",
            f"++workload.dataset.num_files_train={num_data}",
            "++workload.dataset.num_files_eval=0",
            "++workload.dataset.num_subfolders_train=0",
            "++workload.dataset.num_subfolders_eval=0",
            "++workload.workflow.evaluate=False",
            "++workload.workflow.train=True",
            f"++workload.train.epochs={num_epochs}",
        ]

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"++workload.dataset.record_element_type={dtype}",
                f"++workload.dataset.record_dims={list(shape)}",
                f"++workload.reader.transformed_record_dims={list(transformed_sample)}",
                f"++workload.reader.transformed_record_element_type={transformed_dtype}",
                "++workload.reader.batch_size=1",
                "++workload.reader.read_threads=1",
            ] + generate_dlio_param(framework=framework,
                                    storage_root=storage_root,
                                    fmt=fmt,
                                    num_data=num_data),
        )
        comm.Barrier()
        ConfigArguments.reset()
        benchmark = DLIOBenchmark(cfg["workload"])
        benchmark.initialize()
        epoch = 1
        benchmark.args.reconfigure(epoch)
        if comm.rank == 0:
            print(f"Initializing data loader ({benchmark.args.data_loader}) with format {benchmark.args.format} and num epoch {epoch}")
        benchmark.framework.init_loader(benchmark.args.format, epoch=epoch, data_loader=benchmark.args.data_loader)
        benchmark.framework.get_loader(dataset_type=DatasetType.TRAIN).read()
        loader = benchmark.framework.get_loader(dataset_type=DatasetType.TRAIN)
        for epoch in range(1, epoch + 1):
            for batch in loader.next():
                bbatch = batch
                break
            benchmark.framework.get_loader(DatasetType.TRAIN).finalize()
        benchmark.finalize()

    # Verify on rank 0
    if comm.rank == 0:
        assert bbatch is not None, "Batch is None"
        assert list(bbatch.shape) == [1, *transformed_sample], f"Shape mismatch: {bbatch.shape} != {[1, *transformed_sample]}"
        assert torch_to_numpy_dtype_map.get(bbatch.dtype) == np.dtype(transformed_dtype), f"Dtype mismatch: {bbatch.dtype} != {transformed_dtype}"
        print(f"✓ Batch shape: {bbatch.shape}, dtype: {bbatch.dtype}")

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("dtype, transformed_dtype, dim", [
    (dtype, transformed_dtype, dim)
    for dtype in DTYPES
    for transformed_dtype in ["uint8", "float32"]
    for dim in DIMENSIONS
])
def test_transformed_sample(setup_test_env, dtype, transformed_dtype, dim) -> None:
    """Test transformed sample using subprocess with spawn context to isolate MPI."""
    import multiprocessing as mp

    storage_root = setup_test_env
    shape = generate_random_shape(dim)
    transformed_sample = generate_random_shape(2)
    print(f"Transformed sample shape: {transformed_sample}")

    # Use spawn context to run the test in a subprocess
    ctx = mp.get_context('spawn')
    p = ctx.Process(
        target=_run_transformed_sample_worker,
        args=(storage_root, dtype, transformed_dtype, dim, shape, transformed_sample)
    )
    p.start()
    p.join()

    # Check if subprocess succeeded
    assert p.exitcode == 0, f"Subprocess failed with exit code {p.exitcode}"
