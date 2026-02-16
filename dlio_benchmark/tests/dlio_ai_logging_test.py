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

AI Logging Tests for DLIO Benchmark
====================================

These tests verify AI event logging functionality by running benchmarks as subprocesses
to ensure DFTracer traces are properly flushed before verification.

Running Tests:
--------------
# Run all tests sequentially:
pytest tests/dlio_ai_logging_test.py -v

# Run specific test:
pytest tests/dlio_ai_logging_test.py::test_ai_logging_train -k "pytorch-9-2" -v

# Run tests in parallel:
pytest tests/dlio_ai_logging_test.py -n auto -v
pytest tests/dlio_ai_logging_test.py -n 4 -v  # Use 4 workers

# Run with specific number of MPI processes (auto-detected):
# - If flux is available: uses flux run -n 2
# - Else if mpirun is available: uses mpirun -np 2
# - Otherwise: falls back to single process

Notes:
------
- Each test runs in its own subprocess with isolated storage directory
- Tests are safe to run in parallel (use pytest-xdist: -n auto)
- Item/preprocess events are counted globally across all trace files
- Per-rank events (root, epoch, train, etc.) are verified per rank
"""

#!/usr/bin/env python
import uuid
import pytest
import os
import glob
from datetime import datetime
from collections import Counter

from tests.utils import delete_folder, run_mpi_benchmark, NUM_PROCS, TEST_TIMEOUT_SECONDS


@pytest.fixture
def setup_test_env():
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    storage_root = os.path.join("outputs", f"{now}-{str(uuid.uuid4())}")

    if os.path.exists(storage_root):
        delete_folder(storage_root)
    os.makedirs(storage_root, exist_ok=True)

    yield storage_root

    delete_folder(storage_root)

def check_ai_events(path):
    counter = Counter(root=0, compute=0, item=0, preprocess=0, fetch_iter=0, train=0, eval=0, epoch=0, ckpt_capture=0, ckpt_restart=0)
    with open(path, mode="r") as f:
        for line in f:
            if "[" in line or "]" in line:
                continue
            if '"cat":"ai_root"' in line and '"name":"ai_root"' in line:
                counter["root"] += 1
            if '"cat":"compute"' in line and '"name":"compute"' in line:
                counter["compute"] += 1
            if '"cat":"data"' in line and '"name":"item"' in line:
                counter["item"] += 1
            if '"cat":"data"' in line and '"name":"preprocess"' in line:
                counter["preprocess"] += 1
            if '"cat":"dataloader"' in line and '"name":"fetch.iter"' in line:
                counter["fetch_iter"] += 1
            if '"cat":"checkpoint"' in line and '"name":"capture"' in line:
                counter["ckpt_capture"] += 1
            if '"cat":"checkpoint"' in line and '"name":"restart"' in line:
                counter["ckpt_restart"] += 1
            if '"cat":"pipeline"' in line and '"name":"train"' in line:
                counter["train"] += 1
            if '"cat":"pipeline"' in line and '"name":"evaluate"' in line:
                counter["eval"] += 1
            if '"cat":"pipeline"' in line and '"name":"epoch.block"' in line:
                counter["epoch"] += 1
    return counter

def get_rank_trace_files(all_paths, num_procs):
    """
    Find main trace files for each MPI rank.

    Args:
        all_paths: List of all .pfw trace file paths
        num_procs: Expected number of MPI processes

    Returns:
        Dictionary mapping rank number to trace file path
    """
    # Filter to main trace files only (exclude worker traces like trace-{hash}-app.pfw)
    main_traces = [p for p in all_paths if "-of-" in p and "-app.pfw" not in p]

    rank_traces = {}
    for rank in range(num_procs):
        # Match pattern: trace-{rank}-of-{num_procs}.pfw
        matching = [p for p in main_traces if f"trace-{rank}-of-{num_procs}.pfw" in p]
        if matching:
            rank_traces[rank] = matching[0]
        else:
            print(f"WARNING: No main trace file found for rank {rank}")

    return rank_traces

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("framework, num_data, batch_size", [
    (framework, num_data, batch_size)
    for framework in ["pytorch", "tensorflow"]
    for num_data in [9, 10]  # even and odd
    for batch_size in [2, 3]  # even and odd
])
def test_ai_logging_train(setup_test_env, framework, num_data, batch_size):
    storage_root = setup_test_env
    num_epochs = 2
    num_data_pp = num_data
    total_data = num_data_pp * NUM_PROCS

    overrides = [
        f"++workload.framework={framework}",
        f"++workload.reader.data_loader={framework}",
        "++workload.workflow.train=True",
        "++workload.workflow.evaluation=False",
        "++workload.workflow.generate_data=True",
        f"++workload.output.folder={storage_root}",
        f"++workload.dataset.data_folder={storage_root}/data",
        f"++workload.dataset.num_files_train={total_data}",
        "++workload.dataset.num_files_eval=0",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0",
        f"++workload.train.epochs={num_epochs}",
        f"++workload.reader.batch_size={batch_size}"
    ]

    # Run benchmark in MPI subprocess
    run_mpi_benchmark(overrides, num_procs=NUM_PROCS)

    paths = glob.glob(os.path.join(storage_root, "*.pfw"))

    assert len(paths) > 0, "No pfw files found"

    # Aggregate item and preprocess counts globally
    global_item_count = 0
    global_preprocess_count = 0

    for path in paths:
        count = check_ai_events(path=path)
        global_item_count += count["item"]
        global_preprocess_count += count["preprocess"]

    # Get main trace files for each rank
    rank_traces = get_rank_trace_files(paths, NUM_PROCS)

    # Check events from each rank's main trace file
    for rank, trace_path in rank_traces.items():
        count = check_ai_events(path=trace_path)
        print(f"[Rank {rank}] AI events count:", count)

        # check single file from single rank only
        assert count["root"]       == 1, f"Rank {rank}: Expected 1 root event, got {count['root']}"
        assert count["epoch"]      == num_epochs, f"Rank {rank}: Expected {num_epochs} epoch events, got {count['epoch']}"
        assert count["train"]      == num_epochs, f"Rank {rank}: Expected {num_epochs} train events, got {count['train']}"
        assert count["eval"]       == 0, f"Rank {rank}: Expected 0 eval events, got {count['eval']}"

        expected_iters = num_epochs * (num_data_pp // batch_size)
        assert count["fetch_iter"] == expected_iters, f"Rank {rank}: Expected {expected_iters} fetch_iter events, got {count['fetch_iter']}"
        assert count["compute"]    == expected_iters, f"Rank {rank}: Expected {expected_iters} compute events, got {count['compute']}"

        assert count["ckpt_capture"] == 0, f"Rank {rank}: Expected 0 ckpt_capture events, got {count['ckpt_capture']}"
        assert count["ckpt_restart"] == 0, f"Rank {rank}: Expected 0 ckpt_restart events, got {count['ckpt_restart']}"

    expected_total_iters = NUM_PROCS * num_epochs * (num_data_pp // batch_size)
    print(f"Global item count: {global_item_count}, preprocess count: {global_preprocess_count}")
    assert global_item_count       >= expected_total_iters, f"Expected at least {expected_total_iters} item events globally, got {global_item_count}"
    assert global_preprocess_count >= expected_total_iters, f"Expected at least {expected_total_iters} preprocess events globally, got {global_preprocess_count}"

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("framework, step, read_threads", [
    (framework, step, read_threads)
    for framework in ["pytorch", "tensorflow"]
    for step in [2, 3]  # even and odd
    for read_threads in [2, 3]  # even and odd
])
def test_ai_logging_train_with_step(setup_test_env, framework, step, read_threads):
    storage_root = setup_test_env
    num_epochs = 2
    batch_size = 2
    num_data_pp = 8
    total_data = num_data_pp * NUM_PROCS

    overrides = [
        f"++workload.framework={framework}",
        f"++workload.reader.data_loader={framework}",
        "++workload.workflow.train=True",
        "++workload.workflow.evaluation=False",
        "++workload.workflow.generate_data=True",
        f"++workload.output.folder={storage_root}",
        f"++workload.dataset.data_folder={storage_root}/data",
        f"++workload.dataset.num_files_train={total_data}",
        "++workload.dataset.num_files_eval=0",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0",
        f"++workload.reader.batch_size={batch_size}",
        f"++workload.train.epochs={num_epochs}",
        f"++workload.train.total_training_steps={step}",
        f"++workload.reader.read_threads={read_threads}",
    ]

    # Run benchmark in MPI subprocess
    run_mpi_benchmark(overrides, num_procs=NUM_PROCS)

    paths = glob.glob(os.path.join(storage_root, "*.pfw"))
    assert len(paths) > 0, "No pfw files found"

    # Aggregate item and preprocess counts globally
    global_item_count = 0
    global_preprocess_count = 0

    for path in paths:
        count = check_ai_events(path=path)
        global_item_count += count["item"]
        global_preprocess_count += count["preprocess"]

    # Get main trace files for each rank
    rank_traces = get_rank_trace_files(paths, NUM_PROCS)

    # Check events from each rank's main trace file
    for rank, trace_path in rank_traces.items():
        count = check_ai_events(path=trace_path)
        print(f"[Rank {rank}] AI events count:", count)

        assert count["root"]       == 1
        assert count["epoch"]      == num_epochs
        assert count["train"]      == num_epochs
        assert count["eval"]       == 0
        assert count["fetch_iter"] == num_epochs * step
        assert count["compute"]    == num_epochs * step

        assert count["ckpt_capture"] == 0
        assert count["ckpt_restart"] == 0

    expected_total = NUM_PROCS * num_epochs * step
    print(f"Global item count: {global_item_count}, preprocess count: {global_preprocess_count}")
    assert global_item_count       >= expected_total, f"Expected at least {expected_total} item events globally, got {global_item_count}"
    assert global_preprocess_count >= expected_total, f"Expected at least {expected_total} preprocess events globally, got {global_preprocess_count}"


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_ai_logging_with_eval(setup_test_env, framework):
    storage_root = setup_test_env
    num_epochs = 2
    batch_size = 1
    num_data_pp = 8
    total_data = num_data_pp * NUM_PROCS

    overrides = [
        f"++workload.framework={framework}",
        f"++workload.reader.data_loader={framework}",
        "++workload.workflow.train=True",
        "++workload.workflow.evaluation=True",
        "++workload.workflow.generate_data=True",
        f"++workload.output.folder={storage_root}",
        f"++workload.dataset.data_folder={storage_root}/data",
        f"++workload.dataset.num_files_train={total_data}",
        f"++workload.dataset.num_files_eval={total_data}",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0",
        f"++workload.reader.batch_size={batch_size}",
        f"++workload.train.epochs={num_epochs}"
    ]

    # Run benchmark in MPI subprocess
    run_mpi_benchmark(overrides, num_procs=NUM_PROCS)

    paths = glob.glob(os.path.join(storage_root, "*.pfw"))
    assert len(paths) > 0, "No pfw files found"

    # Aggregate item and preprocess counts globally
    global_item_count = 0
    global_preprocess_count = 0

    for path in paths:
        count = check_ai_events(path=path)
        global_item_count += count["item"]
        global_preprocess_count += count["preprocess"]

    # Get main trace files for each rank
    rank_traces = get_rank_trace_files(paths, NUM_PROCS)

    # Check events from each rank's main trace file
    for rank, trace_path in rank_traces.items():
        count = check_ai_events(path=trace_path)
        print(f"[Rank {rank}] AI events count:", count)

        assert count["root"]         == 1
        assert count["epoch"]        == num_epochs
        assert count["train"]        == num_epochs
        assert count["eval"]         == num_epochs
        assert count["fetch_iter"]   == 2 * num_epochs * (num_data_pp // batch_size)
        assert count["compute"]      == 2 * num_epochs * (num_data_pp // batch_size)

        assert count["ckpt_capture"] == 0
        assert count["ckpt_restart"] == 0

    expected_total = NUM_PROCS * 2 * num_epochs * num_data_pp
    print(f"Global item count: {global_item_count}, preprocess count: {global_preprocess_count}")
    assert global_item_count       >= expected_total, f"Expected at least {expected_total} item events globally, got {global_item_count}"
    assert global_preprocess_count >= expected_total, f"Expected at least {expected_total} preprocess events globally, got {global_preprocess_count}"

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("framework, fmt", [
    (framework, fmt)
    for framework in ["pytorch", "tensorflow"]
    for fmt in ["hdf5", "npy", "npz", "tfrecord", "csv", "jpeg", "png", "indexed_binary", "mmap_indexed_binary", "synthetic"]
    if not (fmt == "tfrecord" and framework == "pytorch")  # Exclude tfrecord + pytorch
])
def test_ai_logging_with_reader(setup_test_env, framework, fmt):
    storage_root = setup_test_env
    num_epochs = 2
    batch_size = 1
    num_data_pp = 8
    total_data = num_data_pp * NUM_PROCS

    overrides = [
        f"++workload.framework={framework}",
        f"++workload.reader.data_loader={framework}",
        "++workload.workflow.train=True",
        "++workload.workflow.evaluation=True",
        "++workload.workflow.generate_data=True",
        f"++workload.output.folder={storage_root}",
        f"++workload.dataset.data_folder={storage_root}/data",
        f"++workload.dataset.num_files_train={total_data}",
        f"++workload.dataset.num_files_eval={total_data}",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0",
        f"++workload.reader.batch_size={batch_size}",
        f"++workload.train.epochs={num_epochs}",
        f"++workload.dataset.format={fmt}",
    ]

    # Run benchmark in MPI subprocess
    run_mpi_benchmark(overrides, num_procs=NUM_PROCS)

    paths = glob.glob(os.path.join(storage_root, "*.pfw"))
    assert len(paths) > 0, "No pfw files found"

    # Aggregate item and preprocess counts globally
    global_item_count = 0
    global_preprocess_count = 0

    for path in paths:
        count = check_ai_events(path=path)
        global_item_count += count["item"]
        global_preprocess_count += count["preprocess"]

    # Get main trace files for each rank
    rank_traces = get_rank_trace_files(paths, NUM_PROCS)

    # Check events from each rank's main trace file
    for rank, trace_path in rank_traces.items():
        count = check_ai_events(path=trace_path)
        print(f"[Rank {rank}] AI events count:", count)

        assert count["root"]       == 1
        assert count["epoch"]      == num_epochs
        assert count["train"]      == num_epochs
        assert count["eval"]       == num_epochs
        assert count["fetch_iter"] == 2 * num_epochs * (num_data_pp // batch_size)
        assert count["compute"]    == 2 * num_epochs * (num_data_pp // batch_size)

        assert count["ckpt_capture"] == 0
        assert count["ckpt_restart"] == 0

    # Now check item and preprocess globally
    if fmt == "tfrecord":
        # @ray: tfrecord reader does not have notion of data item since our function
        # will be fused into execution graph, making it impossible to count the events
        # by just using decorator in python
        assert global_item_count == 0
        assert global_preprocess_count == 0
    else:
        expected_total_items = NUM_PROCS * 2 * num_epochs * num_data_pp
        print(f"Global item count: {global_item_count}, preprocess count: {global_preprocess_count}")
        assert global_item_count >= expected_total_items, f"Expected at least {expected_total_items} item events, got {global_item_count}"
        if fmt == "synthetic":
            # @ray: synthetic reader has no preprocess
            assert global_preprocess_count == 0
        else:
            assert global_preprocess_count >= expected_total_items, f"Expected at least {expected_total_items} preprocess events, got {global_preprocess_count}"

# @ray: future note: it seems DLIO hasn't implemented the all_ranks checkpointing yet
# this test suite is only for checkpointing on rank_zero only
# @todo: add test-cases to test all_ranks by adding ++workload.checkpoint.type=<VALUE>
@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("framework, epoch_per_ckpt, steps_per_ckpt", [
    (framework, epoch_per_ckpt, steps_per_ckpt)
    for framework in ["pytorch", "tensorflow"]
    for epoch_per_ckpt in [1, 2]
    for steps_per_ckpt in ["na", 1, 2]
])
def test_ai_logging_train_with_checkpoint(setup_test_env, framework, epoch_per_ckpt, steps_per_ckpt):
    storage_root = setup_test_env
    num_epochs = 2
    batch_size = 1
    num_data_pp = 4
    total_data = num_data_pp * NUM_PROCS
    if steps_per_ckpt == "na":
        steps_per_ckpt = -1

    overrides = [
        f"++workload.framework={framework}",
        f"++workload.reader.data_loader={framework}",
        "++workload.workflow.generate_data=True",
        "++workload.workflow.train=True",
        "++workload.workflow.evaluation=False",
        "++workload.workflow.checkpoint=True",
        f"++workload.output.folder={storage_root}",
        f"++workload.dataset.data_folder={storage_root}/data",
        f"++workload.dataset.num_files_train={total_data}",
        "++workload.dataset.num_files_eval=0",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0",
        f"++workload.train.epochs={num_epochs}",
        f"++workload.reader.batch_size={batch_size}",
        f"++workload.checkpoint.epochs_between_checkpoints={epoch_per_ckpt}",
        f"++workload.checkpoint.steps_between_checkpoints={steps_per_ckpt}",
    ]

    # Run benchmark in MPI subprocess
    run_mpi_benchmark(overrides, num_procs=NUM_PROCS)

    paths = glob.glob(os.path.join(storage_root, "*.pfw"))
    assert len(paths) > 0, "No pfw files found"

    # Aggregate item and preprocess counts globally
    global_item_count = 0
    global_preprocess_count = 0

    for path in paths:
        count = check_ai_events(path=path)
        global_item_count += count["item"]
        global_preprocess_count += count["preprocess"]

    # Get main trace files for each rank
    rank_traces = get_rank_trace_files(paths, NUM_PROCS)

    # Check events from each rank's main trace file
    # For checkpoint test, we need to find the specific rank trace files
    ckpt_capture_total = 0

    for rank, trace_path in rank_traces.items():
        count = check_ai_events(path=trace_path)
        print(f"[Rank {rank}] AI events count: {count}")

        assert count["root"]       == 1
        assert count["epoch"]      == num_epochs
        assert count["train"]      == num_epochs
        assert count["eval"]       == 0
        assert count["fetch_iter"] == num_epochs * (num_data_pp // batch_size)
        assert count["compute"]    == num_epochs * (num_data_pp // batch_size)

        assert count["ckpt_restart"] == 0

        # @ray: this assertion below is only for rank 0
        # @todo: when DLIO supports all_ranks checkpointing, adjust this
        if rank == 0:
            ckpt_capture_total = count["ckpt_capture"]

    expected_total_iters = NUM_PROCS * num_epochs * (num_data_pp // batch_size)
    print(f"Global item count: {global_item_count}, preprocess count: {global_preprocess_count}")
    assert global_item_count       >= expected_total_iters, f"Expected at least {expected_total_iters} item events, got {global_item_count}"
    assert global_preprocess_count >= expected_total_iters, f"Expected at least {expected_total_iters} preprocess events, got {global_preprocess_count}"

    # @ray: in DLIO step has more precedence compared to epoch
    if steps_per_ckpt != -1:
        expected_checkpoints = num_epochs * (num_data_pp // batch_size) // steps_per_ckpt
    else:
        expected_checkpoints = num_epochs // epoch_per_ckpt

    assert ckpt_capture_total == expected_checkpoints, f"Expected {expected_checkpoints} checkpoint captures, got {ckpt_capture_total}"

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("framework, num_checkpoint_write, num_checkpoint_read", [
    (framework, num_checkpoint_write, num_checkpoint_read)
    for framework in ["pytorch", "tensorflow"]
    for num_checkpoint_write in [3, 4]
    for num_checkpoint_read in [1, 2, 3]
])
def test_ai_logging_checkpoint_only(setup_test_env, framework, num_checkpoint_write, num_checkpoint_read):
    storage_root = setup_test_env

    overrides = [
        f"++workload.framework={framework}",
        f"++workload.reader.data_loader={framework}",
        "++workload.workflow.generate_data=False",
        "++workload.workflow.train=False",
        "++workload.workflow.evaluation=False",
        "++workload.workflow.checkpoint=True",
        f"++workload.output.folder={storage_root}",
        f"++workload.dataset.data_folder={storage_root}/data",
        "++workload.dataset.num_files_eval=0",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0",
        f"++workload.checkpoint.checkpoint_folder={storage_root}/checkpoint",
        f"++workload.checkpoint.num_checkpoints_write={num_checkpoint_write}",
        f"++workload.checkpoint.num_checkpoints_read={num_checkpoint_read}",
    ]

    # Run benchmark in MPI subprocess
    run_mpi_benchmark(overrides, num_procs=NUM_PROCS)

    paths = glob.glob(os.path.join(storage_root, "*.pfw"))
    assert len(paths) > 0, "No pfw files found"

    # Get main trace files for each rank
    rank_traces = get_rank_trace_files(paths, NUM_PROCS)

    # Check events from each rank's main trace file
    # For checkpoint test, only rank 0 does checkpointing
    ckpt_capture_total = 0
    ckpt_restart_total = 0

    for rank, trace_path in rank_traces.items():
        count = check_ai_events(path=trace_path)
        print(f"[Rank {rank}] AI events count: {count}")

        assert count["root"]       == 1
        assert count["epoch"]      == 0
        assert count["train"]      == 0
        assert count["eval"]       == 0
        assert count["fetch_iter"] == 0
        assert count["item"]       == 0
        assert count["preprocess"] == 0

        # @ray: this assertion below is only for rank 0
        # @todo: when DLIO supports all_ranks checkpointing, adjust this
        if rank == 0:
            ckpt_capture_total = count["ckpt_capture"]
            ckpt_restart_total = count["ckpt_restart"]
            assert count["compute"] == num_checkpoint_write + num_checkpoint_read

    assert ckpt_capture_total == num_checkpoint_write, f"Expected {num_checkpoint_write} checkpoint writes, got {ckpt_capture_total}"
    assert ckpt_restart_total == num_checkpoint_read, f"Expected {num_checkpoint_read} checkpoint reads, got {ckpt_restart_total}"
