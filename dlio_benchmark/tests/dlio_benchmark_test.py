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
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import unittest
import shutil
from mpi4py import MPI
import pathlib
comm = MPI.COMM_WORLD
import pytest
import time
import subprocess
import logging
import os
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI
import dlio_benchmark
from tests.utils import TEST_TIMEOUT_SECONDS

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
import glob

def init():
    DLIOMPI.get_instance().initialize()

def finalize():
    # DLIOMPI.get_instance().finalize()
    pass

def clean(storage_root="./") -> None:
    comm.Barrier()
    if (comm.rank == 0):
        shutil.rmtree(os.path.join(storage_root, "checkpoints"), ignore_errors=True)
        shutil.rmtree(os.path.join(storage_root, "data/"), ignore_errors=True)
        shutil.rmtree(os.path.join(storage_root, "output"), ignore_errors=True)
    comm.Barrier()


def run_benchmark(cfg, storage_root="./", verify=True):

    comm.Barrier()
    if (comm.rank == 0):
        shutil.rmtree(os.path.join(storage_root, "output"), ignore_errors=True)
    comm.Barrier()
    t0 = time.time()
    ConfigArguments.reset()
    benchmark = DLIOBenchmark(cfg['workload'])
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    t1 = time.time()
    if (comm.rank==0):
        logging.info("Time for the benchmark: %.10f" %(t1-t0)) 
        if (verify):
            assert(len(glob.glob(benchmark.output_folder+"./*_output.json"))==benchmark.comm_size)
    return benchmark


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("fmt, framework", [("png", "tensorflow"), ("npz", "tensorflow"),
                                            ("jpeg", "tensorflow"), ("tfrecord", "tensorflow"),
                                            ("hdf5", "tensorflow"), ("indexed_binary", "tensorflow"), ("mmap_indexed_binary", "tensorflow")])
def test_gen_data(fmt, framework) -> None:
    init()
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for generating {fmt} dataset")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=[f'++workload.framework={framework}',
                                                       f'++workload.reader.data_loader={framework}',
                                                       '++workload.workflow.train=False',
                                                       '++workload.workflow.generate_data=True',
                                                       f"++workload.dataset.format={fmt}", 
                                                       "++workload.dataset.num_files_train=8", 
                                                       "++workload.dataset.num_files_eval=8"])
        benchmark = run_benchmark(cfg, verify=False)
        if benchmark.args.num_subfolders_train <= 1:
            train = pathlib.Path(f"{cfg.workload.dataset.data_folder}/train")
            train_files = list(train.glob(f"*.{fmt}"))
            valid = pathlib.Path(f"{cfg.workload.dataset.data_folder}/valid")
            valid_files = list(valid.glob(f"*.{fmt}"))
            assert (len(train_files) == cfg.workload.dataset.num_files_train)
            assert (len(valid_files) == cfg.workload.dataset.num_files_eval)
        else:
            train = pathlib.Path(f"{cfg.workload.dataset.data_folder}/train")
            train_files = list(train.rglob(f"**/*.{fmt}"))
            valid = pathlib.Path(f"{cfg.workload.dataset.data_folder}/valid")
            valid_files = list(valid.rglob(f"**/*.{fmt}"))
            assert (len(train_files) == cfg.workload.dataset.num_files_train)
            assert (len(valid_files) == cfg.workload.dataset.num_files_eval)
        clean()
    finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_subset() -> None:
    init()
    clean()
    if comm.rank == 0:
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO training test for subset")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        set_dftracer_finalize(False)
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=False', \
                    '++workload.workflow.generate_data=True'])
        benchmark=run_benchmark(cfg, verify=False)
        set_dftracer_initialize(False)
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True', \
                        '++workload.workflow.generate_data=False', \
                            '++workload.dataset.num_files_train=8', \
                            '++workload.train.computation_time=0.01'])
        benchmark=run_benchmark(cfg, verify=True)
    clean()
    finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("fmt, framework", [("png", "tensorflow"), ("npz", "tensorflow"),
                                            ("jpeg", "tensorflow"), ("tfrecord", "tensorflow"),
                                            ("hdf5", "tensorflow"), ("indexed_binary", "tensorflow"),
                                            ("mmap_indexed_binary", "tensorflow")])
def test_storage_root_gen_data(fmt, framework) -> None:
    init()
    storage_root = "runs"

    clean(storage_root)
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for generating {fmt} dataset")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=[f'++workload.framework={framework}',
                                                       f'++workload.reader.data_loader={framework}',
                                                       '++workload.workflow.train=False',
                                                       '++workload.workflow.generate_data=True',
                                                       f"++workload.storage.storage_root={storage_root}",
                                                       f"++workload.dataset.format={fmt}", 
                                                       "++workload.dataset.num_files_train=16"])
        benchmark = run_benchmark(cfg, verify=False)
        if benchmark.args.num_subfolders_train <= 1:
            assert (
                    len(glob.glob(
                        os.path.join(storage_root, cfg.workload.dataset.data_folder, f"train/*.{fmt}"))) ==
                    cfg.workload.dataset.num_files_train)
            assert (
                    len(glob.glob(
                        os.path.join(storage_root, cfg.workload.dataset.data_folder, f"valid/*.{fmt}"))) ==
                    cfg.workload.dataset.num_files_eval)
        else:
            logging.info(os.path.join(storage_root, cfg.workload.dataset.data_folder, f"train/*/*.{fmt}"))
            assert (
                    len(glob.glob(
                        os.path.join(storage_root, cfg.workload.dataset.data_folder, f"train/*/*.{fmt}"))) ==
                    cfg.workload.dataset.num_files_train)
            assert (
                    len(glob.glob(
                        os.path.join(storage_root, cfg.workload.dataset.data_folder, f"valid/*/*.{fmt}"))) ==
                    cfg.workload.dataset.num_files_eval)
        clean(storage_root)
    finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_iostat_profiling() -> None:
    init()
    clean()
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for iostat profiling")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=False',
                                                       '++workload.workflow.generate_data=True'])

        benchmark = run_benchmark(cfg, verify=False)
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True',
                                                       '++workload.workflow.generate_data=False',
                                                       'workload.train.computation_time=0.01',
                                                       'workload.evaluation.eval_time=0.005',
                                                       'workload.train.epochs=1',
                                                       'workload.workflow.profiling=True',
                                                       'workload.profiling.profiler=iostat'])
        benchmark = run_benchmark(cfg)
        assert (os.path.isfile(benchmark.output_folder + "/iostat.json"))
        if (comm.rank == 0):
            logging.info("generating output data")
            hydra = f"{benchmark.output_folder}/.hydra"
            os.makedirs(hydra, exist_ok=True)
            yl: str = OmegaConf.to_yaml(cfg)
            with open(f"{hydra}/config.yaml", "w") as f:
                OmegaConf.save(cfg, f)
            with open(f"{hydra}/overrides.yaml", "w") as f:
                f.write('[]')
            subprocess.run(["ls", "-l", "/dev/null"], capture_output=True)
            cmd = f"dlio_postprocessor --output-folder={benchmark.output_folder}"
            cmd = cmd.split()
            subprocess.run(cmd, capture_output=True, timeout=120)
        clean()
    finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("framework, model_size, optimizers, num_layers, layer_params, zero_stage, randomize", [("tensorflow", 1024, [1024, 128], 2, [16], 0, True),
                                                                                         ("pytorch", 1024, [1024, 128], 2, [16], 0, True),
                                                                                         ("tensorflow", 1024, [1024, 128], 2, [16], 3, True),
                                                                                         ("pytorch", 1024, [1024, 128], 2, [16], 3, True),
                                                                                         ("tensorflow", 1024, [128], 1, [16], 0, True),
                                                                                         ("pytorch", 1024, [128], 1, [16], 0, True),
                                                                                         ("tensorflow", 1024, [1024, 128], 2, [16], 0, False),
                                                                                         ("pytorch", 1024, [1024, 128], 2, [16], 0, False),
                                                                                         ("tensorflow", 1024, [1024, 128], 2, [16], 3, False),
                                                                                         ("pytorch", 1024, [1024, 128], 2, [16], 3, False),
                                                                                         ("tensorflow", 1024, [128], 1, [16], 0, False),
                                                                                         ("pytorch", 1024, [128], 1, [16], 0, False)])
def test_checkpoint_epoch(framework, model_size, optimizers, num_layers, layer_params, zero_stage, randomize) -> None:
    init()
    clean()
    if comm.rank == 0:
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for checkpointing at the end of epochs")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        epochs = 8
        epoch_per_ckp = 2
        cfg = compose(config_name='config',
                      overrides=[f'++workload.framework={framework}',
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
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree("./checkpoints", ignore_errors=True)
            os.makedirs("./checkpoints", exist_ok=True)
        comm.Barrier()
        benchmark = run_benchmark(cfg)
        output = pathlib.Path("./checkpoints")
        load_bin = list(output.glob(f"*/*"))
        n = 0
        if len(layer_params) > 0:
            n = num_layers
        nranks = comm.size
        num_model_files = 1
        num_optimizer_files = 1
        # We are setting num_layer_files to be one because pipeline parallelism is not used. 
        num_layer_files = 1
        files_per_checkpoint = (num_model_files + num_optimizer_files + num_layer_files) * nranks
        if framework == "tensorflow":
            file_per_ckp = 2
            num_check_files = epochs / epoch_per_ckp * (files_per_checkpoint * file_per_ckp + 1)
            assert (len(load_bin) == num_check_files), f"files produced are {len(load_bin)} {num_check_files} {load_bin} "
        if framework == "pytorch":
            num_check_files = epochs / epoch_per_ckp * files_per_checkpoint
            assert (len(load_bin) == num_check_files), f"files produced are {len(load_bin)} {num_check_files} {load_bin}"
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree("./checkpoints", ignore_errors=True)
        comm.Barrier()
        clean()
    finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_checkpoint_step() -> None:
    init()
    clean()
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for checkpointing at the end of steps")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config',
                      overrides=['++workload.workflow.train=True', \
                                 '++workload.workflow.generate_data=True', \
                                 '++workload.train.computation_time=0.01', \
                                 '++workload.evaluation.eval_time=0.005', \
                                 '++workload.train.epochs=8', '++workload.workflow.checkpoint=True', \
                                 '++workload.checkpoint.steps_between_checkpoints=2'])
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree("./checkpoints", ignore_errors=True)
            os.makedirs("./checkpoints", exist_ok=True)
        comm.Barrier()
        benchmark = run_benchmark(cfg)
        dataset = cfg['workload']['dataset']
        nstep = dataset.num_files_train * dataset.num_samples_per_file // cfg['workload']['reader'].batch_size // benchmark.comm_size
        ncheckpoints = nstep // 2 * 8
        output = pathlib.Path("./checkpoints")
        load_bin = list(output.glob(f"*/*"))
        assert (len(load_bin) == ncheckpoints)
        clean()
    finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_checkpoint_ksm_config() -> None:
    """
    Tests the loading and derivation of KSM configuration parameters
    based on the presence and content of the checkpoint.ksm subsection.
    """
    init()
    clean()
    if comm.rank == 0:
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for KSM checkpoint configuration loading")
        logging.info("=" * 80)

    # --- Test Case 1: KSM enabled with defaults ---
    # KSM is enabled just by adding the 'ksm: {}' section in overrides
    logging.info("Testing KSM enabled with defaults...")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config',
                      overrides=[
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
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config',
                      overrides=[
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
    with initialize_config_dir(version_base=None, config_dir=config_dir):
         cfg = compose(config_name='config',
                      overrides=[
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

    clean()
    finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_eval() -> None:
    init()
    clean()
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for evaluation")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config',
                      overrides=['++workload.workflow.train=True', \
                                 '++workload.workflow.generate_data=True', \
                                 'workload.train.computation_time=0.01', \
                                 'workload.evaluation.eval_time=0.005', \
                                 '++workload.train.epochs=4', '++workload.workflow.evaluation=True'])
        benchmark = run_benchmark(cfg)
        clean()
    finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("framework, nt", [("tensorflow", 0), ("tensorflow", 1),("tensorflow", 2),
                                           ("pytorch", 0), ("pytorch", 1), ("pytorch", 2)])
def test_multi_threads(framework, nt) -> None:
    init()
    clean()
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for generating multithreading read_threads={nt} {framework} framework")
        logging.info("=" * 80)
        # with subTest(f"Testing full benchmark for format: {framework}-NT{nt}", nt=nt, framework=framework):
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True',
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
    clean()
    finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("nt, context", [(0, None), (1, "fork"), (2, "spawn"), (2, "forkserver")])
def test_pytorch_multiprocessing_context(nt, context) -> None:
    init()
    clean()
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for pytorch multiprocessing_context={context} read_threads={nt}")
        logging.info("=" * 80)
        # with subTest(f"Testing full benchmark for format: {framework}-NT{nt}", nt=nt, framework=pytorch):
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True',
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
    clean()
    finalize()

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("fmt, framework, dataloader, is_even", [("png", "tensorflow","tensorflow", True), ("npz", "tensorflow","tensorflow", True),
                                            ("jpeg", "tensorflow","tensorflow", True), ("tfrecord", "tensorflow","tensorflow", True),
                                            ("hdf5", "tensorflow","tensorflow", True), ("csv", "tensorflow","tensorflow", True),
                                            ("indexed_binary", "tensorflow","tensorflow", True), ("mmap_indexed_binary", "tensorflow","tensorflow", True),
                                            ("png", "pytorch", "pytorch", True), ("npz", "pytorch", "pytorch", True),
                                            ("jpeg", "pytorch", "pytorch", True), ("hdf5", "pytorch", "pytorch", True),
                                            ("csv", "pytorch", "pytorch", True), ("indexed_binary", "pytorch", "pytorch", True),
                                            ("mmap_indexed_binary", "pytorch", "pytorch", True),
                                            ("png", "tensorflow", "dali", True), ("npz", "tensorflow", "dali", True),
                                            ("jpeg", "tensorflow", "dali", True), ("hdf5", "tensorflow", "dali", True),
                                            ("csv", "tensorflow", "dali", True), ("indexed_binary", "tensorflow", "dali", True),
                                            ("mmap_indexed_binary", "tensorflow", "dali", True),
                                            ("png", "pytorch", "dali", True), ("npz", "pytorch", "dali", True),
                                            ("jpeg", "pytorch", "dali", True), ("hdf5", "pytorch", "dali", True),
                                            ("csv", "pytorch", "dali", True), ("indexed_binary", "pytorch", "dali", True),
                                            ("mmap_indexed_binary", "pytorch", "dali", True),
                                            ("png", "tensorflow","tensorflow", False), ("npz", "tensorflow","tensorflow", False),
                                            ("jpeg", "tensorflow","tensorflow", False), ("tfrecord", "tensorflow","tensorflow", False),
                                            ("hdf5", "tensorflow","tensorflow", False), ("csv", "tensorflow","tensorflow", False),
                                            ("indexed_binary", "tensorflow","tensorflow", False), ("mmap_indexed_binary", "tensorflow","tensorflow", False),
                                            ("png", "pytorch", "pytorch", False), ("npz", "pytorch", "pytorch", False),
                                            ("jpeg", "pytorch", "pytorch", False), ("hdf5", "pytorch", "pytorch", False),
                                            ("csv", "pytorch", "pytorch", False), ("indexed_binary", "pytorch", "pytorch", False),
                                            ("mmap_indexed_binary", "pytorch", "pytorch", False),
                                            ("png", "tensorflow", "dali", False), ("npz", "tensorflow", "dali", False),
                                            ("jpeg", "tensorflow", "dali", False), ("hdf5", "tensorflow", "dali", False),
                                            ("csv", "tensorflow", "dali", False), ("indexed_binary", "tensorflow", "dali", False),
                                            ("mmap_indexed_binary", "tensorflow", "dali", False),
                                            ("png", "pytorch", "dali", False), ("npz", "pytorch", "dali", False),
                                            ("jpeg", "pytorch", "dali", False), ("hdf5", "pytorch", "dali", False),
                                            ("csv", "pytorch", "dali", False), ("indexed_binary", "pytorch", "dali", False),
                                            ("mmap_indexed_binary", "pytorch", "dali", False),
                                            ])
def test_train(fmt, framework, dataloader, is_even) -> None:
    init()
    clean()
    if is_even:
        num_files = 16
    else:
        num_files = 17
    if comm.rank == 0:
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO training test: Generating data for {fmt} format")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True',
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
    #clean()
    finalize()


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("fmt, framework", [("png", "tensorflow"), ("npz", "tensorflow"),
                                            ("jpeg", "tensorflow"), ("tfrecord", "tensorflow"),
                                            ("hdf5", "tensorflow"), ("csv", "tensorflow"),
                                            ("indexed_binary", "tensorflow"), ("mmap_indexed_binary", "tensorflow"),
                                            ("png", "pytorch"), ("npz", "pytorch"),
                                            ("jpeg", "pytorch"), ("hdf5", "pytorch"),
                                            ("csv", "pytorch"), ("indexed_binary", "pytorch"),
                                            ("mmap_indexed_binary", "pytorch"),
                                            ])
def test_custom_storage_root_train(fmt, framework) -> None:
    init()
    storage_root = "root_dir"
    clean(storage_root)
    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO training test for {fmt} format in {framework} framework")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True', \
                                                       '++workload.workflow.generate_data=True', \
                                                       f"++workload.framework={framework}", \
                                                       f"++workload.reader.data_loader={framework}", \
                                                       f"++workload.dataset.format={fmt}",
                                                       f"++workload.storage.storage_root={storage_root}", \
                                                       'workload.train.computation_time=0.01', \
                                                       'workload.evaluation.eval_time=0.005', \
                                                       '++workload.train.epochs=1', \
                                                       '++workload.dataset.num_files_train=16', \
                                                       '++workload.reader.read_threads=1'])
        benchmark = run_benchmark(cfg)
    clean(storage_root)
    finalize()

compute_time_distributions = {
    "uniform": {"type": "uniform", "min": 1.0, "max": 2.0},
    "normal": {"type": "normal", "mean": 1.0, "stdev": 1.0},
    "gamma": {"type": "gamma", "shape": 1.0, "scale": 1.0},
    "exp": {"type": "exponential", "scale": 1.0},
    "poisson": {"type": "poisson", "lam": 1.0},
    "normal_v2": {"mean": 1.0}, # mean, dist: normal
    "normal_v3": {"mean": 1.0, "stdev": 1.0}, # mean, stdev, dist: normal
    "normal_v4": 2.0, # mean, dist: normal
}

@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("dist", list(compute_time_distributions.keys()))
def test_computation_time_distribution(request, dist) -> None:
    init()
    clean()
    compute_time_overrides = []
    dist_val = compute_time_distributions[dist]
    if isinstance(dist_val, dict):
        for key, value in dist_val.items():
            compute_time_overrides.append(f"++workload.train.computation_time.{key}={value}")
    else:
        compute_time_overrides.append(f"++workload.train.computation_time={dist_val}")

    if (comm.rank == 0):
        logging.info("")
        logging.info("=" * 80)
        logging.info(f" DLIO test for computation time distribution")
        logging.info("=" * 80)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        if request.config.is_dftracer_initialized:
            set_dftracer_initialize(False)
        else:
            set_dftracer_finalize(False)

        cfg = compose(config_name='config',
                      overrides=['++workload.workflow.train=True', \
                                 '++workload.workflow.generate_data=True', \
                                 '++workload.train.epochs=1'] + compute_time_overrides)
        benchmark = run_benchmark(cfg)
        if not request.config.is_dftracer_initialized:
            request.config.is_dftracer_initialized = True
        clean()
    finalize()

if __name__ == '__main__':
    unittest.main()
