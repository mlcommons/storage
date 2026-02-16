"""
   Copyright (c) 2025, UChicago Argonne, LLC
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
import math
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow, Profile, DLIOLogger, dft_ai

dlp = Profile(MODULE_DATA_LOADER)

class DaliIndexDataset(object):

    def __init__(self, format_type, dataset_type, epoch, worker_index,
                 total_num_workers, total_num_samples, samples_per_worker, batch_size):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch = epoch
        self.total_num_workers = total_num_workers
        self.total_num_samples = total_num_samples
        self.samples_per_worker = samples_per_worker
        self.batch_size = batch_size
        self.worker_index = worker_index
        self.total_num_steps = self.samples_per_worker//batch_size
        self.reader = ReaderFactory.get_reader(type=self.format_type,
                                               dataset_type=self.dataset_type,
                                               thread_index=worker_index,
                                               epoch_number=self.epoch)
        assert(self.reader.is_index_based())
        start_sample = self.worker_index * samples_per_worker
        end_sample = (self.worker_index + 1) * samples_per_worker - 1
        if end_sample > total_num_samples - 1:
            end_sample = total_num_samples - 1
        if not hasattr(self, 'indices'):
            self.indices = list(range(start_sample, end_sample + 1))
        self.samples_per_worker = len(self.indices)
    def __call__(self, sample_info):
        DLIOLogger.get_instance().debug(
            f"{utcnow()} Reading {sample_info.idx_in_epoch} out of {self.samples_per_worker} by worker {self.worker_index} with {self.indices} indices")
        step = sample_info.iteration       
        if step >= self.total_num_steps or sample_info.idx_in_epoch >= self.samples_per_worker:
            # Indicate end of the epoch
            raise StopIteration()
        sample_idx = self.indices[sample_info.idx_in_epoch]
        with Profile(MODULE_DATA_LOADER, epoch=self.epoch, image_idx=sample_idx, step=step):
            image = self.reader.read_index(sample_idx, step)
        return image, np.uint8([sample_idx])

class DaliIteratorDataset(object):
    def __init__(self, format_type, dataset_type, epoch, worker_index,
                 total_num_workers, total_num_samples, samples_per_worker, batch_size):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch = epoch
        self.total_num_workers = total_num_workers
        self.total_num_samples = total_num_samples
        self.samples_per_worker = samples_per_worker
        self.batch_size = batch_size
        self.worker_index = worker_index
        self.total_num_steps = self.samples_per_worker//batch_size
        self.reader = ReaderFactory.get_reader(type=self.format_type,
                                               dataset_type=self.dataset_type,
                                               thread_index=worker_index,
                                               epoch_number=self.epoch)
        assert(self.reader.is_iterator_based())
    def __iter__(self):
        with Profile(MODULE_DATA_LOADER):
            for image in self.reader.next():
                yield image.numpy(), np.uint8([0])

class DaliDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch):
        super().__init__(format_type, dataset_type, epoch, DataLoaderType.DALI)
        self.pipelines = []
        self.dataset = None

    @dlp.log
    def read(self, init=False):
        if not init:
            return 0
        parallel = True if self._args.read_threads > 0 else False
        self.pipelines = []
        num_threads = 1
        if self._args.read_threads > 0:
            num_threads = self._args.read_threads
        prefetch_size = 2
        if self._args.prefetch_size > 0:
            prefetch_size = self._args.prefetch_size
        num_pipelines = 1
        samples_per_worker = int(math.ceil(self.num_samples/num_pipelines/self._args.comm_size))
        for worker_index in range(num_pipelines):
            global_worker_index = self._args.my_rank * num_pipelines + worker_index
            # None executes pipeline on CPU and the reader does the batching
            self.dataset = DaliIndexDataset(self.format_type, self.dataset_type, self.epoch_number, global_worker_index,
                                self._args.comm_size * num_pipelines, self.num_samples, samples_per_worker, 1)
            pipeline = Pipeline(batch_size=self.batch_size, num_threads=num_threads, device_id=None, py_num_workers=num_threads//num_pipelines,
                                prefetch_queue_depth=prefetch_size, py_start_method=self._args.multiprocessing_context, exec_async=True)
            with pipeline:
                images, labels = fn.external_source(source=self.dataset, num_outputs=2, dtype=[types.UINT8, types.UINT8],
                                                    parallel=parallel, batch=False)
                pipeline.set_outputs(images, labels)
            self.pipelines.append(pipeline)
        for pipe in self.pipelines:
            pipe.start_py_workers()
        for pipe in self.pipelines:
            pipe.build()
        for pipe in self.pipelines:
            pipe.schedule_run()            
        self.logger.debug(f"{utcnow()} Starting {num_threads} pipelines by {self._args.my_rank} rank ")

    @dlp.log
    def next(self):
        super().next()
        self.logger.debug(f"{utcnow()} Iterating pipelines by {self._args.my_rank} rank ")
        step = 0
        self.read(True)
        while step < self.num_samples // self.batch_size:
            for pipe in self.pipelines:
                dft_ai.dataloader.fetch.start()
                try:
                    outputs = pipe.share_outputs()
                except StopIteration:
                    # it is fine to not stop `dft_ai.dataloader.fetch` here since
                    # it will be reset at the next run
                    return
                dft_ai.dataloader.fetch.stop()
                self.logger.debug(f"{utcnow()} Output batch {step} {len(outputs)}")
                yield outputs
                step += 1
                dft_ai.update(step=step)
                pipe.release_outputs()
                pipe.schedule_run()
        self.epoch_number += 1
        dft_ai.update(epoch=self.epoch_number)

    @dlp.log
    def finalize(self):
        pass
