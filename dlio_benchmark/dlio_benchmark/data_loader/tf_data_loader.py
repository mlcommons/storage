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

import tensorflow as tf

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DataLoaderType, FormatType, DatasetType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow, Profile, DLIOLogger, dft_ai

import numpy as np

dlp = Profile(MODULE_DATA_LOADER)


class TensorflowDataset(tf.data.Dataset):
    @staticmethod
    @dlp.log
    def _generator(format_type, dataset_type, epoch_number, thread_index):
        format_type = format_type.decode('ascii')
        dataset_type = dataset_type.decode('ascii')
        DLIOLogger.get_instance().debug(f"{utcnow()} format_type {format_type} dataset_type {dataset_type} tensors")
        reader = ReaderFactory.get_reader(type=FormatType.get_enum(format_type),
                                          dataset_type=DatasetType.get_enum(dataset_type),
                                          thread_index=thread_index,
                                          epoch_number=epoch_number)
        for batch in reader.next():
            yield batch

    @dlp.log
    def __new__(cls, format_type, dataset_type, epoch, shape, thread_index):
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.uint8,
            output_shapes=shape,
            args=(format_type.value, dataset_type.value, epoch, thread_index,),
        )
        return dataset


class TFDataLoader(BaseDataLoader):

    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch):
        super().__init__(format_type, dataset_type, epoch, DataLoaderType.TENSORFLOW)
        self._dataset = None

    @dlp.log
    def read(self):
        read_threads = self._args.read_threads
        if read_threads == 0:
            if self._args.my_rank == 0:
                self.logger.warning(
                    f"{utcnow()} `read_threads` is set to be 0 for tf.data loader. We change it to 1")
            read_threads = 1

        options = tf.data.Options()
        if "threading" in dir(options):
            options.threading.private_threadpool_size = read_threads
            options.threading.max_intra_op_parallelism = read_threads
        elif "experimental_threading" in dir(options):
            options.experimental_threading.private_threadpool_size = read_threads
            options.experimental_threading.max_intra_op_parallelism = read_threads
        if self.format_type != FormatType.TFRECORD:
            self._dataset = tf.data.Dataset.from_tensor_slices(np.arange(read_threads)).with_options(options)
            self._dataset = self._dataset.interleave(lambda x: TensorflowDataset(self.format_type, self.dataset_type,
                                                                                self.epoch_number, (
                                                                                self.batch_size,
                                                                                self._args.max_dimension,
                                                                                self._args.max_dimension), x),
                                                                                cycle_length=read_threads,
                                                                                num_parallel_calls=read_threads)
            if self._args.prefetch_size > 0:
                self._dataset = self._dataset.prefetch(buffer_size=self._args.prefetch_size)
        else:
            self._dataset = ReaderFactory.get_reader(type=self.format_type,
                                          dataset_type=self.dataset_type,
                                          thread_index=-1,
                                          epoch_number=self.epoch_number).next()

    @dlp.log
    def next(self):
        super().next()
        step = 1
        for batch in dft_ai.dataloader.fetch.iter(self._dataset):
            dlp.update(step=step)
            dft_ai.update(step=step)
            step += 1
            yield batch
        self.epoch_number += 1
        dlp.update(epoch=self.epoch_number)
        dft_ai.update(epoch=self.epoch_number)

    @dlp.log
    def finalize(self):
        pass
