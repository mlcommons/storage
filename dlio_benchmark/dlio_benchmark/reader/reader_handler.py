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
from abc import ABC, abstractmethod

import numpy as np

from dlio_benchmark.common.enumerations import DatasetType, ReadType
from dlio_benchmark.utils.utility import utcnow
from dlio_benchmark.utils.utility import Profile, sleep, dft_ai
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.common.constants import MODULE_DATA_READER

dlp = Profile(MODULE_DATA_READER)

class FormatReader(ABC):
    read_images = None

    def __init__(self, dataset_type, thread_index):
        self.thread_index = thread_index
        self._args = ConfigArguments.get_instance()
        self.logger = self._args.logger
        self.logger.debug(
            f"Loading {self.__class__.__qualname__} reader on thread {self.thread_index} from rank {self._args.my_rank}")
        self.dataset_type = dataset_type
        self.open_file_map = {}

        if FormatReader.read_images is None:
            FormatReader.read_images = 0
        self.step = 1
        self.image_idx = 0
        self._file_list = self._args.file_list_train if self.dataset_type is DatasetType.TRAIN else self._args.file_list_eval 
        self.batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        if dataset_type is DatasetType.TRAIN:
            self.global_index_map = self._args.train_global_index_map
            self.file_map = self._args.train_file_map
        else:
            self.file_map = self._args.val_file_map
            self.global_index_map = self._args.val_global_index_map

    @dft_ai.data.preprocess
    def preprocess(self, a=None):
        sleep(self._args.preprocess_time)
        return a

    @abstractmethod
    def open(self, filename):
        return 

    @abstractmethod
    def close(self, filename):
        pass

    @abstractmethod
    def get_sample(self, filename, sample_index):
        return

    @abstractmethod
    def next(self):
        batch = []
        image_processed = 0
        self.step = 1
        total_images = len(self.file_map[self.thread_index])
        self.logger.debug(f"{utcnow()} Reading {total_images} images thread {self.thread_index} rank {self._args.my_rank}")

        for global_sample_idx, filename, sample_index in self.file_map[self.thread_index]:
            dft_ai.data.item.start()
            self.image_idx = global_sample_idx
            if filename not in self.open_file_map or self.open_file_map[filename] is None:
                self.open_file_map[filename] = self.open(filename)
            self.get_sample(filename, sample_index)
            self.preprocess()
            batch.append(self._args.resized_image)
            image_processed += 1
            is_last = 0 if image_processed < total_images else 1
            if is_last:
                while len(batch) is not self.batch_size:
                    batch.append(self._args.resized_image)
            dft_ai.data.item.stop()
            if len(batch) == self.batch_size:
                self.step += 1
                batch = np.array(batch)
                yield batch
                batch = []
            if image_processed % self._args.num_samples_per_file == 0:
                self.close(filename)
                self.open_file_map[filename] = None
            if is_last:
                break

    @abstractmethod
    @dft_ai.data.item
    def read_index(self, global_sample_idx, step):
        self.step = step
        self.image_idx = global_sample_idx
        self.logger.debug(f"{self.global_index_map}")
        filename, sample_index = self.global_index_map[global_sample_idx]
        self.logger.debug(f"{utcnow()} read_index {filename}, {sample_index}")
        FormatReader.read_images += 1
        if self._args.read_type is ReadType.ON_DEMAND or filename not in self.open_file_map or self.open_file_map[filename] is None:
            self.open_file_map[filename] = self.open(filename)
        self.get_sample(filename, sample_index)
        self.preprocess()
        if self._args.read_type is ReadType.ON_DEMAND:
            self.close(filename)
            self.open_file_map[filename] = None
        return self._args.resized_image

    @abstractmethod
    def finalize(self):
        for filename, sample_index in self._args.file_map:
            if filename in self.open_file_map:
                self.close(filename)
                self.open_file_map[filename] = None

    @dlp.log
    def resize(self, image):
        return self._args.resized_image

    def __del__(self):
        self.thread_index = None
        self._args = None
        self.dataset_type = None
        self.open_file_map = None
        self.step = None
        self.image_idx = None
        self.batch_size = None

    @abstractmethod
    def is_index_based(self):
        return False

    @abstractmethod
    def is_iterator_based(self):
        return False
