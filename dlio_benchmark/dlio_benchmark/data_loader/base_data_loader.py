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
import os
from abc import ABC, abstractmethod

from numpy import random

from dlio_benchmark.common.enumerations import FileAccess, DatasetType, MetadataType, Shuffle
from dlio_benchmark.framework.framework_factory import FrameworkFactory
from dlio_benchmark.storage.storage_factory import StorageFactory
from dlio_benchmark.utils.config import ConfigArguments


class BaseDataLoader(ABC):
    def __init__(self, format_type, dataset_type, epoch_number, data_loader_type):
        self._args = ConfigArguments.get_instance()
        self.dataset_type = dataset_type
        self.format_type = format_type
        self.epoch_number = epoch_number
        self.data_loader_type = data_loader_type
        self.num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
        self.batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        self.logger = self._args.logger

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def finalize(self):
        pass
