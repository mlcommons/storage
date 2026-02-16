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

from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.storage.storage_factory import StorageFactory
import numpy as np
from dlio_benchmark.utils.utility import utcnow, add_padding, DLIOMPI


class DataGenerator(ABC):

    def __init__(self):
        self._args = ConfigArguments.get_instance()
        self._args.derive_configurations()
        self._dimension = self._args.dimension
        self._dimension_stdev = self._args.dimension_stdev
        self.data_dir = self._args.data_folder
        self.file_prefix = self._args.file_prefix
        self.num_files_train = self._args.num_files_train
        self.do_eval = self._args.do_eval
        self.num_files_eval = self._args.num_files_eval
        self.num_samples = self._args.num_samples_per_file
        self.my_rank = self._args.my_rank
        self.comm_size = self._args.comm_size
        self.compression = self._args.compression
        self.compression_level = self._args.compression_level
        self._file_prefix = None
        self._file_list = None
        self.num_subfolders_train = self._args.num_subfolders_train
        self.num_subfolders_eval = self._args.num_subfolders_eval
        self.format = self._args.format
        self.logger = self._args.logger
        self.storage = StorageFactory().get_storage(self._args.storage_type, self._args.storage_root,
                                                                        self._args.framework)

    def get_dimension(self, num_samples=1):
        if isinstance(self._dimension, list):
            if self._dimension_stdev > 0:
                # Generated shape (2*num_samples, len(self._dimension))
                random_values = np.random.normal(
                    loc=self._dimension,
                    scale=self._dimension_stdev,
                    size=(2 * num_samples, len(self._dimension))
                )
                dim = np.maximum(random_values.astype(int), 1).tolist()
            else:
                dim = [self._dimension for _ in range(2 * num_samples)]

            return dim

        if (self._dimension_stdev>0):
            dim = [max(int(d), 1) for d in np.random.normal(self._dimension, self._dimension_stdev, 2*num_samples)]
        else:
            dim = np.ones(2*num_samples, dtype=np.int64)*int(self._dimension)
        return dim 

    @abstractmethod
    def generate(self):
        nd_f_train = len(str(self.num_files_train))
        nd_f_eval = len(str(self.num_files_eval))
        nd_sf_train = len(str(self.num_subfolders_train))
        nd_sf_eval = len(str(self.num_subfolders_eval))

        if self.my_rank == 0:
            self.storage.create_node(self.data_dir, exist_ok=True)
            self.storage.create_node(self.data_dir + "/train/", exist_ok=True)
            self.storage.create_node(self.data_dir + "/valid/", exist_ok=True)
            if self.num_subfolders_train > 1: 
                for i in range(self.num_subfolders_train):
                    self.storage.create_node(self.data_dir + f"/train/{add_padding(i, nd_sf_train)}", exist_ok=True)
            if self.num_subfolders_eval > 1: 
                for i in range(self.num_subfolders_eval):
                    self.storage.create_node(self.data_dir + f"/valid/{add_padding(i, nd_sf_eval)}", exist_ok=True)
            self.logger.info(f"{utcnow()} Generating dataset in {self.data_dir}/train and {self.data_dir}/valid")
            self.logger.info(f"{utcnow()} Number of files for training dataset: {self.num_files_train}")
            self.logger.info(f"{utcnow()} Number of files for validation dataset: {self.num_files_eval}")


        DLIOMPI.get_instance().comm().barrier()
        # What is the logic behind this formula? 
        # Will probably have to adapt to generate non-images
        self.total_files_to_generate = self.num_files_train
        if self.num_files_eval > 0:
            self.total_files_to_generate += self.num_files_eval
        self._file_list = []


        if self.num_subfolders_train > 1:
            ns = np.ceil(self.num_files_train / self.num_subfolders_train)
            for i in range(self.num_files_train):
                file_spec = "{}/train/{}/{}_{}_of_{}.{}".format(self.data_dir, add_padding(i%self.num_subfolders_train, nd_sf_train), self.file_prefix, add_padding(i, nd_f_train), self.num_files_train, self.format)
                self._file_list.append(file_spec)
        else:
            for i in range(self.num_files_train):
                file_spec = "{}/train/{}_{}_of_{}.{}".format(self.data_dir, self.file_prefix, add_padding(i, nd_f_train), self.num_files_train, self.format)
                self._file_list.append(file_spec)
        if self.num_subfolders_eval > 1:
            ns = np.ceil(self.num_files_eval / self.num_subfolders_eval)
            for i in range(self.num_files_eval):
                file_spec = "{}/valid/{}/{}_{}_of_{}.{}".format(self.data_dir, add_padding(i%self.num_subfolders_eval, nd_sf_eval), self.file_prefix, add_padding(i, nd_f_eval), self.num_files_eval, self.format)
                self._file_list.append(file_spec)
        else:
            for i in range(self.num_files_eval):
                file_spec = "{}/valid/{}_{}_of_{}.{}".format(self.data_dir, self.file_prefix, add_padding(i, nd_f_eval), self.num_files_eval, self.format)
                self._file_list.append(file_spec)
