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
import numpy as np

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.common.enumerations import DataLoaderSampler
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, dft_ai

dlp = Profile(MODULE_DATA_READER)


class IndexedBinaryMMapReader(FormatReader):
    """
    Reader for Indexed Binary Memory mapped files
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self.file_map_ibr = {}
        self.buffer_map = {}
        self.load_index()

    def index_file_path_off(self, prefix_path):
        return prefix_path + '.off.idx'

    def index_file_path_size(self, prefix_path):
        return prefix_path + '.sz.idx'

    def read_longs(self, f, n):
        a = np.empty(n, dtype=np.int64)
        f.readinto(a)
        return a

    def load_index_file(self, global_sample_idx, filename, sample_index):
        if filename not in self.file_map_ibr:
            offset_file = self.index_file_path_off(filename)
            sz_file = self.index_file_path_size(filename)
            self.file_map_ibr[filename] = []
            bin_buffer_mmap = np.memmap(offset_file, mode='r', order='C')
            bin_buffer = memoryview(bin_buffer_mmap)
            self.file_map_ibr[filename].append(np.frombuffer(bin_buffer, dtype=np.uint64))
            bin_buffer_mmap = np.memmap(sz_file, mode='r', order='C')
            bin_buffer = memoryview(bin_buffer_mmap)
            self.file_map_ibr[filename].append(np.frombuffer(bin_buffer, dtype=np.uint64))
            bin_buffer_mmap = np.memmap(filename, mode='r', order='C')
            bin_buffer = memoryview(bin_buffer_mmap)
            self.buffer_map[filename] = np.frombuffer(bin_buffer, dtype=np.uint8)

    @dlp.log
    def load_index(self):
        if self._args.data_loader_sampler == DataLoaderSampler.ITERATIVE:
            for global_sample_idx, filename, sample_index in self.file_map[self.thread_index]:
                self.load_index_file(global_sample_idx, filename, sample_index)
        elif self._args.data_loader_sampler == DataLoaderSampler.INDEX:
            for global_sample_idx, (filename, sample_index) in self.global_index_map.items():
                self.load_index_file(global_sample_idx, filename, sample_index)

    @dlp.log
    def open(self, filename):
        super().open(filename)        
        return self.buffer_map[filename]

    @dlp.log
    def close(self, filename):
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        buffer = self.buffer_map[filename]
        offset = self.file_map_ibr[filename][0][sample_index]
        size = self.file_map_ibr[filename][1][sample_index]
        image = buffer[offset:offset+size]
        dlp.update(image_size=size)
        dft_ai.update(image_size=size)

    def next(self):
        for batch in super().next():
            yield batch

    @dft_ai.data.item
    def read_index(self, image_idx, step):
        filename, sample_index = self.global_index_map[image_idx]
        self.get_sample(filename, sample_index)
        self.preprocess()
        return self._args.resized_image

    @dlp.log
    def finalize(self):
        super().finalize()
        if self._args.data_loader_sampler == DataLoaderSampler.ITERATIVE:
            for global_sample_idx, filename, sample_index in self.file_map[self.thread_index]:
                self.buffer_map[filename]._mmap.close()
                self.file_map_ibr[filename][0]._mmap.close()
                self.file_map_ibr[filename][1]._mmap.close()
        elif self._args.data_loader_sampler == DataLoaderSampler.INDEX:
            for global_sample_idx, (filename, sample_index) in self.global_index_map.items():
                self.buffer_map[filename]._mmap.close()
                self.file_map_ibr[filename][0]._mmap.close()
                self.file_map_ibr[filename][1]._mmap.close()
            

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
