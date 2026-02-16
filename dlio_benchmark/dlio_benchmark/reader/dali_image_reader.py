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
import nvidia.dali.fn as fn
from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import utcnow
from dlio_benchmark.common.enumerations import Shuffle
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_DATA_READER)


class DaliImageReader(FormatReader):
    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

    @dlp.log        
    def open(self, filename):
        super().open(filename)
    
    def close(self):
        super().close()
    
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        raise Exception("get sample method is not implemented in dali readers")

    def next(self):
        super().next()
        raise Exception("next method is not implemented in dali readers")

    def read_index(self):
        super().read_index()
        raise Exception("read_index method is not implemented in dali readers")

    @dlp.log
    def pipeline(self):
        self.logger.debug(
            f"{utcnow()} Reading {len(self._file_list)} files rank {self._args.my_rank}")
        random_shuffle = False
        seed = -1
        seed_change_epoch = False
        if self._args.sample_shuffle is not Shuffle.OFF:
            if self._args.sample_shuffle is not Shuffle.SEED:
                seed = self._args.seed
            random_shuffle = True
            seed_change_epoch = True
        initial_fill = 1024
        if self._args.shuffle_size > 0:
            initial_fill = self._args.shuffle_size
        prefetch_size = 1
        if self._args.prefetch_size > 0:
            prefetch_size = self._args.prefetch_size

        stick_to_shard = True
        if seed_change_epoch:
            stick_to_shard = False
        images, labels = fn.readers.file(files=self._file_list, num_shards=self._args.comm_size, 
                                         prefetch_queue_depth=prefetch_size, 
                                         initial_fill=initial_fill, random_shuffle=random_shuffle, 
                                         shuffle_after_epoch=seed_change_epoch, 
                                         stick_to_shard=stick_to_shard, pad_last_batch=True, 
                                         dont_use_mmap=self._args.dont_use_mmap)
        images = fn.decoders.image(images, device='cpu')
        images = fn.python_function(images, function=self.preprocess, num_outputs=1)
        dataset = fn.python_function(images, function=self.resize, num_outputs=1)
        return dataset

    @dlp.log
    def finalize(self):
        pass

    def is_index_based(self):
        return False

    def is_iterator_based(self):
        return True
