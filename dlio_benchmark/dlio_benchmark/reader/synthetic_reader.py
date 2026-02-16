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

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.common.enumerations import DatasetType
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, dft_ai

dlp = Profile(MODULE_DATA_READER)


class SyntheticReader(FormatReader):
    """
    Reader for Synethic dataset
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

    @dlp.log
    def open(self, filename):
        super().open(filename)

    @dlp.log
    def close(self, filename):
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)

    @dlp.log
    def next(self):
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        step = 1
        while True:
            dft_ai.data.item.start()
            batch = []
            for i in range(self.batch_size):
                batch.append(self._args.resized_image)
            dft_ai.data.item.stop()
            yield batch
            step += 1
            if step > total:
                break

    @dft_ai.data.item
    def read_index(self, image_idx, step):
        dft_ai.update(step=step)
        return self._args.resized_image

    @dlp.log
    def finalize(self):
        return super().finalize()
    
    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True

