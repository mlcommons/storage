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

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.utils.utility import utcnow, Profile, dft_ai

dlp = Profile(MODULE_DATA_LOADER)

class SyntheticDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch):
        super().__init__(format_type, dataset_type, epoch, DataLoaderType.SYNTHETIC)
        shape = self._args.resized_image.shape
        self.batch = np.zeros((self.batch_size, shape[0], shape[1]))

    @dlp.log
    def read(self, init=False):
        return
    
    @dft_ai.data.item
    def getitem(self):
        return self.batch

    @dlp.log
    def next(self):
        super().next()
        self.logger.debug(f"{utcnow()} Iterating pipelines by {self._args.my_rank} rank ")
        self.read(True)

        step = 1
        dft_ai.dataloader.fetch.start()
        while step < self.num_samples // self.batch_size:
            dft_ai.dataloader.fetch.stop()
            dft_ai.update(step=step)
            step += 1
            yield self.getitem()
            dft_ai.dataloader.fetch.start()

        self.epoch_number += 1
        dft_ai.update(epoch=self.epoch_number)

    @dlp.log
    def finalize(self):
        return