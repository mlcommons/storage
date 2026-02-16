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
import io

from dlio_benchmark.data_generator.data_generator import DataGenerator

from dlio_benchmark.utils.utility import Profile, progress, gen_random_tensor
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)

"""
Generator for creating data in NPY format for S3 Storage.
"""
class NPYGeneratorS3(DataGenerator):
    def __init__(self):
        super().__init__()

    @dlp.log
    def generate(self):
        """
        Generator for creating data in NPY format of 3d dataset.
        """
        super().generate()
        np.random.seed(10)
        rng = np.random.default_rng()
        dim = self.get_dimension(self.total_files_to_generate)
        for i in dlp.iter(range(self.my_rank, int(self.total_files_to_generate), self.comm_size)):
            dim_ = dim[2*i]
            if isinstance(dim_, list):
                records = gen_random_tensor(shape=(*dim_, self.num_samples), dtype=self._args.record_element_dtype, rng=rng)
            else:
                dim1 = dim_
                dim2 = dim[2*i+1]
                records = gen_random_tensor(shape=(dim1, dim2, self.num_samples), dtype=self._args.record_element_dtype, rng=rng)

            out_path_spec = self.storage.get_uri(self._file_list[i])
            progress(i+1, self.total_files_to_generate, "Generating NPY Data")
            buffer = io.BytesIO()
            np.save(buffer, records)
            self.storage.put_data(out_path_spec, buffer)
        np.random.seed()
