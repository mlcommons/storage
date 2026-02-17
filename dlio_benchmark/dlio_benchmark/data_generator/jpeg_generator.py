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
import PIL.Image as im

from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import progress, utcnow
from dlio_benchmark.utils.utility import Profile
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR


dlp = Profile(MODULE_DATA_GENERATOR)

"""
Generator for creating data in JPEG format.
"""
class JPEGGenerator(DataGenerator):
    @dlp.log
    def generate(self):
        """
        Generator for creating data in JPEG format of 3d dataset.
        """
        super().generate()
        np.random.seed(10)
        dim = self.get_dimension(self.total_files_to_generate)
        for i in dlp.iter(range(self.my_rank, int(self.total_files_to_generate), self.comm_size)):
            dim_ = dim[2*i]
            if isinstance(dim_, list):
                dim1 = dim_[0]
                dim2 = dim_[1]
            else:
                dim1 = dim_
                dim2 = dim[2*i+1]
            records = np.random.randint(255, size=(dim1, dim2), dtype=np.uint8)
            if self.my_rank==0:
                self.logger.debug(f"{utcnow()} Dimension of images: {dim1} x {dim2}")
            img = im.fromarray(records)
            if self.my_rank == 0 and i % 100 == 0:
                self.logger.info(f"Generated file {i}/{self.total_files_to_generate}")
            out_path_spec = self.storage.get_uri(self._file_list[i])
            progress(i+1, self.total_files_to_generate, "Generating JPEG Data")
            img.save(out_path_spec, format='JPEG', bits=8)
        np.random.seed()
