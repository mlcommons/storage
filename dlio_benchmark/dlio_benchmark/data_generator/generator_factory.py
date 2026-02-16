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
from dlio_benchmark.utils.config import ConfigArguments

from dlio_benchmark.common.enumerations import FormatType, StorageType
from dlio_benchmark.common.error_code import ErrorCodes

class GeneratorFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_generator(type):
        _args = ConfigArguments.get_instance()
        if type == FormatType.TFRECORD:
            from dlio_benchmark.data_generator.tf_generator import TFRecordGenerator
            return TFRecordGenerator()
        elif type == FormatType.HDF5:
            from dlio_benchmark.data_generator.hdf5_generator import HDF5Generator
            return HDF5Generator()
        elif type == FormatType.CSV:
            from dlio_benchmark.data_generator.csv_generator import CSVGenerator
            return CSVGenerator()
        elif type == FormatType.NPZ:
            if _args.storage_type == StorageType.S3:
                from dlio_benchmark.data_generator.npz_generator_s3 import NPZGeneratorS3
                return NPZGeneratorS3()
            else:
                from dlio_benchmark.data_generator.npz_generator import NPZGenerator
                return NPZGenerator()
        elif type == FormatType.NPY:
            if _args.storage_type == StorageType.S3:
                from dlio_benchmark.data_generator.npy_generator_s3 import NPYGeneratorS3
                return NPYGeneratorS3()
            else:
                from dlio_benchmark.data_generator.npy_generator import NPYGenerator
                return NPYGenerator()            
        elif type == FormatType.JPEG:
            from dlio_benchmark.data_generator.jpeg_generator import JPEGGenerator
            return JPEGGenerator()
        elif type == FormatType.PNG:
            from dlio_benchmark.data_generator.png_generator import PNGGenerator
            return PNGGenerator()
        elif type == FormatType.SYNTHETIC:
            from dlio_benchmark.data_generator.synthetic_generator import SyntheticGenerator
            return SyntheticGenerator()
        elif type == FormatType.INDEXED_BINARY or type == FormatType.MMAP_INDEXED_BINARY:
            from dlio_benchmark.data_generator.indexed_binary_generator import IndexedBinaryGenerator
            return IndexedBinaryGenerator()
        else:
            raise Exception(str(ErrorCodes.EC1001))
