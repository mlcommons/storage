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
import logging
from dlio_benchmark.utils.utility import utcnow, DLIOMPI

from dlio_benchmark.utils.config import ConfigArguments

from dlio_benchmark.common.enumerations import FormatType, DataLoaderType, StorageType
from dlio_benchmark.common.error_code import ErrorCodes


class ReaderFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_reader(type, dataset_type, thread_index, epoch_number):
        """
        This function set the data reader based on the data format and the data loader specified. 
        """

        _args = ConfigArguments.get_instance()
        if _args.reader_class is not None:
            if DLIOMPI.get_instance().rank() == 0:
                self.logger.info(f"{utcnow()} Running DLIO with custom data loader class {_args.reader_class.__name__}")
            return _args.reader_class(dataset_type, thread_index, epoch_number)
        elif type == FormatType.HDF5:
            if _args.odirect == True:
                raise Exception("Odirect for %s format is not yet supported." %type)
            else:
                from dlio_benchmark.reader.hdf5_reader import HDF5Reader
                return HDF5Reader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.CSV:
            if _args.odirect == True:
                raise Exception("Odirect for %s format is not yet supported." %type)
            else:
                from dlio_benchmark.reader.csv_reader import CSVReader
            return CSVReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.JPEG or type == FormatType.PNG:
            if _args.odirect == True:
                raise Exception("Odirect for %s format is not yet supported." %type)
            elif _args.data_loader == DataLoaderType.NATIVE_DALI:
                from dlio_benchmark.reader.dali_image_reader import DaliImageReader
                return DaliImageReader(dataset_type, thread_index, epoch_number)
            else:
                from dlio_benchmark.reader.image_reader import ImageReader
                return ImageReader(dataset_type, thread_index, epoch_number)   
        elif type == FormatType.NPY:
            if _args.data_loader == DataLoaderType.NATIVE_DALI:
                from dlio_benchmark.reader.dali_npy_reader import DaliNPYReader
                return DaliNPYReader(dataset_type, thread_index, epoch_number)
            else:
                if _args.odirect == True:
                    from dlio_benchmark.reader.npy_reader_odirect import NPYReaderODirect
                    return NPYReaderODirect(dataset_type, thread_index, epoch_number)
                elif _args.storage_type == StorageType.S3:
                    from dlio_benchmark.reader.npy_reader_s3 import NPYReaderS3
                    return NPYReaderS3(dataset_type, thread_index, epoch_number)
                else:
                    from dlio_benchmark.reader.npy_reader import NPYReader
                    return NPYReader(dataset_type, thread_index, epoch_number)                           
        elif type == FormatType.NPZ:
            if _args.data_loader == DataLoaderType.NATIVE_DALI:
                raise Exception("Loading data of %s format is not supported without framework data loader; please use npy format instead." %type)
            else:
                if _args.odirect == True:
                    from dlio_benchmark.reader.npz_reader_odirect import NPZReaderODIRECT
                    return NPZReaderODIRECT(dataset_type, thread_index, epoch_number)         
                elif _args.storage_type == StorageType.S3:
                    from dlio_benchmark.reader.npz_reader_s3 import NPZReaderS3
                    return NPZReaderS3(dataset_type, thread_index, epoch_number)
                else:
                    from dlio_benchmark.reader.npz_reader import NPZReader
                    return NPZReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.TFRECORD:
            if _args.odirect == True:
                raise Exception("O_DIRECT for %s format is not yet supported." %type)
            elif _args.data_loader == DataLoaderType.NATIVE_DALI: 
                from dlio_benchmark.reader.dali_tfrecord_reader import DaliTFRecordReader
                return DaliTFRecordReader(dataset_type, thread_index, epoch_number)
            else:
                from dlio_benchmark.reader.tf_reader import TFReader
                return TFReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.INDEXED_BINARY:
            if _args.odirect == True:
                raise Exception("O_DIRECT for %s format is not yet supported." %type)
            else:
                from dlio_benchmark.reader.indexed_binary_reader import IndexedBinaryReader
                return IndexedBinaryReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.MMAP_INDEXED_BINARY:
            if _args.odirect == True:
                raise Exception("O_DIRECT for %s format is not yet supported." %type)
            else:
                from dlio_benchmark.reader.indexed_binary_mmap_reader import IndexedBinaryMMapReader
                return IndexedBinaryMMapReader(dataset_type, thread_index, epoch_number)
        elif type == FormatType.SYNTHETIC:
            if _args.odirect == True:
                raise Exception("O_DIRECT for %s format is not yet supported." %type)
            else:
                from dlio_benchmark.reader.synthetic_reader import SyntheticReader
                return SyntheticReader(dataset_type, thread_index, epoch_number)

        else:
            raise Exception("Loading data of %s format is not supported without framework data loader" %type)
