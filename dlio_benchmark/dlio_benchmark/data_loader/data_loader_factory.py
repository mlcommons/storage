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
from dlio_benchmark.utils.config import ConfigArguments

from dlio_benchmark.utils.utility import utcnow, DLIOMPI

from dlio_benchmark.common.enumerations import DataLoaderType
from dlio_benchmark.common.error_code import ErrorCodes


class DataLoaderFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_loader(type, format_type, dataset_type, epoch):
        """
        This function set the data reader based on the data format and the data loader specified.
        """
        _args = ConfigArguments.get_instance()
        if _args.data_loader_class is not None:
            if DLIOMPI.get_instance().rank() == 0:
                _args.logger.info(f"{utcnow()} Running DLIO with custom data loader class {_args.data_loader_class.__name__}")
            return _args.data_loader_class(format_type, dataset_type, epoch)
        elif type == DataLoaderType.PYTORCH:
            from dlio_benchmark.data_loader.torch_data_loader import TorchDataLoader
            return TorchDataLoader(format_type, dataset_type, epoch)
        elif type == DataLoaderType.TENSORFLOW:
            from dlio_benchmark.data_loader.tf_data_loader import TFDataLoader
            return TFDataLoader(format_type, dataset_type, epoch)
        elif type == DataLoaderType.DALI:
            from dlio_benchmark.data_loader.dali_data_loader import DaliDataLoader
            return DaliDataLoader(format_type, dataset_type, epoch)
        elif type == DataLoaderType.NATIVE_DALI:
            from dlio_benchmark.data_loader.native_dali_data_loader import NativeDaliDataLoader
            return NativeDaliDataLoader(format_type, dataset_type, epoch)
        elif type == DataLoaderType.SYNTHETIC:
            from dlio_benchmark.data_loader.synthetic_data_loader import SyntheticDataLoader
            return SyntheticDataLoader(format_type, dataset_type, epoch)
        else:
            if DLIOMPI.get_instance().rank() == 0:
                print("Data Loader %s not supported or plugins not found" % type)
                raise Exception(str(ErrorCodes.EC1004))
