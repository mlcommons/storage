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

from dlio_benchmark.common.enumerations import CheckpointMechanismType
from dlio_benchmark.common.error_code import ErrorCodes
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import utcnow, DLIOMPI

class CheckpointingFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_mechanism(checkpoint_mechanism_type):
        _args = ConfigArguments.get_instance()
        if _args.checkpoint_mechanism_class is not None:
            if DLIOMPI.get_instance().rank() == 0:
                _args.logger.info(f"{utcnow()} Running DLIO with custom checkpointing mechanism "
                             f"class {_args.checkpoint_mechanism_class.__name__}")
            return _args.checkpoint_mechanism_class.get_instance()
        elif checkpoint_mechanism_type == CheckpointMechanismType.TF_SAVE:
            from dlio_benchmark.checkpointing.tf_checkpointing import TFCheckpointing
            return TFCheckpointing.get_instance()
        elif checkpoint_mechanism_type == CheckpointMechanismType.PT_SAVE:
            from dlio_benchmark.checkpointing.pytorch_checkpointing import PyTorchCheckpointing
            return PyTorchCheckpointing.get_instance()
        elif checkpoint_mechanism_type == CheckpointMechanismType.PT_S3_SAVE:
            from dlio_benchmark.checkpointing.pytorch_s3_checkpointing import PyTorchS3Checkpointing
            return PyTorchS3Checkpointing.get_instance()
        else:
            raise Exception(str(ErrorCodes.EC1005))
