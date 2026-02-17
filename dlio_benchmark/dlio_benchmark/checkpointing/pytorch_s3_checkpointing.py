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
import os
import torch
from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
from dlio_benchmark.checkpointing.pytorch_checkpointing import PyTorchCheckpointing
from dlio_benchmark.utils.utility import Profile, dft_ai

from dlio_benchmark.common.constants import MODULE_CHECKPOINT

dlp = Profile(MODULE_CHECKPOINT)

class PyTorchS3Checkpointing(PyTorchCheckpointing):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if PyTorchS3Checkpointing.__instance is None:
            PyTorchS3Checkpointing.__instance = PyTorchS3Checkpointing()
        return PyTorchS3Checkpointing.__instance

    @dft_ai.checkpoint.capture
    def save_state(self, suffix, state, fsync = False):
        name = f"s3://{self.get_name(suffix)}"
        # Save checkpoint to S3
        with self.checkpoint_storage.s3_checkpoint.writer(name) as writer:
            torch.save(state, writer)

    @dft_ai.checkpoint.restart
    def load_state(self, suffix, state):
        name = self.get_name(suffix)
        state = dict() # clear up
        # Load checkpoint from S3
        with self.checkpoint_storage.s3_checkpoint.reader(name) as reader:
            state = torch.load(reader)
        self.logger.debug(f"checkpoint state loaded: {state}")
        assert(len(state.keys())>0)

    @dlp.log
    def save_checkpoint(self, epoch, step_number):
        super().save_checkpoint(epoch, step_number)

    @dlp.log
    def load_checkpoint(self, epoch, step_number):
        super().load_checkpoint(epoch, step_number)

    @dlp.log
    def finalize(self):
        super().finalize()

    def get_name(self, suffix):
        return f"{self.checkpoint_storage.get_namespace()}/{self.args.checkpoint_folder}/{suffix}.{self.ext}"