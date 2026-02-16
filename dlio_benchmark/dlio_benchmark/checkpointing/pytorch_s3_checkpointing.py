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
import ctypes
from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
from dlio_benchmark.checkpointing.pytorch_checkpointing import PyTorchCheckpointing
from dlio_benchmark.utils.utility import Profile, dft_ai

from dlio_benchmark.common.constants import MODULE_CHECKPOINT
from s3torchconnector import S3Checkpoint, S3ClientConfig

dlp = Profile(MODULE_CHECKPOINT)

class PyTorchS3Checkpointing(PyTorchCheckpointing):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if PyTorchS3Checkpointing.__instance is None:
            PyTorchS3Checkpointing.__instance = PyTorchS3Checkpointing()
        return PyTorchS3Checkpointing.__instance

    @dft_ai.checkpoint.init
    def __init__(self):
        BaseCheckpointing.__init__(self, "pts3")

        # Access config values from self.args (inherited from BaseCheckpointing)
        storage_options = getattr(self.args, "storage_options", {}) or {}

        self.access_key_id = storage_options.get("access_key_id")
        self.secret_access_key = storage_options.get("secret_access_key")
        self.endpoint = storage_options.get("endpoint_url")
        self.region = storage_options.get("region", self.args.s3_region)

        if self.access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self.access_key_id
        if self.secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_access_key

        # Build connector config, possibly with config overrides
        force_path_style_opt = self.args.s3_force_path_style
        if "s3_force_path_style" in storage_options:
            force_path_style_opt = storage_options["s3_force_path_style"].strip().lower() == "true"
        max_attempts_opt = self.args.s3_max_attempts
        if "s3_max_attempts" in storage_options:
            try:
                max_attempts_opt = int(storage_options["s3_max_attempts"])
            except (TypeError, ValueError):
                max_attempts_opt = self.args.s3_max_attempt
        self.s3_client_config = S3ClientConfig(
            force_path_style=force_path_style_opt,
            max_attempts=max_attempts_opt,
        )

        # Initialize the S3Checkpoint instance
        self.s3_checkpoint = S3Checkpoint(
            region=self.region,
            endpoint=self.endpoint,
            s3client_config=self.s3_client_config,
        )

    @dft_ai.checkpoint.capture
    def save_state(self, suffix, state, fsync = False):
        name = self.get_name(suffix)
        # Save checkpoint to S3
        with self.s3_checkpoint.writer(name) as writer:
            torch.save(state, writer)

    @dft_ai.checkpoint.restart
    def load_state(self, suffix, state):
        name = self.get_name(suffix)
        state = dict() # clear up
        # Load checkpoint from S3
        with self.s3_checkpoint.reader(name) as reader:
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

