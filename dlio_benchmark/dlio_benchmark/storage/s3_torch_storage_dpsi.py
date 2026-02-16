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

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.storage.s3_storage import S3Storage
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
import os
from s3torchconnector._s3client import S3Client, S3ClientConfig
from s3torchconnector import S3Checkpoint
import torch

from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)

class S3PyTorchConnectorStorage(S3Storage):
    """
    Storage APIs for S3 objects.
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(namespace, framework)
        # Access config values from self._args (inherited from DataStorage)
        storage_options = getattr(self._args, "storage_options", {}) or {}
        # Build connector config, possibly with config overrides
        max_attempts_opt = self._args.s3_max_attempts
        if "s3_max_attempts" in storage_options:
            try:
                max_attempts_opt = int(storage_options["s3_max_attempts"])
            except (TypeError, ValueError):
                max_attempts_opt = self._args.s3_max_attempt
        self.s3_client_config = S3ClientConfig(
            force_path_style=self.force_path_style,
            max_attempts=max_attempts_opt,
        )

        # Initialize the S3Client instance
        self.s3_client = S3Client(
            region=self.region,
            endpoint=self.endpoint,
            s3client_config=self.s3_client_config,
        )

        self.s3_checkpoint = S3Checkpoint(
            region=self.region,
            endpoint=self.endpoint,
            s3client_config=self.s3_client_config,
        )

    @dlp.log
    def get_uri(self, id):
        return id

    @dlp.log
    def create_namespace(self, exist_ok=False):
        self.logger.info(f"skipping create S3 bucket namespace, not implemented: {self.namespace.name}, exist_ok: {exist_ok}")
        return True

    @dlp.log
    def create_node(self, id, exist_ok=False):
        return super().create_node(self.get_uri(id), exist_ok)

    @dlp.log
    def get_node(self, id=""):
        return super().get_node(self.get_uri(id))

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        if not use_pattern:
            return self.list_objects(id)
        else:
            ext = id.split('.')[-1]
            if ext != ext.lower():
                raise Exception(f"Unknown file format {ext}")

            # Pattern matching: check both lowercase and uppercase extensions
            lower_results = self.list_objects(id)
            upper_prefix = id.replace(ext, ext.upper())
            upper_results = self.list_objects(upper_prefix)

            return lower_results + upper_results

    @dlp.log
    def delete_node(self, id):
        return super().delete_node(self.get_uri(id))

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        bucket_name = self.get_namespace()
        writer = self.s3_client.put_object(bucket_name, id)
        writer.write(data.getvalue())
        writer.close()
        return None

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        obj_name = id  # or just s3_key = id
        bucket_name = self.get_namespace()

        if offset is not None and length is not None:
            start = offset
            end = offset + length - 1
            reader = self.s3_client.get_object(bucket_name, obj_name, start=start, end=end)
        else:
            reader = self.s3_client.get_object(bucket_name, obj_name)

        return reader.read()        

    @dlp.log
    def list_objects(self, prefix=None):
        paths = []
        # list_objects returns an iterable stream of ObjectInfo
        prefix = prefix.lstrip("/") + '/'
        obj_stream = self.s3_client.list_objects(self.get_namespace(), prefix or "")

        for list_obj_result in obj_stream:
            for obj_info in list_obj_result.object_info:
                key = obj_info.key
                if prefix:
                    stripped_key = key[len(prefix):] if key.startswith(prefix) else key
                    paths.append(stripped_key)
                else:
                    paths.append(key)

        return paths

    @dlp.log
    def isfile(self, id):
        return super().isfile(self.get_uri(id))
