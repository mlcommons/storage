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
from time import time

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
import os

from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)


class S3Storage(DataStorage):
    """
    Storage APIs for creating files.
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        if namespace is None or namespace.strip() == "":
            raise ValueError("Namespace cannot be None or empty for S3Storage")
        self.namespace = Namespace(namespace, NamespaceType.FLAT)
        # Access config values from self._args (inherited from DataStorage)
        storage_options = getattr(self._args, "storage_options", {}) or {}
        self.access_key_id = storage_options.get("access_key_id")
        self.secret_access_key = storage_options.get("secret_access_key")
        self.endpoint = storage_options.get("endpoint_url")
        self.region = storage_options.get("region", self._args.s3_region)

        if self.access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self.access_key_id
        if self.secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_access_key

        # Build connector config, possibly with config overrides
        if "s3_force_path_style" in storage_options:
            self.force_path_style = storage_options["s3_force_path_style"]
        else:
            self.force_path_style = True

    @dlp.log
    def get_namespace(self):
        return self.namespace.name