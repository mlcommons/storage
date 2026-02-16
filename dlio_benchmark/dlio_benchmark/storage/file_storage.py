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
from abc import ABC, abstractmethod
from time import time

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
import os
import glob
import shutil

from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)

class FileStorage(DataStorage):
    """
    Storage APIs for creating files.
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.HIERARCHICAL)

    @dlp.log
    def get_uri(self, id):
        return os.path.join(self.namespace.name, id)

    # Namespace APIs
    @dlp.log
    def create_namespace(self, exist_ok=False):
        os.makedirs(self.namespace.name, exist_ok=exist_ok)
        return True

    @dlp.log
    def get_namespace(self):
        return self.namespace.name

    # Metadata APIs
    @dlp.log
    def create_node(self, id, exist_ok=False):
        os.makedirs(self.get_uri(id), exist_ok=exist_ok)
        return True

    @dlp.log
    def get_node(self, id=""):
        path = self.get_uri(id)
        if os.path.exists(path):
            if os.path.isdir(path):
                return MetadataType.DIRECTORY
            else:
                return MetadataType.FILE
        else:
            return None

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        if not use_pattern:
            return os.listdir(self.get_uri(id))
        else:
            format= self.get_uri(id).split(".")[-1]
            upper_case = self.get_uri(id).replace(format, format.upper())
            lower_case = self.get_uri(id).replace(format, format.lower())
            if format != format.lower():
                raise Exception(f"Unknown file format {format}")
            return glob.glob(self.get_uri(id)) + glob.glob(upper_case)


    @dlp.log
    def delete_node(self, id):
        shutil.rmtree(self.get_uri(id))
        return True

    # TODO Handle partial read and writes
    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        with open(self.get_uri(id), "w") as fd:
            fd.write(data)

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        with open(self.get_uri(id), "r") as fd:
            data = fd.read()
        return data
    
    @dlp.log
    def isfile(self, id):
        return os.path.isfile(id)

    def get_basename(self, id):
        return os.path.basename(id)
