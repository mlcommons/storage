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
from dlio_benchmark.framework.framework_factory import FrameworkFactory
from dlio_benchmark.utils.config import ConfigArguments

class Namespace:
    def __init__(self, name, type):
        self.name = name
        self.type = type

class DataStorage(ABC):
    def __init__(self, framework=None):
        self._args = ConfigArguments.get_instance()
        self.logger = self._args.logger
        if framework is not None:
            self.framework = FrameworkFactory().get_framework(self._args.framework, profiling=False)
            self.is_framework_nativeio_available = self.framework.is_nativeio_available()
        else:
            self.framework = None
            self.is_framework_nativeio_available = False

    @abstractmethod
    def get_uri(self, id):
        """
            This method returns URI of an id based on the implemented file system.
            eg: For a file in S3, s3:// has to be prefixed to the file name.
            eg: For a file in hdfs, hdfs:// has to be prefixed to the file name.
        """
        pass

   
    # Namespace APIs
    @abstractmethod
    def create_namespace(self, exist_ok=False):
        """
            This method creates the namespace for the storage which refers to the 
            mount point of the storage. Eg: For files, namespace refers to the root directoy
            where input and checkpoint directories are created. For Objects, namespace refers
            to the bucket where input and checkpoint directories are created.
        """
        pass

    @abstractmethod
    def get_namespace(self):
        """
            This method returns the namespace of the storage.
        """
        pass

    # Metadata APIs
    @abstractmethod
    def create_node(self, id, exist_ok=False):
        """
            This method creates a node within the storage namespace. 
            For files/objects, nodes refer to the subdirectories.
        """
        if self.is_framework_nativeio_available:
            return self.framework.create_node(id, exist_ok)
        return True

    @abstractmethod
    def get_node(self, id):
        """
            This method returns the node info for a specific node id. 
            For Files/Objects, it returns node type if node is a
            file or directory
        """
        if self.is_framework_nativeio_available:
            return self.framework.get_node(id)
        return None

    @abstractmethod
    def walk_node(self, id, use_pattern=False):
        """
            This method lists the sub nodes under the specified node
        """
        if self.is_framework_nativeio_available:
            return self.framework.walk_node(id, use_pattern)
        return None

    @abstractmethod
    def delete_node(self, id):
        """
            This method deletes a specified node
        """
        if self.is_framework_nativeio_available:
            return self.framework.delete_node(id)
        return False

    
    # Data APIs
    def put_data(self, id, data, offset=None, length=None):
        """
            This method adds data content to a node.
            eg: For files, this method writes data to a file.
                For objects, this method writes data to a object
        """
        if self.is_framework_nativeio_available:
            return self.framework.put_data(id, data, offset, length)
        return False
    
    def get_data(self, id, data, offset=None, length=None):
        """
            This method retrieves data content of a node.
            eg: For files, this method returns file data.
                For objects, this method returns object data.
        """
        if self.is_framework_nativeio_available:
            return self.framework.get_data(id, data, offset, length)
        return None

    def isfile(self, id):
        """
            This method checks if the given path is a file
        """
        if self.is_framework_nativeio_available:
            return self.framework.isfile(id)
        return None
