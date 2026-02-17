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
from dlio_benchmark.storage.file_storage import FileStorage
from dlio_benchmark.storage.s3_storage import S3Storage
from dlio_benchmark.common.enumerations import StorageType, StorageLibrary
from dlio_benchmark.common.error_code import ErrorCodes
import os

class StorageFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_storage(storage_type, namespace, framework=None, storage_library=None):
        """
        Create appropriate storage handler based on storage type and library.
        
        Args:
            storage_type: StorageType enum value (LOCAL_FS, PARALLEL_FS, S3)
            namespace: Storage root path (bucket name or file path)
            framework: Framework type (PyTorch, TensorFlow, etc.)
            storage_library: StorageLibrary enum (s3torchconnector, s3dlio, minio) - only for S3
        """
        # Normalize storage_type to enum if it's a string
        if isinstance(storage_type, str):
            storage_type = StorageType(storage_type)
        
        # Handle FILE-based storage (local/network filesystem)
        if storage_type in [StorageType.LOCAL_FS, StorageType.PARALLEL_FS]:
            return FileStorage(namespace, framework)
        
        # Handle S3 object storage with multi-library support
        elif storage_type == StorageType.S3:
            # Default to s3torchconnector (dpsi fork baseline)
            if storage_library is None:
                storage_library = StorageLibrary.S3TORCHCONNECTOR
            elif isinstance(storage_library, str):
                storage_library = StorageLibrary(storage_library)
            
            # Route to appropriate storage implementation
            if storage_library == StorageLibrary.S3DLIO:
                from dlio_benchmark.storage.s3dlio_storage import S3DlioStorage
                return S3DlioStorage(namespace, framework)
            
            elif storage_library == StorageLibrary.MINIO:
                from dlio_benchmark.storage.minio_storage import MinioStorage
                return MinioStorage(namespace, framework)
            
            else:  # S3TORCHCONNECTOR (default)
                from dlio_benchmark.storage.s3_torch_storage import S3PyTorchConnectorStorage
                return S3PyTorchConnectorStorage(namespace, framework)
        
        else:
            raise Exception(f"Unsupported storage type: {storage_type} ({ErrorCodes.EC1001})")
