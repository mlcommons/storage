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
from dlio_benchmark.common.enumerations import StorageType
from dlio_benchmark.common.error_code import ErrorCodes
import os

class StorageFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_storage(storage_type, namespace, framework=None):
        if storage_type == StorageType.LOCAL_FS or storage_type == StorageType.DIRECT_FS:
            return FileStorage(namespace, framework)
        elif storage_type == StorageType.S3:
            from dlio_benchmark.common.enumerations import FrameworkType
            if framework == FrameworkType.PYTORCH:
                # Allow testing both implementations via environment variable
                # DLIO_S3_IMPLEMENTATION=dpsi - use dpsi's architecture (bucket+key separation)
                # DLIO_S3_IMPLEMENTATION=mlp (default) - use mlp-storage's multi-library architecture
                impl = os.environ.get("DLIO_S3_IMPLEMENTATION", "mlp").lower()
                
                if impl == "dpsi":
                    print(f"[StorageFactory] Using dpsi S3 implementation (bucket+key architecture)")
                    from dlio_benchmark.storage.s3_torch_storage_dpsi import S3PyTorchConnectorStorage
                    return S3PyTorchConnectorStorage(namespace, framework)
                else:
                    print(f"[StorageFactory] Using mlp-storage S3 implementation (multi-library, URI-based)")
                    from dlio_benchmark.storage.s3_torch_storage import S3PyTorchConnectorStorage
                    return S3PyTorchConnectorStorage(namespace, framework)
            return S3Storage(namespace, framework)
        else:
            raise Exception(str(ErrorCodes.EC1001))
