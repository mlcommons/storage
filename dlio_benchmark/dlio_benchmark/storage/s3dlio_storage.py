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
from dlio_benchmark.storage.s3_torch_storage import S3PyTorchConnectorStorage
import os

from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)

class S3DlioStorage(S3PyTorchConnectorStorage):
    """
    Storage APIs for S3 objects using s3dlio library.
    Inherits all initialization and metadata operations from S3PyTorchConnectorStorage,
    but overrides put_data and get_data to use s3dlio for data transfer.
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        # Call parent to get full S3PyTorchConnector initialization
        super().__init__(namespace, framework)
        
        # Import s3dlio here to avoid hard dependency
        try:
            import s3dlio
            self.s3dlio = s3dlio
        except ImportError:
            raise ImportError("s3dlio library not installed. Install with: pip install s3dlio")
        
        # Build S3 URI for s3dlio (functional API, no store object needed)
        bucket_name = self.get_namespace()
        self.s3_uri_base = f"s3://{bucket_name}/"
        
        # Configure s3dlio with endpoint override if provided
        if self.endpoint:
            os.environ["AWS_ENDPOINT_URL_S3"] = self.endpoint

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        """Write data to S3 using s3dlio - overrides parent method"""
        bucket_name = self.get_namespace()
        full_uri = f"s3://{bucket_name}/{id}"
        
        try:
            # s3dlio.put_bytes() is the correct API (not put())
            data_bytes = data.getvalue()
            self.s3dlio.put_bytes(full_uri, data_bytes)
            return None
        except Exception as e:
            self.logger.error(f"Error putting data to {full_uri}: {e}")
            raise

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        """Read data from S3 using s3dlio - overrides parent method"""
        bucket_name = self.get_namespace()
        full_uri = f"s3://{bucket_name}/{id}"
        
        try:
            if offset is not None and length is not None:
                # Range read
                result_bytes = self.s3dlio.get_range(full_uri, offset, length)
            else:
                # Full object read
                result_bytes = self.s3dlio.get(full_uri)
            
            # Return bytes directly (same as parent S3PyTorchConnectorStorage behavior)
            return result_bytes
        except Exception as e:
            self.logger.error(f"Error getting data from {full_uri}: {e}")
            raise
