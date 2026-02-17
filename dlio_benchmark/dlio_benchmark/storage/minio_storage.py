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
from io import BytesIO

from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)

class MinioStorage(S3PyTorchConnectorStorage):
    """
    Storage APIs for S3 objects using minio library.
    Inherits all initialization and metadata operations from S3PyTorchConnectorStorage,
    but overrides put_data and get_data to use minio for data transfer.
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        # Call parent to get full S3PyTorchConnector initialization
        super().__init__(namespace, framework)
        
        # Import minio here to avoid hard dependency
        try:
            from minio import Minio
            self.Minio = Minio
        except ImportError:
            raise ImportError("minio library not installed. Install with: pip install minio")
        
        # Parse endpoint URL to extract hostname:port and secure flag
        # Minio client expects "hostname:port" format, not full URL
        endpoint_url = self.endpoint
        if not endpoint_url:
            raise ValueError("Endpoint URL is required for minio storage")
        
        if endpoint_url.startswith("https://"):
            endpoint = endpoint_url[8:]
            secure = True
        elif endpoint_url.startswith("http://"):
            endpoint = endpoint_url[7:]
            secure = False
        else:
            # No protocol specified, assume http
            endpoint = endpoint_url
            secure = False
        
        # Initialize minio client
        self.client = self.Minio(
            endpoint,
            access_key=self.access_key_id,
            secret_key=self.secret_access_key,
            secure=secure,
            region="us-east-1"
        )
        
        # Performance tuning parameters
        # Default part_size=0 lets minio auto-calculate (usually 5MB minimum)
        # Increase for better throughput with large objects
        self.part_size = 16 * 1024 * 1024  # 16 MB parts for better performance
        self.num_parallel_uploads = 8  # Increase from default 3 for better PUT speed

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        """Write data to S3 using minio - overrides parent method"""
        bucket_name = self.get_namespace()
        
        try:
            # Convert BytesIO to bytes for minio
            data_bytes = data.getvalue()
            data_stream = BytesIO(data_bytes)
            data_size = len(data_bytes)
            
            # Use put_object with performance tuning
            result = self.client.put_object(
                bucket_name=bucket_name,
                object_name=id,
                data=data_stream,
                length=data_size,
                part_size=self.part_size,
                num_parallel_uploads=self.num_parallel_uploads
            )
            return None
        except Exception as e:
            self.logger.error(f"Error putting data to {bucket_name}/{id}: {e}")
            raise

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        """Read data from S3 using minio - overrides parent method"""
        bucket_name = self.get_namespace()
        
        try:
            if offset is not None and length is not None:
                # Range read - minio supports range via get_object parameters
                response = self.client.get_object(
                    bucket_name=bucket_name,
                    object_name=id,
                    offset=offset,
                    length=length
                )
            else:
                # Full object read
                response = self.client.get_object(
                    bucket_name=bucket_name,
                    object_name=id
                )
            
            # Read all data from response stream
            result_bytes = response.read()
            response.close()
            response.release_conn()
            
            # Return bytes directly (same as parent S3PyTorchConnectorStorage behavior)
            return result_bytes
        except Exception as e:
            self.logger.error(f"Error getting data from {bucket_name}/{id}: {e}")
            raise
