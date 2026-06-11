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
from io import BytesIO

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.storage.s3_storage import S3Storage
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
from urllib.parse import urlparse
import os

from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)


class MinIOAdapter:
    """Adapter to make Minio client compatible with S3Client API"""
    
    def __init__(self, endpoint, access_key, secret_key, region=None, secure=True):
        from minio import Minio
        # Parse endpoint to extract host and determine secure
        if endpoint:
            parsed = urlparse(endpoint if '://' in endpoint else f'http://{endpoint}')
            host = parsed.netloc or parsed.path
            secure = parsed.scheme == 'https' if parsed.scheme else secure
        else:
            host = "localhost:9000"
            
        self.client = Minio(
            host,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region
        )
        
    def get_object(self, bucket_name, object_name, start=None, end=None):
        """Adapter for get_object to match S3Client API"""
        class MinioReader:
            def __init__(self, response):
                self.response = response
                
            def read(self):
                return self.response.read()
                
            def close(self):
                self.response.close()
                self.response.release_conn()
        
        if start is not None and end is not None:
            length = end - start + 1
            response = self.client.get_object(bucket_name, object_name, offset=start, length=length)
        else:
            response = self.client.get_object(bucket_name, object_name)
        return MinioReader(response)
    
    def put_object(self, bucket_name, object_name):
        """Adapter for put_object to match S3Client API"""
        class MinioWriter:
            def __init__(self, client, bucket, obj_name):
                self.client = client
                self.bucket = bucket
                self.obj_name = obj_name
                self.buffer = BytesIO()
                
            def write(self, data):
                if isinstance(data, bytes):
                    self.buffer.write(data)
                else:
                    self.buffer.write(data.encode())
                    
            def close(self):
                self.buffer.seek(0)
                length = len(self.buffer.getvalue())
                self.client.put_object(
                    self.bucket,
                    self.obj_name,
                    self.buffer,
                    length
                )
                self.buffer.close()
        
        return MinioWriter(self.client, bucket_name, object_name)
    
    def list_objects(self, bucket_name, prefix=None):
        """Adapter for list_objects to match S3Client API"""
        class MinioListResult:
            def __init__(self, objects, prefix):
                self.object_info = []
                for obj in objects:
                    obj_info = type('ObjectInfo', (), {'key': obj.object_name})()
                    self.object_info.append(obj_info)
                self.prefix = prefix
        
        objects = self.client.list_objects(bucket_name, prefix=prefix or "", recursive=True)
        # Convert generator to list for iteration
        obj_list = list(objects)
        return [MinioListResult(obj_list, prefix)]


class S3PyTorchConnectorStorage(S3Storage):
    """
    Storage APIs for S3-compatible object storage with multi-library support.
    
    Supports 3 storage libraries via YAML config:
      storage_library: s3dlio           # s3dlio (zero-copy, multi-protocol)  
      storage_library: s3torchconnector # AWS s3torchconnector (default)
      storage_library: minio            # MinIO native SDK
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.FLAT)

        # Access config values from self._args (inherited from DataStorage)
        storage_options = getattr(self._args, "storage_options", {}) or {}
        
        # Get storage library selection (default to s3torchconnector for backward compatibility)
        # Check multiple sources: storage_options dict, env var, or direct config attribute
        if "storage_library" in storage_options:
            storage_library = storage_options["storage_library"]
        elif os.environ.get("STORAGE_LIBRARY"):
            storage_library = os.environ.get("STORAGE_LIBRARY")
        else:
            storage_library = "s3torchconnector"  # default
        self.storage_library = storage_library
        
        print(f"[S3PyTorchConnectorStorage] Using storage library: {storage_library}")
        
        # Get credentials and endpoint config
        self.access_key_id = storage_options.get("access_key_id")
        self.secret_access_key = storage_options.get("secret_access_key")
        self.endpoint = storage_options.get("endpoint_url")
        self.region = storage_options.get("region", self._args.s3_region)
        
        # Object key format configuration:
        # - False/"path": Pass path-only keys (e.g., "path/to/object") - default, works with most APIs
        # - True/"uri": Pass full URIs (e.g., "s3://bucket/path/to/object")
        # Configurable via DLIO_OBJECT_KEY_USE_FULL_URI env var or storage_options
        use_full_uri_str = os.environ.get("DLIO_OBJECT_KEY_USE_FULL_URI", 
                                          storage_options.get("use_full_object_uri", "false"))
        self.use_full_object_uri = use_full_uri_str.lower() in ("true", "1", "yes")
        
        if self.use_full_object_uri:
            print(f"  → Object key format: Full URI (s3://bucket/path/object)")
        else:
            print(f"  → Object key format: Path-only (path/object)")

        # Set environment variables for libraries that use them
        if self.access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self.access_key_id
        if self.secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_access_key

        # Dynamically import and initialize the appropriate library
        if storage_library == "s3dlio":
            print(f"  → s3dlio: Zero-copy multi-protocol (20-30 GB/s)")
            try:
                import s3dlio
                # s3dlio uses native API - no client wrapper needed
                # Just store the module for put_bytes/get_bytes calls
                self.s3_client = None  # Not used for s3dlio
                self._s3dlio = s3dlio
                
            except ImportError as e:
                raise ImportError(
                    f"s3dlio is not installed. "
                    f"Install with: pip install s3dlio\nError: {e}"
                )
                
        elif storage_library == "s3torchconnector":
            print(f"  → s3torchconnector: AWS official S3 connector (5-10 GB/s)")
            try:
                from s3torchconnector._s3client import S3Client, S3ClientConfig
                
                force_path_style_opt = self._args.s3_force_path_style
                if "s3_force_path_style" in storage_options:
                    force_path_style_opt = storage_options["s3_force_path_style"].strip().lower() == "true"
                    
                max_attempts_opt = self._args.s3_max_attempts
                if "s3_max_attempts" in storage_options:
                    try:
                        max_attempts_opt = int(storage_options["s3_max_attempts"])
                    except (TypeError, ValueError):
                        max_attempts_opt = self._args.s3_max_attempts
                        
                s3_client_config = S3ClientConfig(
                    force_path_style=force_path_style_opt,
                    max_attempts=max_attempts_opt,
                )
                
                self.s3_client = S3Client(
                    region=self.region,
                    endpoint=self.endpoint,
                    s3client_config=s3_client_config,
                )
            except ImportError as e:
                raise ImportError(
                    f"s3torchconnector is not installed. "
                    f"Install with: pip install s3torchconnector\nError: {e}"
                )
                
        elif storage_library == "minio":
            print(f"  → minio: MinIO native SDK (10-15 GB/s)")
            try:
                secure = storage_options.get("secure", True)
                self.s3_client = MinIOAdapter(
                    endpoint=self.endpoint,
                    access_key=self.access_key_id,
                    secret_key=self.secret_access_key,
                    region=self.region,
                    secure=secure
                )
            except ImportError as e:
                raise ImportError(
                    f"minio is not installed. "
                    f"Install with: pip install minio\nError: {e}"
                )
        else:
            raise ValueError(
                f"Unknown storage_library: {storage_library}. "
                f"Supported: s3dlio, s3torchconnector, minio"
            )

    @dlp.log
    def get_uri(self, id):
        """
        Construct full S3 URI from bucket (namespace) + object key (id).
        MLP uses URI-based architecture: namespace is bucket, id is object key.
        Returns: s3://bucket/path/to/object
        """
        # Handle both absolute paths (s3://...) and relative paths
        if id.startswith('s3://'):
            return id  # Already a full URI
        return f"s3://{self.namespace.name}/{id.lstrip('/')}"
    
    def _normalize_object_key(self, uri):
        """
        Convert s3:// URI to appropriate format for underlying storage library.
        Returns: (bucket_name, object_key)
        
        If use_full_object_uri=True: object_key is full URI (s3://bucket/path/object)
        If use_full_object_uri=False: object_key is path-only (path/object)
        """
        parsed = urlparse(uri)
        if parsed.scheme != 's3':
            raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")
        
        bucket_name = parsed.netloc
        
        if self.use_full_object_uri:
            # Return full URI as object key
            object_key = uri
        else:
            # Return path-only as object key (strip s3://bucket/ prefix)
            object_key = parsed.path.lstrip('/')
        
        return bucket_name, object_key

    @dlp.log
    def create_namespace(self, exist_ok=False):
        return True

    @dlp.log
    def get_namespace(self):
        return self.get_node(self.namespace.name)

    @dlp.log
    def create_node(self, id, exist_ok=False):
        return super().create_node(self.get_uri(id), exist_ok)

    @dlp.log
    def get_node(self, id=""):
        return super().get_node(self.get_uri(id))

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        # Parse s3://bucket/prefix path
        parsed = urlparse(id)
        if parsed.scheme != 's3':
            raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")
    
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')

        if not use_pattern:
            return self.list_objects(bucket, prefix)
        else:
            ext = prefix.split('.')[-1]
            if ext != ext.lower():
                raise Exception(f"Unknown file format {ext}")

            # Pattern matching: check both lowercase and uppercase extensions
            lower_results = self.list_objects(bucket, prefix)
            upper_prefix = prefix.replace(ext, ext.upper())
            upper_results = self.list_objects(bucket, upper_prefix)

            return lower_results + upper_results

    @dlp.log
    def delete_node(self, id):
        return super().delete_node(self.get_uri(id))

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        if self.storage_library == "s3dlio":
            # Use s3dlio native API - simple put_bytes call
            # id is already full s3:// URI from get_uri()
            payload = data.getvalue() if hasattr(data, 'getvalue') else data
            self._s3dlio.put_bytes(id, payload)
        else:
            # s3torchconnector or minio - use S3Client API
            bucket_name, object_key = self._normalize_object_key(id)
            writer = self.s3_client.put_object(bucket_name, object_key)
            writer.write(data.getvalue())
            writer.close()
        return None

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        if self.storage_library == "s3dlio":
            # Use s3dlio native API - simple get_bytes call
            result = self._s3dlio.get_bytes(id)
            return result
        else:
            # s3torchconnector or minio - use S3Client API
            bucket_name, object_key = self._normalize_object_key(id)

            if offset is not None and length is not None:
                start = offset
                end = offset + length - 1
                reader = self.s3_client.get_object(bucket_name, object_key, start=start, end=end)
            else:
                reader = self.s3_client.get_object(bucket_name, object_key)

            return reader.read()

    @dlp.log
    def list_objects(self, bucket_name, prefix=None):
        paths = []
        try:
            if self.storage_library == "s3dlio":
                # Use s3dlio native list API - takes full URI
                uri = f"s3://{bucket_name}/{prefix.lstrip('/')}" if prefix else f"s3://{bucket_name}/"
                full_uris = self._s3dlio.list(uri)
                # Return relative paths (strip bucket prefix)
                for full_uri in full_uris:
                    if full_uri.startswith(f"s3://{bucket_name}/"):
                        key = full_uri[len(f"s3://{bucket_name}/"):]
                        paths.append(key)
            else:
                # s3torchconnector or minio - use S3Client API
                # Normalize prefix based on use_full_object_uri setting
                if self.use_full_object_uri:
                    # Pass prefix as-is or reconstruct full URI format
                    list_prefix = f"s3://{bucket_name}/{prefix.lstrip('/')}" if prefix else f"s3://{bucket_name}/"
                else:
                    # Pass path-only prefix (default - works with most APIs)
                    list_prefix = prefix.lstrip('/') if prefix else ""
                
                if list_prefix and not list_prefix.endswith('/'):
                    list_prefix += '/'
                
                # Pass normalized prefix to underlying storage library
                obj_stream = self.s3_client.list_objects(bucket_name, list_prefix)

                for list_obj_result in obj_stream:
                    for obj_info in list_obj_result.object_info:
                        key = obj_info.key
                        # Strip the prefix from returned keys to get relative paths
                        if list_prefix and key.startswith(list_prefix):
                            stripped_key = key[len(list_prefix):]
                            paths.append(stripped_key)
                        else:
                            paths.append(key)
        except Exception as e:
            print(f"Error listing objects in bucket '{bucket_name}': {e}")

        return paths

    @dlp.log
    def isfile(self, id):
        return super().isfile(self.get_uri(id))

    def get_basename(self, id):
        return os.path.basename(id)
