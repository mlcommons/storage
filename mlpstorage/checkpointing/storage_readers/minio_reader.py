"""MinIO byte-range reader for streaming checkpoint load.

Uses minio.Minio.get_object(bucket, key, offset=, length=) which issues
a single HTTP Range-GET.  Each response is read and immediately released,
so peak RAM = chunk_size bytes regardless of object size.

Client setup mirrors MinIOStorageWriter exactly (same env vars).
"""

import os
import re
from typing import Dict, Any, List, Optional

from .base import StorageReader


class MinIOStorageReader(StorageReader):
    """Chunked byte-range reader using the minio Python SDK."""

    @staticmethod
    def _expand_template(template: str) -> List[str]:
        match = re.search(r'\{(\d+)\.\.\.(\d+)\}', template)
        if not match:
            return [template]
        start, end = int(match.group(1)), int(match.group(2))
        prefix, suffix = template[:match.start()], template[match.end():]
        return [f"{prefix}{i}{suffix}" for i in range(start, end + 1)]

    @staticmethod
    def _detect_endpoint() -> Optional[str]:
        """Mirror endpoint detection from MinIOStorageWriter."""
        uris_str = os.environ.get('S3_ENDPOINT_URIS')
        if uris_str:
            endpoints = [u.strip() for u in uris_str.split(',') if u.strip()]
            if endpoints:
                return endpoints[0]

        template = os.environ.get('S3_ENDPOINT_TEMPLATE')
        if template:
            endpoints = MinIOStorageReader._expand_template(template)
            if endpoints:
                return endpoints[0]

        endpoint_file = os.environ.get('S3_ENDPOINT_FILE')
        if endpoint_file:
            try:
                with open(endpoint_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            return line
            except OSError:
                pass

        return None

    def __init__(self, uri: str, chunk_size: int = None):
        if not uri.startswith('s3://'):
            raise ValueError(f"MinIOStorageReader requires s3:// URI, got: {uri}")

        try:
            from minio import Minio
        except ImportError:
            raise ImportError("minio library required. Install with: pip install minio")

        parts = uri[5:].split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI (expected s3://bucket/key): {uri}")

        self.bucket_name = parts[0]
        self.object_name = parts[1]
        self.uri = uri
        self.total_bytes = 0

        access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        secret_key  = os.environ.get('AWS_SECRET_ACCESS_KEY')
        if not access_key or not secret_key:
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set")

        endpoint = self._detect_endpoint() or os.environ.get('AWS_ENDPOINT_URL') or os.environ.get('S3_ENDPOINT')

        if not endpoint:
            endpoint = 's3.amazonaws.com'
            secure = True
        elif endpoint.startswith('https://'):
            endpoint = endpoint[8:]
            secure = True
        elif endpoint.startswith('http://'):
            endpoint = endpoint[7:]
            secure = False
        else:
            secure = False

        # Support custom CA certificate via AWS_CA_BUNDLE (same env var as s3dlio/boto3).
        http_client = None
        ca_bundle = os.environ.get('AWS_CA_BUNDLE')
        if secure and ca_bundle:
            import urllib3
            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout.DEFAULT_TIMEOUT,
                maxsize=10,
                cert_reqs='CERT_REQUIRED',
                ca_certs=ca_bundle,
                retries=urllib3.Retry(total=5, backoff_factor=0.2,
                                      status_forcelist=[500, 502, 503, 504]),
            )
            print(f"[MinIOReader] TLS: using CA bundle from AWS_CA_BUNDLE={ca_bundle}")

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=os.environ.get('AWS_REGION', 'us-east-1'),
            http_client=http_client,
        )
        print(f"[MinIOReader] endpoint={endpoint}, bucket={self.bucket_name}, key={self.object_name}")

    def read_chunk(self, offset: int, size: int) -> int:
        response = self.client.get_object(
            self.bucket_name, self.object_name,
            offset=offset, length=size,
        )
        try:
            data = response.read()
            nbytes = len(data)
        finally:
            response.close()
            response.release_conn()
        self.total_bytes += nbytes
        # data goes out of scope → freed
        return nbytes

    def close(self) -> Dict[str, Any]:
        return {'backend': 'minio', 'total_bytes': self.total_bytes}
