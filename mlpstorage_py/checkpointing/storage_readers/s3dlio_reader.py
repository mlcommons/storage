"""s3dlio byte-range reader for streaming checkpoint load.

Uses s3dlio.get_range(uri, offset, length) which:
  - Issues a single HTTP Range-GET request
  - Returns a BytesView (zero-copy view into s3dlio's internal buffer)
  - The BytesView is freed when it goes out of scope

Peak RAM per call = chunk_size bytes, regardless of object size.
"""

from typing import Dict, Any
from .base import StorageReader


class S3DLIOStorageReader(StorageReader):
    """Chunked reader using s3dlio.get_range() byte-range GETs.

    Credentials are picked up from the environment (AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL) exactly as the writer does —
    no extra setup required.
    """

    def __init__(self, uri: str, chunk_size: int = None):
        try:
            import s3dlio
            self.s3dlio = s3dlio
        except ImportError:
            raise ImportError("s3dlio not available. Install with: pip install s3dlio")

        self.uri = uri
        self.total_bytes = 0
        print(f"[S3DLIOReader] uri={uri}")

    def read_chunk(self, offset: int, size: int) -> int:
        data = self.s3dlio.get_range(self.uri, offset, size)
        nbytes = len(data)
        self.total_bytes += nbytes
        # data (BytesView) goes out of scope here — memory freed by s3dlio
        return nbytes

    def close(self) -> Dict[str, Any]:
        return {'backend': 's3dlio', 'total_bytes': self.total_bytes}
