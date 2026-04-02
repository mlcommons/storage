"""s3torchconnector streaming reader for checkpoint load.

Uses S3Client._get_object_stream(bucket, key, start, end) directly — the same
native CRT call that backs RangedS3Reader internally — but held open across
chunk iterations so each worker issues exactly ONE HTTP connection per block.

Key design facts (from the installed library source at
s3torchconnector/s3reader/ranged.py):

  - RangedS3Reader._read_unbuffered() calls self._get_stream(start, end) on
    EVERY read() call, opening a brand-new HTTP range-GET each time.
    range_based(buffer_size=0) therefore gives one request per read() call,
    which is why we saw 0.07 GB/s per worker regardless of chunk size.

  - _get_object_stream(bucket, key, start, end) returns a GetObjectStream
    (native Rust/CRT iterator) that streams [start, end) over ONE connection.
    Iterating it yields bytes chunks (~8 MB each from the CRT).

  - Each chunk is released immediately after len() — the caller holds no
    large buffers. Peak RAM per stream ≈ one CRT chunk (~8–64 MB). With 8
    workers: ~64–512 MB total, independent of object or block size.

  - S3Client holds a MountpointS3Client with an internal connection pool.
    One S3Client per worker is sufficient; connections are reused by the CRT.

RAM budget:
  stream_block path (parallel):    n_workers × ~32 MB ≈ 256 MB  (8 workers)
  read_chunk path (serial):        ~8 MB (one leftover CRT chunk at a time)
  Both are constant regardless of total object size (16 GB or 759 GB).
"""

import os
import re
from typing import Dict, Any, List, Optional

from .base import StorageReader


class S3TorchStorageReader(StorageReader):
    """Streaming byte-range reader using s3torchconnector's native CRT client."""

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
        uris_str = os.environ.get('S3_ENDPOINT_URIS')
        if uris_str:
            endpoints = [u.strip() for u in uris_str.split(',') if u.strip()]
            if endpoints:
                return endpoints[0]
        template = os.environ.get('S3_ENDPOINT_TEMPLATE')
        if template:
            endpoints = S3TorchStorageReader._expand_template(template)
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

    def __init__(self, uri: str, chunk_size: int = 32 * 1024 * 1024):
        if not uri.startswith('s3://'):
            raise ValueError(f"S3TorchStorageReader requires s3:// URI, got: {uri}")

        try:
            from s3torchconnector._s3client import S3Client, S3ClientConfig
        except ImportError:
            raise ImportError(
                "s3torchconnector library required. Install with: pip install s3torchconnector"
            )

        parts = uri[5:].split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI (expected s3://bucket/key): {uri}")

        self.bucket_name = parts[0]
        self.object_key  = parts[1]
        self.uri         = uri
        self.chunk_size  = chunk_size
        self.total_bytes = 0

        region   = os.environ.get('AWS_REGION', 'us-east-1')
        endpoint = (self._detect_endpoint()
                    or os.environ.get('AWS_ENDPOINT_URL')
                    or os.environ.get('S3_ENDPOINT'))

        s3_client_config = S3ClientConfig(
            force_path_style=bool(endpoint),
            max_attempts=3,
        )
        self.s3_client = S3Client(
            region=region,
            endpoint=endpoint,
            s3client_config=s3_client_config,
        )

        # Streaming state for the read_chunk() serial/fallback path.
        # The GetObjectStream is opened lazily and kept alive across
        # sequential read_chunk() calls — one HTTP connection per run.
        self._stream_iter = None  # iter() over open GetObjectStream, or None
        self._position    = 0     # current logical read position
        self._leftover    = b''   # bytes pulled from CRT not yet returned

        print(f"[S3TorchReader] endpoint={endpoint or 'AWS S3'}, "
              f"bucket={self.bucket_name}, key={self.object_key} [streaming]")

    # ------------------------------------------------------------------
    # stream_block — optimal path for the parallel worker
    # ------------------------------------------------------------------
    def stream_block(self, start: int, end: int) -> int:
        """Stream bytes [start, end) via a single CRT range-GET.

        Opens ONE HTTP connection for the block [start, end) and iterates
        the native CRT chunks until complete. Each chunk is discarded
        immediately after counting. Peak RAM ≈ one CRT chunk (~8–64 MB),
        independent of how large [start, end) is.

        Args:
            start: First byte (inclusive).
            end:   Last byte (exclusive).

        Returns:
            Number of bytes received.
        """
        total = 0
        for chunk in self.s3_client._get_object_stream(
            self.bucket_name, self.object_key, start, end
        ):
            total += len(chunk)
            # chunk drops here → RAM freed immediately
        self.total_bytes += total
        return total

    # ------------------------------------------------------------------
    # read_chunk — serial / fallback path
    # ------------------------------------------------------------------
    def _open_stream(self, offset: int) -> None:
        """Open a streaming connection from `offset` to end-of-object."""
        self._stream_iter = iter(self.s3_client._get_object_stream(
            self.bucket_name, self.object_key, offset, None
        ))
        self._position = offset
        self._leftover = b''

    def _close_stream(self) -> None:
        """Drop the stream — CRT releases the underlying connection."""
        self._stream_iter = None
        self._leftover    = b''

    def read_chunk(self, offset: int, size: int) -> int:
        """Read exactly `size` bytes starting at `offset`.

        A streaming connection is opened the first time and kept alive for
        all subsequent calls with adjacent offsets, so a sequential loop of
        read_chunk() calls uses exactly ONE HTTP connection for the full run.

        Returns:
            Number of bytes read (may be < size at end-of-object).
        """
        if self._stream_iter is None or offset != self._position:
            self._close_stream()
            self._open_stream(offset)

        needed = size

        # Consume leftover bytes from the previous CRT chunk first.
        if self._leftover:
            if len(self._leftover) >= needed:
                self._position   += needed
                self.total_bytes += needed
                self._leftover    = self._leftover[needed:]
                return needed
            # Leftover is smaller than needed; account for it, then pull more.
            needed        -= len(self._leftover)
            self._leftover = b''

        # Pull CRT chunks until we have `size` bytes or hit EOF.
        collected = size - needed   # bytes already from leftover
        while needed > 0:
            chunk = next(self._stream_iter, None)
            if chunk is None:
                break               # EOF
            if len(chunk) > needed:
                self._leftover = chunk[needed:]
                collected     += needed
                needed         = 0
            else:
                collected += len(chunk)
                needed    -= len(chunk)

        self._position   += collected
        self.total_bytes += collected
        return collected

    def close(self) -> Dict[str, Any]:
        self._close_stream()
        return {'backend': 's3torchconnector', 'total_bytes': self.total_bytes}
        return {'backend': 's3torchconnector', 'total_bytes': self.total_bytes}
