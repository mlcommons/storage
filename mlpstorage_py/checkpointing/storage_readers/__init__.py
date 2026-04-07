"""Storage reader backends for streaming checkpoint load.

Mirrors storage_writers/ — each backend issues byte-range reads and
discards each chunk immediately, so peak RAM = chunk_size bytes regardless
of total checkpoint size.

Use StorageReaderFactory.create() to select the appropriate backend.
"""

from .base import StorageReader
from .s3dlio_reader import S3DLIOStorageReader

from typing import Optional, Any


class StorageReaderFactory:
    """Factory for creating storage reader instances."""

    @staticmethod
    def create(
        uri: str,
        backend: Optional[str] = None,
        fadvise_mode: str = 'dontneed',
        **kwargs: Any,
    ) -> StorageReader:
        """Create a storage reader instance.

        Args:
            uri:          Full URI (s3://, file://, etc.) or plain filesystem path.
            backend:      Explicit backend name: 'file', 's3dlio', 'minio',
                          's3torchconnector'.  If None, auto-detects from URI scheme.
            fadvise_mode: Page-cache strategy for the 'file' backend.
                          'dontneed' (default) — drop pages after each read;
                          'sequential' — hint sequential access only;
                          'none' — no hints.
            **kwargs:     Passed to the reader constructor (e.g. chunk_size).

        Returns:
            StorageReader configured for the requested backend.
        """
        if backend:
            if backend == 'file':
                from .file_reader import FileStorageReader
                # Strip file:// prefix if present; reader expects a plain path
                path = uri[7:] if uri.startswith('file://') else uri
                return FileStorageReader(path, fadvise_mode=fadvise_mode, **kwargs)

            elif backend == 'direct_fs':
                # O_DIRECT via s3dlio's direct:// URI — bypasses page cache entirely.
                # No fadvise needed: the kernel never sees the data.
                path = uri
                for prefix in ('direct://', 'file://'):
                    if path.startswith(prefix):
                        path = path[len(prefix):]
                        break
                return S3DLIOStorageReader('direct://' + path, **kwargs)

            elif backend == 's3dlio':
                return S3DLIOStorageReader(uri, **kwargs)

            elif backend == 'minio':
                from .minio_reader import MinIOStorageReader
                return MinIOStorageReader(uri, **kwargs)

            elif backend == 's3torchconnector':
                from .s3torch_reader import S3TorchStorageReader
                return S3TorchStorageReader(uri, **kwargs)

            else:
                raise ValueError(
                    f"Unknown backend: {backend!r}. "
                    f"Supported: file, direct_fs, s3dlio, minio, s3torchconnector"
                )

        # Auto-detect from URI scheme
        if uri.startswith('s3://') or uri.startswith('gs://') or uri.startswith('az://'):
            return S3DLIOStorageReader(uri, **kwargs)

        if uri.startswith('direct://'):
            return S3DLIOStorageReader(uri, **kwargs)

        if uri.startswith('file://') or uri.startswith('/'):
            from .file_reader import FileStorageReader
            path = uri[7:] if uri.startswith('file://') else uri
            return FileStorageReader(path, fadvise_mode=fadvise_mode, **kwargs)

        raise ValueError(
            f"Cannot auto-detect reader backend for URI: {uri!r}. "
            f"Specify backend= explicitly."
        )
