"""Storage writer backends for streaming checkpoints.

This package provides unified interfaces to multiple storage systems:
- Local filesystem (with optional O_DIRECT)
- s3dlio multi-protocol (S3, Azure, GCS, file, direct)
- s3torchconnector (AWS S3-specific)
- MinIO S3-compatible storage

Note: Azure Blob Storage is supported exclusively via s3dlio (az:// URIs).

Use StorageWriterFactory.create() to automatically select the appropriate
backend based on URI scheme or explicit backend name.
"""

import os

from .base import StorageWriter
from .file_writer import FileStorageWriter
from .s3dlio_writer import S3DLIOStorageWriter

from typing import Optional, Any


CHECKPOINT_URI_SCHEME_ENV = 'MLPSTORAGE_CHECKPOINT_URI_SCHEME'


def _normalize_checkpoint_uri(uri: str) -> str:
    """Reconstruct an object-storage URI when mlpstorage stripped the scheme.

    Issue #583: ``CheckpointingBenchmark.add_checkpoint_params`` strips the
    URI scheme from ``checkpoint_folder`` so DLIO's
    ``ObjStoreLibStorage._preflight`` (which prepends the scheme itself,
    the #392 / #459 pattern) doesn't produce ``s3://s3://…``. The writer
    process — forked from DLIO, inheriting env — reads
    ``MLPSTORAGE_CHECKPOINT_URI_SCHEME`` to put the scheme back before
    dispatch. Without this bridge the writer would silently fall back to
    the local FileStorageWriter on a scheme-less path, a data integrity
    failure rather than a loud one.

    No-op when the URI already has a scheme OR the env var is unset/empty.
    """
    if '://' in uri:
        return uri
    scheme = os.environ.get(CHECKPOINT_URI_SCHEME_ENV)
    if not scheme:
        return uri
    return f'{scheme}://{uri}'


class StorageWriterFactory:
    """Factory for creating storage writer instances based on URI or explicit backend."""

    @staticmethod
    def create(
        uri_or_path: str,
        backend: Optional[str] = None,
        use_direct_io: bool = False,
        fadvise_mode: str = 'none',
        **kwargs: Any
    ) -> StorageWriter:
        """Create a storage writer instance.
        
        Args:
            uri_or_path: URI or file path (file://, s3://, az://, gs://, direct://, or path)
            backend: Explicit backend name ('file', 's3dlio', 's3torchconnector', 'minio')
                    If None, auto-detects from URI scheme
                    Note: For Azure (az://), use backend='s3dlio'
            use_direct_io: Enable O_DIRECT for file:// backend (requires aligned buffers)
            use_fadvise: Use posix_fadvise hints to bypass page cache (default: True)
            **kwargs: Backend-specific options
            
        Returns:
            StorageWriter instance configured for the specified backend
            
        Raises:
            ValueError: If backend is unknown or URI scheme not supported
            ImportError: If required backend library not installed
            
        Examples:
            >>> # Auto-detect from URI
            >>> writer = StorageWriterFactory.create('file:///tmp/checkpoint.dat')
            >>> writer = StorageWriterFactory.create('s3://bucket/checkpoint.dat')
            
            >>> # Explicit backend
            >>> writer = StorageWriterFactory.create(
            ...     '/tmp/checkpoint.dat',
            ...     backend='file',
            ...     use_direct_io=True
            ... )
        """
        # Issue #583: reconstruct the URI scheme mlpstorage stripped to
        # work around the DLIO preflight double-prefix bug. No-op if the
        # URI already has a scheme or the env hint isn't set.
        uri_or_path = _normalize_checkpoint_uri(uri_or_path)

        # Explicit backend selection
        if backend:
            if backend == 'file':
                # File backend expects path, not URI
                path = uri_or_path[7:] if uri_or_path.startswith('file://') else uri_or_path
                return FileStorageWriter(path, use_direct_io=use_direct_io, fadvise_mode=fadvise_mode)
            
            elif backend == 'direct_fs':
                # O_DIRECT via s3dlio's direct:// URI — bypasses page cache entirely.
                # fadvise_mode is ignored; O_DIRECT never populates the page cache.
                path = uri_or_path
                for prefix in ('direct://', 'file://'):
                    if path.startswith(prefix):
                        path = path[len(prefix):]
                        break
                return S3DLIOStorageWriter('direct://' + path, **kwargs)
            
            elif backend == 's3dlio':
                return S3DLIOStorageWriter(uri_or_path, **kwargs)
            
            elif backend == 's3torchconnector':
                # Lazy import
                try:
                    from .s3torch_writer import S3TorchConnectorWriter
                    return S3TorchConnectorWriter(uri_or_path, **kwargs)
                except ImportError:
                    raise ImportError(
                        "s3torchconnector backend requires s3torchconnector package. "
                        "Install with: pip install s3torchconnector"
                    )
            
            elif backend == 'minio':
                try:
                    from .minio_writer import MinIOStorageWriter
                    return MinIOStorageWriter(uri_or_path, **kwargs)
                except ImportError:
                    raise ImportError(
                        "minio backend requires minio package. "
                        "Install with: pip install minio"
                    )
            
            else:
                raise ValueError(
                    f"Unknown backend: {backend}. "
                    f"Supported: file, s3dlio, s3torchconnector, minio\n"
                    f"Note: For Azure Blob Storage, use backend='s3dlio' with az:// URIs"
                )
        
        # Auto-detect from URI scheme
        if uri_or_path.startswith('s3://'):
            # Prefer s3dlio (multi-protocol), fallback to s3torchconnector
            try:
                return S3DLIOStorageWriter(uri_or_path, **kwargs)
            except ImportError:
                try:
                    from .s3torch_writer import S3TorchConnectorWriter
                    return S3TorchConnectorWriter(uri_or_path, **kwargs)
                except ImportError:
                    raise ImportError(
                        "No S3-capable backend found. "
                        "Install s3dlio or s3torchconnector"
                    )
        
        elif (uri_or_path.startswith('az://') or
              (uri_or_path.startswith('https://') and 'blob.core.windows.net' in uri_or_path)):
            # Azure Blob Storage via s3dlio only
            try:
                return S3DLIOStorageWriter(uri_or_path, **kwargs)
            except ImportError:
                raise ImportError(
                    "Azure Blob Storage requires s3dlio. Install with: pip install s3dlio"
                )
        
        elif uri_or_path.startswith('gs://'):
            return S3DLIOStorageWriter(uri_or_path, **kwargs)
        
        elif uri_or_path.startswith('file://'):
            path = uri_or_path[7:]  # Remove file:// prefix
            return FileStorageWriter(path, use_direct_io=use_direct_io, fadvise_mode=fadvise_mode)
        
        elif uri_or_path.startswith('direct://'):
            return S3DLIOStorageWriter(uri_or_path, **kwargs)
        
        else:
            # Default to file backend for plain paths
            return FileStorageWriter(uri_or_path, use_direct_io=use_direct_io, fadvise_mode=fadvise_mode)


__all__ = [
    'StorageWriter',
    'StorageWriterFactory',
    'FileStorageWriter',
    'S3DLIOStorageWriter',
    'MinIOStorageWriter',
    'S3TorchConnectorWriter',
]
