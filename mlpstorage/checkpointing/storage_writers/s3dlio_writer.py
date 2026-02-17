"""s3dlio multi-protocol storage writer.

Supports file://, direct://, s3://, az://, gs:// protocols through the
unified s3dlio library interface.
"""

from typing import Dict, Any
from .base import StorageWriter


class S3DLIOStorageWriter(StorageWriter):
    """Multi-protocol writer using s3dlio library.
    
    Supports:
    - file:// - Local filesystem (buffered)
    - direct:// - Local filesystem (O_DIRECT, unbuffered)
    - s3:// - AWS S3, MinIO, S3-compatible
    - az:// - Azure Blob Storage
    - gs:// - Google Cloud Storage
    
    Uses zero-copy write_chunk() via PyBuffer protocol for optimal performance.
    
    Examples:
        >>> # Local file
        >>> writer = S3DLIOStorageWriter('file:///tmp/checkpoint.dat')
        
        >>> # AWS S3
        >>> writer = S3DLIOStorageWriter('s3://my-bucket/checkpoints/ckpt.dat')
        
        >>> # Azure Blob
        >>> writer = S3DLIOStorageWriter('az://container/checkpoint.dat')
    """
    
    def __init__(self, uri: str, chunk_size: int = 32 * 1024 * 1024):
        """Initialize s3dlio writer.
        
        Args:
            uri: Full URI including scheme (file://, s3://, az://, gs://, direct://)
            chunk_size: Internal buffer size (default: 32 MB)
            
        Raises:
            ImportError: If s3dlio not installed
            ValueError: If URI scheme not supported
        """
        try:
            import s3dlio
            self.s3dlio = s3dlio
        except ImportError:
            raise ImportError(
                "s3dlio not available. Install with: pip install s3dlio"
            )
        
        self.uri = uri
        self.chunk_size = chunk_size
        self.total_bytes = 0
        
        # Create writer options
        options = s3dlio.PyWriterOptions().with_buffer_size(chunk_size)
        
        # Initialize writer based on URI scheme
        if uri.startswith('s3://'):
            self.writer = s3dlio.create_s3_writer(uri, options)
        elif uri.startswith('az://') or (uri.startswith('https://') and 'blob.core.windows.net' in uri):
            self.writer = s3dlio.create_azure_writer(uri, options)
        elif uri.startswith('gs://'):
            # GCS via S3 compatibility
            self.writer = s3dlio.create_s3_writer(uri, options)
        elif uri.startswith('file://'):
            self.writer = s3dlio.create_filesystem_writer(uri, options)
        elif uri.startswith('direct://'):
            self.writer = s3dlio.create_direct_filesystem_writer(uri, options)
        else:
            raise ValueError(
                f"Unsupported URI scheme: {uri}. "
                f"Supported: file://, direct://, s3://, az://, gs://"
            )
    
    def write_chunk(self, buffer: memoryview, size: int) -> int:
        """Write chunk using s3dlio (zero-copy via PyBuffer protocol).
        
        Args:
            buffer: Memory buffer (memoryview, numpy array, shared_memory)
            size: Number of bytes to write
            
        Returns:
            Number of bytes written
        """
        # s3dlio's write_chunk() supports PyBuffer protocol (memoryview)
        # This avoids copying data - memoryview passed directly to Rust
        self.writer.write_chunk(buffer[:size])
        self.total_bytes += size
        return size
    
    def close(self) -> Dict[str, Any]:
        """Finalize write and return statistics.
        
        Returns:
            Dictionary with backend info and bytes written
        """
        if self.writer:
            self.writer.finalize()
        
        return {
            'backend': 's3dlio',
            'total_bytes': self.total_bytes,
            'uri': self.uri,
            'chunk_size': self.chunk_size
        }
