"""s3dlio multi-protocol storage writer.

Supports file://, direct://, s3://, az://, gs:// protocols through the
unified s3dlio library interface with multi-endpoint load balancing.
"""

import os
import time
from typing import Dict, Any, List, Optional
from .base import StorageWriter


class S3DLIOStorageWriter(StorageWriter):
    """Multi-protocol writer using s3dlio library.
    
    Supports:
    - file:// - Local filesystem (buffered)
    - direct:// - Local filesystem (O_DIRECT, unbuffered)
    - s3:// - AWS S3, MinIO, S3-compatible (with proper multipart upload)
    - az:// - Azure Blob Storage
    - gs:// - Google Cloud Storage
    
    Multi-Endpoint Support (S3/Az/GCS only):
    - Supports round-robin and least-connections load balancing
    - Configure via environment variables:
      * S3_ENDPOINT_URIS: Comma-separated list "http://host1:9000,http://host2:9000"
      * S3_ENDPOINT_TEMPLATE: Template with expansion "http://172.16.21.{1...8}:9000"
      * S3_ENDPOINT_FILE: Path to file with one URI per line
      * S3_LOAD_BALANCE_STRATEGY: "round_robin" (default) or "least_connections"
    - MPI-aware: Uses OMPI_COMM_WORLD_RANK to select endpoint for distributed runs
    
    Uses zero-copy write_chunk() via PyBuffer protocol for optimal performance.
    For S3, uses MultipartUploadWriter for proper concurrent multipart uploads.
    
    Examples:
        >>> # Local file
        >>> writer = S3DLIOStorageWriter('file:///tmp/checkpoint.dat')
        
        >>> # AWS S3 (uses MultipartUploadWriter)
        >>> writer = S3DLIOStorageWriter('s3://my-bucket/checkpoints/ckpt.dat')
        
        >>> # Multi-endpoint S3 (via environment variables)
        >>> os.environ['S3_ENDPOINT_URIS'] = 'http://172.16.21.1:9000,http://172.16.21.2:9000'
        >>> writer = S3DLIOStorageWriter('s3://bucket/checkpoint.dat')
    """
    
    def __init__(self, uri: str, chunk_size: int = 32 * 1024 * 1024, 
                 part_size: int = 32 * 1024 * 1024, max_in_flight: int = 16,
                 use_multi_endpoint: bool = True):
        """Initialize s3dlio writer.
        
        Args:
            uri: Full URI including scheme (file://, s3://, az://, gs://, direct://)
            chunk_size: Internal buffer size (default: 32 MB)
            part_size: Multipart upload part size (default: 32 MB, minimum for S3)
            max_in_flight: Concurrent multipart uploads (default: 16, range: 1-64)
                         Aligned with dgen-py's optimal 32 MB buffer size for impedance matching
            use_multi_endpoint: Enable multi-endpoint load balancing (default: True)
                              Only applies to S3/Azure/GCS URIs
            
        Raises:
            ImportError: If s3dlio not installed
            ValueError: If URI scheme not supported or parameters out of range
        """
        # Validate parameters
        if part_size < 5 * 1024 * 1024:
            raise ValueError(f"part_size must be >= 5 MB (S3 minimum), got {part_size / (1024**2):.1f} MB")
        if not 1 <= max_in_flight <= 64:
            raise ValueError(f"max_in_flight must be between 1 and 64, got {max_in_flight}")
        
        try:
            import s3dlio
            self.s3dlio = s3dlio
        except ImportError:
            raise ImportError(
                "s3dlio not available. Install with: pip install s3dlio"
            )
        
        self.uri = uri
        self.chunk_size = chunk_size
        self.part_size = part_size
        self.max_in_flight = max_in_flight
        self.total_bytes = 0
        self._start_time = time.monotonic()
        self.writer = None
        self.writer_type = None
        self.multi_endpoint_mode = False
        
        # Check for multi-endpoint configuration (S3/Azure/GCS only)
        endpoint_uris = self._detect_multi_endpoint_config() if use_multi_endpoint else None
        
        # Initialize writer based on URI scheme
        if uri.startswith('s3://') or uri.startswith('gs://'):
            # S3/GCS: Check for multi-endpoint configuration first
            if endpoint_uris:
                self._init_multi_endpoint_s3(uri, endpoint_uris)
            else:
                self._init_single_endpoint_s3(uri)
            
        elif uri.startswith('az://') or (uri.startswith('https://') and 'blob.core.windows.net' in uri):
            # Azure Blob Storage
            if endpoint_uris:
                self._init_multi_endpoint_azure(uri, endpoint_uris)
            else:
                options = s3dlio.PyWriterOptions().with_buffer_size(chunk_size)
                self.writer = s3dlio.create_azure_writer(uri, options)
                self.writer_type = 'streaming'
            
        elif uri.startswith('file://'):
            # Local filesystem uses streaming writer
            options = s3dlio.PyWriterOptions().with_buffer_size(chunk_size)
            self.writer = s3dlio.create_filesystem_writer(uri, options)
            self.writer_type = 'streaming'
            
        elif uri.startswith('direct://'):
            # Direct I/O uses streaming writer
            options = s3dlio.PyWriterOptions().with_buffer_size(chunk_size)
            self.writer = s3dlio.create_direct_filesystem_writer(uri, options)
            self.writer_type = 'streaming'
            
        else:
            raise ValueError(
                f"Unsupported URI scheme: {uri}. "
                f"Supported: file://, direct://, s3://, az://, gs://"
            )
    
    def _detect_multi_endpoint_config(self) -> Optional[List[str]]:
        """Detect multi-endpoint configuration from environment variables.
        
        Priority order:
        1. S3_ENDPOINT_URIS - Comma-separated list
        2. S3_ENDPOINT_TEMPLATE - Template with {N...M} expansion  
        3. S3_ENDPOINT_FILE - File with one URI per line
        4. MPI rank-based single endpoint selection from AWS_ENDPOINT_URL
        
        Returns:
            List of endpoint URIs if multi-endpoint configured, None otherwise
        """
        # Option 1: Explicit URI list
        uris_str = os.environ.get('S3_ENDPOINT_URIS')
        if uris_str:
            uris = [u.strip() for u in uris_str.split(',') if u.strip()]
            if len(uris) > 1:
                print(f"[S3DLIOWriter] Multi-endpoint mode: {len(uris)} endpoints from S3_ENDPOINT_URIS")
                return uris
        
        # Option 2: Template expansion
        template = os.environ.get('S3_ENDPOINT_TEMPLATE')
        if template:
            uris = self._expand_template(template)
            if len(uris) > 1:
                print(f"[S3DLIOWriter] Multi-endpoint mode: {len(uris)} endpoints from template")
                return uris
        
        # Option 3: File with URIs
        file_path = os.environ.get('S3_ENDPOINT_FILE')
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                uris = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            if len(uris) > 1:
                print(f"[S3DLIOWriter] Multi-endpoint mode: {len(uris)} endpoints from file")
                return uris
        
        # Option 4: MPI rank-based single endpoint (distributed mode)
        mpi_rank = self._get_mpi_rank()
        if mpi_rank is not None and uris_str:
            # Select endpoint based on rank (round-robin)
            uris = [u.strip() for u in uris_str.split(',') if u.strip()]
            if len(uris) > 1:
                selected = uris[mpi_rank % len(uris)]
                print(f"[S3DLIOWriter] MPI mode: rank {mpi_rank} using endpoint {selected}")
                # Return single endpoint (no multi-endpoint store needed)
                os.environ['AWS_ENDPOINT_URL'] = selected
        
        return None  # No multi-endpoint configuration
    
    def _get_mpi_rank(self) -> Optional[int]:
        """Get MPI rank from Open MPI environment variables.
        
        Returns:
            MPI rank (0-based) or None if not in MPI environment
        """
        # Open MPI v4+ uses OMPI_COMM_WORLD_RANK
        rank_str = os.environ.get('OMPI_COMM_WORLD_RANK')
        if rank_str:
            try:
                return int(rank_str)
            except ValueError:
                pass
        
        # MPICH uses PMI_RANK
        rank_str = os.environ.get('PMI_RANK')
        if rank_str:
            try:
                return int(rank_str)
            except ValueError:
                pass
        
        return None
    
    def _expand_template(self, template: str) -> List[str]:
        """Expand URI template with {N...M} syntax.
        
        Example:
            "http://172.16.21.{1...8}:9000" -> 
            ["http://172.16.21.1:9000", "http://172.16.21.2:9000", ...]
        """
        import re
        match = re.search(r'\{(\d+)\.\.\.(\d+)\}', template)
        if not match:
            return [template]
        
        start, end = int(match.group(1)), int(match.group(2))
        prefix = template[:match.start()]
        suffix = template[match.end():]
        
        return [f"{prefix}{i}{suffix}" for i in range(start, end + 1)]
    
    def _init_single_endpoint_s3(self, uri: str):
        """Initialize single-endpoint S3 writer (traditional mode)."""
        print(f"[S3DLIOWriter] Using MultipartUploadWriter (single endpoint)")
        print(f"[S3DLIOWriter]   part_size={self.part_size / (1024**2):.0f} MB, max_in_flight={self.max_in_flight}")
        
        self.writer = self.s3dlio.MultipartUploadWriter.from_uri(
            uri,
            part_size=self.part_size,
            max_in_flight=self.max_in_flight,
            abort_on_drop=True
        )
        self.writer_type = 'multipart'
    
    def _init_multi_endpoint_s3(self, uri: str, endpoint_uris: List[str]):
        """Initialize multi-endpoint S3 writer with load balancing."""
        strategy = os.environ.get('S3_LOAD_BALANCE_STRATEGY', 'round_robin')
        
        print(f"[S3DLIOWriter] Using MultiEndpointStore")
        print(f"[S3DLIOWriter]   endpoints={len(endpoint_uris)}, strategy={strategy}")
        print(f"[S3DLIOWriter]   part_size={self.part_size / (1024**2):.0f} MB, max_in_flight={self.max_in_flight}")
        
        # Create multi-endpoint store
        self.multi_endpoint_store = self.s3dlio.create_multi_endpoint_store(
            uris=endpoint_uris,
            strategy=strategy
        )
        
        # Create multipart writer using the multi-endpoint store
        # Note: s3dlio will handle routing through the store
        self.writer = self.s3dlio.MultipartUploadWriter.from_uri(
            uri,
            part_size=self.part_size,
            max_in_flight=self.max_in_flight,
            abort_on_drop=True
        )
        self.writer_type = 'multipart'
        self.multi_endpoint_mode = True
    
    def _init_multi_endpoint_azure(self, uri: str, endpoint_uris: List[str]):
        """Initialize multi-endpoint Azure writer with load balancing."""
        strategy = os.environ.get('S3_LOAD_BALANCE_STRATEGY', 'round_robin')
        
        print(f"[S3DLIOWriter] Using MultiEndpointStore for Azure")
        print(f"[S3DLIOWriter]   endpoints={len(endpoint_uris)}, strategy={strategy}")
        
        # Create multi-endpoint store for Azure
        self.multi_endpoint_store = self.s3dlio.create_multi_endpoint_store(
            uris=endpoint_uris,
            strategy=strategy
        )
        
        # Use streaming writer with multi-endpoint support
        options = self.s3dlio.PyWriterOptions().with_buffer_size(self.chunk_size)
        self.writer = self.s3dlio.create_azure_writer(uri, options)
        self.writer_type = 'streaming'
        self.multi_endpoint_mode = True
    
    def write_chunk(self, buffer: memoryview, size: int) -> int:
        """Write chunk using s3dlio (zero-copy via PyBuffer protocol).
        
        Args:
            buffer: Memory buffer (memoryview, numpy array, shared_memory)
            size: Number of bytes to write
            
        Returns:
            Number of bytes written
        """
        if self.writer_type == 'multipart':
            # MultipartUploadWriter.write() accepts buffer protocol objects
            self.writer.write(buffer[:size])
        else:
            # Streaming writer uses write_chunk()
            self.writer.write_chunk(buffer[:size])
        
        self.total_bytes += size
        elapsed = time.monotonic() - self._start_time
        written_gb = self.total_bytes / 1e9
        rate = written_gb / elapsed if elapsed > 0 else 0.0
        print(f'\r[Writer] {written_gb:.2f} GB, {rate:.2f} GB/s   ', end='', flush=True)
        return size
    
    def close(self) -> Dict[str, Any]:
        """Finalize write and return statistics.
        
        Returns:
            Dictionary with backend info and bytes written
        """
        if not self.writer:
            return {
                'backend': 's3dlio',
                'total_bytes': self.total_bytes,
                'uri': self.uri,
                'chunk_size': self.chunk_size,
                'multi_endpoint': self.multi_endpoint_mode
            }
        
        if self.writer_type == 'multipart':
            # MultipartUploadWriter.close() returns detailed stats
            stats = self.writer.close()
            print()  # end the \r progress line
            result = {
                'backend': 's3dlio-multipart',
                'total_bytes': stats.get('total_bytes', self.total_bytes),
                'parts': stats.get('parts', 0),
                'etag': stats.get('etag', None),
                'uri': self.uri,
                'chunk_size': self.chunk_size,
                'multi_endpoint': self.multi_endpoint_mode
            }
            
            # Add multi-endpoint stats if available
            if self.multi_endpoint_mode and hasattr(self, 'multi_endpoint_store'):
                try:
                    ep_stats = self.multi_endpoint_store.get_stats()
                    result['endpoint_stats'] = ep_stats
                except:
                    pass  # Stats not available
            
            return result
        else:
            # Streaming writer uses finalize()
            self.writer.finalize()
            print()  # end the \r progress line
            return {
                'backend': 's3dlio-streaming',
                'total_bytes': self.total_bytes,
                'uri': self.uri,
                'chunk_size': self.chunk_size,
                'multi_endpoint': self.multi_endpoint_mode
            }
