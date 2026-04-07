"""S3 storage writer using AWS s3torchconnector library.

Provides high-performance checkpointing to AWS S3 using the official
s3torchconnector library with auto-managed multipart uploads.

Multi-Endpoint Support:
- MPI rank-based endpoint selection (no native load balancing)
- Configure via S3_ENDPOINT_URIS, S3_ENDPOINT_TEMPLATE, or S3_ENDPOINT_FILE
- Each MPI rank selects different endpoint (round-robin)
"""

import os
import re
import time
from io import BytesIO
from typing import Optional, Dict, Any, List

from .base import StorageWriter


class S3TorchConnectorWriter(StorageWriter):
    """Storage writer for AWS S3 using s3torchconnector library.
    
    Features:
    - AWS S3-optimized with s3torchconnector
    - Automatic multipart upload management
    - Buffered writes with single upload on close
    - MPI rank-based endpoint selection for distributed workloads
    
    Multi-Endpoint Support:
    - Detects S3_ENDPOINT_URIS, S3_ENDPOINT_TEMPLATE, or S3_ENDPOINT_FILE
    - Each MPI rank selects different endpoint (round-robin)
    - No native load balancing (unlike s3dlio)
    
    Note: s3torchconnector manages multipart uploads internally - no manual tuning.
    For explicit multipart control or native multi-endpoint support, use S3DLIOStorageWriter.
    """
    
    @staticmethod
    def _get_mpi_rank() -> Optional[int]:
        """Get MPI rank from environment variables.
        
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
    
    @staticmethod
    def _expand_template(template: str) -> List[str]:
        """Expand URI template with {N...M} syntax.
        
        Example:
            "http://172.16.21.{1...8}:9000" -> 
            ["http://172.16.21.1:9000", "http://172.16.21.2:9000", ...]
        """
        match = re.search(r'\{(\d+)\.\.\.(\d+)\}', template)
        if not match:
            return [template]
        
        start, end = int(match.group(1)), int(match.group(2))
        prefix = template[:match.start()]
        suffix = template[match.end():]
        
        return [f"{prefix}{i}{suffix}" for i in range(start, end + 1)]
    
    @staticmethod
    def _detect_and_select_endpoint() -> Optional[str]:
        """Detect multi-endpoint configuration and select based on MPI rank.
        
        Priority order:
        1. S3_ENDPOINT_URIS - Comma-separated list
        2. S3_ENDPOINT_TEMPLATE - Template with {N...M} expansion
        3. S3_ENDPOINT_FILE - File with one URI per line
        
        Returns:
            Selected endpoint URI or None if no multi-endpoint config
        """
        endpoints = []
        
        # Option 1: Explicit URI list
        uris_str = os.environ.get('S3_ENDPOINT_URIS')
        if uris_str:
            endpoints = [u.strip() for u in uris_str.split(',') if u.strip()]
        
        # Option 2: Template expansion
        if not endpoints:
            template = os.environ.get('S3_ENDPOINT_TEMPLATE')
            if template:
                endpoints = S3TorchConnectorWriter._expand_template(template)
        
        # Option 3: File with URIs
        if not endpoints:
            file_path = os.environ.get('S3_ENDPOINT_FILE')
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    endpoints = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not endpoints:
            return None
        
        # Select endpoint based on MPI rank (round-robin)
        mpi_rank = S3TorchConnectorWriter._get_mpi_rank()
        if mpi_rank is not None and len(endpoints) > 1:
            selected = endpoints[mpi_rank % len(endpoints)]
            print(f"[S3TorchWriter] MPI rank {mpi_rank}: selected endpoint {selected} from {len(endpoints)} endpoints")
            return selected
        elif len(endpoints) == 1:
            return endpoints[0]
        else:
            # No MPI but multiple endpoints - use first one with warning
            print(f"[S3TorchWriter] WARNING: Multiple endpoints configured but no MPI rank detected")
            print(f"[S3TorchWriter]          Using first endpoint: {endpoints[0]}")
            return endpoints[0]
    
    def __init__(
        self,
        uri: str,
        chunk_size: int = 32 * 1024 * 1024,
        **kwargs
    ):
        """Initialize S3TorchConnector storage writer.
        
        Args:
            uri: S3 URI (s3://bucket/key)
            chunk_size: Buffer size for accumulating writes (default: 32 MB)
            **kwargs: Additional options (ignored - s3torchconnector has auto-tuning)
        
        Raises:
            ValueError: If URI is invalid
            ImportError: If s3torchconnector library not installed
        """
        if not uri.startswith('s3://'):
            raise ValueError(f"S3TorchConnector writer requires s3:// URI, got: {uri}")
        
        try:
            from s3torchconnector._s3client import S3Client, S3ClientConfig
        except ImportError:
            raise ImportError(
                "s3torchconnector library required for S3TorchConnector storage writer. "
                "Install with: pip install s3torchconnector"
            )
        
        # Parse S3 URI: s3://bucket/key
        parts = uri[5:].split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI format (expected s3://bucket/key): {uri}")
        
        self.bucket_name = parts[0]
        self.object_key = parts[1]
        self.uri = uri
        self.chunk_size = chunk_size
        
        # Get S3 configuration from environment
        region = os.environ.get('AWS_REGION', 'us-east-1')
        
        # Check for multi-endpoint configuration first
        endpoint = self._detect_and_select_endpoint()
        if not endpoint:
            # Fall back to single endpoint from AWS_ENDPOINT_URL
            endpoint = os.environ.get('AWS_ENDPOINT_URL', os.environ.get('S3_ENDPOINT'))
        
        # S3Client config - use defaults for AWS best practices
        s3_client_config = S3ClientConfig(
            force_path_style=bool(endpoint),  # Use path style for custom endpoints
            max_attempts=3
        )
        
        # Initialize S3TorchConnector client
        self.s3_client = S3Client(
            region=region,
            endpoint=endpoint,
            s3client_config=s3_client_config
        )
        
        # Start streaming writer immediately (supports incremental writes)
        self.writer = self.s3_client.put_object(self.bucket_name, self.object_key)
        self.total_bytes = 0
        self._start_time = time.monotonic()
        
        print(f"[S3TorchWriter] Using s3torchconnector library (streaming)")
        print(f"[S3TorchWriter]   region={region}, endpoint={endpoint or 'AWS S3'}")
        print(f"[S3TorchWriter]   (multipart auto-managed by s3torchconnector)")
    
    def write_chunk(self, buffer: memoryview, size: int) -> int:
        """Write chunk directly to S3 (streaming).
        
        Args:
            buffer: Memory buffer containing data to write
            size: Number of bytes to write from buffer
            
        Returns:
            Number of bytes written
        """
        data = bytes(buffer[:size])
        self.writer.write(data)  # Stream directly to S3
        self.total_bytes += size
        elapsed = time.monotonic() - self._start_time
        written_gb = self.total_bytes / 1e9
        rate = written_gb / elapsed if elapsed > 0 else 0.0
        print(f'\r[Writer] {written_gb:.2f} GB, {rate:.2f} GB/s   ', end='', flush=True)
        return size
    
    def close(self) -> Dict[str, Any]:
        """Finalize streaming upload and return metadata.
        
        Returns:
            Dictionary with backend, total_bytes, etag, uri, chunk_size
        """
        # Close the streaming writer (completes multipart upload)
        self.writer.close()
        print()  # end the \r progress line
        
        return {
            'backend': 's3torchconnector',
            'total_bytes': self.total_bytes,
            'etag': 'auto-managed',  # s3torchconnector doesn't expose ETag
            'uri': self.uri,
            'chunk_size': self.chunk_size
        }
