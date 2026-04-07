"""MinIO S3-compatible storage writer using native minio library.

Provides high-performance checkpointing to MinIO, S3, and S3-compatible storage using
the official Python minio SDK with true streaming multipart upload API.

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


class MinIOStorageWriter(StorageWriter):
    """Storage writer for MinIO/S3 using native minio library with streaming multipart.
    
    Features:
    - True streaming multipart uploads using MinIO's S3-compatible API
    - Constant memory usage (only buffers one part at a time)
    - Support for MinIO, AWS S3, and S3-compatible storage
    - MPI rank-based endpoint selection for distributed workloads
    
    Multi-Endpoint Support:
    - Detects S3_ENDPOINT_URIS, S3_ENDPOINT_TEMPLATE, or S3_ENDPOINT_FILE
    - Each MPI rank selects different endpoint (round-robin)
    - No native load balancing (unlike s3dlio)
    
    Performance tuning:
    - part_size: Size of each multipart part (default: 32 MB, minimum: 5 MB)
    - num_parallel_uploads: Currently unused (sequential for simplicity)
    
    Uses MinIO's multipart upload API:
    - _create_multipart_upload() to initiate
    - _upload_part() for each part
    - _complete_multipart_upload() to finalize
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
                endpoints = MinIOStorageWriter._expand_template(template)
        
        # Option 3: File with URIs
        if not endpoints:
            file_path = os.environ.get('S3_ENDPOINT_FILE')
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    endpoints = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not endpoints:
            return None
        
        # Select endpoint based on MPI rank (round-robin)
        mpi_rank = MinIOStorageWriter._get_mpi_rank()
        if mpi_rank is not None and len(endpoints) > 1:
            selected = endpoints[mpi_rank % len(endpoints)]
            print(f"[MinIOWriter] MPI rank {mpi_rank}: selected endpoint {selected} from {len(endpoints)} endpoints")
            return selected
        elif len(endpoints) == 1:
            return endpoints[0]
        else:
            # No MPI but multiple endpoints - use first one with warning
            print(f"[MinIOWriter] WARNING: Multiple endpoints configured but no MPI rank detected")
            print(f"[MinIOWriter]          Using first endpoint: {endpoints[0]}")
            return endpoints[0]
    
    def __init__(
        self,
        uri: str,
        chunk_size: int = 32 * 1024 * 1024,
        part_size: int = 32 * 1024 * 1024,
        num_parallel_uploads: int = 8
    ):
        """Initialize MinIO storage writer with streaming multipart upload.
        
        Args:
            uri: S3 URI (s3://bucket/key)
            chunk_size: Buffer size for accumulating writes (default: 32 MB)
            part_size: Multipart part size (default: 32 MB, minimum: 5 MB)
            num_parallel_uploads: Concurrent uploads (default: 8) - currently unused
        
        Raises:
            ValueError: If URI is invalid or parameters out of range
            ImportError: If minio library not installed
        """
        if not uri.startswith('s3://'):
            raise ValueError(f"MinIO writer requires s3:// URI, got: {uri}")
        
        # Validate multipart parameters
        if part_size < 5 * 1024 * 1024:
            raise ValueError("part_size must be >= 5 MB (S3 minimum)")
        if not 1 <= num_parallel_uploads <= 64:
            raise ValueError("num_parallel_uploads must be between 1 and 64")
        
        try:
            from minio import Minio
        except ImportError:
            raise ImportError(
                "minio library required for MinIO storage writer. "
                "Install with: pip install minio"
            )
        
        # Parse S3 URI: s3://bucket/key
        parts = uri[5:].split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI format (expected s3://bucket/key): {uri}")
        
        self.bucket_name = parts[0]
        self.object_name = parts[1]
        self.uri = uri
        self.chunk_size = chunk_size
        self.part_size = part_size
        self.num_parallel_uploads = num_parallel_uploads
        
        # Get S3 credentials from environment
        access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        
        # Check for multi-endpoint configuration first
        endpoint = self._detect_and_select_endpoint()
        if not endpoint:
            # Fall back to single endpoint from AWS_ENDPOINT_URL
            endpoint = os.environ.get('AWS_ENDPOINT_URL', os.environ.get('S3_ENDPOINT'))
        
        if not access_key or not secret_key:
            raise ValueError(
                "AWS credentials required in environment: "
                "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
            )
        
        if not endpoint:
            # Default to AWS S3
            endpoint = "s3.amazonaws.com"
            secure = True
        else:
            # Parse endpoint to extract hostname:port and secure flag
            if endpoint.startswith("https://"):
                endpoint = endpoint[8:]
                secure = True
            elif endpoint.startswith("http://"):
                endpoint = endpoint[7:]
                secure = False
            else:
                # No protocol specified, assume http
                secure = False
        
        # Initialize MinIO client
        # Support custom CA certificate via AWS_CA_BUNDLE (same env var as s3dlio/boto3).
        # Required when the MinIO server uses a self-signed or private CA certificate.
        http_client = None
        ca_bundle = os.environ.get('AWS_CA_BUNDLE')
        if secure and ca_bundle:
            import urllib3
            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout.DEFAULT_TIMEOUT,
                maxsize=max(num_parallel_uploads, 10),
                cert_reqs='CERT_REQUIRED',
                ca_certs=ca_bundle,
                retries=urllib3.Retry(total=5, backoff_factor=0.2,
                                      status_forcelist=[500, 502, 503, 504]),
            )
            print(f"[MinIOWriter] TLS: using CA bundle from AWS_CA_BUNDLE={ca_bundle}")

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=os.environ.get('AWS_REGION', 'us-east-1'),
            http_client=http_client,
        )
        
        # Create multipart upload using MinIO's S3-compatible API
        self.upload_id = self.client._create_multipart_upload(
            self.bucket_name,
            self.object_name,
            {}  # headers
        )
        
        # Multipart state
        self.current_part_number = 1
        self.part_buffer = BytesIO()
        self.part_buffer_size = 0
        self.total_bytes = 0
        self._flushed_bytes = 0
        self._start_time = time.monotonic()

        # Parallel upload state
        from concurrent.futures import ThreadPoolExecutor
        self._max_in_flight = num_parallel_uploads
        self._executor = ThreadPoolExecutor(
            max_workers=num_parallel_uploads,
            thread_name_prefix='minio-part'
        )
        self._futures: dict = {}          # part_number -> Future
        self._inflight_parts: list = []  # (part_number, Part) collected from completed futures

        print(f"[Writer] Using minio library (streaming multipart, {num_parallel_uploads} parallel parts, {part_size//(1024**2)} MB/part)")
    
    
    def _upload_part_sync(self, part_data: bytes, part_number: int):
        """Upload a single multipart part — runs inside ThreadPoolExecutor."""
        etag = self.client._upload_part(
            bucket_name=self.bucket_name,
            object_name=self.object_name,
            data=part_data,
            headers=None,
            upload_id=self.upload_id,
            part_number=part_number,
        )
        from minio.datatypes import Part
        return Part(part_number, etag)

    def _throttle(self) -> None:
        """Block until fewer than _max_in_flight parts are uploading."""
        from concurrent.futures import FIRST_COMPLETED, wait as futures_wait
        while len(self._futures) >= self._max_in_flight:
            done_set, _ = futures_wait(
                list(self._futures.values()), return_when=FIRST_COMPLETED
            )
            to_remove = [pn for pn, f in self._futures.items() if f in done_set]
            for pn in to_remove:
                part = self._futures.pop(pn).result()  # raises on upload error
                self._inflight_parts.append((pn, part))

    def _flush_part(self) -> None:
        """Submit current part buffer to the thread pool (non-blocking)."""
        if self.part_buffer_size == 0:
            return

        part_data = self.part_buffer.getvalue()   # bytes copy
        part_number = self.current_part_number

        # Throttle: wait if too many parts already in flight
        self._throttle()

        # Reset buffer immediately so the producer can fill the next chunk
        # while uploads are happening in parallel
        self.part_buffer.close()
        self.part_buffer = BytesIO()
        self.part_buffer_size = 0
        self.current_part_number += 1
        self._flushed_bytes += len(part_data)

        # Submit upload asynchronously
        self._futures[part_number] = self._executor.submit(
            self._upload_part_sync, part_data, part_number
        )

        # Progress report
        elapsed = time.monotonic() - self._start_time
        written_gb = self._flushed_bytes / 1e9
        rate = written_gb / elapsed if elapsed > 0 else 0.0
        print(f'\r[Writer] {written_gb:.2f} GB, {rate:.2f} GB/s   ', end='', flush=True)
    
    def write_chunk(self, buffer: memoryview, size: int) -> int:
        """Write chunk, flushing parts as they fill up.
        
        Args:
            buffer: Memory buffer containing data to write
            size: Number of bytes to write from buffer
            
        Returns:
            Number of bytes written
        """
        data = bytes(buffer[:size])
        offset = 0
        
        while offset < size:
            # Calculate how much we can add to current part
            remaining_in_part = self.part_size - self.part_buffer_size
            chunk_remaining = size - offset
            to_write = min(remaining_in_part, chunk_remaining)
            
            # Add to part buffer
            self.part_buffer.write(data[offset:offset + to_write])
            self.part_buffer_size += to_write
            offset += to_write
            
            # Flush if part is full
            if self.part_buffer_size >= self.part_size:
                self._flush_part()
        
        self.total_bytes += size
        return size
    
    def close(self) -> Dict[str, Any]:
        """Wait for all in-flight uploads, then finalize multipart upload.

        Returns:
            Dictionary with backend, total_bytes, parts, etag, uri, chunk_size
        """
        from concurrent.futures import wait as futures_wait, ALL_COMPLETED
        try:
            # Flush any remaining buffered data
            if self.part_buffer_size > 0:
                self._flush_part()

            # Wait for all still-in-flight parts to complete
            if self._futures:
                futures_wait(list(self._futures.values()), return_when=ALL_COMPLETED)
                for pn, f in self._futures.items():
                    part = f.result()   # raises on upload error
                    self._inflight_parts.append((pn, part))
                self._futures.clear()

            self._executor.shutdown(wait=False)
            print()  # end the carriage-return progress line

            # Sort all collected parts by part number (S3 requires ordered list)
            self._inflight_parts.sort(key=lambda x: x[0])
            sorted_parts = [p for _, p in self._inflight_parts]

            # Complete multipart upload
            result = self.client._complete_multipart_upload(
                self.bucket_name,
                self.object_name,
                self.upload_id,
                sorted_parts,
            )

            return {
                'backend': 'minio-multipart',
                'total_bytes': self.total_bytes,
                'parts': len(sorted_parts),
                'etag': result.etag if hasattr(result, 'etag') else 'unknown',
                'uri': self.uri,
                'chunk_size': self.chunk_size,
            }

        except Exception as e:
            # Abort multipart upload on error (best effort)
            try:
                self.client._abort_multipart_upload(
                    self.bucket_name,
                    self.object_name,
                    self.upload_id,
                )
            except Exception:
                pass
            raise e

        finally:
            self.part_buffer.close()
