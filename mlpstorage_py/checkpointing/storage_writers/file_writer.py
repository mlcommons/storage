"""Native filesystem writer with optional O_DIRECT support."""

import os
from typing import Dict, Any
from .base import StorageWriter


class FileStorageWriter(StorageWriter):
    """Native file I/O writer with optional O_DIRECT (bypassing page cache).
    
    This is the simplest backend and serves as a baseline for performance
    comparisons. Supports O_DIRECT on Linux for unbuffered I/O.
    
    Examples:
        >>> writer = FileStorageWriter('/tmp/checkpoint.dat', use_direct_io=False)
        >>> import shared_memory
        >>> shm = shared_memory.SharedMemory(create=True, size=1024)
        >>> writer.write_chunk(shm.buf, 1024)
        1024
        >>> stats = writer.close()
        >>> print(stats['total_bytes'])
        1024
    """
    
    def __init__(self, filepath: str, use_direct_io: bool = False, fadvise_mode: str = 'none'):
        """Initialize file writer.
        
        Args:
            filepath: Absolute path to output file
            use_direct_io: Enable O_DIRECT (requires aligned buffers on Linux)
            fadvise_mode: 'none', 'sequential', or 'dontneed'
        """
        self.filepath = filepath
        self.use_direct_io = use_direct_io
        self.fadvise_mode = fadvise_mode
        self.total_bytes = 0
        
        # Create parent directory if needed
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        # Open file with appropriate flags
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if use_direct_io and hasattr(os, 'O_DIRECT'):
            flags |= os.O_DIRECT
            self.direct_io = True
        else:
            self.direct_io = False
            if use_direct_io:
                import warnings
                warnings.warn(
                    "O_DIRECT requested but not available on this platform",
                    RuntimeWarning
                )
        
        self.fd = os.open(filepath, flags, 0o644)
        
        # No SEQUENTIAL hint: readahead is meaningless on a write-only fd and
        # would only inflate page cache.  DONTNEED is applied per-write below
        # to flush and drop dirty pages as we go.
    
    def write_chunk(self, buffer: memoryview, size: int) -> int:
        """Write chunk to file.
        
        Args:
            buffer: Memory buffer (typically from shared_memory.SharedMemory)
            size: Number of bytes to write
            
        Returns:
            Number of bytes written
        """
        offset_before = self.total_bytes
        written = os.write(self.fd, buffer[:size])
        self.total_bytes += written
        
        # Drop pages for data we just wrote so the load phase cannot serve
        # them from DRAM — checkpoint reads must hit the actual storage device
        # to produce a valid throughput measurement.
        if self.fadvise_mode == 'dontneed' and hasattr(os, 'posix_fadvise'):
            try:
                os.posix_fadvise(self.fd, offset_before, written, os.POSIX_FADV_DONTNEED)
            except (OSError, AttributeError):
                pass  # Ignore if not supported
        
        return written
    
    def close(self) -> Dict[str, Any]:
        """Close file and return statistics.
        
        Returns:
            Dictionary with backend info and bytes written
        """
        # Single fsync at the very end (not incremental)
        os.fsync(self.fd)  # Ensure all data is on disk
        os.close(self.fd)
        
        return {
            'backend': 'file',
            'total_bytes': self.total_bytes,
            'filepath': self.filepath,
            'direct_io': self.direct_io,
            'fadvise': self.fadvise_mode
        }
