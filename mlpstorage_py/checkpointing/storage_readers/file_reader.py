"""Native filesystem reader with posix_fadvise(POSIX_FADV_DONTNEED) support.

After each read_chunk() call, the just-read pages are dropped from the kernel
page cache via POSIX_FADV_DONTNEED.  This ensures that every read issues a
real I/O request to the underlying storage device — if the pages were allowed
to remain cached, a second (or even the first) read could be served entirely
from DRAM, making the benchmark report DRAM bandwidth instead of storage
throughput.

For write-path equivalent behaviour see file_writer.py which applies the
same fadvise hint after each write_chunk(), ensuring that checkpoint data
written in the save phase is not cached in DRAM when the load phase reads
it back.
"""

import os
from typing import Dict, Any
from .base import StorageReader

# POSIX_FADV_DONTNEED: "The specified data will not be accessed in the near
# future."  The kernel is free to drop the corresponding page-cache pages.
_FADV_DONTNEED = getattr(os, 'POSIX_FADV_DONTNEED', 4)  # 4 is the Linux value


class FileStorageReader(StorageReader):
    """Chunked sequential reader for local-filesystem checkpoint files.

    Reads exactly *size* bytes at *offset* per read_chunk() call, then
    immediately advises the kernel to reclaim those pages.  Without this,
    the OS would cache each chunk in DRAM and a subsequent read (or even the
    first read if kernel readahead pre-populated the cache) would report DRAM
    bandwidth rather than actual storage I/O throughput.

    Args:
        filepath:     Absolute path to the checkpoint file.
        fadvise_mode: 'dontneed' — drop pages after each read (default, recommended).
                      'sequential' — hint sequential access only (pages kept).
                      'none' — no fadvise hints at all.
        chunk_size:   Ignored (kept for factory interface compatibility).
    """

    def __init__(self, filepath: str, fadvise_mode: str = 'dontneed', chunk_size: int = None):
        self.filepath = filepath
        self.fadvise_mode = fadvise_mode
        self.total_bytes = 0
        self._fadvise_available = hasattr(os, 'posix_fadvise')

        self.fd = os.open(filepath, os.O_RDONLY)

        # Disable kernel readahead (RANDOM = no speculative prefetch).
        # With SEQUENTIAL the kernel pre-fills the page cache with pages ahead
        # of the current file position, so reads may never reach the storage
        # device at all.  RANDOM disables that prefetch, ensuring that only the
        # pages explicitly requested by pread() are fetched from storage.
        # DONTNEED then drops each chunk immediately, keeping the live footprint
        # to one chunk window and guaranteeing subsequent reads are not served
        # from DRAM.
        if self._fadvise_available:
            try:
                os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
            except (OSError, AttributeError):
                pass

        print(f"[FileReader] path={filepath}  fadvise={fadvise_mode}")

    def read_chunk(self, offset: int, size: int) -> int:
        """Read *size* bytes at *offset* and drop those pages from page cache.

        Returns:
            Number of bytes actually read.
        """
        # pread() reads at an arbitrary offset without moving the fd position,
        # which is safe when multiple reader threads share-nothing file descriptors.
        data = os.pread(self.fd, size, offset)
        nbytes = len(data)
        self.total_bytes += nbytes

        if nbytes > 0 and self.fadvise_mode == 'dontneed' and self._fadvise_available:
            try:
                # Drop exactly the pages we just read — forces the next read of
                # the same region to go to storage rather than DRAM cache.
                os.posix_fadvise(self.fd, offset, nbytes, _FADV_DONTNEED)
            except (OSError, AttributeError):
                pass

        # Discard buffer immediately — this is a throughput benchmark.
        return nbytes

    def close(self) -> Dict[str, Any]:
        """Close the file descriptor."""
        os.close(self.fd)
        return {'backend': 'file', 'total_bytes': self.total_bytes, 'fadvise': self.fadvise_mode}
