"""Base class for storage readers (streaming checkpoint load)."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class StorageReader(ABC):
    """Abstract base class for chunked byte-range readers.

    Each read_chunk() call issues a byte-range GET for exactly *size* bytes
    starting at *offset*.  The caller discards the data immediately, so
    peak RAM = one chunk regardless of the total checkpoint size.
    """

    @abstractmethod
    def read_chunk(self, offset: int, size: int) -> int:
        """Issue a byte-range GET and discard the result.

        Args:
            offset: Byte offset into the object.
            size:   Number of bytes to read.

        Returns:
            Number of bytes actually received.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> Dict[str, Any]:
        """Release any open connections / handles.

        Returns:
            Dict with at minimum: backend (str), total_bytes (int).
        """
        raise NotImplementedError
