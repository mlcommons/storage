"""Base classes for storage writers.

This module defines the abstract interface that all storage backend
implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class StorageWriter(ABC):
    """Abstract base class for all storage backend writers.
    
    All storage backends (file, s3dlio, s3torchconnector, etc.) must implement
    this interface to provide consistent behavior for streaming checkpoints.
    """
    
    @abstractmethod
    def write_chunk(self, buffer: memoryview, size: int) -> int:
        """Write a chunk of data from the buffer.
        
        Args:
            buffer: Memory buffer containing data to write
            size: Number of bytes to write from buffer
            
        Returns:
            Number of bytes actually written
            
        Raises:
            IOError: If write operation fails
        """
        raise NotImplementedError
    
    @abstractmethod
    def close(self) -> Dict[str, Any]:
        """Finalize the write operation and return statistics.
        
        This typically involves flushing buffers, closing file descriptors,
        and collecting performance metrics.
        
        Returns:
            Dictionary containing:
                - backend: str - Backend name
                - total_bytes: int - Total bytes written
                - Additional backend-specific metrics
                
        Raises:
            IOError: If close/flush operation fails
        """
        raise NotImplementedError
