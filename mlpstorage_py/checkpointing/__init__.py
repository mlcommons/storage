"""Streaming checkpoint plugin for mlp-storage.

This package implements a producer-consumer pattern for efficient checkpoint I/O
with minimal training interruption. Supports multiple storage backends through
a unified interface.
"""

from .streaming_checkpoint import StreamingCheckpointing
from .storage_writers import (
    StorageWriter,
    StorageWriterFactory,
    FileStorageWriter,
    S3DLIOStorageWriter,
)

__all__ = [
    'StreamingCheckpointing',
    'StorageWriter',
    'StorageWriterFactory',
    'FileStorageWriter', 
    'S3DLIOStorageWriter',
]
