#!/usr/bin/env python3
"""
Test DLIO s3dlio backend with file:// URIs to verify zero-copy.

This test bypasses full DLIO benchmark to test just the storage layer.
"""

import sys
import os
from pathlib import Path

# Add DLIO to path
sys.path.insert(0, str(Path.home() / "Documents/Code/mlp-storage/.venv/lib/python3.12/site-packages"))

print("Testing DLIO s3dlio storage backend with zero-copy...")
print("="*60)

# Import DLIO components
from dlio_benchmark.common.enumerations import StorageType
from dlio_benchmark.storage.storage_factory import StorageFactory

# Create a mock namespace for storage options
class MockNamespace:
    def __init__(self):
        self.storage_type = StorageType.S3DLIO
        self.storage_root = "file:///tmp/dlio-zerocopy-test/"
        self.storage_options = {}

namespace = MockNamespace()

# Get storage backend
print(f"\n1. Creating storage backend...")
print(f"   Type: {namespace.storage_type}")
print(f"   Root: {namespace.storage_root}")

storage = StorageFactory.get_storage(
    namespace.storage_type, 
    namespace
)

print(f"   ✓ Storage backend created: {type(storage).__name__}")

# List files
print(f"\n2. Listing files...")
files = storage.walk_node("", use_pattern=False)
print(f"   ✓ Found {len(files)} files:")
for i, f in enumerate(files[:5]):  # Show first 5
    print(f"      {i}: {f}")

# Read a file
if files:
    print(f"\n3. Reading first file (zero-copy test)...")
    file_id = files[0]
    print(f"   File: {file_id}")
    
    data = storage.get_data(file_id)
    print(f"   ✓ Data received")
    print(f"      Type: {type(data).__name__}")
    print(f"      Length: {len(data)} bytes")
    print(f"      Has buffer protocol: {hasattr(data, '__buffer__')}")
    
    # Verify it's BytesView (zero-copy)
    if type(data).__name__ == "BytesView":
        print(f"   ✅ ZERO-COPY confirmed! (BytesView)")
    elif type(data).__name__ == "bytes":
        print(f"   ⚠️  bytes returned (creates copy, not zero-copy)")
    else:
        print(f"   ❓ Unknown type: {type(data)}")
    
    # Test buffer protocol with NumPy
    print(f"\n4. Testing buffer protocol with NumPy...")
    try:
        import numpy as np
        arr = np.frombuffer(data, dtype=np.uint8)
        print(f"   ✓ NumPy array created (zero-copy)")
        print(f"      Shape: {arr.shape}")
        print(f"      First 20 bytes: {arr[:20]}")
    except Exception as e:
        print(f"   ✗ NumPy failed: {e}")
    
    # Test with PyTorch
    print(f"\n5. Testing buffer protocol with PyTorch...")
    try:
        import torch
        tensor = torch.frombuffer(data, dtype=torch.uint8)
        print(f"   ✓ PyTorch tensor created (zero-copy)")
        print(f"      Shape: {tensor.shape}")
    except Exception as e:
        print(f"   ✗ PyTorch failed: {e}")

print("\n" + "="*60)
print("DLIO Storage Backend Test Complete!")
print("="*60)
