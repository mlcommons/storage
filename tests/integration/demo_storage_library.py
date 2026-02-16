#!/usr/bin/env python3
"""
Demo: storage_library configuration in action

Shows how different storage libraries are loaded based on config.
"""

import os
import sys

print("="*60)
print("Storage Library Selection Demo")
print("="*60)

# Simulate DLIO config args
class MockArgs:
    """Mock DLIO configuration arguments"""
    def __init__(self, storage_library="s3torchconnector"):
        self.storage_library = storage_library
        self.s3_region = "us-east-1"
        self.s3_force_path_style = False
        self.s3_max_attempts = 5

def test_import(storage_library):
    """Test importing the appropriate library"""
    print(f"\nTest: storage_library = '{storage_library}'")
    print("-" * 60)
    
    # This is the exact logic from our patched s3_torch_storage.py
    if storage_library == "s3dlio":
        print(f"  ✅ Using s3dlio compatibility layer (zero-copy)")
        from s3dlio.compat.s3torchconnector import S3Client, S3ClientConfig
        print(f"  📦 Imported: {S3Client.__module__}.S3Client")
    else:
        print(f"  ℹ️  Using AWS s3torchconnector")
        try:
            from s3torchconnector._s3client import S3Client, S3ClientConfig
            print(f"  📦 Imported: {S3Client.__module__}.S3Client")
        except ImportError:
            print(f"  ⚠️  s3torchconnector not installed, falling back to s3dlio")
            from s3dlio.compat.s3torchconnector import S3Client, S3ClientConfig
            print(f"  📦 Imported: {S3Client.__module__}.S3Client")
    
    # Create client instance
    config = S3ClientConfig(force_path_style=True, max_attempts=5)
    client = S3Client(
        region="us-east-1",
        endpoint="http://localhost:9000",
        s3client_config=config
    )
    print(f"  ✅ S3Client initialized successfully")
    print(f"  📍 Endpoint: {client.endpoint if hasattr(client, 'endpoint') else 'default'}")
    
    return client

# Test both options
print("\n" + "="*60)
print("Option 1: s3dlio (Recommended)")
print("="*60)
client1 = test_import("s3dlio")

print("\n" + "="*60)
print("Option 2: s3torchconnector (AWS Original)")
print("="*60)
client2 = test_import("s3torchconnector")

print("\n" + "="*60)
print("Summary")
print("="*60)
print("\n✅ storage_library configuration works!")
print("\nTo use in YAML config:")
print("\nreader:")
print("  storage_library: s3dlio  # High-performance zero-copy")
print("  # OR")
print("  storage_library: s3torchconnector  # AWS original")
print("\nSee configs/dlio/workload/pytorch_s3dlio.yaml for example")
print("="*60)
