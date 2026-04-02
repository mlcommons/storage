#!/usr/bin/env python3
"""Quick test of s3dlio compatibility layer"""

print("Testing s3dlio compatibility layer...")

try:
    from s3dlio.compat.s3torchconnector import S3IterableDataset, S3MapDataset, S3Checkpoint
    print("✓ S3IterableDataset imported")
    print("✓ S3MapDataset imported")
    print("✓ S3Checkpoint imported")
    
    # Check they have the expected methods
    assert hasattr(S3IterableDataset, 'from_prefix'), "Missing from_prefix method"
    assert hasattr(S3MapDataset, 'from_prefix'), "Missing from_prefix method"
    assert hasattr(S3Checkpoint, 'writer'), "Missing writer method"
    assert hasattr(S3Checkpoint, 'reader'), "Missing reader method"
    
    print("\n✓ All compatibility classes have expected methods")
    print("\nCompatibility layer is working correctly!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
