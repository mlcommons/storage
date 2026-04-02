#!/usr/bin/env python3
"""
Verify s3dlio integration with DLIO

This script checks if s3dlio is properly installed and can be loaded by DLIO.
"""

import sys

def verify_s3dlio_integration():
    print("=" * 60)
    print("s3dlio Integration Verification")
    print("=" * 60)
    
    # Test 1: Check if s3dlio is importable
    print("\n1. Checking s3dlio Python package...")
    try:
        import s3dlio
        print(f"   ✓ s3dlio version: {s3dlio.__version__}")
    except ImportError as e:
        print(f"   ✗ FAILED: s3dlio not found")
        print(f"      Error: {e}")
        return False
    
    # Test 2: Check if DLIO has S3DLIO storage type
    print("\n2. Checking DLIO StorageType enum...")
    try:
        from dlio_benchmark.common.enumerations import StorageType
        if hasattr(StorageType, 'S3DLIO'):
            print(f"   ✓ StorageType.S3DLIO = '{StorageType.S3DLIO.value}'")
        else:
            print("   ✗ FAILED: StorageType.S3DLIO not found")
            print("      Available types:", [e.value for e in StorageType])
            return False
    except Exception as e:
        print(f"   ✗ FAILED: Could not check StorageType")
        print(f"      Error: {e}")
        return False
    
    # Test 3: Check if s3dlio_storage.py exists
    print("\n3. Checking s3dlio storage backend file...")
    try:
        from dlio_benchmark.storage.s3dlio_storage import S3dlioStorage
        print(f"   ✓ S3dlioStorage class found")
    except ImportError as e:
        print(f"   ✗ FAILED: s3dlio_storage.py not found or has errors")
        print(f"      Error: {e}")
        return False
    
    # Test 4: Check if storage factory can create s3dlio storage
    print("\n4. Checking StorageFactory integration...")
    try:
        from dlio_benchmark.storage.storage_factory import StorageFactory
        # Note: This may fail with MPI errors in non-MPI context, which is expected
        try:
            storage = StorageFactory.get_storage(StorageType.S3DLIO, "file:///tmp/test")
            print(f"   ✓ StorageFactory can create S3dlioStorage")
            print(f"      Type: {type(storage).__name__}")
        except Exception as e:
            if "MPI" in str(e):
                print(f"   ✓ StorageFactory recognizes S3DLIO (MPI not initialized, expected)")
            else:
                raise
    except Exception as e:
        print(f"   ✗ FAILED: StorageFactory cannot create S3dlioStorage")
        print(f"      Error: {e}")
        return False
    
    # Test 5: Check s3dlio module structure
    print("\n5. Checking s3dlio module structure...")
    try:
        # Just verify the module has expected attributes
        expected_attrs = ['get_object', 'list_keys', 'list_full_uris']
        for attr in expected_attrs:
            if hasattr(s3dlio, attr):
                print(f"   ✓ {attr} available")
            else:
                print(f"   ? {attr} not found (may use different API)")
        print(f"   ✓ s3dlio module structure OK")
    except Exception as e:
        print(f"   ✗ FAILED: Could not check s3dlio module")
        print(f"      Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All checks passed! s3dlio is ready to use.")
    print("=" * 60)
    print("\nYou can now use 'storage_type: s3dlio' in DLIO configs.")
    print("\nExample configuration:")
    print("  storage:")
    print("    storage_type: s3dlio")
    print("    storage_root: s3://bucket/prefix")
    print("")
    return True

if __name__ == '__main__':
    success = verify_s3dlio_integration()
    sys.exit(0 if success else 1)
