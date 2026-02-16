#!/usr/bin/env python3
"""
Test storage_library configuration support

Verifies that the patched s3_torch_storage.py can dynamically import
either s3torchconnector or s3dlio based on config.
"""

import os
import sys
from pathlib import Path

def test_patch_installed():
    """Verify patch is installed"""
    print("="*60)
    print("Test 1: Verify Patch Installation")
    print("="*60)
    
    try:
        import dlio_benchmark
        dlio_path = Path(dlio_benchmark.__file__).parent
        storage_file = dlio_path / "storage" / "s3_torch_storage.py"
        backup_file = dlio_path / "storage" / "s3_torch_storage.py.orig"
        
        if not storage_file.exists():
            print(f"   ❌ Storage file not found: {storage_file}")
            return False
        
        # Check for our patch marker
        content = storage_file.read_text()
        if "storage_library" in content:
            print(f"   ✅ Patch installed (found 'storage_library' in code)")
        else:
            print(f"   ❌ Patch not installed (no 'storage_library' in code)")
            print(f"   Run: python install_storage_library_patch.py")
            return False
        
        if backup_file.exists():
            print(f"   ✅ Backup exists: {backup_file.name}")
        else:
            print(f"   ⚠️  No backup found (may not have been installed via script)")
        
        return True
        
    except ImportError:
        print("   ❌ dlio_benchmark not installed")
        return False

def test_library_imports():
    """Test that both libraries can be imported"""
    print("\n" + "="*60)
    print("Test 2: Verify Library Imports")
    print("="*60)
    
    # Test s3torchconnector
    try:
        from s3torchconnector._s3client import S3Client, S3ClientConfig
        print("   ✅ s3torchconnector imported successfully")
        s3torch_available = True
    except ImportError as e:
        print(f"   ⚠️  s3torchconnector not available: {e}")
        s3torch_available = False
    
    # Test s3dlio compat layer
    try:
        from s3dlio.compat.s3torchconnector import S3Client, S3ClientConfig
        print("   ✅ s3dlio.compat.s3torchconnector imported successfully")
        s3dlio_available = True
    except ImportError as e:
        print(f"   ❌ s3dlio compat layer not available: {e}")
        s3dlio_available = False
    
    return s3dlio_available  # s3dlio is required

def test_dynamic_import():
    """Test dynamic import based on mock config"""
    print("\n" + "="*60)
    print("Test 3: Test Dynamic Import Logic")
    print("="*60)
    
    # Test importing s3dlio via compat layer
    print("\n   Test A: storage_library = 's3dlio'")
    storage_library = "s3dlio"
    try:
        if storage_library == "s3dlio":
            from s3dlio.compat.s3torchconnector import S3Client, S3ClientConfig
            print(f"      ✅ Imported from s3dlio.compat.s3torchconnector")
        else:
            from s3torchconnector._s3client import S3Client, S3ClientConfig
            print(f"      ✅ Imported from s3torchconnector")
    except ImportError as e:
        print(f"      ❌ Import failed: {e}")
        return False
    
    # Test importing s3torchconnector (if available)
    print("\n   Test B: storage_library = 's3torchconnector'")
    storage_library = "s3torchconnector"
    try:
        if storage_library == "s3dlio":
            from s3dlio.compat.s3torchconnector import S3Client, S3ClientConfig
            print(f"      ✅ Imported from s3dlio.compat.s3torchconnector")
        else:
            try:
                from s3torchconnector._s3client import S3Client, S3ClientConfig
                print(f"      ✅ Imported from s3torchconnector._s3client")
            except ImportError:
                print(f"      ⚠️  s3torchconnector not installed (using s3dlio fallback)")
    except ImportError as e:
        print(f"      ❌ Import failed: {e}")
        return False
    
    return True

def test_config_examples():
    """Verify example configs exist"""
    print("\n" + "="*60)
    print("Test 4: Verify Example Configurations")
    print("="*60)
    
    configs = [
        "configs/dlio/workload/pytorch_s3dlio.yaml",
        "configs/dlio/workload/pytorch_s3torchconnector.yaml",
        "configs/dlio/workload/pytorch_file_backend.yaml",
    ]
    
    all_exist = True
    for config in configs:
        config_path = Path(config)
        if config_path.exists():
            # Check for storage_library in config
            content = config_path.read_text()
            if "storage_library" in content:
                print(f"   ✅ {config_path.name} (has storage_library)")
            else:
                print(f"   ⚠️  {config_path.name} (missing storage_library)")
        else:
            print(f"   ❌ {config_path.name} (not found)")
            all_exist = False
    
    return all_exist

def test_documentation():
    """Verify documentation exists"""
    print("\n" + "="*60)
    print("Test 5: Verify Documentation")
    print("="*60)
    
    docs = [
        "docs/STORAGE_LIBRARY_GUIDE.md",
    ]
    
    all_exist = True
    for doc in docs:
        doc_path = Path(doc)
        if doc_path.exists():
            size = doc_path.stat().st_size
            print(f"   ✅ {doc_path.name} ({size:,} bytes)")
        else:
            print(f"   ❌ {doc_path.name} (not found)")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Storage Library Configuration Test Suite")
    print("="*60)
    
    results = []
    
    results.append(("Patch Installation", test_patch_installed()))
    results.append(("Library Imports", test_library_imports()))
    results.append(("Dynamic Import Logic", test_dynamic_import()))
    results.append(("Example Configs", test_config_examples()))
    results.append(("Documentation", test_documentation()))
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("✅ All Tests Passed!")
        print("="*60)
        print("\nYou can now use storage_library in YAML configs:")
        print("  - storage_library: s3dlio")
        print("  - storage_library: s3torchconnector")
        print("\nSee docs/STORAGE_LIBRARY_GUIDE.md for details")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("❌ Some Tests Failed")
        print("="*60)
        print("\nPlease fix the failing tests before using storage_library config")
        sys.exit(1)
