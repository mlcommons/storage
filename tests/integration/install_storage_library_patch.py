#!/usr/bin/env python3
"""
Install storage_library config support for DLIO benchmark.

This patches s3_torch_storage.py to support dynamic selection between:
  - s3torchconnector (AWS original)
  - s3dlio (zero-copy drop-in replacement)

Usage:
  python install_storage_library_patch.py          # Install patch
  python install_storage_library_patch.py restore  # Restore original
"""

import os
import shutil
import sys
from pathlib import Path

# Find DLIO installation
try:
    import dlio_benchmark
    dlio_path = Path(dlio_benchmark.__file__).parent
    storage_path = dlio_path / "storage"
    target_file = storage_path / "s3_torch_storage.py"
    backup_file = storage_path / "s3_torch_storage.py.orig"
except ImportError:
    print("❌ Error: dlio_benchmark not installed")
    print("   Install with: uv pip install dlio-benchmark")
    sys.exit(1)

# Patch file
patch_file = Path(__file__).parent / "patches" / "s3_torch_storage.py"

def install_patch():
    """Install the storage_library patch"""
    print("="*60)
    print("Installing storage_library Config Support")
    print("="*60)
    
    if not target_file.exists():
        print(f"❌ Target file not found: {target_file}")
        sys.exit(1)
    
    if not patch_file.exists():
        print(f"❌ Patch file not found: {patch_file}")
        sys.exit(1)
    
    # Backup original if not already backed up
    if not backup_file.exists():
        print(f"📦 Backing up original: {backup_file.name}")
        shutil.copy2(target_file, backup_file)
    else:
        print(f"ℹ️  Backup already exists: {backup_file.name}")
    
    # Install patch
    print(f"✅ Installing patched version")
    shutil.copy2(patch_file, target_file)
    
    print("="*60)
    print("✅ Installation Complete!")
    print("="*60)
    print("\nYou can now use 'storage_library' in YAML configs:")
    print("\nreader:")
    print("  storage_library: s3dlio           # Use s3dlio (zero-copy)")
    print("  # OR")
    print("  storage_library: s3torchconnector # Use AWS original (default)")
    print("\nSee configs/dlio/workload/pytorch_s3dlio.yaml for example")
    print("="*60)

def restore_original():
    """Restore the original file"""
    print("="*60)
    print("Restoring Original s3_torch_storage.py")
    print("="*60)
    
    if not backup_file.exists():
        print(f"❌ Backup not found: {backup_file}")
        print("   Patch may not have been installed")
        sys.exit(1)
    
    print(f"✅ Restoring from backup")
    shutil.copy2(backup_file, target_file)
    
    print(f"🗑️  Removing backup")
    backup_file.unlink()
    
    print("="*60)
    print("✅ Restore Complete!")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_original()
    else:
        install_patch()
