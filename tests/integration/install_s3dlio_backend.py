#!/usr/bin/env python3
"""
Install s3dlio storage backend into DLIO

This script installs the s3dlio storage backend into the DLIO installation
in the virtual environment, making it available as a storage type.
"""

import os
import sys

# Add s3dlio to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../s3dlio/python'))

from s3dlio.integrations.dlio import install_s3dlio_storage

if __name__ == '__main__':
    # Find DLIO installation
    import dlio_benchmark
    dlio_path = os.path.dirname(dlio_benchmark.__file__)
    
    print(f"Installing s3dlio storage backend into DLIO at: {dlio_path}")
    print("=" * 60)
    
    # Install s3dlio storage
    installed_file = install_s3dlio_storage(dlio_path)
    
    print(f"\n✓ Installation complete!")
    print(f"\nYou can now use 'storage_type: s3dlio' in your DLIO configs.")
