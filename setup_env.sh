#!/bin/bash
# MLPerf Storage Environment Setup
# Supports both uv and traditional venv/pip

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S3DLIO_PATH="${SCRIPT_DIR}/../s3dlio"

echo "=========================================="
echo "MLPerf Storage Environment Setup"
echo "=========================================="

# Detect if uv is available
if command -v uv &> /dev/null; then
    echo "✓ Using uv (recommended)"
    USE_UV=1
else
    echo "ℹ Using traditional venv/pip"
    USE_UV=0
fi

# Create and activate virtual environment
if [ $USE_UV -eq 1 ]; then
    # uv workflow
    if [ ! -d ".venv" ]; then
        echo "Creating uv virtual environment..."
        uv venv
    fi
    source .venv/bin/activate
    
    # Install s3dlio from local path first
    if [ -d "$S3DLIO_PATH" ]; then
        echo "Installing s3dlio from local path: $S3DLIO_PATH"
        uv pip install -e "$S3DLIO_PATH"
    else
        echo "WARNING: s3dlio not found at $S3DLIO_PATH"
        echo "Installing s3dlio from PyPI instead..."
        uv pip install s3dlio
    fi
    
    # Install mlpstorage with dependencies
    echo "Installing mlpstorage and dependencies..."
    uv pip install -e .
    
else
    # Traditional venv/pip workflow
    if [ ! -d ".venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    
    # Upgrade pip
    echo "Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install s3dlio from local path first
    if [ -d "$S3DLIO_PATH" ]; then
        echo "Installing s3dlio from local path: $S3DLIO_PATH"
        pip install -e "$S3DLIO_PATH"
    else
        echo "WARNING: s3dlio not found at $S3DLIO_PATH"
        echo "Installing s3dlio from PyPI instead..."
        pip install s3dlio
    fi
    
    # Install mlpstorage with dependencies
    echo "Installing mlpstorage and dependencies..."
    pip install -e .
fi

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Run benchmark: mlpstorage training run --model unet3d --accelerator-type h100 ..."
echo ""
echo "To use s3dlio backend, add to your DLIO config:"
echo "  storage:"
echo "    storage_type: s3dlio"
echo "    storage_root: s3://bucket/prefix"
echo ""
