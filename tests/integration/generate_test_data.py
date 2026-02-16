#!/usr/bin/env python3
"""Generate test dataset for DLIO benchmarking with file:// backend."""

import os
import numpy as np
from pathlib import Path

# Create test directory
test_dir = Path("/tmp/dlio-zerocopy-test")
test_dir.mkdir(exist_ok=True)

print(f"Creating test dataset in {test_dir}...")

# Generate small NPZ files (like ResNet50 training data)
num_files = 10
samples_per_file = 2
image_shape = (224, 224, 3)  # ResNet50 input size

for file_idx in range(num_files):
    samples = []
    labels = []
    
    for sample_idx in range(samples_per_file):
        # Generate random image (uint8, 0-255)
        img = np.random.randint(0, 256, image_shape, dtype=np.uint8)
        label = np.random.randint(0, 1000)  # ImageNet 1k classes
        
        samples.append(img)
        labels.append(label)
    
    # Save as NPZ
    file_path = test_dir / f"train_{file_idx:04d}.npz"
    np.savez_compressed(file_path, x=np.array(samples), y=np.array(labels))
    
    if file_idx == 0:
        print(f"  Sample file: {file_path}")
        print(f"    Shape: {samples[0].shape}, dtype: {samples[0].dtype}")
        print(f"    Size: {file_path.stat().st_size / 1024:.1f} KB")

print(f"\n✓ Created {num_files} NPZ files")
print(f"✓ {samples_per_file} samples per file")
print(f"✓ Total samples: {num_files * samples_per_file}")
print(f"\nDataset ready at: file://{test_dir}/")
print(f"\nUsage in DLIO config:")
print(f"  storage:")
print(f"    storage_type: s3dlio")
print(f"    storage_root: file://{test_dir}/")
