# Parquet and Data Format Support

Guide to using Parquet, HDF5, TFRecord, and other data formats with byte-range reads.

---

## Overview

All 4 storage libraries support **byte-range reads**, enabling efficient access to columnar formats like Parquet without downloading entire files.

**Architecture:**
- **Storage Layer** (s3dlio, minio, etc.): Provides `get_range(uri, offset, length)` API
- **Application Layer** (PyArrow, h5py): Understands file format, calculates byte ranges
- **Benchmark Layer** (your code): Measures performance

**Key Insight:** Storage libraries are format-agnostic. They just move bytes. Format understanding lives in application libraries like PyArrow.

---

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 3: Benchmark/Application Layer (YOUR CODE)               │
│  • Decides WHICH columns to read                               │
│  • Measures performance and data transfer                       │
│  • Uses PyArrow to parse Parquet format                        │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 2: Application Format Layer (PyArrow)                    │
│  • Understands Parquet structure (footer, row groups, chunks)  │
│  • Reads footer to get column chunk byte ranges                │
│  • Calculates WHICH byte ranges to request                     │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1: Storage Layer (s3dlio, minio, s3torchconnector, etc.) │
│  • Provides byte-range API: get_range(uri, offset, length)     │
│  • Translates to S3/Azure/GCS GetObject with Range header      │
│  • Format-agnostic (doesn't know about Parquet structure)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Supported Formats

| Format | Byte-Range Critical? | Library | Notes |
|--------|---------------------|---------|-------|
| **Parquet** | ✅ **YES** | PyArrow | Columnar - read only needed columns |
| **HDF5** | ✅ **YES** | h5py | Hierarchical - read specific datasets |
| **TFRecord** | ⚠️ Maybe | TensorFlow | Sequential but index helps |
| **NPZ** | ⚠️ Maybe | NumPy | ZIP-based - footer has directory |

---

## Byte-Range APIs by Library

### s3dlio
```python
# Full object
data = s3dlio.get('s3://bucket/file.parquet')

# Byte range
chunk = s3dlio.get_range('s3://bucket/file.parquet', offset=5001, length=999)
```

### minio
```python
# Byte range
response = client.get_object('bucket', 'file.parquet', offset=5001, length=999)
data = response.read()
```

### s3torchconnector
```python
# Byte range (start/end inclusive)
reader = client.get_object('bucket', 'file.parquet', start=5001, end=5999)
data = reader.read()
```

### azstoragetorch
```python
# Byte range via seek + read
blob = BlobIO(container, 'file.parquet', 'r')
blob.seek(5001)
data = blob.read(999)
```

---

## Parquet Efficiency Example

**Scenario:** 100 GB Parquet file with 50 columns, you only need 2 columns.

**WITHOUT byte-ranges (inefficient):**
```python
table = pq.read_table('s3://bucket/train.parquet')  # Read all 100 GB
features = table['image_data']
labels = table['label']
```

**WITH byte-ranges (efficient):**
```python
table = pq.read_table('s3://bucket/train.parquet',
                      columns=['image_data', 'label'])  # Read only 4 GB!
```

**Savings:** 96 GB of data transfer eliminated (96% reduction)!

---

## Working Example

See **`parquet_byte_range_example.py`** for complete working demonstration:

**What it shows:**
- Create sample Parquet file
- Read footer only (99.5% data savings)
- Read specific columns with PyArrow
- Benchmark full vs partial reads
- Demonstrate all 3 layers working together

**Run it:**
```bash
# Install dependencies
pip install pyarrow s3dlio

# Run example (local file)
python parquet_byte_range_example.py

# Run with S3
export AWS_ENDPOINT_URL=http://localhost:9000
python parquet_byte_range_example.py --uri s3://bucket/test.parquet
```

**Expected output:**
```
Creating Parquet file: file:///tmp/test.parquet
File size: 308,941 bytes

=== Footer-Only Read (Byte-Range) ===
Read 1,410 bytes (0.5% of file)
Data transfer savings: 99.5%

=== Column Subset Read ===
Reading columns: ['feature_1', 'label']
Read 45,234 bytes (14.6% of file)
Data transfer savings: 85.4%
```

---

## Integration with Benchmarks

### Add Parquet to Benchmark Tools

To benchmark Parquet performance across libraries:

1. **Generate Parquet files:**
   ```python
   # See parquet_byte_range_example.py create_sample_parquet()
   ```

2. **Benchmark full read:**
   ```python
   # Use benchmark_read_comparison.py with Parquet files
   ```

3. **Benchmark column-subset reads:**
   ```python
   # Modify benchmarks to use PyArrow with columns parameter
   table = pq.read_table(uri, columns=['col1', 'col2'])
   ```

### Measuring Actual Bytes Transferred

To track actual network I/O:

```python
# Instrument storage layer to count bytes
# See parquet_byte_range_example.py for example
```

---

## HDF5 Support

HDF5 files also benefit from byte-range reads:

```python
import h5py

# Read specific dataset (not entire file)
with h5py.File('s3://bucket/data.h5', 'r') as f:
    dataset = f['images'][0:100]  # Read first 100 only
```

**Note:** Requires h5py with S3 support (via s3dlio or s3fs)

---

## Format Support in s3dlio

s3dlio has **built-in support** for some formats:

### NPZ (NumPy)
```python
import s3dlio

# Build NPZ file
s3dlio.build_npz(uri, arrays={'data': array1, 'labels': array2})

# Read arrays
arrays = s3dlio.read_npz_array(uri, array_name='data')
```

### HDF5
```python
# Build HDF5 file
s3dlio.build_hdf5(uri, datasets={'data': array1, 'labels': array2})
```

### TFRecord
```python
# Build TFRecord with index
s3dlio.build_tfrecord_with_index(uri, records=[...])
```

**See:** s3dlio documentation for complete format support

---

## No Changes Needed to s3dlio

**Important:** You do **NOT** need to add Parquet support to s3dlio.

**Why?**
- s3dlio already provides `get_range()` API (format-agnostic)
- PyArrow handles Parquet structure (application layer)
- All storage libraries work the same way for Parquet

**What you DO need:**
- PyArrow library installed
- Use PyArrow's `read_table()` with `columns` parameter
- PyArrow automatically uses storage byte-range APIs

---

## Performance Tips

### 1. Read Only Needed Columns
```python
# BAD: Read all columns
table = pq.read_table(uri)

# GOOD: Read specific columns
table = pq.read_table(uri, columns=['feature1', 'label'])
```

### 2. Use Row Group Filtering
```python
# Read specific row groups
table = pq.read_table(uri, 
                      columns=['feature1', 'label'],
                      filters=[('label', '==', 5)])
```

### 3. Benchmark Data Transfer
```python
# Measure actual bytes transferred vs file size
# See parquet_byte_range_example.py for implementation
```

---

## Troubleshooting

### Problem: PyArrow reads entire file

**Cause:** PyArrow doesn't have byte-range access to storage

**Solution:** Use PyArrow with S3FileSystem:
```python
from pyarrow.fs import S3FileSystem

fs = S3FileSystem(endpoint_override='http://localhost:9000')
table = pq.read_table('bucket/file.parquet', 
                      filesystem=fs,
                      columns=['col1'])
```

### Problem: Slow Parquet reads

**Check:**
1. Are you using `columns` parameter? (Should see < 20% data transfer)
2. Is network fast enough? (Run `iperf3`)
3. Is Parquet file well-structured? (Check row group size)

---

## Related Documentation

- **[Storage Libraries](STORAGE_LIBRARIES.md)** - All 4 libraries support byte-ranges
- **[Performance Testing](PERFORMANCE_TESTING.md)** - Benchmark byte-range efficiency
- **[Quick Start](QUICK_START.md)** - Get started quickly

---

## Summary

- **All 4 libraries** (s3dlio, minio, s3torchconnector, azstoragetorch) support byte-range reads
- **PyArrow** handles Parquet structure, calculates byte ranges
- **Storage libraries** are format-agnostic, just provide `get_range()` API
- **No s3dlio changes needed** for Parquet support
- **See `parquet_byte_range_example.py`** for working demonstration

**For Parquet:** Use PyArrow with `columns` parameter → automatic byte-range optimization!
