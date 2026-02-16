#!/usr/bin/env python3
"""
Parquet Byte-Range Read Example

Demonstrates how to efficiently read Parquet files using byte-range requests.
Shows where byte-range information is specified and how libraries cooperate.

Architecture:
- Storage Layer (s3dlio): Provides get_range(uri, offset, length) API
- Application Layer (PyArrow): Knows Parquet structure, calculates byte ranges
- Benchmark Layer (this file): Measures performance and efficiency
"""

import time
import struct
from typing import List, Tuple, Dict

# Storage layer - provides byte-range API
import s3dlio

# Application layer - understands Parquet format
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False
    print("⚠️  PyArrow not installed: pip install pyarrow")


def create_sample_parquet(uri: str, num_rows: int = 1000) -> Dict[str, any]:
    """
    Create a sample Parquet file and return metadata.
    
    Returns:
        dict: File metadata including size and column info
    """
    if not HAVE_PYARROW:
        raise ImportError("PyArrow required to create Parquet files")
    
    # Create sample data with multiple columns (like a real ML dataset)
    data = {
        'id': list(range(num_rows)),
        'feature_1': [i * 1.5 for i in range(num_rows)],
        'feature_2': [i * 2.0 for i in range(num_rows)],
        'feature_3': [i * 3.0 for i in range(num_rows)],
        'label': [i % 10 for i in range(num_rows)],
        'metadata': [f"row_{i}" for i in range(num_rows)],
    }
    
    # Create PyArrow table
    table = pa.table(data)
    
    # Write to bytes buffer
    import io
    buf = io.BytesIO()
    pq.write_table(table, buf)
    parquet_bytes = buf.getvalue()
    
    # Upload to storage
    s3dlio.put_bytes(uri, parquet_bytes)
    
    # Get file metadata
    meta = s3dlio.stat(uri)
    
    return {
        'uri': uri,
        'size': meta['size'],
        'num_rows': num_rows,
        'num_columns': len(data),
        'columns': list(data.keys()),
    }


def read_parquet_footer(uri: str) -> Tuple[bytes, Dict]:
    """
    Read Parquet footer using byte-range request.
    
    Parquet footer is at the END of file and contains:
    - Schema
    - Row group metadata
    - Column chunk byte ranges
    
    Returns:
        tuple: (footer_bytes, metadata_dict)
    """
    # Get file size
    meta = s3dlio.stat(uri)
    file_size = meta['size']
    
    print(f"\n📊 Reading Parquet footer...")
    print(f"   File size: {file_size:,} bytes")
    
    # Parquet footer format:
    # [...data...] [footer_metadata] [4-byte footer length] [4-byte "PAR1" magic]
    
    # Step 1: Read last 8 bytes to get footer length
    magic_and_length = s3dlio.get_range(uri, offset=file_size - 8, length=8)
    magic_and_length = bytes(magic_and_length)
    
    # Parse footer length (4 bytes before final magic)
    footer_length = struct.unpack('<I', magic_and_length[:4])[0]
    magic = magic_and_length[4:8]
    
    if magic != b'PAR1':
        raise ValueError(f"Invalid Parquet file (magic={magic})")
    
    print(f"   Footer length: {footer_length:,} bytes")
    
    # Step 2: Read full footer metadata
    footer_offset = file_size - 8 - footer_length
    footer_bytes = s3dlio.get_range(uri, offset=footer_offset, length=footer_length)
    footer_bytes = bytes(footer_bytes)
    
    print(f"   Footer read: {len(footer_bytes):,} bytes")
    print(f"   Bytes transferred: {8 + len(footer_bytes):,} / {file_size:,} ({(8 + len(footer_bytes)) / file_size * 100:.1f}%)")
    
    return footer_bytes, {
        'file_size': file_size,
        'footer_length': footer_length,
        'footer_offset': footer_offset,
    }


def benchmark_full_read(uri: str) -> Dict:
    """Read entire Parquet file (baseline)."""
    print(f"\n🔍 Benchmark: Full File Read")
    
    start = time.time()
    data = s3dlio.get(uri)
    elapsed = time.time() - start
    
    bytes_read = len(bytes(data))
    throughput = bytes_read / (1024**3) / elapsed if elapsed > 0 else 0
    
    print(f"   Bytes read: {bytes_read:,}")
    print(f"   Time: {elapsed:.3f} seconds")
    print(f"   Throughput: {throughput:.2f} GB/s")
    
    return {
        'method': 'full_read',
        'bytes_read': bytes_read,
        'time': elapsed,
        'throughput': throughput,
    }


def benchmark_footer_only(uri: str) -> Dict:
    """Read only Parquet footer (metadata extraction)."""
    print(f"\n🔍 Benchmark: Footer-Only Read")
    
    start = time.time()
    footer_bytes, meta = read_parquet_footer(uri)
    elapsed = time.time() - start
    
    bytes_read = 8 + len(footer_bytes)  # magic/length + footer
    throughput = bytes_read / (1024**3) / elapsed if elapsed > 0 else 0
    savings = (1 - bytes_read / meta['file_size']) * 100
    
    print(f"   Bytes read: {bytes_read:,} ({savings:.1f}% savings)")
    print(f"   Time: {elapsed:.3f} seconds")
    print(f"   Throughput: {throughput:.2f} GB/s")
    
    return {
        'method': 'footer_only',
        'bytes_read': bytes_read,
        'time': elapsed,
        'throughput': throughput,
        'savings_pct': savings,
    }


def benchmark_column_subset(uri: str, columns: List[str]) -> Dict:
    """
    Read only specific columns using PyArrow + s3dlio.
    
    This is where PyArrow determines the byte ranges based on footer metadata,
    then uses the storage layer's byte-range API to fetch only needed chunks.
    """
    if not HAVE_PYARROW:
        print("⚠️  Skipping column subset benchmark (PyArrow not available)")
        return {}
    
    print(f"\n🔍 Benchmark: Column Subset Read ({', '.join(columns)})")
    
    # PyArrow will:
    # 1. Read footer to get column chunk locations
    # 2. Request only byte ranges for specified columns
    # 3. Use storage layer's byte-range API (S3's GetObject with Range header)
    
    start = time.time()
    
    # Parse URI to get bucket/key for PyArrow
    if uri.startswith('file://'):
        # Local file - PyArrow can read directly
        file_path = uri.replace('file://', '')
        table = pq.read_table(file_path, columns=columns)
    else:
        # Object storage - need filesystem adapter
        # For now, read full object and filter columns
        data = s3dlio.get(uri)
        import io
        buf = io.BytesIO(bytes(data))
        table = pq.read_table(buf, columns=columns)
    
    elapsed = time.time() - start
    
    # Note: We can't easily measure actual byte-range requests without
    # instrumenting the storage layer. In production, you'd add logging
    # to s3dlio.get_range() to track actual bytes transferred.
    
    print(f"   Rows read: {len(table):,}")
    print(f"   Columns: {table.column_names}")
    print(f"   Time: {elapsed:.3f} seconds")
    print(f"   Note: PyArrow handles byte-range logic internally")
    
    return {
        'method': 'column_subset',
        'columns': columns,
        'rows': len(table),
        'time': elapsed,
    }


def main():
    """Demonstrate Parquet byte-range reads with s3dlio."""
    
    print("=" * 70)
    print("Parquet Byte-Range Read Benchmarks")
    print("=" * 70)
    
    # Configuration
    uri = "file:///tmp/sample_parquet_data.parquet"
    num_rows = 10000
    
    # Create sample Parquet file
    print("\n📝 Creating sample Parquet file...")
    meta = create_sample_parquet(uri, num_rows)
    print(f"   URI: {meta['uri']}")
    print(f"   Size: {meta['size']:,} bytes")
    print(f"   Rows: {meta['num_rows']:,}")
    print(f"   Columns: {', '.join(meta['columns'])}")
    
    # Benchmark 1: Full file read (baseline)
    result_full = benchmark_full_read(uri)
    
    # Benchmark 2: Footer-only read (metadata extraction)
    result_footer = benchmark_footer_only(uri)
    
    # Benchmark 3: Column subset (realistic ML workflow)
    if HAVE_PYARROW:
        result_columns = benchmark_column_subset(uri, columns=['feature_1', 'label'])
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary: Byte-Range Benefits")
    print("=" * 70)
    print(f"\n📊 Data Transfer Savings:")
    print(f"   Full file:    {result_full['bytes_read']:,} bytes (baseline)")
    print(f"   Footer only:  {result_footer['bytes_read']:,} bytes ({result_footer['savings_pct']:.1f}% savings)")
    
    print(f"\n⚡ Performance Impact:")
    print(f"   Full read: {result_full['time']:.3f}s")
    print(f"   Footer:    {result_footer['time']:.3f}s ({result_footer['time'] / result_full['time'] * 100:.1f}% of full read time)")
    
    print("\n✅ Key Takeaways:")
    print("   1. Byte-range reads reduce data transfer (critical for large files)")
    print("   2. Footer-only reads enable fast metadata extraction")
    print("   3. Column subsets avoid transferring unused data")
    print("   4. s3dlio provides get_range() API - PyArrow uses it internally")
    print("   5. Your benchmarks can measure byte-range efficiency")
    
    print("\n📍 Where Byte-Range Info is Specified:")
    print("   - Storage Layer (s3dlio):  get_range(uri, offset, length)")
    print("   - Application Layer (PyArrow): Calculates byte ranges from footer")
    print("   - Benchmark Layer (yours): Measures performance and savings")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
