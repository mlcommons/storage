#!/usr/bin/env python3
"""High-performance S3 read benchmark with library comparison.

Supports comparison between:
- s3dlio: Zero-copy reads using BytesView (S3/Azure/GCS/file/direct)
- s3torchconnector: AWS official library
- minio: MinIO Python SDK (S3-compatible)
- azstoragetorch: Azure Storage for PyTorch (BlobIO API)

Target: 20-30 GB/s read throughput with 200+ GB total data.

Example usage:
    # Compare all installed libraries
    python benchmark_read_comparison.py --compare-all --endpoint http://localhost:9000 --bucket benchmark
    
    # Compare specific libraries
    python benchmark_read_comparison.py --compare s3dlio minio --endpoint http://localhost:9000
    
    # Test single library  
    python benchmark_read_comparison.py --library s3dlio --endpoint http://localhost:9000
    python benchmark_read_comparison.py --library minio --endpoint http://localhost:9000
    
    # Legacy 2-way comparison
    python benchmark_read_comparison.py --compare-libraries --endpoint http://localhost:9000
"""

import argparse
import time
import sys
import os
from io import BytesIO
from urllib.parse import urlparse

# Will import libraries based on --library flag
s3dlio = None
S3Client = None
S3ClientConfig = None
Minio = None
BlobIO = None


def test_read_performance(endpoint, bucket, num_files, file_size, library_name):
    """Read benchmark for a single library."""
    use_s3dlio = (library_name == "s3dlio")
    
    file_size_mb = file_size / (1024 * 1024)
    total_gb = (num_files * file_size) / (1024**3)
    
    print("=" * 70)
    print(f"Read Performance Test - {library_name.upper()}")
    print("=" * 70)
    print(f"Library:     {library_name}")
    print(f"Endpoint:    {endpoint}")
    print(f"Bucket:      {bucket}")
    print(f"Files:       {num_files:,}")
    print(f"File Size:   {file_size_mb:.0f} MB ({file_size:,} bytes)")
    print(f"Total Data:  {total_gb:.2f} GB")
    print("=" * 70)
    
    # Setup client based on library
    client = None
    if library_name == "s3torchconnector":
        if endpoint.startswith("s3://"):
            from s3torchconnector import S3ClientConfig as S3ClientConfigClass
            config = S3ClientConfigClass(region="us-east-1")
        else:
            endpoint_url = endpoint if endpoint.startswith("http") else f"http://{endpoint}"
            from s3torchconnector import S3ClientConfig as S3ClientConfigClass
            config = S3ClientConfigClass(endpoint_url=endpoint_url, region="us-east-1")
        
        from s3torchconnector import S3Client as S3ClientClass
        client = S3ClientClass(config)
    
    elif library_name == "minio":
        # MinIO: S3-compatible API
        parsed = urlparse(endpoint if endpoint.startswith("http") else f"http://{endpoint}")
        
        # Get credentials from environment or use defaults for local testing
        import os
        access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
        
        # Create MinIO client
        client = Minio(
            parsed.netloc,
            access_key=access_key,
            secret_key=secret_key,
            secure=(parsed.scheme == "https")
        )
    
    # Read files
    print(f"\nReading {num_files:,} files from storage...")
    
    start_time = time.time()
    total_bytes_read = 0
    
    for i in range(num_files):
        if use_s3dlio:
            # s3dlio: ZERO-COPY read (returns BytesView)
            uri = f"{endpoint}/{bucket}/test-data/file_{i:06d}.bin"
            data = s3dlio.get(uri)
            
            # Access via memoryview (zero-copy)
            view = memoryview(data)
            total_bytes_read += len(view)
        
        elif library_name == "s3torchconnector":
            # s3torchconnector: Standard read
            key = f"test-data/file_{i:06d}.bin"
            obj = client.get_object(bucket, key)
            data = obj.read()
            total_bytes_read += len(data)
        
        elif library_name == "minio":
            # MinIO: S3-compatible API
            object_name = f"test-data/file_{i:06d}.bin"
            response = client.get_object(bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            total_bytes_read += len(data)
        
        elif library_name == "azstoragetorch":
            # Azure Blob Storage: BlobIO file-like API
            blob_name = f"test-data/file_{i:06d}.bin"
            if endpoint.endswith("/"):
                blob_url = f"{endpoint}{bucket}/{blob_name}"
            else:
                blob_url = f"{endpoint}/{bucket}/{blob_name}"
            
            with BlobIO(blob_url, "rb") as f:
                data = f.read()
            total_bytes_read += len(data)
        
        else:
            raise ValueError(f"Unknown library: {library_name}")
        
        # Progress update every 10%
        if (i + 1) % max(1, num_files // 10) == 0:
            elapsed = time.time() - start_time
            progress = (i + 1) / num_files
            current_throughput = (total_bytes_read / (1024**3)) / elapsed
            print(f"  Progress: {progress*100:5.1f}% | {i+1:,}/{num_files:,} files | {current_throughput:.2f} GB/s")
    
    total_time = time.time() - start_time
    throughput_gbs = total_gb / total_time
    files_per_sec = num_files / total_time
    
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total Data:       {total_gb:.2f} GB")
    print(f"Total Time:       {total_time:.2f} seconds")
    print(f"Throughput:       {throughput_gbs:.2f} GB/s")
    print(f"Files/second:     {files_per_sec:.1f}")
    print(f"Avg per file:     {total_time/num_files*1000:.2f} ms")
    
    # Performance assessment
    if throughput_gbs >= 30:
        print(f"\n🏆 EXCELLENT: {throughput_gbs:.2f} GB/s (Target: 20-30 GB/s)")
    elif throughput_gbs >= 20:
        print(f"\n✅ GOOD: {throughput_gbs:.2f} GB/s (Within target range)")
    elif throughput_gbs >= 10:
        print(f"\n⚠️  MODERATE: {throughput_gbs:.2f} GB/s (Below 20 GB/s target)")
    else:
        print(f"\n❌ LOW: {throughput_gbs:.2f} GB/s (Needs investigation)")
    
    print("=" * 70)
    print()
    
    return {
        'library': library_name,
        'throughput_gbs': throughput_gbs,
        'total_time': total_time,
        'files_per_sec': files_per_sec,
        'total_gb': total_gb,
        'num_files': num_files,
        'file_size_mb': file_size_mb
    }


def import_library(library_name):
    """Import a specific library and return success status."""
    global s3dlio, S3Client, S3ClientConfig, Minio, BlobIO
    
    if library_name == "s3dlio":
        try:
            import s3dlio as s3dlio_mod
            s3dlio = s3dlio_mod
            return True
        except ImportError:
            print(f"❌ ERROR: s3dlio not installed")
            print("Install: uv pip install s3dlio")
            return False
    
    elif library_name == "s3torchconnector":
        try:
            from s3torchconnector import S3Client as S3ClientClass, S3ClientConfig as S3ClientConfigClass
            S3Client = S3ClientClass
            S3ClientConfig = S3ClientConfigClass
            return True
        except ImportError:
            print(f"❌ ERROR: s3torchconnector not installed")
            print("Install: uv pip install s3torchconnector")
            return False
    
    elif library_name == "minio":
        try:
            from minio import Minio as MinioClass
            Minio = MinioClass
            globals()['Minio'] = Minio
            return True
        except ImportError:
            print(f"❌ ERROR: minio not installed")
            print("Install: pip install minio")
            return False
    
    elif library_name == "azstoragetorch":
        try:
            from azstoragetorch.io import BlobIO as BlobIOClass
            BlobIO = BlobIOClass
            globals()['BlobIO'] = BlobIO
            return True
        except ImportError:
            print(f"❌ ERROR: azstoragetorch not installed")
            print("Install: pip install azstoragetorch")
            return False
    
    else:
        print(f"❌ ERROR: Unknown library '{library_name}'")
        return False


def compare_libraries(endpoint, bucket, num_files, file_size, libraries_to_test=None):
    """Run multiple libraries back-to-back for direct comparison.
    
    Args:
        libraries_to_test: List of library names to test (e.g., ['s3dlio', 'minio']).
                          If None, defaults to ['s3dlio', 's3torchconnector'] for backward compatibility.
    """
    if libraries_to_test is None:
        libraries_to_test = ['s3dlio', 's3torchconnector']
    
    print("\n" + "=" * 80)
    if len(libraries_to_test) == 2:
        print("HEAD-TO-HEAD LIBRARY COMPARISON MODE (READS)")
    else:
        print(f"MULTI-LIBRARY COMPARISON MODE ({len(libraries_to_test)} libraries, READS)")
    print("=" * 80)
    print(f"\nTesting libraries: {', '.join(libraries_to_test)}")
    print(f"Total test: {num_files:,} files × {file_size/(1024**2):.0f} MB = {num_files*file_size/(1024**3):.1f} GB per library")
    print(f"Combined: {len(libraries_to_test)*num_files*file_size/(1024**3):.1f} GB total data read")
    print()
    
    results = {}
    
    # Test each library
    for i, lib in enumerate(libraries_to_test, 1):
        print(f"\n>>> TESTING {lib.upper()} ({i}/{len(libraries_to_test)}) <<<\n")
        try:
            results[lib] = test_read_performance(endpoint, bucket, num_files, file_size, lib)
            if i < len(libraries_to_test):
                time.sleep(2)  # Brief pause between tests
        except Exception as e:
            print(f"❌ Error testing {lib}: {e}")
            print(f"Skipping {lib} and continuing...\n")
            continue
    
    if not results:
        print("\n❌ No libraries completed successfully!")
        return results
    
    # Print detailed comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nTest Configuration:")
    print(f"  Files:       {num_files:,}")
    print(f"  File Size:   {file_size/(1024**2):.0f} MB")
    
    # Get total_gb from any result
    first_result = next(iter(results.values()))
    print(f"  Total Data:  {first_result['total_gb']:.2f} GB (per library)")
    
    # Dynamic table with variable column count
    lib_names = list(results.keys())
    col_width = 18
    metric_width = 30
    
    # Table header
    header = f"\n{'Metric':<{metric_width}}"
    for lib in lib_names:
        header += f" {lib:<{col_width}}"
    print(header)
    print("-" * (metric_width + col_width * len(lib_names)))
    
    # Throughput row
    row = f"{'Throughput (GB/s)':<{metric_width}}"
    for lib in lib_names:
        row += f" {results[lib]['throughput_gbs']:<{col_width}.2f}"
    print(row)
    
    # Total time row
    row = f"{'Total Time (seconds)':<{metric_width}}"
    for lib in lib_names:
        row += f" {results[lib]['total_time']:<{col_width}.2f}"
    print(row)
    
    # Files/second row
    row = f"{'Files/second':<{metric_width}}"
    for lib in lib_names:
        row += f" {results[lib]['files_per_sec']:<{col_width}.1f}"
    print(row)
    
    print("-" * (metric_width + col_width * len(lib_names)))
    
    # Find fastest library
    fastest_lib = max(results.items(), key=lambda x: x[1]['throughput_gbs'])
    fastest_name = fastest_lib[0]
    fastest_throughput = fastest_lib[1]['throughput_gbs']
    
    print(f"\n🏁 FINAL VERDICT:")
    print(f"   Fastest: {fastest_name.upper()} at {fastest_throughput:.2f} GB/s")
    
    # Show speedup comparisons
    if len(results) >= 2:
        print(f"\n   Relative Performance:")
        for lib in lib_names:
            if lib != fastest_name:
                speedup = fastest_throughput / results[lib]['throughput_gbs']
                print(f"   • {fastest_name} is {speedup:.2f}x faster than {lib}")
    
    print("\n" + "=" * 80)
    print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="S3 read benchmark with library comparison (s3dlio vs s3torchconnector)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Head-to-head comparison (RECOMMENDED)
  python benchmark_read_comparison.py --compare-libraries --endpoint http://localhost:9000 --bucket benchmark
  
  # Test single library
  python benchmark_read_comparison.py --library s3dlio --endpoint http://localhost:9000
  python benchmark_read_comparison.py --library s3torchconnector --endpoint http://localhost:9000
  
  # Large-scale test (200 GB)
  python benchmark_read_comparison.py --files 2000 --size 100 --compare-libraries
        """
    )
    
    parser.add_argument("--library", 
                        choices=["s3dlio", "s3torchconnector", "minio", "azstoragetorch"], 
                        default="s3dlio",
                        help="Library to use (default: s3dlio)")
    parser.add_argument("--compare-libraries", action="store_true",
                        help="Run s3dlio vs s3torchconnector (legacy 2-way comparison)")
    parser.add_argument("--compare", nargs="+", metavar="LIB",
                        help="Compare specific libraries (e.g., --compare s3dlio minio azstoragetorch)")
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all installed libraries")
    
    parser.add_argument("--endpoint", default="s3://", help="S3 endpoint URL (default: s3://)")
    parser.add_argument("--bucket", default="benchmark", help="S3 bucket name (default: benchmark)")
    parser.add_argument("--files", type=int, default=2000,
                        help="Number of files to read (default: 2000 = 200 GB with 100 MB files)")
    parser.add_argument("--size", type=int, default=100,
                        help="Expected file size in MB (default: 100 MB)")
    
    args = parser.parse_args()
    
    # Determine which libraries to test
    libraries_to_test = []
    
    if args.compare_all:
        # Test all installed libraries
        print("🔍 Checking for installed libraries...")
        all_libs = ["s3dlio", "s3torchconnector", "minio", "azstoragetorch"]
        for lib in all_libs:
            if import_library(lib):
                libraries_to_test.append(lib)
                print(f"  ✅ {lib}")
            else:
                print(f"  ⏭️  {lib} not installed, skipping")
        
        if not libraries_to_test:
            print("\n❌ ERROR: No libraries installed!")
            print("Install at least one: uv pip install s3dlio s3torchconnector minio azstoragetorch")
            sys.exit(1)
        
        print(f"\nWill test {len(libraries_to_test)} libraries: {', '.join(libraries_to_test)}\n")
    
    elif args.compare:
        # Test specific libraries
        print("🔍 Checking for requested libraries...")
        for lib in args.compare:
            if lib not in ["s3dlio", "s3torchconnector", "minio", "azstoragetorch"]:
                print(f"❌ ERROR: Unknown library '{lib}'")
                print("Valid options: s3dlio, s3torchconnector, minio, azstoragetorch")
                sys.exit(1)
            
            if import_library(lib):
                libraries_to_test.append(lib)
                print(f"  ✅ {lib}")
            else:
                print(f"  ❌ {lib} not installed")
                print(f"     Install: uv pip install {lib}")
                sys.exit(1)
        
        print(f"\nWill test: {', '.join(libraries_to_test)}\n")
    
    elif args.compare_libraries:
        # Legacy mode: s3dlio vs s3torchconnector
        print("🔍 Checking for s3dlio and s3torchconnector...")
        libraries_to_test = []
        
        if import_library("s3dlio"):
            libraries_to_test.append("s3dlio")
            print("  ✅ s3dlio")
        else:
            print("  ❌ s3dlio not installed")
            sys.exit(1)
        
        if import_library("s3torchconnector"):
            libraries_to_test.append("s3torchconnector")
            print("  ✅ s3torchconnector")
        else:
            print("  ❌ s3torchconnector not installed")
            sys.exit(1)
        
        print()
    
    else:
        # Single library mode
        print(f"🔍 Checking for {args.library}...")
        if not import_library(args.library):
            sys.exit(1)
        libraries_to_test = [args.library]
        print(f"  ✅ {args.library}\n")
    
    file_size = args.size * 1024 * 1024  # Convert MB to bytes
    total_gb = (args.files * file_size) / (1024**3)
    
    # Validate parameters
    if args.size >= 16:
        print(f"✅ File size: {args.size} MB (meets recommendation: ≥16 MB)")
    else:
        print(f"⚠️  File size: {args.size} MB (below recommended 16 MB)")
    
    if total_gb >= 200:
        print(f"✅ Total data: {total_gb:.1f} GB (meets recommendation: ≥200 GB)")
    else:
        print(f"⚠️  Total data: {total_gb:.1f} GB (below recommended 200 GB)")
    
    print()
    
    # Run tests
    if len(libraries_to_test) > 1:
        # Comparison mode: run multiple libraries
        compare_libraries(args.endpoint, args.bucket, args.files, file_size, libraries_to_test)
    else:
        # Single library mode
        lib = libraries_to_test[0]
        test_read_performance(args.endpoint, args.bucket, args.files, file_size, lib)


if __name__ == "__main__":
    main()
