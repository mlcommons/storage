#!/usr/bin/env python3
"""Test multi-endpoint integration with S3dlioStorage class"""

import os
import sys

# Add s3dlio to path
sys.path.insert(0, '/home/eval/Documents/Code/s3dlio/python')

def test_endpoint_selection_methods():
    print("="*60)
    print("Test 1: Endpoint Selection Methods")
    print("="*60)
    
    from s3dlio.integrations.dlio.s3dlio_storage import S3dlioStorage
    
    # Create a storage instance to access the methods
    storage = S3dlioStorage("file:///tmp/test")
    
    # Test MPI-based selection
    print("\n1. MPI-based endpoint selection:")
    os.environ['OMPI_COMM_WORLD_RANK'] = '5'
    endpoints = [
        "http://endpoint1:9000",
        "http://endpoint2:9000",
        "http://endpoint3:9000",
        "http://endpoint4:9000",
    ]
    selected = storage._select_endpoint_via_mpi(endpoints)
    print(f"   MPI Rank 5 → {selected}")
    print(f"   Expected: endpoint[1] (5 % 4 = 1)")
    assert selected == "http://endpoint2:9000", f"Expected endpoint2, got {selected}"
    print(f"   ✅ Correct endpoint selected!")
    
    # Clean up
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        del os.environ['OMPI_COMM_WORLD_RANK']
    
    # Test round-robin selection
    print("\n2. Round-robin endpoint selection:")
    pid = os.getpid()
    selected = storage._select_endpoint_via_strategy(endpoints, "round_robin")
    expected_idx = pid % len(endpoints)
    print(f"   PID {pid} → {selected}")
    print(f"   Expected: endpoint[{expected_idx}]")
    assert selected == endpoints[expected_idx], f"Expected endpoint[{expected_idx}], got {selected}"
    print(f"   ✅ Correct endpoint selected!")
    
    # Test random selection
    print("\n3. Random endpoint selection:")
    selected = storage._select_endpoint_via_strategy(endpoints, "random")
    print(f"   Selected: {selected}")
    assert selected in endpoints, f"Selected endpoint not in list: {selected}"
    print(f"   ✅ Valid endpoint selected!")

def test_config_based_usage():
    print("\n" + "="*60)
    print("Test 2: Config-Based Usage (How DLIO Uses It)")
    print("="*60)
    
    print("\nNote: S3dlioStorage gets config from DLIO framework via self._args")
    print("Config fields used:")
    print("  - endpoint_uris: List of endpoint URLs")
    print("  - load_balance_strategy: 'round_robin' or 'random'")
    print("  - use_mpi_endpoint_distribution: bool")
    print("  - storage_options: Dict with access keys, endpoint_url, etc.")
    print("\nSee configs/dlio/workload/multi_endpoint_*.yaml for examples")
    print("   ✅ Config structure documented")


def test_config_patterns():
    print("\n" + "="*60)
    print("Test 3: Common Configuration Patterns")
    print("="*60)
    
    patterns = [
        {
            "name": "Single MinIO",
            "yaml": """
reader:
  data_loader: s3dlio
  data_loader_root: s3://bucket/data
  storage_options:
    endpoint_url: http://minio:9000
    access_key_id: minioadmin
    secret_access_key: minioadmin
""",
        },
        {
            "name": "Multi-MinIO (s3dlio native)",
            "yaml": """
reader:
  data_loader: s3dlio
  data_loader_root: s3://bucket/data
  endpoint_uris:
    - http://minio1:9000
    - http://minio2:9000
    - http://minio3:9000
    - http://minio4:9000
  load_balance_strategy: round_robin
  storage_options:
    access_key_id: minioadmin
    secret_access_key: minioadmin
""",
        },
        {
            "name": "Multi-MinIO (MPI-based)",
            "yaml": """
reader:
  data_loader: s3dlio
  data_loader_root: s3://bucket/data
  endpoint_uris:
    - http://minio1:9000
    - http://minio2:9000
    - http://minio3:9000
    - http://minio4:9000
  use_mpi_endpoint_distribution: true
  storage_options:
    access_key_id: minioadmin
    secret_access_key: minioadmin
""",
        },
        {
            "name": "Hybrid Storage",
            "yaml": """
reader:
  data_loader: s3dlio
  data_loader_root: s3://bucket/data
  endpoint_uris:
    - http://minio1:9000
    - http://minio2:9000
  load_balance_strategy: round_robin
  checkpoint_folder: file:///nvme/checkpoints
  storage_options:
    access_key_id: minioadmin
    secret_access_key: minioadmin
""",
        },
    ]
    
    for i, pattern in enumerate(patterns, 1):
        print(f"\n{i}. {pattern['name']}:")
        print(f"   Config snippet:")
        for line in pattern['yaml'].strip().split('\n'):
            print(f"     {line}")

if __name__ == "__main__":
    try:
        test_endpoint_selection_methods()
        test_config_based_usage()
        test_config_patterns()
        
        print("\n" + "="*60)
        print("✅ All integration tests passed!")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

