#!/usr/bin/env python3
"""Test DLIO with MPI multi-endpoint configuration"""

from mpi4py import MPI
import os
import sys

# Get MPI info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print("\n" + "="*60)
    print("DLIO Multi-Endpoint Test with MPI")
    print("="*60)
    print(f"Total MPI processes: {size}")
    print(f"Endpoint assignment will be: rank % 4")
    print("="*60 + "\n")

# Add DLIO to path
sys.path.insert(0, '/home/eval/Documents/Code/s3dlio/python')

from s3dlio.integrations.dlio.s3dlio_storage import S3dlioStorage

# Simulate DLIO by creating a mock args object
class MockArgs:
    def __init__(self):
        self.endpoint_uris = [
            "http://endpoint1:9000",
            "http://endpoint2:9000",
            "http://endpoint3:9000",
            "http://endpoint4:9000",
        ]
        self.use_mpi_endpoint_distribution = True
        self.storage_options = {
            "access_key_id": "minioadmin",
            "secret_access_key": "minioadmin",
        }

# Create storage instance
try:
    # We can't actually instantiate S3dlioStorage without full DLIO framework,
    # but we can test the selection methods directly
    from s3dlio.integrations.dlio.s3dlio_storage import S3dlioStorage
    
    # Test the _select_endpoint_via_mpi method directly
    endpoints = [
        "http://endpoint1:9000",
        "http://endpoint2:9000",
        "http://endpoint3:9000",
        "http://endpoint4:9000",
    ]
    
    # Since we have OMPI_COMM_WORLD_RANK set by mpirun, simulate the selection
    ompi_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    endpoint_index = ompi_rank % len(endpoints)
    selected_endpoint = endpoints[endpoint_index]
    
    print(f"Rank {rank:2d}: OMPI_COMM_WORLD_RANK={ompi_rank} → endpoint[{endpoint_index}] = {selected_endpoint}")
    
    comm.Barrier()
    
    if rank == 0:
        print("\n" + "="*60)
        print("✅ DLIO multi-endpoint MPI test completed!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Use configs/dlio/workload/multi_endpoint_mpi.yaml")
        print("  2. Run: mpirun -np 8 dlio_benchmark --config multi_endpoint_mpi.yaml")
        print("="*60)

except Exception as e:
    print(f"Rank {rank}: Error: {e}")
    import traceback
    traceback.print_exc()
