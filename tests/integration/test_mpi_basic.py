#!/usr/bin/env python3
"""Test basic MPI functionality"""

from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Test environment variables set by mpirun
ompi_rank = os.environ.get('OMPI_COMM_WORLD_RANK', 'not set')
ompi_size = os.environ.get('OMPI_COMM_WORLD_SIZE', 'not set')

print(f"Rank {rank}/{size}: OMPI_COMM_WORLD_RANK={ompi_rank}, OMPI_COMM_WORLD_SIZE={ompi_size}")

# Test endpoint distribution logic
if rank == 0:
    print("\n" + "="*60)
    print("Testing Multi-Endpoint Distribution")
    print("="*60)

endpoints = [
    "http://endpoint1:9000",
    "http://endpoint2:9000",
    "http://endpoint3:9000",
    "http://endpoint4:9000",
]

endpoint_index = rank % len(endpoints)
my_endpoint = endpoints[endpoint_index]

print(f"Rank {rank:2d} → endpoint[{endpoint_index}] = {my_endpoint}")

comm.Barrier()

if rank == 0:
    print("="*60)
    print("✅ MPI test completed successfully!")
    print("="*60)
