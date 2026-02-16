#!/usr/bin/env python3
"""Test multi-endpoint selection logic"""

import os
import sys

# Simulate MPI environment
def test_mpi_distribution():
    print("="*60)
    print("Test 1: MPI-Based Endpoint Distribution")
    print("="*60)
    
    endpoints = [
        "http://endpoint1:9000",
        "http://endpoint2:9000",
        "http://endpoint3:9000",
        "http://endpoint4:9000",
    ]
    
    print(f"\nEndpoints: {len(endpoints)}")
    for i, ep in enumerate(endpoints):
        print(f"  [{i}] {ep}")
    
    print(f"\nSimulating 16 MPI ranks:")
    for rank in range(16):
        os.environ['OMPI_COMM_WORLD_RANK'] = str(rank)
        endpoint_index = rank % len(endpoints)
        endpoint = endpoints[endpoint_index]
        print(f"  Rank {rank:2d} → endpoint[{endpoint_index}] = {endpoint}")
    
    # Clean up
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        del os.environ['OMPI_COMM_WORLD_RANK']

def test_round_robin():
    print("\n" + "="*60)
    print("Test 2: Round-Robin (PID-based)")
    print("="*60)
    
    endpoints = [
        "http://endpoint1:9000",
        "http://endpoint2:9000",
        "http://endpoint3:9000",
        "http://endpoint4:9000",
    ]
    
    print(f"\nCurrent PID: {os.getpid()}")
    pid = os.getpid()
    endpoint_index = pid % len(endpoints)
    endpoint = endpoints[endpoint_index]
    
    print(f"Selected: endpoint[{endpoint_index}] = {endpoint}")
    
    print(f"\nSimulating different PIDs:")
    for pid in range(1000, 1016): 
        endpoint_index = pid % len(endpoints)
        endpoint = endpoints[endpoint_index]
        print(f"  PID {pid} → endpoint[{endpoint_index}] = {endpoint}")

def test_fallback():
    print("\n" + "="*60)
    print("Test 3: Fallback Behavior (No MPI)")
    print("="*60)
    
    endpoints = [
        "http://endpoint1:9000",
        "http://endpoint2:9000",
    ]
    
    # Ensure no MPI vars
    for key in list(os.environ.keys()):
        if 'OMPI_' in key or 'SLURM' in key or 'PMI' in key:
            del os.environ[key]
    
    rank = None
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
    elif 'PMI_RANK' in os.environ:
        rank = int(os.environ['PMI_RANK'])
    
    if rank is not None:
        endpoint_index = rank % len(endpoints)
        endpoint = endpoints[endpoint_index]
        print(f"MPI rank {rank} → {endpoint}")
    else:
        print("No MPI environment detected")
        print(f"Using fallback: endpoint[0] = {endpoints[0]}")

def test_slurm_fallback():
    print("\n" + "="*60)
    print("Test 4: SLURM Fallback")
    print("="*60)
    
    endpoints = [
        "http://endpoint1:9000",
        "http://endpoint2:9000",
        "http://endpoint3:9000",
    ]
    
    # Clear OpenMPI vars, set SLURM
    for key in list(os.environ.keys()):
        if 'OMPI_' in key:
            del os.environ[key]
    
    print(f"\nSimulating SLURM ranks:")
    for rank in range(12):
        os.environ['SLURM_PROCID'] = str(rank)
        endpoint_index = rank % len(endpoints)
        endpoint = endpoints[endpoint_index]
        print(f"  SLURM rank {rank:2d} → endpoint[{endpoint_index}] = {endpoint}")
    
    # Clean up
    if 'SLURM_PROCID' in os.environ:
        del os.environ['SLURM_PROCID']

if __name__ == "__main__":
    test_mpi_distribution()
    test_round_robin()
    test_fallback()
    test_slurm_fallback()
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
