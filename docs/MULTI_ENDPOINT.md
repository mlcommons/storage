# Multi-Endpoint and Advanced Storage Configuration Guide

**Date**: February 7, 2026  
**s3dlio Version**: 0.9.39+  

## Overview

s3dlio provides advanced multi-endpoint capabilities that s3pytorchconnector lacks:

1. **Multiple S3 Endpoints** - Load balance across multiple object storage servers
2. **MPI-Based Distribution** - Deterministic endpoint assignment using MPI rank
3. **Separate Checkpoint Storage** - Different storage for training data vs checkpoints
4. **Multi-Protocol** - Mix S3, Azure, GCS, and file:// in one workflow

---

## 1. Multi-Endpoint Load Balancing

### Why Use Multiple Endpoints?

**Performance**: Distribute I/O load across multiple servers
- Aggregate bandwidth: 4 endpoints → 4x throughput potential
- Avoid single-server bottlenecks
- NUMA-aware data placement

**Reliability**: Redundancy and failover capabilities

**Cost**: Distribute storage across tiers (hot/warm/cold)

### Configuration Options

#### Option A: s3dlio Native Round-Robin

```yaml
storage:
  storage_type: s3dlio
  storage_root: s3://bucket/data/
  
  endpoint_uris:
    - http://endpoint1:9000
    - http://endpoint2:9000
    - http://endpoint3:9000
    - http://endpoint4:9000
  
  load_balance_strategy: round_robin  # Each process picks based on PID
```

**How it works**:
- Each process selects endpoint using: `endpoint[PID % num_endpoints]`
- Semi-stable distribution across processes
- No coordination required

**Best for**: Single-node training, simple distributed setups

#### Option B: MPI-Based Distribution (Recommended)

```yaml
storage:
  storage_type: s3dlio
  storage_root: s3://bucket/data/
  
  endpoint_uris:
    - http://numa-node-0:9000  # Close to CPU 0-15
    - http://numa-node-1:9000  # Close to CPU 16-31
    - http://numa-node-2:9000  # Close to CPU 32-47
    - http://numa-node-3:9000  # Close to CPU 48-63
  
  use_mpi_endpoint_distribution: true
```

**How it works**:
- Uses MPI rank: `endpoint[rank % num_endpoints]`
- Deterministic assignment
- Supports OpenMPI, SLURM, MPICH

**MPI Variables Used**:
1. `OMPI_COMM_WORLD_RANK` (OpenMPI)
2. `SLURM_PROCID` (SLURM)
3. `PMI_RANK` (MPICH)

**Example Distribution** (4 endpoints, 16 ranks):
```
Rank 0-3   → endpoint[0] (http://numa-node-0:9000)
Rank 4-7   → endpoint[1] (http://numa-node-1:9000)
Rank 8-11  → endpoint[2] (http://numa-node-2:9000)
Rank 12-15 → endpoint[3] (http://numa-node-3:9000)
```

**Best for**:
- Multi-node HPC training
- NUMA-aware architectures
- Consistent performance needs
- Research reproducibility

---

## 2. MPI Environment Variables Reference

### OpenMPI Variables (Primary)

| Variable | Description | Example |
|----------|-------------|---------|
| `OMPI_COMM_WORLD_RANK` | Global process rank | 0, 1, 2, ... |
| `OMPI_COMM_WORLD_SIZE` | Total processes | 16 |
| `OMPI_COMM_WORLD_LOCAL_RANK` | Rank on current node | 0-7 (if 8 per node) |
| `OMPI_COMM_WORLD_LOCAL_SIZE` | Processes on node | 8 |
| `OMPI_COMM_WORLD_NODE_RANK` | Node number | 0, 1, 2, 3 |

### SLURM Variables (Fallback)

| Variable | Description | Example |
|----------|-------------|---------|
| `SLURM_PROCID` | Global task ID | 0-15 |
| `SLURM_LOCALID` | Local task ID on node | 0-7 |
| `SLURM_NODEID` | Node index | 0-3 |

### Advanced Endpoint Selection Strategies

**By Node** (all ranks on same node use same endpoint):
```python
# Future enhancement - not yet implemented
node_rank = int(os.environ.get('OMPI_COMM_WORLD_NODE_RANK', 0))
endpoint = endpoint_uris[node_rank % len(endpoint_uris)]
```

**By NUMA Domain** (group ranks by CPU affinity):
```python
# Future enhancement - requires CPU affinity detection
local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
numa_domain = local_rank // cpus_per_numa
endpoint = endpoint_uris[numa_domain % len(endpoint_uris)]
```

---

## 3. Separate Checkpoint Storage

### Why Separate Checkpoints?

**Performance**: Checkpoints don't compete with training data I/O

**Cost**: Store checkpoints on cheaper/slower storage

**Simplicity**: Fast local NVMe for checkpoints, distributed S3 for data

### Configuration

```yaml
storage:
  storage_type: s3dlio
  storage_root: s3://training-data-bucket/imagenet/
  endpoint_uris:
    - http://fast-s3-1:9000
    - http://fast-s3-2:9000
  use_mpi_endpoint_distribution: true

checkpoint:
  # Option 1: Different S3 bucket
  checkpoint_folder: s3://checkpoint-bucket/resnet50/
  
  # Option 2: Local NVMe (fastest for checkpoint I/O)
  checkpoint_folder: file:///nvme/checkpoints/resnet50/
  
  # Option 3: Azure Blob (cross-cloud)
  checkpoint_folder: az://account/container/checkpoints/
```

### Checkpoint Storage Patterns

#### Pattern 1: Local NVMe During Training

```yaml
checkpoint:
  checkpoint_folder: file:///nvme/checkpoints/
  checkpoint_after_epoch: 1
  epochs_between_checkpoints: 1
```

**Benefits**:
- Fastest checkpoint save/load
- No network congestion
- No S3 API costs

**After training**: Copy best checkpoint to S3 for archival
```bash
aws s3 cp /nvme/checkpoints/best_model.pt s3://archive/models/
```

#### Pattern 2: Separate S3 Bucket

```yaml
storage:
  storage_root: s3://training-data/  # Multi-endpoint, read-heavy
  endpoint_uris: [...]

checkpoint:
  checkpoint_folder: s3://checkpoints/  # Single endpoint, write-heavy
  # Uses same S3 credentials but different bucket policy
```

**Benefits**:
- Separate I/O patterns (read vs write)
- Different replication policies
- Easier lifecycle management

#### Pattern 3: Tiered Storage

```yaml
# Training: Fast S3/MinIO cluster
storage:
  storage_root: s3://fast-tier/training/
  endpoint_uris: [local-minio-1, local-minio-2, local-minio-3]

# Checkpoints: Cloud S3 for durability  
checkpoint:
  checkpoint_folder: s3://aws-s3-bucket/checkpoints/
  # Uses AWS S3 endpoint (different from training endpoints)
```

---

## 4. Complete Examples

### Example 1: Single-Node Multi-GPU

```yaml
# 8 GPUs, 4 local MinIO servers
storage:
  storage_type: s3dlio
  storage_root: s3://training/imagenet/
  endpoint_uris:
    - http://localhost:9001  # MinIO instance 1
    - http://localhost:9002  # MinIO instance 2
    - http://localhost:9003  # MinIO instance 3
    - http://localhost:9004  # MinIO instance 4
  load_balance_strategy: round_robin

checkpoint:
  checkpoint_folder: file:///nvme/checkpoints/

# Run: python -m torch.distributed.launch --nproc_per_node=8 train.py
```

### Example 2: Multi-Node HPC Cluster

```yaml
# 4 nodes × 8 GPUs = 32 ranks
# 4 S3 endpoints (1 per node for NUMA affinity)
storage:
  storage_type: s3dlio
  storage_root: s3://shared-training-data/imagenet/
  endpoint_uris:
    - http://node1-ib0:9000  # Node 1 InfiniBand IP
    - http://node2-ib0:9000  # Node 2 InfiniBand IP
    - http://node3-ib0:9000  # Node 3 InfiniBand IP
    - http://node4-ib0:9000  # Node 4 InfiniBand IP
  use_mpi_endpoint_distribution: true

checkpoint:
  checkpoint_folder: s3://checkpoint-bucket/job-12345/

# Run: mpirun -np 32 -hostfile hosts.txt dlio_benchmark --config config.yaml
#
# Distribution:
#   Node 1 (ranks 0-7)   → endpoint node1-ib0:9000
#   Node 2 (ranks 8-15)  → endpoint node2-ib0:9000
#   Node 3 (ranks 16-23) → endpoint node3-ib0:9000
#   Node 4 (ranks 24-31) → endpoint node4-ib0:9000
```

### Example 3: Hybrid Cloud

```yaml
# Training data: On-prem S3 cluster (high bandwidth)
storage:
  storage_type: s3dlio
  storage_root: s3://on-prem/training-cache/
  endpoint_uris:
    - http://datacenter-s3-1:9000
    - http://datacenter-s3-2:9000
  
# Checkpoints: Cloud S3 (durability, archival)
checkpoint:
  checkpoint_folder: s3://aws-bucket/experiments/run-001/
  # Auto-uses AWS S3 endpoint
```

---

## 5. Performance Tuning

### Endpoint Count Guidelines

| Setup | Recommended Endpoints | Rationale |
|-------|----------------------|-----------|
| Single node, 8 GPUs | 2-4 endpoints | Match GPU pairs or NUMA domains |
| Multi-node, 4 nodes × 8 GPUs | 4 endpoints (1/node) | Minimize network hops |
| Large cluster (16+ nodes) | 8-16 endpoints | Balance load vs connection overhead |

### MPI vs Round-Robin

**Use MPI-based** when:
- ✅ Running under mpirun/srun
- ✅ Need deterministic assignment
- ✅ NUMA-aware setup important
- ✅ Reproducible performance required

**Use Round-Robin** when:
- ✅ Single-node training
- ✅ No MPI environment
- ✅ Simple setup preferred
- ✅ Dynamic process count

### Network Topology Considerations

**NUMA-Aware** (recommended):
```yaml
endpoint_uris:
  - http://10.0.0.1:9000  # CPU 0-31, NIC 0
  - http://10.0.0.2:9000  # CPU 32-63, NIC 1
use_mpi_endpoint_distribution: true
```

**Rack-Aware** (large clusters):
```yaml
# Assign endpoints based on rack
# Rank 0-15 (Rack 1) → endpoint1
# Rank 16-31 (Rack 2) → endpoint2
```

---

## 6. Testing & Validation

### Test MPI Distribution

```bash
# Create test script
cat > test_mpi_distribution.py << 'EOF'
import os
endpoints = [
    "http://endpoint1:9000",
    "http://endpoint2:9000",
    "http://endpoint3:9000",
    "http://endpoint4:9000",
]
rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
endpoint = endpoints[rank % len(endpoints)]
print(f"Rank {rank}/{size} → {endpoint}")
EOF

# Run with MPI
mpirun -np 16 python test_mpi_distribution.py

# Expected output:
#   Rank 0/16 → http://endpoint1:9000
#   Rank 1/16 → http://endpoint2:9000
#   Rank 2/16 → http://endpoint3:9000
#   Rank 3/16 → http://endpoint4:9000
#   Rank 4/16 → http://endpoint1:9000
#   ...
```

### Verify Endpoint Selection

Add to config for debugging:
```yaml
storage:
  storage_type: s3dlio
  storage_root: s3://bucket/
  endpoint_uris: [...]
  use_mpi_endpoint_distribution: true

# Check logs for:
#   [s3dlio] MPI-based endpoint selection: http://endpoint2:9000
```

---

## 7. Troubleshooting

### Issue: MPI rank not detected

**Symptom**: Warning: "MPI distribution requested but no MPI rank found"

**Solution**: Ensure running under MPI launcher:
```bash
# ✅ Correct
mpirun -np 16 dlio_benchmark --config config.yaml

# ❌ Wrong
python dlio_benchmark --config config.yaml  # No MPI!
```

### Issue: All ranks use same endpoint

**Cause**: `use_mpi_endpoint_distribution: true` but not running under MPI

**Solution**: Either:
1. Run with `mpirun`/`srun`, OR
2. Use `load_balance_strategy: round_robin` instead

### Issue: Poor load distribution

**Symptom**: One endpoint gets all traffic

**Debug**: Check endpoint selection logs and MPI rank distribution

**Solution**: Verify endpoint count divides evenly into rank count

---

## 8. Future Enhancements

**Planned** (not yet implemented):

1. **Native s3dlio.MultiEndpointStore**: Use Rust-based multi-endpoint with true least_connections
2. **Node-aware distribution**: Auto-detect node topology and assign endpoints
3. **Dynamic endpoint health**: Remove failed endpoints from pool
4. **Per-endpoint statistics**: Track throughput, latency per endpoint
5. **Checkpoint-specific endpoints**: Override endpoint list for checkpoints

---

## Summary

**Multi-endpoint support gives you**:
- ✅ Higher aggregate throughput (4 endpoints → 4x potential)
- ✅ NUMA/topology-aware data placement
- ✅ Separate storage for training vs checkpoints
- ✅ Flexibility (MPI or simple round-robin)

**Advantages over s3pytorchconnector**:
- ✅ Multi-endpoint support (s3torch has none)
- ✅ MPI-aware distribution
- ✅ Multi-protocol (S3/Azure/GCS/file)
- ✅ Zero-copy performance

**Get started**:
1. Use example configs in `configs/dlio/workload/multi_endpoint_*.yaml`
2. Start with round-robin for testing
3. Switch to MPI-based for production HPC deployments
