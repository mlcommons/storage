# Multi-Endpoint Load Balancing - Complete Guide

**Last Updated**: February 18, 2026  
**Status**: All three backends (s3dlio, minio, s3torchconnector) support multi-endpoint

---

## Overview

Multi-endpoint support allows distributing storage I/O across multiple object storage servers for higher aggregate throughput and better load distribution. This guide covers all three supported backends and their different approaches to multi-endpoint configuration.

**Supported backends**:
- **s3dlio** - Native multi-endpoint with true load balancing (recommended)
- **minio** - MPI rank-based endpoint selection
- **s3torchconnector** - MPI rank-based endpoint selection

---

## Quick Start

### Single-Node Multi-Endpoint (s3dlio recommended)

```bash
# Set multiple endpoints
export S3_ENDPOINT_URIS='http://172.16.21.1:9000,http://172.16.21.2:9000'
export S3_LOAD_BALANCE_STRATEGY=round_robin  # or least_connections

# Run your workload
python train.py
```

### Multi-Node MPI Distributed (all backends)

```bash
# Set multiple endpoints
export S3_ENDPOINT_URIS='http://172.16.21.{1...4}:9000'

# Run with MPI - each rank uses different endpoint
mpirun -np 16 python train.py
```

---

## Configuration Methods

All backends support three configuration methods:

### Method 1: Comma-Separated List

```bash
export S3_ENDPOINT_URIS='http://172.16.21.1:9000,http://172.16.21.2:9000,http://172.16.21.3:9000'
```

### Method 2: Template Expansion

```bash
# Expands to http://172.16.21.1:9000, http://172.16.21.2:9000, ... http://172.16.21.8:9000
export S3_ENDPOINT_TEMPLATE='http://172.16.21.{1...8}:9000'
```

### Method 3: File with URIs

```bash
cat > endpoints.txt << EOF
http://172.16.21.1:9000
http://172.16.21.2:9000
http://172.16.21.3:9000
# Comments are supported
http://172.16.21.4:9000
EOF

export S3_ENDPOINT_FILE=endpoints.txt
```

### Method 4: Load Balancing Strategy (s3dlio only)

```bash
export S3_LOAD_BALANCE_STRATEGY=round_robin       # Default: distribute requests evenly
# OR
export S3_LOAD_BALANCE_STRATEGY=least_connections # Route to endpoint with fewest active connections
```

---

## Backend Capabilities Comparison

| Feature | s3dlio | minio | s3torchconnector |
|---------|--------|-------|------------------|
| **Native multi-endpoint** | ✅ Yes | ❌ No | ❌ No |
| **MPI rank-based** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Per-request load balancing** | ✅ Yes | ❌ No | ❌ No |
| **Strategies** | round_robin, least_connections | round_robin (via rank) | round_robin (via rank) |
| **Automatic failover** | ✅ Yes | ❌ No | ❌ No |
| **Per-endpoint stats** | ✅ Yes | ❌ No | ❌ No |
| **Single-process multi-endpoint** | ✅ Yes | ❌ No | ❌ No |

### Implementation Differences

#### s3dlio (Native Multi-Endpoint)
- **Architecture**: Uses Rust-based `MultiEndpointStore` with true load balancing
- **Routing**: Per-request routing across all configured endpoints
- **Performance**: Highest throughput potential from single process
- **Overhead**: Minimal (~1-5 µs per request for endpoint selection)
- **Best for**: Maximum single-node performance, automatic failover, complex load balancing

#### minio (MPI Rank-Based)
- **Architecture**: Each MPI rank selects one endpoint at initialization
- **Routing**: All requests from a rank go to same endpoint (no per-request balancing)
- **Performance**: Perfect for distributed MPI workloads
- **Overhead**: Zero (endpoint selected once)
- **Best for**: MPI distributed workloads, Python SDK preference, wide compatibility

#### s3torchconnector (MPI Rank-Based)
- **Architecture**: Same as minio - rank-based selection
- **Routing**: One endpoint per rank
- **Performance**: AWS-optimized, PyTorch integration
- **Overhead**: Zero (endpoint selected once)
- **Best for**: AWS S3 workloads, PyTorch-specific optimizations, MPI distributed

---

## Use Cases

### Use Case 1: Single-Node, Multiple Endpoints → **Use s3dlio**

**Scenario**: 8-GPU workstation with 4 local MinIO servers

```bash
export S3_ENDPOINT_URIS='http://localhost:9001,http://localhost:9002,http://localhost:9003,http://localhost:9004'
export S3_LOAD_BALANCE_STRATEGY=least_connections

python train.py
```

**Why s3dlio**:
- True load balancing across all endpoints
- Single process can utilize all 4 endpoints
- Automatic failover if one endpoint fails
- Per-endpoint statistics

**Result**: Aggregate bandwidth from all 4 endpoints

---

### Use Case 2: MPI Distributed Training → **Any backend works**

**Scenario**: 4 nodes × 8 GPUs = 32 MPI ranks, 4 storage endpoints

```bash
export S3_ENDPOINT_URIS='http://172.16.21.1:9000,http://172.16.21.2:9000,http://172.16.21.3:9000,http://172.16.21.4:9000'

mpirun -np 32 python train.py
```

**Distribution** (all backends):
```
Ranks 0,4,8,12,16,20,24,28  → endpoint 1 (172.16.21.1)
Ranks 1,5,9,13,17,21,25,29  → endpoint 2 (172.16.21.2)
Ranks 2,6,10,14,18,22,26,30 → endpoint 3 (172.16.21.3)
Ranks 3,7,11,15,19,23,27,31 → endpoint 4 (172.16.21.4)
```

**Round-robin formula**: `endpoint[rank % num_endpoints]`

**Result**: Each rank uses different endpoint, no contention

---

### Use Case 3: NUMA-Aware Distribution → **Use s3dlio or MPI**

**Scenario**: 2 NUMA nodes, 2 storage endpoints (one per NUMA node)

```bash
# Each endpoint is close to one NUMA domain
export S3_ENDPOINT_URIS='http://numa0-storage:9000,http://numa1-storage:9000'

# Option A: s3dlio native (automatic distribution)
python train.py

# Option B: MPI-based (deterministic assignment)
mpirun -np 16 python train.py
```

**Benefits**:
- Minimizes cross-NUMA traffic
- Higher aggregate memory bandwidth
- Better cache locality

---

## MPI Environment Variables

The following MPI environment variables are automatically detected:

| Variable | MPI Implementation | Priority |
|----------|-------------------|----------|
| `OMPI_COMM_WORLD_RANK` | Open MPI v4+ | 1 (checked first) |
| `PMI_RANK` | MPICH, Intel MPI | 2 (fallback) |

**Example MPI rank detection**:
```python
# Automatically done by all backends
rank = os.environ.get('OMPI_COMM_WORLD_RANK') or os.environ.get('PMI_RANK')
if rank:
    endpoint = endpoints[int(rank) % len(endpoints)]
```

**Note**: SLURM support (`SLURM_PROCID`) is not yet implemented but can be added if needed.

---

## Complete Examples

### Example 1: s3dlio Native Multi-Endpoint
```python
from mlpstorage.checkpointing import StreamingCheckpointing

# Configure multi-endpoint via environment
os.environ['S3_ENDPOINT_URIS'] = 'http://ep1:9000,http://ep2:9000,http://ep3:9000'
os.environ['S3_LOAD_BALANCE_STRATEGY'] = 'least_connections'

# Use s3dlio backend
checkpoint = StreamingCheckpointing(backend='s3dlio')
results = checkpoint.save('s3://bucket/checkpoint.dat', total_size_bytes=100*1024**3)

# Results will show:
# - MultiEndpointStore used
# - 3 endpoints active
# - Per-endpoint statistics (if available)
```

### Example 2: minio MPI Rank-Based
```bash
#!/bin/bash
# Configure endpoints
export S3_ENDPOINT_TEMPLATE='http://172.16.21.{1...4}:9000'

# Run with MPI
mpirun -np 16 python -c "
from mlpstorage.checkpointing import StreamingCheckpointing

# Each rank automatically selects different endpoint
checkpoint = StreamingCheckpointing(backend='minio')
results = checkpoint.save('s3://bucket/checkpoint.dat', total_size_bytes=10*1024**3)
print(f'Rank {checkpoint.backend.rank}: {results}')
"

# Output shows each rank using different endpoint:
# [MinIOWriter] MPI rank 0: selected endpoint http://172.16.21.1:9000 from 4 endpoints
# [MinIOWriter] MPI rank 1: selected endpoint http://172.16.21.2:9000 from 4 endpoints
# ...
```

### Example 3: s3torchconnector MPI Distributed
```bash
export S3_ENDPOINT_URIS='http://ep1:9000,http://ep2:9000'

mpirun -np 8 python train.py
# Ranks 0,2,4,6 → ep1
# Ranks 1,3,5,7 → ep2
```

---

## Configuration Priority

All backends follow this priority order:

1. **S3_ENDPOINT_URIS** (highest priority)
2. **S3_ENDPOINT_TEMPLATE** (if URIS not set)
3. **S3_ENDPOINT_FILE** (if neither URIS nor TEMPLATE set)
4. **AWS_ENDPOINT_URL** (fallback - single endpoint, original behavior)

**Backward Compatibility**: If none of the multi-endpoint variables are set, all backends fall back to `AWS_ENDPOINT_URL` (single-endpoint mode).

---

## Testing Multi-Endpoint Setup

### Quick Test - Verify MPI Rank Detection
```bash
export OMPI_COMM_WORLD_RANK=0
python3 -c "from mlpstorage.checkpointing.storage_writers.minio_writer import MinIOStorageWriter; print(f'Rank: {MinIOStorageWriter._get_mpi_rank()}')"
# Output: Rank: 0

export OMPI_COMM_WORLD_RANK=5
python3 -c "from mlpstorage.checkpointing.storage_writers.minio_writer import MinIOStorageWriter; print(f'Rank: {MinIOStorageWriter._get_mpi_rank()}')"
# Output: Rank: 5
```

### Test Template Expansion
```bash
python3 -c "
from mlpstorage.checkpointing.storage_writers.minio_writer import MinIOStorageWriter
template = 'http://172.16.21.{1...8}:9000'
endpoints = MinIOStorageWriter._expand_template(template)
print(f'Template: {template}')
print(f'Expanded: {len(endpoints)} endpoints')
for i, ep in enumerate(endpoints):
    print(f'  {i}: {ep}')
"
```

### Test Endpoint Selection with Simulated MPI
```bash
export S3_ENDPOINT_URIS='http://172.16.21.1:9000,http://172.16.21.2:9000,http://172.16.21.3:9000'

for rank in 0 1 2 3 4 5 6 7; do
    OMPI_COMM_WORLD_RANK=$rank python3 -c "
from mlpstorage.checkpointing.storage_writers.minio_writer import MinIOStorageWriter
endpoint = MinIOStorageWriter._detect_and_select_endpoint()
" 2>&1 | grep "MPI rank"
done

# Expected output:
# [MinIOWriter] MPI rank 0: selected endpoint http://172.16.21.1:9000 from 3 endpoints
# [MinIOWriter] MPI rank 1: selected endpoint http://172.16.21.2:9000 from 3 endpoints
# [MinIOWriter] MPI rank 2: selected endpoint http://172.16.21.3:9000 from 3 endpoints
# [MinIOWriter] MPI rank 3: selected endpoint http://172.16.21.1:9000 from 3 endpoints (wraps)
# ...
```

---

## Performance Tuning

### Endpoint Count Guidelines

| Workload Type | Recommended Endpoints | Rationale |
|---------------|----------------------|-----------|
| Single node, 8 GPUs | 2-4 endpoints | Match NUMA domains or GPU pairs |
| Multi-node, 4 nodes | 4 endpoints (1/node) | Minimize network hops, locality |
| Large cluster (16+ nodes) | 8-16 endpoints | Balance load vs connection overhead |
| Cloud S3 | 1 endpoint | AWS S3 auto-scales, multiple endpoints not needed |

### When to Use s3dlio vs minio/s3torch

**Use s3dlio when**:
- ✅ Single-node training with multiple storage servers
- ✅ Need maximum throughput from single process
- ✅ Want automatic failover on endpoint failure
- ✅ Need per-endpoint statistics

**Use minio/s3torch when**:
- ✅ Multi-node MPI distributed training
- ✅ Each rank should use different endpoint (no per-request switching)
- ✅ Python SDK preference (minio) or AWS integration (s3torch)
- ✅ Simple round-robin sufficient

### Load Balancing Strategies (s3dlio only)

**round_robin** (default):
- Distributes requests evenly across endpoints
- Predictable, deterministic
- Best for: Uniform endpoint capabilities

**least_connections**:
- Routes to endpoint with fewest active connections
- Adapts to endpoint load
- Best for: Varying endpoint performance, dynamic workloads

---

## Troubleshooting

### Issue: "WARNING: Multiple endpoints configured but no MPI rank detected"

**Symptom**: minio or s3torch shows warning, uses only first endpoint

**Cause**: Multiple endpoints configured but not running under MPI

**Solutions**:
1. Run with MPI: `mpirun -np <N> python train.py`
2. Use s3dlio for single-process multi-endpoint
3. Accept the warning (will use first endpoint only)

### Issue: All ranks use same endpoint (MPI mode)

**Symptom**: No load distribution despite multiple endpoints

**Debug**: Check MPI rank detection
```bash
mpirun -np 4 python -c "import os; print(f'Rank: {os.environ.get(\"OMPI_COMM_WORLD_RANK\", \"NOT SET\")}')"
```

**Solutions**:
- Ensure running with `mpirun`, `mpiexec`, or `srun`
- Verify MPI environment variables are set
- Check logs for endpoint selection messages

### Issue: Poor load distribution

**Symptom**: One endpoint receiving most traffic

**Causes**:
- Endpoint count doesn't divide evenly into rank count
- Network topology issues
- Backend doesn't support per-request balancing (minio/s3torch)

**Solutions**:
- Use s3dlio for true per-request load balancing
- Adjust endpoint count to divide evenly (e.g., 4 endpoints for 16 ranks)
- Check network topology (NUMA, IB fabric)

---

## Performance Expectations

### s3dlio Native Multi-Endpoint
- **Per-process throughput**: Aggregate of all endpoints
- **Overhead**: Minimal (~1-5 µs per request)
- **Scalability**: Limited by client CPU/memory bandwidth
- **Example**: 4 endpoints × 2 GB/s each = ~8 GB/s aggregate

### minio/s3torch MPI Rank-Based
- **Per-process throughput**: Single endpoint bandwidth
- **Overhead**: Zero (selected once at init)
- **Scalability**: Linear with number of ranks
- **Example**: 4 endpoints, 16 ranks → each endpoint serves 4 ranks

**Tested Performance** (single client, s3dlio):
- Up to **7 GB/s per client** (varies by library and storage target)
- Network and storage backend are typical bottlenecks

---

## Known Limitations

The following gaps were identified during code review and have **not** been
addressed in the current implementation. They are documented here to prevent
data loss and to inform future contributors.

### 1. SLURM not supported for MPI rank detection

**Affected**: all three backends (`minio_writer.py`, `s3torch_writer.py`,
`s3dlio_writer.py`)

`_get_mpi_rank()` checks only two environment variables:
- `OMPI_COMM_WORLD_RANK` (Open MPI v4+)
- `PMI_RANK` (MPICH, Intel MPI, MVAPICH2)

`SLURM_PROCID` (set by SLURM's `srun`) is **not checked**. On SLURM-managed
HPC clusters, MPI rank detection will silently return `None`, causing all ranks
to fall back to the first endpoint rather than distributing across endpoints.

**Workaround**: Set `OMPI_COMM_WORLD_RANK` manually in your SLURM job script:
```bash
export OMPI_COMM_WORLD_RANK=$SLURM_PROCID
```

**Fix**: Add `SLURM_PROCID` to `_get_mpi_rank()` in all three writer files,
before the MPICH check:
```python
# SLURM uses SLURM_PROCID
rank_str = os.environ.get('SLURM_PROCID')
if rank_str:
    try:
        return int(rank_str)
    except ValueError:
        pass
```

---

### 2. Template expansion handles only the first `{N...M}` pattern

**Affected**: all three backends (`_expand_template()`)

`S3_ENDPOINT_TEMPLATE` uses `re.search()`, which stops at the first match.
A template with multiple patterns (e.g., `http://{1...2}.{10...12}:9000`) only
expands the first `{N...M}`, leaving the second as a literal string:

```
Input:    http://{1...2}.rack{1...4}.example.com
Output:   http://1.rack{1...4}.example.com
          http://2.rack{1...4}.example.com   ← second pattern NOT expanded
```

**Workaround**: Enumerate endpoints explicitly using `S3_ENDPOINT_URIS` instead
of a template with multiple ranges.

**Fix**: Replace `re.search()` with `re.findall()` and apply recursive
expansion, or raise a clear error when more than one pattern is detected.

---

### 3. No URI validation — malformed endpoints pass through silently

**Affected**: all three backends (`_detect_and_select_endpoint()`)

Endpoint URIs from `S3_ENDPOINT_URIS`, `S3_ENDPOINT_TEMPLATE`, or
`S3_ENDPOINT_FILE` are accepted without format checking. Missing `http://` or
`https://` prefix, extra whitespace, or typographical errors result in confusing
failures deep in the storage client rather than a clear error at startup.

**Workaround**: Double-check your endpoint URIs manually before running.

**Fix**: Add a validation step after endpoint list construction:
```python
import re
_URI_RE = re.compile(r'^https?://.+:\d+$')
for uri in endpoints:
    if not _URI_RE.match(uri):
        raise ValueError(f"Malformed endpoint URI: {uri!r} — expected http(s)://host:port")
```

---

## Summary

**Multi-endpoint support provides**:
- ✅ Higher aggregate throughput (N endpoints → Nx potential bandwidth)
- ✅ Better load distribution across storage infrastructure
- ✅ NUMA/topology-aware data placement
- ✅ Flexibility: Choose native load balancing (s3dlio) or MPI distribution (all backends)

**Recommendations**:
1. **Single-node**: Use s3dlio with `S3_LOAD_BALANCE_STRATEGY=least_connections`
2. **Multi-node MPI**: Any backend works, configure via `S3_ENDPOINT_URIS` or `S3_ENDPOINT_TEMPLATE`
3. **Production HPC**: Use MPI-based distribution for deterministic performance

**Get started**:
```bash
# Quick demo with multi-endpoint
export S3_ENDPOINT_URIS='http://ep1:9000,http://ep2:9000'
export TEST_CHECKPOINT_DIR=/fast/storage
./quickstart_demo.sh
```

