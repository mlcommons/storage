# s3dlio Performance Notes — DLIO Training Workload

**Date:** March 20, 2026  
**Status:** Historical — issues identified here are substantially resolved in s3dlio v0.9.84.  
See [dlio_mpi_object_results.md](dlio_mpi_object_results.md) for current benchmark results.

---

## Background

During March 2026 testing with DLIO (168 × ~147 MB NPZ files, UNet3D profile, MinIO backend
at ~1.2 GB/s network ceiling), s3dlio showed lower single-rank throughput than minio-py
under default settings. Root-cause analysis identified six issues, all of which have since
been addressed.

## What Was Found and Fixed

Six issues were identified in s3dlio v0.9.82 and earlier:

| # | Issue | Resolution |
|---|-------|-----------|
| 1 | Redundant HEAD request per object on the `get_many` code path | Fixed in v0.9.84 |
| 2 | Range splitting threshold too aggressive for 1 Gbps environments (37 sub-requests per 147 MB file) | Fixed in v0.9.84; `S3DLIO_RANGE_THRESHOLD_MB` env var now correctly controls the `get_many` path |
| 3 | Tokio runtime thread over-provisioning (32 threads/process × 16 worker processes) | Mitigated: set `S3DLIO_RT_THREADS=8`; architectural fix pending in a future release |
| 4 | Unnecessary Python-side memory copy in the DLIO NPZ reader (`bytes(data)` discarding zero-copy view) | Fixed in mlp-storage reader: zero-copy `_BytesViewIO` wrapper applied |
| 5 | Mutex contention during parallel range-chunk assembly | Fixed in v0.9.82 |
| 6 | O(N²) sort in `get_objects_parallel` for input-order preservation | Fixed in v0.9.82 |

## Outcome

After fixes, s3dlio and minio-py converge to within 1% of each other at NP=4
(~1087–1097 MB/s), confirming all issues were caused by the above bugs rather than
any fundamental capability difference between the libraries.

On high-bandwidth systems (10/100 Gbps), s3dlio's adaptive range-splitting provides
significant advantages that minio-py (which never issues range requests) cannot match.
The threshold defaults are now better calibrated for typical deployment environments.

## Useful Environment Variables

For 1 Gbps or bandwidth-saturated environments, these env vars can further tune behavior:

```bash
# Raise range-split threshold above your largest file size to use single-stream GET
export S3DLIO_RANGE_THRESHOLD_MB=1000

# Reduce Tokio threads per worker process (recommended for high MPI rank counts)
export S3DLIO_RT_THREADS=8
```

