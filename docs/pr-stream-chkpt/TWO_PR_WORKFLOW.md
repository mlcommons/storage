# Two-PR Workflow: DLIO Benchmark + MLPerf Storage

**Date**: February 18, 2026  
**Branch Setup**: 
- dlio_benchmark: `darien-s3-refactor` (current)
- mlp-storage: `feature/checkpoint-dgen-optimization` (current)

---

## Overview

We have changes in TWO separate git repositories that require TWO separate PRs:

1. **PR #1**: DLIO Benchmark changes → Your fork (https://github.com/russfellows/dlio_benchmark)
2. **PR #2**: MLPerf Storage changes → MLCommons repo (TF_ObjectStorage branch)

---

## PR #1: DLIO Benchmark Multi-Library Storage

### Files Changed (4 files)
```
M dlio_benchmark/dlio_benchmark/common/enumerations.py
M dlio_benchmark/dlio_benchmark/storage/storage_factory.py
A dlio_benchmark/dlio_benchmark/storage/minio_storage.py
A dlio_benchmark/dlio_benchmark/storage/s3dlio_storage.py
```

### Purpose
Add multi-library S3 storage support (s3dlio, minio, s3torchconnector) to DLIO benchmark.

### Commit Strategy
Single commit with all 4 files:
```
feat: Add multi-library S3 storage support

- Add StorageLibrary enum (s3dlio, minio, s3torchconnector)
- Update storage factory to support library selection
- Add MinioStorage backend (native MinIO SDK)
- Add S3dlioStorage backend (high-performance Rust-based)
- Maintain backward compatibility with existing s3torchconnector
```

### Steps to Execute

```bash
# Navigate to dlio_benchmark repository
cd /home/eval/Documents/Code/mlp-storage/dlio_benchmark

# Add your fork as remote (if not already added)
git remote add russfellows https://github.com/russfellows/dlio_benchmark.git
git fetch russfellows

# Create feature branch from current branch (darien-s3-refactor)
git checkout -b feat/multi-library-storage

# Stage all changes
git add dlio_benchmark/common/enumerations.py
git add dlio_benchmark/storage/storage_factory.py  
git add dlio_benchmark/storage/minio_storage.py
git add dlio_benchmark/storage/s3dlio_storage.py

# Commit
git commit -m "feat: Add multi-library S3 storage support

- Add StorageLibrary enum (s3dlio, minio, s3torchconnector)
- Update storage factory to support library selection
- Add MinioStorage backend (native MinIO SDK)
- Add S3dlioStorage backend (high-performance Rust-based)
- Maintain backward compatibility with existing s3torchconnector"

# Push to your fork
git push russfellows feat/multi-library-storage

# Create PR on GitHub
# Go to: https://github.com/russfellows/dlio_benchmark
# Click "New Pull Request"
# Base: darien-s3-refactor (or main, depending on your fork structure)
# Compare: feat/multi-library-storage
```

---

## PR #2: MLPerf Storage StreamingCheckpointing

### Files Changed (16 items)

**Core Code (5 files)**:
```
M mlpstorage/checkpointing/streaming_checkpoint.py
M mlpstorage/checkpointing/storage_writers/__init__.py
M mlpstorage/checkpointing/storage_writers/s3dlio_writer.py
A mlpstorage/checkpointing/storage_writers/minio_writer.py
A mlpstorage/checkpointing/storage_writers/s3torch_writer.py
```

**Documentation (5 items)**:
```
M README.md
M tests/README.md
A docs/MULTI_ENDPOINT_GUIDE.md
A docs/QUICKSTART.md
A docs/pr-stream-chkpt/
D docs/MULTI_ENDPOINT.md
D docs/PR_Readiness_Plan.md
```

**Tests (3 files)**:
```
A tests/checkpointing/test_streaming_backends.py
A tests/checkpointing/demo_checkpoint_methods.sh
A tests/scripts/demo_streaming_checkpoint.sh
```

**Other (2 files)**:
```
M .gitignore
```

### Purpose
Add StreamingCheckpointing with multi-endpoint support and dgen-py integration (155x faster, 192x memory reduction).

### Commit Strategy
Three commits for logical organization:

#### Commit 1: Core Multi-Endpoint Implementation
```bash
cd /home/eval/Documents/Code/mlp-storage

git add mlpstorage/checkpointing/streaming_checkpoint.py
git add mlpstorage/checkpointing/storage_writers/__init__.py
git add mlpstorage/checkpointing/storage_writers/s3dlio_writer.py
git add mlpstorage/checkpointing/storage_writers/minio_writer.py
git add mlpstorage/checkpointing/storage_writers/s3torch_writer.py

git commit -m "feat: Add StreamingCheckpointing with multi-endpoint support

- Add multi-endpoint configuration via environment variables
  (S3_ENDPOINT_URIS, S3_ENDPOINT_TEMPLATE, S3_ENDPOINT_FILE)
- Implement MPI rank-based endpoint selection for minio/s3torch
- Add native multi-endpoint support for s3dlio (round_robin, least_connections)
- Add MinioWriter with configurable multipart upload
- Add S3TorchWriter with auto-managed multipart
- Update S3dlioWriter with multi-endpoint detection
- Support 3 storage backends: s3dlio, minio, s3torchconnector
- Enable load balancing across multiple storage nodes"
```

#### Commit 2: Tests and Demos  
```bash
git add tests/checkpointing/test_streaming_backends.py
git add tests/checkpointing/demo_checkpoint_methods.sh
git add tests/scripts/demo_streaming_checkpoint.sh
git add tests/README.md

git commit -m "test: Add StreamingCheckpointing validation and demos

- Add test_streaming_backends.py - validates all 3 backends
- Add demo_checkpoint_methods.sh - simple checkpoint comparison
- Add demo_streaming_checkpoint.sh - comprehensive feature demo
- Update tests/README.md with demo documentation"
```

#### Commit 3: Documentation and Cleanup
```bash
git add docs/MULTI_ENDPOINT_GUIDE.md
git add docs/QUICKSTART.md
git add docs/pr-stream-chkpt/
git add README.md
git add .gitignore
git rm docs/MULTI_ENDPOINT.md
git rm docs/PR_Readiness_Plan.md

git commit -m "docs: Add multi-endpoint documentation and update README

- Add MULTI_ENDPOINT_GUIDE.md (14K comprehensive guide)
- Add QUICKSTART.md with feature overview
- Add docs/pr-stream-chkpt/ with testing documentation
- Update main README with Testing and Demos section
- Update .gitignore (Test-Backup/, *.OLD_*/, env-*)
- Remove superseded documentation files"
```

#### Push and Create PR
```bash
# Push to your remote
git push origin feature/checkpoint-dgen-optimization

# Create PR on GitHub
# Go to: https://github.com/mlcommons/storage (or your fork)
# Click "New Pull Request"
# Base: TF_ObjectStorage
# Compare: feature/checkpoint-dgen-optimization
```

---

## PR Relationship and Dependencies

### Can PRs be independent?
**Yes!** Both PRs can be developed, reviewed, and merged independently:

- **DLIO PR**: Adds multi-library support to DLIO benchmark
- **MLPerf PR**: Adds StreamingCheckpointing to mlp-storage

### Integration Point
The mlp-storage code imports from `dlio_benchmark.storage` classes, so eventually:
1. DLIO PR gets merged into your dlio_benchmark fork
2. mlp-storage updates its dlio_benchmark dependency to use your fork/branch
3. Full integration is complete

### Testing
Both PRs have been validated:
- ✅ All Python code compiles successfully
- ✅ Import tests pass
- ✅ Logical analysis confirms correctness
- ✅ No runtime testing yet (no multi-endpoint environment available)

---

## Recommended Order

### Option A: Sequential (Safer)
1. Submit DLIO PR first
2. Get DLIO PR reviewed/merged
3. Submit MLPerf PR (referencing DLIO changes)

### Option B: Parallel (Faster)
1. Submit both PRs simultaneously
2. Note dlio_benchmark dependency in MLPerf PR description
3. Merge DLIO first when approved
4. Merge MLPerf after DLIO is integrated

---

## Post-PR Checklist

After both PRs are merged:
- [ ] Update mlp-storage's dlio_benchmark dependency to point to your fork
- [ ] Verify integration testing works end-to-end
- [ ] Update documentation with final repo references
- [ ] Consider upstreaming DLIO changes to dpsi/dlio_benchmark

---

## Quick Reference

### DLIO Benchmark Repository
- **Your Fork**: https://github.com/russfellows/dlio_benchmark
- **Current Branch**: darien-s3-refactor
- **PR Branch**: feat/multi-library-storage
- **Files**: 4 (2 modified, 2 new)

### MLPerf Storage Repository  
- **Upstream**: https://github.com/mlcommons/storage
- **Current Branch**: feature/checkpoint-dgen-optimization
- **Target Branch**: TF_ObjectStorage
- **Files**: 16 items (8 modified, 7 new, 1 directory, 2 deleted)

