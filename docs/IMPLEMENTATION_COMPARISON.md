# MLP vs dpsi Implementation Comparison

## Critical Finding: DIFFERENT BASE CODE

### Repository Origins

**MLP Implementation (mlp-storage/dlio_benchmark):**
- Repository: `https://github.com/russfellows/dlio_benchmark.git`
- Branch: `main`
- HEAD Commit: `ed7f476` "Add configurable dgen-py data generation support"

**dpsi Implementation (mlp-storage-dpsi):**
- Wrapper Repository: `https://github.com/dpsi/storage.git` (branch: darien-TF_ObjectStorage)
- Embedded DLIO: `https://github.com/dpsi/dlio_benchmark.git@darien-s3-refactor`
- HEAD Commit: `7078286` "Refactor S3 pytorch implementation. Change code to use storage_root config option and namespace. Removes urlparsing for each I/O..."

### Common Ancestor

Both implementations **diverged from a common upstream** around commit `3c2be85`:
```
3c2be85 - Fix the first epoch AU calculation (#318) (#319)
0207330 - feat(s3 checkpointing support): added pytorch s3 for checkpointing (#315)
002424d - docs(profiling): fix dftracer broken link (#314)
...
```

**Divergence Point:**
- **After 3c2be85**, russfellows added: `ed7f476` (dgen-py support)
- **After 3c2be85**, dpsi added: `585f375` + `7078286` (S3 refactor)

## Implementation Differences

### File Sizes
- **dpsi**: 145 lines (simple, focused)
- **MLP**: 382 lines (complex, multi-library)

### Architecture Philosophy

**dpsi Approach:**
```python
# Bucket+key separation via config
storage_root = "bucket-name"        # The S3 bucket
data_folder = "prefix/path"         # Object key prefix
namespace = "train"                 # Subdirectory

# Result: s3://bucket-name/prefix/path/train/file.npz
```

**MLP Approach:**
```python
# URI-based with runtime parsing
data_dir = "s3://bucket-name/prefix/path"
namespace = "train"

# Runtime: urlparse(data_dir) → bucket="bucket-name", key="prefix/path"
# Result: s3://bucket-name/prefix/path/train/file.npz
```

### Library Support

**dpsi:**
- **Single library**: s3torchconnector only
- Simple, well-tested
- 145-line implementation

**MLP:**
- **Multi-library**: s3torchconnector, minio, s3dlio
- Environment variable selector: `STORAGE_LIBRARY`
- MinIOAdapter wrapper class (83 lines)
- Dynamic library loading
- 382-line implementation

### Modified Files Overlap (MERGE CONFLICTS EXPECTED)

Both implementations modified the SAME core files:

1. **dlio_benchmark/storage/s3_torch_storage.py**
   - dpsi: Simplified to 145 lines, removed URL parsing
   - MLP: Expanded to 382 lines, added multi-library support

2. **dlio_benchmark/storage/storage_handler.py**
   - dpsi: Added namespace handling
   - MLP: Added `self.logger` attribute

3. **dlio_benchmark/storage/storage_factory.py**
   - dpsi: No changes
   - MLP: Added DLIO_S3_IMPLEMENTATION env var selector

## Code Changes Breakdown

### dpsi Refactor (commit 7078286, 9 files changed)
```
dlio_benchmark/checkpointing/base_checkpointing.py       |  4 +-
dlio_benchmark/checkpointing/pytorch_s3_checkpointing.py | 49 ++---------
dlio_benchmark/configs/workload/unet3d_a100_s3.yaml      |  4 +-
dlio_benchmark/configs/workload/unet3d_h100_s3.yaml      |  4 +-
dlio_benchmark/main.py                                   |  3 +-
dlio_benchmark/storage/s3_storage.py                     | 56 ++++---------
dlio_benchmark/storage/s3_torch_storage.py               | 98 +++++++---------------
dlio_benchmark/storage/storage_handler.py                |  1 +
dlio_benchmark/utils/config.py                           |  7 +-
```
**Goal**: Simplify S3 implementation, eliminate per-I/O URL parsing overhead

### MLP Changes (custom modifications)
```
dlio_benchmark/storage/storage_factory.py         | Added implementation selector
dlio_benchmark/storage/s3_torch_storage.py        | 383 lines (multi-library)
dlio_benchmark/storage/s3_torch_storage_dpsi.py   | 145 lines (dpsi copy)
dlio_benchmark/storage/s3_storage_dpsi.py         | dpsi base class copy
dlio_benchmark/storage/storage_handler.py         | Added self.logger
```
**Goal**: Enable runtime library selection (s3torchconnector/minio/s3dlio)

## Merge Implications

### Option 1: Keep Separate (Current State)
✅ **Pros:**
- Clean comparison possible
- No merge conflicts
- Can benchmark both approaches independently

❌ **Cons:**
- Two codebases to maintain
- Can't combine dpsi simplifications with MLP multi-library

### Option 2: Merge dpsi into MLP
**Strategy**: Add dpsi as 4th library option
```python
STORAGE_LIBRARY options:
- s3torchconnector  (MLP URI-based)
- minio             (MLP URI-based)
- s3dlio            (MLP URI-based, currently broken)
- s3torch-dpsi      (dpsi bucket+key architecture)
```

✅ **Pros:**
- Best of both worlds
- Structured comparison
- Single codebase

❌ **Cons:**
- Requires careful refactoring
- Must preserve both URI and bucket+key approaches

### Option 3: Replace MLP with dpsi + Add Libraries
**Strategy**: Use dpsi's 145-line base, add minio/s3dlio adapters

✅ **Pros:**
- Simpler base (145 lines)
- Cleaner architecture
- Less URL parsing overhead

❌ **Cons:**
- Lose MLP's URI convenience
- Must adapt configs to bucket+key format

## Testing Status

### ✅ Completed Tests
1. **dpsi + s3torchconnector** (BASELINE)
   - Bucket: dpsi-s3torch
   - Result: ✅ 3 NPZ files created in ~23 seconds

### ⏳ Pending Tests
2. **MLP + s3torchconnector**
   - Bucket: mlp-s3torch
   - Expected: ✅ Should match baseline

3. **MLP + minio**
   - Bucket: mlp-minio
   - Expected: ✅ Should work

4. **MLP + s3dlio**
   - Bucket: mlp-s3dlio
   - Expected: ❌ Known bug at compat layer line 571

## Recommendations

### Immediate Actions (Phase 1)
1. ✅ Run MLP + s3torchconnector test (validate MLP URI parsing works)
2. ✅ Run MLP + minio test (validate multi-library switching)
3. Fix s3dlio bug and test
4. **Compare performance**: dpsi (145 lines, no URL parsing) vs MLP (382 lines, runtime parsing)

### Decision Point (Phase 2)
Based on test results, decide:
- **If dpsi is faster**: Adopt bucket+key architecture, add libraries to it
- **If MLP matches dpsi**: Keep MLP approach, incorporate dpsi's simplifications
- **If both equal**: Choose based on config convenience (URI vs bucket+key)

### Integration Strategy (Phase 3)
Likely approach:
```python
# Hybrid: Support both config styles
if config.storage_root and config.data_folder:
    # dpsi bucket+key mode
    bucket = config.storage_root
    prefix = config.data_folder
else:
    # MLP URI mode (backward compatible)
    bucket, prefix = parse_s3_uri(config.data_dir)

# Then use selected library (s3torchconnector/minio/s3dlio)
```

## Key Takeaway

**The implementations started from the SAME upstream DLIO codebase but diverged:**
- dpsi focused on **simplification** (145 lines, bucket+key)
- MLP focused on **flexibility** (382 lines, multi-library, URI-based)

Both are valid approaches. Testing will reveal which architecture performs better.
