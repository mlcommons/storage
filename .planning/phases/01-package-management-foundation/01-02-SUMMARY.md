---
phase: 01-package-management-foundation
plan: 02
subsystem: package-management
tags: [uv, pytorch, dependencies, cpu-only]

# Dependency graph
requires: []
provides:
  - CPU-only PyTorch index configuration
  - Packaging library for version parsing
affects:
  - 01-03 (uv lockfile generation will use CPU-only index)
  - All future dependency management

# Tech tracking
tech-stack:
  added:
    - packaging>=21.0
  patterns:
    - uv index configuration for CPU-only builds

# File tracking
key-files:
  created: []
  modified:
    - pyproject.toml

# Decisions
decisions:
  - id: cpu-only-pytorch
    choice: Use uv index configuration to enforce CPU-only PyTorch
    rationale: Storage benchmark doesn't need GPU support; avoids large CUDA downloads
  - id: packaging-library
    choice: Add packaging as core dependency
    rationale: Required for PEP 440 version parsing in lockfile validator

# Metrics
duration: 76 seconds
completed: 2026-01-23
---

# Phase 01 Plan 02: CPU-Only PyTorch Configuration Summary

**One-liner:** Configure uv to use CPU-only PyTorch wheels via explicit index, add packaging for version parsing

## What Was Built

Added uv package manager configuration to pyproject.toml to enforce CPU-only PyTorch installations. This prevents accidental inclusion of GPU dependencies (torch-cuda, nvidia-*) which are unnecessary for a storage benchmark and add significant download/install overhead.

### Key Components

1. **UV Index Configuration**
   - Added `[[tool.uv.index]]` section pointing to PyTorch CPU wheel index
   - Configured explicit index for torch, torchvision, torchaudio packages
   - Documented rationale for CPU-only builds in comments

2. **Packaging Dependency**
   - Added `packaging>=21.0` to core dependencies
   - Enables PEP 440 compliant version comparison in lockfile validation
   - Lightweight pure-Python library with no transitive dependencies

## Tasks Completed

| Task | Description | Commit | Files Modified |
|------|-------------|--------|----------------|
| 1 | Add uv CPU-only index configuration | ec1e089 | pyproject.toml |
| 2 | Add packaging dependency for version parsing | 889e20c | pyproject.toml |

## Decisions Made

### CPU-Only PyTorch Index
**Context:** MLPerf Storage is a storage benchmark suite that does not require GPU acceleration. DLIO (the underlying execution engine) has PyTorch as a transitive dependency.

**Decision:** Use uv's explicit index feature to direct PyTorch packages to CPU-only wheels.

**Alternatives considered:**
- Let users manage this manually (error-prone, poor UX)
- Use pip constraints files (less elegant, not package-manager native)

**Rationale:**
- Prevents accidental 10+ GB downloads of CUDA packages
- Reduces installation time significantly
- Aligns with benchmark purpose (storage, not compute)
- Transparent via uv's native index mechanism

### Packaging Library for Version Parsing
**Context:** Future lockfile validator needs to compare version strings correctly.

**Decision:** Add `packaging>=21.0` as core dependency.

**Rationale:**
- Official PyPA library for PEP 440 version parsing
- Pure Python, no transitive dependencies (~300KB)
- Already used by pip internally
- Avoids error-prone manual version comparisons

## Technical Details

### UV Index Configuration
```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]
torchaudio = [{ index = "pytorch-cpu" }]
```

**How it works:**
- `explicit = true` means this index is only used for packages listed in `[tool.uv.sources]`
- PyTorch packages resolve to `https://download.pytorch.org/whl/cpu` instead of PyPI
- Other packages continue to use PyPI normally
- Compatible with both `uv pip install` and future `uv lock` workflows

### Impact on Installation
- **Default install** (`pip install .`): No PyTorch, no change in behavior
- **Full install** (`pip install .[full]`): PyTorch comes from CPU index, not CUDA
- **Test install** (`pip install .[test]`): No PyTorch, no change
- **UV users**: Automatic CPU-only PyTorch when resolving dependencies

## Deviations from Plan

None - plan executed exactly as written.

## Verification

✅ All verification criteria met:
- `[[tool.uv.index]]` section exists with pytorch-cpu URL
- `[tool.uv.sources]` directs torch packages to CPU index
- `packaging>=21.0` added to dependencies
- pyproject.toml is valid TOML syntax
- Existing extras (test/full/dlio) unchanged in functionality

## Known Limitations

1. **Only affects uv users:** Traditional `pip install` will still use PyPI's default PyTorch (which may include CUDA). This is acceptable as uv is the recommended package manager for this project.

2. **Requires uv 0.1.0+:** The index configuration syntax is uv-specific. Projects not using uv can ignore these sections.

## Next Phase Readiness

**Blocks:** None

**Enables:**
- 01-03 can now generate uv lockfile with CPU-only PyTorch
- Future dependency validation can use `packaging.version.Version` for comparisons

**Dependencies for next plan:**
- None - 01-03 can proceed immediately

## Integration Points

### With Other Plans
- **01-01 (uv installation)**: Provides the uv binary that reads this configuration
- **01-03 (lockfile generation)**: Will consume this index configuration
- **01-04 (lockfile validation)**: Will use packaging library for version checks

### With Existing System
- No runtime changes to mlpstorage CLI
- Only affects dependency resolution during installation
- Transparent to end users (they see faster installs, no CUDA bloat)

## Testing Notes

**Manual verification performed:**
```bash
# Confirmed index configuration present
grep -A2 "tool.uv.index" pyproject.toml

# Confirmed pytorch-cpu URL
grep "pytorch-cpu" pyproject.toml

# Confirmed packaging dependency
grep "packaging" pyproject.toml

# Validated TOML syntax
python3 -c "import tomllib; f=open('pyproject.toml','rb'); tomllib.load(f)"
```

**Not tested (requires 01-03):**
- Actual uv lockfile generation with CPU-only PyTorch
- Installation size comparison (CPU vs CUDA wheels)

These will be verified in plan 01-03 when lockfile is generated.

## Artifacts

### Modified Files
- `pyproject.toml` (22 lines added)

### Configuration Added
- UV index configuration (10 lines)
- UV sources mapping (4 lines)
- Packaging dependency (1 line)
- Enhanced comments (7 lines)

## Future Considerations

### Potential Improvements
1. **Platform-specific indexes:** Could add separate indexes for macOS ARM (MPS) if needed
2. **Version pinning:** Could pin specific PyTorch CPU version in lockfile (handled in 01-03)
3. **Documentation:** User-facing docs should explain CPU-only choice (handled in later phases)

### Monitoring
- Watch for uv syntax changes in future versions
- Monitor PyTorch CPU index availability/reliability
- Track packaging library updates (currently stable)

---

**Completion:** 2026-01-23 22:21:10 UTC
**Duration:** 76 seconds
**Commits:** ec1e089, 889e20c
