# Phase 9: DLIO Parquet Support - Research

**Researched:** 2026-01-25
**Domain:** DLIO benchmark data format extension, Apache Parquet integration
**Confidence:** MEDIUM

## Summary

This research investigates adding Apache Parquet format support to the DLIO benchmark for use in MLPerf Storage training benchmarks. Parquet is a columnar storage format widely used in recommendation systems (DLRM uses Criteo data in Parquet format) and modern ML data pipelines.

The key finding is that **DLIO does not currently support Parquet format natively**. The supported formats are: tfrecord, hdf5, npz, csv, jpeg, and png. Adding parquet support requires implementing a custom reader plugin for DLIO and extending the data generator. This involves contributing code to the upstream DLIO repository (argonne-lcf/dlio_benchmark) or maintaining a fork.

The mlpstorage project already has `pyarrow` as a dependency, which provides full Parquet read/write capabilities. The DLIO architecture is modular and designed for format extensions via `FormatReader` base class and `DataGenerator` inheritance.

**Primary recommendation:** Implement parquet support as a DLIO plugin following the established CSV reader/generator patterns. Contribute upstream to argonne-lcf/dlio_benchmark to benefit the broader MLPerf community. Use PyArrow for parquet I/O operations with snappy or zstd compression.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyArrow | 23.x | Parquet read/write | Apache Arrow's official Python binding, high performance |
| DLIO Benchmark | mlperf_storage_v2.0 | I/O workload emulation | MLPerf Storage standard engine |
| Pandas | 2.x | DataFrame conversion | PyArrow integration for tabular data |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Petastorm | 0.12.x | Parquet-PyTorch bridge | Complex multi-worker dataloading from parquet |
| fastparquet | 2024.x | Alternative parquet engine | When PyArrow compatibility issues arise |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyArrow | Petastorm | Petastorm adds complexity but provides DataLoader integration |
| PyArrow | fastparquet | fastparquet is pure-Python, slower but sometimes more compatible |
| Custom reader | Existing CSV reader | CSV is text-based, much slower than binary Parquet |

**Installation:**
```bash
pip install pyarrow>=17.0  # Already a dependency in mlpstorage
```

## Architecture Patterns

### Recommended Project Structure
```
dlio_benchmark/
  reader/
    parquet_reader.py       # New: ParquetReader extends FormatReader
    reader_factory.py       # Modify: Add PARQUET format mapping
  data_generator/
    parquet_generator.py    # New: ParquetGenerator extends DataGenerator
  common/
    enumerations.py         # Modify: Add PARQUET to FormatType enum
```

### Pattern 1: DLIO FormatReader Implementation

**What:** Custom reader class extending FormatReader base
**When to use:** For reading parquet files during training benchmark
**Example:**
```python
# Source: Pattern derived from dlio_benchmark/reader/csv_reader.py
from dlio_benchmark.reader.format_reader import FormatReader
from dlio_benchmark.utils.config import ConfigArguments
import pyarrow.parquet as pq

class ParquetReader(FormatReader):
    """Parquet Reader implementation for DLIO."""

    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index, epoch)

    @dlp.log
    def open(self, filename):
        """Load parquet file into memory."""
        table = pq.read_table(filename)
        return table.to_pandas().values  # Convert to numpy array

    @dlp.log
    def close(self, filename):
        """Cleanup operations."""
        pass

    @dlp.log
    def get_sample(self, filename, sample_index):
        """Get a single sample from parquet file."""
        super().get_sample(filename, sample_index)
        data = self.open_file_map[filename]
        image = data[sample_index]
        dlp.update(image_size=image.nbytes)
        return image
```

### Pattern 2: DLIO DataGenerator Implementation

**What:** Data generator class for creating synthetic parquet files
**When to use:** For datagen command to create test datasets
**Example:**
```python
# Source: Pattern derived from dlio_benchmark/data_generator/csv_generator.py
from dlio_benchmark.data_generator.data_generator import DataGenerator
import pyarrow as pa
import pyarrow.parquet as pq

class ParquetGenerator(DataGenerator):
    """Parquet data generator for DLIO."""

    def __init__(self):
        super().__init__()

    def generate(self):
        """Generate synthetic parquet files."""
        super().generate()
        np.random.seed(10)
        record_label = 0

        for i in dlp.iter(range(self.my_rank, self.total_files_to_generate, self.comm_size)):
            out_path_spec = self.storage.get_uri(self._file_list[i])
            dim = self.get_dimension(i)

            # Generate random data as numpy array
            records = [self.gen_random_tensor(dim, dtype='float32')
                      for _ in range(self.num_samples)]
            data = np.stack(records)

            # Convert to Arrow table and write as parquet
            table = pa.table({'data': [row.flatten() for row in data]})
            pq.write_table(table, out_path_spec,
                          compression=self.compression or 'snappy')

        np.random.seed()
```

### Pattern 3: Reader Factory Registration

**What:** Registering parquet format in the factory pattern
**When to use:** Required for DLIO to instantiate the correct reader
**Example:**
```python
# Source: Pattern from dlio_benchmark/reader/reader_factory.py
from dlio_benchmark.common.enumerations import FormatType

class ReaderFactory:
    @staticmethod
    def get_reader(type, dataset_type, thread_index, epoch_number):
        _args = ConfigArguments.get_instance()
        # ... existing format checks ...

        elif _args.format == FormatType.PARQUET:
            from dlio_benchmark.reader.parquet_reader import ParquetReader
            return ParquetReader(dataset_type, thread_index, epoch_number)
```

### Anti-Patterns to Avoid
- **Loading entire parquet dataset into memory:** Use row groups for chunked reading
- **Ignoring column projection:** Read only needed columns via `columns` parameter
- **Single-threaded parquet reads:** Enable `use_threads=True` in PyArrow
- **Uncompressed parquet:** Always use compression (snappy for speed, zstd for size)

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parquet I/O | Custom binary parser | PyArrow pq.read_table/write_table | PyArrow handles schema, compression, row groups |
| Multi-worker loading | Manual process coordination | PyTorch DataLoader + PyArrow | DataLoader handles worker spawning, batching |
| Parquet schema | Ad-hoc column naming | Arrow schema definition | Schema ensures type consistency |
| Compression | Manual compression | PyArrow compression parameter | PyArrow handles snappy/zstd/gzip natively |
| Large file chunking | Manual seek/read | pq.ParquetFile row_group iteration | Row groups are designed for chunked access |

**Key insight:** PyArrow provides a complete, high-performance implementation of Parquet. The work is in integrating it with DLIO's reader/generator patterns, not in parquet handling itself.

## Common Pitfalls

### Pitfall 1: Multi-Worker DataLoader Conflicts with PyArrow
**What goes wrong:** PyArrow errors or data corruption when using num_workers > 0
**Why it happens:** PyArrow file handles not fork-safe across processes
**How to avoid:** Initialize parquet file handles in worker_init() function, not in __init__()
**Warning signs:** "ArrowIOError" or segfaults when num_workers > 0

### Pitfall 2: Memory Explosion from Full File Loading
**What goes wrong:** Out of memory when loading large parquet files
**Why it happens:** Default pq.read_table() loads entire file into memory
**How to avoid:** Use row group iteration or memory mapping:
```python
parquet_file = pq.ParquetFile(filename)
for batch in parquet_file.iter_batches(batch_size=1024):
    process(batch.to_pandas())
```
**Warning signs:** OOM errors, swap thrashing during data loading

### Pitfall 3: Ignoring Parquet Schema Requirements
**What goes wrong:** Type mismatches between generated and expected data
**Why it happens:** Parquet is strongly typed, DLIO uses arbitrary tensors
**How to avoid:** Define explicit Arrow schema matching DLIO tensor expectations:
```python
schema = pa.schema([
    ('data', pa.list_(pa.float32())),
    ('label', pa.int64())
])
```
**Warning signs:** Arrow type conversion errors, NaN values appearing

### Pitfall 4: Forgetting num_samples_per_file Limitation
**What goes wrong:** DLIO expects one sample per file for non-tfrecord formats
**Why it happens:** Current DLIO architecture limitation noted in documentation
**How to avoid:** For Phase 9, either:
  1. Generate one parquet file per sample (like npz/jpeg behavior)
  2. Implement multi-sample support as part of the parquet reader
**Warning signs:** Sample count mismatches, training data exhaustion

### Pitfall 5: Upstream vs Fork Maintenance
**What goes wrong:** Parquet support drifts from DLIO mainline
**Why it happens:** Maintaining fork without upstream contribution
**How to avoid:** Submit PR to argonne-lcf/dlio_benchmark early; coordinate with maintainers
**Warning signs:** Merge conflicts, feature incompatibilities with DLIO updates

## Code Examples

Verified patterns from official sources:

### PyArrow Parquet Read/Write
```python
# Source: https://arrow.apache.org/docs/python/parquet.html
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

# Writing parquet
data = np.random.randn(1000, 128).astype(np.float32)
table = pa.table({'features': [row for row in data]})
pq.write_table(table, 'data.parquet', compression='snappy')

# Reading parquet
table = pq.read_table('data.parquet')
data = np.stack(table['features'].to_pylist())
```

### Row Group Iteration for Large Files
```python
# Source: https://arrow.apache.org/docs/python/parquet.html
parquet_file = pq.ParquetFile('large_data.parquet')

# Get metadata
num_row_groups = parquet_file.metadata.num_row_groups

# Iterate by row group
for i in range(num_row_groups):
    table = parquet_file.read_row_group(i)
    process_batch(table.to_pandas())
```

### DLIO Configuration for Parquet Format
```yaml
# Proposed configuration for parquet-based workload
model:
  name: dlrm
  type: recommendation

framework: pytorch

workflow:
  generate_data: False
  train: True
  checkpoint: False

dataset:
  data_folder: data/dlrm/
  format: parquet  # New format option
  num_files_train: 65536
  num_samples_per_file: 1
  record_length_bytes: 512

reader:
  data_loader: pytorch
  batch_size: 8192
  read_threads: 8
```

### PyTorch-Parquet Integration with Worker Init
```python
# Source: PyTorch documentation and community patterns
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader

class ParquetDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.pq_file = None  # Lazy initialization

    def worker_init(self, worker_id):
        """Called in worker process - safe to open files here."""
        # Each worker can open its own file handles
        pass

    def __getitem__(self, idx):
        if self.pq_file is None:
            self.pq_file = pq.ParquetFile(self.file_list[idx // self.samples_per_file])
        # Read specific row
        ...

# Use worker_init_fn to handle per-worker initialization
dataloader = DataLoader(
    dataset,
    num_workers=4,
    worker_init_fn=dataset.worker_init
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CSV for tabular ML data | Parquet for columnar data | 2018-2020 | 10-100x faster, better compression |
| TFRecord for all TF workloads | Format per use case | 2020+ | Parquet for tabular, TFRecord for images |
| Petastorm required for Parquet | Native PyArrow sufficient | 2022+ | Reduced dependency complexity |
| LZ4 compression default | Snappy/ZSTD preference | 2023+ | Better compression-speed tradeoffs |

**Deprecated/outdated:**
- Using pandas.read_parquet() directly in performance-critical code (use PyArrow directly)
- LZ4 compression in parquet (use ZSTD or snappy)
- Petastorm for simple use cases (PyArrow sufficient when not needing full Petastorm features)

## Integration Requirements for mlpstorage

### Changes Required in mlpstorage

1. **YAML Configuration Files:**
   - Create parquet variants: `dlrm_parquet_h100.yaml`, `dlrm_parquet_datagen.yaml`
   - Add `format: parquet` specification

2. **Validation Rules:**
   - `dataset.format: parquet` is an OPEN-category parameter change
   - No changes to CLOSED_ALLOWED_PARAMS needed

3. **CLI Arguments:**
   - Optional: Add `--format` flag for datagen command
   - Or: Let YAML config control format entirely (simpler)

4. **Documentation:**
   - Document parquet format support
   - Explain when parquet vs npz vs tfrecord is appropriate

### Dependencies

- PyArrow already present in mlpstorage dependencies
- DLIO fork or upstream PR required for actual support

## Implementation Options

### Option 1: Upstream Contribution (Recommended)
**Approach:** Submit PR to argonne-lcf/dlio_benchmark adding parquet support
**Pros:**
- Community benefit
- Maintenance shared with upstream
- No fork management overhead
**Cons:**
- Longer timeline (PR review process)
- Must follow upstream conventions
**Effort:** 2-3 weeks for implementation + review

### Option 2: DLIO Fork with Parquet
**Approach:** Maintain mlpstorage-specific DLIO fork with parquet support
**Pros:**
- Full control over implementation
- Faster initial deployment
**Cons:**
- Fork maintenance burden
- Drift from upstream
**Effort:** 1-2 weeks for implementation, ongoing maintenance

### Option 3: DLIO Plugin Architecture
**Approach:** Use DLIO's `data_loader_classname` custom loader support
**Pros:**
- No DLIO code changes required
- Plugin can live in mlpstorage
**Cons:**
- Limited to data loading, not generation
- Less integrated than native support
**Effort:** 1 week for implementation

## Open Questions

Things that couldn't be fully resolved:

1. **DLIO upstream acceptance likelihood**
   - What we know: DLIO welcomes format contributions
   - What's unclear: Timeline and review requirements
   - Recommendation: File issue first to gauge maintainer interest

2. **Multi-sample per file support**
   - What we know: DLIO only supports 1 sample/file for non-tfrecord
   - What's unclear: Whether parquet reader should break this pattern
   - Recommendation: Start with 1 sample/file; consider multi-sample as enhancement

3. **Compression default**
   - What we know: Snappy is fastest, ZSTD best compression
   - What's unclear: What matches real DLRM workload patterns
   - Recommendation: Default to snappy, make configurable

4. **Performance validation**
   - What we know: PyArrow is highly optimized
   - What's unclear: Actual I/O patterns match real DLRM training
   - Recommendation: Benchmark against real Criteo data access patterns

## Sources

### Primary (HIGH confidence)
- [Apache Arrow Parquet Documentation](https://arrow.apache.org/docs/python/parquet.html) - PyArrow API reference
- [DLIO Benchmark GitHub](https://github.com/argonne-lcf/dlio_benchmark) - Source code patterns
- [DLIO Configuration Documentation](https://dlio-benchmark.readthedocs.io/en/latest/config.html) - Supported formats

### Secondary (MEDIUM confidence)
- [DLIO Custom Data Loader Plugin](https://dlio-benchmark.readthedocs.io/en/v2.0.0/custom_data_loader.html) - Extension architecture
- [Petastorm Documentation](https://petastorm.readthedocs.io/en/latest/readme_include.html) - Alternative approach
- [NVIDIA DLRM Implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM) - Real DLRM data format reference

### Tertiary (LOW confidence)
- WebSearch results for parquet multi-worker patterns - Community practices
- WebSearch results for compression benchmarks - Performance characteristics

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyArrow is the definitive parquet library for Python
- Architecture: MEDIUM - Based on DLIO patterns, not verified implementation
- Pitfalls: MEDIUM - Based on documentation and community patterns
- Implementation options: MEDIUM - Upstream acceptance unknown

**Research date:** 2026-01-25
**Valid until:** 90 days (parquet ecosystem is stable, DLIO may evolve)
