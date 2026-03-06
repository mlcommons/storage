---
id: "PLAN-01"
phase: "12-dlrm-dataset-columns"
wave: 1
depends_on: []
goal: "Add int8/float16 dtype support to DLIO parquet generator, default size to 1, update DLRM configs with 200 mixed-dtype columns, and verify end-to-end"
context_target: "50%"
---

# Plan: Expand DLRM workload to 200 individual columns with mixed dtype support

## Background

The current DLRM workload configs define 3 columns where two columns (`numerical_features` size 13, `categorical_features` size 26) represent multiple features packed together. The goal is to break out all features into 200 individual columns with mixed dtypes, where 40 randomly distributed columns are read during the workload (totaling exactly 160 bytes) and 160 are not read.

**Current state:**
- 3 columns: `label` (float32, size 1), `numerical_features` (float32, size 13), `categorical_features` (float32, size 26)
- `record_length_bytes: 160` (40 values × 4 bytes)
- DLIO parquet generator only supports float32, float64, list, string, binary, bool dtypes
- DLIO defaults `size` to 1024 when not specified in config

**Target state:**
- 200 columns: `feature_000` through `feature_199`, each with size 1 (implicit default)
- 40 columns randomly distributed across all 200 have `read: true`
- 160 columns have `read: false`
- Read columns total exactly 160 bytes
- Mixed dtypes: int8, float16, float32, float64
- `record_length_bytes` = total bytes of all 200 columns
- `size` defaults to 1 when omitted from config (not 1024)
- Scalar PyArrow types used for size=1 columns (not FixedSizeListArray)

**Key code locations:**
- Generator schema: `dlio_benchmark/dlio_benchmark/data_generator/parquet_generator.py` lines 65-98 (`_build_schema`)
- Generator data: `dlio_benchmark/dlio_benchmark/data_generator/parquet_generator.py` lines 100-153 (`_generate_column_data_batch`)
- Config parsing: `dlio_benchmark/dlio_benchmark/utils/config.py` lines 805-821 (parquet column parsing with `read` flag)
- Reader column filtering: `dlio_benchmark/dlio_benchmark/reader/parquet_reader.py` lines 189-191 (filters by `read` flag)
- DLRM configs: `configs/dlio/workload/dlrm_b200.yaml`, `configs/dlio/workload/dlrm_mi355.yaml`, `configs/dlio/workload/dlrm_datagen.yaml`

## Wave 1 — DLIO fork: int8/float16 support, scalar size=1 path, default size=1

## Task 1: Add int8 and float16 dtype support with scalar size=1 path and default size=1

**Files:** `dlio_benchmark/dlio_benchmark/data_generator/parquet_generator.py`, `dlio_benchmark/dlio_benchmark/utils/config.py`

**Action:**

### Part A: Update `_build_schema()` in `parquet_generator.py` (line 65)

1. Change the default `size` from `1024` to `1` on line 76:
   ```python
   size = int(col_spec.get('size', 1))
   ```

2. Add scalar handling for `size == 1` and add `int8`/`float16` dtype cases. Replace the `if dtype in ('float32', 'float64'):` block (lines 82-85) with:

   ```python
   if dtype == 'int8':
       if size == 1:
           fields.append(pa.field(name, pa.int8()))
       else:
           fields.append(pa.field(name, pa.list_(pa.int8(), size)))
   elif dtype == 'float16':
       if size == 1:
           fields.append(pa.field(name, pa.float16()))
       else:
           fields.append(pa.field(name, pa.list_(pa.float16(), size)))
   elif dtype in ('float32', 'float64'):
       pa_inner = pa.float32() if dtype == 'float32' else pa.float64()
       if size == 1:
           fields.append(pa.field(name, pa_inner))
       else:
           fields.append(pa.field(name, pa.list_(pa_inner, size)))
   ```

### Part B: Update `_generate_column_data_batch()` in `parquet_generator.py` (line 100)

1. Change the default `size` from `1024` to `1` on line 111:
   ```python
   size = int(col_spec.get('size', 1))
   ```

2. Add `int8` and `float16` cases and scalar handling for `size == 1`. Replace the `if dtype in ('float32', 'float64'):` block (lines 117-125) with:

   ```python
   if dtype == 'int8':
       if size == 1:
           data = np.random.randint(-128, 127, size=batch_size, dtype=np.int8)
           return name, pa.array(data, type=pa.int8())
       else:
           data = np.random.randint(-128, 127, size=(batch_size, size), dtype=np.int8)
           flat_data = data.ravel()
           arrow_flat = pa.array(flat_data, type=pa.int8())
           arrow_data = pa.FixedSizeListArray.from_arrays(arrow_flat, size)
           return name, arrow_data

   if dtype == 'float16':
       if size == 1:
           data = np.random.rand(batch_size).astype(np.float16)
           return name, pa.array(data, type=pa.float16())
       else:
           data = np.random.rand(batch_size, size).astype(np.float16)
           flat_data = data.ravel()
           arrow_flat = pa.array(flat_data, type=pa.float16())
           arrow_data = pa.FixedSizeListArray.from_arrays(arrow_flat, size)
           return name, arrow_data

   if dtype in ('float32', 'float64'):
       np_dtype = np.float32 if dtype == 'float32' else np.float64
       if size == 1:
           data = np.random.rand(batch_size).astype(np_dtype)
           pa_type = pa.float32() if dtype == 'float32' else pa.float64()
           return name, pa.array(data, type=pa_type)
       else:
           data = np.random.rand(batch_size, size).astype(np_dtype)
           flat_data = data.ravel()
           arrow_flat = pa.array(flat_data)
           arrow_data = pa.FixedSizeListArray.from_arrays(arrow_flat, size)
           return name, arrow_data
   ```

### Part C: Update `config.py` default size (line 812)

1. Change the default `size` from `1024` to `1` on line 812:
   ```python
   'size': col_spec.get('size', 1),
   ```

2. Also change the fallback default on line 819:
   ```python
   'size': 1,
   ```

**Verify:** Run the following Python script to confirm int8, float16, float32, float64 all work as scalar and list types:
```
python -c "
import pyarrow as pa
import numpy as np

# Test scalar types
s = pa.schema([('a', pa.int8()), ('b', pa.float16()), ('c', pa.float32()), ('d', pa.float64())])
print('Schema OK:', s)

# Test scalar arrays
d1 = pa.array(np.array([1,2,3], dtype=np.int8), type=pa.int8())
d2 = pa.array(np.array([1.0,2.0,3.0], dtype=np.float16), type=pa.float16())
d3 = pa.array(np.array([1.0,2.0,3.0], dtype=np.float32), type=pa.float32())
d4 = pa.array(np.array([1.0,2.0,3.0], dtype=np.float64), type=pa.float64())
print('Scalar arrays OK:', len(d1), len(d2), len(d3), len(d4))

# Test list types
flat_i8 = pa.array(np.array([1,2,3,4,5,6], dtype=np.int8), type=pa.int8())
list_i8 = pa.FixedSizeListArray.from_arrays(flat_i8, 2)
print('List int8 OK:', len(list_i8))

flat_f16 = pa.array(np.array([1.0,2.0,3.0,4.0,5.0,6.0], dtype=np.float16), type=pa.float16())
list_f16 = pa.FixedSizeListArray.from_arrays(flat_f16, 2)
print('List float16 OK:', len(list_f16))
print('All dtype tests passed')
"
```
Also verify the default size change doesn't break existing configs by checking that `config.py` line 812 now reads `'size': col_spec.get('size', 1)`.

**Done:** The DLIO parquet generator handles `int8` and `float16` dtypes for both scalar (size=1) and list columns. Scalar PyArrow types are used for size=1 (not FixedSizeListArray). Default `size` is 1 in generator, data generation, and config parsing.

## Wave 2 — DLRM config files and E2E verification

## Task 2: Generate and apply 200-column DLRM config with randomly distributed read flags

**Files:** `configs/dlio/workload/dlrm_b200.yaml`, `configs/dlio/workload/dlrm_mi355.yaml`, `configs/dlio/workload/dlrm_datagen.yaml`

**Action:**

1. Use the following Python script to generate the column YAML section. The script produces 200 columns with:
   - 40 read columns totaling exactly 160 bytes, randomly distributed across all 200 positions
   - Read columns follow DLRM patterns: 1 label (float32), 13 numerical (float32), 26 categorical (4×int8, 4×float16, 13×float32, 5×float64)
   - 160 unread columns with random dtypes (seeded for reproducibility)
   - `size` is omitted from all columns (defaults to 1)
   - `record_length_bytes` = total bytes of all 200 columns

   ```python
   import random
   
   random.seed(42)
   
   DTYPE_SIZES = {'int8': 1, 'float16': 2, 'float32': 4, 'float64': 8}
   
   # Define the 40 read columns with their dtypes (totaling 160 bytes)
   # 1 label (float32=4) + 13 numerical (float32=52) + 26 categorical (4×int8=4, 4×float16=8, 13×float32=52, 5×float64=40)
   # Total: 4 + 52 + 4 + 8 + 52 + 40 = 160 bytes
   read_dtypes = (
       ['float32'] * 1 +      # label
       ['float32'] * 13 +     # numerical features
       ['int8'] * 4 +         # categorical (int8)
       ['float16'] * 4 +      # categorical (float16)
       ['float32'] * 13 +     # categorical (float32)
       ['float64'] * 5         # categorical (float64)
   )
   random.shuffle(read_dtypes)  # Shuffle dtype assignment within read columns
   
   # Randomly select 40 positions out of 200 for read columns
   all_positions = list(range(200))
   read_positions = set(random.sample(all_positions, 40))
   
   # Assign dtypes to unread columns (random mix)
   unread_dtypes_pool = ['int8', 'float16', 'float32', 'float64']
   
   # Build column list
   columns = []
   read_idx = 0
   total_bytes = 0
   read_bytes = 0
   
   for i in range(200):
       if i in read_positions:
           dtype = read_dtypes[read_idx]
           read_idx += 1
           read = True
           read_bytes += DTYPE_SIZES[dtype]
       else:
           dtype = random.choice(unread_dtypes_pool)
           read = False
       
       total_bytes += DTYPE_SIZES[dtype]
       columns.append({'name': f'feature_{i:03d}', 'dtype': dtype, 'read': read})
   
   # Verify constraints
   assert read_bytes == 160, f"Read bytes: {read_bytes}, expected 160"
   assert sum(1 for c in columns if c['read']) == 40, "Expected 40 read columns"
   assert sum(1 for c in columns if not c['read']) == 160, "Expected 160 unread columns"
   
   # Print YAML
   print(f"    # record_length_bytes should be: {total_bytes}")
   print(f"    # Read columns: {sum(1 for c in columns if c['read'])} totaling {read_bytes} bytes")
   print(f"    # Unread columns: {sum(1 for c in columns if not c['read'])}")
   print(f"    columns:")
   for col in columns:
       read_str = 'true' if col['read'] else 'false'
       print(f"      - name: {col['name']}")
       print(f"        dtype: {col['dtype']}")
       print(f"        read: {read_str}")
   
   print(f"\n# Set record_length_bytes to: {total_bytes}")
   ```

2. Run the script and capture the output. Use the generated `columns:` YAML section and `record_length_bytes` value.

3. Update all three DLRM config files with the generated columns. For each file:
   - Replace `record_length_bytes: 160` with `record_length_bytes: <total_bytes from script>`
   - Replace the existing `columns:` section (the 3 columns: label, numerical_features, categorical_features) with the generated 200-column section
   - Keep all other settings unchanged (workflow, reader, train, metric sections for b200/mi355; workflow section for datagen)
   - Do NOT include `size:` in any column definition (it defaults to 1)

4. Verify all 3 files have identical column definitions by diffing the `columns:` sections.

**Verify:** Run:
```
python -c "
import yaml

for fname in ['configs/dlio/workload/dlrm_b200.yaml', 'configs/dlio/workload/dlrm_mi355.yaml', 'configs/dlio/workload/dlrm_datagen.yaml']:
    with open(fname) as f:
        d = yaml.safe_load(f)
    cols = d['dataset']['parquet']['columns']
    read_cols = [c for c in cols if c.get('read', True)]
    unread = [c for c in cols if not c.get('read', True)]
    dtypes = set(c['dtype'] for c in cols)
    
    # Check no column has 'size' key
    has_size = any('size' in c for c in cols)
    
    # Calculate read bytes
    dtype_sizes = {'int8': 1, 'float16': 2, 'float32': 4, 'float64': 8}
    read_bytes = sum(dtype_sizes[c['dtype']] for c in read_cols)
    
    # Check read columns are NOT all sequential
    read_indices = [int(c['name'].split('_')[1]) for c in read_cols]
    is_sequential = read_indices == list(range(min(read_indices), max(read_indices)+1))
    
    print(f'{fname}:')
    print(f'  Total columns: {len(cols)}')
    print(f'  Read columns: {len(read_cols)} ({read_bytes} bytes)')
    print(f'  Unread columns: {len(unread)}')
    print(f'  Dtypes: {dtypes}')
    print(f'  Has size key: {has_size}')
    print(f'  Read sequential: {is_sequential}')
    print(f'  record_length_bytes: {d[\"dataset\"][\"record_length_bytes\"]}')
    print()
"
```
Expected output for each file: Total=200, Read=40 (160 bytes), Unread=160, Dtypes={int8, float16, float32, float64}, Has size key=False, Read sequential=False.

**Done:** All three DLRM config files contain 200 individual columns with mixed dtypes, 40 randomly distributed as `read: true` totaling 160 bytes, 160 as `read: false`, no `size` key in any column, and `record_length_bytes` set to the total of all 200 columns.

## Task 3: End-to-end verification — generate parquet data and read it back

**Files:** (no files created — verification only)

**Action:**

1. Run a small-scale data generation using the DLRM datagen config to produce parquet files with the new 200-column schema. Use a reduced dataset size for quick verification:

   ```bash
   cd dlio_benchmark && python -m dlio_benchmark.main \
     workload=dlrm_datagen \
     ++dataset.num_files_train=2 \
     ++dataset.num_samples_per_file=100 \
     ++dataset.data_folder=/tmp/dlrm_e2e_test
   ```

   This should generate 2 parquet files with 100 samples each, using the 200-column schema.

2. Read back the generated files and verify:
   ```python
   import pyarrow.parquet as pq
   import os
   
   data_dir = '/tmp/dlrm_e2e_test'
   files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
   assert len(files) >= 1, f"Expected parquet files, found: {files}"
   
   pf = pq.ParquetFile(os.path.join(data_dir, files[0]))
   schema = pf.schema_arrow
   
   print(f"Columns: {schema.names[:5]}... ({len(schema.names)} total)")
   assert len(schema.names) == 200, f"Expected 200 columns, got {len(schema.names)}"
   
   # Verify dtype diversity
   type_names = set(str(schema.field(i).type) for i in range(len(schema)))
   print(f"PyArrow types found: {type_names}")
   assert 'int8' in type_names, "Missing int8 columns"
   assert 'halffloat' in type_names or 'float16' in type_names, "Missing float16 columns"
   assert 'float' in type_names or 'float32' in type_names, "Missing float32 columns"
   assert 'double' in type_names or 'float64' in type_names, "Missing float64 columns"
   
   # Verify all columns are scalar (not list types)
   for i in range(len(schema)):
       field_type = schema.field(i).type
       assert not hasattr(field_type, 'list_size'), f"Column {schema.field(i).name} is a list type, expected scalar"
   
   # Read a row group and verify data is present
   table = pf.read_row_group(0)
   print(f"Rows: {table.num_rows}")
   assert table.num_rows > 0, "No rows in file"
   
   # Verify selective column reading works
   import yaml
   with open('../configs/dlio/workload/dlrm_datagen.yaml') as f:
       config = yaml.safe_load(f)
   read_cols = [c['name'] for c in config['dataset']['parquet']['columns'] if c.get('read', True)]
   print(f"Read columns: {len(read_cols)}")
   
   # Read only the read columns
   table_selective = pf.read_row_group(0, columns=read_cols)
   print(f"Selective read columns: {table_selective.num_columns}")
   assert table_selective.num_columns == 40, f"Expected 40 read columns, got {table_selective.num_columns}"
   
   # Calculate read bytes per row
   row = table_selective.slice(0, 1)
   total_read_bytes = sum(row.column(i).nbytes for i in range(row.num_columns))
   print(f"Read bytes per row: {total_read_bytes}")
   
   print("E2E verification PASSED")
   ```

3. Clean up the test data:
   ```bash
   rm -rf /tmp/dlrm_e2e_test
   ```

**Verify:** The Python verification script above prints "E2E verification PASSED" without errors. All assertions pass: 200 columns, all 4 dtypes present, all scalar types (no lists), selective reading returns 40 columns.

**Done:** End-to-end verification confirms that DLRM parquet data generation produces 200 scalar columns with mixed dtypes (int8, float16, float32, float64), and selective column reading correctly returns only the 40 read columns.

## Notes

- The column generation script uses `random.seed(42)` for reproducibility — the same seed produces the same column layout every time
- Read column dtype distribution: 1 label(f32) + 13 numerical(f32) + 4 categorical(i8) + 4 categorical(f16) + 13 categorical(f32) + 5 categorical(f64) = 40 columns, 160 bytes
- The `size` key is intentionally omitted from all column definitions — it defaults to 1 in the updated config parser and generator
- The `read` flag filtering in `parquet_reader.py` (lines 189-191) already works correctly — no changes needed
- PyArrow's `pa.float16()` maps to IEEE 754 half-precision; NumPy's `np.float16` is compatible
- PyArrow's `pa.int8()` maps to signed 8-bit integer; NumPy's `np.int8` is compatible
- The E2E test uses a small dataset (2 files × 100 samples) for quick verification — full-scale testing is done during actual benchmark runs
- Existing configs with explicit `size` values continue to work (backward compatible)
