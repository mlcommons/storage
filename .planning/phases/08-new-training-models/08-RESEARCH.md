# Phase 8: New Training Models - Research

**Researched:** 2026-01-24
**Domain:** MLPerf Training benchmark workload configurations for DLRM, RetinaNet, and Flux models
**Confidence:** MEDIUM

## Summary

This research investigates adding three new training models (dlrm, retinanet, flux) to the MLPerf Storage benchmark suite. These models represent diverse machine learning workloads: recommendation systems (DLRM), object detection (RetinaNet), and text-to-image generation (Flux).

The MLPerf Storage benchmark emulates I/O patterns of training workloads via DLIO (Deep Learning I/O) benchmark without requiring actual GPUs. Each new model requires YAML configuration files for both training execution (per accelerator type) and data generation, plus updates to validation rules for CLOSED/OPEN submission categories.

The existing pattern in this codebase (unet3d, resnet50, cosmoflow) provides a clear template: model constants in `config.py`, YAML workload configs in `configs/dlio/workload/`, model validation in `rules/run_checkers/training.py`, and CLI integration in `cli/training_args.py`.

**Primary recommendation:** Follow the existing model pattern precisely. Add model constants to `config.py`, create accelerator-specific and datagen YAML configs in `configs/dlio/workload/`, and extend the validation rules. The key challenge is determining accurate I/O characteristics (record sizes, batch sizes, computation times) that emulate real training behavior.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| DLIO Benchmark | 2.0+ | I/O workload emulation | MLPerf Storage standard engine |
| PyTorch/TensorFlow | Various | Framework for data loaders | Standard ML framework support |
| Hydra | 1.x | YAML configuration management | DLIO dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| NumPy | 1.x | npz file format support | UNet3D-style workloads |
| TFRecords | - | TensorFlow record format | ResNet50/CosmoFlow-style workloads |
| WebDataset | - | Tar-based sharded format | Large-scale image datasets (potential for Flux) |
| Parquet | - | Column-oriented format | DLRM Criteo data (Phase 9) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Synthetic data | Real datasets | Synthetic allows arbitrary sizing; real requires download/preprocessing |
| DLIO emulation | Actual training | DLIO removes GPU requirement; real training needs GPUs |

**Installation:**
```bash
pip install -e ".[full]"  # Includes DLIO and framework dependencies
```

## Architecture Patterns

### Recommended Project Structure
```
mlpstorage/
  config.py                           # Add model constants: DLRM, RETINANET, FLUX
  cli/training_args.py                # Update MODELS list if needed
  rules/run_checkers/training.py      # Add model-specific validation rules

configs/dlio/workload/
  dlrm_h100.yaml                      # DLRM H100 accelerator config
  dlrm_a100.yaml                      # DLRM A100 accelerator config
  dlrm_datagen.yaml                   # DLRM data generation config
  retinanet_h100.yaml                 # RetinaNet H100 accelerator config
  retinanet_a100.yaml                 # RetinaNet A100 accelerator config
  retinanet_datagen.yaml              # RetinaNet data generation config
  flux_h100.yaml                      # Flux H100 accelerator config
  flux_a100.yaml                      # Flux A100 accelerator config
  flux_datagen.yaml                   # Flux data generation config
```

### Pattern 1: Model Configuration in config.py

**What:** Define model constants and add to MODELS list
**When to use:** For every new training model
**Example:**
```python
# Source: Existing pattern in mlpstorage/config.py
COSMOFLOW = "cosmoflow"
RESNET = "resnet50"
UNET = "unet3d"
DLRM = "dlrm"         # New
RETINANET = "retinanet"  # New
FLUX = "flux"          # New
MODELS = [COSMOFLOW, RESNET, UNET, DLRM, RETINANET, FLUX]
```

### Pattern 2: DLIO Workload YAML Configuration

**What:** YAML file defining model I/O characteristics for DLIO emulation
**When to use:** Each model-accelerator combination needs a config
**Example:**
```yaml
# Source: Existing pattern from configs/dlio/workload/unet3d_h100.yaml
model:
  name: modelname
  type: modeltype  # cnn, recommendation, transformer, etc.

framework: pytorch  # or tensorflow

workflow:
  generate_data: False  # True for datagen configs
  train: True
  checkpoint: False

dataset:
  data_folder: data/modelname/
  format: npz  # npz, tfrecord, jpeg, png, hdf5
  num_files_train: 168
  num_samples_per_file: 1
  record_length_bytes: 146600628
  record_length_bytes_stdev: 68341808
  record_length_bytes_resize: 2097152

reader:
  data_loader: pytorch  # pytorch, tensorflow
  batch_size: 7
  read_threads: 4
  file_shuffle: seed
  sample_shuffle: seed

train:
  epochs: 5
  computation_time: 0.323  # Seconds per batch

metric:
  au: 0.90  # Accelerator utilization target
```

### Pattern 3: Datagen Configuration

**What:** YAML configuration for generating synthetic training data
**When to use:** Each model needs a datagen config
**Example:**
```yaml
# Source: Existing pattern from configs/dlio/workload/unet3d_datagen.yaml
model:
  name: modelname

framework: pytorch

workflow:
  generate_data: True
  train: False
  checkpoint: False

dataset:
  data_folder: data/modelname/
  format: npz
  num_files_train: 168
  num_samples_per_file: 1
  record_length_bytes: 146600628
  record_length_bytes_stdev: 68341808
```

### Anti-Patterns to Avoid
- **Hardcoding accelerator configs in code:** Use YAML files that DLIO loads dynamically
- **Ignoring record_length_bytes accuracy:** Incorrect values lead to meaningless benchmark results
- **Skipping datagen config:** Users need to generate synthetic data; real datasets are impractical

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| I/O emulation | Custom data readers | DLIO benchmark | DLIO handles MPI, shuffling, prefetching, threading |
| Config management | Argparse overrides | Hydra YAML + DLIO | Hydra provides composable configs, override syntax |
| Data generation | Custom synthetic data | DLIO generate_data workflow | DLIO generates correct formats with proper distribution |
| Validation rules | Ad-hoc checks | TrainingRunRulesChecker | Existing checker handles CLOSED/OPEN logic |

**Key insight:** The DLIO benchmark and existing codebase patterns handle all the complexity. New models only require configuration, not new code logic.

## Common Pitfalls

### Pitfall 1: Incorrect Record Length Calculations
**What goes wrong:** Benchmark doesn't accurately reflect real workload I/O intensity
**Why it happens:** Record lengths estimated rather than measured from real data
**How to avoid:** Calculate from actual dataset specifications:
  - DLRM: Multi-hot Criteo samples have variable sizes (~hundreds of bytes to KB range per sample)
  - RetinaNet: OpenImages JPEGs vary by resolution (avg ~150KB at 800x800)
  - Flux: Image + caption pairs at 1024x1024 resolution (~3-4MB per sample)
**Warning signs:** Benchmark completes too fast or too slow compared to real training

### Pitfall 2: Mismatched Framework/Data Loader
**What goes wrong:** DLIO errors about incompatible format/loader combinations
**Why it happens:** Not all formats work with all data loaders
**How to avoid:** Follow known working combinations:
  - `tfrecord` -> `tensorflow` loader
  - `npz` -> `pytorch` loader
  - `jpeg/png` -> either loader
**Warning signs:** DLIO fails to read generated data

### Pitfall 3: Overlooking Multiple Samples Per File
**What goes wrong:** Data generation creates wrong number of files
**Why it happens:** Assuming 1 sample per file for all models
**How to avoid:** Check DLIO limitations:
  - DLIO currently supports multiple samples per file for tfrecord only
  - npz, jpeg, hdf5 expect 1 sample per file
**Warning signs:** `num_files_train * num_samples_per_file != expected_total_samples`

### Pitfall 4: Missing Accelerator Utilization Target
**What goes wrong:** CLOSED submission fails validation
**Why it happens:** Forgetting `metric.au` setting
**How to avoid:** Always include accelerator utilization metric (typically 0.70-0.90)
**Warning signs:** Validation errors about missing AU target

### Pitfall 5: Incorrect Computation Time
**What goes wrong:** I/O and compute overlap is unrealistic
**Why it happens:** Using arbitrary computation_time instead of measured values
**How to avoid:** Computation time should reflect actual model forward/backward pass time
**Warning signs:** Benchmark is I/O bound when real training is compute bound (or vice versa)

## Code Examples

Verified patterns from official sources:

### Adding Model Constant to config.py
```python
# Source: mlpstorage/config.py pattern
DLRM = "dlrm"
RETINANET = "retinanet"
FLUX = "flux"
MODELS = [COSMOFLOW, RESNET, UNET, DLRM, RETINANET, FLUX]
```

### DLRM H100 Workload Config
```yaml
# Derived from MLPerf Training DLRMv2 specifications
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
  format: npz  # Note: parquet preferred but requires Phase 9
  num_files_train: 65536  # High file count for recommendation data
  num_samples_per_file: 1
  record_length_bytes: 512  # Approximate per-sample size for multi-hot encoding

reader:
  data_loader: pytorch
  batch_size: 8192  # DLRMv2 uses large batches (65536 global / 8 GPUs)
  read_threads: 8
  file_shuffle: seed
  sample_shuffle: seed

train:
  epochs: 1  # DLRM trains for 1 epoch
  computation_time: 0.005  # Fast per-batch computation

metric:
  au: 0.70  # Lower AU due to high I/O bandwidth requirements
```

### RetinaNet H100 Workload Config
```yaml
# Derived from MLPerf Training RetinaNet specifications
model:
  name: retinanet
  type: cnn

framework: pytorch

workflow:
  generate_data: False
  train: True
  checkpoint: False

dataset:
  data_folder: data/retinanet/
  format: jpeg
  num_files_train: 1743042  # OpenImages training set size
  num_samples_per_file: 1
  record_length_bytes: 153600  # ~150KB average JPEG at 800x800
  record_length_bytes_stdev: 51200

reader:
  data_loader: pytorch
  batch_size: 16  # Per-GPU batch size
  read_threads: 8
  file_shuffle: seed
  sample_shuffle: seed

train:
  epochs: 5
  computation_time: 0.180  # Object detection is compute intensive

metric:
  au: 0.85
```

### Flux H100 Workload Config
```yaml
# Derived from MLPerf Training Flux.1 specifications
model:
  name: flux
  type: diffusion

framework: pytorch

workflow:
  generate_data: False
  train: True
  checkpoint: False

dataset:
  data_folder: data/flux/
  format: jpeg  # CC12M uses image+caption pairs
  num_files_train: 1099776  # CC12M subset used by MLPerf
  num_samples_per_file: 1
  record_length_bytes: 307200  # 1024x1024 JPEG average
  record_length_bytes_stdev: 102400

reader:
  data_loader: pytorch
  batch_size: 8  # Memory constrained for 11.9B model
  read_threads: 8
  file_shuffle: seed
  sample_shuffle: seed

train:
  epochs: 5
  computation_time: 0.850  # Large transformer model

metric:
  au: 0.80
```

### Validation Rule Example
```python
# Source: Pattern from mlpstorage/rules/run_checkers/training.py
def check_model_specific_params(self) -> Optional[Issue]:
    """Check model-specific parameter requirements."""
    if self.benchmark_run.model == DLRM:
        # DLRM requires high batch sizes
        batch_size = self.benchmark_run.parameters.get('reader', {}).get('batch_size', 0)
        if batch_size < 2048:
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message=f"DLRM CLOSED requires batch_size >= 2048",
                parameter="reader.batch_size",
                expected=">= 2048",
                actual=batch_size
            )
    return None
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| DLRM v1 | DLRMv2 (DCNv2) | MLPerf v3.0 | Multi-hot encoding, larger batches |
| Stable Diffusion v2 | Flux.1 | MLPerf v5.1 (Oct 2025) | 11.9B params, rectified flows |
| SSD for detection | RetinaNet | MLPerf v2.0 | Single-stage detection, OpenImages |
| FID/CLIP eval | Validation loss | MLPerf v5.1 | Faster evaluation for image gen |

**Deprecated/outdated:**
- DLRM v1: Replaced by DLRMv2 with DCNv2 cross-layer
- Stable Diffusion: Replaced by Flux.1 for training benchmarks (still in inference)
- COCO for RetinaNet training: Now uses OpenImages (COCO still for eval)

## Model-Specific Details

### DLRM (Deep Learning Recommendation Model)

**MLPerf Training Reference:**
- **Model:** DLRMv2 (DCNv2) with 167M parameters
- **Dataset:** Criteo 3.5TB multi-hot (24 days of click logs)
- **Format:** Parquet (converted from raw); DLIO may use npz as synthetic approximation
- **Batch Size:** 65536 global (8192 per GPU with 8 GPUs reference)
- **Epochs:** 1 (reaches target in < 1 epoch)
- **Target:** AUROC 0.8030
- **I/O Characteristics:** High bandwidth requirements (13.4 GB/s training, 41.4 GB/s eval)

**Key Implementation Notes:**
- Very high file counts (millions of samples)
- Parquet format ideal but requires Phase 9 DLIO support
- Can approximate with npz or tfrecord for initial implementation

### RetinaNet (Object Detection)

**MLPerf Training Reference:**
- **Model:** RetinaNet with ResNeXt50 backbone, 37M parameters
- **Dataset:** OpenImages (~1.7M training images)
- **Format:** JPEG images with annotations
- **Batch Size:** Varies by hardware (16-96 per GPU typical)
- **Quality Target:** mAP 0.3757 (IoU=0.50:0.95)
- **I/O Characteristics:** Image loading with preprocessing

**Key Implementation Notes:**
- JPEG format native support in DLIO
- Large per-sample size (images vary 100KB-500KB)
- Evaluation after each epoch with validation set

### Flux.1 (Text-to-Image Generation)

**MLPerf Training Reference:**
- **Model:** Flux.1 transformer-based diffusion, 11.9B parameters
- **Dataset:** CC12M subset (1,099,776 samples)
- **Format:** JPEG images with text captions
- **Batch Size:** 8-16 per GPU (memory constrained)
- **Training Duration:** ~95 minutes on 64 B200 GPUs
- **Target:** Validation loss 0.586

**Key Implementation Notes:**
- Largest model being added (11.9B params)
- Image-caption pairs require handling both modalities
- BF16 precision standard
- Rectified flow training differs from traditional diffusion

## Open Questions

Things that couldn't be fully resolved:

1. **Exact computation_time values for H100/A100**
   - What we know: General ratios between models
   - What's unclear: Precise per-batch times for storage benchmark emulation
   - Recommendation: Start with estimates, validate with real benchmarks, iterate

2. **DLRM synthetic data format**
   - What we know: Real data is Parquet/binary; DLIO supports npz/tfrecord
   - What's unclear: Best synthetic approximation without parquet support
   - Recommendation: Use npz initially; update to parquet in Phase 9

3. **Flux caption handling in DLIO**
   - What we know: Flux needs image+caption pairs
   - What's unclear: How DLIO handles multi-modal data
   - Recommendation: Research DLIO capabilities; may need image-only approximation

4. **CLOSED vs OPEN validation rules specifics**
   - What we know: MLPerf defines allowed variations
   - What's unclear: Exact parameter constraints for new models
   - Recommendation: Review MLPerf training rules; start with OPEN flexibility

## Sources

### Primary (HIGH confidence)
- mlpstorage/config.py - Existing model constant pattern
- mlpstorage/benchmarks/dlio.py - TrainingBenchmark implementation
- mlpstorage/rules/run_checkers/training.py - Validation rule pattern
- configs/dlio/workload/*.yaml - Existing YAML configuration patterns

### Secondary (MEDIUM confidence)
- [MLCommons Training Repository](https://github.com/mlcommons/training) - Model specifications
- [MLPerf Training Flux.1 Announcement](https://mlcommons.org/2025/10/training-flux1/) - Flux benchmark details
- [NVIDIA MLPerf Blog](https://developer.nvidia.com/blog/boosting-mlperf-training-performance-with-full-stack-optimization/) - RetinaNet details
- [DLIO Benchmark Documentation](https://dlio-benchmark.readthedocs.io/en/latest/config.html) - Configuration options

### Tertiary (LOW confidence)
- WebSearch results for DLRM Criteo data format - Needs validation with actual reference
- WebSearch results for computation times - Estimates only
- Flux training parameters - Based on announcement, not reference implementation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Uses existing DLIO/YAML pattern established in codebase
- Architecture: HIGH - Direct extension of existing model configuration pattern
- Pitfalls: MEDIUM - Based on DLIO documentation and existing model issues
- Model specifications: MEDIUM - Based on MLPerf announcements, may differ from final reference implementations

**Research date:** 2026-01-24
**Valid until:** 60 days (model specifications may update with MLPerf releases)
