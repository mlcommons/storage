#!/usr/bin/env bash
# =============================================================================
# run_flux_bench.sh — MLPerf Storage Flux benchmark runner
# =============================================================================
#
# Usage:
#   ./run_flux_bench.sh <NP> [s3dlio|s3torchconnector|minio] [simulate [log_secs]]
#
# NP = number of accelerators / MPI ranks (1, 2, 4, 8)
#
# Prerequisites:
#   - s3-ultra must be running on port 9200 (see start_s3ultra.sh)
#   - mlp-storage venv must be at /home/eval/Documents/Code/mlp-storage/.venv
#   - .env file must be present in /home/eval/Documents/Code/mlp-storage/
#   - Flux Parquet data must be present at dataset.data_folder on the S3 system
#
# Flux dataset characteristics:
#   4296 files × 288 samples/file ≈ 594 MiB/file (uncompressed)
#   Columns: t5_encodings (524328×f32), clip_encodings (409×f32),
#            mean (8232×f32), logvar (8232×f32), timestamp (7×f32)
#   Full dataset: ~2.4 TiB total
#   Default run:  64 files (~37 GiB subset)
#
# Results are written to:
#   /home/eval/Documents/Code/mlp-storage/results/flux/
#
# =============================================================================

set -euo pipefail

NP="${1:?Usage: $0 <NP> [s3dlio|s3torchconnector|minio] [simulate [log_secs]]  (e.g. ./run_flux_bench.sh 1 s3dlio simulate 30)}"
LIBRARY="${2:-s3dlio}"
SIMULATE="${3:-}"
SIM_LOG_SECS="${4:-60}"

REPO=/home/eval/Documents/Code/mlp-storage
RESULTS_DIR="${REPO}/results/flux"
VENV="${REPO}/.venv"

# 64 parquet files, 288 samples each, ~594 MiB each = ~37 GiB subset
# (full scale: 4296 files = ~2.4 TiB)
NUM_FILES=64
SAMPLES_PER_FILE=288
DATA_FOLDER="data/flux"

mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "  Flux benchmark  NP=${NP}  library=${LIBRARY}${SIMULATE:+  SIMULATE}"
echo "  Results dir: ${RESULTS_DIR}"
echo "  Files: ${NUM_FILES} × ${SAMPLES_PER_FILE} samples/file (~594 MiB each)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

cd "${REPO}"
source .env

RUST_LOG=s3dlio=info \
"${VENV}/bin/python3" -c "from mlpstorage_py.main import main; main()" \
    training run \
    --model flux \
    --accelerator-type b200 \
    --num-accelerators "${NP}" \
    --num-client-hosts 1 \
    --client-host-memory-in-gb 64 \
    --dlio-bin-path "${VENV}/bin" \
    --object s3 \
    --skip-validation \
    --results-dir "${RESULTS_DIR}" \
    --params \
        dataset.num_files_train=${NUM_FILES} \
        dataset.num_samples_per_file=${SAMPLES_PER_FILE} \
        dataset.data_folder=${DATA_FOLDER} \
        storage.storage_options.decode_mode=none \
        storage.storage_options.storage_library=${LIBRARY} \
        ${SIMULATE:+storage.storage_options.simulate_io=true} \
        ${SIMULATE:+storage.storage_options.sim_log_secs=${SIM_LOG_SECS}}

echo ""
echo "============================================================"
echo "  Run complete — parsing results"
echo "============================================================"

# Print throughput from the most recent run's metadata.json
"${VENV}/bin/python3" - <<'PYEOF'
import json, glob, os

results_dir = "/home/eval/Documents/Code/mlp-storage/results/flux"
files = sorted(glob.glob(f"{results_dir}/**/training_*_metadata.json", recursive=True))
if not files:
    print("  No metadata.json found.")
    exit(0)

latest = files[-1]
d = json.load(open(latest))
np_ = d.get("num_processes", "?")
runtime = d.get("runtime", None)

print(f"  Run dir:    {os.path.dirname(latest).split('/')[-1]}")
print(f"  NP:         {np_}")

if runtime:
    # 64 files × ~594 MiB each
    total_gb = 64 * 594 / 1024
    mbps = total_gb * 1024 / runtime
    print(f"  Runtime:    {runtime:.1f} s")
    print(f"  Throughput: {mbps:.0f} MB/s  ({total_gb:.1f} GiB / {runtime:.1f} s)")
else:
    print("  Runtime not found in metadata")

# Also print DLIO's own summary if it exists
run_dir = os.path.dirname(latest)
summary_path = os.path.join(run_dir, "summary.json")
if os.path.exists(summary_path):
    s = json.load(open(summary_path))
    m = s.get("metric", {})
    au_mean = m.get("train_au_mean_percentage")
    tput_mean = m.get("train_throughput_mean_samples_per_second")
    io_mean = m.get("train_io_mean_MB_per_second")
    au_ok = m.get("train_au_meet_expectation", "?")
    if au_mean is not None:
        print(f"  AU mean:    {au_mean:.1f}%  ({au_ok})")
    if tput_mean is not None:
        print(f"  Samples/s:  {tput_mean:.0f}")
    if io_mean is not None:
        print(f"  DLIO I/O:   {io_mean:.0f} MB/s")
else:
    print("  (no summary.json — DLIO may have crashed during finalize)")
PYEOF

echo "============================================================"
