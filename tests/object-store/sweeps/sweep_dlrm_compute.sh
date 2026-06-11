#!/usr/bin/env bash
# =============================================================================
# sweep_dlrm_compute.sh — DLRM computation_time sweep
#
# Phase 1 (this script):
#   Sweep computation_time at NP=1: 375us, 1ms, 5ms, 10ms
#   Uses s3dlio Rust-based Parquet generator and s3dlio reader throughout.
#
# Dataset: 200 files × 1,536,000 samples ≈ 234 GB in bucket mlp-dlrm
#   (20% of full 1024-file spec; footer ~3.1 MiB < s3-ultra 4 MiB limit)
#
# Usage:
#   cd /home/eval/Documents/Code/mlp-storage
#
#   # Step 1 — generate data (one-time, takes a while):
#   tests/object-store/sweep_dlrm_compute.sh datagen
#
#   # Step 2 — run the sweep:
#   tests/object-store/sweep_dlrm_compute.sh
#
# After reviewing Phase 1 results, run Phase 2 (NP sweep) separately.
# =============================================================================
set -euo pipefail

REPO=/home/eval/Documents/Code/mlp-storage
VENV="${REPO}/.venv"
RESULTS_DIR="${REPO}/results/dlrm_sweep"
PYTHON="${VENV}/bin/python3"

# Dataset: 20% of spec
NUM_FILES=200
SAMPLES_PER_FILE=1536000  # 250 RGs × 6144 → ~3.1 MiB footer (under s3-ultra 4 MiB limit)
DATA_FOLDER="data/dlrm"

# Phase 1: NP=1 only
NP=1

# computation_time values to sweep (seconds)
COMP_TIMES=("0.000375" "0.001" "0.005" "0.010")
COMP_LABELS=("375us"   "1ms"   "5ms"   "10ms")

mkdir -p "${RESULTS_DIR}"

cd "${REPO}"
source .env

# Override BUCKET to the dlrm-specific bucket
export BUCKET=mlp-dlrm

# ─── datagen ──────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "datagen" ]]; then
    echo "============================================================"
    echo "  DLRM datagen — s3dlio Rust Parquet generator"
    echo "  ${NUM_FILES} files x ${SAMPLES_PER_FILE} samples = 718 GB"
    echo "  Bucket: ${BUCKET}  Path: ${DATA_FOLDER}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    RUST_LOG=s3dlio=info \
    "${PYTHON}" -c "from mlpstorage_py.main import main; main()" \
        training datagen \
        --model dlrm \
        --num-processes 1 \
        --dlio-bin-path "${VENV}/bin" \
        --object s3 \
        --skip-validation \
        --open \
        --results-dir "${RESULTS_DIR}" \
        --params \
            dataset.num_files_train=${NUM_FILES} \
            dataset.num_samples_per_file=${SAMPLES_PER_FILE} \
            dataset.data_folder=${DATA_FOLDER} \
            storage.storage_options.decode_mode=none \
            storage.storage_options.storage_library=s3dlio

    echo "============================================================"
    echo "  Datagen complete: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    exit 0
fi

# ─── Phase 1 sweep: computation_time at NP=1 ─────────────────────────────────
SUMMARY_TSV="${RESULTS_DIR}/sweep_compute_NP1_$(date '+%Y%m%d_%H%M%S').tsv"
echo -e "computation_time\tlabel\tNP\tau_pct\tsamples_per_sec\tio_mb_per_sec\tau_met" \
    > "${SUMMARY_TSV}"

for i in "${!COMP_TIMES[@]}"; do
    CT="${COMP_TIMES[$i]}"
    LABEL="${COMP_LABELS[$i]}"

    echo ""
    echo "============================================================"
    echo "  computation_time=${CT} (${LABEL})  NP=${NP}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    RUN_RESULTS="${RESULTS_DIR}/run_ct${LABEL}_NP${NP}"
    mkdir -p "${RUN_RESULTS}"

    RUST_LOG=s3dlio=info \
    "${PYTHON}" -c "from mlpstorage_py.main import main; main()" \
        training run \
        --model dlrm \
        --accelerator-type b200 \
        --num-accelerators "${NP}" \
        --num-client-hosts 1 \
        --client-host-memory-in-gb 47 \
        --dlio-bin-path "${VENV}/bin" \
        --object s3 \
        --skip-validation \
        --open \
        --results-dir "${RUN_RESULTS}" \
        --params \
            dataset.num_files_train=${NUM_FILES} \
            dataset.num_samples_per_file=${SAMPLES_PER_FILE} \
            dataset.data_folder=${DATA_FOLDER} \
            train.computation_time=${CT} \
            storage.storage_options.decode_mode=none \
            storage.storage_options.storage_library=s3dlio

    # Parse and append to summary
    "${PYTHON}" - "${CT}" "${LABEL}" "${NP}" "${RUN_RESULTS}" \
        >> "${SUMMARY_TSV}" 2>&1 <<'PYEOF'
import json, glob, os, sys

ct, label, np_, run_results = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

files = sorted(glob.glob(f"{run_results}/**/summary.json", recursive=True))
if not files:
    print(f"{ct}\t{label}\t{np_}\tN/A\tN/A\tN/A\tN/A")
    sys.exit(0)

d = json.load(open(files[-1]))
m = d.get("metric", {})

au   = m.get("train_au_mean_percentage",                 "N/A")
sps  = m.get("train_throughput_mean_samples_per_second", "N/A")
ioMB = m.get("train_io_mean_MB_per_second",              "N/A")
met  = m.get("train_au_meet_expectation",                "N/A")

def fmt(v): return f"{v:.2f}" if isinstance(v, float) else str(v)
print(f"{ct}\t{label}\t{np_}\t{fmt(au)}\t{fmt(sps)}\t{fmt(ioMB)}\t{met}")
PYEOF

done

# ─── summary table ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Phase 1 complete — computation_time sweep at NP=1"
echo "  Results: ${SUMMARY_TSV}"
echo "============================================================"
echo ""
column -t -s $'\t' "${SUMMARY_TSV}"
echo ""
echo "Next: review AU and I/O columns, pick 1-2 values, then run Phase 2 (NP sweep)."
