#!/usr/bin/env bash
# =============================================================================
# sweep_dlrm_np.sh — DLRM NP (num-accelerators) sweep — Phase 2
#
# Sweeps NP=1,2,4,8 at two computation_time values (1ms and 5ms) that were
# selected from the Phase 1 compute-time sweep results.
#
#   1ms  → I/O-bound baseline  (AU ~20% at NP=1, storage bottleneck)
#   5ms  → balanced / AU sweet spot (AU ~79% at NP=1)
#
# All runs use a single host (127.0.0.1); NP controls both mpirun -n and
# the --num-accelerators argument passed to the mlpstorage_py wrapper.
#
# Dataset: 200 files × 1,536,000 samples  (bucket: mlp-dlrm / data/dlrm)
#
# Usage:
#   cd /home/eval/Documents/Code/mlp-storage
#   bash tests/object-store/sweep_dlrm_np.sh 2>&1
# =============================================================================
set -euo pipefail

REPO=/home/eval/Documents/Code/mlp-storage
VENV="${REPO}/.venv"
RESULTS_DIR="${REPO}/results/dlrm_sweep"
PYTHON="${VENV}/bin/python3"

# Dataset (matches Phase 1)
NUM_FILES=200
SAMPLES_PER_FILE=1536000
DATA_FOLDER="data/dlrm"

# Fixed computation_time values chosen from Phase 1 results
COMP_TIMES=("0.001" "0.005")
COMP_LABELS=("1ms"  "5ms")

# NP sweep
NP_VALUES=(1 2 4 8)

mkdir -p "${RESULTS_DIR}"

cd "${REPO}"
source .env
export BUCKET=mlp-dlrm

SUMMARY_TSV="${RESULTS_DIR}/sweep_np_$(date '+%Y%m%d_%H%M%S').tsv"
echo -e "computation_time\tlabel\tNP\tau_pct\tsamples_per_sec\tio_mb_per_sec\tau_met" \
    > "${SUMMARY_TSV}"

for NP in "${NP_VALUES[@]}"; do
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

        # Parse summary.json and append row to TSV
        "${PYTHON}" - "${CT}" "${LABEL}" "${NP}" "${RUN_RESULTS}" \
            >> "${SUMMARY_TSV}" 2>&1 <<'PYEOF'
import json, glob, sys

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
done

# ─── summary table ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Phase 2 complete — NP sweep (1ms + 5ms compute time)"
echo "  Results: ${SUMMARY_TSV}"
echo "============================================================"
echo ""
column -t -s $'\t' "${SUMMARY_TSV}"
echo ""
echo "Expected pattern:"
echo "  1ms: AU stays low (I/O-bound), throughput scales with NP until storage saturates"
echo "  5ms: AU stays high (~80%), throughput scales linearly with NP (compute-bound)"
