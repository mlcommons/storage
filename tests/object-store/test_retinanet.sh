#!/usr/bin/env bash
# =============================================================================
# test_retinanet.sh — Single-run smoke test for RetinaNet training benchmark
#
# True smoke test: NP=1, 200 files, 1 epoch — fast end-to-end sanity check
# that the pipeline works at all before turning up the heat.
#
# The [DATALOADER] log line should show:
#   TorchIterableDatasetSimple(bulk-prefetch, N workers)
# and the [INFO] streaming lines should show small chunk counts,
# confirming the bounded sliding-window path (not thundering-herd) is active.
#
# Prerequisites:
#   - s3-ultra running           (bash s3-ultra/scripts/start_s3ultra2.sh)
#   - Dataset already generated  (bash tests/object-store/gen_retinanet_jpeg.sh)
#
# Usage:
#   cd /home/eval/Documents/Code/mlp-storage
#   bash tests/object-store/test_retinanet.sh
#
#   # Override NP or file count:
#   NP=2 bash tests/object-store/test_retinanet.sh
#   NP=1 NUM_FILES=50000 bash tests/object-store/test_retinanet.sh
# =============================================================================
set -euo pipefail

REPO=/home/eval/Documents/Code/mlp-storage
NP="${NP:-1}"
NUM_FILES="${NUM_FILES:-200}"           # smoke test: just 200 files; full dataset has 500k
DATA_FOLDER="data/retinanet"
STORAGE_ROOT="${STORAGE_ROOT:-mlp-retinanet}"

cd "${REPO}"

# Load credentials; unset BUCKET so env never controls the target bucket
set -o allexport; source .env.s3-ultra; set +o allexport
unset BUCKET

source .venv/bin/activate

echo ""
echo "════════════════════════════════════════════════════════"
echo "  RetinaNet Smoke Test"
echo "  NP=${NP}   Bucket: s3://${STORAGE_ROOT}/${DATA_FOLDER}/"
echo "  Files: ${NUM_FILES}   Endpoint: ${AWS_ENDPOINT_URL}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════"
echo ""

RUST_LOG=s3dlio=info \
.venv/bin/python3 -c "from mlpstorage_py.main import main; main()" \
    training run \
    --model retinanet \
    --accelerator-type b200 \
    --num-accelerators "${NP}" \
    --num-client-hosts 1 \
    --client-host-memory-in-gb 47 \
    --dlio-bin-path "${REPO}/.venv/bin" \
    --object s3 \
    --skip-validation \
    --open \
    --params \
        storage.storage_root="${STORAGE_ROOT}" \
        dataset.num_files_train="${NUM_FILES}" \
        dataset.num_samples_per_file=1 \
        dataset.data_folder="${DATA_FOLDER}" \
        train.computation_time=0.04755 \
        train.epochs=1 \
        storage.storage_options.storage_library=s3dlio \
    2>&1

echo ""
echo "════════════════════════════════════════════════════════"
echo "  test_retinanet.sh complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Check [DATALOADER] lines above for:"
echo "    TorchIterableDatasetSimple(bulk-prefetch, N workers)"
echo "════════════════════════════════════════════════════════"
