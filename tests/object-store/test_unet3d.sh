#!/usr/bin/env bash
# Quick single-run test for UNet3D training benchmark (NP=1, B200)
# Dataset: s3://mlp-unet3d/data/unet3d/  (7,200 NPZ files ~984 GiB)
#
# Usage:
#   cd /home/eval/Documents/Code/mlp-storage
#   bash tests/object-store/test_unet3d.sh
#
#   # Override NP:
#   NP=2 bash tests/object-store/test_unet3d.sh
set -euo pipefail

REPO=/home/eval/Documents/Code/mlp-storage
NP="${NP:-1}"

cd "${REPO}"
# Load credentials only — unset BUCKET so env never controls the target bucket
set -o allexport; source .env.s3-ultra; set +o allexport
unset BUCKET

source .venv/bin/activate

RUST_LOG=s3dlio=info \
.venv/bin/python3 -c "from mlpstorage_py.main import main; main()" \
    training run \
    --model unet3d \
    --accelerator-type b200 \
    --num-accelerators "${NP}" \
    --num-client-hosts 1 \
    --client-host-memory-in-gb 47 \
    --dlio-bin-path "${REPO}/.venv/bin" \
    --object s3 \
    --skip-validation \
    --open \
    --params \
        storage.storage_root="${STORAGE_ROOT:-mlp-unet3d}" \
        dataset.num_files_train=7200 \
        dataset.num_samples_per_file=1 \
        dataset.data_folder=data/unet3d \
        train.computation_time=0.162 \
        storage.storage_options.decode_mode=none \
        storage.storage_options.storage_library=s3dlio \
    2>&1
