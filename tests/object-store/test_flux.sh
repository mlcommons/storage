cd /home/eval/Documents/Code/mlp-storage && \
source .env && \
RUST_LOG=s3dlio=info \
.venv/bin/python3 -c "from mlpstorage_py.main import main; main()" \
  training run \
  --model flux --accelerator-type b200 --num-accelerators 1 \
  --num-client-hosts 1 --client-host-memory-in-gb 64 \
  --dlio-bin-path /home/eval/Documents/Code/mlp-storage/.venv/bin \
  --object s3 --skip-validation \
  --params \
    dataset.num_files_train=64 \
    dataset.num_samples_per_file=288 \
    dataset.data_folder=data/flux \
    storage.storage_options.decode_mode=none \
  2>&1
