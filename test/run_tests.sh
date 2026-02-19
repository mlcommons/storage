#!/usr/bin/env bash

# Fail if any command fails:
set -e

# Function to run a command and handle errors
run_command() {
    local cmd="$1"
    local expected_exit_code="${2:-0}"  # Default expected exit code is 0 (success)

    echo "Running: $cmd"
    echo "Expected exit code: $expected_exit_code"

    # Create temporary files for stdout and stderr
    local stdout_file=$(mktemp)
    local stderr_file=$(mktemp)

    # Run the command, capturing stdout, stderr, and exit status
    set +e  # Temporarily disable automatic exit on error
    eval "$cmd" > "$stdout_file" 2> "$stderr_file"
    local exit_status=$?
    set -e  # Re-enable automatic exit on error

    # Display stdout regardless of success/failure
    cat "$stdout_file"

    # Display stderr if there is any
    if [ -s "$stderr_file" ]; then
        echo "STDERR output:"
        cat "$stderr_file"
    fi

    # Check if exit code matches expected exit code
    if [ $exit_status -eq $expected_exit_code ]; then
        echo "Command completed with expected exit status $exit_status"
    else
        echo "ERROR: Command failed with unexpected exit status"
        echo "Command: $cmd"
        echo "Actual exit code: $exit_status"
        echo "Expected exit code: $expected_exit_code"

        # Clean up temp files
        rm -f "$stdout_file" "$stderr_file"

        # Exit with the actual error code
        exit $exit_status
    fi

    # Clean up temp files
    rm -f "$stdout_file" "$stderr_file"
    echo "----------------------------------------"
}

# Define the list of commands to run with their expected exit codes
# Format: "command|expected_exit_code"
# If expected_exit_code is omitted, 0 (success) is assumed
commands=(
    "mlpstorage --version|0"
    "mlpstorage history show|0"

#     Example of a command expected to fail with exit code 2 (INVALID_ARGUMENTS)
    "mlpstorage training datasize --invalid-flag|2"

    "mlpstorage training datasize --model resnet50 --client-host-memory-in-gb 256 --max-accelerators 8 --num-client-hosts 2 --accelerator-type a100|0"
    "mlpstorage training datasize --model resnet50 --client-host-memory-in-gb 256 --max-accelerators 8 --num-client-hosts 2 --accelerator-type h100|0"
    "mlpstorage training datasize --model cosmoflow --client-host-memory-in-gb 256 --max-accelerators 8 --num-client-hosts 2 --accelerator-type h100|0"
    "mlpstorage training datasize --model unet3d --client-host-memory-in-gb 256 --max-accelerators 8 --num-client-hosts 2 --accelerator-type h100|0"

    "mlpstorage training datagen --hosts 127.0.0.1,127.0.0.1 --model resnet50 --num-processes 16 --param dataset.num_files_train=12 --data-dir /mnt/nvme/test_data --results-dir /mnt/nvme/results --allow-run-as-root|0"
    "mlpstorage training datagen --hosts 127.0.0.1,127.0.0.1 --model cosmoflow --num-processes 16 --param dataset.num_files_train=12 --data-dir /mnt/nvme/test_data --results-dir /mnt/nvme/results --allow-run-as-root|0"
    "mlpstorage training datagen --hosts 127.0.0.1,127.0.0.1 --model unet3d --num-processes 16 --param dataset.num_files_train=12 --data-dir /mnt/nvme/test_data --results-dir /mnt/nvme/results --allow-run-as-root|0"

    "mlpstorage training run --hosts 127.0.0.1,127.0.0.1 --num-client-hosts 2 --client-host-memory-in-gb 256 --num-accelerators 2 --accelerator-type a100 --model resnet50 --param dataset.num_files_train=12 --data-dir /mnt/nvme/test_data --results-dir /mnt/nvme/results --allow-run-as-root|0"
    "mlpstorage training run --hosts 127.0.0.1,127.0.0.1 --num-client-hosts 2 --client-host-memory-in-gb 256 --num-accelerators 2 --accelerator-type a100 --model cosmoflow --param dataset.num_files_train=12 --data-dir /mnt/nvme/test_data --results-dir /mnt/nvme/results --allow-run-as-root|0"

    # Checkpoint folder required for unet3d
    "mlpstorage training run --hosts 127.0.0.1,127.0.0.1 --num-client-hosts 2 --client-host-memory-in-gb 256 --num-accelerators 2 --accelerator-type a100 --model unet3d --param dataset.num_files_train=12 --data-dir /mnt/nvme/test_data --results-dir /mnt/nvme/results --allow-run-as-root|2"
    "mlpstorage training run --hosts 127.0.0.1,127.0.0.1 --num-client-hosts 2 --client-host-memory-in-gb 256 --num-accelerators 2 --accelerator-type a100 --model unet3d --param dataset.num_files_train=12 --data-dir /mnt/nvme/test_data --checkpoint-folder /mnt/nvme/test_data/unet3d_checkpoints --results-dir /mnt/nvme/results --allow-run-as-root|0"

    "mlpstorage checkpointing datasize --hosts 127.0.0.1,127.0.0.1 --client-host-memory-in-gb 256 --model llama3-8b --num-processes 1 --checkpoint-folder /mnt/nvme/test_data --results-dir /mnt/nvme/results|0"
    "mlpstorage checkpointing datasize --hosts 127.0.0.1,127.0.0.1 --client-host-memory-in-gb 256 --model llama3-70b --num-processes 1 --checkpoint-folder /mnt/nvme/test_data --results-dir /mnt/nvme/results|0"
    "mlpstorage checkpointing datasize --hosts 127.0.0.1,127.0.0.1 --client-host-memory-in-gb 256 --model llama3-405b --num-processes 1 --checkpoint-folder /mnt/nvme/test_data --results-dir /mnt/nvme/results|0"
    "mlpstorage checkpointing datasize --hosts 127.0.0.1,127.0.0.1 --client-host-memory-in-gb 256 --model llama3-1t --num-processes 1 --checkpoint-folder /mnt/nvme/test_data --results-dir /mnt/nvme/results|0"

    "mlpstorage checkpointing run --hosts 127.0.0.1 --model llama3-8b --client-host-memory-in-gb 512 --num-processes 1 --checkpoint-folder /mnt/nvme/test_data --results-dir /mnt/nvme/results --num-checkpoints-read 1 --num-checkpoints-write 1 --allow-run-as-root|0"
    "mlpstorage checkpointing run --hosts 127.0.0.1 --model llama3-70b --client-host-memory-in-gb 512 --num-processes 1 --checkpoint-folder /mnt/nvme/test_data --results-dir /mnt/nvme/results --num-checkpoints-read 1 --num-checkpoints-write 1 --allow-run-as-root|0"
    "mlpstorage checkpointing run --hosts 127.0.0.1 --model llama3-405b --client-host-memory-in-gb 512 --num-processes 1 --checkpoint-folder /mnt/nvme/test_data --results-dir /mnt/nvme/results --num-checkpoints-read 1 --num-checkpoints-write 1 --allow-run-as-root|0"
    "mlpstorage checkpointing run --hosts 127.0.0.1 --model llama3-1t --client-host-memory-in-gb 512 --num-processes 1 --checkpoint-folder /mnt/nvme/test_data --results-dir /mnt/nvme/results --num-checkpoints-read 1 --num-checkpoints-write 1 --allow-run-as-root|0"

    "mlpstorage reports reportgen --results-dir ./test_results|3"
    "mlpstorage reports reportgen --results-dir /mnt/nvme/results|0"
)

# Loop through all commands and run them
for cmd_with_code in "${commands[@]}"; do
    # Split the command and expected exit code
    IFS='|' read -r cmd expected_code <<< "$cmd_with_code"

    # If no expected code was provided, default to 0
    if [ -z "$expected_code" ]; then
        expected_code=0
    fi

    run_command "$cmd" "$expected_code"
done

echo "All tests completed successfully!"
