#!/bin/bash

#KVCache Discovery Loop - ShareGPT Dataset Version

cache_dir="/mnt/nvme"
DEVICE="nvme3n1"
cpumemory=(0 4 8 16 32 64)
maxallocs=(0 2 4 8 16 32 64)
genmode=("realistic" "none")
model=(llama2-7b mistral-7b llama3.1-8b llama3.1-70b-instruct tiny-1b)
output_dir="results_sharegpt"
zip_file="sharegpt_experiment_results_$(date +%Y%m%d_%H%M%S).zip"

# ShareGPT dataset configuration
SHAREGPT_PATH="${SHAREGPT_PATH:-/path/to/ShareGPT_V3_unfiltered_cleaned_split.json}"
MAX_CONVERSATIONS=500

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Check if ShareGPT dataset exists
if [[ ! -f "$SHAREGPT_PATH" ]]; then
    echo "ERROR: ShareGPT dataset not found at $SHAREGPT_PATH"
    echo "Please set SHAREGPT_PATH environment variable or update the script"
    echo "Example: export SHAREGPT_PATH=/mnt/data/ShareGPT_V3_unfiltered_cleaned_split.json"
    exit 1
fi

echo "Using ShareGPT dataset: $SHAREGPT_PATH"
echo "Max conversations: $MAX_CONVERSATIONS"
echo "Output directory: $output_dir"
echo "=================================================="

for m in "${model[@]}"; do
    # Set numusers based on model (controls worker threads for processing)
    if [[ "$m" == "llama3.1-70b-instruct" ]]; then
        numusers=(10 20 30 40 50 60 70)
    elif [[ "$m" == "tiny-1b" ]]; then
        numusers=(200 300 400 500)
    else
        numusers=(50 100 150 200)
    fi

    for dram in "${cpumemory[@]}"; do
        for qd in "${maxallocs[@]}"; do
            for gen in "${genmode[@]}"; do
                for nu in "${numusers[@]}"; do
                    # Define output filenames
                    base_name="${m}_cpu${dram}GB_qd${qd}_gen${gen}_users${nu}_sharegpt"
                    iostat_file="$output_dir/iostat_${DEVICE}_${base_name}.txt"
                    json_file="$output_dir/mlperf_v3_storage_${base_name}.json"

                    echo "Running: model=$m, cpu_mem=$dram, qd=$qd, gen=$gen, users=$nu (ShareGPT)"

                    # Start iostat in background
                    iostat -mx 1 /dev/${DEVICE} > "$iostat_file" &
                    iostat_pid=$!

                    # Run the Python script with ShareGPT dataset
                    python3 kv-cache.py \
                        --model "${m}" \
                        --num-users "${nu}" \
                        --duration 120 \
                        --gpu-mem-gb 0 \
                        --cpu-mem-gb "${dram}" \
                        --max-concurrent-allocs "${qd}" \
                        --generation-mode "${gen}" \
                        --performance-profile throughput \
                        --cache-dir "${cache_dir}" \
                        --seed 42 \
                        --dataset-path "${SHAREGPT_PATH}" \
                        --max-conversations "${MAX_CONVERSATIONS}" \
                        --output "$json_file"

                    # Kill iostat after Python script completes
                    kill $iostat_pid 2>/dev/null
                    wait $iostat_pid 2>/dev/null
                done
            done
        done
    done
done

# Run xlsx conversion
echo "Converting JSON results to Excel..."
python3 json_to_xlsx.py --input-dir "$output_dir" --output "$output_dir/mlperf_storage_summary_sharegpt.xlsx"

# Zip all results
cd "$output_dir" || exit
zip -r "../$zip_file" .
echo "Results zipped to ../$zip_file"
