#!/bin/bash
# Performance benchmark: Compare s3torchconnector, minio, s3dlio for 100GB workload

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
CONFIG_PATH="$PROJECT_ROOT/tests/configs/perf_test_100gb.yaml"

# Test parameters
TOTAL_SIZE_GB=100
NUM_FILES=100
SAMPLES_PER_FILE=1000
RECORD_SIZE_MB=1

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DLIO Performance Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Target size: ${YELLOW}${TOTAL_SIZE_GB} GB${NC}"
echo -e "Files: ${NUM_FILES}, Samples/file: ${SAMPLES_PER_FILE}, Record size: ${RECORD_SIZE_MB}MB"
echo -e "Config: $(basename $CONFIG_PATH)"
echo ""

# S3 credentials from environment variables
# Prefer generic (ACCESS_KEY_ID) over AWS_* if both exist
if [ -n "$ACCESS_KEY_ID" ]; then
    export AWS_ACCESS_KEY_ID="$ACCESS_KEY_ID"
    echo -e "${YELLOW}Using ACCESS_KEY_ID from environment${NC}"
elif [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo -e "${RED}Error: Neither ACCESS_KEY_ID nor AWS_ACCESS_KEY_ID is set${NC}"
    exit 1
else
    echo -e "${YELLOW}Using AWS_ACCESS_KEY_ID from environment${NC}"
fi

if [ -n "$SECRET_ACCESS_KEY" ]; then
    export AWS_SECRET_ACCESS_KEY="$SECRET_ACCESS_KEY"
    echo -e "${YELLOW}Using SECRET_ACCESS_KEY from environment${NC}"
elif [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo -e "${RED}Error: Neither SECRET_ACCESS_KEY nor AWS_SECRET_ACCESS_KEY is set${NC}"
    exit 1
else
    echo -e "${YELLOW}Using AWS_SECRET_ACCESS_KEY from environment${NC}"
fi

if [ -n "$ENDPOINT_URL" ]; then
    export AWS_ENDPOINT_URL="$ENDPOINT_URL"
    echo -e "${YELLOW}Using ENDPOINT_URL from environment${NC}"
elif [ -z "$AWS_ENDPOINT_URL" ]; then
    echo -e "${RED}Error: Neither ENDPOINT_URL nor AWS_ENDPOINT_URL is set${NC}"
    exit 1
else
    echo -e "${YELLOW}Using AWS_ENDPOINT_URL from environment${NC}"
fi

echo ""

# Activate virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    exit 1
fi

source "$VENV_PATH/bin/activate"

# Function to run test for a specific library
run_test() {
    local library=$1
    local bucket=$2
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Testing: $library${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Bucket: ${bucket}"
    echo -e "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Update config with library and bucket
    local temp_config="/tmp/perf_test_${library}.yaml"
    sed "s/storage_library: .*/storage_library: $library/" "$CONFIG_PATH" | \
    sed "s|storage_root: .*|storage_root: s3://$bucket|" > "$temp_config"
    
    # Create bucket if it doesn't exist (ignore errors if it exists)
    python3 - <<EOF 2>/dev/null || true
import boto3
from botocore.client import Config
import os
s3 = boto3.client('s3',
    endpoint_url=os.environ['AWS_ENDPOINT_URL'],
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    config=Config(signature_version='s3v4'))
try:
    s3.create_bucket(Bucket='$bucket')
    print("Created bucket: $bucket")
except:
    pass
EOF
    
    echo -e "\n${YELLOW}--- WRITE Test (Data Generation) ---${NC}"
    local write_start=$(date +%s)
    
    if ! dlio_benchmark run --config-name perf_test_100gb --config-path /tmp 2>&1 | tee "/tmp/perf_${library}_write.log"; then
        echo -e "${RED}ERROR: Write test failed for $library${NC}"
        echo "$library,FAILED,0,FAILED,0,0" >> /tmp/perf_results.csv
        return 1
    fi
    
    local write_end=$(date +%s)
    local write_time=$((write_end - write_start))
    
    # Verify data was written using s3-cli
    echo -e "\n${YELLOW}Verifying data in bucket $bucket...${NC}"
    local files_in_bucket=$(s3-cli ls -cr s3://$bucket/ 2>&1 | grep -oP "Total: \K\d+" || echo "0")
    echo -e "Files in bucket: ${GREEN}$files_in_bucket${NC}"
    
    if [ "$files_in_bucket" -eq 0 ]; then
        echo -e "${RED}WARNING: No files found in bucket!${NC}"
    fi
    
    # Extract file count from output
    local files_created=$(grep -oP "Generated \K\d+" "/tmp/perf_${library}_write.log" | tail -1 || echo "$files_in_bucket")
    
    echo -e "\n${YELLOW}--- READ Test (Training Epoch) ---${NC}"
    
    # Now run a read test - update config for training mode
    sed "s/generate_data: True/generate_data: False/" "$temp_config" | \
    sed "s/train: False/train: True/" > "${temp_config}.read"
    
    local read_start=$(date +%s)
    
    if ! dlio_benchmark run --config-name "$(basename ${temp_config}.read .yaml)" --config-path /tmp 2>&1 | tee "/tmp/perf_${library}_read.log"; then
        echo -e "${RED}ERROR: Read test failed for $library${NC}"
        echo "$library,$write_time,$write_throughput,FAILED,0,$files_in_bucket" >> /tmp/perf_results.csv
        return 1
    fi
    
    local read_end=$(date +%s)
    local read_time=$((read_end - read_start))
    
    # Calculate throughput
    local write_throughput=$(awk "BEGIN {printf \"%.2f\", $TOTAL_SIZE_GB / $write_time}")
    local read_throughput=$(awk "BEGIN {printf \"%.2f\", $TOTAL_SIZE_GB / $read_time}")
    
    echo -e "\n${GREEN}Results for $library:${NC}"
    echo -e "  Files in bucket: $files_in_bucket"
    echo -e "  Files created: $files_created"
    echo -e "  Write time: ${write_time}s (${write_throughput} GB/s)"
    echo -e "  Read time:  ${read_time}s (${read_throughput} GB/s)"
    echo -e "  End time: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Save results
    echo "$library,$write_time,$write_throughput,$read_time,$read_throughput,$files_in_bucket" >> /tmp/perf_results.csv
    
    # Cleanup temp config
    rm -f "$temp_config" "${temp_config}.read"
}

# Check for s3-cli
if ! command -v s3-cli &> /dev/null; then
    echo -e "${RED}ERROR: s3-cli not found. Please install it first.${NC}"
    echo -e "Run: cd /path/to/s3dlio && cargo install --path ."
    exit 1
fi

echo -e "${BLUE}Using s3-cli version: $(s3-cli -V)${NC}"
echo ""

# Initialize results file
echo "Library,Write_Time_s,Write_Throughput_GBps,Read_Time_s,Read_Throughput_GBps,Files_In_Bucket" > /tmp/perf_results.csv

# Test each library
echo -e "\n${BLUE}Starting performance tests...${NC}\n"

run_test "s3torchconnector" "perf-s3torch"
echo -e "\n${YELLOW}Waiting 5 seconds before next test...${NC}"
sleep 5

# Final verification - list all buckets
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Final Bucket Verification${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
for bucket in "perf-s3torch" "perf-minio" "perf-s3dlio"; do
    echo -e "${YELLOW}Checking s3://$bucket:${NC}"
    s3-cli ls -cr s3://$bucket/ 2>&1 || echo "  (bucket may not exist or is empty)"
    echo ""
done

# Display summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Performance Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
column -t -s, /tmp/perf_results.csv

# Find winner (excluding FAILED entries)
echo -e "\n${GREEN}Winners:${NC}"
fastest_write=$(tail -n +2 /tmp/perf_results.csv | grep -v FAILED | sort -t, -k3 -rn | head -1 | cut -d, -f1)
fastest_read=$(tail -n +2 /tmp/perf_results.csv | grep -v FAILED | sort -t, -k5 -rn | head -1 | cut -d, -f1)
if [ -n "$fastest_write" ]; then
    echo -e "  Fastest WRITE: ${GREEN}$fastest_write${NC}"
else
    echo -e "  Fastest WRITE: ${RED}All tests failed${NC}"
fi
if [ -n "$fastest_read" ]; then
    echo -e "  Fastest READ:  ${GREEN}$fastest_read${NC}"
else
    echo -e "  Fastest READ:  ${RED}All tests failed${NC}"
fi

# Find winner
echo -e "\n${GREEN}Winners:${NC}"
fastest_write=$(tail -n +2 /tmp/perf_results.csv | sort -t, -k3 -rn | head -1 | cut -d, -f1)
fastest_read=$(tail -n +2 /tmp/perf_results.csv | sort -t, -k5 -rn | head -1 | cut -d, -f1)
echo -e "  Fastest WRITE: ${GREEN}$fastest_write${NC}"
echo -e "  Fastest READ:  ${GREEN}$fastest_read${NC}"

echo -e "\n${BLUE}Full results saved to: /tmp/perf_results.csv${NC}"
echo -e "${BLUE}Logs saved to: /tmp/perf_*_*.log${NC}"
