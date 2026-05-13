#!/usr/bin/env bash
# Flux read-thread × NP scaling sweep
# NP ∈ {1,2,4,8}, read_threads ∈ {1,2,4,8}  → 16 combos
# (NP=8, RT=8) is gated on (NP=4, RT=4) passing
#
# Fixed params across all runs:
#   computation_time = 0.05 s
#   coalesce_rgs     = 1
#   prefetch_workers = 2
#   dataset.num_files_train = 500
#
# Usage: bash sweep_flux.sh [--logdir DIR]
#        (default logdir: ./sweep_logs/<timestamp>)

set -uo pipefail

LOGDIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --logdir) LOGDIR="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done
[[ -z "$LOGDIR" ]] && LOGDIR="./sweep_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

SUMMARY="$LOGDIR/summary.tsv"
printf "NP\tRT\texitcode\tthroughput_GBs\tAU_pct\tduration_s\tlog\n" > "$SUMMARY"

log_and_echo() { echo "$1" | tee -a "$2"; }

run_combo() {
    local np=$1
    local rt=$2
    local logfile="$LOGDIR/np${np}_rt${rt}.log"
    local t_start t_end duration exitcode throughput au

    {
        echo ""
        echo "========================================================"
        echo "  NP=${np}  read_threads=${rt}  started: $(date)"
        echo "========================================================"
    } | tee "$logfile"

    t_start=$(date +%s)

    uv run mlpstorage training run \
        --model flux \
        --num-accelerators "$np" \
        --accelerator-type b200 \
        --client-host-memory-in-gb 47 \
        --object s3 \
        --skip-validation \
        --open \
        --params \
            dataset.num_files_train=500 \
            "train.computation_time=0.05" \
            "storage.storage_options.coalesce_rgs=1" \
            "storage.storage_options.prefetch_workers=2" \
            "reader.read_threads=${rt}" \
        2>&1 | tee -a "$logfile"
    exitcode=${PIPESTATUS[0]}

    t_end=$(date +%s)
    duration=$(( t_end - t_start ))

    # Extract throughput: match patterns like "1.923 GB/s" or "1923.4 MB/s"
    throughput=$(grep -oP '\d+\.\d+\s*GB/s' "$logfile" 2>/dev/null \
                 | tail -1 | grep -oP '\d+\.\d+' || true)
    if [[ -z "$throughput" ]]; then
        # try MB/s and convert
        local mbs
        mbs=$(grep -oP '\d+\.\d+\s*MB/s' "$logfile" 2>/dev/null \
              | tail -1 | grep -oP '\d+\.\d+' || true)
        [[ -n "$mbs" ]] && throughput=$(awk "BEGIN{printf \"%.3f\", $mbs/1024}") || throughput="N/A"
    fi

    # Extract accelerator utilisation: "AU=96.8" / "accelerator_util.*96.8" / "util.*96.8 %"
    au=$(grep -iP 'accelerator.util|AU\s*[=:]\s*' "$logfile" 2>/dev/null \
         | grep -oP '\d+\.\d+' | tail -1 || true)
    [[ -z "$au" ]] && au="N/A"

    local status="OK"
    [[ $exitcode -ne 0 ]] && status="FAIL"

    printf "%-4s\t%-4s\t%s(%s)\t%-14s\t%-8s\t%-12s\t%s\n" \
        "$np" "$rt" "$exitcode" "$status" \
        "${throughput}" "${au}" "${duration}" "${logfile}" >> "$SUMMARY"

    {
        echo ""
        echo "  Finished: $(date)  exit=${exitcode}  duration=${duration}s"
        echo "  throughput=${throughput} GB/s  AU=${au}%"
        echo "========================================================"
    } | tee -a "$logfile"

    return $exitcode
}

# ── Print plan ────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Flux scaling sweep  —  $(date)"
echo "  LOGDIR: $LOGDIR"
echo "  Fixed: computation_time=0.05  coalesce_rgs=1  prefetch_workers=2"
echo "  NP ∈ {1,2,4,8}  ×  read_threads ∈ {1,2,4,8}"
echo "  (NP=8, RT=8) gated on (NP=4, RT=4) passing"
echo "========================================================"
echo ""

NPS=(1 2 4 8)
RTS=(1 2 4 8)
np4_rt4_ok=false
total=0
passed=0

for np in "${NPS[@]}"; do
    for rt in "${RTS[@]}"; do
        # Gate: skip (8,8) here — handled below
        [[ $np -eq 8 && $rt -eq 8 ]] && continue

        total=$(( total + 1 ))
        echo ""
        echo "─── Combo ${total}/15 : NP=${np}  RT=${rt} ───"

        if run_combo "$np" "$rt"; then
            passed=$(( passed + 1 ))
            [[ $np -eq 4 && $rt -eq 4 ]] && np4_rt4_ok=true
        else
            echo "  *** NP=${np} RT=${rt} FAILED — continuing sweep ***"
        fi
    done
done

# ── Gate: (NP=8, RT=8) ────────────────────────────────────────────────────────
echo ""
echo "========================================================"
if $np4_rt4_ok; then
    echo "  GATE: NP=4 RT=4 PASSED → running NP=8 RT=8"
    echo "========================================================"
    total=$(( total + 1 ))
    if run_combo 8 8; then
        passed=$(( passed + 1 ))
    fi
else
    echo "  GATE: NP=4 RT=4 did NOT pass → SKIPPING NP=8 RT=8"
    echo "========================================================"
    printf "%-4s\t%-4s\t%s\t%-14s\t%-8s\t%-12s\t%s\n" \
        "8" "8" "SKIPPED" "N/A" "N/A" "N/A" "gated_on_4x4" >> "$SUMMARY"
fi

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  SWEEP COMPLETE  —  $(date)"
echo "  Passed: ${passed}/${total}"
echo "  Summary: $SUMMARY"
echo "========================================================"
echo ""
cat "$SUMMARY"
