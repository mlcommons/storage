#!/usr/bin/env bash
# =============================================================================
# sweep_unet3d_np.sh — UNet3D NP (num-accelerators) scaling sweep
#
# Sweeps NP=1, 2, 4 at the B200 computation_time (0.162 s = H100 ÷ 2).
# NP=8 is intentionally excluded — co-located s3-ultra saturates at NP≥4.
#
# Dataset : s3://mlp-unet3d/data/unet3d/  (7,200 NPZ files ≈ 984 GiB)
# Model   : unet3d, B200 accelerator, computation_time=0.162 s
# AU goal : ≥ 0.90 (90%)
#
# Results per run are written to  results/unet3d_np_sweep/<timestamp>/
# A TSV summary row is appended after each run, printed at the end.
# A Markdown results doc is auto-generated at the end of the sweep.
#
# Usage:
#   cd /home/eval/Documents/Code/mlp-storage
#   bash tests/object-store/sweep_unet3d_np.sh 2>&1 | tee sweep_unet3d_$(date +%Y%m%d_%H%M%S).log
# =============================================================================
set -euo pipefail

REPO=/home/eval/Documents/Code/mlp-storage
VENV="${REPO}/.venv"
PYTHON="${VENV}/bin/python3"

SWEEP_TS=$(date '+%Y%m%d_%H%M%S')
RESULTS_BASE="${REPO}/results/unet3d_np_sweep"
RESULTS_DIR="${RESULTS_BASE}/${SWEEP_TS}"
mkdir -p "${RESULTS_DIR}"

# ── Dataset parameters (must match the generated dataset) ────────────────────
NUM_FILES=7200
SAMPLES_PER_FILE=1
DATA_FOLDER="data/unet3d"
STORAGE_ROOT="${STORAGE_ROOT:-mlp-unet3d}"   # override: STORAGE_ROOT=mlp-flux bash sweep_unet3d_np.sh
COMP_TIME="0.162"   # B200: H100 (0.323 s) ÷ 2

# ── NP values to sweep ────────────────────────────────────────────
NP_VALUES=(1 2 4)   # NP=8 excluded — co-located s3-ultra saturates at NP≥4

# ── Load s3-ultra credentials ───────────────────────────────────────────────
# NOTE: .env.s3-ultra sets BUCKET=mlp-flux (its default).  We do NOT export
# BUCKET — instead we pass storage.storage_root on the CLI so the correct
# bucket is always used regardless of what the env file contains.
if [[ ! -f "${REPO}/.env.s3-ultra" ]]; then
    echo "ERROR: ${REPO}/.env.s3-ultra not found" >&2; exit 1
fi
set -o allexport
source "${REPO}/.env.s3-ultra"
set +o allexport
unset BUCKET   # prevent env BUCKET from leaking into mlpstorage

# ── Activate venv ─────────────────────────────────────────────────────────────
source "${VENV}/bin/activate"

# ── TSV header ────────────────────────────────────────────────────────────────
SUMMARY_TSV="${RESULTS_DIR}/sweep_unet3d_np_${SWEEP_TS}.tsv"
printf "NP\tau_pct\tsamples_per_sec\tio_mb_per_sec\twall_s\tau_met\n" \
    > "${SUMMARY_TSV}"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  UNet3D NP Scaling Sweep"
echo "  NP values : ${NP_VALUES[*]}"
echo "  Dataset   : s3://${STORAGE_ROOT}/${DATA_FOLDER}  (${NUM_FILES} files)"
echo "  ct        : ${COMP_TIME} s  (B200 = H100 ÷ 2)"
echo "  Results   : ${RESULTS_DIR}"
echo "  Started   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════"
echo ""

for NP in "${NP_VALUES[@]}"; do
    RUN_DIR="${RESULTS_DIR}/NP${NP}"
    mkdir -p "${RUN_DIR}"

    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  NP=${NP}   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "────────────────────────────────────────────────────────────────"

    t_start=$(date +%s)

    RUST_LOG=s3dlio=info \
    "${PYTHON}" -c "from mlpstorage_py.main import main; main()" \
        training run \
        --model unet3d \
        --accelerator-type b200 \
        --num-accelerators "${NP}" \
        --num-client-hosts 1 \
        --client-host-memory-in-gb 47 \
        --dlio-bin-path "${VENV}/bin" \
        --object s3 \
        --skip-validation \
        --open \
        --results-dir "${RUN_DIR}" \
        --params \
            storage.storage_root=${STORAGE_ROOT} \
            dataset.num_files_train=${NUM_FILES} \
            dataset.num_samples_per_file=${SAMPLES_PER_FILE} \
            dataset.data_folder=${DATA_FOLDER} \
            train.computation_time=${COMP_TIME} \
            storage.storage_options.decode_mode=none \
            storage.storage_options.storage_library=s3dlio

    t_end=$(date +%s)
    wall=$(( t_end - t_start ))

    # ── Parse summary.json → append TSV row ──────────────────────────────
    "${PYTHON}" - "${NP}" "${wall}" "${RUN_DIR}" \
        >> "${SUMMARY_TSV}" 2>&1 <<'PYEOF'
import json, glob, sys

np_, wall, run_dir = sys.argv[1], sys.argv[2], sys.argv[3]

files = sorted(glob.glob(f"{run_dir}/**/summary.json", recursive=True))
if not files:
    print(f"{np_}\tN/A\tN/A\tN/A\t{wall}\tN/A")
    sys.exit(0)

d    = json.load(open(files[-1]))
m    = d.get("metric", {})

au   = m.get("train_au_mean_percentage",                 None)
sps  = m.get("train_throughput_mean_samples_per_second", None)
ioMB = m.get("train_io_mean_MB_per_second",              None)
met  = m.get("train_au_meet_expectation",                "N/A")

def fmt(v, digits=2):
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "N/A"

print(f"{np_}\t{fmt(au)}\t{fmt(sps,1)}\t{fmt(ioMB,1)}\t{wall}\t{met}")
PYEOF

done

# ── Print summary table ───────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  UNet3D NP Sweep — Summary"
echo "════════════════════════════════════════════════════════════════"
column -t -s $'\t' "${SUMMARY_TSV}"
echo ""

# ── Auto-generate Markdown results doc ───────────────────────────────────────
MD_OUT="${RESULTS_DIR}/UNet3D_NP_Scaling_Results_${SWEEP_TS}.md"

"${PYTHON}" - "${SUMMARY_TSV}" "${SWEEP_TS}" "${COMP_TIME}" \
    "${NUM_FILES}" "${STORAGE_ROOT}/${DATA_FOLDER}" \
    > "${MD_OUT}" 2>&1 <<'PYEOF'
import csv, sys, datetime

tsv_path, ts, ct, nfiles, path = sys.argv[1:]

rows = []
with open(tsv_path) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        rows.append(row)

date_str = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%B %d, %Y")

def pass_fail(met):
    if met == "True" or met is True:
        return "✅ PASS"
    if met == "False" or met is False:
        return "❌ FAIL"
    return met

lines = []
lines.append(f"# UNet3D Training — NP Scaling Study")
lines.append(f"")
lines.append(f"**Sweep date**: {date_str}")
lines.append(f"")
lines.append(f"---")
lines.append(f"")
lines.append(f"## Test Environment")
lines.append(f"")
lines.append(f"| Parameter | Value |")
lines.append(f"|-----------|-------|")
lines.append(f"| Host | 24 vCPU VM (with hyperthreading), 48 GB RAM |")
lines.append(f"| Object storage | s3-ultra (`http://127.0.0.1:9000`, co-located on test host) |")
lines.append(f"| Bucket / path | `{path}` |")
lines.append(f"| Dataset | {nfiles} NPZ files × 1 sample/file (≈ 984 GiB) |")
lines.append(f"| Record length | 146,600,628 bytes avg (σ = 68,341,808) |")
lines.append(f"| Batch size | 7 |")
lines.append(f"| Read threads | 4 |")
lines.append(f"| `computation_time` | {ct} s  (B200 = H100 0.323 s ÷ 2) |")
lines.append(f"| `decode_mode` | `none` |")
lines.append(f"| Epochs | 5 |")
lines.append(f"| AU target | ≥ 90% |")
lines.append(f"| Model config | `unet3d_b200.yaml` |")
lines.append(f"| MPI invocation | `mpirun -n NP -host 127.0.0.1:NP` |")
lines.append(f"")
lines.append(f"> **⚠️ Co-located test configuration.** The s3-ultra storage server and all benchmark")
lines.append(f"> processes run on the **same** 24 vCPU / 48 GB RAM host, sharing CPU cores, memory,")
lines.append(f"> and the loopback network interface. In a real deployment storage would be a dedicated")
lines.append(f"> remote system; the CPU/memory pressure that limits scaling here would not apply.")
lines.append(f">")
lines.append(f"> **AU (Accelerator Utilization)** — fraction of wall time the simulated GPU was")
lines.append(f"> computing rather than waiting for I/O. AU ≥ 90% is the target threshold for a")
lines.append(f"> \"pass\" on unet3d.")
lines.append(f"")
lines.append(f"---")
lines.append(f"")
lines.append(f"## NP Scaling Results")
lines.append(f"")
lines.append(f"| NP | AU% | Samples/s | I/O MiB/s | Wall time (s) | AU ≥ 90%? |")
lines.append(f"|----|-----|-----------|-----------|---------------|-----------|")
for r in rows:
    pf = pass_fail(r.get("au_met", "N/A"))
    lines.append(
        f"| {r['NP']} "
        f"| {r['au_pct']} "
        f"| {r['samples_per_sec']} "
        f"| {r['io_mb_per_sec']} "
        f"| {r['wall_s']} "
        f"| {pf} |"
    )
lines.append(f"")
lines.append(f"---")
lines.append(f"")
lines.append(f"## Scaling Analysis")
lines.append(f"")
if len(rows) >= 2:
    try:
        au1 = float(rows[0]['au_pct'])
        au2 = float(rows[1]['au_pct']) if len(rows) > 1 else None
        au4 = float(rows[2]['au_pct']) if len(rows) > 2 else None
        sps1 = float(rows[0]['samples_per_sec'])
        sps2 = float(rows[1]['samples_per_sec']) if len(rows) > 1 else None
        sps4 = float(rows[2]['samples_per_sec']) if len(rows) > 2 else None

        lines.append(f"### Throughput Scaling Efficiency")
        lines.append(f"")
        lines.append(f"| Transition | Samples/s | Ideal | Efficiency |")
        lines.append(f"|------------|-----------|-------|------------|")
        if sps2 is not None:
            eff = sps2 / (sps1 * 2) * 100
            lines.append(f"| NP=1 → NP=2 | {sps1:.1f} → {sps2:.1f} | {sps1*2:.1f} | {eff:.1f}% |")
        if sps4 is not None:
            eff4 = sps4 / (sps1 * 4) * 100
            lines.append(f"| NP=1 → NP=4 | {sps1:.1f} → {sps4:.1f} | {sps1*4:.1f} | {eff4:.1f}% |")
        lines.append(f"")
    except (ValueError, IndexError):
        lines.append(f"*(throughput scaling table: parse error — check TSV)*")
        lines.append(f"")

lines.append(f"### Key Observations")
lines.append(f"")
lines.append(f"1. **NP=1 baseline** — establishes single-accelerator AU and throughput floor.")
lines.append(f"2. **NP=2 scaling** — first scaling step; throughput should nearly double if I/O-bound,")
lines.append(f"   or AU should improve if NP=1 was CPU-throttled by co-located s3-ultra.")
lines.append(f"3. **NP=4** — highest tested NP; co-located s3-ultra competes for CPU at this level.")
lines.append(f"   If AU drops or throughput plateaus relative to NP=2, storage bandwidth is saturated.")
lines.append(f"")
lines.append(f"---")
lines.append(f"")
lines.append(f"## Raw Results Location")
lines.append(f"")
lines.append(f"Full per-run output in `results/unet3d_np_sweep/{ts}/NP{{1,2,4}}/` —")
lines.append(f"each contains `summary.json`, per-epoch logs, and DLIO output.")

print('\n'.join(lines))
PYEOF

echo "════════════════════════════════════════════════════════════════"
echo "  Markdown results doc: ${MD_OUT}"
echo "  TSV summary         : ${SUMMARY_TSV}"
echo "  Finished            : $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════"
