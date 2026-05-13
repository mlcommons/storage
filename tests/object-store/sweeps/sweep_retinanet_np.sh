#!/usr/bin/env bash
# =============================================================================
# sweep_retinanet_np.sh — RetinaNet NP (num-accelerators) scaling sweep
#
# Sweeps NP=1, 2, 4 using the B200 computation_time (0.04755 s).
# NP=8 is intentionally excluded — co-located s3-ultra saturates at NP≥4.
#
# Dataset : s3://mlp-retinanet/data/retinanet/  (50,000 JPEG files ≈ 15.4 GiB)
# Format  : JPEG, 1 sample/file, ~323 KiB/file
# Model   : retinanet, B200 accelerator
# AU goal : ≥ 0.85 (85%)
#
# Key difference from UNet3D:
#   RetinaNet uses many small objects (315 KiB × 50,000) vs few large objects
#   (140 MiB × 7,200 for UNet3D). The iterable DataLoader path
#   (TorchIterableDatasetSimple) issues 64 × NP concurrent GETs, which is
#   essential for saturating the storage backend with small objects.
#
# Results per run are written to  results/retinanet_np_sweep/<timestamp>/
# A TSV summary row is appended after each run, printed at the end.
# A Markdown results doc is auto-generated at the end of the sweep.
#
# Usage:
#   cd /home/eval/Documents/Code/mlp-storage
#   bash tests/object-store/sweep_retinanet_np.sh 2>&1 | tee sweep_retinanet_$(date +%Y%m%d_%H%M%S).log
#
#   # Full MLPerf dataset (must have been generated with NUM_FILES=1170301):
#   NUM_FILES=1170301 bash tests/object-store/sweep_retinanet_np.sh 2>&1 | tee ...
# =============================================================================
set -euo pipefail

REPO=/home/eval/Documents/Code/mlp-storage
VENV="${REPO}/.venv"
PYTHON="${VENV}/bin/python3"

SWEEP_TS=$(date '+%Y%m%d_%H%M%S')
RESULTS_BASE="${REPO}/results/retinanet_np_sweep"
RESULTS_DIR="${RESULTS_BASE}/${SWEEP_TS}"
mkdir -p "${RESULTS_DIR}"

# ── Dataset parameters (must match the generated dataset) ────────────────────
NUM_FILES="${NUM_FILES:-50000}"          # full MLPerf: 1170301
SAMPLES_PER_FILE=1
DATA_FOLDER="data/retinanet"
STORAGE_ROOT="${STORAGE_ROOT:-mlp-retinanet}"
COMP_TIME="0.04755"   # B200 retinanet computation time

# ── NP values to sweep ────────────────────────────────────────────────────────
NP_VALUES=(1 2 4)   # NP=8 excluded — co-located s3-ultra saturates at NP≥4

# ── Load s3-ultra credentials ─────────────────────────────────────────────────
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
SUMMARY_TSV="${RESULTS_DIR}/sweep_retinanet_np_${SWEEP_TS}.tsv"
printf "NP\tau_pct\tsamples_per_sec\tio_mb_per_sec\twall_s\tau_met\n" \
    > "${SUMMARY_TSV}"

# ── Size estimate ─────────────────────────────────────────────────────────────
TOTAL_MIB=$(( NUM_FILES * 322957 / 1024 / 1024 ))

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  RetinaNet NP Scaling Sweep"
echo "  NP values : ${NP_VALUES[*]}"
echo "  Dataset   : s3://${STORAGE_ROOT}/${DATA_FOLDER}  (${NUM_FILES} files ≈ ${TOTAL_MIB} MiB)"
echo "  Format    : JPEG, 1 sample/file, ~323 KiB/file"
echo "  ct        : ${COMP_TIME} s  (B200)"
echo "  DataLoader: TorchIterableDatasetSimple (64 in-flight GETs/worker)"
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
        --model retinanet \
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
            storage.storage_options.storage_library=s3dlio

    t_end=$(date +%s)
    wall=$(( t_end - t_start ))

    # ── Parse summary.json → append TSV row ──────────────────────────────────
    "${PYTHON}" - "${NP}" "${wall}" "${RUN_DIR}" \
        >> "${SUMMARY_TSV}" 2>&1 <<'PYEOF'
import json, glob, sys

np_, wall, run_dir = sys.argv[1], sys.argv[2], sys.argv[3]

files = sorted(glob.glob(f"{run_dir}/**/summary.json", recursive=True))
if not files:
    print(f"{np_}\tN/A\tN/A\tN/A\t{wall}\tN/A")
    sys.exit(0)

d   = json.load(open(files[-1]))
m   = d.get("metric", {})

au  = m.get("train_au_mean_percentage",                 None)
sps = m.get("train_throughput_mean_samples_per_second", None)
ioMB = m.get("train_io_mean_MB_per_second",             None)
met = m.get("train_au_meet_expectation",                "N/A")

def fmt(v, digits=2):
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "N/A"

print(f"{np_}\t{fmt(au)}\t{fmt(sps)}\t{fmt(ioMB)}\t{wall}\t{met}")
PYEOF

    echo "  NP=${NP} done  (wall=${wall}s)"
    echo "  Results: ${RUN_DIR}"
done

# ── Print TSV summary ─────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Sweep complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  TSV summary:"
cat "${SUMMARY_TSV}"
echo "════════════════════════════════════════════════════════════════"

# ── Auto-generate Markdown results doc ───────────────────────────────────────
MD_OUT="${REPO}/docs/RetinaNet_NP_Scaling_Results.md"

"${PYTHON}" - "${SWEEP_TS}" "${COMP_TIME}" "${SUMMARY_TSV}" \
             "${STORAGE_ROOT}/${DATA_FOLDER}" "${NUM_FILES}" \
             "${MD_OUT}" <<'PYEOF'
import sys, csv, datetime

ts, ct, tsv_path, path, nfiles_str, md_out = sys.argv[1:]
nfiles = int(nfiles_str)
record_bytes = 322957
total_mib = nfiles * record_bytes // (1024 * 1024)

rows = []
with open(tsv_path) as fh:
    reader = csv.DictReader(fh, delimiter='\t')
    for row in reader:
        rows.append(row)

date_str = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")

def pass_fail(v):
    if v in ("True", True):  return "✅ PASS"
    if v in ("False", False): return "❌ FAIL"
    return "—"

lines = []
lines.append(f"# RetinaNet NP Scaling Results")
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
lines.append(f"| Dataset | {nfiles:,} JPEG files × 1 sample/file (≈ {total_mib:,} MiB) |")
lines.append(f"| Record length | 322,957 bytes (~315 KiB / file) |")
lines.append(f"| Batch size | 24 |")
lines.append(f"| Read threads | 8 |")
lines.append(f"| `computation_time` | {ct} s  (B200) |")
lines.append(f"| DataLoader | `TorchIterableDatasetSimple` (64 in-flight GETs/worker) |")
lines.append(f"| Epochs | 8 |")
lines.append(f"| AU target | ≥ 85% |")
lines.append(f"| Model config | `retinanet_b200.yaml` |")
lines.append(f"| MPI invocation | `mpirun -n NP -host 127.0.0.1:NP` |")
lines.append(f"")
lines.append(f"> **⚠️ Co-located test configuration.** The s3-ultra storage server and all benchmark")
lines.append(f"> processes run on the **same** 24 vCPU / 48 GB RAM host, sharing CPU cores, memory,")
lines.append(f"> and the loopback network interface. In a real deployment storage would be a dedicated")
lines.append(f"> remote system; the CPU/memory pressure that limits scaling here would not apply.")
lines.append(f">")
lines.append(f"> **AU (Accelerator Utilization)** — fraction of wall time the simulated GPU was")
lines.append(f"> computing rather than waiting for I/O. AU ≥ 85% is the target threshold for")
lines.append(f"> retinanet.")
lines.append(f"")
lines.append(f"---")
lines.append(f"")
lines.append(f"## NP Scaling Results")
lines.append(f"")
lines.append(f"| NP | AU% | Samples/s | I/O MiB/s | Wall time (s) | AU ≥ 85%? |")
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
        au1  = float(rows[0]['au_pct'])
        au2  = float(rows[1]['au_pct']) if len(rows) > 1 else None
        au4  = float(rows[2]['au_pct']) if len(rows) > 2 else None
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
lines.append(f"   RetinaNet I/O is dominated by many small GETs (~315 KiB × files-per-worker);")
lines.append(f"   the `TorchIterableDatasetSimple` path with 64 in-flight GETs/worker is")
lines.append(f"   essential to keep the storage backend saturated.")
lines.append(f"2. **NP=2 scaling** — first scaling step; both AU and throughput should improve")
lines.append(f"   if the NP=1 run was I/O-bound (AU < 85%).")
lines.append(f"3. **NP=4** — highest tested NP; co-located s3-ultra competes for CPU at this")
lines.append(f"   level. If AU plateaus or degrades, the bottleneck has shifted from I/O to")
lines.append(f"   SHA-256 signing CPU on this Cascade Lake host (no SHA-NI instruction).")
lines.append(f"")
lines.append(f"---")
lines.append(f"")
lines.append(f"## Raw Results Location")
lines.append(f"")
lines.append(f"Full per-run output in `results/retinanet_np_sweep/{ts}/NP{{1,2,4}}/` —")
lines.append(f"each contains `summary.json`, per-epoch logs, and DLIO output.")

with open(md_out, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')

print(f"Markdown written to: {md_out}")
PYEOF

echo "════════════════════════════════════════════════════════════════"
echo "  Markdown results doc: ${MD_OUT}"
echo "  TSV summary         : ${SUMMARY_TSV}"
echo "  Finished            : $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════"
