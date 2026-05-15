#!/usr/bin/env bash
# =============================================================================
# show_results.sh — Print a summary table of all completed DLRM runs
# =============================================================================
#
# Usage: ./show_results.sh
#
# =============================================================================

VENV=/home/eval/Documents/Code/mlp-storage/.venv

"${VENV}/bin/python3" - <<'PYEOF'
import json, glob, os

results_dir = "/home/eval/Documents/Code/mlp-storage/results/dlrm"
files = sorted(glob.glob(f"{results_dir}/**/training_*_metadata.json", recursive=True))

if not files:
    print("No results found in", results_dir)
    exit(0)

print(f"{'Run':20s}  {'NP':>3}  {'Runtime(s)':>11}  {'MB/s':>8}  {'AU%':>6}  {'DLIO IO MB/s':>12}  {'Summary':8}")
print("-" * 85)

for f in files:
    run_id = os.path.dirname(f).split("/")[-1]
    d = json.load(open(f))
    np_ = d.get("num_processes", "?")
    runtime = d.get("runtime")

    if runtime:
        total_mb = 64 * 970
        mbps = total_mb / runtime
        rt_str = f"{runtime:.1f}"
        mbps_str = f"{mbps:.0f}"
    else:
        rt_str = "?"
        mbps_str = "?"

    # DLIO summary
    summary_path = os.path.join(os.path.dirname(f), "summary.json")
    if os.path.exists(summary_path):
        s = json.load(open(summary_path))
        m = s.get("metric", {})
        au_str = f"{m.get('train_au_mean_percentage', '?'):.1f}" if isinstance(m.get('train_au_mean_percentage'), float) else "?"
        io_str = f"{m.get('train_io_mean_MB_per_second', '?'):.0f}" if isinstance(m.get('train_io_mean_MB_per_second'), float) else "?"
        ok = m.get("train_au_meet_expectation", "?")
    else:
        au_str = "-"
        io_str = "-"
        ok = "no summary"

    print(f"{run_id:20s}  {str(np_):>3}  {rt_str:>11}  {mbps_str:>8}  {au_str:>6}  {io_str:>12}  {ok}")

PYEOF
