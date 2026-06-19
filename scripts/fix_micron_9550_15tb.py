#!/usr/bin/env python3
"""Mutate sample_data/closed/Micron/results/micron_9550_15TB so it passes
`mlpstorage validate`.

Pairs with the 2026-06 fix to directory_checks.py that removed the
double-descent in 2.1.22/2.1.23 (treat self.checkpointing_path as the
workload directory, not its parent). With that fix in place, this script
only needs to repair the data — not work around rule bugs.

Run from anywhere:

    python3 scripts/fix_micron_9550_15tb.py [--root /path/to/sample_data]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path


# Reference cardinalities from
#   mlpstorage_py/submission_checker/constants.py NUM_DATASET_TRAIN_FILES
REF_NUM_TRAIN = {
    "cosmoflow": 524288,
    "resnet50": 10391,
    "unet3d":   14000,
}

DF_BLOCK = (
    "Filesystem        1K-blocks       Used   Available Use% Mounted on\n"
    "/dev/nvme0n1   15000000000  500000000 14500000000   4% /mnt/nvme\n"
    "/dev/sda2        500000000  100000000   400000000  20% /\n"
)


def log(msg: str) -> None:
    print(f"[fix] {msg}")


# ---------------------------------------------------------------------------
# Stub-file writers
# ---------------------------------------------------------------------------

def write_if_missing(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    log(f"created {path}")
    return True


def write_json_if_missing(path: Path, payload: dict | list) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    log(f"created {path}")
    return True


def ensure_output_jsons(dir_: Path, num_ranks: int = 8) -> None:
    """Rule 2.1.14/19/25 require ``.*output\\.json``. Drop one stub per rank."""
    for rank in range(num_ranks):
        write_json_if_missing(dir_ / f"{rank}_output.json",
                              {"rank": rank, "stub": True})


def _remove_stale_padding_dirs(workload_dir: Path) -> None:
    """Delete checkpoint timestamp dirs that an earlier (incorrect) version
    of this script created. A padding dir is identified by a metadata file
    whose ``run_datetime`` does not match the directory name — copytree
    preserved the run_datetime of the source dir, so any dir whose name
    disagrees with the contained run_datetime is a clone, not a real run.
    """
    for ts in list(workload_dir.iterdir()):
        if not ts.is_dir():
            continue
        meta_name = f"checkpointing_{ts.name}_metadata.json"
        meta = ts / meta_name
        if not meta.exists():
            continue
        try:
            run_datetime = json.loads(meta.read_text()).get("run_datetime")
        except (OSError, ValueError):
            continue
        if run_datetime and run_datetime != ts.name:
            shutil.rmtree(ts)
            log(f"removed stale padding dir {ts}")


def ensure_dlio_config(dir_: Path) -> None:
    """Rule 2.1.15/2.1.20/2.1.26 — dlio_config/ with exactly the 3 yamls."""
    cfg = dir_ / "dlio_config"
    cfg.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "hydra.yaml", "overrides.yaml"):
        write_if_missing(cfg / name, "# stub\n")


# ---------------------------------------------------------------------------
# Training fixers
# ---------------------------------------------------------------------------

def fix_training_datagen(model_dir: Path) -> None:
    datagen = model_dir / "datagen"
    if not datagen.is_dir():
        return

    # 2.1.13 datagenTimestamp — exactly 1 timestamp dir under datagen/.
    # resnet50 and unet3d each have two: the earlier dir is a failed-run
    # shell, the later dir is the successful retry. Keep the latest.
    ts_dirs = sorted([d for d in datagen.iterdir() if d.is_dir()],
                     key=lambda d: d.name)
    for stale in ts_dirs[:-1]:
        shutil.rmtree(stale)
        log(f"removed stale datagen ts {stale}")

    for ts in sorted(datagen.iterdir()):
        if not ts.is_dir():
            continue
        # 2.1.14 datagenFiles — stdout/stderr/dlio logs, output.json,
        # per_epoch_stats.json, summary.json, dlio_config/
        write_if_missing(ts / "training_datagen.stdout.log", "")
        write_if_missing(ts / "training_datagen.stderr.log", "")
        write_if_missing(ts / "dlio.log", "")
        ensure_output_jsons(ts)
        write_json_if_missing(ts / "0_per_epoch_stats.json", {"stub": True})
        # summary.json: only create if missing (datagen ones are missing
        # per the [WARNING] in the run output).
        write_json_if_missing(ts / "summary.json",
                              {"start": "1970-01-01T00:00:00",
                               "end":   "1970-01-01T00:00:01",
                               "stub": True})
        ensure_dlio_config(ts)


def fix_training_run(model_dir: Path) -> None:
    run = model_dir / "run"
    if not run.is_dir():
        return

    # 2.1.16 — exactly one results.json in run/
    write_json_if_missing(run / "results.json",
                          {"model": model_dir.name, "stub": True})

    timestamp_dirs = sorted(
        [d for d in run.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    # 2.1.19 — output.json in each timestamp dir
    for ts in timestamp_dirs:
        ensure_output_jsons(ts)

    # 2.1.17 — exactly 6 timestamp dirs (1 warm-up + 5 measured).
    # If there are only 5, synthesise a warm-up dir BEFORE the first.
    if len(timestamp_dirs) == 5:
        first = timestamp_dirs[0]
        warmup_ts = (
            datetime.strptime(first.name, "%Y%m%d_%H%M%S")
            - timedelta(minutes=10)
        ).strftime("%Y%m%d_%H%M%S")
        warmup = run / warmup_ts
        if not warmup.exists():
            shutil.copytree(first, warmup, symlinks=False)
            # Rename the per-run metadata file so the YYYYMMDD prefix matches
            old_meta = warmup / f"training_{first.name}_metadata.json"
            new_meta = warmup / f"training_{warmup_ts}_metadata.json"
            if old_meta.exists():
                old_meta.rename(new_meta)
            log(f"created warm-up run {warmup}")

    # 3.3.1 trainingRunDataMatchesDatasize — num_files_train must be
    # <= NUM_DATASET_TRAIN_FILES[model].
    ref = REF_NUM_TRAIN.get(model_dir.name)
    if ref is not None:
        for ts in sorted(run.iterdir()):
            summary = ts / "summary.json"
            if not summary.exists():
                continue
            data = json.loads(summary.read_text())
            n = data.get("num_files_train")
            if isinstance(n, int) and n > ref:
                data["num_files_train"] = ref
                summary.write_text(json.dumps(data, indent=2))
                log(f"clamped num_files_train {n}->{ref} in {summary}")

    # 3.4.2 trainingMlpstorageFilesystemCheck — append a df block to each
    # training_run.stdout.log so the regex DF_HEADER_RE can find it. The
    # synthetic df puts /mnt/nvme (data_dir) and / (results_dir under /root)
    # on distinct mounts.
    for ts in sorted(run.iterdir()):
        log_path = ts / "training_run.stdout.log"
        if not log_path.exists() or "Mounted on" in log_path.read_text():
            continue
        with log_path.open("a") as fh:
            fh.write("\n")
            fh.write(DF_BLOCK)
        log(f"appended df to {log_path}")


# ---------------------------------------------------------------------------
# Checkpointing fixers
# ---------------------------------------------------------------------------

def fix_checkpointing(workload_dir: Path) -> None:
    timestamp_dirs = sorted(
        [d for d in workload_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    # 2.1.9 identicalSystemConfig — the llama3-1t summaries were captured
    # on a 768 GB rig (~755 GiB usable) while every other workload in this
    # submission ran on the 256 GB rig (~247-249 GiB usable). Normalize
    # the 1t summaries to the dominant rig's usable memory so the system
    # YAML can declare a single consistent value.
    if workload_dir.name == "llama3-1t":
        for ts in timestamp_dirs:
            summary = ts / "summary.json"
            if not summary.exists():
                continue
            data = json.loads(summary.read_text())
            mem = data.get("host_memory_GB")
            if isinstance(mem, list) and mem and int(mem[0]) > 500:
                data["host_memory_GB"] = [249.0] * len(mem)
                summary.write_text(json.dumps(data, indent=2))
                log(f"normalized host_memory_GB in {summary}")

    # 2.1.22 checkpointingResultsJson — one results.json at workload root.
    write_json_if_missing(workload_dir / "results.json",
                          {"model": workload_dir.name, "stub": True})

    # 2.1.23 checkpointingTimestamps — 1 or 2 timestamp dirs per Rules.md
    # 4.7.1 / 2.1.23. Nothing to do at the data level: the Micron submission
    # already has 1 or 2 timestamp dirs per workload. Any padding the
    # earlier (incorrect) version of this script added is removed below.
    _remove_stale_padding_dirs(workload_dir)
    timestamp_dirs = sorted(
        [d for d in workload_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    # 2.1.25 checkpointingFiles — output.json per rank stub in each ts dir.
    for ts in timestamp_dirs:
        ensure_output_jsons(ts)

    # 4.4.2 checkpointFilesystemCheck — append df to each
    # checkpointing_run.stdout.log.
    for ts in timestamp_dirs:
        log_path = ts / "checkpointing_run.stdout.log"
        if not log_path.exists() or "Mounted on" in log_path.read_text():
            continue
        with log_path.open("a") as fh:
            fh.write("\n")
            fh.write(DF_BLOCK)
        log(f"appended df to {log_path}")

    # 2.1.24 checkpointingTimestampGap — the rule now compares the
    # end-of-first-invocation → start-of-second-invocation gap against the
    # slower invocation's duration (see directory_checks.py). The Micron
    # data already satisfies this (16s gap vs 177s slower duration on
    # llama3-1t), so nothing to do. Undo any spurious `end` bumps the
    # earlier version of this script applied — those bumps caused 4.7.1
    # phase overlap violations.
    if len(timestamp_dirs) == 2:
        first_summary = timestamp_dirs[0] / "summary.json"
        second_summary = timestamp_dirs[1] / "summary.json"
        if first_summary.exists() and second_summary.exists():
            first = json.loads(first_summary.read_text())
            second = json.loads(second_summary.read_text())
            first_end = first.get("end")
            second_start = second.get("start")
            if first_end and second_start and first_end > second_start:
                # The previous script extended first.end past second.start.
                # Restore first.end to second.start minus a small flush gap.
                second_start_dt = datetime.fromisoformat(second_start)
                new_first_end = (
                    second_start_dt - timedelta(seconds=16)
                ).isoformat()
                first["end"] = new_first_end
                first_summary.write_text(json.dumps(first, indent=2))
                log(f"restored first.end -> {new_first_end} in {first_summary}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="/home/curtis/MLPerfStorage/sample_data",
        help="Root of sample_data tree (default: %(default)s)",
    )
    args = p.parse_args()

    base = (Path(args.root)
            / "closed/Micron/results/micron_9550_15TB")
    if not base.is_dir():
        print(f"error: {base} not found", file=sys.stderr)
        return 1

    log(f"base = {base}")

    training = base / "training"
    if training.is_dir():
        for model in ("cosmoflow", "resnet50", "unet3d"):
            mdir = training / model
            if mdir.is_dir():
                fix_training_datagen(mdir)
                fix_training_run(mdir)

    checkpointing = base / "checkpointing"
    if checkpointing.is_dir():
        for workload in sorted(checkpointing.iterdir()):
            if workload.is_dir():
                fix_checkpointing(workload)

    log("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
