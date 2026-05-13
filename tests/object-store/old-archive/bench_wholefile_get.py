#!/usr/bin/env python3
"""
Simulate the proposed batch-iterator architecture for DLRM Parquet files.

Current architecture:  16 byte-range GETs per file, 1,024 GETs/epoch,
                       64,000,000 read_index() calls/epoch  → Python-bound
Proposed architecture: 1 whole-object GET per file, 64 GETs/epoch,
                       ~64 iterator.__next__() calls/epoch  → I/O-bound

This script issues real full-file GETs against the S3 endpoint (no Parquet
decode) to measure the I/O ceiling of the proposed design.

  --pipeline  concurrent GETs per NP process (default: 2)
  --np        number of NP processes to simulate (default: 1)
              total outstanding GETs = np × pipeline
              e.g. --np 4 --pipeline 2  →  8 concurrent GETs in flight

Usage:
    python3 bench_wholefile_get.py [--np N] [--pipeline N] [--files N] [--epochs N]
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Credentials / endpoint from .env
# ---------------------------------------------------------------------------
_ENV = os.path.join(os.path.dirname(__file__), "../../.env")
if os.path.exists(_ENV):
    with open(_ENV) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

import s3dlio  # noqa: E402  (needs env vars before import)

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
BUCKET   = "mlp-flux"
PREFIX   = "data/dlrm/train/train"
N_FILES  = 64

def file_uris(n: int = N_FILES) -> list[str]:
    return [f"s3://{BUCKET}/{PREFIX}/img_{i:02d}_of_64.parquet" for i in range(n)]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def fetch_file(uri: str) -> tuple[str, int, float]:
    """GET the entire object and discard bytes.  Returns (uri, nbytes, elapsed_s)."""
    t0 = time.monotonic()
    data = s3dlio.get(uri)          # releases GIL internally → concurrent with other threads
    elapsed = time.monotonic() - t0
    nbytes = len(data)
    del data                        # release immediately
    return uri, nbytes, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--np",       type=int, default=1,
                    help="Simulated NP (number of processes); multiplies pipeline (default: 1)")
    ap.add_argument("--pipeline", type=int, default=2,
                    help="Concurrent GETs per NP process (default: 2)")
    ap.add_argument("--files",    type=int, default=N_FILES,
                    help=f"Number of files to fetch per epoch (default: {N_FILES})")
    ap.add_argument("--epochs",   type=int, default=2,
                    help="Number of epochs to run (default: 2)")
    args = ap.parse_args()

    total_pipeline = args.np * args.pipeline
    uris = file_uris(args.files)
    total_dataset_bytes: int | None = None  # set after first epoch

    print(f"Proposed batch-iterator benchmark")
    print(f"  files={args.files}  np={args.np}  pipeline={args.pipeline}  "
          f"total_outstanding={total_pipeline}  epochs={args.epochs}")
    print(f"  endpoint: {os.environ.get('AWS_ENDPOINT_URL_S3', '(default)')}")
    print(f"  target: ≥400 MB/s")
    print()

    epoch_results: list[tuple[int, float, float]] = []  # (epoch, total_gb, mbps)

    for ep in range(1, args.epochs + 1):
        print(f"═══ Epoch {ep} ════════════════════════════════════════════════════")
        print(f"  {'File':<35} {'MiB':>8}  {'s':>7}  {'MB/s':>9}")
        print(f"  {'-'*35} {'-'*8}  {'-'*7}  {'-'*9}")

        epoch_bytes = 0
        file_results: list[tuple[str, int, float]] = []

        t_epoch = time.monotonic()
        with ThreadPoolExecutor(max_workers=total_pipeline) as ex:
            futs = {ex.submit(fetch_file, u): u for u in uris}
            for fut in as_completed(futs):
                uri, nbytes, elapsed = fut.result()
                mbps = nbytes / elapsed / 1e6
                epoch_bytes += nbytes
                file_results.append((uri, nbytes, elapsed, mbps))
                print(f"  {os.path.basename(uri):<35} {nbytes/1024**2:>8.1f}  {elapsed:>7.3f}  {mbps:>9.1f}")
        t_epoch = time.monotonic() - t_epoch

        if total_dataset_bytes is None:
            total_dataset_bytes = epoch_bytes

        epoch_mbps = epoch_bytes / t_epoch / 1e6
        epoch_results.append((ep, epoch_bytes / 1024**3, epoch_mbps))

        print(f"  {'-'*35} {'-'*8}  {'-'*7}  {'-'*9}")
        print(f"  {'EPOCH TOTAL':<35} {epoch_bytes/1024**3:>7.2f}G  {t_epoch:>7.3f}  {epoch_mbps:>9.1f}")
        print()

    # Summary
    print("═══ Summary ════════════════════════════════════════════════════════")
    print(f"  {'Epoch':<8}  {'Data GiB':>10}  {'Throughput MB/s':>16}  {'vs 400 MB/s':>12}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*16}  {'-'*12}")
    for ep, gb, mbps in epoch_results:
        vs = f"+{mbps-400:.0f}" if mbps >= 400 else f"{mbps-400:.0f}"
        label = "PASS" if mbps >= 400 else "FAIL"
        print(f"  {ep:<8}  {gb:>10.2f}  {mbps:>16.1f}  {vs:>8} ({label})")

    if len(epoch_results) >= 2:
        ep2_mbps = epoch_results[1][2]
        print()
        print(f"  Epoch 2 (OS/server cache): {ep2_mbps:.1f} MB/s  "
              f"{'≥ 400 MB/s ✓' if ep2_mbps >= 400 else '< 400 MB/s ✗'}")


if __name__ == "__main__":
    main()
