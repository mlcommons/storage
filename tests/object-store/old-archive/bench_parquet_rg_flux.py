#!/usr/bin/env python3
"""
Flux Parquet Row-Group Benchmark — Mode 1 (s3dlio raw + discard)

Reads Parquet row groups from Flux training files using
s3dlio.parquet_get_rg(decode="raw") and discards bytes immediately.
This benchmarks pure storage throughput with zero Python decode overhead.

Flux dataset characteristics (MLPerf Storage):
  Files:   4296 train files
  Samples: 288 per file  (~594 MiB each uncompressed, no compression)
  Columns: t5_encodings (524328×f32), clip_encodings (409×f32),
           mean (8232×f32), logvar (8232×f32), timestamp (7×f32)
  Record:  2,164,832 bytes per sample
  Full dataset: ~2.4 TiB total

Row-group granularity:
  --rg-per-file controls how many row groups each file is split into.
  Default is 6 (matching batch_size=48 from flux_b200.yaml: 288/48 = 6).
  Each row group = 48 samples × 2,164,832 bytes ≈ 99 MiB.

Mode 1 = s3dlio.parquet_get_rg(decode="raw")
  Returns compressed column-chunk bytes directly from the Parquet file.
  NOT a standalone .parquet file — no magic bytes or footer.
  The bytes are discarded immediately; only storage throughput is measured.

Usage:
    python3 bench_parquet_rg_flux.py [OPTIONS]

File naming matches gen_flux_parquet.py: train_{i:04d}.parquet

Options:
    --prefix URI_PREFIX      Base URI prefix for flux files
                             (default: file:///mnt/test/data/flux/train)
    --files N                Number of files to benchmark per epoch (default: 8)
    --rg-per-file N          Row groups per file (default: 6 = 288 samples / 48)
    --np N                   Simulated MPI ranks — multiplies pipeline (default: 1)
    --pipeline N             Concurrent parquet_get_rg calls per rank (default: 4)
    --epochs N               Number of epochs to run (default: 2)
    --footer-cap BYTES        Footer prefetch size in bytes (default: 4194304 = 4 MiB)
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Load .env credentials / endpoint (walk up from script location)
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.join(_here, "../../.env"),
    os.path.join(_here, "../.env"),
    os.path.join(_here, ".env"),
]:
    if os.path.exists(_candidate):
        with open(_candidate) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())
        break

import s3dlio  # noqa: E402  (needs env vars set first)

# ---------------------------------------------------------------------------
# Dataset defaults
# ---------------------------------------------------------------------------
DEFAULT_PREFIX    = "file:///mnt/test/data/flux/train"
DEFAULT_N_FILES   = 8
DEFAULT_RG_PER_FILE = 6          # 288 samples / batch_size 48
DEFAULT_FOOTER_CAP  = 4 * 1024 * 1024   # 4 MiB — covers all RG metadata

# Flux file size for reference reporting (uncompressed, no Snappy)
FLUX_FILE_MIB = 594.0            # ~594 MiB per file


def file_uris(prefix: str, n: int, start: int = 0) -> list[str]:
    """Return s3dlio URIs for n Flux training files.

    Naming matches gen_flux_parquet.py: train_{i:04d}.parquet.
    """
    return [f"{prefix.rstrip('/')}/train_{i:04d}.parquet" for i in range(start, start + n)]


# ---------------------------------------------------------------------------
# Worker: fetch one (file, rg_idx) pair — Mode 1, raw bytes, immediate discard
# ---------------------------------------------------------------------------
def fetch_rg(uri: str, rg_idx: int, footer_cap: int) -> tuple[str, int, int, float]:
    """
    Read one Parquet row group (raw compressed bytes) and discard.

    Returns (uri, rg_idx, nbytes_compressed, elapsed_s).
    nbytes reflects compressed column-chunk bytes, not uncompressed payload.
    """
    t0 = time.monotonic()
    data = s3dlio.parquet_get_rg(uri, rg_idx, footer_cap=footer_cap, decode="raw")
    elapsed = time.monotonic() - t0
    nbytes = len(bytes(data))
    del data                    # release immediately — we measure storage, not decode
    return uri, rg_idx, nbytes, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--prefix",       default=DEFAULT_PREFIX,
                    help=f"Base URI prefix for Flux files (default: {DEFAULT_PREFIX})")
    ap.add_argument("--files",        type=int, default=DEFAULT_N_FILES,
                    help=f"Files per epoch (default: {DEFAULT_N_FILES})")
    ap.add_argument("--rg-per-file",  type=int, default=DEFAULT_RG_PER_FILE,
                    help=f"Row groups per file (default: {DEFAULT_RG_PER_FILE})")
    ap.add_argument("--np",           type=int, default=1,
                    help="Simulated MPI ranks; multiplies pipeline (default: 1)")
    ap.add_argument("--pipeline",     type=int, default=4,
                    help="Concurrent parquet_get_rg calls per rank (default: 4)")
    ap.add_argument("--epochs",       type=int, default=2,
                    help="Epochs to run (default: 2)")
    ap.add_argument("--footer-cap",   type=int, default=DEFAULT_FOOTER_CAP,
                    help=f"Footer prefetch bytes (default: {DEFAULT_FOOTER_CAP})")
    args = ap.parse_args()

    total_workers = args.np * args.pipeline
    uris = file_uris(args.prefix, args.files)
    total_rgs = args.files * args.rg_per_file

    # Partition Tokio threads across simulated MPI ranks
    s3dlio.configure_tokio_threads()

    print("Flux Parquet RG benchmark — Mode 1 (s3dlio raw + discard)")
    print(f"  files={args.files}  rg_per_file={args.rg_per_file}  "
          f"total_rgs={total_rgs}")
    print(f"  np={args.np}  pipeline={args.pipeline}  "
          f"total_workers={total_workers}  epochs={args.epochs}")
    print(f"  prefix:   {args.prefix}")
    print(f"  endpoint: {os.environ.get('AWS_ENDPOINT_URL_S3', '(default AWS)')}")
    print(f"  footer_cap: {args.footer_cap // 1024} KiB")
    print(f"  est. uncompressed data/epoch: "
          f"{args.files * FLUX_FILE_MIB / 1024:.1f} GiB "
          f"({args.files} files × {FLUX_FILE_MIB:.0f} MiB)")
    print()

    epoch_results: list[tuple[int, float, float]] = []  # (epoch, total_gb, mbps)

    for ep in range(1, args.epochs + 1):
        print(f"══ Epoch {ep} ═════════════════════════════════════════════════")

        # Build all (uri, rg_idx) tasks for this epoch
        tasks = [
            (uri, rg_idx)
            for uri in uris
            for rg_idx in range(args.rg_per_file)
        ]

        epoch_bytes = 0
        rg_count = 0

        t_epoch = time.monotonic()
        with ThreadPoolExecutor(max_workers=total_workers) as ex:
            futs = {
                ex.submit(fetch_rg, uri, rg_idx, args.footer_cap): (uri, rg_idx)
                for uri, rg_idx in tasks
            }
            for fut in as_completed(futs):
                uri, rg_idx, nbytes, elapsed = fut.result()
                epoch_bytes += nbytes
                rg_count += 1
                if rg_idx == 0:
                    # Print first RG of each file as a progress indicator
                    fname = os.path.basename(uri)
                    mbps = nbytes / elapsed / 1e6 if elapsed > 0 else 0
                    print(f"  {fname}  rg=0  {nbytes/1024:.0f} KiB  "
                          f"{elapsed*1000:.1f} ms  {mbps:.0f} MB/s")
        t_epoch = time.monotonic() - t_epoch

        epoch_mbps = epoch_bytes / t_epoch / 1e6
        epoch_gib  = epoch_bytes / 1024**3
        epoch_results.append((ep, epoch_gib, epoch_mbps))

        print(f"  ── epoch {ep} total: {rg_count} RGs  "
              f"{epoch_gib:.3f} GiB compressed  "
              f"{t_epoch:.2f} s  {epoch_mbps:.0f} MB/s")
        print()

    # Summary
    print("══ Summary ═══════════════════════════════════════════════")
    print(f"  {'Epoch':<8}  {'Compressed GiB':>15}  {'Throughput MB/s':>16}")
    print(f"  {'-'*8}  {'-'*15}  {'-'*16}")
    for ep, gib, mbps in epoch_results:
        print(f"  {ep:<8}  {gib:>15.3f}  {mbps:>16.0f}")

    if len(epoch_results) >= 2:
        ep2_mbps = epoch_results[1][2]
        print()
        print(f"  Epoch 2 reflects warm OS/server cache: {ep2_mbps:.0f} MB/s")

    # Note on compressed vs uncompressed
    print()
    print("  Note: bytes reported are compressed column-chunk bytes")
    print("  (decode='raw' returns Parquet payload before decompression).")
    print(f"  Flux files have compression=none so raw ≈ uncompressed payload.")


if __name__ == "__main__":
    main()
