#!/usr/bin/env python3
"""
gen_flux_parquet.py — Generate Flux-schema Parquet files for storage benchmarking.

Uses s3dlio.generate_and_write_parquet_schema() — pure Rust Xoshiro256++
RollingPool data generation with zero Python data involvement and zero numpy.

Flux schema (from flux_b200.yaml / flux_mi355.yaml):
  t5_encodings   FixedSizeList<float32>[524328]  — text encoder embedding
  clip_encodings FixedSizeList<float32>[409]      — CLIP embedding
  mean           FixedSizeList<float32>[8232]     — VAE latent mean
  logvar         FixedSizeList<float32>[8232]     — VAE latent log-variance
  timestamp      FixedSizeList<float32>[7]        — diffusion timestep encoding

Per-file characteristics:
  288 rows (samples) × 541,208 float32 values/row = ~594.6 MiB uncompressed
  6 row groups × 48 rows each  (batch_size=48 from flux_b200.yaml)
  compression: none  (Flux data is already compressed/incompressible embeddings)

Destination URIs:
  file:///mnt/test/data/flux/train/train_{i:04d}.parquet   (local filesystem)
  s3://mlp-flux/data/flux/train/train_{i:04d}.parquet      (S3 / s3-ultra)

Usage:
    # Quick local smoke test — 8 files (~4.6 GiB)
    python3 gen_flux_parquet.py --dest file:///mnt/test/data/flux/train --files 8

    # Larger local batch — 64 files (~37 GiB, fits in /mnt/test 816 GB free)
    python3 gen_flux_parquet.py --dest file:///mnt/test/data/flux/train --files 64

    # Full-scale on S3 (2 PB capacity)
    python3 gen_flux_parquet.py --dest s3://mlp-flux/data/flux/train --files 4296 --workers 16

Options:
    --dest URI         Base URI prefix for output files (no trailing slash)
    --files N          Number of files to generate (default: 8)
    --rows-per-file N  Rows (samples) per file (default: 288, matches spec)
    --rows-per-rg N    Rows per row group (default: 48 = batch_size)
    --workers N        Concurrent generation threads (default: 4)
    --start-idx N      First file index (default: 0, for resuming partial runs)
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
# Flux column specification  (name, num_float32_values_per_row)
# Source: flux_b200.yaml and flux_mi355.yaml
# ---------------------------------------------------------------------------
FLUX_COLUMNS: list[tuple[str, int]] = [
    ("t5_encodings",   524_328),  # text encoder output  (2.0 MiB/row)
    ("clip_encodings", 409),      # CLIP embedding
    ("mean",           8_232),    # VAE latent mean
    ("logvar",         8_232),    # VAE latent log-variance
    ("timestamp",      7),        # diffusion timestep encoding
]
ROWS_PER_FILE_DEFAULT = 288
ROWS_PER_RG_DEFAULT   = 48       # = batch_size in flux_b200.yaml; 288/48 = 6 RGs


# ---------------------------------------------------------------------------
# Write one file — pure Rust, GIL released for full duration
# ---------------------------------------------------------------------------
def write_one(
    idx: int,
    dest_prefix: str,
    columns: list[tuple[str, int]],
    rows_per_rg: int,
    num_row_groups: int,
) -> tuple[int, float]:
    """Generate and write one Flux Parquet file entirely in Rust.

    Returns (idx, elapsed_s).  s3dlio.generate_and_write_parquet_schema()
    releases the GIL for the entire pipeline: Xoshiro256++ data gen,
    Parquet serialization, and store write — zero Python data handling.
    """
    uri = f"{dest_prefix.rstrip('/')}/train_{idx:04d}.parquet"

    # For local file:// URIs we need the directory to exist first
    if dest_prefix.startswith("file://"):
        local_dir = dest_prefix[len("file://"):]
        os.makedirs(local_dir, exist_ok=True)

    t0 = time.monotonic()
    s3dlio.generate_and_write_parquet_schema(uri, columns, rows_per_rg, num_row_groups)
    elapsed = time.monotonic() - t0

    return idx, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dest",
        default="file:///mnt/test/data/flux/train",
        help="Base URI prefix for output files (default: file:///mnt/test/data/flux/train)",
    )
    ap.add_argument(
        "--files", type=int, default=8,
        help="Number of files to generate (default: 8)",
    )
    ap.add_argument(
        "--rows-per-file", type=int, default=ROWS_PER_FILE_DEFAULT,
        help=f"Rows per file (default: {ROWS_PER_FILE_DEFAULT})",
    )
    ap.add_argument(
        "--rows-per-rg", type=int, default=ROWS_PER_RG_DEFAULT,
        help=f"Rows per row group (default: {ROWS_PER_RG_DEFAULT}, = batch_size)",
    )
    ap.add_argument(
        "--workers", type=int, default=4,
        help="Concurrent generation+write threads (default: 4)",
    )
    ap.add_argument(
        "--start-idx", type=int, default=0,
        help="First file index (default: 0, use to resume partial runs)",
    )
    args = ap.parse_args()

    num_row_groups = args.rows_per_file // args.rows_per_rg
    est_mib = args.rows_per_file * sum(s for _, s in FLUX_COLUMNS) * 4 / 1024**2

    # Partition Tokio threads for s3dlio (MPI-aware)
    s3dlio.configure_tokio_threads()

    print("Flux Parquet Generator  (pure Rust — Xoshiro256++ RollingPool, zero numpy)")
    print(f"  dest:          {args.dest}")
    print(f"  files:         {args.files}  (idx {args.start_idx}..{args.start_idx + args.files - 1})")
    print(f"  rows/file:     {args.rows_per_file}  →  {num_row_groups} row groups × {args.rows_per_rg} rows")
    print(f"  est. size:     {est_mib:.1f} MiB/file  ×  {args.files} = {est_mib * args.files / 1024:.1f} GiB total")
    print(f"  workers:       {args.workers}")
    print(f"  schema:        {', '.join(f'{n}[{s}]' for n, s in FLUX_COLUMNS)}")
    print()

    indices = list(range(args.start_idx, args.start_idx + args.files))
    results: list[tuple[int, float]] = []

    t_wall = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                write_one, i, args.dest, FLUX_COLUMNS, args.rows_per_rg, num_row_groups
            ): i
            for i in indices
        }
        for fut in as_completed(futs):
            idx, elapsed = fut.result()
            results.append((idx, elapsed))
            mbps = est_mib / elapsed if elapsed > 0 else 0
            print(f"  train_{idx:04d}.parquet  {est_mib:6.1f} MiB  {elapsed:.2f}s  {mbps:.0f} MB/s")
    t_wall = time.monotonic() - t_wall

    total_mib = est_mib * args.files
    wall_mbps = total_mib / t_wall if t_wall > 0 else 0
    print()
    print(f"  ── Total: {len(results)} files  "
          f"{total_mib/1024:.2f} GiB  "
          f"{t_wall:.1f} s  "
          f"{wall_mbps:.0f} MB/s (wall-clock throughput)")
    print()
    print(f"  Benchmark command:")
    print(f"    python3 bench_parquet_rg_flux.py \\")
    print(f"      --prefix '{args.dest}' \\")
    print(f"      --files {args.files} \\")
    print(f"      --rg-per-file {num_row_groups}")


if __name__ == "__main__":
    main()

