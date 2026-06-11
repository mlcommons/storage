#!/usr/bin/env python3
"""
distill_fio.py — Convert bpftrace histogram output to a fio workload file.

Parses the output of storage_latency_stack.bt and generates a standalone
fio .ini that reproduces the same I/O pattern: bssplit, read/write ratio,
queue depth, and idle time.

Usage:
    # Pipe from bpftrace:
    sudo bpftrace storage_latency_stack.bt "vllm" 2>&1 | python3 distill_fio.py

    # From saved file:
    python3 distill_fio.py < trace_output.txt

    # With custom output name:
    python3 distill_fio.py -o vllm_workload.ini < trace_output.txt

    # Standalone (called by storage_latency_stack.sh --fio):
    python3 distill_fio.py -i /tmp/trace_raw.txt -o fio_traced.ini --process vllm

Author: Hazem Awadallah, Kingston Digital <hazem_awadallah@kingston.com>
License: Apache-2.0, donated to MLCommons
"""

import re
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def parse_bpftrace_output(text: str) -> Dict:
    """Parse bpftrace histogram output into structured dict."""
    result = {}
    current_hist = None

    for line in text.split('\n'):
        # Match histogram name: @histogram_name:
        hist_match = re.match(r'^\s*@(\w+):\s*$', line)
        if hist_match:
            current_hist = hist_match.group(1)
            result[current_hist] = []
            continue

        if current_hist is None:
            continue

        # Match numeric bucket: [low, high)  count  |@@@@|
        m = re.match(r'^\[(\d+),\s*(\d+)\)\s+(\d+)\s+\|', line)
        if m:
            result[current_hist].append({
                'low': int(m.group(1)),
                'high': int(m.group(2)),
                'count': int(m.group(3)),
            })
            continue

        # Match K/M bucket: [1K, 2K)  count  |@@@@|
        m = re.match(r'^\[(\d+)([KM]),\s*(\d+)([KM])\)\s+(\d+)\s+\|', line)
        if m:
            def to_val(n, s):
                v = int(n)
                return v * 1024 if s == 'K' else v * 1048576
            result[current_hist].append({
                'low': to_val(m.group(1), m.group(2)),
                'high': to_val(m.group(3), m.group(4)),
                'count': int(m.group(5)),
            })

    return result


def hist_percentile(buckets: List[Dict], pct: float) -> int:
    """Return the lower bound of the bucket containing the given percentile."""
    total = sum(b['count'] for b in buckets)
    if total == 0:
        return 0
    target = total * pct / 100.0
    cumulative = 0
    for b in buckets:
        cumulative += b['count']
        if cumulative >= target:
            return b['low']
    return buckets[-1]['low'] if buckets else 0


def hist_to_bssplit(buckets: List[Dict]) -> str:
    """Convert a bpftrace size histogram (KiB) to fio bssplit format."""
    total = sum(b['count'] for b in buckets)
    if total == 0:
        return "4k/100"
    parts = []
    for b in buckets:
        if b['count'] == 0:
            continue
        size_kb = b['low']
        pct = int(round(b['count'] * 100.0 / total))
        if pct == 0 and b['count'] > 0:
            pct = 1
        if size_kb >= 1024:
            size_str = f"{size_kb // 1024}m"
        elif size_kb == 0:
            continue  # skip zero-size buckets
        else:
            size_str = f"{size_kb}k"
        parts.append(f"{size_str}/{pct}")
    return ":".join(parts) if parts else "4k/100"


def generate_fio(histograms: Dict, process_name: str = "") -> str:
    """Generate a fio .ini config from parsed histograms."""

    # ── bssplit ──
    read_bs = histograms.get('bssplit_read_kb', [])
    write_bs = histograms.get('bssplit_write_kb', [])
    read_bssplit = hist_to_bssplit(read_bs)
    write_bssplit = hist_to_bssplit(write_bs)

    # ── rwmixread ──
    read_count = sum(b['count'] for b in read_bs)
    write_count = sum(b['count'] for b in write_bs)
    total_io = read_count + write_count
    rwmixread = int(round(read_count * 100.0 / total_io)) if total_io > 0 else 50

    # ── iodepth from QD histogram P50 ──
    iodepth = 32
    for qd_key in ('qd_read', 'qd_write'):
        buckets = histograms.get(qd_key, [])
        if buckets:
            candidate = max(1, hist_percentile(buckets, 50))
            iodepth = max(iodepth, candidate)

    # ── thinktime from write_to_fsync gap ──
    thinktime_us = 0
    wt_buckets = histograms.get('write_to_fsync_us', [])
    if sum(b['count'] for b in wt_buckets) >= 4:
        thinktime_us = hist_percentile(wt_buckets, 50)

    # ── thinktime_iotime from fsync latency ──
    thinktime_iotime_us = 0
    fs_buckets = histograms.get('fsync_us', [])
    if sum(b['count'] for b in fs_buckets) >= 4:
        thinktime_iotime_us = hist_percentile(fs_buckets, 50)

    # ── LBA summary ──
    lba_summary = ""
    for lba_key, direction in [('lba_read_gb', 'Read'), ('lba_write_gb', 'Write')]:
        buckets = histograms.get(lba_key, [])
        total = sum(b['count'] for b in buckets)
        if total > 0:
            hot = [b for b in buckets if b['count'] > total * 0.01]
            if hot:
                lba_summary += f"#   {direction} hot zone: {hot[0]['low']}-{hot[-1]['high']}GiB ({sum(b['count'] for b in hot)*100//total}% of I/O)\n"

    # ── D2C summary ──
    d2c_summary = ""
    for key, direction in [('d2c_read_us', 'Read'), ('d2c_write_us', 'Write')]:
        buckets = histograms.get(key, [])
        total = sum(b['count'] for b in buckets)
        if total > 0:
            p50 = hist_percentile(buckets, 50)
            p99 = hist_percentile(buckets, 99)
            d2c_summary += f"#   D2C {direction}: P50={p50} us, P99={p99} us ({total} samples)\n"

    # ── Build config ──
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    job_name = process_name.replace('-', '_') if process_name else "traced"

    lines = [
        f"# Storage Latency Stack; Distilled fio Workload",
        f"# Generated: {timestamp}",
        f"# Source process: {process_name or '(all)'}",
        f"# Total traced I/Os: {total_io:,} ({read_count:,} reads, {write_count:,} writes)",
        f"#",
    ]
    if d2c_summary:
        lines.append(f"# Device latency (D2C):")
        lines.append(d2c_summary.rstrip())
    if lba_summary:
        lines.append(f"# LBA spatial distribution:")
        lines.append(lba_summary.rstrip())
    lines.extend([
        f"#",
        f"# Usage:",
        f"#   fio <this_file> --filename=/dev/nvmeXn1",
        f"#   fio <this_file> --filename=/mnt/nvme/fio_test --size=100G",
        f"",
        f"[{job_name}_workload]",
        f"ioengine=libaio",
        f"direct=1",
        f"time_based",
        f"runtime=300",
        f"rw=randrw",
        f"rwmixread={rwmixread}",
        f"bssplit={read_bssplit},{write_bssplit}",
        f"iodepth={iodepth}",
        f"iodepth_batch_submit={iodepth}",
        f"iodepth_batch_complete_min=1",
        f"size=100%",
    ])

    if thinktime_us > 0:
        lines.append(f"thinktime={thinktime_us}")
        lines.append(f"thinktime_blocks={iodepth}")
        if thinktime_iotime_us > 0:
            iotime_s = max(1, thinktime_iotime_us // 1000000)
            lines.append(f"# thinktime_iotime={iotime_s}s  # uncomment for fio 3.28+")

    lines.extend([
        f"refill_buffers=1",
        f"norandommap=1",
        f"randrepeat=0",
        f"numjobs=1",
        f"group_reporting",
        f"percentile_list=50:95:99:99.9:99.99",
    ])

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description='Distill bpftrace trace output into a fio workload file.'
    )
    parser.add_argument('-i', '--input', default=None,
                        help='Input file (bpftrace output). Default: stdin')
    parser.add_argument('-o', '--output', default=None,
                        help='Output fio .ini file. Default: fio_traced_TIMESTAMP.ini')
    parser.add_argument('--process', default='',
                        help='Process name (for fio job naming and comments)')
    parser.add_argument('--stdout', action='store_true',
                        help='Print fio config to stdout instead of file')
    args = parser.parse_args()

    # Read input
    if args.input:
        with open(args.input, 'r') as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    # Parse
    histograms = parse_bpftrace_output(text)
    if not histograms:
        print("Error: no histograms found in input", file=sys.stderr)
        sys.exit(1)

    # Check for required histograms
    has_bssplit = 'bssplit_read_kb' in histograms or 'bssplit_write_kb' in histograms
    if not has_bssplit:
        print("Error: no bssplit histograms found. Was storage_latency_stack.bt used?", file=sys.stderr)
        sys.exit(1)

    # Generate
    fio_config = generate_fio(histograms, args.process)

    if args.stdout:
        print(fio_config)
    else:
        if args.output:
            outpath = args.output
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            outpath = f"fio_traced_{ts}.ini"
        with open(outpath, 'w') as f:
            f.write(fio_config)
        print(f"fio workload saved to {outpath}")


if __name__ == '__main__':
    main()
