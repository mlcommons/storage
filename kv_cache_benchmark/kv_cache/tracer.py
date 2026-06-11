"""
I/O Trace Logger for KV Cache Benchmark.

When --io-trace-log is specified, the benchmark runs in trace mode:
no actual GPU/CPU/NVMe I/O is performed, but every KV cache operation
is recorded to a CSV log file. The output can be replayed by an external
storage benchmarking tool (e.g. fio, sai3-bench) to measure real hardware
performance independently of the Python benchmark runtime.

Output format (one row per operation):
    Timestamp,Operation,Object_Size_Bytes,Tier,Key,Phase

    Timestamp        Unix epoch (float, 6 decimal places)
    Operation        'Read' or 'Write'
    Object_Size_Bytes  Exact byte size of the KV cache object
    Tier             'Tier-0' (GPU), 'Tier-1' (CPU), 'Tier-2' (NVMe)
    Key              Cache entry identifier — use as the object name /
                     file path in the replay tool (e.g. S3 key, fio filename)
    Phase            'Prefill' (initial write), 'Decode' (per-token read),
                     or 'Evict' (tier-demotion read/write pair)

Tier mapping:
    Tier-0  = GPU VRAM
    Tier-1  = CPU / system RAM
    Tier-2  = NVMe / persistent storage

Compression:
    If the output path ends with '.zst', the CSV is written through a
    streaming zstd compressor (requires the 'zstandard' package).
    This is strongly recommended for runs longer than a few minutes —
    a 1-hour run can produce 500 MB–5 GB of uncompressed CSV, which
    zstd typically reduces by 10–20× at the default compression level.

    Example:
        --io-trace-log kv_ops.csv         # plain CSV
        --io-trace-log kv_ops.csv.zst     # zstd-compressed CSV
"""

import csv
import io
import time
import threading
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Internal tier name → external Tier-N label
_TIER_LABELS = {
    'gpu':  'Tier-0',
    'cpu':  'Tier-1',
    'nvme': 'Tier-2',
}

# Default zstd compression level (1=fastest, 22=smallest; 3 is a good balance)
_DEFAULT_ZSTD_LEVEL = 3


class IOTracer:
    """
    Thread-safe CSV writer that records every KV cache I/O decision.

    Plain CSV usage:
        tracer = IOTracer('/tmp/kv_trace.csv')
        tracer.log('Write', 131072, 'gpu')
        tracer.log('Read',  131072, 'gpu')
        tracer.close()

    zstd-compressed usage (path must end in '.zst'):
        tracer = IOTracer('/tmp/kv_trace.csv.zst')
        # identical API — compression is transparent
        tracer.close()

    Context manager:
        with IOTracer('/tmp/kv_trace.csv.zst') as tracer:
            tracer.log('Write', 131072, 'gpu')
    """

    HEADER = ['Timestamp', 'Operation', 'Object_Size_Bytes', 'Tier', 'Key', 'Phase']

    def __init__(self, path: str, zstd_level: int = _DEFAULT_ZSTD_LEVEL):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ops_logged = 0
        self._closed = False

        # Compression handles
        self._raw_file = None
        self._zstd_writer = None
        self._text_wrapper = None

        use_zstd = self.path.suffix == '.zst'

        if use_zstd:
            try:
                import zstandard as zstd
            except ImportError:
                raise ImportError(
                    "The 'zstandard' package is required for .zst trace output. "
                    "Install it with: uv pip install zstandard"
                )
            self._raw_file = open(self.path, 'wb')
            cctx = zstd.ZstdCompressor(level=zstd_level)
            # stream_writer produces a binary writable stream
            self._zstd_writer = cctx.stream_writer(self._raw_file, closefd=False)
            # Wrap in TextIOWrapper so csv.writer can write text
            self._text_wrapper = io.TextIOWrapper(
                self._zstd_writer, encoding='utf-8', newline=''
            )
            self._writer = csv.writer(self._text_wrapper)
            logger.info(
                f"IOTracer: trace mode active (zstd level {zstd_level}), "
                f"writing to {self.path}"
            )
        else:
            # Plain CSV — line-buffered for low latency flushing
            self._plain_file = open(self.path, 'w', newline='', buffering=1)
            self._writer = csv.writer(self._plain_file)
            logger.info(f"IOTracer: trace mode active (plain CSV), writing to {self.path}")

        self._use_zstd = use_zstd
        self._writer.writerow(self.HEADER)

    def log(self, operation: str, size_bytes: int, tier: str,
             key: str = '', phase: str = '') -> None:
        """
        Record a single KV cache I/O event.

        Args:
            operation:  'Read' or 'Write'
            size_bytes: Total byte size of the KV cache object
            tier:       Internal tier name: 'gpu', 'cpu', or 'nvme'
            key:        Cache entry identifier (object name for replay tools).
                        Links writes to their subsequent reads — essential for
                        accurate workload replay with warp / sai3-bench / fio.
            phase:      Inference phase: 'Prefill' (initial write), 'Decode'
                        (per-token read), or 'Evict' (tier demotion pair).
        """
        if self._closed:
            return
        tier_label = _TIER_LABELS.get(tier, tier)
        ts = time.time()
        with self._lock:
            self._writer.writerow([f'{ts:.6f}', operation, size_bytes, tier_label, key, phase])
            self._ops_logged += 1

    def close(self) -> None:
        """
        Flush and close the trace file.

        For zstd output this finalises the compressed frame so the file
        is a valid, self-contained .zst archive.
        """
        if self._closed:
            return
        with self._lock:
            if self._closed:
                return
            if self._use_zstd:
                # Flush the text layer without letting it close the binary layer
                self._text_wrapper.flush()
                self._text_wrapper.detach()   # detach so TextIOWrapper doesn't close zstd_writer
                self._zstd_writer.close()     # finalise the zstd frame
                self._raw_file.close()
            else:
                self._plain_file.flush()
                self._plain_file.close()
            self._closed = True
        logger.info(
            f"IOTracer: closed — {self._ops_logged:,} operations logged to {self.path}"
        )

    # -------------------------------------------------------------------------
    # Context manager support
    # -------------------------------------------------------------------------

    def __enter__(self) -> 'IOTracer':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
