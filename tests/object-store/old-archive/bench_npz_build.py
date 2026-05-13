"""
NPZ serialization speed benchmark.

Tests several approaches to building a valid .npz file from raw bytes,
measuring wall-clock time for 139.8 MiB (unet3d shape).

Usage:
    uv run python3 tests/object-store/bench_npz_build.py
"""
import io
import struct
import time
import zipfile
import zlib

import dgen_py
import numpy as np

SHAPE = (6053, 6053, 1)           # actual unet3d datagen shape
DTYPE_STR = "<f4"                  # float32 little-endian
TOTAL_ELEMENTS = 6053 * 6053 * 1
TOTAL_BYTES = TOTAL_ELEMENTS * 4   # 139.8 MiB
NRUNS = 5


# ---------------------------------------------------------------------------
# NPY header builder (pure Python, no numpy)
# ---------------------------------------------------------------------------
def build_npy_header(shape, dtype_str="<f4"):
    """Return bytes for a valid NPY 1.0 header for the given shape/dtype."""
    shape_str = ", ".join(str(d) for d in shape)
    if len(shape) == 1:
        shape_str += ","
    # NPY dict: descr, fortran_order, shape — order matters for compatibility
    header_dict = (
        f"{{'descr': '{dtype_str}', 'fortran_order': False, "
        f"'shape': ({shape_str},), }}"
    )
    hdr = header_dict.encode("latin1")
    # Pad so that (PREFIX + len(hdr) + 1) % 64 == 0  (NPY 1.0 spec)
    PREFIX = 10  # magic(6) + version(2) + hlen(2)
    pad = (64 - ((PREFIX + len(hdr) + 1) % 64)) % 64
    hdr = hdr + b" " * pad + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr


# ---------------------------------------------------------------------------
# Tiny y-array NPY bytes (int64 [0])
# ---------------------------------------------------------------------------
def build_y_npy():
    buf = io.BytesIO()
    np.save(buf, np.array([0], dtype=np.int64))
    return buf.getvalue()


_NPY_HDR_X = build_npy_header(SHAPE, DTYPE_STR)
_Y_NPY = build_y_npy()


# ---------------------------------------------------------------------------
# Method 1: np.savez baseline
# ---------------------------------------------------------------------------
def method1_savez(raw_view):
    """Current dlio_benchmark production path."""
    arr = np.frombuffer(raw_view, dtype=DTYPE_STR).reshape(SHAPE)
    buf = io.BytesIO()
    np.savez(buf, x=arr, y=[0])
    return buf


# ---------------------------------------------------------------------------
# Method 2: zipfile.ZipFile + zf.open() streaming
# ---------------------------------------------------------------------------
def method2_zipfile_stream(raw_view):
    """zipfile wrapper; still pays CRC32 + write overhead."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        with zf.open("x.npy", "w") as f:
            f.write(_NPY_HDR_X)
            f.write(raw_view)  # buffer protocol: no extra copy
        zf.writestr("y.npy", _Y_NPY)
    return buf


# ---------------------------------------------------------------------------
# Method 3: raw ZIP construction (manual, no Python zipfile overhead)
#
# Build a minimal ZIP archive manually:
#   - Local file header  (30 + len(name) bytes)
#   - File data (npy_header + raw bytes)
#   - Central directory entry
#   - End-of-central-directory record
#
# Uses zlib.crc32() (C code) for incremental CRC over npy_hdr + raw data.
# Pre-allocates bytearray to exact final size — zero reallocation.
# ---------------------------------------------------------------------------
def _zip_local_header(name: bytes, data_size: int, crc: int) -> bytes:
    # PK local file header signature
    # version needed: 20 (2.0)
    # general purpose bit flag: 0
    # compression method: 0 (STORED)
    # last mod time/date: 0
    # crc-32, compressed size, uncompressed size
    return struct.pack(
        "<4sHHHHHIIIHH",
        b"PK\x03\x04",  # local file header signature
        20,              # version needed
        0,               # flags
        0,               # compression: STORED
        0,               # mod time
        0,               # mod date
        crc,             # CRC-32
        data_size,       # compressed size
        data_size,       # uncompressed size
        len(name),       # file name length
        0,               # extra field length
    ) + name


def _zip_central_dir_entry(
    name: bytes, data_size: int, crc: int, local_header_offset: int
) -> bytes:
    return struct.pack(
        "<4sHHHHHHIIIHHHHHII",
        b"PK\x01\x02",  # central dir signature
        20,              # version made by
        20,              # version needed
        0,               # flags
        0,               # compression: STORED
        0,               # mod time
        0,               # mod date
        crc,             # CRC-32
        data_size,       # compressed size
        data_size,       # uncompressed size
        len(name),       # file name length
        0,               # extra field length
        0,               # comment length
        0,               # disk number start
        0,               # internal attributes
        0,               # external attributes
        local_header_offset,  # relative offset of local header
    ) + name


def _zip_eocd(num_entries: int, cd_size: int, cd_offset: int) -> bytes:
    return struct.pack(
        "<4sHHHHIIH",
        b"PK\x05\x06",  # end of central directory signature
        0,               # disk number
        0,               # disk with start of central directory
        num_entries,     # entries on this disk
        num_entries,     # total entries
        cd_size,         # central directory size
        cd_offset,       # central directory offset
        0,               # comment length
    )


def _build_raw_zip_parts(raw_view):
    """Compute CRC32 and return list of parts for the raw ZIP/NPZ structure."""
    name_x = b"x.npy"
    name_y = b"y.npy"

    crc_x = zlib.crc32(_NPY_HDR_X)
    crc_x = zlib.crc32(raw_view, crc_x) & 0xFFFFFFFF  # buffer protocol: 1× read
    crc_y = zlib.crc32(_Y_NPY) & 0xFFFFFFFF
    data_size_x = len(_NPY_HDR_X) + TOTAL_BYTES
    data_size_y = len(_Y_NPY)

    lh_x = _zip_local_header(name_x, data_size_x, crc_x)
    lh_y = _zip_local_header(name_y, data_size_y, crc_y)
    offset_x = 0
    offset_y = offset_x + len(lh_x) + data_size_x
    cd_x = _zip_central_dir_entry(name_x, data_size_x, crc_x, offset_x)
    cd_y = _zip_central_dir_entry(name_y, data_size_y, crc_y, offset_y)
    cd_offset = offset_y + len(lh_y) + data_size_y
    eocd = _zip_eocd(2, len(cd_x) + len(cd_y), cd_offset)

    return [lh_x, _NPY_HDR_X, raw_view, lh_y, _Y_NPY, cd_x, cd_y, eocd]


def method3_raw_zip(raw_view):
    """
    WRONG method3: used bytes(out) causing 3× copies — kept as reference.
    Replaced by method3b and method3c.
    """
    parts = _build_raw_zip_parts(raw_view)
    # b''.join: 1× copy of raw_view via buffer protocol → produces bytes object
    data = b"".join(parts)
    # BytesIO(data) copies again → 2× total copies of raw_view
    return io.BytesIO(data)


def method3b_bjoin_bytes(raw_view):
    """
    b''.join → bytes.  Return the bytes object directly (NO BytesIO wrapper).
    put_data() in obj_store_lib.py handles bytes directly: payload = data.
    So this avoids the extra BytesIO copy.
    Total copies of raw_view: CRC32 read (1×) + b''.join copy (1×) = 2× passes.
    """
    parts = _build_raw_zip_parts(raw_view)
    return b"".join(parts)  # returns bytes, not BytesIO


def method3c_preallocated_ba(raw_view):
    """
    Pre-allocate a bytearray to the exact NPZ size, fill it, wrap in BytesIO.
    Avoids BytesIO reallocation overhead but still makes 2× copies of raw_view
    (CRC32 read + bytearray write; BytesIO wraps the bytearray without copy).

    NOTE: io.BytesIO(bytearray) still copies the bytearray in CPython.
    This method exists to measure whether pre-allocation helps.
    """
    parts = _build_raw_zip_parts(raw_view)
    # Compute exact total size
    total = sum(len(p) if not isinstance(p, (bytes, bytearray)) else len(p)
                for p in parts)
    # b''.join: pre-allocates a bytes of exactly the right size, 1× copy each part
    data = b"".join(parts)
    return io.BytesIO(data)  # BytesIO copies the bytes object again


# ---------------------------------------------------------------------------
# Method 4: pre-allocated BytesIO + np.savez
# (avoids BytesIO reallocation overhead)
# ---------------------------------------------------------------------------
def method4_preallocated_savez(raw_view):
    """
    Pre-allocate BytesIO to exact NPZ size before calling np.savez.
    Avoids BytesIO reallocation overhead.
    """
    arr = np.frombuffer(raw_view, dtype=DTYPE_STR).reshape(SHAPE)
    # NPZ size = local_hdr_x + npy_hdr + raw_data + local_hdr_y + y_data + central_dir + eocd
    # Slightly overestimate (extra 2 KiB) to avoid re-alloc at boundary
    estimated_size = TOTAL_BYTES + len(_NPY_HDR_X) + len(_Y_NPY) + 2048
    output = io.BytesIO()
    # Pre-allocate by seeking to end and writing a zero byte
    output.seek(estimated_size - 1)
    output.write(b"\x00")
    output.seek(0)
    np.savez(output, x=arr, y=[0])
    actual_size = output.tell()
    output.truncate(actual_size)
    output.seek(0)
    return output


# ---------------------------------------------------------------------------
# Microbenchmarks — isolate individual operations
# ---------------------------------------------------------------------------
def micro_crc32(raw_view):
    """How long does zlib.crc32 take over 140 MiB?"""
    crc = zlib.crc32(_NPY_HDR_X)
    crc = zlib.crc32(raw_view, crc) & 0xFFFFFFFF
    return crc


def micro_bjoin(raw_view):
    """How long does b''.join([...raw_view...]) take for 140 MiB?"""
    return b"".join([b"\x00" * 100, raw_view, b"\x00" * 100])


def micro_bytesio_write(raw_view):
    """How long does BytesIO.write(140 MiB) take (from scratch)?"""
    buf = io.BytesIO()
    buf.write(raw_view)
    return buf


# ---------------------------------------------------------------------------
# Verify method3b produces a valid NPZ that numpy can read
# ---------------------------------------------------------------------------
def verify_method3b():
    raw = dgen_py.generate_buffer(TOTAL_BYTES)
    data = method3b_bjoin_bytes(raw)
    assert isinstance(data, bytes), f"expected bytes, got {type(data)}"
    npz = np.load(io.BytesIO(data))
    assert "x" in npz.files, f"'x' key missing, got: {npz.files}"
    arr = npz["x"]
    assert arr.shape == SHAPE, f"shape mismatch: {arr.shape} != {SHAPE}"
    assert arr.dtype == np.dtype(DTYPE_STR), f"dtype: {arr.dtype}"
    assert "y" in npz.files, "'y' key missing"
    print(f"[verify] method3b ok: shape={arr.shape}, dtype={arr.dtype}, size={len(data)/1024/1024:.1f} MiB")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def bench(label, fn, raw_fn, result_is_bytes=False):
    times = []
    sizes = []
    for _ in range(NRUNS):
        raw = raw_fn()  # fresh data each run (excludes generation time)
        t0 = time.perf_counter()
        result = fn(raw)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if result_is_bytes:
            sizes.append(len(result))
        elif hasattr(result, "tell"):
            result.seek(0, 2)
            sizes.append(result.tell())
        else:
            sizes.append(0)

    # Drop first (warm-up), average rest
    warm = times[0]
    avg = sum(times[1:]) / len(times[1:])
    tput = TOTAL_BYTES / avg / 1024 / 1024
    print(
        f"  {label:<46s}  warm={warm*1000:.0f}ms  avg={avg*1000:.0f}ms  "
        f"{tput:.0f} MB/s  size={sizes[0]/1024/1024:.1f} MiB"
    )
    return avg


def bench_micro(label, fn, raw_fn):
    times = []
    for _ in range(NRUNS):
        raw = raw_fn()
        t0 = time.perf_counter()
        fn(raw)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    warm = times[0]
    avg = sum(times[1:]) / len(times[1:])
    tput = TOTAL_BYTES / avg / 1024 / 1024
    print(
        f"  {label:<46s}  warm={warm*1000:.0f}ms  avg={avg*1000:.0f}ms  {tput:.0f} MB/s"
    )
    return avg


def main():
    print(f"Shape: {SHAPE}  dtype: {DTYPE_STR}  size: {TOTAL_BYTES/1024/1024:.1f} MiB")
    print(f"Runs: {NRUNS} (first is warm-up, avg of rest)")
    print()

    raw_fn = lambda: dgen_py.generate_buffer(TOTAL_BYTES)

    print("Verifying method3b produces valid NPZ...")
    verify_method3b()
    print()

    print("Microbenchmarks (component timings):")
    bench_micro("M1. zlib.crc32(raw_view) 140 MiB",   micro_crc32,        raw_fn)
    bench_micro("M2. b''.join([tiny, raw_view, tiny])", micro_bjoin,       raw_fn)
    bench_micro("M3. BytesIO().write(raw_view) 140 MiB", micro_bytesio_write, raw_fn)
    print()

    print("NPZ build benchmarks (returning file-like or bytes for upload):")
    bench("1. np.savez → BytesIO (baseline)",          method1_savez,             raw_fn)
    bench("2. zipfile.ZipFile stream → BytesIO",       method2_zipfile_stream,    raw_fn)
    bench("3a. raw ZIP → bytearray+bytes+BytesIO (bad)", method3_raw_zip,         raw_fn)
    bench("3b. raw ZIP → bytes (b''.join, no BytesIO)",  method3b_bjoin_bytes,    raw_fn, result_is_bytes=True)
    bench("3c. raw ZIP → bytes+BytesIO",               method3c_preallocated_ba,  raw_fn)
    bench("4.  pre-alloc BytesIO + np.savez",          method4_preallocated_savez, raw_fn)


if __name__ == "__main__":
    main()
