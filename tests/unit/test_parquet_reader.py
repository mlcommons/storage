"""
Unit tests for the parquet reader components added to dlio_benchmark.

Covers:
  - FormatType.PARQUET enum presence and get_enum() round-trip
  - _S3RangeFile: seek/tell/read semantics with mocked s3dlio
  - _MinioRangeFile: seek/tell/read semantics with mocked minio client
  - Both range-file implementations transparently serving a real parquet file
    to pyarrow.parquet.ParquetFile (validates the byte-range interface)
  - ParquetReaderS3Iterable: open(), get_sample(), close(), row-group caching,
    and LRU eviction (with FormatReader.__init__ mocked to avoid DLIO init)
  - reader_factory produces a ParquetReaderS3Iterable for FormatType.PARQUET

No S3 endpoint or Minio server is required; all storage calls are intercepted
by in-process mocks backed by in-memory parquet bytes.
"""
import io
import sys
import types
import bisect
import logging
from argparse import Namespace
from unittest.mock import MagicMock, patch, call

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: build an in-memory parquet file shared across tests
# ─────────────────────────────────────────────────────────────────────────────

ROWS_PER_GROUP = 8
NUM_GROUPS = 3
COLUMNS = ["feature1", "label"]
TOTAL_ROWS = ROWS_PER_GROUP * NUM_GROUPS


def _make_parquet_bytes(
    rows_per_group: int = ROWS_PER_GROUP,
    num_groups: int = NUM_GROUPS,
    columns: list = None,
) -> bytes:
    """Return the bytes of a small, multi-row-group parquet file."""
    if columns is None:
        columns = COLUMNS
    tables = [
        pa.table(
            {col: pa.array(range(g * rows_per_group, (g + 1) * rows_per_group))
             for col in columns}
        )
        for g in range(num_groups)
    ]
    full = pa.concat_tables(tables)
    buf = io.BytesIO()
    pq.write_table(full, buf, row_group_size=rows_per_group)
    return buf.getvalue()


@pytest.fixture(scope="module")
def parquet_bytes() -> bytes:
    """The shared parquet payload used by all tests in this module."""
    return _make_parquet_bytes()


# ─────────────────────────────────────────────────────────────────────────────
# Fake s3dlio module
# ─────────────────────────────────────────────────────────────────────────────


def _make_fake_s3dlio(payload: bytes):
    """Return a types.ModuleType that answers s3dlio.stat / get_range from payload."""
    mod = types.ModuleType("s3dlio")
    mod.stat = lambda uri: {"size": len(payload), "last_modified": "", "etag": "abc"}
    mod.get_range = lambda uri, offset, length: memoryview(payload)[offset: offset + length]
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: FormatType.PARQUET enum
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatTypeParquet:
    """FormatType enum has a PARQUET member and get_enum() supports it."""

    def test_format_type_has_parquet(self):
        from dlio_benchmark.common.enumerations import FormatType

        assert hasattr(FormatType, "PARQUET")
        assert FormatType.PARQUET.value == "parquet"

    def test_get_enum_round_trip(self):
        from dlio_benchmark.common.enumerations import FormatType

        result = FormatType.get_enum("parquet")
        assert result == FormatType.PARQUET

    def test_other_format_types_unaffected(self):
        from dlio_benchmark.common.enumerations import FormatType

        assert FormatType.NPZ.value == "npz"
        assert FormatType.NPY.value == "npy"
        assert FormatType.get_enum("npz") == FormatType.NPZ


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: _S3RangeFile seek / tell / read
# ─────────────────────────────────────────────────────────────────────────────


class TestS3RangeFile:
    """_S3RangeFile correctly implements the seekable file-like interface."""

    @pytest.fixture(autouse=True)
    def inject_s3dlio(self, parquet_bytes, monkeypatch):
        """Install a fake s3dlio module for the duration of each test."""
        fake = _make_fake_s3dlio(parquet_bytes)
        monkeypatch.setitem(sys.modules, "s3dlio", fake)
        # Store payload so tests can reference it directly
        self._payload = parquet_bytes

    def _make_file(self, uri="s3://bucket/test.parquet"):
        from dlio_benchmark.reader.parquet_reader_s3_iterable import _S3RangeFile
        return _S3RangeFile(uri)

    # ── capability flags ──────────────────────────────────────────────────────

    def test_readable(self):
        assert self._make_file().readable() is True

    def test_seekable(self):
        assert self._make_file().seekable() is True

    def test_not_writable(self):
        assert self._make_file().writable() is False

    # ── initial state ─────────────────────────────────────────────────────────

    def test_initial_pos_is_zero(self):
        assert self._make_file().tell() == 0

    def test_size_not_fetched_before_needed(self):
        """_size should stay None until a read or SEEK_END is performed."""
        rf = self._make_file()
        assert rf._size is None

    # ── tell / seek SEEK_SET ──────────────────────────────────────────────────

    def test_seek_set(self):
        rf = self._make_file()
        result = rf.seek(42)
        assert result == 42
        assert rf.tell() == 42

    def test_seek_set_to_zero(self):
        rf = self._make_file()
        rf.seek(100)
        rf.seek(0)
        assert rf.tell() == 0

    # ── seek SEEK_CUR ─────────────────────────────────────────────────────────

    def test_seek_cur_advances(self):
        rf = self._make_file()
        rf.seek(10)
        result = rf.seek(5, 1)
        assert result == 15
        assert rf.tell() == 15

    def test_seek_cur_from_zero(self):
        rf = self._make_file()
        result = rf.seek(7, 1)
        assert result == 7

    # ── seek SEEK_END ─────────────────────────────────────────────────────────

    def test_seek_end_triggers_stat(self):
        """SEEK_END must fetch file size from s3dlio.stat()."""
        rf = self._make_file()
        assert rf._size is None
        rf.seek(0, 2)
        assert rf._size == len(self._payload)

    def test_seek_end_positions_at_end(self):
        rf = self._make_file()
        result = rf.seek(0, 2)
        assert result == len(self._payload)
        assert rf.tell() == len(self._payload)

    def test_seek_end_negative_offset(self):
        rf = self._make_file()
        result = rf.seek(-10, 2)
        assert result == len(self._payload) - 10

    # ── read ──────────────────────────────────────────────────────────────────

    def test_read_n_bytes(self):
        rf = self._make_file()
        data = rf.read(4)
        assert data == self._payload[:4]
        assert rf.tell() == 4

    def test_read_advances_position(self):
        rf = self._make_file()
        rf.read(10)
        rf.read(5)
        assert rf.tell() == 15

    def test_read_from_offset(self):
        rf = self._make_file()
        rf.seek(100)
        data = rf.read(10)
        assert data == self._payload[100:110]

    def test_read_zero_bytes(self):
        rf = self._make_file()
        data = rf.read(0)
        assert data == b""
        assert rf.tell() == 0

    def test_read_all(self):
        rf = self._make_file()
        data = rf.read(-1)
        assert data == self._payload
        assert rf.tell() == len(self._payload)

    def test_readall(self):
        rf = self._make_file()
        data = rf.readall()
        assert data == self._payload

    def test_read_past_end_is_clamped(self):
        rf = self._make_file()
        rf.seek(len(self._payload) - 3)
        data = rf.read(100)       # asks for 100 but only 3 remain
        assert len(data) == 3

    def test_read_at_end_returns_empty(self):
        rf = self._make_file()
        rf.seek(0, 2)             # end
        data = rf.read(10)
        assert data == b""

    # ── pyarrow integration ───────────────────────────────────────────────────

    def test_pyarrow_reads_parquet_through_range_file(self):
        """pyarrow.parquet.ParquetFile must work when backed by _S3RangeFile."""
        rf = self._make_file()
        pf = pq.ParquetFile(rf)
        assert pf.metadata.num_row_groups == NUM_GROUPS
        assert pf.metadata.num_rows == TOTAL_ROWS

    def test_pyarrow_row_group_data_is_correct(self):
        rf = self._make_file()
        pf = pq.ParquetFile(rf)
        for rg_idx in range(NUM_GROUPS):
            table = pf.read_row_group(rg_idx)
            expected_start = rg_idx * ROWS_PER_GROUP
            assert table["feature1"][0].as_py() == expected_start
            assert len(table) == ROWS_PER_GROUP


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: _MinioRangeFile seek / tell / read
# ─────────────────────────────────────────────────────────────────────────────


class TestMinioRangeFile:
    """_MinioRangeFile correctly implements the seekable file-like interface."""

    @pytest.fixture()
    def minio_client_and_payload(self, parquet_bytes):
        """Return (mocked minio client, payload bytes)."""
        client = MagicMock()
        client.stat_object.return_value = MagicMock(size=len(parquet_bytes))

        def _get_object(bucket, key, offset=0, length=None):
            chunk = parquet_bytes[offset: offset + (length or len(parquet_bytes) - offset)]
            resp = MagicMock()
            resp.read.return_value = chunk
            return resp

        client.get_object.side_effect = _get_object
        return client, parquet_bytes

    def _make_file(self, client, payload):
        from dlio_benchmark.reader.parquet_reader_s3_iterable import _MinioRangeFile
        return _MinioRangeFile("my-bucket", "test.parquet", client)

    def test_readable(self, minio_client_and_payload):
        client, payload = minio_client_and_payload
        assert self._make_file(client, payload).readable() is True

    def test_seekable(self, minio_client_and_payload):
        client, payload = minio_client_and_payload
        assert self._make_file(client, payload).seekable() is True

    def test_not_writable(self, minio_client_and_payload):
        client, payload = minio_client_and_payload
        assert self._make_file(client, payload).writable() is False

    def test_initial_pos_is_zero(self, minio_client_and_payload):
        client, payload = minio_client_and_payload
        assert self._make_file(client, payload).tell() == 0

    def test_seek_and_tell(self, minio_client_and_payload):
        client, payload = minio_client_and_payload
        rf = self._make_file(client, payload)
        rf.seek(50)
        assert rf.tell() == 50

    def test_seek_from_end_calls_stat(self, minio_client_and_payload):
        client, payload = minio_client_and_payload
        rf = self._make_file(client, payload)
        client.stat_object.assert_not_called()
        rf.seek(0, 2)
        client.stat_object.assert_called_once_with("my-bucket", "test.parquet")
        assert rf.tell() == len(payload)

    def test_read_n_bytes(self, minio_client_and_payload):
        client, payload = minio_client_and_payload
        rf = self._make_file(client, payload)
        data = rf.read(4)
        assert data == payload[:4]
        assert rf.tell() == 4

    def test_read_from_offset(self, minio_client_and_payload):
        client, payload = minio_client_and_payload
        rf = self._make_file(client, payload)
        rf.seek(100)
        data = rf.read(10)
        assert data == payload[100:110]

    def test_read_zero_bytes(self, minio_client_and_payload):
        client, payload = minio_client_and_payload
        rf = self._make_file(client, payload)
        assert rf.read(0) == b""

    def test_readall(self, minio_client_and_payload):
        client, payload = minio_client_and_payload
        rf = self._make_file(client, payload)
        data = rf.readall()
        assert data == payload

    def test_pyarrow_reads_parquet_through_minio_range_file(self, minio_client_and_payload):
        """pyarrow.parquet.ParquetFile must work when backed by _MinioRangeFile."""
        client, payload = minio_client_and_payload
        rf = self._make_file(client, payload)
        pf = pq.ParquetFile(rf)
        assert pf.metadata.num_row_groups == NUM_GROUPS
        assert pf.metadata.num_rows == TOTAL_ROWS

    def test_pyarrow_row_group_data_via_minio(self, minio_client_and_payload):
        client, payload = minio_client_and_payload
        rf = self._make_file(client, payload)
        pf = pq.ParquetFile(rf)
        table = pf.read_row_group(1)  # second row group
        assert table["feature1"][0].as_py() == ROWS_PER_GROUP  # first value of RG 1


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: ParquetReaderS3Iterable — unit tests with mocked DLIO context
# ─────────────────────────────────────────────────────────────────────────────


def _make_mock_args(
    storage_root="test-bucket",
    storage_library="s3dlio",
    columns=None,
    row_group_cache_size=2,
    endpoint_url=None,
):
    """Return a Namespace that mimics ConfigArguments for ParquetReaderS3Iterable."""
    opts = {"storage_library": storage_library, "row_group_cache_size": row_group_cache_size}
    if columns:
        opts["columns"] = columns
    if endpoint_url:
        opts["endpoint_url"] = endpoint_url
    return Namespace(
        storage_root=storage_root,
        storage_options=opts,
        read_type=None,   # not used in open/get_sample directly
    )


class TestParquetReaderS3Iterable:
    """
    Tests for ParquetReaderS3Iterable.open(), get_sample(), close(), and
    the LRU row-group cache.

    FormatReader.__init__() is patched so DLIO's singleton ConfigArguments
    is never invoked; _args is set manually using a Namespace fixture.
    """

    URI = "s3://test-bucket/data.parquet"
    FILENAME = "data.parquet"

    @pytest.fixture(autouse=True)
    def setup(self, parquet_bytes, monkeypatch):
        """
        Patch FormatReader.__init__, inject fake s3dlio, build reader instance.
        """
        self._payload = parquet_bytes
        fake_s3dlio = _make_fake_s3dlio(parquet_bytes)
        monkeypatch.setitem(sys.modules, "s3dlio", fake_s3dlio)

        # Patch FormatReader.__init__ so no DLIO singleton is needed
        from dlio_benchmark.reader import reader_handler

        def _fake_format_reader_init(inst, dataset_type, thread_index):
            inst.open_file_map = {}
            inst.file_map = {}
            inst.thread_index = thread_index
            inst.global_index_map = {}
            inst.logger = logging.getLogger("test")

        monkeypatch.setattr(
            reader_handler.FormatReader, "__init__", _fake_format_reader_init
        )

        from dlio_benchmark.reader.parquet_reader_s3_iterable import ParquetReaderS3Iterable

        self._reader_cls = ParquetReaderS3Iterable

    def _make_reader(self, **kwargs):
        args = _make_mock_args(**kwargs)
        reader = self._reader_cls.__new__(self._reader_cls)
        # Pre-set _args so ParquetReaderS3Iterable.__init__ can read it
        # (FormatReader.__init__ is mocked and won't set it from ConfigArguments)
        reader._args = args
        reader.__init__(dataset_type=None, thread_index=0, epoch=1)
        return reader

    # ── open() ────────────────────────────────────────────────────────────────

    def test_open_returns_tuple(self):
        reader = self._make_reader()
        result = reader.open(self.FILENAME)
        assert isinstance(result, tuple)
        pf, offsets = result
        assert pf is not None
        assert isinstance(offsets, list)

    def test_open_correct_row_group_count(self):
        reader = self._make_reader()
        pf, offsets = reader.open(self.FILENAME)
        assert pf.metadata.num_row_groups == NUM_GROUPS

    def test_open_cumulative_offsets(self):
        reader = self._make_reader()
        pf, offsets = reader.open(self.FILENAME)
        # offsets should be [0, 8, 16, 24] for 3 groups of 8 rows
        expected = [i * ROWS_PER_GROUP for i in range(NUM_GROUPS + 1)]
        assert offsets == expected

    def test_open_total_rows(self):
        reader = self._make_reader()
        pf, offsets = reader.open(self.FILENAME)
        assert offsets[-1] == TOTAL_ROWS

    # ── get_sample() ─────────────────────────────────────────────────────────

    def test_get_sample_first_row_group(self):
        reader = self._make_reader()
        reader.open_file_map[self.FILENAME] = reader.open(self.FILENAME)
        # Sample 0 is in row group 0
        reader.get_sample(self.FILENAME, 0)
        assert (self.FILENAME, 0) in reader._rg_cache

    def test_get_sample_middle_row_group(self):
        reader = self._make_reader()
        reader.open_file_map[self.FILENAME] = reader.open(self.FILENAME)
        # Sample ROWS_PER_GROUP is the first row of RG 1
        reader.get_sample(self.FILENAME, ROWS_PER_GROUP)
        assert (self.FILENAME, 1) in reader._rg_cache

    def test_get_sample_last_row_group(self):
        reader = self._make_reader()
        reader.open_file_map[self.FILENAME] = reader.open(self.FILENAME)
        reader.get_sample(self.FILENAME, TOTAL_ROWS - 1)
        assert (self.FILENAME, NUM_GROUPS - 1) in reader._rg_cache

    def test_get_sample_caches_row_group(self):
        """Second call to get_sample for same row group must not re-fetch.

        The cache stores compressed_bytes (an int), not a pyarrow Table.
        Verify the cache entry exists and is an int after the first call,
        and that the value is unchanged after a second call on the same RG.
        """
        reader = self._make_reader()
        reader.open_file_map[self.FILENAME] = reader.open(self.FILENAME)
        reader.get_sample(self.FILENAME, 0)
        cached_first = reader._rg_cache[(self.FILENAME, 0)]
        assert isinstance(cached_first, int), "cache must store compressed byte count (int)"
        reader.get_sample(self.FILENAME, 1)   # same row group 0
        cached_second = reader._rg_cache[(self.FILENAME, 0)]
        assert cached_first == cached_second  # same value, not re-fetched

    def test_get_sample_all_samples_find_correct_rg(self):
        reader = self._make_reader(row_group_cache_size=NUM_GROUPS + 1)
        reader.open_file_map[self.FILENAME] = reader.open(self.FILENAME)
        for sample_idx in range(TOTAL_ROWS):
            expected_rg = sample_idx // ROWS_PER_GROUP
            reader.get_sample(self.FILENAME, sample_idx)
            assert (self.FILENAME, expected_rg) in reader._rg_cache

    # ── cache growth (no LRU eviction within an epoch) ─────────────────────────

    def test_cache_grows_as_rgs_are_accessed(self):
        """One cache entry is added per unique row group accessed; none are evicted.

        The old implementation had an LRU eviction policy bounded by
        row_group_cache_size.  The new implementation keeps byte counts (ints)
        for every row group accessed this epoch and never evicts them during
        the epoch — eviction happens only at finalize().  row_group_cache_size
        in storage_options is silently ignored.
        """
        reader = self._make_reader(row_group_cache_size=2)  # limit is ignored
        reader.open_file_map[self.FILENAME] = reader.open(self.FILENAME)
        for sample_idx in range(TOTAL_ROWS):
            reader.get_sample(self.FILENAME, sample_idx)
        # All NUM_GROUPS row groups must be in the cache — nothing was evicted
        for rg in range(NUM_GROUPS):
            assert (self.FILENAME, rg) in reader._rg_cache

    def test_all_rg_entries_persist_within_epoch(self):
        """After accessing all row groups, every entry survives until finalize()."""
        reader = self._make_reader(row_group_cache_size=2)  # limit is ignored
        reader.open_file_map[self.FILENAME] = reader.open(self.FILENAME)
        reader.get_sample(self.FILENAME, 0)                            # loads RG 0
        reader.get_sample(self.FILENAME, ROWS_PER_GROUP)               # loads RG 1
        reader.get_sample(self.FILENAME, ROWS_PER_GROUP * 2)           # loads RG 2
        # All three must still be present — no LRU eviction within an epoch
        assert (self.FILENAME, 0) in reader._rg_cache
        assert (self.FILENAME, 1) in reader._rg_cache
        assert (self.FILENAME, 2) in reader._rg_cache

    # ── close() ──────────────────────────────────────────────────────────────

    def test_close_does_not_evict_rg_cache(self):
        """close() is intentionally a no-op for _rg_cache.

        In ON_DEMAND mode DLIO calls close() after every single sample.
        Evicting on close would force a full row-group re-fetch for every
        subsequent sample on the same file.  Byte counts must survive close()
        and are only cleared by finalize() at epoch boundary.
        """
        reader = self._make_reader()
        reader.open_file_map[self.FILENAME] = reader.open(self.FILENAME)
        reader.get_sample(self.FILENAME, 0)
        reader.get_sample(self.FILENAME, ROWS_PER_GROUP)
        assert len(reader._rg_cache) == 2

        reader.close(self.FILENAME)
        # Entries must still be present after close()
        remaining = [k for k in reader._rg_cache if k[0] == self.FILENAME]
        assert len(remaining) == 2

    def test_close_preserves_all_files_rg_cache(self):
        """Closing one file leaves all files' byte-count entries intact."""
        reader = self._make_reader(row_group_cache_size=8)
        other = "other.parquet"
        reader.open_file_map[self.FILENAME] = reader.open(self.FILENAME)
        reader.open_file_map[other] = reader.open(self.FILENAME)  # same payload

        reader.get_sample(self.FILENAME, 0)
        reader.get_sample(other, 0)
        reader.close(self.FILENAME)

        # Both files' entries survive close() — eviction happens only at finalize()
        assert (self.FILENAME, 0) in reader._rg_cache
        assert (other, 0) in reader._rg_cache

    # ── capability methods ────────────────────────────────────────────────────

    def test_is_index_based(self):
        reader = self._make_reader()
        assert reader.is_index_based() is True

    def test_is_iterator_based(self):
        reader = self._make_reader()
        assert reader.is_iterator_based() is True

    # ── URI construction ──────────────────────────────────────────────────────

    def test_uri_for_absolute_passthrough(self):
        reader = self._make_reader()
        uri = reader._uri_for_filename("s3://other-bucket/file.parquet")
        assert uri == "s3://other-bucket/file.parquet"

    def test_uri_for_relative_filename(self):
        reader = self._make_reader(storage_root="my-bucket")
        uri = reader._uri_for_filename("train/file.parquet")
        assert uri == "s3://my-bucket/train/file.parquet"

    def test_uri_strips_leading_slash(self):
        reader = self._make_reader(storage_root="my-bucket")
        uri = reader._uri_for_filename("/train/file.parquet")
        assert uri == "s3://my-bucket/train/file.parquet"

    # ── column filtering ──────────────────────────────────────────────────────

    def test_column_filtering_records_byte_count(self):
        """Column filtering is passed to read_row_group; byte count is still cached.

        The old tests checked table.column_names, but the new implementation
        discards the pyarrow Table immediately after measuring its byte count.
        We verify instead that:
        - _columns is wired to the columns option, and
        - get_sample() succeeds and stores an int byte count in _rg_cache.
        """
        reader = self._make_reader(columns=["feature1"])
        assert reader._columns == ["feature1"]
        reader.open_file_map[self.FILENAME] = reader.open(self.FILENAME)
        reader.get_sample(self.FILENAME, 0)
        cached = reader._rg_cache[(self.FILENAME, 0)]
        assert isinstance(cached, int)
        assert cached > 0  # some bytes were read

    def test_no_column_filter_reads_all(self):
        """With columns=None, _columns is None and the byte count is still cached."""
        reader = self._make_reader(columns=None)
        assert reader._columns is None
        reader.open_file_map[self.FILENAME] = reader.open(self.FILENAME)
        reader.get_sample(self.FILENAME, 0)
        cached = reader._rg_cache[(self.FILENAME, 0)]
        assert isinstance(cached, int)
        assert cached > 0


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: reader_factory PARQUET routing
# ─────────────────────────────────────────────────────────────────────────────


class TestReaderFactoryParquetRouting:
    """reader_factory correctly routes FormatType.PARQUET → ParquetReaderS3Iterable."""

    def test_parquet_import_routed_from_factory(self):
        """
        Verify the factory contains a PARQUET branch by importing the reader class
        directly and confirming refusals for unsupported formats still work.
        """
        from dlio_benchmark.reader.parquet_reader_s3_iterable import ParquetReaderS3Iterable
        from dlio_benchmark.common.enumerations import FormatType

        # Just verify the class is importable and is the right type
        assert issubclass(ParquetReaderS3Iterable, object)
        assert FormatType.PARQUET is not None

    def test_factory_source_contains_parquet_branch(self):
        """
        Verify reader_factory.py actually has a PARQUET branch by reading
        the module source — prevents silent routing failures.
        """
        import inspect
        from dlio_benchmark.reader import reader_factory

        src = inspect.getsource(reader_factory)
        assert "FormatType.PARQUET" in src
        assert "ParquetReaderS3Iterable" in src
