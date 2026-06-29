"""Unit tests for ``generate_output_location`` and the orgname/systemname
args contract.

The runtime output path is:
    <results_dir>/<mode>/<orgname>/results/<systemname>/<benchmark>/<model>/<command>/<datetime>/

``generate_output_location`` reads orgname and systemname from ``benchmark.args``
(NOT from kwargs and NOT from env vars). At runtime:

  * ``args.orgname`` is pinned by ``main._main_impl()``'s orgname-resolution
    gate, which reads ``orgname.yaml`` written by ``mlpstorage init``.
  * ``args.systemname`` is populated by argparse from ``--systemname`` (with
    an ``MLPSTORAGE_SYSTEMNAME`` env-var fallback).

This test file exercises:

  * the path prefix for CLOSED, OPEN, and ``whatif`` modes,
  * the typed ``ConfigurationError`` raised when ``args.orgname`` or
    ``args.systemname`` is missing or empty (with ``parameter`` field set
    for the dispatch layer to surface),
  * the assertion that the function does NOT consult ``os.environ`` for
    ``MLPSTORAGE_*`` — that is the CLI dispatch layer's job.
  * the module-level env-var-name constants
    ``MLPSTORAGE_ORGNAME_ENVVAR`` / ``MLPSTORAGE_SYSTEMNAME_ENVVAR``
    exported as the single source of truth for the dispatch helper.
"""

import types

import pytest

from mlpstorage_py.config import BENCHMARK_TYPES
from mlpstorage_py.errors import ConfigurationError


def _benchmark(mode: str, model: str = "unet3d", command: str = "datagen",
               benchmark_type=BENCHMARK_TYPES.training, results_dir: str = "/tmp/r",
               index_type: str | None = None, vdb_engine: str | None = None,
               orgname: str | None = "acme", systemname: str | None = "sys-1"):
    """Build a minimal benchmark stand-in with the attributes
    ``generate_output_location`` reads.

    ``index_type`` is set for vector_database benchmarks; the runtime path for
    that type includes a per-index_type segment so AISAQ results are kept
    separate from DISKANN/HNSW (they're not comparable). The on-disk index
    directory uses the UPPERCASE token (DISKANN / HNSW / AISAQ), matching
    ``args.index_type`` and ``summary.json.index_type``. ``vdb_engine`` adds
    the engine segment between <type> and <index>.

    Pass ``orgname=None`` or ``systemname=None`` to omit the attribute
    entirely (simulating an args Namespace built before the upstream
    orgname-resolution gate populates it). Pass an empty string to simulate
    a present-but-empty value.
    """
    args = types.SimpleNamespace(
        mode=mode,
        results_dir=results_dir,
        model=model,
        command=command,
    )
    if orgname is not None:
        args.orgname = orgname
    if systemname is not None:
        args.systemname = systemname
    if index_type is not None:
        args.index_type = index_type
    if vdb_engine is not None:
        args.vdb_engine = vdb_engine
    return types.SimpleNamespace(args=args, BENCHMARK_TYPE=benchmark_type)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

def test_envvar_constants_exported():
    """The module exports the two env-var-name constants for the dispatch
    helper to consume."""
    from mlpstorage_py.rules.utils import (
        MLPSTORAGE_ORGNAME_ENVVAR,
        MLPSTORAGE_SYSTEMNAME_ENVVAR,
    )

    assert MLPSTORAGE_ORGNAME_ENVVAR == "MLPSTORAGE_ORGNAME"
    assert MLPSTORAGE_SYSTEMNAME_ENVVAR == "MLPSTORAGE_SYSTEMNAME"


# ---------------------------------------------------------------------------
# CLOSED prefix
# ---------------------------------------------------------------------------

def test_closed_training_prefix():
    """CLOSED training path: {results_dir}/closed/<orgname>/results/<systemname>/training/<model>/<command>/<datetime>/."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="closed")
    path = generate_output_location(b, datetime_str="X")
    assert path.startswith("/tmp/r/closed/acme/results/sys-1/training/unet3d/datagen/"), path
    assert path.endswith("/X"), path


def test_closed_checkpointing_prefix():
    """CLOSED checkpointing path omits the <command> segment per LAY-05."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(
        mode="closed",
        model="llama3-8b",
        command="run",
        benchmark_type=BENCHMARK_TYPES.checkpointing,
    )
    path = generate_output_location(b, datetime_str="X")
    assert path.startswith("/tmp/r/closed/acme/results/sys-1/checkpointing/llama3-8b/"), path
    assert path.endswith("/X"), path


# ---------------------------------------------------------------------------
# OPEN prefix
# ---------------------------------------------------------------------------

def test_open_training_prefix():
    """OPEN training has the same shape as CLOSED — both modes thread through
    the orgname/results/systemname prefix."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="open")
    path = generate_output_location(b, datetime_str="X")
    assert path.startswith(
        "/tmp/r/open/acme/results/sys-1/training/unet3d/datagen/"
    ), path


def test_open_vector_database_prefix_includes_index_type():
    """vector_database results are split by engine/index_type because AISAQ
    results are not comparable to DISKANN/HNSW. The runtime path includes
    the <engine>/<index_type> segments between <type> and <command>.

    On-disk type segment is `vector_database` (BENCHMARK_TYPES.name) and the
    index directory is the UPPERCASE token (`DISKANN`), matching
    ``args.index_type`` and ``summary.json.index_type``."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(
        mode="open",
        command="run",
        benchmark_type=BENCHMARK_TYPES.vector_database,
        index_type="DISKANN",
        vdb_engine="milvus",
    )
    path = generate_output_location(b, datetime_str="X")
    assert path.startswith(
        "/tmp/r/open/acme/results/sys-1/vector_database/milvus/DISKANN/run/"
    ), path


def test_closed_vector_database_prefix_includes_index_type():
    """Same contract on the CLOSED side: <engine>/<index_type> sits between
    <type> and <command>.

    The type segment is `vector_database` and the index directory is the
    UPPERCASE token `AISAQ`, matching ``args.index_type``."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(
        mode="closed",
        command="run",
        benchmark_type=BENCHMARK_TYPES.vector_database,
        index_type="AISAQ",
        vdb_engine="milvus",
    )
    path = generate_output_location(b, datetime_str="X")
    assert path.startswith(
        "/tmp/r/closed/acme/results/sys-1/vector_database/milvus/AISAQ/run/"
    ), path


# ---------------------------------------------------------------------------
# whatif mode — uniform canonical shape (no special-case legacy)
# ---------------------------------------------------------------------------

def test_whatif_uses_canonical_shape():
    """`whatif` flows through the same orgname-resolution gate as `closed`/`open`,
    so the path has the same canonical shape — only the leading mode segment
    differs."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="whatif")
    path = generate_output_location(b, datetime_str="X")
    assert path.startswith(
        "/tmp/r/whatif/acme/results/sys-1/training/unet3d/datagen/"
    ), path
    assert path.endswith("/X"), path


# ---------------------------------------------------------------------------
# Typed-error trust contract: missing kwargs for closed/open modes
# ---------------------------------------------------------------------------

def test_missing_orgname_raises_configuration_error():
    """No args.orgname raises a typed ConfigurationError. The dispatch layer
    surfaces it as actionable user-facing text via `parameter` + suggestion."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="closed", orgname=None)
    with pytest.raises(ConfigurationError) as exc_info:
        generate_output_location(b, datetime_str="X")
    assert exc_info.value.parameter == "orgname"
    # Suggestion text points the user at the right remediation.
    assert "mlpstorage init" in str(exc_info.value)


def test_empty_orgname_raises_configuration_error():
    """An empty-string args.orgname is treated as missing (avoids producing
    a path with an empty segment)."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="closed", orgname="")
    with pytest.raises(ConfigurationError) as exc_info:
        generate_output_location(b, datetime_str="X")
    assert exc_info.value.parameter == "orgname"


def test_missing_systemname_raises_configuration_error():
    """With orgname present but systemname missing, the function raises a
    typed ConfigurationError identifying systemname as the missing parameter."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="open", systemname=None)
    with pytest.raises(ConfigurationError) as exc_info:
        generate_output_location(b, datetime_str="X")
    assert exc_info.value.parameter == "systemname"
    assert "MLPSTORAGE_SYSTEMNAME" in str(exc_info.value)


def test_orgname_reported_before_systemname():
    """When BOTH are missing, orgname is reported first (it is the outer
    segment in the path so the error name surfaces the first thing the
    user needs to set)."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="open", orgname=None, systemname=None)
    with pytest.raises(ConfigurationError) as exc_info:
        generate_output_location(b, datetime_str="X")
    assert exc_info.value.parameter == "orgname"


# ---------------------------------------------------------------------------
# Negative assertion: no os.environ reads for MLPSTORAGE_* names
# ---------------------------------------------------------------------------

def test_function_does_not_read_mlpstorage_env_vars(monkeypatch):
    """The function MUST NOT touch os.environ for MLPSTORAGE_* — that is the
    CLI dispatch layer's job. We assert by setting env vars to wrong values
    and confirming the path uses args.orgname/args.systemname instead."""
    monkeypatch.setenv("MLPSTORAGE_ORGNAME", "ENV-ORGNAME-WRONG")
    monkeypatch.setenv("MLPSTORAGE_SYSTEMNAME", "ENV-SYSTEMNAME-WRONG")

    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="closed", orgname="acme", systemname="sys-1")
    path = generate_output_location(b, datetime_str="X")
    # args wins; the env-var value never appears.
    assert "/closed/acme/results/sys-1/" in path, path
    assert "ENV-ORGNAME-WRONG" not in path, path
    assert "ENV-SYSTEMNAME-WRONG" not in path, path


# ---------------------------------------------------------------------------
# Path-component safety: reject path-traversal / unsafe segments at the
# trust boundary (defense in depth — argparse choices= covers the CLI
# entrypoint; this catches programmatic callers that bypass argparse).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_orgname", [
    "../etc",          # parent-dir traversal
    "..",              # reserved
    ".",               # reserved
    "/absolute",       # absolute reset (would clobber results_dir via os.path.join)
    "acme/sub",        # embedded separator
    "acme\x00",        # null byte
    "acme name",       # whitespace
])
def test_orgname_rejects_unsafe_path_components(bad_orgname):
    """args.orgname is pinned upstream from orgname.yaml — but a programmatic
    caller (test fixture, future internal API) that bypasses the sentinel
    schema validation must still hit the path-traversal guard here."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="closed", orgname=bad_orgname)
    with pytest.raises((ValueError, ConfigurationError)):
        generate_output_location(b, datetime_str="X")


@pytest.mark.parametrize("bad_systemname", ["../etc", "..", "/absolute", "sys/sub"])
def test_systemname_rejects_unsafe_path_components(bad_systemname):
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="open", systemname=bad_systemname)
    with pytest.raises(ValueError):
        generate_output_location(b, datetime_str="X")


@pytest.mark.parametrize("bad_index", ["../etc", "..", "/absolute", "DISKANN/sub"])
def test_vdb_index_rejects_unsafe_path_components(bad_index):
    """A programmatic caller (test fixture, future internal API) that
    bypasses cli.vectordb_args.validate_vectordb_arguments and feeds an
    arbitrary string as args.vdb_index must NOT land in a traversal path."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(
        mode="closed",
        command="run",
        benchmark_type=BENCHMARK_TYPES.vector_database,
        index_type=bad_index,
        vdb_engine="milvus",
    )
    with pytest.raises(ValueError):
        generate_output_location(b, datetime_str="X")


@pytest.mark.parametrize("bad_value", ["../bad", "..", "/abs", "a/b"])
def test_model_rejects_unsafe_path_components(bad_value):
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="closed", model=bad_value)
    with pytest.raises(ValueError):
        generate_output_location(b, datetime_str="X")


def test_datetime_str_rejects_unsafe_path_components():
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="closed")
    with pytest.raises(ValueError):
        generate_output_location(b, datetime_str="../escape")
