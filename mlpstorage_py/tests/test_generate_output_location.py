"""Unit tests for ``generate_output_location`` and the orgname/systemname
keyword-only contract (Plan 02-01, Task 2).

Per 02-CONTEXT.md D-03 the runtime output path is restructured so results
land under ``{results_dir}/{closed|open}/<orgname>/...`` (with an additional
``results/<systemname>/`` segment for OPEN). Per the Gemini MEDIUM
trust-contract review (02-REVIEWS.md), ``generate_output_location`` does
NOT read environment variables — it accepts ``orgname`` and ``systemname``
as keyword-only parameters threaded by the CLI dispatch layer (Plan 02-02).

This test file exercises:

  * the new path prefix for CLOSED and OPEN,
  * the back-compat shape for ``whatif`` and any other non-{closed,open} mode,
  * the typed ``ConfigurationError`` raised when the kwargs are missing for
    closed/open modes (NOT a bare ``KeyError`` from a hidden env read),
  * the module-level env-var-name constants
    ``MLPSTORAGE_ORGNAME_ENVVAR`` / ``MLPSTORAGE_SYSTEMNAME_ENVVAR``
    exported for Plan 02-02's helper to consume as a single source of truth.
"""

import types

import pytest

from mlpstorage_py.config import BENCHMARK_TYPES
from mlpstorage_py.errors import ConfigurationError


def _benchmark(mode: str, model: str = "unet3d", command: str = "datagen",
               benchmark_type=BENCHMARK_TYPES.training, results_dir: str = "/tmp/r",
               index_type: str | None = None, vdb_engine: str | None = None):
    """Build a minimal benchmark stand-in with the attributes
    ``generate_output_location`` reads.

    ``index_type`` is set for vector_database benchmarks; the runtime path for
    that type includes a per-index_type segment so AISAQ results are kept
    separate from DISKANN/HNSW (they're not comparable). The on-disk index
    directory uses display-case spellings (DiskANN / HNSW / AiSAQ) routed via
    ``INDEX_TYPE_TOKEN_TO_DIR``, while ``args.index_type`` itself stays
    UPPERCASE (the CLI / summary.json form). ``vdb_engine`` adds the engine
    segment between <type> and <DisplayIndex>.
    """
    args = types.SimpleNamespace(
        mode=mode,
        results_dir=results_dir,
        model=model,
        command=command,
    )
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
    """CLOSED training/<model>/<command>/<datetime> sits under
    {results_dir}/closed/<orgname>/."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="closed")
    path = generate_output_location(b, datetime_str="X", orgname="acme")
    assert path.startswith("/tmp/r/closed/acme/training/unet3d/datagen/"), path
    assert path.endswith("/X"), path


def test_closed_checkpointing_prefix():
    """CLOSED checkpointing/<model>/<datetime> sits under
    {results_dir}/closed/<orgname>/."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(
        mode="closed",
        model="llama3-8b",
        command="run",
        benchmark_type=BENCHMARK_TYPES.checkpointing,
    )
    path = generate_output_location(b, datetime_str="X", orgname="acme")
    assert path.startswith("/tmp/r/closed/acme/checkpointing/llama3-8b/"), path
    assert path.endswith("/X"), path


# ---------------------------------------------------------------------------
# OPEN prefix
# ---------------------------------------------------------------------------

def test_open_training_prefix():
    """OPEN training prepends both closed/open-segment and
    results/<systemname>/ before the per-type tail."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="open")
    path = generate_output_location(
        b, datetime_str="X", orgname="acme", systemname="sys-1",
    )
    assert path.startswith(
        "/tmp/r/open/acme/results/sys-1/training/unet3d/datagen/"
    ), path


def test_open_vector_database_prefix_includes_index_type():
    """vector_database results are split by engine/index_type because AISAQ
    results are not comparable to DISKANN/HNSW. The runtime path includes
    the <engine>/<DisplayIndex> segments between <type> and <command>.

    On-disk type segment is `vector_database` (BENCHMARK_TYPES.name). The
    index directory uses display-case spellings (e.g. `DiskANN`) while
    ``args.index_type`` stays UPPERCASE."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(
        mode="open",
        command="run",
        benchmark_type=BENCHMARK_TYPES.vector_database,
        index_type="DISKANN",
        vdb_engine="milvus",
    )
    path = generate_output_location(
        b, datetime_str="X", orgname="acme", systemname="sys-1",
    )
    assert path.startswith(
        "/tmp/r/open/acme/results/sys-1/vector_database/milvus/DiskANN/run/"
    ), path


def test_closed_vector_database_prefix_includes_index_type():
    """Same contract on the CLOSED side: <engine>/<DisplayIndex> sits between
    <type> and <command>.

    The type segment is `vector_database` and the index directory is the
    display-case spelling `AiSAQ`; the CLI/summary.json token
    ``args.index_type`` stays UPPERCASE `AISAQ`."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(
        mode="closed",
        command="run",
        benchmark_type=BENCHMARK_TYPES.vector_database,
        index_type="AISAQ",
        vdb_engine="milvus",
    )
    path = generate_output_location(b, datetime_str="X", orgname="acme")
    assert path.startswith(
        "/tmp/r/closed/acme/vector_database/milvus/AiSAQ/run/"
    ), path


# ---------------------------------------------------------------------------
# Back-compat: whatif (and any non-{closed,open} mode) — unchanged shape
# ---------------------------------------------------------------------------

def test_whatif_has_no_closed_open_prefix():
    """Mode=whatif keeps the legacy shape — no closed/open segment,
    no orgname/systemname."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="whatif")
    path = generate_output_location(b, datetime_str="X")
    # No prefix segments appear.
    assert "/closed/" not in path, path
    assert "/open/" not in path, path
    assert "/acme/" not in path, path
    # Legacy shape preserved.
    assert path.startswith("/tmp/r/training/unet3d/datagen/"), path
    assert path.endswith("/X"), path


def test_missing_mode_attribute_keeps_legacy_shape():
    """If args.mode is missing entirely (older callers), the function
    returns the legacy shape and does not raise."""
    from mlpstorage_py.rules.utils import generate_output_location

    args = types.SimpleNamespace(results_dir="/tmp/r", model="unet3d", command="datagen")
    b = types.SimpleNamespace(args=args, BENCHMARK_TYPE=BENCHMARK_TYPES.training)
    path = generate_output_location(b, datetime_str="X")
    assert path == "/tmp/r/training/unet3d/datagen/X"


# ---------------------------------------------------------------------------
# Typed-error trust contract: missing kwargs for closed/open modes
# ---------------------------------------------------------------------------

def test_closed_missing_orgname_raises_configuration_error():
    """CLOSED without orgname raises a typed ConfigurationError that
    identifies the missing parameter."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="closed")
    with pytest.raises(ConfigurationError) as exc_info:
        generate_output_location(b, datetime_str="X")
    # The CLI dispatch layer can recover the parameter name to surface in
    # its own user-facing error.
    assert exc_info.value.parameter == "orgname"
    # And the suggestion text references the env-var name constant so the
    # user sees actionable guidance.
    assert "MLPSTORAGE_ORGNAME" in str(exc_info.value)


def test_closed_empty_orgname_raises_configuration_error():
    """An empty-string orgname is treated as missing (avoids producing
    a path with an empty path segment)."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="closed")
    with pytest.raises(ConfigurationError) as exc_info:
        generate_output_location(b, datetime_str="X", orgname="")
    assert exc_info.value.parameter == "orgname"


def test_open_missing_systemname_raises_configuration_error():
    """OPEN with orgname but no systemname raises a typed
    ConfigurationError that identifies systemname as the missing
    parameter."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="open")
    with pytest.raises(ConfigurationError) as exc_info:
        generate_output_location(b, datetime_str="X", orgname="acme")
    assert exc_info.value.parameter == "systemname"
    assert "MLPSTORAGE_SYSTEMNAME" in str(exc_info.value)


def test_open_missing_orgname_raises_configuration_error():
    """OPEN missing orgname is also a typed error — orgname is reported
    first because it is the outer segment in the path."""
    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="open")
    with pytest.raises(ConfigurationError) as exc_info:
        generate_output_location(b, datetime_str="X", systemname="sys-1")
    assert exc_info.value.parameter == "orgname"


# ---------------------------------------------------------------------------
# Negative assertion: no os.environ reads for MLPSTORAGE_* names
# ---------------------------------------------------------------------------

def test_function_does_not_read_mlpstorage_env_vars(monkeypatch):
    """The function MUST NOT touch os.environ for MLPSTORAGE_* — that is the
    CLI dispatch layer's job. We assert by patching the values to something
    that would produce a wrong path if the function read them; the function's
    explicit kwargs must win."""
    monkeypatch.setenv("MLPSTORAGE_ORGNAME", "ENV-ORGNAME-WRONG")
    monkeypatch.setenv("MLPSTORAGE_SYSTEMNAME", "ENV-SYSTEMNAME-WRONG")

    from mlpstorage_py.rules.utils import generate_output_location

    b = _benchmark(mode="closed")
    path = generate_output_location(b, datetime_str="X", orgname="acme")
    # Kwargs win: 'acme' appears, the env-var value does NOT.
    assert "/closed/acme/" in path, path
    assert "ENV-ORGNAME-WRONG" not in path, path
