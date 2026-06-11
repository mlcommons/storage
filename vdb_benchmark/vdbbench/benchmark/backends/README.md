# Vector Database Backends

This package provides a **pluggable backend system** for the VDB benchmark
framework. Every database adapter implements the same abstract interface
(`VectorDBBackend`), and the framework discovers and registers backends
automatically at import time -- no manual wiring required.

## Directory Layout

```
backends/
├── __init__.py          # BackendRegistry + auto-discovery
├── base.py              # Abstract VectorDBBackend + descriptor dataclasses
├── _env.py              # Environment variable loading for connection params
├── _help.py             # CLI help formatting utilities
├── elasticsearch/       # Elasticsearch adapter
│   ├── __init__.py      #   backend_descriptor() + exports
│   ├── backend.py       #   ElasticsearchBackend implementation
│   └── README.md        #   Elasticsearch-specific documentation
├── milvus/              # Milvus / Zilliz Cloud adapter
│   ├── __init__.py      #   backend_descriptor() + exports
│   ├── backend.py       #   MilvusBackend implementation
│   └── README.md        #   Milvus-specific documentation
└── pgvector/            # PostgreSQL + pgvector adapter
    ├── __init__.py      #   backend_descriptor() + exports
    ├── backend.py       #   PGVectorBackend implementation
    └── README.md        #   pgvector-specific documentation
```

## Abstract Interface

`VectorDBBackend` (defined in `base.py`) is the contract that every adapter
must satisfy. The benchmark orchestrator only calls methods on this interface,
so adding a new database requires **zero changes** to the generation,
ground-truth, or search pipelines.

### Method Reference

#### Lifecycle

| Method | Signature | Purpose |
|--------|-----------|---------|
| `connect` | `connect(self, **kwargs) -> None` | Open a connection. Keyword arguments come from the backend's `connection_params`. |
| `disconnect` | `disconnect(self) -> None` | Close the connection and release resources. |

#### Collection Management

| Method | Signature | Purpose |
|--------|-----------|---------|
| `create_collection` | `create_collection(self, name, dimension, metric_type="COSINE", index_type="HNSW", index_params=None, num_shards=1, force=False) -> CollectionInfo` | Create a collection (or drop + recreate when `force=True`) and build its index. |
| `collection_exists` | `collection_exists(self, name: str) -> bool` | Check whether a collection already exists. |
| `drop_collection` | `drop_collection(self, name: str) -> None` | Drop a collection if it exists. |

#### Data Ingestion

| Method | Signature | Purpose |
|--------|-----------|---------|
| `insert_batch` | `insert_batch(self, name, ids: np.ndarray, vectors: np.ndarray) -> int` | Insert a batch of vectors. `ids` is `(n,)` int64; `vectors` is `(n, dim)` float32. Returns the number of vectors inserted. |
| `flush` | `flush(self, name: str) -> None` | Commit pending writes to durable storage. |

#### Search

| Method | Signature | Purpose |
|--------|-----------|---------|
| `search` | `search(self, name, query_vectors: np.ndarray, top_k: int, search_params=None) -> List[List[int]]` | Run an ANN (or exact) search. Returns a list of `top_k` primary-key IDs per query, ordered closest-first. |

#### Status / Info

| Method | Signature | Purpose |
|--------|-----------|---------|
| `row_count` | `row_count(self, name: str) -> int` | Return the number of vectors currently in the collection. |
| `get_index_progress` | `get_index_progress(self, name: str) -> IndexProgress` | **(Abstract)** Return a point-in-time snapshot of the index build. Each backend fills in whatever fields it can (see `IndexProgress` below). |

#### Concrete (provided by base class)

| Method | Signature | Purpose |
|--------|-----------|---------|
| `wait_for_index` | `wait_for_index(self, name, interval=5.0, timeout=0, compacted=False) -> None` | Polls `get_index_progress()` in a loop with unified progress logging. When row counts are available (e.g. Milvus) it logs percentage, overall/recent rates, and ETA; otherwise it logs a simpler status line. Raises `TimeoutError` if `timeout > 0` is exceeded. **Do not override** -- implement `get_index_progress()` instead. |
| `compact` | `compact(self, name: str) -> None` | Trigger segment compaction. Default is a no-op; override if your backend needs it (e.g. Milvus). |

## Descriptor System

Every backend exposes a `BackendDescriptor` that tells the framework what the
backend supports. This descriptor drives:

- CLI `--help` output and argument validation
- Index type and metric validation before a run starts
- The `--plan` execution planner

### Descriptor Dataclasses

```python
@dataclass
class ParamDescriptor:
    name: str              # e.g. "M", "ef", "host"
    description: str       # shown in --help
    type: str = "int"      # "int" | "float" | "str" | "bool"
    default: Any = None
    required: bool = False

@dataclass
class IndexDescriptor:
    name: str              # e.g. "HNSW", "DISKANN"
    description: str
    build_params:  List[ParamDescriptor]   # used during create_collection
    search_params: List[ParamDescriptor]   # used during search

@dataclass
class BackendDescriptor:
    name: str                          # short key used in --backend flag
    display_name: str                  # human-readable name
    description: str                   # one-paragraph overview
    backend_class: Type[VectorDBBackend]
    supported_metrics:  List[str]              # e.g. ["COSINE", "L2", "IP"]
    supported_indexes:  List[IndexDescriptor]
    connection_params:  List[ParamDescriptor]
    active: bool = True                # set False to hide from CLI / help

@dataclass
class CollectionInfo:
    name: str
    dimension: int
    metric_type: str
    index_type: str
    row_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IndexProgress:
    """Snapshot of index-build progress returned by get_index_progress()."""
    is_ready: bool = False       # True when the build is complete
    total_rows: int = 0          # total rows to index (0 if unknown)
    indexed_rows: int = 0        # rows indexed so far
    pending_rows: int = 0        # rows waiting to be indexed
    status: str = ""             # free-form backend status (e.g. "yellow")
```

When `total_rows > 0` the base-class `wait_for_index()` logs detailed
progress:

```
Building index: 55.17% complete... (551,660/1,000,000 rows) | Pending rows: 681,000 | Overall rate: 227.28 rows/sec | Recent rate: 4065.85 rows/sec | ETA: 2026-03-31 17:45:23 | Est. remaining: 0:32:52
```

When only `status` is available (e.g. Elasticsearch health), a simpler
line is shown:

```
Waiting for index on 'my_collection' ... (status: yellow)  [5s elapsed]
```

## Auto-Discovery

Backend packages are discovered automatically when the `backends` package is
imported. The mechanism (in `__init__.py`) works as follows:

1. Walk every sub-directory of `backends/` that is a Python package.
2. Import the package.
3. Look for a module-level `backend_descriptor` attribute.
4. If it is callable, call it; otherwise use it directly.
5. If the result is a `BackendDescriptor`, register it in the global
   `registry`.
6. If import fails (missing dependency, etc.), log a warning and skip.

This means installing a new backend is as simple as dropping a package into
`backends/` -- the framework will pick it up on the next import.

## Existing Backends

| Backend | `--backend` name | Supported Indexes | Supported Metrics | Active | Required packages |
|---------|-------------------|-------------------|-------------------|--------|-------------------|
| Milvus | `milvus` | HNSW, DISKANN, AISAQ, FLAT | COSINE, L2, IP | Yes | `pymilvus` |
| pgvector | `pgvector` | HNSW, IVFFLAT, FLAT | COSINE, L2, IP | Yes | `psycopg2-binary`, `pgvector` |
| Elasticsearch | `elasticsearch` | HNSW, FLAT | COSINE, L2, IP | Yes | `elasticsearch` |

### Active vs Inactive Backends

A backend can be present in the source tree but hidden from users by setting
`active=False` in its `BackendDescriptor`. Inactive backends:

- Are **not** listed in `--help` or `help backends` output.
- Are **not** returned by `registry.names()`, `registry.list_backends()`,
  or `registry.get()`.
- **Cannot** be selected via `--backend` (the CLI will report "unknown
  backend").
- **Are** still registered internally and can be inspected via
  `registry.all_backends(include_inactive=True)`.

This is useful for backends that are under development or not yet ready for
general use. To activate a backend, simply change `active=False` to
`active=True` in its `backend_descriptor()` function.

## Environment Variable Configuration

Backend connection parameters can be set via **environment variables** or a
**`.env` file** instead of (or in addition to) CLI flags and YAML configs.

### Naming Convention

```
{BACKEND}__{PARAM}
```

Both parts are **upper-cased** and separated by a **double underscore** (`__`).
`PARAM` matches the `name` field of the backend's `connection_params`
descriptors.

| Backend | Example variables |
|---------|-------------------|
| Milvus | `MILVUS__HOST`, `MILVUS__PORT`, `MILVUS__MAX_MESSAGE_LENGTH` |
| pgvector | `PGVECTOR__HOST`, `PGVECTOR__PORT`, `PGVECTOR__DBNAME`, `PGVECTOR__USER`, `PGVECTOR__PASSWORD` |
| Elasticsearch | `ELASTICSEARCH__HOST`, `ELASTICSEARCH__API_KEY`, `ELASTICSEARCH__CLOUD_ID` |

### .env File

If the [`python-dotenv`](https://pypi.org/project/python-dotenv/) package
is installed, the benchmark CLI automatically loads a `.env` file from the
current working directory on startup. See `.env.example` in the benchmark
directory for a template.

```bash
pip install python-dotenv   # optional; enables .env file support
cp benchmark/.env.example .env
# edit .env with your values
```

When `python-dotenv` is not installed, only real shell environment variables
are read.

### Precedence

Connection parameters are resolved with the following precedence (highest
wins):

```
CLI flags  >  environment variables / .env  >  YAML config  >  built-in defaults
```

For example, if `MILVUS__HOST=10.0.0.5` is set in `.env` and
`host: 127.0.0.1` is in the YAML config, the env value `10.0.0.5` wins.
But `--host 192.168.1.1` on the CLI overrides both.

### Debugging

Use `--what-if` to see where each connection parameter came from:

```bash
python -m vdbbench.benchmark \
    --backend milvus --config configs/1m_hnsw.yaml --what-if
```

Output includes a "Connection parameters (source)" section showing each
parameter's resolved value and whether it came from CLI, env, YAML, or
default.

### Type Coercion

Environment variables are always strings. The framework automatically
coerces them to the type declared in `ParamDescriptor.type`:

| `type` | Conversion |
|--------|-----------|
| `"str"` | Used as-is |
| `"int"` | `int(value)` |
| `"float"` | `float(value)` |
| `"bool"` | `true` / `1` / `yes` / `on` → `True`; everything else → `False` |

Invalid conversions (e.g. `MILVUS__PORT=abc`) are logged as warnings and
skipped.

---

## Creating a New Backend

Follow these steps to add support for a new vector database.

### 1. Create the package directory

```
backends/
└── mydb/
    ├── __init__.py
    └── backend.py
```

### 2. Implement the backend class (`backend.py`)

Subclass `VectorDBBackend` and implement every abstract method:

```python
"""MyDB backend implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..base import CollectionInfo, IndexProgress, VectorDBBackend

logger = logging.getLogger(__name__)


class MyDBBackend(VectorDBBackend):
    """Concrete backend for MyDB."""

    def __init__(self) -> None:
        self._client = None

    # -- Lifecycle --------------------------------------------------------

    def connect(self, host: str = "127.0.0.1", port: str = "6333", **kwargs) -> None:
        from mydb_client import Client          # import here to keep it optional
        self._client = Client(host=host, port=int(port))
        logger.info("Connected to MyDB at %s:%s", host, port)

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        logger.info("Disconnected from MyDB")

    # -- Collection management --------------------------------------------

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric_type: str = "COSINE",
        index_type: str = "HNSW",
        index_params: Optional[Dict[str, Any]] = None,
        num_shards: int = 1,
        force: bool = False,
    ) -> CollectionInfo:
        if self.collection_exists(name):
            if force:
                self.drop_collection(name)
            else:
                raise ValueError(f"Collection '{name}' already exists")

        params = index_params or {}
        # ... create the collection and index using your DB client ...

        return CollectionInfo(
            name=name,
            dimension=dimension,
            metric_type=metric_type,
            index_type=index_type,
            row_count=0,
            extra={"index_params": params},
        )

    def collection_exists(self, name: str) -> bool:
        return self._client.has_collection(name)

    def drop_collection(self, name: str) -> None:
        if self.collection_exists(name):
            self._client.delete_collection(name)
            logger.info("Dropped collection '%s'", name)

    # -- Data ingestion ---------------------------------------------------

    def insert_batch(self, name: str, ids: np.ndarray, vectors: np.ndarray) -> int:
        # ids: (n,) int64, vectors: (n, dim) float32
        self._client.upsert(
            collection=name,
            ids=ids.tolist(),
            vectors=vectors.tolist(),
        )
        return len(ids)

    def flush(self, name: str) -> None:
        self._client.flush(collection=name)
        logger.info("Flushed '%s'", name)

    # -- Search -----------------------------------------------------------

    def search(
        self,
        name: str,
        query_vectors: np.ndarray,
        top_k: int,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[List[int]]:
        results = []
        for qvec in query_vectors:
            hits = self._client.search(
                collection=name,
                vector=qvec.tolist(),
                limit=top_k,
                **(search_params or {}),
            )
            results.append([hit.id for hit in hits])
        return results

    # -- Status -----------------------------------------------------------

    def row_count(self, name: str) -> int:
        return self._client.count(collection=name)

    def get_index_progress(self, name: str) -> IndexProgress:
        info = self._client.index_status(collection=name)
        return IndexProgress(
            is_ready=info.get("ready", False),
            total_rows=info.get("total", 0),
            indexed_rows=info.get("indexed", 0),
            pending_rows=info.get("pending", 0),
            status=info.get("state", ""),
        )

    # -- Optional overrides -----------------------------------------------

    def load_collection(self, name: str) -> None:
        """Load collection into memory (if your DB requires it)."""
        self._client.load(collection=name)
        logger.info("Loaded collection '%s' into memory", name)
```

**Guidelines:**

- Import your database client library **inside** `connect()` (not at
  module level). This keeps the dependency optional -- the framework can
  still import the package and show help text even when the client library
  is not installed.
- Always accept `**kwargs` in `connect()` so the framework can pass
  connection parameters defined in your descriptor.
- `search()` must return results sorted **closest-first**.
- `insert_batch()` receives NumPy arrays. Convert to lists or native types
  as needed by your client library.
- Implement `get_index_progress()` -- **not** `wait_for_index()`.  The
  base class owns the polling loop and all progress logging.  Your method
  just returns a single `IndexProgress` snapshot.  If your database has a
  synchronous index build (like pgvector), simply return
  `IndexProgress(is_ready=True)` once the index exists.

### 3. Write the descriptor (`__init__.py`)

The `__init__.py` must expose a `backend_descriptor` attribute -- either a
callable (function) that returns a `BackendDescriptor`, or a
`BackendDescriptor` instance directly.

```python
"""MyDB backend package."""

from ..base import BackendDescriptor, IndexDescriptor, ParamDescriptor
from .backend import MyDBBackend

__all__ = ["MyDBBackend", "backend_descriptor"]


def backend_descriptor() -> BackendDescriptor:
    """Return the capability descriptor for the MyDB backend."""
    return BackendDescriptor(
        name="mydb",                         # used in --backend mydb
        display_name="MyDB",                 # shown in CLI help
        description=(
            "A scalable vector database with support for HNSW "
            "and brute-force search. Requires the mydb-client "
            "Python package."
        ),
        backend_class=MyDBBackend,
        supported_metrics=["COSINE", "L2", "IP"],
        supported_indexes=[
            IndexDescriptor(
                name="HNSW",
                description="Graph-based approximate search.",
                build_params=[
                    ParamDescriptor(
                        name="M",
                        description="Max connections per node.",
                        type="int",
                        default=16,
                    ),
                    ParamDescriptor(
                        name="efConstruction",
                        description="Build-time search width.",
                        type="int",
                        default=200,
                    ),
                ],
                search_params=[
                    ParamDescriptor(
                        name="ef",
                        description="Query-time search width.",
                        type="int",
                        default=128,
                    ),
                ],
            ),
            IndexDescriptor(
                name="FLAT",
                description="Brute-force exact search.",
                build_params=[],
                search_params=[],
            ),
        ],
        connection_params=[
            ParamDescriptor(
                name="host",
                description="Server hostname or IP.",
                type="str",
                default="127.0.0.1",
            ),
            ParamDescriptor(
                name="port",
                description="Server port.",
                type="str",
                default="6333",
            ),
        ],
    )
```

**Key rules for the descriptor:**

- `name` must be a unique, lower-case identifier. This is used as the
  `--backend` CLI value.
- `supported_indexes` must list every index algorithm your backend
  supports. `build_params` describe the parameters passed to
  `create_collection(index_params=...)`. `search_params` describe the
  parameters passed to `search(search_params=...)`.
- `connection_params` should list every keyword accepted by your
  `connect()` method so the framework can generate the correct CLI flags.
- Set `active=False` to keep the backend in the tree but hidden from
  users. This is useful during development. Omit the field or set
  `active=True` (the default) to make it available.

### 4. Verify

No manual registration code is needed. Simply restart Python and the
auto-discovery will find your package:

```bash
# Confirm the backend is discovered
python -c "
from vdbbench.benchmark.backends import registry
print(registry.names())          # should include 'mydb'
print(registry.get('mydb'))      # should show your BackendDescriptor
"

# Check CLI help
python -m vdbbench.benchmark help backend mydb

# Run a benchmark
python -m vdbbench.benchmark \
    --backend mydb \
    --config configs/1m_hnsw.yaml \
    --mode both
```

### 5. Checklist

- [ ] `backend.py` subclasses `VectorDBBackend` and implements all abstract
      methods.
- [ ] `__init__.py` exposes a `backend_descriptor` callable returning a
      `BackendDescriptor`.
- [ ] Client library imported inside `connect()`, not at module top level.
- [ ] `connect()` accepts `**kwargs`.
- [ ] `create_collection()` respects the `force` flag (drop + recreate).
- [ ] `search()` returns IDs sorted closest-first.
- [ ] `get_index_progress()` returns an `IndexProgress` snapshot.
      `wait_for_index()` is provided by the base class -- do **not**
      override it.
- [ ] `supported_indexes` lists every index type the backend handles.
- [ ] `connection_params` matches the keyword arguments of `connect()`.
- [ ] The backend appears in `registry.names()` after import.
