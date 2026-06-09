"""VectorDB benchmark CLI argument builder.

This module defines the CLI arguments for the VectorDB benchmark, including
datasize, datagen, and run commands.
"""

from mlpstorage_py.config import (
    DISTRIBUTIONS,
    SEARCH_METRICS,
    VECTOR_DTYPES,
    VECTORDB_DEFAULT_RUNTIME,
    VDB_BENCHMARK_MODES,
    VDB_INDEX_TYPES,
)
from mlpstorage_py.cli.common_args import (
    HELP_MESSAGES,
    add_mpi_arguments,
    add_storage_type_arguments,
    add_timeseries_arguments,
    add_universal_arguments,
)


VECTORDB_DISTRIBUTED_HELP_MESSAGES = {
    "distributed": (
        "Launch the VectorDB benchmark across one or more client hosts using MPI. "
        "When omitted, VectorDB keeps the existing single-node execution path."
    ),
    "hosts": (
        "Space-separated or comma-separated benchmark client hosts for MPI ranks. "
        "This is NOT the Milvus database host. The Milvus endpoint remains --host/-s. "
        "Examples: '--hosts node01 node02' or '--hosts=node01,node02'."
    ),
    "npernode": (
        "Number of VectorDB MPI ranks to start on each benchmark client host. "
        "Effective MPI world size is len(--hosts) * --npernode."
    ),
    "mpi_impl": (
        "MPI command dialect used by the VectorDB orchestrator. "
        "'mpich' builds an MPICH/Hydra-style launch command; "
        "'openmpi' builds an Open MPI-style launch command."
    ),
    "seed": (
        "Base random seed for rank-local VectorDB work. "
        "The effective seed is base seed + MPI rank."
    ),
    "ready_timeout": (
        "Timeout in seconds for rank synchronization markers during distributed "
        "VectorDB load/run orchestration."
    ),
}


def _add_vectordb_distributed_arguments(parser):
    """Add distributed execution arguments for VectorDB datagen/run.

    Important naming rule:
      * --host / -s is the Milvus database endpoint.
      * --hosts is the list of benchmark client hosts.

    Do not add a short -s alias to --hosts here because VectorDB already uses
    -s for --host.
    """
    distributed_group = parser.add_argument_group("Distributed Execution")

    distributed_group.add_argument(
        "--distributed",
        action="store_true",
        help=VECTORDB_DISTRIBUTED_HELP_MESSAGES["distributed"],
    )

    distributed_group.add_argument(
        "--hosts",
        nargs="+",
        default=None,
        help=VECTORDB_DISTRIBUTED_HELP_MESSAGES["hosts"],
    )

    distributed_group.add_argument(
        "--npernode",
        "--num-processes-per-client",
        dest="npernode",
        type=int,
        default=1,
        help=VECTORDB_DISTRIBUTED_HELP_MESSAGES["npernode"],
    )

    distributed_group.add_argument(
        "--mpi-impl",
        choices=["mpich", "openmpi"],
        default="mpich",
        help=VECTORDB_DISTRIBUTED_HELP_MESSAGES["mpi_impl"],
    )

    distributed_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help=VECTORDB_DISTRIBUTED_HELP_MESSAGES["seed"],
    )

    distributed_group.add_argument(
        "--ready-timeout",
        type=int,
        default=7200,
        help=VECTORDB_DISTRIBUTED_HELP_MESSAGES["ready_timeout"],
    )

    # Reuse common MPI options:
    #   --mpi-bin
    #   --oversubscribe
    #   --allow-run-as-root
    #   --mpi-btl
    #   --mpi-params
    add_mpi_arguments(parser)

    # Common MPI defaults to mpirun. VectorDB multi-node support is designed
    # around MPICH by default, so use mpiexec unless the user overrides it.
    parser.set_defaults(mpi_bin="mpiexec")


def add_vectordb_arguments(parser):
    """Add VectorDB benchmark arguments to the parser.

    Args:
        parser: Argparse subparser for the VectorDB benchmark.
    """
    vectordb_subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="sub_commands",
    )
    parser.required = True

    # ---- Subcommand parsers ----
    datasize = vectordb_subparsers.add_parser(
        "datasize",
        help="Calculate storage requirements for a VDB dataset",
    )
    datagen = vectordb_subparsers.add_parser(
        "datagen",
        help=HELP_MESSAGES["vdb_datagen"],
    )
    run_benchmark = vectordb_subparsers.add_parser(
        "run",
        help=HELP_MESSAGES["vdb_run"],
    )

    # ---- Common arguments for datagen and run ----
    for _parser in [datagen, run_benchmark]:
        _parser.add_argument(
            "--host",
            "-s",
            type=str,
            default="127.0.0.1",
            help=HELP_MESSAGES["db_ip_address"],
        )
        _parser.add_argument(
            "--port",
            "-p",
            type=int,
            default=19530,
            help=HELP_MESSAGES["db_port"],
        )
        _parser.add_argument(
            "--config",
            help="VectorDB benchmark config name or config file reference.",
        )
        _parser.add_argument(
            "--collection",
            type=str,
            help=HELP_MESSAGES["db_collection"],
        )

    # ---- Datasize arguments ----
    datasize.add_argument(
        "--dimension",
        type=int,
        default=1536,
        help=HELP_MESSAGES["dimension"],
    )
    datasize.add_argument(
        "--num-vectors",
        type=int,
        default=1_000_000,
        help=HELP_MESSAGES["num_vectors"],
    )
    datasize.add_argument(
        "--index-type",
        choices=VDB_INDEX_TYPES,
        default="DISKANN",
        help="Index type for storage estimation",
    )
    datasize.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help=HELP_MESSAGES["num_shards"],
    )
    datasize.add_argument(
        "--vector-dtype",
        choices=VECTOR_DTYPES,
        default="FLOAT_VECTOR",
        help=HELP_MESSAGES["vector_dtype"],
    )

    # ---- Datagen / load specific arguments ----
    datagen.add_argument(
        "--dimension",
        type=int,
        default=1536,
        help=HELP_MESSAGES["dimension"],
    )
    datagen.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help=HELP_MESSAGES["num_shards"],
    )
    datagen.add_argument(
        "--vector-dtype",
        choices=VECTOR_DTYPES,
        default="FLOAT_VECTOR",
        help=HELP_MESSAGES["vector_dtype"],
    )
    datagen.add_argument(
        "--num-vectors",
        type=int,
        default=1_000_000,
        help=HELP_MESSAGES["num_vectors"],
    )
    datagen.add_argument(
        "--distribution",
        choices=DISTRIBUTIONS,
        default="uniform",
        help=HELP_MESSAGES["distribution"],
    )
    datagen.add_argument(
        "--batch-size",
        type=int,
        default=1_000,
        help=HELP_MESSAGES["vdb_datagen_batch_size"],
    )
    datagen.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help=HELP_MESSAGES["vdb_datagen_chunk_size"],
    )
    datagen.add_argument(
        "--index-type",
        choices=VDB_INDEX_TYPES,
        default="DISKANN",
        help="Vector index type to create during load.",
    )
    datagen.add_argument(
        "--metric-type",
        choices=SEARCH_METRICS,
        default="COSINE",
        help="Vector search metric type for the created index.",
    )
    datagen.add_argument(
        "--max-degree",
        type=int,
        default=16,
        help="DiskANN MaxDegree parameter.",
    )
    datagen.add_argument(
        "--search-list-size",
        type=int,
        default=200,
        help="DiskANN SearchListSize parameter.",
    )
    datagen.add_argument(
        "--M",
        type=int,
        default=16,
        help="HNSW M parameter.",
    )
    datagen.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="HNSW efConstruction parameter.",
    )
    datagen.add_argument(
        "--inline-pq",
        type=int,
        default=16,
        help="AISAQ inline_pq parameter.",
    )
    datagen.add_argument(
        "--monitor-interval",
        type=int,
        default=5,
        help="Interval in seconds for monitoring index build progress.",
    )
    datagen.add_argument(
        "--compact",
        action="store_true",
        help="Perform collection compaction after loading.",
    )
    datagen.add_argument(
        "--force",
        action="store_true",
        help="Force recreate collection if it exists.",
    )

    # ---- Run specific arguments ----
    run_benchmark.add_argument(
        "--num-query-processes",
        type=int,
        default=1,
        help=HELP_MESSAGES["num_query_processes"],
    )
    run_benchmark.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=HELP_MESSAGES["query_batch_size"],
    )
    run_benchmark.add_argument(
        "--report-count",
        type=int,
        default=100,
        help=HELP_MESSAGES["vdb_report_count"],
    )
    run_benchmark.add_argument(
        "--mode",
        choices=VDB_BENCHMARK_MODES,
        default="timed",
        help=(
            "Benchmark mode: timed or query_count use simple_bench; "
            "sweep uses enhanced_bench."
        ),
    )

    # Common simple_bench search/recall knobs. These are optional and preserve
    # the existing defaults when not supplied.
    run_benchmark.add_argument(
        "--vector-dim",
        type=int,
        default=1536,
        help="Vector dimension used by query generation.",
    )
    run_benchmark.add_argument(
        "--search-limit",
        type=int,
        default=10,
        help="Number of nearest neighbors to request per query.",
    )
    run_benchmark.add_argument(
        "--search-ef",
        type=int,
        default=200,
        help="Search ef parameter for ANN query execution.",
    )
    run_benchmark.add_argument(
        "--gt-collection",
        type=str,
        default=None,
        help=(
            "Ground-truth FLAT collection name. "
            "Defaults to '<collection>_flat_gt' when omitted."
        ),
    )
    run_benchmark.add_argument(
        "--num-query-vectors",
        type=int,
        default=1000,
        help="Number of deterministic query vectors to pre-generate for recall.",
    )
    run_benchmark.add_argument(
        "--recall-k",
        type=int,
        default=None,
        help="K value for recall@k. Defaults to --search-limit.",
    )

    # End condition group for run.
    end_group = run_benchmark.add_argument_group(
        "Provide an end condition of runtime in seconds or total number of "
        "queries to execute. If neither is provided, the VectorDB config or "
        f"default runtime is used; default runtime is {VECTORDB_DEFAULT_RUNTIME} seconds."
    )
    end_condition = end_group.add_mutually_exclusive_group()
    end_condition.add_argument(
        "--runtime",
        type=int,
        help="Run for a specific duration in seconds.",
    )
    end_condition.add_argument(
        "--queries",
        type=int,
        help=(
            "Run for a specific number of queries. In distributed mode this is "
            "interpreted as the global query count and split across MPI ranks."
        ),
    )

    # Add distributed execution arguments to datagen and run only.
    _add_vectordb_distributed_arguments(datagen)
    _add_vectordb_distributed_arguments(run_benchmark)

    # Add universal arguments to all subcommands.
    for _parser in [datasize, datagen, run_benchmark]:
        add_universal_arguments(_parser)
        add_storage_type_arguments(_parser)

    # Add time-series arguments to run command only.
    add_timeseries_arguments(run_benchmark)
