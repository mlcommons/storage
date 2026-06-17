"""VectorDB benchmark CLI argument builder.

This module defines the CLI arguments for the VectorDB benchmark, including
datasize, datagen, and run commands.

Distributed VectorDB terminology:

  --host / -s
      Milvus / vector database endpoint host.

  --hosts
      Benchmark client hosts used for MPI ranks.

  --coordination filesystem
      Legacy distributed mode. Uses a shared results directory and marker files.

  --coordination mpi
      No-shared-filesystem mode. Uses mpi4py bcast/barrier/gather for
      synchronization and metric aggregation. Rank-local detailed files are
      written under --rank-output-dir on each node.
"""

from mlpstorage_py.config import (
    DISTRIBUTIONS,
    SEARCH_METRICS,
    VECTOR_DTYPES,
    VECTORDB_DEFAULT_RUNTIME,
    VDB_BENCHMARK_MODES,
    VDB_ENGINE_DEFAULT,
    VDB_ENGINES,
    VDB_INDEX_TYPES,
    VDB_INDEX_TYPES_CLOSED,
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
        "Launch the VectorDB benchmark across one or more benchmark client "
        "hosts using MPI. When omitted, VectorDB keeps the existing single-node "
        "execution path."
    ),
    "hosts": (
        "Space-separated or comma-separated benchmark client hosts for MPI "
        "ranks. This is NOT the Milvus database host. The Milvus endpoint "
        "remains --host/-s. Examples: '--hosts node01 node02' or "
        "'--hosts=node01,node02'."
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
        "Timeout in seconds for rank synchronization. In filesystem "
        "coordination mode this controls marker-file waits. In MPI "
        "coordination mode it is kept for CLI compatibility."
    ),
    "coordination": (
        "Distributed coordination backend. 'filesystem' uses the legacy shared "
        "results directory and marker-file workflow. 'mpi' uses mpi4py "
        "bcast/barrier/gather and does not require a shared filesystem."
    ),
    "rank_output_dir": (
        "Node-local directory used by each MPI rank when --coordination mpi is "
        "selected. This directory does not need to be shared across nodes. "
        "Rank-local simple/enhanced detailed outputs are written here."
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
        "--coordination",
        choices=["filesystem", "mpi"],
        default="filesystem",
        help=VECTORDB_DISTRIBUTED_HELP_MESSAGES["coordination"],
    )

    distributed_group.add_argument(
        "--rank-output-dir",
        type=str,
        default="/tmp/mlps_vdb",
        help=VECTORDB_DISTRIBUTED_HELP_MESSAGES["rank_output_dir"],
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

    # Reuse common MPI options (--mpi-bin, --oversubscribe, --allow-run-as-root,
    # --mpi-btl, --mpi-params). VectorDB multi-node support was first designed
    # around MPICH, so keep mpiexec as the default. Users running Open MPI
    # should pass: --mpi-impl openmpi --mpi-bin mpirun.
    add_mpi_arguments(parser)
    parser.set_defaults(mpi_bin="mpiexec")


def add_vectordb_arguments(parser, mode):
    """Add VectorDB benchmark arguments to the parser.

    Args:
        parser: Argparse subparser for the VectorDB benchmark.
        mode: One of 'closed', 'open', or 'whatif'.
    """
    index_choices = VDB_INDEX_TYPES_CLOSED if mode == "closed" else VDB_INDEX_TYPES

    vectordb_subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="sub_commands",
    )
    parser.required = True

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

    for cmd_name, cmd_parser in [("datasize", datasize), ("datagen", datagen), ("run", run_benchmark)]:
        _add_vectordb_core_args(cmd_parser, cmd_name, index_choices)
        if mode in ("open", "whatif"):
            _add_vectordb_open_args(cmd_parser, cmd_name)
        if cmd_name in ("datagen", "run"):
            _add_vectordb_distributed_arguments(cmd_parser)


def _add_vectordb_core_args(parser, command, index_choices):
    """Add core VectorDB arguments shared across all modes.

    Args:
        parser: Argparse parser to add arguments to.
        command: The subcommand name ('datasize', 'datagen', or 'run').
        index_choices: Allowed index type choices based on mode.
    """
    # Set defaults for open-gated attrs so they always exist in the namespace.
    parser.set_defaults(loops=1, params='', allow_invalid_params=False)

    # The engine identifies which vector database implementation is being
    # benchmarked (milvus today). It is recorded in the results-dir path
    # and metadata so accumulated runs from different engines coexist.
    parser.add_argument(
        '--vdb-engine',
        choices=VDB_ENGINES,
        default=VDB_ENGINE_DEFAULT,
        help=(
            "Vector database engine being benchmarked. "
            "Recorded in the result path so multiple engines can accumulate "
            "in one --results-dir without collision."
        ),
    )

    # ---- Common args for datagen and run ----
    if command in ("datagen", "run"):
        parser.add_argument(
            '--host', '-s',
            type=str,
            default="127.0.0.1",
            help=HELP_MESSAGES["db_ip_address"],
        )
        parser.add_argument(
            '--port', '-p',
            type=int,
            default=19530,
            help=HELP_MESSAGES["db_port"],
        )
        parser.add_argument(
            '--config',
            help="VectorDB benchmark config name or config file reference.",
        )
        parser.add_argument(
            '--collection',
            type=str,
            help=HELP_MESSAGES["db_collection"],
        )

    # ---- Datasize args ----
    if command == "datasize":
        parser.add_argument(
            '--dimension',
            type=int,
            default=1536,
            help=HELP_MESSAGES['dimension'],
        )
        parser.add_argument(
            '--num-vectors',
            type=int,
            default=1_000_000,
            help=HELP_MESSAGES['num_vectors'],
        )
        parser.add_argument(
            '--index-type',
            choices=index_choices,
            default="DISKANN",
            help="Index type for storage estimation",
        )
        parser.add_argument(
            '--num-shards',
            type=int,
            default=1,
            help=HELP_MESSAGES['num_shards'],
        )
        parser.add_argument(
            '--vector-dtype',
            choices=VECTOR_DTYPES,
            default="FLOAT_VECTOR",
            help=HELP_MESSAGES['vector_dtype'],
        )

    # ---- Datagen args ----
    if command == "datagen":
        parser.add_argument(
            '--dimension',
            type=int,
            default=1536,
            help=HELP_MESSAGES['dimension'],
        )
        parser.add_argument(
            '--num-shards',
            type=int,
            default=1,
            help=HELP_MESSAGES['num_shards'],
        )
        parser.add_argument(
            '--vector-dtype',
            choices=VECTOR_DTYPES,
            default="FLOAT_VECTOR",
            help=HELP_MESSAGES['vector_dtype'],
        )
        parser.add_argument(
            '--num-vectors',
            type=int,
            default=1_000_000,
            help=HELP_MESSAGES['num_vectors'],
        )
        parser.add_argument(
            '--distribution',
            choices=DISTRIBUTIONS,
            default="uniform",
            help=HELP_MESSAGES['distribution'],
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=1_000,
            help=HELP_MESSAGES['vdb_datagen_batch_size'],
        )
        parser.add_argument(
            '--chunk-size',
            type=int,
            default=10_000,
            help=HELP_MESSAGES['vdb_datagen_chunk_size'],
        )
        parser.add_argument(
            '--index-type',
            choices=index_choices,
            default="DISKANN",
            help="Vector index type to create during load.",
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help="Force recreate collection if it exists",
        )

    # ---- Run args ----
    if command == "run":
        parser.add_argument(
            '--num-query-processes',
            type=int,
            default=1,
            help=HELP_MESSAGES['num_query_processes'],
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=1,
            help=HELP_MESSAGES['query_batch_size'],
        )
        parser.add_argument(
            '--report-count',
            type=int,
            default=100,
            help=HELP_MESSAGES['vdb_report_count'],
        )
        parser.add_argument(
            '--benchmark-mode',
            dest='benchmark_mode',
            choices=VDB_BENCHMARK_MODES,
            default='timed',
            help=(
                "Benchmark mode: timed or query_count use simple_bench; "
                "sweep uses enhanced_bench."
            ),
        )
        parser.add_argument(
            '--vector-dim',
            type=int,
            default=1536,
            help="Vector dimension used by query generation.",
        )
        parser.add_argument(
            '--search-limit',
            type=int,
            default=10,
            help="Number of nearest neighbors to request per query.",
        )
        parser.add_argument(
            '--search-ef',
            type=int,
            default=200,
            help="Search ef parameter for ANN query execution.",
        )
        parser.add_argument(
            '--gt-collection',
            type=str,
            default=None,
            help=(
                "Ground-truth FLAT collection name. "
                "Defaults to '<collection>_flat_gt' when omitted."
            ),
        )
        parser.add_argument(
            '--num-query-vectors',
            type=int,
            default=1000,
            help="Number of deterministic query vectors to pre-generate for recall.",
        )
        parser.add_argument(
            '--recall-k',
            type=int,
            default=None,
            help="K value for recall@k. Defaults to --search-limit.",
        )

        end_group = parser.add_argument_group(
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

    add_universal_arguments(parser, req_results=(command in ("datagen", "run")))

    if command in ("datagen", "run"):
        add_storage_type_arguments(parser, required=True)

    if command == "run":
        add_timeseries_arguments(parser)


def _add_vectordb_open_args(parser, command):
    """Add open/whatif-only VectorDB arguments.

    Args:
        parser: Argparse parser to add arguments to.
        command: The subcommand name.
    """
    parser.add_argument(
        '--loops',
        type=int,
        default=1,
        help="Number of times to repeat the benchmark run",
    )
    parser.add_argument(
        '--allow-invalid-params', '-aip',
        action='store_true',
        help="Allow parameters that would otherwise be flagged as invalid",
    )
    parser.add_argument(
        '--params',
        nargs="+",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=HELP_MESSAGES['params'],
    )

    if command == "datagen":
        # Advanced index-build knobs are open-gated. Closed mode pins these to
        # the underlying tool's defaults; open/whatif may tune them.
        parser.add_argument(
            '--metric-type',
            choices=SEARCH_METRICS,
            default="COSINE",
            help="Vector search metric type for the created index.",
        )
        # DiskANN parameters.
        parser.add_argument(
            '--max-degree',
            type=int,
            default=16,
            help="DiskANN MaxDegree parameter.",
        )
        parser.add_argument(
            '--search-list-size',
            type=int,
            default=200,
            help="DiskANN SearchListSize parameter.",
        )
        # HNSW parameters.
        parser.add_argument(
            '--M',
            type=int,
            default=16,
            help="HNSW M parameter.",
        )
        parser.add_argument(
            '--ef-construction',
            type=int,
            default=200,
            help="HNSW efConstruction parameter.",
        )
        # AISAQ parameters.
        parser.add_argument(
            '--inline-pq',
            type=int,
            default=16,
            help="AISAQ inline_pq parameter.",
        )
        parser.add_argument(
            '--monitor-interval',
            type=int,
            default=5,
            help="Interval in seconds for monitoring index build progress.",
        )
        parser.add_argument(
            '--compact',
            action='store_true',
            help="Perform collection compaction after loading.",
        )


def validate_vectordb_arguments(args):
    """Validate the whole set of args given that we're doing a vectordb benchmark.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
