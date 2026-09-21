"""
Help text constants and context-sensitive help for MLPerf Storage CLI.

Exports:
    HELP_ALL_TEXT: verbatim full command reference (from plans/help_all_spec.md)
    get_context_help_tokens: returns "next: X | Y | Z" or None for a positional token path
"""

# ---------------------------------------------------------------------------
# HELP_ALL_TEXT
# Verbatim content extracted from plans/help_all_spec.md code fences.
# Section headers between blocks are included as plain text (no markdown).
# ---------------------------------------------------------------------------

_HEADER_TEXT = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MLPSTORAGE — COMPLETE COMMAND REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

SYNOPSIS_TEXT = """\
SYNOPSIS
  mlpstorage <closed|open|whatif> training <model> <command> <file|object> [OPTIONS]
  mlpstorage <closed|open|whatif> checkpointing <command> <file|object> [OPTIONS]
  mlpstorage <closed|open|whatif> vectordb <command> <file|object> [OPTIONS]
  mlpstorage <closed|open|whatif> kvcache <command> [OPTIONS]
  mlpstorage (reports|history|runs|lockfile|config|version) [subcommand] [OPTIONS]
  mlpstorage init <orgname> [results-dir]
  mlpstorage validate <submission-dir> [OPTIONS]
  mlpstorage rules-coverage [--rules-md PATH]

  <closed|open|whatif>  — required first positional for benchmark commands
  <model>               — training only: required model positional
                          (choices vary by mode; see the tree below).
                          The other benchmarks select models via flags:
                          checkpointing --model/-m (required),
                          vectordb --vdb-index, kvcache --model/-m
                          (open|whatif only; fixed in closed)
  <file|object>         — required storage selector for commands that touch storage
                          (absent on datasize; absent on all kvcache commands)"""

_TREE_AND_BODY_TEXT = """\
mlpstorage
│
├── closed ──────────────────────────────────────────────────────
│   ├── training
│   │   └── unet3d | retinanet            ← model positional
│   │       ├── datasize                          {TR_DATASIZE_CLOSED}
│   │       ├── datagen    file | object           {TR_DATAGEN_CLOSED}
│   │       ├── run        file | object           {TR_RUN_CLOSED}
│   │       └── configview file | object           {TR_CONFIGVIEW_CLOSED}
│   │
│   ├── checkpointing                     ← model via --model/-m flag (required)
│   │   ├── datasize                              {CK_DATASIZE_CLOSED}
│   │   ├── run        file | object               {CK_RUN_CLOSED}
│   │   └── configview file | object               {CK_CONFIGVIEW_CLOSED}
│   │
│   ├── vectordb                          ← index via --vdb-index flag
│   │   ├── datasize                              {VDB_DATASIZE_CLOSED}
│   │   ├── datagen    file | object               {VDB_DATAGEN_CLOSED}
│   │   └── run        file | object               {VDB_RUN_CLOSED}
│   │
│   └── kvcache                           ← model fixed in closed
│       ├── datasize                              {KV_DATASIZE_CLOSED}
│       └── run                                   {KV_RUN_CLOSED}
│
├── open ────────────────────────────────────────────────────────
│   ├── training
│   │   └── unet3d | retinanet            ← model positional
│   │       ├── datasize                          {TR_DATASIZE_OPEN}
│   │       ├── datagen    file | object           {TR_DATAGEN_OPEN}
│   │       ├── run        file | object           {TR_RUN_OPEN}
│   │       └── configview file | object           {TR_CONFIGVIEW_OPEN}
│   │
│   ├── checkpointing                     ← model via --model/-m flag (required)
│   │   ├── datasize                              {CK_DATASIZE_OPEN}
│   │   ├── run        file | object               {CK_RUN_OPEN}
│   │   └── configview file | object               {CK_CONFIGVIEW_OPEN}
│   │
│   ├── vectordb                          ← index via --vdb-index flag
│   │   ├── datasize                              {VDB_DATASIZE_OPEN}
│   │   ├── datagen    file | object               {VDB_DATAGEN_OPEN}
│   │   └── run        file | object               {VDB_RUN_OPEN}
│   │
│   └── kvcache                           ← model via --model/-m flag (default: tiny-1b)
│       ├── datasize                              {KV_DATASIZE_OPEN}
│       └── run                                   {KV_RUN_OPEN}
│
├── whatif ──────────────────────────────────────────────────────
│   ├── training
│   │   └── cosmoflow | resnet50 | unet3d | dlrm | retinanet | flux
│   │       ├── datasize                          {TR_DATASIZE_WHATIF}
│   │       ├── datagen    file | object           {TR_DATAGEN_WHATIF}
│   │       ├── run        file | object           {TR_RUN_WHATIF}
│   │       └── configview file | object           {TR_CONFIGVIEW_WHATIF}
│   │
│   ├── checkpointing                     ← model via --model/-m flag (required)
│   │   ├── datasize                              {CK_DATASIZE_WHATIF}
│   │   ├── run        file | object               {CK_RUN_WHATIF}
│   │   └── configview file | object               {CK_CONFIGVIEW_WHATIF}
│   │
│   ├── vectordb                          ← index via --vdb-index flag
│   │   ├── datasize                              {VDB_DATASIZE_WHATIF}
│   │   ├── datagen    file | object               {VDB_DATAGEN_WHATIF}
│   │   └── run        file | object               {VDB_RUN_WHATIF}
│   │
│   └── kvcache                           ← model via --model/-m flag (default: tiny-1b)
│       ├── datasize                              {KV_DATASIZE_WHATIF}
│       └── run                                   {KV_RUN_WHATIF}
│
├── reports
│   └── reportgen                                {RP_REPORTGEN}
│
├── history
│   ├── show                                     {HI_SHOW}
│   └── rerun <id>                               {HI_RERUN}
│
├── runs                                  ← manage the run leaves of a results-dir
│   ├── list                                     {RUNS_LIST}
│   ├── show <id>                                {RUNS_SHOW}
│   ├── rm [<id>...]                             {RUNS_RM}
│   ├── purge                                    {RUNS_PURGE}
│   └── gc                                       {RUNS_GC}
│
├── lockfile
│   ├── generate                                 {LF_GENERATE}
│   └── verify                                   {LF_VERIFY}
│
├── init <orgname> [results-dir]                Pin orgname to a results-dir (default ~/mlpstorage-results)
│                                                and record it as the default for every later command
│
├── config                                ← manage the per-user config file (~/.config/mlpstorage/config.yaml)
│   ├── show                                     {CONFIG_SHOW}
│   ├── set <key> <value>...                     {CONFIG_SET}
│   ├── unset <key>                              {CONFIG_UNSET}
│   └── path                                     {CONFIG_PATH}
│
├── validate <submission-dir>                    {VALIDATE}
│
├── rules-coverage                               {RULES_COVERAGE}
│
└── version                                      {VERSION}

Common argument groups

CORE_STD — Standard arguments, every benchmark command and most utilities
  Every flag below (and every flag marked "layered" elsewhere) is filled in
  one order: typed flag > --config-file YAML > MLPSTORAGE_* env var >
  ~/.config/mlpstorage/config.yaml > built-in default. The per-user file may
  carry only environment keys: results_dir, systemname, data_dir,
  checkpoint_folder, hosts, mpi_bin, mpi_btl, oversubscribe,
  allow_run_as_root, mpi_params, dlio_bin_path, exec_type, color,
  stream_log_level. main prints which file supplied what.
  --results-dir/-rd PATH        Benchmark results directory
                                (layered: this flag > --config-file > MLPSTORAGE_RESULTS_DIR
                                 > the results-dir recorded by `mlpstorage init`;
                                 no tempdir fallback)
  --systemname/-sn NAME         System-under-test name — folder under results/
                                (layered: MLPSTORAGE_SYSTEMNAME env var, config.yaml)
  --config-file/-c PATH         YAML of flag values for this invocation; fills
                                what was not typed, may name any flag of the leaf
  --debug                       Enable debug output
  --verbose                     Enable verbose output
  --stream-log-level LEVEL      Logging level (default: INFO)
  --quiet                       Suppress the run configuration summary table
  --color {auto,always,never}   Colour the log level tag (default: auto — only when
                                stderr is a terminal; honours NO_COLOR)
  --dry-run                     Print the command that would execute; do not run
  --verify-lockfile PATH        Validate installed packages against lockfile
  --skip-validation             Skip MPI/SSH/DLIO pre-run environment checks
  --skip-ssh-check              Skip only the SSH connectivity preflight (for
                                scheduler-launched runs: PALS mpiexec, srun)
  --skip-fs-separation-gate     Bypass the CAP-03 same-filesystem hard gate
                                (probe still runs; Rules.md 3.4.2/4.4.2/5.4.2
                                still fail at validation time)

OPEN_STD — Additional standard arguments, open and whatif modes
            (all commands except kvcache datasize)
  --loops N                     Repeat benchmark N times (default: 1)
  --allow-invalid-params/-aip   Do not abort on invalid DLIO parameters

MPI_ARGS — MPI execution arguments
  --mpi-bin {mpirun,mpiexec}    MPI launcher binary (default: mpirun)
  --mpi-btl {auto,vader,tcp}    Byte Transport Layer for single-host runs (default: auto)
  --oversubscribe               Allow more ranks than CPU slots
  --allow-run-as-root           Permit execution as root (OpenMPI flag)
  --mpi-params PARAM...         Additional raw MPI parameters (repeatable)

TIMESERIES — Time-series host metrics, open and whatif run commands only
             (a closed run always collects at the defaults below and
              cannot opt out)
  --timeseries-interval SECS    Sample interval in seconds (default: 10.0)
  --skip-timeseries             Disable time-series collection entirely
  --max-timeseries-samples N    Per-host sample cap (default: 3600)

Placeholder definitions — TRAINING

TR_DATASIZE_CLOSED
  Required:
    --max-accelerators/-ma N
    --accelerator-type/-at {b200,mi355}
    --client-host-memory-in-gb/-cm N
    --systemname/-sn NAME           (layered; see CORE_STD)
  Optional:
    --data-dir/-dd PATH
    --num-client-hosts/-nc N        Derived from --hosts count if unset
    --dlio-bin-path/-dp PATH
    --exec-type/-et {mpi,docker}    (default: mpi)
    --hosts/-s HOST...              (default: 127.0.0.1)
    --params/-p/--param KEY=VALUE...  DLIO overrides (CLOSED: restricted subset)
  + MPI_ARGS
  + CORE_STD  (--results-dir layered: flag > --config-file > env > init default)

TR_DATASIZE_OPEN
  = TR_DATASIZE_CLOSED  (flags identical; --params unrestricted)
  + OPEN_STD

TR_DATASIZE_WHATIF
  = TR_DATASIZE_OPEN but:
    --accelerator-type choices: {h100,a100,b200,mi355}

──────────────────────────────────────────────────────────────────

TR_DATAGEN_CLOSED
  Required:
    --num-processes/-np N
    --systemname/-sn NAME           (layered; see CORE_STD)
    --data-dir/-dd PATH             (required with file storage; object mode
                                     may supply data_dir via --config-file)
    [storage positional: file | object]
  Optional:
    --dlio-bin-path/-dp PATH
    --exec-type/-et {mpi,docker}    (default: mpi)
    --hosts/-s HOST...              (default: 127.0.0.1)
    --o-direct                      Route I/O through s3dlio's O_DIRECT local-fs mode
    --params/-p/--param KEY=VALUE...  DLIO overrides (CLOSED: restricted subset)
  + MPI_ARGS
  + CORE_STD  (--results-dir layered: flag > --config-file > env > init default)

TR_DATAGEN_OPEN
  = TR_DATAGEN_CLOSED  (flags identical; --params unrestricted)
  + OPEN_STD

TR_DATAGEN_WHATIF
  = TR_DATAGEN_OPEN  (model positional choices differ; flags identical)

──────────────────────────────────────────────────────────────────

TR_RUN_CLOSED
  Required:
    --num-accelerators/-na N
    --accelerator-type/-at {b200,mi355}
    --client-host-memory-in-gb/-cm N
    --results-dir/-rd PATH          (layered; see CORE_STD)
    --systemname/-sn NAME           (layered; see CORE_STD)
    --data-dir/-dd PATH             (required with file storage; object mode
                                     may supply data_dir via --config-file)
    [storage positional: file | object]
  Optional:
    --num-client-hosts/-nc N
    --dlio-bin-path/-dp PATH
    --exec-type/-et {mpi,docker}    (default: mpi)
    --hosts/-s HOST...              (default: 127.0.0.1)
    --o-direct                      Route I/O through s3dlio's O_DIRECT local-fs mode
    --drop-caches-timeout-seconds N   Per-call timeout for the per-epoch
                                      page-cache flush
    --params/-p/--param KEY=VALUE...  DLIO overrides (CLOSED: restricted subset)
  + MPI_ARGS
  + CORE_STD

TR_RUN_OPEN
  = TR_RUN_CLOSED  (flags identical; --params unrestricted)
  + OPEN_STD
  + TIMESERIES

TR_RUN_WHATIF
  = TR_RUN_OPEN but:
    --accelerator-type choices: {h100,a100,b200,mi355}

──────────────────────────────────────────────────────────────────

TR_CONFIGVIEW_CLOSED
  Required:
    --num-accelerators/-na N
    --accelerator-type/-at {b200,mi355}
    --client-host-memory-in-gb/-cm N
    --results-dir/-rd PATH          (layered; see CORE_STD)
    --systemname/-sn NAME           (layered; see CORE_STD)
    [storage positional: file | object]
  Optional:
    --data-dir/-dd PATH
    --num-client-hosts/-nc N
    --dlio-bin-path/-dp PATH
    --exec-type/-et {mpi,docker}    (default: mpi)
    --hosts/-s HOST...              (default: 127.0.0.1)
    --o-direct
    --params/-p/--param KEY=VALUE...
  + MPI_ARGS
  + CORE_STD

TR_CONFIGVIEW_OPEN
  = TR_CONFIGVIEW_CLOSED  (flags identical; --params unrestricted)
  + OPEN_STD

TR_CONFIGVIEW_WHATIF
  = TR_CONFIGVIEW_OPEN but:
    --accelerator-type choices: {h100,a100,b200,mi355}

Placeholder definitions — CHECKPOINTING

CK_DATASIZE_CLOSED
  Required:
    --model/-m {llama3-8b,llama3-70b,llama3-405b,llama3-1t}
    --num-processes/-np N
    --client-host-memory-in-gb/-cm N
  Optional:
    --hosts/-s HOST...              (default: 127.0.0.1)
    --exec-type/-et {mpi,docker}    (default: mpi)
    --dlio-bin-path/-dp PATH
    --num-checkpoints-read/-ncr N   (default: 10; closed allows 10 or 0)
    --num-checkpoints-write/-ncw N  (default: 10; closed allows 10 or 0)
  + MPI_ARGS
  + CORE_STD  (--results-dir layered: flag > --config-file > env > init default; --systemname optional)
  Note: closed runs use 10/10 by default. Use 10/0 then 0/10 in two
        invocations when a cache flush is required between phases
        (see Rules.md §4.7.1 and checkpointing/README.md).

CK_DATASIZE_OPEN
  = CK_DATASIZE_CLOSED plus:
    --params/-p KEY=VALUE...        DLIO parameter overrides (repeatable)
  + OPEN_STD
  Note: open allows any non-negative integer for --num-checkpoints-read/-write

CK_DATASIZE_WHATIF
  = CK_DATASIZE_OPEN  (--model choices identical; flags identical)

──────────────────────────────────────────────────────────────────

CK_RUN_CLOSED
  Required:
    --model/-m {llama3-8b,llama3-70b,llama3-405b,llama3-1t}
    --num-processes/-np N
    --accelerator-type/-at {b200,mi355}
    --checkpoint-folder/-cf PATH
    --client-host-memory-in-gb/-cm N
    --results-dir/-rd PATH          (layered; see CORE_STD)
    --systemname/-sn NAME           (layered; see CORE_STD)
    [storage positional: file | object]
  Optional:
    --checkpoint-subset             (8B at 8 processes only; declares a Subset run)
    --exec-type/-et {mpi,docker}    (default: mpi)
    --hosts/-s HOST...              (default: 127.0.0.1)
    --dlio-bin-path/-dp PATH
    --num-checkpoints-read/-ncr N   (default: 10; closed allows 10 or 0)
    --num-checkpoints-write/-ncw N  (default: 10; closed allows 10 or 0)
    --o-direct                      Route I/O through s3dlio's O_DIRECT local-fs mode
  + MPI_ARGS
  + CORE_STD
  Note: closed runs use 10/10 by default. Use 10/0 then 0/10 in two
        invocations when a cache flush is required between phases
        (see Rules.md §4.7.1 and checkpointing/README.md).

  Closed rank constraints by model:
    llama3-1t:   8 or 1024
    llama3-405b: 8 or 512
    llama3-70b:  8 or 64
    llama3-8b:   8

CK_RUN_OPEN
  = CK_RUN_CLOSED plus:
    --params/-p KEY=VALUE...
  + OPEN_STD
  + TIMESERIES
  Note: open allows any multiple of the per-model GPU-per-DP-instance count
        and any non-negative integer for --num-checkpoints-read/-write

CK_RUN_WHATIF
  = CK_RUN_OPEN but:
    --accelerator-type choices: {h100,a100,b200,mi355}

──────────────────────────────────────────────────────────────────

CK_CONFIGVIEW_CLOSED
  Required:
    --model/-m {llama3-8b,llama3-70b,llama3-405b,llama3-1t}
    --num-processes/-np N
    --accelerator-type/-at {b200,mi355}
    --client-host-memory-in-gb/-cm N
    --results-dir/-rd PATH          (layered; see CORE_STD)
    --systemname/-sn NAME           (layered; see CORE_STD)
    [storage positional: file | object]
  Optional:
    --exec-type/-et {mpi,docker}    (default: mpi)
    --hosts/-s HOST...              (default: 127.0.0.1)
    --dlio-bin-path/-dp PATH
    --num-checkpoints-read/-ncr N   (default: 10)
    --num-checkpoints-write/-ncw N  (default: 10)
    --o-direct
  + MPI_ARGS
  + CORE_STD

CK_CONFIGVIEW_OPEN
  = CK_CONFIGVIEW_CLOSED plus:
    --params/-p KEY=VALUE...
  + OPEN_STD

CK_CONFIGVIEW_WHATIF
  = CK_CONFIGVIEW_OPEN but:
    --accelerator-type choices: {h100,a100,b200,mi355}

Placeholder definitions — VECTORDB

VDB_DATASIZE_CLOSED
  Optional:
    --vdb-engine {milvus}           (default: milvus)
    --vdb-index {DISKANN,HNSW,AISAQ}   Index family; names the result path
                                       vector_database/<engine>/<index>/...
    --index-type {DISKANN,HNSW,AISAQ}  Milvus index for storage estimation
                                       (defaults to --vdb-index)
    --dimension N                   (default: 1536)
    --num-vectors N                 (default: 1,000,000)
    --num-shards N                  (default: 1)
    --vector-dtype {FLOAT_VECTOR}   (default: FLOAT_VECTOR)
  + CORE_STD  (--results-dir layered: flag > --config-file > env > init default; --systemname optional)

VDB_DATASIZE_OPEN
  = VDB_DATASIZE_CLOSED plus:
    --vdb-index {DISKANN,HNSW,AISAQ,IVF_FLAT,IVF_SQ8,FLAT}   (open widens choices)
    --index-type {DISKANN,HNSW,AISAQ,IVF_FLAT,IVF_SQ8,FLAT}
    --params KEY=VALUE...
  + OPEN_STD

VDB_DATASIZE_WHATIF
  = VDB_DATASIZE_OPEN  (flags identical)

──────────────────────────────────────────────────────────────────

VDB_DATAGEN_CLOSED
  Required:
    --results-dir/-rd PATH          (layered; see CORE_STD)
    --systemname/-sn NAME           (layered; see CORE_STD)
    [storage positional: file | object]
  Optional:
    --vdb-engine {milvus}           (default: milvus)
    --vdb-index {DISKANN,HNSW,AISAQ}
    --index-type {DISKANN,HNSW,AISAQ}  Milvus index to create during load
                                       (defaults to --vdb-index)
    --host/-s IP                    Milvus server address (default: 127.0.0.1)
    --port/-p N                     Milvus port (default: 19530)
    --collection NAME
    --config PATH
    --dimension N                   (default: 1536)
    --num-shards N                  (default: 1)
    --vector-dtype {FLOAT_VECTOR}   (default: FLOAT_VECTOR)
    --num-vectors N                 (default: 1,000,000)
    --distribution {uniform,normal,zipfian}  (default: uniform)
    --batch-size N                  (default: 1,000)
    --chunk-size N                  (default: 10,000)
    --force
  VDB storage location (recorded for Rules.md 5.4.1; overrides config storage.*):
    --storage-root PATH             Where the engine stores its data
                                    (must differ from --results-dir)
    --storage-type TYPE             Storage medium, e.g. local_fs, s3
                                    (default: local_fs)
  Distributed launch:
    --distributed                   Fan datagen out across --hosts via MPI
    --hosts HOST...                 (no -s short form here; -s is --host)
    --npernode/--num-processes-per-client N  (default: 1)
    --mpi-impl {mpich,openmpi}      (default: mpich)
    --coordination {filesystem,mpi} (default: filesystem)
    --rank-output-dir PATH          (default: /tmp/mlps_vdb)
    --seed N                        (default: 42)
    --ready-timeout SECS            (default: 7200)
  + MPI_ARGS
  + CORE_STD

VDB_DATAGEN_OPEN
  = VDB_DATAGEN_CLOSED plus:
    --vdb-index {DISKANN,HNSW,AISAQ,IVF_FLAT,IVF_SQ8,FLAT}   (open widens choices)
    --index-type {DISKANN,HNSW,AISAQ,IVF_FLAT,IVF_SQ8,FLAT}
    --M N                           HNSW M parameter (default: 16)
    --ef-construction N             (default: 200)
    --max-degree N                  (default: 16)
    --inline-pq N                   (default: 16)
    --search-list-size N            (default: 200)
    --metric-type {COSINE,L2,IP}    (default: COSINE)
    --compact                       Compact the collection after load
    --monitor-interval SECS         (default: 5)
    --params KEY=VALUE...
  + OPEN_STD

VDB_DATAGEN_WHATIF
  = VDB_DATAGEN_OPEN  (flags identical)

──────────────────────────────────────────────────────────────────

VDB_RUN_CLOSED
  Required:
    --results-dir/-rd PATH          (layered; see CORE_STD)
    --systemname/-sn NAME           (layered; see CORE_STD)
    [storage positional: file | object]
  Optional:
    --vdb-engine {milvus}           (default: milvus)
    --vdb-index {DISKANN,HNSW,AISAQ}   Index already loaded in the target collection
    --host/-s IP                    (default: 127.0.0.1)
    --port/-p N                     (default: 19530)
    --collection NAME
    --config PATH
    --num-query-processes N         (default: 1)
    --batch-size N                  (default: 1)
    --report-count N                (default: 100)
    --benchmark-mode {timed,query_count,sweep}  (default: timed)
    --runtime N                     Seconds; mutually exclusive with --queries
    --queries N                     Mutually exclusive with --runtime
    --num-query-vectors N           (default: 1000)
    --search-limit N                (default: 10)
    --search-ef N                   (default: 200)
    --recall-k N                    K for recall@k (defaults to --search-limit)
    --gt-collection NAME            Ground-truth FLAT collection
                                    (default: <collection>_flat_gt)
    --vector-dim N                  (default: 1536)
  VDB storage location (recorded for Rules.md 5.4.1; overrides config storage.*):
    --storage-root PATH
    --storage-type TYPE
  Distributed launch:
    --distributed
    --hosts HOST...                 (no -s short form here; -s is --host)
    --npernode/--num-processes-per-client N  (default: 1)
    --mpi-impl {mpich,openmpi}      (default: mpich)
    --coordination {filesystem,mpi} (default: filesystem)
    --rank-output-dir PATH          (default: /tmp/mlps_vdb)
    --seed N                        (default: 42)
    --ready-timeout SECS            (default: 7200)
  + MPI_ARGS
  + CORE_STD

VDB_RUN_OPEN
  = VDB_RUN_CLOSED plus:
    --vdb-index {DISKANN,HNSW,AISAQ,IVF_FLAT,IVF_SQ8,FLAT}   (open widens choices)
    --params KEY=VALUE...
  + OPEN_STD
  + TIMESERIES

VDB_RUN_WHATIF
  = VDB_RUN_OPEN  (flags identical)

Placeholder definitions — KVCACHE

Note: kvcache never has a file|object storage positional (architectural constraint).
No object storage support at any level.

KV_DATASIZE_CLOSED
  (Model and cache-tier sizes fixed in closed: the phase sequence uses
   llama3.1-8b + llama3.1-70b-instruct automatically)
  Optional:
    --cache-dir PATH                NVMe cache tier directory
                                    (default: subdirectory of results)
  + CORE_STD  (--results-dir layered: flag > --config-file > env > init default; --systemname optional)
  Note: --gpu-mem-gb=16.0 and --cpu-mem-gb=32.0 fixed; not shown

KV_DATASIZE_OPEN
  = KV_DATASIZE_CLOSED plus:
    --gpu-mem-gb FLOAT              (default: 16.0)
    --cpu-mem-gb FLOAT              (default: 32.0)
  Note: OPEN_STD (--loops / --allow-invalid-params) is NOT available on
        kvcache datasize in any mode

KV_DATASIZE_WHATIF
  = KV_DATASIZE_OPEN  (flags identical)

──────────────────────────────────────────────────────────────────

KV_RUN_CLOSED
  (Fixed 3-phase sequence; model pair and load parameters are pinned)
  Required:
    --results-dir/-rd PATH          (layered; see CORE_STD)
    --systemname/-sn NAME           (layered; see CORE_STD)
  Optional:
    --cache-dir PATH
    --kvcache-bin-path PATH
    --exec-type/-et {mpi,docker}    (default: mpi)
    --num-processes/-np N
    --hosts/-s HOST...              (default: 127.0.0.1)
  + MPI_ARGS
  + CORE_STD
  Note: the following are fixed in closed and not shown:
    gpu-mem-gb=16.0, cpu-mem-gb=32.0, duration=60s,
    generation-mode=realistic, performance-profile=throughput,
    seed=42, trials=3, inter-option-delay=90s,
    disable-multi-turn=False, disable-prefix-caching=False,
    enable-rag=True, rag-num-docs=10,
    enable-autoscaling=True, autoscaler-mode=qos

KV_RUN_OPEN
  = KV_RUN_CLOSED plus:
    --model/-m {tiny-1b,mistral-7b,llama2-7b,llama3.1-8b,llama3.1-70b-instruct}
                                    (default: tiny-1b)
    --num-users/-nu N               Concurrent users to simulate (default: 100)
    --npernode/--num-processes-per-client N  (default: 1)
    --gpu-mem-gb FLOAT              (default: 16.0)
    --cpu-mem-gb FLOAT              (default: 32.0)
    --duration/-d N                 Seconds (default: 60)
    --generation-mode {none,fast,realistic}  (default: realistic)
    --performance-profile {latency,throughput}  (default: throughput)
    --disable-multi-turn
    --disable-prefix-caching
    --enable-rag
    --rag-num-docs N                (default: 10)
    --enable-autoscaling
    --autoscaler-mode {qos,capacity}  (default: qos)
    --seed N
    --trials N
    --inter-option-delay N
    --config PATH
    --max-concurrent-allocs N       Cap on concurrent in-flight cache allocations
    --enable-latency-tracing        bpftrace block-layer device latency tracing
                                    (requires root)
  + OPEN_STD
  + TIMESERIES

KV_RUN_WHATIF
  = KV_RUN_OPEN  (flags identical)

Placeholder definitions — UTILITY COMMANDS

RP_REPORTGEN
  Required:
    --results-dir/-rd PATH          (layered; see CORE_STD)
  + CORE_STD  (every standard argument is accepted)

──────────────────────────────────────────────────────────────────

HI_SHOW
  Optional:
    --limit/-n N                    Show the N most recent entries
    --id/-i N                       Show a specific entry by ID

HI_RERUN
  Required:
    rerun_id  (positional)          History entry ID to re-run

──────────────────────────────────────────────────────────────────

RUNS_LIST
  Optional:
    --results-dir/-rd PATH          Tree to manage (resolved: flag > MLPSTORAGE_RESULTS_DIR
                                    > the results-dir recorded by `mlpstorage init`)
    --mode {closed,open,whatif}     Only runs from this submission mode
    --benchmark {training,checkpointing,vectordb,kvcache}
                                    Only runs of this benchmark
    --model NAME                    Only runs of this model (vectordb: <engine>/<index>)
    --systemname/-sn NAME           Only runs from this system-under-test
    --status {complete,failed,incomplete}
                                    Only runs in this state
    --json                          JSON array instead of a table
  Every canonical run leaf gets a small stable ID in <results-dir>/.mlps/runs.jsonl;
  leaves removed by hand disappear from the list and their IDs are never reused.

RUNS_SHOW
  Required:
    run_id  (positional)            Run ID from `mlpstorage runs list`
  Optional:
    --results-dir/-rd PATH          Tree to manage (resolved as for RUNS_LIST)
  Prints identity, status, code-image pointer resolution, a metadata excerpt
  and every file in the leaf.

RUNS_RM
  Optional:
    run_ids  (positional, repeatable)
                                    Run IDs to remove; omit to select by --status / --older-than
    --results-dir/-rd PATH          Tree to manage (resolved as for RUNS_LIST)
    --status {complete,failed,incomplete}
                                    Remove runs in this state
    --older-than AGE|DATE           Remove runs started before this (12h, 7d, 2w, 2026-09-01)
    --keep-last N                   Keep the N newest runs of the selection
    --yes/-y                        Skip the confirmation (required when stdin is not a terminal)
  Moves leaves to <results-dir>/.mlps/trash/<batch>/<original path>; nothing is
  deleted until `runs purge`. Never touches the sentinel, systems/ or the code-image pool.

RUNS_PURGE
  Optional:
    --results-dir/-rd PATH          Tree to manage (resolved as for RUNS_LIST)
    --yes/-y                        Skip the confirmation (required when stdin is not a terminal)
  Permanently deletes everything under <results-dir>/.mlps/trash.

RUNS_GC
  Optional:
    --results-dir/-rd PATH          Tree to manage (resolved as for RUNS_LIST)
    --yes/-y                        Skip the confirmation (required when stdin is not a terminal)
  Moves code-image pool directories that no run leaf (in the tree or its trash)
  points at into the trash — the CHECK-03 orphans `mlpstorage validate` reports.

──────────────────────────────────────────────────────────────────

LF_GENERATE
  Optional:
    --output/-o PATH                (default: requirements.txt)
    --extra EXTRA                   Include optional dep group (repeatable)
    --hashes                        Include SHA256 hashes
    --python-version VERSION
    --pyproject PATH                (default: pyproject.toml)
    --all                           Generate both requirements.txt and requirements-full.txt
  + CORE_STD  (--results-dir accepted but never required or used: lockfile
               touches only pyproject.toml and the installed environment)

LF_VERIFY
  Optional:
    --lockfile/-l PATH              (default: requirements.txt)
    --skip PKG                      Package to skip (repeatable)
    --allow-missing
    --strict
  + CORE_STD  (--results-dir accepted but never required or used)

──────────────────────────────────────────────────────────────────

VALIDATE
  Required:
    input  (positional)             Submission directory to check
  Optional:
    --submitters CSV                Comma-separated submitter allowlist (default: all)
    --csv PATH                      Summary CSV path (default: summary.csv)
    --skip-output-file              Suppress per-submission output file
    --reference-checksum MD5        Override REFERENCE_CHECKSUMS for code/ MD5 check

RULES_COVERAGE
  Optional:
    --rules-md PATH                 Path to Rules.md (default: project-root Rules.md)

──────────────────────────────────────────────────────────────────

VERSION
  No flags.  Prints the installed package version string and exits 0.
  Resolution order: importlib.metadata("mlpstorage") → pyproject.toml → "unknown"

──────────────────────────────────────────────────────────────────

INIT
  Required (positional):
    orgname                         Organization name to pin to this results-dir
                                    (Rules.md §2.1.5 submitter identity)
  Optional (positional):
    results-dir                     Filesystem path to initialize as a results-dir
                                    (default: ~/mlpstorage-results)
  Behavior:
    Writes <results-dir>/mlperf-results.yaml as the sentinel that subsequent
    commands read to resolve orgname, and records <results-dir> in
    ~/.config/mlpstorage/config.yaml ($XDG_CONFIG_HOME honoured) as the
    default every later command uses when --results-dir and
    MLPSTORAGE_RESULTS_DIR are both absent. Creates <results-dir> if absent
    (parent must exist). Idempotent when the sentinel already pins the same
    orgname (the default is re-recorded, which is how you switch trees);
    refuses to overwrite a sentinel that pins a different orgname.

──────────────────────────────────────────────────────────────────

CONFIG_SHOW
  Optional:
    --json                          The same report as one JSON object
  Prints the per-user config file path, then every key the file may carry
  (results_dir, systemname, data_dir, checkpoint_folder, hosts, mpi_bin,
  mpi_btl, oversubscribe, allow_run_as_root, mpi_params, dlio_bin_path,
  exec_type, color, stream_log_level) with its value or [unset]. A set
  MLPSTORAGE_* env var is named next to the key it outranks. A workload
  key in the file is flagged REJECTED (every other command fails until it
  is removed); a key no command knows is flagged ignored.

CONFIG_SET
  Required (positional):
    key                             One of the keys CONFIG_SHOW lists
    values                          One value; several (or one comma-separated
                                    value for hosts) store a list; true/false for
                                    oversubscribe and allow_run_as_root
  The value is type- and choice-checked exactly as the flag would be
  (`mpi_bin mpi` is refused, `mpi_bin mpiexec` stored). A workload-selecting
  key (model, accelerator_type, ...) or an unknown key is an error. Other keys
  in the file are kept.

CONFIG_UNSET
  Required (positional):
    key                             Key to remove; any key, so a stray workload
                                    key can be cleared
  Removing results_dir prints how to supply one from then on.

CONFIG_PATH
  No flags.  Prints the per-user config file path
  ($XDG_CONFIG_HOME/mlpstorage/config.yaml, else ~/.config/mlpstorage/config.yaml).
  `config` never needs a results-dir, skips the tier resolver every other
  command runs at parse time (so it works on a file that resolver refuses),
  and is not recorded in history.
"""

# HELP_ALL_TEXT is composed from three pieces so that SYNOPSIS_TEXT can be
# printed standalone (mid-tree -h) without parsing the combined string.
HELP_ALL_TEXT = _HEADER_TEXT + "\n" + SYNOPSIS_TEXT + "\n\n" + _TREE_AND_BODY_TEXT

# ---------------------------------------------------------------------------
# get_context_help_tokens
#
# CONTRACT: receives POSITIONAL-ONLY tokens. Caller strips all option-style
# tokens (anything starting with '-') before calling. This function never
# sees flags like '--help', '-cm', etc. — only bare positional words.
# ---------------------------------------------------------------------------

_MODES = frozenset(('closed', 'open', 'whatif'))
_BENCHMARKS = frozenset(('training', 'checkpointing', 'vectordb', 'kvcache'))

# Training models per mode: closed / open from the current rules edition
# (editions.yaml `workloads:`), whatif = everything the tool can run.
from mlpstorage_py.editions import current_edition as _current_edition  # noqa: E402

_TRAINING_MODELS_CLOSED_OPEN = (frozenset(_current_edition().models('training', 'closed'))
                                | frozenset(_current_edition().models('training', 'open')))
_TRAINING_MODELS_WHATIF = frozenset(('cosmoflow', 'resnet50', 'unet3d', 'dlrm', 'retinanet', 'flux'))

# Training commands that have a file|object storage positional
_TRAINING_CMDS_WITH_STORAGE = frozenset(('datagen', 'run', 'configview'))
_TRAINING_CMDS_ALL = frozenset(('datasize', 'datagen', 'run', 'configview'))

# Checkpointing commands
_CKPT_CMDS_WITH_STORAGE = frozenset(('run', 'configview'))
_CKPT_CMDS_ALL = frozenset(('datasize', 'run', 'configview'))

# VectorDB commands
_VDB_CMDS_WITH_STORAGE = frozenset(('datagen', 'run'))
_VDB_CMDS_ALL = frozenset(('datasize', 'datagen', 'run'))

# KVCache commands (NO file|object at any level)
_KV_CMDS_ALL = frozenset(('datasize', 'run'))

_STORAGE_POSITIONALS = frozenset(('file', 'object'))


def get_context_help_tokens(argv: list) -> 'str | None':
    """Return a "next: X | Y | Z" hint string for the given positional token path.

    Parameters
    ----------
    argv : list[str]
        POSITIONAL-ONLY tokens. Caller must strip all option-style tokens
        (anything starting with '-') before calling this function.

    Returns
    -------
    str or None
        "next: X | Y | Z" if the path is a recognised mid-tree position,
        None if the path is a leaf or contains unrecognised tokens (fall
        through to argparse).
    """
    n = len(argv)

    # Root — no tokens
    if n == 0:
        return 'next: closed | open | whatif | init | reports | history | runs | lockfile | version | validate | rules-coverage'

    t0 = argv[0]

    # ── Utility siblings ────────────────────────────────────────────────────
    if t0 == 'reports':
        if n == 1:
            return 'next: reportgen'
        return None  # leaf (reportgen + any further tokens)

    if t0 == 'history':
        if n == 1:
            return 'next: show | rerun'
        return None  # leaf

    if t0 == 'runs':
        if n == 1:
            return 'next: list | show | rm | purge | gc'
        return None  # leaf

    if t0 == 'lockfile':
        if n == 1:
            return 'next: generate | verify'
        return None  # leaf

    if t0 == 'version':
        return None  # leaf — fall through to argparse

    if t0 == 'validate':
        return None  # leaf — fall through to argparse (positional <input> required)

    if t0 == 'rules-coverage':
        return None  # leaf — fall through to argparse

    if t0 == 'init':
        return None  # leaf — fall through to argparse (positionals <orgname> <results-dir> required)

    if t0 == 'config':
        if n == 1:
            return 'next: show | set | unset | path'
        return None  # leaf

    # ── Three-mode benchmark branch ──────────────────────────────────────────
    if t0 not in _MODES:
        return None  # unrecognised first token

    mode = t0

    if n == 1:
        return 'next: training | checkpointing | vectordb | kvcache'

    t1 = argv[1]

    if t1 not in _BENCHMARKS:
        return None  # unrecognised benchmark token

    bench = t1

    # ── kvcache (no model positional, no file|object) ────────────────────────
    if bench == 'kvcache':
        if n == 2:
            return 'next: datasize | run'
        # Any further token (command or beyond) → leaf
        return None

    # ── vectordb ─────────────────────────────────────────────────────────────
    if bench == 'vectordb':
        if n == 2:
            return 'next: datasize | datagen | run'
        t2 = argv[2]
        if t2 not in _VDB_CMDS_ALL:
            return None  # unrecognised command
        if t2 in _VDB_CMDS_WITH_STORAGE:
            if n == 3:
                return 'next: file | object'
            return None  # storage positional supplied → leaf
        # datasize (no storage) → leaf
        return None

    # ── checkpointing (no model positional) ─────────────────────────────────
    if bench == 'checkpointing':
        if n == 2:
            return 'next: datasize | run | configview'
        t2 = argv[2]
        if t2 not in _CKPT_CMDS_ALL:
            return None  # unrecognised command
        if t2 in _CKPT_CMDS_WITH_STORAGE:
            if n == 3:
                return 'next: file | object'
            return None  # storage supplied → leaf
        # datasize → leaf
        return None

    # ── training (has model positional) ─────────────────────────────────────
    if bench == 'training':
        if n == 2:
            if mode == 'whatif':
                return 'next: cosmoflow | resnet50 | unet3d | dlrm | retinanet | flux'
            return 'next: unet3d | retinanet'

        t2 = argv[2]  # model positional
        # Validate model for mode
        if mode == 'whatif':
            valid_models = _TRAINING_MODELS_WHATIF
        else:
            valid_models = _TRAINING_MODELS_CLOSED_OPEN

        if t2 not in valid_models:
            return None  # unrecognised model → fall through

        if n == 3:
            return 'next: datasize | datagen | run | configview'

        t3 = argv[3]  # command positional
        if t3 not in _TRAINING_CMDS_ALL:
            return None  # unrecognised command

        if t3 in _TRAINING_CMDS_WITH_STORAGE:
            if n == 4:
                return 'next: file | object'
            return None  # storage supplied → leaf
        # datasize → leaf
        return None

    return None  # should be unreachable
