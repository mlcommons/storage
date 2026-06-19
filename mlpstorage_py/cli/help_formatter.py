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
  mlpstorage <closed|open|whatif> <benchmark> <model|algorithm> <command> <file|object> [OPTIONS]
  mlpstorage (reports|history|lockfile|version) [subcommand] [OPTIONS]
  mlpstorage validate <submission-dir> [OPTIONS]
  mlpstorage rules-coverage [--rules-md PATH]

  <closed|open|whatif>  — required first positional for benchmark commands
  <model|algorithm>     — required second positional (see per-benchmark choices below)
  <file|object>         — required storage selector for commands that touch storage
                          (absent on datasize; absent on all kvcache commands)"""

_TREE_AND_BODY_TEXT = """\
mlpstorage
│
├── closed ──────────────────────────────────────────────────────
│   ├── training
│   │   └── unet3d | retinanet
│   │       ├── datasize                          {TR_DATASIZE_CLOSED}
│   │       ├── datagen    file | object           {TR_DATAGEN_CLOSED}
│   │       ├── run        file | object           {TR_RUN_CLOSED}
│   │       └── configview file | object           {TR_CONFIGVIEW_CLOSED}
│   │
│   ├── checkpointing
│   │   └── llama3-8b | llama3-70b | llama3-405b | llama3-1t
│   │       ├── datasize                          {CK_DATASIZE_CLOSED}
│   │       ├── run        file | object           {CK_RUN_CLOSED}
│   │       └── configview file | object           {CK_CONFIGVIEW_CLOSED}
│   │
│   ├── vectordb
│   │   └── DISKANN | HNSW | AISAQ
│   │       ├── datasize                          {VDB_DATASIZE_CLOSED}
│   │       ├── datagen    file | object           {VDB_DATAGEN_CLOSED}
│   │       └── run        file | object           {VDB_RUN_CLOSED}
│   │
│   └── kvcache                          ← no model positional in closed
│       ├── datasize                              {KV_DATASIZE_CLOSED}
│       └── run                                   {KV_RUN_CLOSED}
│
├── open ────────────────────────────────────────────────────────
│   ├── training
│   │   └── unet3d | retinanet
│   │       ├── datasize                          {TR_DATASIZE_OPEN}
│   │       ├── datagen    file | object           {TR_DATAGEN_OPEN}
│   │       ├── run        file | object           {TR_RUN_OPEN}
│   │       └── configview file | object           {TR_CONFIGVIEW_OPEN}
│   │
│   ├── checkpointing
│   │   └── llama3-8b | llama3-70b | llama3-405b | llama3-1t
│   │       ├── datasize                          {CK_DATASIZE_OPEN}
│   │       ├── run        file | object           {CK_RUN_OPEN}
│   │       └── configview file | object           {CK_CONFIGVIEW_OPEN}
│   │
│   ├── vectordb
│   │   └── DISKANN | HNSW | AISAQ | IVF_FLAT | IVF_SQ8 | FLAT
│   │       ├── datasize                          {VDB_DATASIZE_OPEN}
│   │       ├── datagen    file | object           {VDB_DATAGEN_OPEN}
│   │       └── run        file | object           {VDB_RUN_OPEN}
│   │
│   └── kvcache
│       └── tiny-1b | mistral-7b | llama2-7b | llama3.1-8b | llama3.1-70b-instruct
│           ├── datasize                          {KV_DATASIZE_OPEN}
│           └── run                               {KV_RUN_OPEN}
│
├── whatif ──────────────────────────────────────────────────────
│   ├── training
│   │   └── cosmoflow | resnet50 | unet3d | dlrm | retinanet | flux
│   │       ├── datasize                          {TR_DATASIZE_WHATIF}
│   │       ├── datagen    file | object           {TR_DATAGEN_WHATIF}
│   │       ├── run        file | object           {TR_RUN_WHATIF}
│   │       └── configview file | object           {TR_CONFIGVIEW_WHATIF}
│   │
│   ├── checkpointing
│   │   └── llama3-8b | llama3-70b | llama3-405b | llama3-1t
│   │       ├── datasize                          {CK_DATASIZE_WHATIF}
│   │       ├── run        file | object           {CK_RUN_WHATIF}
│   │       └── configview file | object           {CK_CONFIGVIEW_WHATIF}
│   │
│   ├── vectordb
│   │   └── DISKANN | HNSW | AISAQ | IVF_FLAT | IVF_SQ8 | FLAT
│   │       ├── datasize                          {VDB_DATASIZE_WHATIF}
│   │       ├── datagen    file | object           {VDB_DATAGEN_WHATIF}
│   │       └── run        file | object           {VDB_RUN_WHATIF}
│   │
│   └── kvcache
│       └── tiny-1b | mistral-7b | llama2-7b | llama3.1-8b | llama3.1-70b-instruct
│           ├── datasize                          {KV_DATASIZE_WHATIF}
│           └── run                               {KV_RUN_WHATIF}
│
├── reports
│   └── reportgen                                {RP_REPORTGEN}
│
├── history
│   ├── list                                     {HI_LIST}
│   └── replay                                   {HI_REPLAY}
│
├── lockfile
│   ├── generate                                 {LF_GENERATE}
│   └── verify                                   {LF_VERIFY}
│
├── validate <submission-dir>                    Run the Rules.md submission checker
│       --submitters CSV                          Comma-separated submitter allowlist (default: all)
│       --mlperf-version VERSION                  Spec version (default: v5.1)
│       --csv PATH                                Summary CSV path (default: summary.csv)
│       --skip-output-file                        Suppress per-submission output file
│       --reference-checksum MD5                  Override REFERENCE_CHECKSUMS for code/ MD5 check
│
├── rules-coverage                               Reconcile Rules.md IDs against @rule-decorated checks
│       --rules-md PATH                           Path to Rules.md (default: project-root Rules.md)
│
└── version                                      {VERSION}

Common argument groups

CORE_STD — Standard arguments, all three modes
  --results-dir/-rd PATH        Benchmark results directory
  --config-file/-c PATH         YAML overrides file (applied after CLI args)
  --debug                       Enable debug output
  --verbose                     Enable verbose output
  --stream-log-level LEVEL      Logging level (default: INFO)
  --dry-run                     Print the command that would execute; do not run
  --verify-lockfile PATH        Validate installed packages against lockfile
  --skip-validation             Skip MPI/SSH/DLIO pre-run environment checks

OPEN_STD — Additional standard arguments, open and whatif modes only
  --loops N                     Repeat benchmark N times (default: 1)
  --allow-invalid-params/-aip   Do not abort on invalid DLIO parameters

MPI_ARGS — MPI execution arguments
  --mpi-bin {mpirun,mpiexec}    MPI launcher binary (default: mpirun)
  --mpi-btl {auto,vader,tcp}    Byte Transport Layer for single-host runs (default: auto)
  --oversubscribe               Allow more ranks than CPU slots
  --allow-run-as-root           Permit execution as root (OpenMPI flag)
  --mpi-params PARAM...         Additional raw MPI parameters (repeatable)

TIMESERIES — Time-series host metrics, open and whatif run commands only
  --timeseries-interval SECS    Sample interval in seconds (default: 10.0)
  --skip-timeseries             Disable time-series collection entirely
  --max-timeseries-samples N    Per-host sample cap (default: 3600)

Placeholder definitions — TRAINING

TR_DATASIZE_CLOSED
  Required:
    --max-accelerators/-ma N
    --accelerator-type/-g {b200,mi355}
    --client-host-memory-in-gb/-cm N
    --data-dir/-dd PATH
  Optional:
    --num-client-hosts/-nc N        Derived from --hosts count if unset
    --dlio-bin-path/-dp PATH
    --exec-type/-et {mpi,docker}    (default: mpi)
    --hosts/-s HOST...              (default: 127.0.0.1)
  + MPI_ARGS
  + CORE_STD  (--results-dir optional)

TR_DATASIZE_OPEN
  = TR_DATASIZE_CLOSED plus:
    --params/-p KEY=VALUE...        DLIO parameter overrides (repeatable)
  + OPEN_STD

TR_DATASIZE_WHATIF
  = TR_DATASIZE_OPEN but:
    --accelerator-type choices: {h100,a100,b200,mi355}

──────────────────────────────────────────────────────────────────

TR_DATAGEN_CLOSED
  Required:
    --num-processes/-np N
    --results-dir/-rd PATH
    --data-dir/-dd PATH
    [storage positional: file | object]
  Optional:
    --dlio-bin-path/-dp PATH
    --exec-type/-et {mpi,docker}    (default: mpi)
    --hosts/-s HOST...              (default: 127.0.0.1)
  + MPI_ARGS
  + CORE_STD

TR_DATAGEN_OPEN
  = TR_DATAGEN_CLOSED plus:
    --params/-p KEY=VALUE...
  + OPEN_STD

TR_DATAGEN_WHATIF
  = TR_DATAGEN_OPEN  (model positional choices differ; flags identical)

──────────────────────────────────────────────────────────────────

TR_RUN_CLOSED
  Required:
    --num-accelerators/-na N
    --accelerator-type/-g {b200,mi355}
    --client-host-memory-in-gb/-cm N
    --checkpoint-folder/-cf PATH
    --results-dir/-rd PATH
    --data-dir/-dd PATH
    [storage positional: file | object]
  Optional:
    --num-client-hosts/-nc N
    --dlio-bin-path/-dp PATH
    --exec-type/-et {mpi,docker}    (default: mpi)
    --hosts/-s HOST...              (default: 127.0.0.1)
  + MPI_ARGS
  + CORE_STD

TR_RUN_OPEN
  = TR_RUN_CLOSED plus:
    --params/-p KEY=VALUE...
  + OPEN_STD
  + TIMESERIES

TR_RUN_WHATIF
  = TR_RUN_OPEN but:
    --accelerator-type choices: {h100,a100,b200,mi355}

──────────────────────────────────────────────────────────────────

TR_CONFIGVIEW_CLOSED
  Required:
    --num-accelerators/-na N
    --results-dir/-rd PATH
    --data-dir/-dd PATH
    [storage positional: file | object]
  Optional:
    --dlio-bin-path/-dp PATH
  + CORE_STD

TR_CONFIGVIEW_OPEN
  = TR_CONFIGVIEW_CLOSED plus:
    --params/-p KEY=VALUE...
  + OPEN_STD

TR_CONFIGVIEW_WHATIF
  = TR_CONFIGVIEW_OPEN  (model positional choices differ; flags identical)

Placeholder definitions — CHECKPOINTING

CK_DATASIZE_CLOSED
  Required:
    --client-host-memory-in-gb/-cm N
  Optional:
    --hosts/-s HOST...              (default: 127.0.0.1)
    --num-checkpoints-read/-ncr N   (default: 10; closed allows 10 or 0)
    --num-checkpoints-write/-ncw N  (default: 10; closed allows 10 or 0)
  + CORE_STD  (--results-dir optional)
  Note: closed runs use 10/10 by default. Use 10/0 then 0/10 in two
        invocations when a cache flush is required between phases
        (see Rules.md §4.7.1 and checkpointing/README.md).

CK_DATASIZE_OPEN
  = CK_DATASIZE_CLOSED plus:
    --dlio-bin-path/-dp PATH
    --params/-p KEY=VALUE...
  + OPEN_STD
  Note: open allows any non-negative integer for --num-checkpoints-read/-write

CK_DATASIZE_WHATIF
  = CK_DATASIZE_OPEN  (model positional choices identical; flags identical)

──────────────────────────────────────────────────────────────────

CK_RUN_CLOSED
  Required:
    --num-processes/-np N
    --checkpoint-folder/-cf PATH
    --client-host-memory-in-gb/-cm N
    --results-dir/-rd PATH
    [storage positional: file | object]
  Optional:
    --exec-type/-et {mpi,docker}    (default: mpi)
    --hosts/-s HOST...              (default: 127.0.0.1)
    --num-checkpoints-read/-ncr N   (default: 10; closed allows 10 or 0)
    --num-checkpoints-write/-ncw N  (default: 10; closed allows 10 or 0)
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
    --dlio-bin-path/-dp PATH
    --params/-p KEY=VALUE...
  + OPEN_STD
  + TIMESERIES
  Note: open allows any multiple of the per-model GPU-per-DP-instance count
        and any non-negative integer for --num-checkpoints-read/-write

CK_RUN_WHATIF
  = CK_RUN_OPEN  (model positional choices identical; flags identical)

──────────────────────────────────────────────────────────────────

CK_CONFIGVIEW_CLOSED
  Required:
    --results-dir/-rd PATH
    [storage positional: file | object]
  Optional:
    --dlio-bin-path/-dp PATH
  + CORE_STD

CK_CONFIGVIEW_OPEN
  = CK_CONFIGVIEW_CLOSED plus:
    --params/-p KEY=VALUE...
  + OPEN_STD

CK_CONFIGVIEW_WHATIF
  = CK_CONFIGVIEW_OPEN  (flags identical)

Placeholder definitions — VECTORDB

VDB_DATASIZE_CLOSED
  Optional:
    --dimension N                   (default: 1536)
    --num-vectors N                 (default: 1,000,000)
    --index-type {DISKANN,HNSW,AISAQ}  (default: DISKANN)
    --num-shards N                  (default: 1)
    --vector-dtype {FLOAT_VECTOR}   (default: FLOAT_VECTOR)
  + CORE_STD  (--results-dir optional)

VDB_DATASIZE_OPEN
  = VDB_DATASIZE_CLOSED but:
    --index-type choices: {DISKANN,HNSW,AISAQ,IVF_FLAT,IVF_SQ8,FLAT}
  + OPEN_STD

VDB_DATASIZE_WHATIF
  = VDB_DATASIZE_OPEN  (algorithm positional choices differ; flags identical)

──────────────────────────────────────────────────────────────────

VDB_DATAGEN_CLOSED
  Required:
    --results-dir/-rd PATH
    [storage positional: file | object]
  Optional:
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
  + CORE_STD

VDB_DATAGEN_OPEN
  = VDB_DATAGEN_CLOSED
  + OPEN_STD

VDB_DATAGEN_WHATIF
  = VDB_DATAGEN_OPEN  (algorithm positional choices differ; flags identical)

──────────────────────────────────────────────────────────────────

VDB_RUN_CLOSED
  Required:
    --results-dir/-rd PATH
    [storage positional: file | object]
  Optional:
    --host/-s IP                    (default: 127.0.0.1)
    --port/-p N                     (default: 19530)
    --collection NAME
    --config PATH
    --num-query-processes N         (default: 1)
    --batch-size N                  (default: 1)
    --report-count N                (default: 100)
    --mode {timed,query_count,sweep}  (default: timed)
    --runtime N                     Mutually exclusive with --queries
    --queries N                     Mutually exclusive with --runtime
  + CORE_STD

VDB_RUN_OPEN
  = VDB_RUN_CLOSED
  + OPEN_STD
  + TIMESERIES

VDB_RUN_WHATIF
  = VDB_RUN_OPEN  (algorithm positional choices differ; flags identical)

Placeholder definitions — KVCACHE

Note: kvcache never has a file|object storage positional (architectural constraint).
No object storage support at any level.

KV_DATASIZE_CLOSED
  (No model positional — closed runs fixed phase sequence using
   llama3.1-8b + llama3.1-70b-instruct automatically)
  Required:
    --num-users/-nu N
  Optional:
    --cache-dir PATH
  + CORE_STD  (--results-dir optional)
  Note: --gpu-mem-gb=16.0 and --cpu-mem-gb=32.0 fixed; not shown

KV_DATASIZE_OPEN
  Required:
    --num-users/-nu N
  Optional:
    --cache-dir PATH
    --gpu-mem-gb FLOAT              (default: 16.0)
    --cpu-mem-gb FLOAT              (default: 32.0)
  + CORE_STD  (--results-dir optional)
  + OPEN_STD

KV_DATASIZE_WHATIF
  = KV_DATASIZE_OPEN  (flags identical)

──────────────────────────────────────────────────────────────────

KV_RUN_CLOSED
  (No model positional — fixed 3-phase sequence)
  Required:
    --num-users/-nu N
    --results-dir/-rd PATH
  Optional:
    --cache-dir PATH
    --kvcache-bin-path PATH
    --npernode/--num-processes-per-client N  (default: 1)
    --exec-type/-et {mpi,docker}    (default: mpi)
    --num-processes/-np N
    --hosts/-s HOST...              (default: 127.0.0.1)
  + MPI_ARGS
  + CORE_STD
  Note: the following are fixed in closed and not shown:
    duration=60s, generation-mode=realistic, performance-profile=throughput,
    seed=42, trials=3, inter-option-delay=20s,
    disable-multi-turn=False, disable-prefix-caching=False,
    enable-rag=True, rag-num-docs=10,
    enable-autoscaling=True, autoscaler-mode=qos

KV_RUN_OPEN
  Required:
    --num-users/-nu N
    --results-dir/-rd PATH
  Optional:
    --cache-dir PATH
    --kvcache-bin-path PATH
    --npernode/--num-processes-per-client N  (default: 1)
    --exec-type/-et {mpi,docker}    (default: mpi)
    --num-processes/-np N
    --hosts/-s HOST...              (default: 127.0.0.1)
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
    --autoscaler-mode {qos,predictive}  (default: qos)
    --seed N
    --trials N
    --inter-option-delay N
    --config PATH
  + MPI_ARGS
  + TIMESERIES
  + OPEN_STD

KV_RUN_WHATIF
  = KV_RUN_OPEN  (flags identical)

Placeholder definitions — UTILITY COMMANDS

RP_REPORTGEN
  Required:
    --results-dir/-rd PATH
  Optional:
    --output-dir PATH
    --config-file/-c PATH
    --debug
    --verbose

──────────────────────────────────────────────────────────────────

HI_LIST
  Optional:
    --limit/-n N                    Show N most recent entries
    --id/-i N                       Show specific entry by ID
    --debug
    --verbose

HI_REPLAY
  Required:
    ID  (positional)                History entry ID to re-run
  Optional:
    --debug
    --verbose

──────────────────────────────────────────────────────────────────

LF_GENERATE
  Optional:
    --output/-o PATH                (default: requirements.txt)
    --extra EXTRA                   Include optional dep group (repeatable)
    --hashes                        Include SHA256 hashes
    --python-version VERSION
    --pyproject PATH                (default: pyproject.toml)
    --all                           Generate both requirements.txt and requirements-full.txt

LF_VERIFY
  Optional:
    --lockfile/-l PATH              (default: requirements.txt)
    --skip PKG                      Package to skip (repeatable)
    --allow-missing
    --strict

──────────────────────────────────────────────────────────────────

VERSION
  No flags.  Prints the installed package version string and exits 0.
  Resolution order: importlib.metadata("mlpstorage") → pyproject.toml → "unknown"
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

# Training models per mode
_TRAINING_MODELS_CLOSED_OPEN = frozenset(('unet3d', 'retinanet'))
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
        return 'next: closed | open | whatif | reports | history | lockfile | version | validate | rules-coverage'

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
