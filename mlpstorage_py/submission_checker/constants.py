from .parsers.json_parser import JSONParser
from .parsers.yaml_parser import YamlParser

VALID_DIVISIONS = ["open", "closed"]

# Where a submission's system description lives, relative to the tree root.
# This is layout, not rules edition: every layout this tool reads puts it
# here. (The former per-version dicts -- VERSIONS, DEFAULT_SPEC_VERSION,
# *_REQUIRED_FILES / *_REQUIRED_FOLDERS -- moved into the rules editions
# table, mlpstorage_py/rules/editions.yaml `checker:`, read through Config.)
SYSTEM_PATH = "{division}/{submitter}/systems/{system}.yaml"

PARSER_MAP = {
    "System": YamlParser,
    "Summary": JSONParser,
    "Metadata": JSONParser,
    "default": JSONParser
}

# Issue #608: NUM_DATASET_TRAIN_FILES / NUM_DATASET_EVAL_FILES /
# NUM_DATASET_TRAIN_FOLDERS / NUM_DATASET_EVAL_FOLDERS placeholder dicts
# (`# TODO: Ask for correct values`) were deleted here. Rule 3.3.1 now
# reads the per-submission `datasize/<ts>/training_<ts>_metadata.json`
# instead, which is the value the datasize phase actually wrote and is
# guaranteed to match the configuration the submitter ran.

CHECKPOINT_FILE_MAP = {
    "llama3-1t": "llama3_1t.yaml",
    "llama3-8b": "llama3_8b.yaml",
    "llama3-70b": "llama3_70b.yaml",
    "llama3-405b": "llama3_405b.yaml",
}

# Rules.md Table 2 (§4.3.4 surface) — CLOSED total MPI processes per model (TP × PP × DP).
# DP (data parallelism) is NOT in the DLIO workload YAMLs (configs/dlio/workload/llama3_*.yaml);
# only tensor and pipeline are recorded there. This constant encodes the BENCHMARK CONTRACT
# for CLOSED submissions (the runtime workload config and the contract are two distinct sources
# of truth — see D-C4 of Phase 2 CONTEXT.md).
CLOSED_MPI_PROCESSES: dict[str, int] = {
    "8b": 8,
    "70b": 64,
    "405b": 512,
    "1t": 1024,
}

# Rules.md 2.1.6 / 3.6.1 codeDirectoryContents / trainingClosedSubmissionChecksum
# Reference hex MD5 of the canonical code/ tree per version. None means
# "not yet pinned" — runtime check will emit a WARNING via warn_violation
# (D-12) and pass. Use `python -m mlpstorage_py.submission_checker.tools.\
# compute_code_checksum <path>` to regenerate.
REFERENCE_CHECKSUMS: dict[str, str | None] = {
    "v2.0": None,
    "v3.0": None,
    "default": None,
}

# Rules.md 2.1.17 runTimestamps — exactly 6 (1 warm-up + 5 measured)
RUN_TIMESTAMP_COUNT = 6

# Directory-name prefixes excluded from the code-tree MD5 (Rules.md 2.1.6).
# Match is against POSIX-joined relative paths with a trailing slash so that
# `.gitignore` (file) does not collide with `.git/` (directory prefix).
#
# Dot-prefixed entries (.git/, .idea/, .planning/, etc.) catch local
# developer / tooling artifacts that the project's .gitignore already
# excludes from version control. They are not part of the benchmark
# source contract, would change every time a contributor's tools change,
# and (in the .gsd-tmp/ case) would even contain transient agent state.
MD5_EXCLUDE_PREFIXES: tuple[str, ...] = (
    ".git/",
    ".idea/",          # JetBrains IDE workspace
    ".vscode/",        # VS Code workspace
    ".claude/",        # Claude CLI runtime / settings
    ".agent/",         # Agent runtime (per project .gitignore "Coding Agents")
    ".agents/",        # Same, alternate name
    ".roo/",           # Roo agent runtime
    ".planning/",      # GSD planning artifacts (project-local)
    ".gsd-tmp/",       # GSD code-fixer worktree (project-local)
    "__pycache__/",
    ".pytest_cache/",
    ".venv/",
    "node_modules/",
    "build/",
    "dist/",
    ".tox/",
    "test/",
    "tests/",
)

# Filename patterns excluded from the code-tree MD5 (Rules.md 2.1.6).
# Matched against the basename. ``.egg-info`` is handled at the prefix level
# (any directory ending in ``.egg-info``) — keep that in the predicate, not here.
MD5_EXCLUDE_FILENAMES: tuple[str, ...] = (
    ".code-hash.json",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "Thumbs.db",
)