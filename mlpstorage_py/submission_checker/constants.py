import re

from mlpstorage_py import VERSION as _PACKAGE_VERSION

from .parsers.json_parser import JSONParser
from .parsers.yaml_parser import YamlParser

VERSIONS = ["v2.0", "v3.0"]
VALID_DIVISIONS = ["open", "closed"]


def _derive_default_spec_version(package_version: str, supported: list) -> str:
    """Return the spec-version string that pairs with this package release.

    The MLPerf Storage spec ("Rules.md") evolves on round boundaries (v2.0,
    v3.0, v4.0, ...). The Python package version evolves on release
    boundaries (e.g. 3.0.0, 3.0.7, 3.0.8) — patch bumps for code fixes that
    do not touch the spec. The package's major.minor therefore IS the
    canonical spec round the code was built for; the patch level is the
    code-only delta. Derive the default for ``--mlperf-version`` from the
    package's major.minor so the two never drift.

    If the derived string is not in ``supported`` (e.g. during a transition
    when the new spec version hasn't been added to VERSIONS yet), fall back
    to the most recently supported round and let the validator surface a
    runtime mismatch via per-version dict lookups.

    Args:
        package_version: e.g. ``"3.0.8"``; falls back to ``"unknown"`` when
            both PEP 621 metadata and pyproject.toml are unavailable.
        supported: the ordered list of spec versions the per-version
            constants dicts ship entries for.

    Returns:
        A spec-version string such as ``"v3.0"``.
    """
    m = re.match(r"^(\d+)\.(\d+)", package_version)
    if m:
        candidate = f"v{m.group(1)}.{m.group(2)}"
        if candidate in supported:
            return candidate
    # Fallback: most recently supported round.
    return supported[-1] if supported else "unknown"


DEFAULT_SPEC_VERSION = _derive_default_spec_version(_PACKAGE_VERSION, VERSIONS)

SYSTEM_PATH = {
    "v2.0": "{division}/{submitter}/systems/{system}.yaml",
    "v3.0": "{division}/{submitter}/systems/{system}.yaml",
    "default": "{division}/{submitter}/systems/{system}.yaml",
}

PARSER_MAP = {
    "System": YamlParser,
    "Summary": JSONParser,
    "Metadata": JSONParser,
    "default": JSONParser
}

DATAGEN_REQUIRED_FILES = {
    "v2.0": [r"training_datagen\.stdout.log", r"training_datagen.stderr\.log", r".*output\.json$", r".*per_epoch_stats\.json$", r".*summary\.json$", r"dlio\.log"],
    "v3.0": [r"training_datagen\.stdout.log", r"training_datagen.stderr\.log", r".*output\.json$", r".*per_epoch_stats\.json$", r",*summary\.json$", r"dlio\.log"],
    "default": [r"training_datagen\.stdout.log", r"training_datagen.stderr\.log", r".*output\.json$", r".*per_epoch_stats\.json$", r".*summary\.json$", r"dlio\.log"],
}

DATAGEN_REQUIRED_FOLDERS = {
    "v2.0": ["dlio_config"],
    "v3.0": ["dlio_config"],
    "default": ["dlio_config"],
}

RUN_REQUIRED_FILES = {
    "v2.0": [r"training_run\.stdout.log", r"training_run\.stderr.log", r".*output\.json", r".*per_epoch_stats\.json", r".*summary\.json", r"dlio\.log"],
    "v3.0": [r"training_run\.stdout.log", r"training_run\.stderr.log", r".*output\.json", r".*per_epoch_stats\.json", r".*summary\.json", r"dlio\.log"],
    "default": [r"training_run\.stdout.log", r"training_run\.stderr.log", r".*output\.json", r".*per_epoch_stats\.json", r".*summary\.json", r"dlio\.log"],
}

RUN_REQUIRED_FOLDERS = {
    "v2.0": ["dlio_config"],
    "v3.0": ["dlio_config"],
    "default": ["dlio_config"],
}

# BUG-02 (D-E2): prior versions used training_run.* prefixes here — should be
# checkpointing_run.* (Rules.md 2.1.25 checkpointingFiles). The dot before
# "log" is also escaped (\.) to avoid latent over-matching.
CHECKPOINT_REQUIRED_FILES = {
    "v2.0": [r"checkpointing_run\.stdout\.log", r"checkpointing_run\.stderr\.log", r".*output\.json", r".*per_epoch_stats\.json", r".*summary\.json", r"dlio\.log"],
    "v3.0": [r"checkpointing_run\.stdout\.log", r"checkpointing_run\.stderr\.log", r".*output\.json", r".*per_epoch_stats\.json", r".*summary\.json", r"dlio\.log"],
    "default": [r"checkpointing_run\.stdout\.log", r"checkpointing_run\.stderr\.log", r".*output\.json", r".*per_epoch_stats\.json", r".*summary\.json", r"dlio\.log"],
}

CHECKPOINT_REQUIRED_FOLDERS = {
    "v2.0": ["dlio_config"],
    "v3.0": ["dlio_config"],
    "default": ["dlio_config"],
}

# TODO: Ask for correct values
NUM_DATASET_TRAIN_FILES = {
    "cosmoflow": 524288,
    "resnet50": 10391,
    "unet3d": 14000
}

NUM_DATASET_EVAL_FILES = {
    "cosmoflow": 0,
    "resnet50": 0,
    "unet3d": 0
}

NUM_DATASET_TRAIN_FOLDERS = {
    "cosmoflow": 0,
    "resnet50": 0,
    "unet3d": 0
}

NUM_DATASET_EVAL_FOLDERS = {
    "cosmoflow": 0,
    "resnet50": 0,
    "unet3d": 0
}

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
MD5_EXCLUDE_PREFIXES: tuple[str, ...] = (
    ".git/",
    "__pycache__/",
    ".pytest_cache/",
    ".venv/",
    "node_modules/",
    "build/",
    "dist/",
    ".tox/",
)

# Filename patterns excluded from the code-tree MD5 (Rules.md 2.1.6).
# Matched against the basename. ``.egg-info`` is handled at the prefix level
# (any directory ending in ``.egg-info``) — keep that in the predicate, not here.
MD5_EXCLUDE_FILENAMES: tuple[str, ...] = (
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "Thumbs.db",
)