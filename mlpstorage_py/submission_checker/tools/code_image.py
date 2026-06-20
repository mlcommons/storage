"""Code-image capture, load, and verification tooling.

Implements the capture-at-runtime and integrity-verification semantics
specified in Phase 1 and 2 of the MLPerf Storage Code-Image initiative.

Design decisions (D-01..D-20):
- D-01: Module lives at mlpstorage_py/submission_checker/tools/code_image.py.
- D-02: Public API: capture_code_image, load_code_image, verify_source_against_image,
  verify_image_self_consistent, find_source_root; CodeImage dataclass.
- D-03: Typed CodeImageError hierarchy for CLI mapping.
- D-04: find_source_root ascends to pyproject.toml.
- D-05: SourceRootNotFound raised at filesystem root.
- D-07: .code-hash.json schema (hash, algorithm, captured_at, mlpstorage_version, git_sha).
- D-08: git_sha captured via best-effort 'git rev-parse HEAD'.
- D-09: algorithm identifier 'md5-tree-v1' is stable.
- D-10: captured_at in canonical ISO-8601 UTC 'Z' form.
- D-11: Runtime check hashes live source against captured image.
- D-12: Submission check hashes captured tree against its own JSON.
- D-14: Missing JSON in existing code/ is a fatal error.
- D-15: Malformed JSON is a fatal error.
- D-16: Never silently re-capture an existing code/ image.
- D-17: Atomic capture via code.tmp/ then os.rename.
- D-18: Cleanup stale code.tmp/ before starting capture.
- D-19: JSON hash is computed from the captured copy, not live source.

Public API:
    find_source_root(start=None) -> Path
    capture_code_image(source_root, target_dir, log) -> CodeImage
    load_code_image(image_dir, log) -> CodeImage
    verify_source_against_image(source_root, image_dir, log) -> bool
    verify_image_self_consistent(image_dir, log) -> bool
    CodeImage (dataclass)
    CodeImageError (Exception)
"""

import datetime
import fnmatch
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mlpstorage_py import __version__ as MLPSTORAGE_VERSION
from mlpstorage_py.config import BENCHMARK_TYPES
from mlpstorage_py.errors import ConfigurationError, ErrorCode
from mlpstorage_py.rules.utils import (
    MLPSTORAGE_ORGNAME_ENVVAR,
    MLPSTORAGE_SYSTEMNAME_ENVVAR,
)
from .code_checksum import compute_code_tree_md5
from ..constants import MD5_EXCLUDE_FILENAMES, MD5_EXCLUDE_PREFIXES


# CLI subparser name → canonical on-disk type segment.
# generate_output_location() writes this same segment, so the captured code/
# must use it to live in the same submission tree. CLI names map to the
# BENCHMARK_TYPES enum value, whose .name is used as the on-disk segment for
# all four types.
_CLI_BENCHMARK_TO_TYPE: dict[str, BENCHMARK_TYPES] = {
    "training": BENCHMARK_TYPES.training,
    "checkpointing": BENCHMARK_TYPES.checkpointing,
    "vectordb": BENCHMARK_TYPES.vector_database,
    "kvcache": BENCHMARK_TYPES.kv_cache,
}

# On-disk type segment is the BENCHMARK_TYPES.name for every benchmark type.
_TYPE_TO_ONDISK_SEGMENT: dict[BENCHMARK_TYPES, str] = {
    BENCHMARK_TYPES.training: BENCHMARK_TYPES.training.name,
    BENCHMARK_TYPES.checkpointing: BENCHMARK_TYPES.checkpointing.name,
    BENCHMARK_TYPES.vector_database: BENCHMARK_TYPES.vector_database.name,
    BENCHMARK_TYPES.kv_cache: BENCHMARK_TYPES.kv_cache.name,
}

# Per-type "leaf attribute" on args. The OPEN capture/verify path includes
# this segment between <type>/ and code/ so each leaf — what the submitter
# would consider a single comparable result group — has its own code image.
#
#   training, checkpointing : per-<model>      → uses args.model
#   vector_database         : per-<index_type> → uses args.index_type
#                             (AISAQ results are not comparable to DISKANN
#                              or HNSW, so they live in separate trees).
#                             The index name is UPPERCASE on disk, matching
#                             args.index_type and summary.json.index_type.
#   kv_cache                : transitional —   → None (no leaf segment)
#                             code lives at <type>/code/ until the kv_cache
#                             directory/file structure below the prefix is
#                             finalized (per follow-up plan).
#
# None means "no leaf segment" — code is captured per benchmark type only.
_TYPE_TO_LEAF_ATTR: dict[BENCHMARK_TYPES, str | None] = {
    BENCHMARK_TYPES.training: "model",
    BENCHMARK_TYPES.checkpointing: "model",
    BENCHMARK_TYPES.vector_database: "index_type",
    BENCHMARK_TYPES.kv_cache: None,
}


class CodeImageError(Exception):
    """Base for all code-image capture/verify failures (D-03)."""


class MissingHashFile(CodeImageError):
    """.code-hash.json not found in an image directory (D-14)."""


class MalformedHashFile(CodeImageError):
    """.code-hash.json present but unparseable or invalid (D-15)."""


class SourceRootNotFound(CodeImageError):
    """find_source_root walked to filesystem root without finding pyproject.toml (D-05)."""


class CodeTreeUnreadable(CodeImageError):
    """compute_code_tree_md5 returned None for a tree that should be readable.

    Raised when a code/ or source tree exists but the hashing walk could not
    complete — e.g., a permission error mid-walk, or a path that is gone by
    the time the walk reaches it. Distinct from MissingHashFile (the
    `.code-hash.json` sidecar is missing) and SourceRootNotFound (no
    pyproject.toml ancestor) so the caller can log the right diagnostic.
    """


@dataclass(frozen=True)
class CodeImage:
    """In-memory representation of a captured code image (D-02)."""
    path: Path
    hash: str
    algorithm: str
    captured_at: str
    mlpstorage_version: str
    git_sha: str | None


# Private constants
_HASH_FILENAME = ".code-hash.json"
_TMP_SUFFIX = "code.tmp"
_CODE_DIRNAME = "code"
_ALGORITHM = "md5-tree-v1"
_GIT_TIMEOUT_SEC = 5
_HASH_HEX_LEN = 32
_GIT_SHA_LEN = 40

# POSIX-safe name pattern per Rules.md §2.1.1 + path-traversal guard for `.` / `..`
# (D-05; T-02-02-05 mitigation made INLINE per Gemini + plan-checker consensus, REVIEWS.md):
_SUBMITTER_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
# The regex above MATCHES the literal strings "." and "..". An additional explicit
# reject is required to prevent path-traversal exploits (Gemini + plan-checker
# consensus, REVIEWS.md). This is checked INLINE in capture_or_verify_code_image,
# not deferred to a follow-up.
_RESERVED_PATH_SEGMENTS = frozenset({".", ".."})

# Submission-mode gating sets (D-10).
_SUBMISSION_MODES = frozenset({"closed", "open"})
_SUBMISSION_COMMANDS = frozenset({"datasize", "datagen", "run"})


def find_source_root(start: Path | None = None) -> Path:
    """Ascend from start until a directory with pyproject.toml is found (D-04).

    Args:
        start: Directory to start searching from. Defaults to the directory
            containing this file.

    Returns:
        Absolute Path to the repository root.

    Raises:
        SourceRootNotFound: If the walk reaches the filesystem root.
    """
    curr = (start or Path(__file__)).resolve()
    if curr.is_file():
        curr = curr.parent

    while True:
        if (curr / "pyproject.toml").exists():
            return curr
        if curr.parent == curr:  # reached root
            break
        curr = curr.parent

    raise SourceRootNotFound(
        f"Could not find source root (pyproject.toml) ascending from {start or Path(__file__)}"
    )


def capture_code_image(source_root: Path, target_dir: Path, log) -> CodeImage:
    """Capture a frozen copy of source_root into target_dir/code/ (D-02, CAP-01/02).

    1. Removes any stale 'code.tmp/' in target_dir (D-18).
    2. Copies source_root into 'code.tmp/' minus exclusions (CAP-03/04).
    3. Hashes the captured copy (D-19, HASH-01).
    4. Writes .code-hash.json into 'code.tmp/' (CAP-05).
    5. Atomically renames 'code.tmp/' to 'code/' (D-17).

    Args:
        source_root: Root of the benchmark source tree.
        target_dir: Directory where the 'code/' subdirectory will be created.
        log: Logger object.

    Returns:
        A CodeImage instance representing the new capture.

    Raises:
        ConfigurationError: If MLPSTORAGE_VERSION resolved to the literal
            "unknown" sentinel (no installed dist metadata and no readable
            pyproject.toml) — refusing to stamp a degenerate version into
            .code-hash.json that would degrade submission-time forensics.
        CodeImageError: If target_dir/code/ already exists (D-16).
        SourceRootNotFound: If source_root is missing or hashing fails.
    """
    # Refuse to capture with a degenerate mlpstorage_version sentinel — fail
    # before any filesystem work so we leave no partial state behind.
    if MLPSTORAGE_VERSION == "unknown":
        raise ConfigurationError(
            "mlpstorage version could not be resolved (no installed distribution "
            "metadata and no readable pyproject.toml); refusing to capture with "
            "mlpstorage_version=\"unknown\" — install the package "
            "(pip install -e . / uv sync) or run from a checkout with pyproject.toml",
            code=ErrorCode.CONFIG_MISSING_REQUIRED,
        )

    code_dir = target_dir / _CODE_DIRNAME
    code_tmp = target_dir / _TMP_SUFFIX

    if code_dir.exists():
        raise CodeImageError(f"Code image already exists at {code_dir} (D-16)")

    if code_tmp.exists():
        log.warning("stale code.tmp/ at %s removed before capture (D-18)", code_tmp)
        shutil.rmtree(code_tmp)

    # Behavior 5: Exclusion delegated to identical logic as hash
    _atomic_capture(source_root, code_tmp, log)

    # D-17 atomicity contract: code.tmp/ must be removed on ANY failure
    # between copy and rename — otherwise the next attempt finds a stale
    # tmp tree and only logs a warning. Wrap hash + JSON-write + rename in
    # try/except BaseException so KeyboardInterrupt / SystemExit also clean up.
    try:
        # Behavior 3/4: Hash the captured copy
        digest = compute_code_tree_md5(str(code_tmp), log)
        if digest is None:
            # This shouldn't happen if _atomic_capture succeeded, but for safety:
            raise SourceRootNotFound(f"Failed to hash captured tree at {code_tmp}")

        # Behavior 6: Build payload
        payload = {
            "hash": digest,
            "algorithm": _ALGORITHM,
            "captured_at": _now_utc_iso(),
            "mlpstorage_version": MLPSTORAGE_VERSION,
            "git_sha": _resolve_git_sha(source_root, log),
        }

        # Behavior 6: Write JSON
        _write_hash_file(code_tmp, payload, log)

        # Behavior 4: Atomic rename
        os.rename(str(code_tmp), str(code_dir))
    except BaseException:
        if code_tmp.exists():
            shutil.rmtree(code_tmp, ignore_errors=True)
        raise

    return CodeImage(path=code_dir, **payload)


def load_code_image(image_dir: Path, log) -> CodeImage:
    """Read and validate .code-hash.json from an image directory (D-02, D-14, D-15).

    Args:
        image_dir: Path to the 'code/' directory.
        log: Logger object.

    Returns:
        CodeImage instance.

    Raises:
        MissingHashFile: If .code-hash.json is absent.
        MalformedHashFile: If JSON is invalid or missing required fields.
    """
    data = _read_hash_file(image_dir, log)
    return CodeImage(path=image_dir, **data)


def verify_source_against_image(source_root: Path, image_dir: Path, log) -> bool:
    """Compare live source tree against a captured image (D-11, VALR-01..04).

    Args:
        source_root: Path to the running benchmark source.
        image_dir: Path to the captured 'code/' directory.
        log: Logger object.

    Returns:
        True if hashes match, False otherwise.

    Raises:
        CodeTreeUnreadable: If source_root exists but the hashing walk could
            not complete (permission error mid-walk, etc.).
        MissingHashFile / MalformedHashFile: If image_dir is missing or has
            an invalid `.code-hash.json` (via load_code_image).
    """
    img = load_code_image(image_dir, log)
    current_hash = compute_code_tree_md5(str(source_root), log)
    if current_hash is None:
        # IN-02: previously raised SourceRootNotFound, but that exception is
        # reserved for "walked to filesystem root without finding pyproject.toml"
        # (D-05) — a structural CLI / config error. compute_code_tree_md5
        # returning None means the walk itself failed, not that source_root
        # is structurally invalid. Use CodeTreeUnreadable instead.
        raise CodeTreeUnreadable(
            f"Source root could not be hashed (unreadable or vanished mid-walk): {source_root}"
        )

    return current_hash == img.hash


def verify_image_self_consistent(image_dir: Path, log) -> bool:
    """Verify that a captured 'code/' tree matches its own recorded hash (D-12, VALS-02/04).

    Used by the submission validator to detect post-capture tampering.

    Args:
        image_dir: Path to the captured 'code/' directory.
        log: Logger object.

    Returns:
        True if the tree hash matches .code-hash.json, False otherwise.

    Raises:
        MissingHashFile: If .code-hash.json is absent (via load_code_image).
        MalformedHashFile: If .code-hash.json is unparseable (via load_code_image).
        CodeTreeUnreadable: If the image_dir tree itself cannot be hashed
            (permission error mid-walk, gone by the time we walk, etc.).
    """
    img = load_code_image(image_dir, log)
    actual_hash = compute_code_tree_md5(str(image_dir), log)
    if actual_hash is None:
        # IN-01: previously raised MissingHashFile here, but load_code_image
        # already succeeded — the JSON IS present. The real failure is that
        # the tree itself didn't hash. Use CodeTreeUnreadable so the log
        # message names the actual root cause.
        raise CodeTreeUnreadable(
            f"Captured code directory is missing or unreadable: {image_dir}"
        )

    return actual_hash == img.hash


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _atomic_capture(source_root: Path, target_dir: Path, log) -> None:
    """Copy source_root to target_dir using identical exclusion logic as hashing (Behavior 5)."""
    source_str = str(source_root)
    # shutil.copytree(..., dirs_exist_ok=True) below creates target_dir on its
    # own (Python ≥3.8). No need to pre-mkdir — keeping the call shrinks the
    # window in which target_dir can be in a partial state when copytree starts.

    # We use shutil.copytree with a custom ignore function to replicate the
    # predicate's exclusion logic exactly.
    def ignore_logic(curr_dir, contents):
        ignored = set()
        # Rel_dir from source_root for prefix matching
        rel_dir = os.path.relpath(curr_dir, source_str).replace(os.sep, "/")
        if rel_dir == ".":
            rel_dir = ""
        else:
            rel_dir += "/"

        for name in contents:
            full_item = os.path.join(curr_dir, name)
            rel_item = rel_dir + name

            # 1. Directory exclusion (Prefixes or .egg-info)
            if os.path.isdir(full_item):
                # Match if basename is in prefixes (stripped) OR if rel_path starts with prefix
                item_prefix = rel_item + "/"
                if any(item_prefix.startswith(p) for p in MD5_EXCLUDE_PREFIXES) or \
                   any(name == p.rstrip("/") for p in MD5_EXCLUDE_PREFIXES):
                    ignored.add(name)
                    continue
                # .egg-info handled specially in predicate
                if name.endswith(".egg-info"):
                    ignored.add(name)
                    continue
            else:
                # 2. Filename-based exclusion
                if any(fnmatch.fnmatch(name, pat) for pat in MD5_EXCLUDE_FILENAMES):
                    ignored.add(name)
                    continue
                # 3. Symlinks (hash skips them, so capture must skip them to stay consistent)
                if os.path.islink(full_item):
                    ignored.add(name)
                    continue
                # 4. Belt-and-suspenders: check if file is in an excluded dir (rel_item prefix match)
                if any(rel_item.startswith(p) for p in MD5_EXCLUDE_PREFIXES):
                    ignored.add(name)
                    continue

        return ignored

    # symlinks=True preserves symlinks in the copy (though we ignore them above).
    # Re-using shutil.copytree is more robust than a manual walk for edge cases.
    # Note: we already ignored symlinks in our ignore_logic to match hash behavior.
    shutil.copytree(source_root, target_dir, symlinks=True, ignore=ignore_logic, dirs_exist_ok=True)


def _write_hash_file(image_dir: Path, payload: dict, log) -> None:
    """Write .code-hash.json with fixed field order (D-07)."""
    hash_path = image_dir / _HASH_FILENAME
    # Ensure field order per specifics §1
    ordered = {
        "hash": payload["hash"],
        "algorithm": payload["algorithm"],
        "captured_at": payload["captured_at"],
        "mlpstorage_version": payload["mlpstorage_version"],
        "git_sha": payload["git_sha"],
    }
    with open(hash_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)
        f.write("\n")


def _read_hash_file(image_dir: Path, log) -> dict:
    """Read and validate the JSON file (D-15)."""
    hash_path = image_dir / _HASH_FILENAME
    if not hash_path.is_file():
        raise MissingHashFile(f"Required file {_HASH_FILENAME} not found at {hash_path}")

    try:
        with open(hash_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise MalformedHashFile(f"Failed to parse {_HASH_FILENAME} at {hash_path}: {e}")

    # Validation
    required = ["hash", "algorithm", "captured_at", "mlpstorage_version", "git_sha"]
    for field in required:
        if field not in data:
            raise MalformedHashFile(f"Missing required field '{field}' in {hash_path}")

    if data["algorithm"] != _ALGORITHM:
        raise MalformedHashFile(f"Unknown algorithm '{data['algorithm']}' (expected '{_ALGORITHM}') in {hash_path}")

    if not re.fullmatch(r"[0-9a-f]{" + str(_HASH_HEX_LEN) + r"}", data["hash"]):
        raise MalformedHashFile(f"Invalid MD5 hash format in {hash_path}")

    if data["git_sha"] is not None:
        if not re.fullmatch(r"[0-9a-f]{" + str(_GIT_SHA_LEN) + r"}", data["git_sha"]):
            raise MalformedHashFile(f"Invalid git_sha format in {hash_path}")

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", data["captured_at"]):
        raise MalformedHashFile(f"Invalid captured_at timestamp format in {hash_path}")

    return data


def _resolve_git_sha(source_root: Path, log) -> str | None:
    """Best-effort capture of HEAD SHA (D-08)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(source_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SEC,
            shell=False,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            if re.fullmatch(r"[0-9a-f]{" + str(_GIT_SHA_LEN) + r"}", sha):
                return sha
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("Failed to resolve git SHA in %s: %s (D-08)", source_root, e)
    
    return None


def _now_utc_iso() -> str:
    """Return canonical ISO-8601 UTC 'Z' timestamp (D-10)."""
    return datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# CLI dispatch helper (Phase 2 — D-07..D-10, D-20, D-21)
# ---------------------------------------------------------------------------

def capture_or_verify_code_image(args, env, log):
    """Capture-or-verify the code image at the submission tree (D-07..D-10).

    The single CLI dispatch chokepoint that owns the entire CAP/VALR contract:

    - Gates on `(args.mode, args.command)`: returns None unless mode is in
      {closed, open} AND command is in {datasize, datagen, run} (D-10).
    - Reads + validates MLPSTORAGE_ORGNAME (and MLPSTORAGE_SYSTEMNAME for OPEN)
      from `env` — this helper is the SOLE reader of those env vars in the
      codebase (Gemini MEDIUM trust-contract finding closed; D-05).
    - Applies POSIX regex (Rules.md §2.1.1) AND inline `.`/`..` path-traversal
      guard for both orgname and systemname (T-02-02-05 mitigation, REVIEWS.md
      consensus finding).
    - Computes the image-parent path matching `generate_output_location`'s
      prefix (Plan 01, D-03). Stores validated values on `args` so downstream
      `generate_output_location` callers can read them without re-reading env.
    - Captures (CAP-01/02/06) on first call, verifies (VALR-01/03 success,
      VALR-02/04 mismatch) on subsequent calls. Re-raises Phase 1 typed errors
      (MissingHashFile, MalformedHashFile) after logging the D-21 recovery
      message; mismatch raises CodeImageError with the literal spec string.

    Args:
        args: argparse.Namespace-like with attributes `mode`, `command`,
            `results_dir`, `benchmark`, `model`.
        env: Mapping (e.g., os.environ) used to look up MLPSTORAGE_* env vars.
        log: Logger object with status/error/info/warning/debug methods.

    Returns:
        Path | None: The captured/verified `code/` directory path, or None
        when gated off.

    Raises:
        ConfigurationError: Missing or invalid MLPSTORAGE_* env var.
        CodeImageError: Hash mismatch (VALR-02/04) — main() maps to
            EXIT_CODE.CODE_IMAGE_ERROR.
        MissingHashFile / MalformedHashFile: Existing code/ has missing or
            unparseable .code-hash.json (D-21) — main() maps to exit code 2.
        SourceRootNotFound: Live source tree could not be located/hashed.

    Notes:
        D-07..D-10, D-20, D-21; inline path-traversal guard per REVIEWS.md
        consensus finding (T-02-02-05). This helper is the SOLE reader of
        MLPSTORAGE_ORGNAME / MLPSTORAGE_SYSTEMNAME env vars.
    """
    # 1. Gate by mode (D-10) — return None for whatif/reports/validate/etc.
    mode = getattr(args, "mode", None)
    if mode not in _SUBMISSION_MODES:
        return None

    # 2. Gate by command (D-10) — return None for configview/etc. under
    # closed|open modes (e.g., `mlpstorage closed configview`).
    command = getattr(args, "command", None)
    if command not in _SUBMISSION_COMMANDS:
        return None

    # 3. Read + validate orgname (D-04, D-05).
    orgname = env.get(MLPSTORAGE_ORGNAME_ENVVAR)
    if not orgname:
        raise ConfigurationError(
            "MLPSTORAGE_ORGNAME environment variable is required for closed|open runs",
            parameter=MLPSTORAGE_ORGNAME_ENVVAR,
            suggestion=(
                "export MLPSTORAGE_ORGNAME=<your_org>  "
                "# future: mlpstorage init <orgname> <results_dir>"
            ),
            code=ErrorCode.CONFIG_MISSING_REQUIRED,
        )
    if not _SUBMITTER_NAME_RE.match(orgname):
        raise ConfigurationError(
            f"MLPSTORAGE_ORGNAME={orgname!r} is not a POSIX-filename-safe identifier "
            f"(Rules.md §2.1.1: ^[A-Za-z0-9._-]+$)",
            parameter=MLPSTORAGE_ORGNAME_ENVVAR,
            suggestion="Use only letters, digits, '.', '_', or '-'",
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )
    # INLINE path-traversal guard for orgname (CONSENSUS FINDING — REVIEWS.md).
    # The regex `^[A-Za-z0-9._-]+$` accepts `.` and `..` literally, so an
    # additional explicit reject is REQUIRED. The substring `"'.' and '..'
    # are reserved path segments"` is the spec contract used by Plan 05's tests.
    if orgname in _RESERVED_PATH_SEGMENTS:
        raise ConfigurationError(
            f"MLPSTORAGE_ORGNAME={orgname!r} is not a permitted value: "
            f"'.' and '..' are reserved path segments",
            parameter=MLPSTORAGE_ORGNAME_ENVVAR,
            suggestion="Choose an orgname that is not '.' or '..'",
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )

    # 4. For OPEN, also read + validate systemname.
    systemname = None
    if mode == "open":
        systemname = env.get(MLPSTORAGE_SYSTEMNAME_ENVVAR)
        if not systemname:
            raise ConfigurationError(
                "MLPSTORAGE_SYSTEMNAME environment variable is required for open runs",
                parameter=MLPSTORAGE_SYSTEMNAME_ENVVAR,
                suggestion=(
                    "export MLPSTORAGE_SYSTEMNAME=<your_system>  "
                    "# future: per-command --system-name flag"
                ),
                code=ErrorCode.CONFIG_MISSING_REQUIRED,
            )
        if not _SUBMITTER_NAME_RE.match(systemname):
            raise ConfigurationError(
                f"MLPSTORAGE_SYSTEMNAME={systemname!r} is not a POSIX-filename-safe identifier "
                f"(Rules.md §2.1.1: ^[A-Za-z0-9._-]+$)",
                parameter=MLPSTORAGE_SYSTEMNAME_ENVVAR,
                suggestion="Use only letters, digits, '.', '_', or '-'",
                code=ErrorCode.CONFIG_INVALID_VALUE,
            )
        # INLINE path-traversal guard for systemname (CONSENSUS FINDING — REVIEWS.md).
        if systemname in _RESERVED_PATH_SEGMENTS:
            raise ConfigurationError(
                f"MLPSTORAGE_SYSTEMNAME={systemname!r} is not a permitted value: "
                f"'.' and '..' are reserved path segments",
                parameter=MLPSTORAGE_SYSTEMNAME_ENVVAR,
                suggestion="Choose a systemname that is not '.' or '..'",
                code=ErrorCode.CONFIG_INVALID_VALUE,
            )

    # 5. Stash validated values on args so downstream generate_output_location
    # callers can consume them without re-reading env (closes the Gemini MEDIUM
    # trust-contract finding — this helper remains the sole env reader).
    args._validated_orgname = orgname
    args._validated_systemname = systemname

    # 6. Compute image_parent — MUST match Plan 01's generate_output_location
    # prefix. The helper only creates the {closed|open}/<orgname>/.../code/
    # subtree inside the already-existing results-directory (D-06); creating
    # the results-directory itself is reserved for the future
    # `mlpstorage init` command.
    results_dir = Path(args.results_dir)
    # IN-03: enforce the "results_dir must already exist" contract from the
    # comment above. Without this gate, image_parent.mkdir(parents=True, ...)
    # below silently creates results_dir if absent, diverging from the
    # documented behavior.
    if not results_dir.exists():
        raise ConfigurationError(
            f"results_dir {str(results_dir)!r} does not exist; the code-image "
            f"helper does not create it (reserved for future `mlpstorage init`)",
            parameter="--results-dir",
            suggestion=f"mkdir -p {str(results_dir)!r} before running, "
                       f"or point --results-dir at an existing directory",
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )
    if mode == "closed":
        image_parent = results_dir / "closed" / orgname
    else:  # mode == "open"
        # Canonicalize the per-type segment via _CLI_BENCHMARK_TO_TYPE +
        # _TYPE_TO_ONDISK_SEGMENT so the captured code/ shares the on-disk
        # tree with generate_output_location's output. The CLI subparser
        # names 'vectordb' and 'kvcache' diverge from the on-disk segments
        # ('vector_database' and 'kv_cache') — without these lookups the
        # captured code/ would live in a different tree than the runtime's
        # results.
        # Use getattr(..., None) + typed raise rather than bare getattr.
        # A bare getattr surfaces AttributeError, which the main.py exit-code
        # mapping treats as an unhandled crash rather than CodeImageError.
        cli_benchmark = getattr(args, "benchmark", None)
        if cli_benchmark is None:
            raise CodeImageError(
                "args.benchmark is required for capture-or-verify in OPEN mode"
            )
        try:
            benchmark_type = _CLI_BENCHMARK_TO_TYPE[cli_benchmark]
        except KeyError:
            raise CodeImageError(
                f"Unknown benchmark CLI name {cli_benchmark!r} — "
                f"expected one of {sorted(_CLI_BENCHMARK_TO_TYPE)}"
            ) from None
        ondisk_segment = _TYPE_TO_ONDISK_SEGMENT[benchmark_type]
        leaf_dir = (
            results_dir / "open" / orgname / "results" / systemname
            / ondisk_segment
        )
        # Per-type leaf segment (see _TYPE_TO_LEAF_ATTR for the design rationale).
        leaf_attr = _TYPE_TO_LEAF_ATTR[benchmark_type]
        if leaf_attr is not None:
            leaf_value = getattr(args, leaf_attr, None)
            if leaf_value is None:
                raise CodeImageError(
                    f"args.{leaf_attr} is required for "
                    f"{benchmark_type.name} OPEN capture"
                )
            leaf_dir = leaf_dir / leaf_value
        image_parent = leaf_dir
    image_parent.mkdir(parents=True, exist_ok=True)

    # 7. Branch capture-vs-verify (D-08).
    code_dir = image_parent / _CODE_DIRNAME
    source_root = find_source_root()

    if not code_dir.exists():
        capture_code_image(source_root, image_parent, log)
        log.status(f"Captured code image at {code_dir}")
        return code_dir

    # code_dir exists → verify path. Catch missing/malformed .code-hash.json
    # so we can attach the D-21 actionable recovery message before re-raising.
    try:
        matched = verify_source_against_image(source_root, code_dir, log)
    except (MissingHashFile, MalformedHashFile) as e:
        log.error(str(e))
        log.error(f"code image at: {code_dir}")
        log.error(
            "either delete `code/` and re-run to re-capture, "
            "or restore the original capture."
        )
        raise

    if matched:
        log.status(f"code unchanged from on-file image at {code_dir}")
        return code_dir

    # Hash mismatch — emit the literal spec string by mode (VALR-02 / VALR-04).
    if mode == "closed":
        msg = "changes to the codebase are not allowed in a CLOSED run"
    else:  # mode == "open"
        msg = "all runs of this type must use the same codebase"
    log.error(msg)
    log.error(f"code image at: {code_dir}")
    raise CodeImageError(msg)
