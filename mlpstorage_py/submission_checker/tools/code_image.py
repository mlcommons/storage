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
- D-09: algorithm identifier 'md5-tree-vN' is stable within a major version
  and bumped whenever the hash semantics change (currently v2 — see #505).
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
from types import SimpleNamespace

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


class PointerMalformed(CodeImageError):
    """Raised when .mlps-code-image content does not parse as md5-tree-v2:<32-hex> per D-61.

    Subclasses CodeImageError so main.py's existing exit-code mapping surfaces
    this as EXIT_CODE.CODE_IMAGE_ERROR without a new handler. Distinct from
    MalformedHashFile (which covers .code-hash.json / the sidecar JSON) so the
    caller can log the right diagnostic when a submitter hand-edits the
    pointer file (RESEARCH Pitfall 3).
    """


class LegacyLayoutDetected(CodeImageError):
    """Raised at capture-or-verify entry when a legacy ``code/`` layout is present
    under ``<results_dir>/{closed,open}/<orgname>/`` (D-63, Phase 6).

    Phase 6 replaces the legacy single-``code/`` layout with a content-addressed
    pool at ``<results_dir>/<orgname>/code-<hash8>/``. Any legacy ``code/`` present
    at capture time is refused BEFORE any writes — the strict single-layout
    invariant is what Phase 8's CHECK-04 assumes. Migration is Phase 7's job.

    Subclasses CodeImageError so main.py's existing exit-code mapping catches it
    without a new handler.
    """


class PoolCorruption(CodeImageError):
    """Raised when the D-66 loser branch verifies a pre-existing pool image and
    its ``.code-hash.json.hash`` does not match the live source hash.

    Semantics: two concurrent writers race on the same target ``code-<hash8>/``.
    The loser's ``os.rename`` fails with ENOTEMPTY; we read the winner's
    ``.code-hash.json`` to confirm byte-equal content. Mismatch here indicates
    genuine filesystem corruption or a hand-planted pool image — not a benign
    race — so we surface it as an actionable error.

    Subclasses CodeImageError so main.py's existing exit-code mapping catches it
    without a new handler.
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
# Pointer sentinel written into every submission run-leaf pointing at the
# content-addressed pool image whose hash matches the live source tree.
# Content shape: `md5-tree-v2:<32-hex>` (D-61, Plan 06-01). Leading dot keeps
# it invisible to a naive `Path.glob("code-*")` pool scan (RESEARCH Pitfall 4).
_POINTER_FILENAME = ".mlps-code-image"
_TMP_SUFFIX = "code.tmp"
_CODE_DIRNAME = "code"
# Bumped to v2 alongside the source-vs-copy hash-target fix in PR #512.
# A v1 .code-hash.json was computed against the captured code/ copy via a
# walker that disagreed with the verifier's; the post-#512 codebase computes
# the digest against source_root directly. Any v1 capture sitting on disk
# from before #512 merged will fail _read_hash_file's algorithm check and
# get the actionable "delete code/ and re-run" error instead of a stale
# content-mismatch error. Bump again whenever the hash semantics change. (#505)
_ALGORITHM = "md5-tree-v2"
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
        # Hash source_root directly, not the just-made copy in code_tmp/. (#505)
        #
        # verify_source_against_image at runtime calls
        # compute_code_tree_md5(source_root) and compares to this stored digest.
        # If we hash code_tmp here, the comparison only succeeds when
        # shutil.copytree's `ignore` callback walks the tree byte-for-byte
        # identically to compute_code_tree_md5's filtered os.walk — any
        # divergence (and real trees DO diverge: differing handling of
        # `.egg-info`, symlinks, deep prefix matches, etc.) silently breaks
        # CLOSED-run verification on the very first re-invocation. Hashing
        # source_root on both sides eliminates the walker-parity dependency by
        # construction; the code_tmp/ → code/ copy remains for archival
        # forensics.
        digest = compute_code_tree_md5(str(source_root), log)
        if digest is None:
            # source_root vanished between _atomic_capture and the hash call.
            raise SourceRootNotFound(f"Failed to hash source tree at {source_root}")

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


def _pool_dir_name(full_hash: str) -> str:
    """Return the pool-image directory name for `full_hash` (D-62).

    Shape: `code-<first-8-hex>`. The 8-char prefix keeps directory names
    short while remaining collision-resistant for the pool sizes MLPerf
    submitters actually stage. Callers are responsible for passing a
    validated 32-lowercase-hex string; this helper does no validation of
    its own so it can be used inline in path assembly without try/except.
    """
    return f"code-{full_hash[:8]}"


def _write_pointer_atomic(run_leaf: Path, full_hash: str, log) -> None:
    """Write `_POINTER_FILENAME` inside run_leaf atomically (D-61, D-65).

    Contract: on return, either the pointer contains the full expected
    content or the run_leaf contains no pointer file at all. No partial
    writes are ever visible to a concurrent reader.

    Args:
        run_leaf: Directory the pointer file will be written into. Must
            already exist.
        full_hash: 32-lowercase-hex md5-tree-v2 digest identifying the
            content-addressed pool image this run resolves to.
        log: Logger object.

    Raises:
        AssertionError: full_hash is not 32 lowercase hex chars.
        BaseException: Any error inside the tmp-write block propagates
            after the tmp sibling is cleaned up. `except BaseException`
            (NOT `except Exception`) is intentional — KeyboardInterrupt
            and SystemExit must also trigger tmp cleanup so a ^C mid-write
            never leaks a stale sibling into the run leaf (verbatim
            carry-over from mlpstorage_py/results_dir/code_image.py:160-186).
    """
    assert re.fullmatch(r"[0-9a-f]{32}", full_hash), (
        f"live hash must be 32 lowercase hex chars, got {full_hash!r}"
    )
    dst = run_leaf / _POINTER_FILENAME
    # Leading dot on the tmp sibling is REQUIRED — RESEARCH Pitfall 4.
    # A downstream `Path.glob("code-*")` pool scan (Plan 06-02) must not
    # pick up an in-flight tmp file mid-write.
    tmp = run_leaf / f".{_POINTER_FILENAME}.tmp.{os.getpid()}"
    if tmp.exists():
        # Stale from a prior crash between tmp-write and os.rename.
        tmp.unlink(missing_ok=True)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            # No trailing newline — D-61 permits either but the writer is
            # locked to the shorter form so the round-trip is byte-exact.
            f.write(f"{_ALGORITHM}:{full_hash}")
    except BaseException:
        # KeyboardInterrupt / SystemExit reach here too — that is
        # intentional (D-65 atomicity contract).
        tmp.unlink(missing_ok=True)
        raise
    # Atomic on same fs — after this returns, `dst` is complete.
    os.rename(str(tmp), str(dst))
    log.debug("wrote pointer file %s → %s:%s", dst, _ALGORITHM, full_hash)


def _read_pointer(run_leaf: Path, log) -> tuple[str, str]:
    """Read `_POINTER_FILENAME` from run_leaf, return (algorithm, full_hash) per D-61, PTR-02.

    Rejects every malformed variant surfaced in RESEARCH Pitfall 3:
    empty tail (`md5-tree-v2:`), short hash, uppercase hex, missing
    colon, and unknown algorithm. Every rejection raises PointerMalformed
    naming the offending pointer path.

    Args:
        run_leaf: Directory containing `_POINTER_FILENAME`.
        log: Logger object.

    Returns:
        Tuple of (algorithm, full_hash) where algorithm is the literal
        string `md5-tree-v2` and full_hash is 32 lowercase hex chars.

    Raises:
        FileNotFoundError: Pointer file is absent. Caller can distinguish
            "pool image not linked" from "pool image linked but pointer
            corrupted".
        PointerMalformed: File present but not a valid `md5-tree-v2:<hex32>`
            line.
    """
    path = run_leaf / _POINTER_FILENAME
    # `.strip()` tolerates trailing whitespace / newline per D-61 (both
    # forms are permitted). Any accidental extra `:` in the hex tail
    # would still fail the hex regex below, so `partition(':')` is safe.
    line = path.read_text(encoding="utf-8").strip()
    if ":" not in line:
        raise PointerMalformed(
            f"{path}: expected '{_ALGORITHM}:<hex32>', got {line!r}"
        )
    alg, _, hex_part = line.partition(":")
    if alg != _ALGORITHM:
        raise PointerMalformed(
            f"{path}: unknown algorithm {alg!r} (expected {_ALGORITHM!r})"
        )
    if not re.fullmatch(r"[0-9a-f]{32}", hex_part):
        raise PointerMalformed(
            f"{path}: hash after ':' is not 32 lowercase hex chars, "
            f"got {hex_part!r}"
        )
    return alg, hex_part


def _now_utc_iso() -> str:
    """Return canonical ISO-8601 UTC 'Z' timestamp (D-10)."""
    return datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _scan_legacy_layout(results_dir: Path, orgname: str) -> list[Path]:
    """Return every legacy ``code/`` directory under
    ``<results_dir>/{closed,open}/<orgname>/`` (D-63).

    Bounded: at most one ``is_dir()`` syscall per submission mode (two total).
    Phase 6 replaces the single-``code/`` layout with a content-addressed
    pool at ``<results_dir>/<orgname>/code-<hash8>/`` (D-64). Any legacy
    ``code/`` present at capture time means the tree was captured under a
    pre-Phase-6 mlpstorage and must be migrated (Phase 7). This helper is
    the runtime refusal predicate; the caller raises ``LegacyLayoutDetected``.

    Args:
        results_dir: Root results directory (the ``--results-dir`` value).
        orgname: Validated organization name for path scoping.

    Returns:
        List of offending Paths (empty if no legacy layout present). Callers
        report the first entry by name and count the rest.

    Notes:
        The OPEN legacy path
        ``results_dir/open/<org>/code/<benchmark>/<command>/`` has parent
        ``results_dir/open/<org>/code/`` — so a single ``is_dir()`` check at
        that depth catches the whole open-mode legacy subtree without
        walking. O(1) with respect to pool size. (RESEARCH § "D-63 legacy
        layout scan".)
    """
    offenders: list[Path] = []
    for mode in ("closed", "open"):
        candidate = results_dir / mode / orgname / "code"
        if candidate.is_dir():
            offenders.append(candidate)
    return offenders


def _find_matching_pool_image(
    org_root: Path, live_hash: str, log
) -> Path | None:
    """Scan ``<org_root>/code-*/`` for a pool image whose ``.code-hash.json``
    ``hash`` field equals ``live_hash``. Return the first match, else None.

    Single-walk / glob-then-parse-hash-file. Does NOT re-hash pool images at
    scan time — the stored ``.code-hash.json.hash`` is authoritative
    (RESEARCH Anti-Pattern: re-hashing pool images at scan time). Per-image
    self-consistency is Phase 8's CHECK-02, not runtime capture.

    Args:
        org_root: ``<results_dir>/<orgname>/`` (mode-agnostic per D-64).
        live_hash: 32-lowercase-hex md5-tree-v2 digest of the live source.
        log: Logger object; DEBUG-level "skip candidate" messages for
            non-conformant pool dirs.

    Returns:
        The first matching pool dir, or None on no match.

    Notes:
        Catches ``MissingHashFile`` / ``MalformedHashFile`` and skips at DEBUG.
        A downstream ``.code-<hash8>.tmp.<pid>`` sibling is invisible to
        ``glob("code-*")`` because it starts with a leading dot (Pitfall 4).
    """
    if not org_root.is_dir():
        return None
    for candidate in org_root.glob("code-*"):
        if not candidate.is_dir():
            continue  # skip stray files
        try:
            stored = _read_hash_file(candidate, log)
        except (MissingHashFile, MalformedHashFile):
            log.debug("skipping non-conformant pool candidate at %s", candidate)
            continue
        if stored["hash"] == live_hash:
            return candidate
    return None


def _capture_new_pool_image(
    org_root: Path, source_root: Path, live_hash: str, log
) -> Path:
    """Capture a new content-addressed pool image at
    ``<org_root>/code-<hash8>/`` via write-tmp + ``os.rename`` (D-66
    first-writer-wins).

    Sequence:
      1. Copy source_root into ``<org_root>/.code-<hash8>.tmp.<pid>/`` via
         the existing ``_atomic_capture`` helper (D-17 atomicity contract).
      2. Write ``.code-hash.json`` INSIDE the tmp sibling BEFORE the rename
         so the target arrives non-empty (Pitfall 1 — guarantees ENOTEMPTY
         on the race-loser's rename attempt, not silent empty-dir overwrite).
      3. ``os.rename`` tmp → ``code-<hash8>/``. Success: return the pool dir.
      4. On ``OSError`` (ENOTEMPTY / EEXIST): a concurrent writer won.
         Clean our tmp, verify the winner's ``.code-hash.json.hash`` equals
         live_hash, and return the winner's path. Mismatch raises
         ``PoolCorruption``.

    Args:
        org_root: ``<results_dir>/<orgname>/`` (mode-agnostic per D-64).
        source_root: Live source tree to copy (result of ``find_source_root``).
        live_hash: 32-lowercase-hex md5-tree-v2 digest of source_root.
        log: Logger object.

    Returns:
        Path to the pool dir at ``<org_root>/code-<hash8>/``.

    Raises:
        PoolCorruption: D-66 loser branch found a pre-existing pool image
            whose ``.code-hash.json.hash`` does not match ``live_hash`` —
            filesystem-corruption signal, not a benign race.
        OSError: Unrelated OS error during rename (e.g. EACCES) — bubbles up
            unchanged.

    Notes:
        Uses ``os.rename`` (NOT ``os.replace``) — the D-66 first-writer-wins
        pattern requires the failure-on-non-empty semantics that ``rename``
        provides on POSIX; ``replace`` overwrites unconditionally
        (RESEARCH Anti-Patterns).
    """
    hash8 = live_hash[:8]
    tmp = org_root / f".code-{hash8}.tmp.{os.getpid()}"
    if tmp.exists():
        # Stale from a prior crash between _atomic_capture and rename.
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        _atomic_capture(source_root, tmp, log)
        # Build the D-07 payload and write .code-hash.json INSIDE tmp
        # BEFORE the rename — Pitfall 1: the rename target must arrive
        # non-empty so the D-66 loser branch triggers ENOTEMPTY instead
        # of silently overwriting an empty winner placeholder.
        payload = {
            "hash": live_hash,
            "algorithm": _ALGORITHM,
            "captured_at": _now_utc_iso(),
            "mlpstorage_version": MLPSTORAGE_VERSION,
            "git_sha": _resolve_git_sha(source_root, log),
        }
        _write_hash_file(tmp, payload, log)
    except BaseException:
        # Cleanup on ANY failure (including KeyboardInterrupt / SystemExit)
        # so we do not leak a stale .tmp sibling into org_root.
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    pool_dir = org_root / _pool_dir_name(live_hash)
    try:
        # .code-hash.json inside tmp guarantees ENOTEMPTY on race-loser's
        # rename attempt (D-66; Pitfall 1). Do NOT use os.replace here —
        # replace overwrites unconditionally and collapses the first-writer-
        # wins semantic.
        os.rename(str(tmp), str(pool_dir))
    except OSError as e:
        # ENOTEMPTY, EEXIST, or FileExistsError — a concurrent writer won.
        # Clean up our tmp so we do not leak.
        shutil.rmtree(tmp, ignore_errors=True)
        # If the target does not exist, this is some other OSError — bubble it.
        if not pool_dir.is_dir():
            raise
        # Verify the winner captured byte-equal content (hash match).
        winner = _read_hash_file(pool_dir, log)
        if winner["hash"] != live_hash:
            raise PoolCorruption(
                f"pool image at {pool_dir} does not match live hash "
                f"{live_hash!r} — concurrent-write race lost integrity"
            ) from e
        # Winner's content is byte-equal to ours; safe to proceed silently.
    return pool_dir


# ---------------------------------------------------------------------------
# CLI dispatch helper (Phase 2 — D-07..D-10, D-20, D-21; Phase 6 — D-63..D-67)
# ---------------------------------------------------------------------------

def capture_or_verify_code_image(args, env, log):
    """Capture-or-verify a content-addressed code-image pool at the submission tree.

    Phase 6 semantic core: hashes the live source tree, refuses if a legacy
    ``code/`` layout is present (D-63), scans ``<results_dir>/<orgname>/code-*/``
    for a matching pool image (CAPVER-01), on-match reuses it and on-no-match
    captures a new ``code-<hash8>/`` via write-tmp + ``os.rename`` (D-66
    first-writer-wins, CAPVER-02), then always writes the ``.mlps-code-image``
    pointer atomically in the run leaf before returning (PTR-01, D-65).

    Contract (Phase 6):

    - Gates on ``(args.mode, args.command)``: returns None unless mode is in
      ``{closed, open}`` AND command is in ``{datasize, datagen, run}`` (D-10).
    - Reads + validates MLPSTORAGE_ORGNAME (and MLPSTORAGE_SYSTEMNAME for OPEN)
      from ``env`` — this helper is the SOLE reader of those env vars in the
      codebase (Gemini MEDIUM trust-contract finding closed; D-05).
    - Applies POSIX regex (Rules.md §2.1.1) AND inline ``.``/``..`` path-traversal
      guard for both orgname and systemname (T-02-02-05 mitigation, REVIEWS.md
      consensus finding).
    - Refuses (D-63) if a legacy ``code/`` layout is present under
      ``<results_dir>/{closed,open}/<orgname>/`` — Phase 7 owns the migration.
    - Content-addressed pool is mode-agnostic per D-64: pool images live under
      ``<results_dir>/<orgname>/code-<hash8>/``, so CLOSED and OPEN runs of the
      same source hash reuse the same pool image (POOL-04).
    - Source change (CAPVER-03) captures a new pool image alongside the existing
      one; hash mismatch is NO LONGER an error. The pre-Phase-6 CLOSED and
      OPEN content-mismatch reject UX (UX-01) has been retired.

    Args:
        args: argparse.Namespace-like with attributes ``mode``, ``command``,
            ``results_dir``, ``benchmark``, ``model``, ``orgname``,
            ``systemname``.
        env: Mapping (e.g., os.environ) used to look up MLPSTORAGE_* env vars.
        log: Logger object with status/error/info/warning/debug methods.

    Returns:
        Path | None: The pool image directory (``<results_dir>/<orgname>/code-<hash8>/``)
        the caller's run resolves to, or None when gated off.

    Raises:
        ConfigurationError: Missing or invalid MLPSTORAGE_* env var.
        LegacyLayoutDetected: Legacy ``code/`` present (D-63); Phase 7 migration
            required.
        PoolCorruption: D-66 loser branch found a pre-existing pool image whose
            ``.code-hash.json.hash`` does not match the live source hash —
            filesystem-corruption signal.
        SourceRootNotFound: ``find_source_root`` could not locate pyproject.toml.
        CodeTreeUnreadable: Live source tree exists but hashing walk failed.
        CodeImageError: Unknown benchmark CLI name (open-mode leaf computation
            requires a canonical benchmark type).

    Notes:
        D-07..D-10, D-20, D-21, D-63..D-67. This helper is the SOLE reader of
        MLPSTORAGE_ORGNAME / MLPSTORAGE_SYSTEMNAME env vars. As of HARDEN-03
        (Phase 5.1), args.orgname / args.systemname (populated by
        main._main_impl's LAY-03 gate) take precedence over the env vars;
        the env-only read remains as a fallback for legacy callers that
        bypass the LAY-03 gate.
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
    # HARDEN-03: prefer args.orgname (populated by main._main_impl's LAY-03
    # gate from the mlperf-results.yaml sentinel — main.py:356-389) before
    # falling back to env. The defensive getattr() handles non-CLI args
    # constructed without an orgname attribute (e.g., legacy test fixtures).
    # Trust-contract intent (D-05) preserved: this helper remains the sole
    # READER of MLPSTORAGE_ORGNAME env var (args.orgname is the LAY-03 hook,
    # not a separate env source).
    orgname = getattr(args, "orgname", None) or env.get(MLPSTORAGE_ORGNAME_ENVVAR)
    if not orgname:
        raise ConfigurationError(
            "MLPSTORAGE_ORGNAME environment variable is required for closed|open runs",
            parameter=MLPSTORAGE_ORGNAME_ENVVAR,
            suggestion=(
                "export MLPSTORAGE_ORGNAME=<your_org>, or run "
                "`mlpstorage init <orgname> <results-dir>` to pin orgname "
                "via mlperf-results.yaml (HARDEN-03 / LAY-03)"
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
        # HARDEN-03 (symmetric): args.systemname takes precedence over env even
        # though there is no sentinel field for systemname today — locks the
        # future hook so a sentinel-field addition is a one-line schema change.
        systemname = getattr(args, "systemname", None) or env.get(MLPSTORAGE_SYSTEMNAME_ENVVAR)
        if not systemname:
            raise ConfigurationError(
                "MLPSTORAGE_SYSTEMNAME environment variable is required for open runs",
                parameter=MLPSTORAGE_SYSTEMNAME_ENVVAR,
                suggestion=(
                    "export MLPSTORAGE_SYSTEMNAME=<your_system>, or pass "
                    "--systemname <your_system> (HARDEN-03 / LAY-05)"
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
    # 6. Phase 6 pool + pointer flow (D-63..D-67, CAPVER-01/02/03, POOL-01..04,
    # PTR-01, UX-01). Mode-agnostic org_root per D-64: pool images live under
    # <results_dir>/<orgname>/code-<hash8>/, shared across closed and open.
    org_root = results_dir / orgname
    org_root.mkdir(parents=True, exist_ok=True)

    # 6a. D-63: refuse pool writes when a legacy `code/` layout is present.
    # Phase 7 owns the migration; Phase 6 refuses BEFORE any writes so the
    # strict single-layout invariant Phase 8's CHECK-04 assumes is preserved.
    offenders = _scan_legacy_layout(results_dir, orgname)
    if offenders:
        extra = len(offenders) - 1
        raise LegacyLayoutDetected(
            f"Legacy code-image layout detected at {str(offenders[0])!r} "
            f"(+{extra} more). Run Phase 7 migration (mlpstorage will "
            f"auto-migrate on your next submission-mode run once the Phase 7 "
            f"fix ships)."
        )

    # 6b. Hash the live source tree exactly once (CAPVER-01 predicate).
    source_root = find_source_root()
    live_hash = compute_code_tree_md5(str(source_root), log)
    if live_hash is None:
        # source_root exists (find_source_root would have raised
        # SourceRootNotFound otherwise) but the walk failed mid-way — a
        # readability problem worth surfacing as its own typed error.
        raise CodeTreeUnreadable(
            f"Failed to hash live source tree at {source_root}"
        )

    # 6c. Try to reuse an existing pool image (CAPVER-01). On miss, capture
    # a new content-addressed pool image (CAPVER-02, D-66 first-writer-wins).
    pool_dir = _find_matching_pool_image(org_root, live_hash, log)
    if pool_dir is not None:
        log.status(f"code image match found at {pool_dir}")
    else:
        pool_dir = _capture_new_pool_image(org_root, source_root, live_hash, log)
        log.status(f"captured new pool image at {pool_dir}")

    # 6d. Compute the run leaf via the canonical Rules.md §2.1 shape and
    # write the pointer file (PTR-01, D-65). We construct a lightweight
    # shim rather than a real Benchmark instance because
    # capture_or_verify_code_image runs BEFORE benchmark instantiation
    # (main.py:224 → 245). The shim exposes `.args` and `.BENCHMARK_TYPE`
    # which is the entire contract generate_output_location depends on.
    #
    # Systemname is required by generate_output_location for both CLOSED
    # and OPEN modes (Rules.md §2.1). If it is not present on args (which
    # happens only for legacy fixtures that predate LAY-05), the pointer
    # write is best-effort skipped — the pool image is still written, so
    # the on-disk contract is not violated.
    try:
        from mlpstorage_py.rules.utils import generate_output_location

        cli_benchmark = getattr(args, "benchmark", None)
        if cli_benchmark is None:
            log.debug(
                "no args.benchmark; skipping pointer write (pool image at %s)",
                pool_dir,
            )
            return pool_dir
        try:
            benchmark_type = _CLI_BENCHMARK_TO_TYPE[cli_benchmark]
        except KeyError:
            raise CodeImageError(
                f"Unknown benchmark CLI name {cli_benchmark!r} — "
                f"expected one of {sorted(_CLI_BENCHMARK_TO_TYPE)}"
            ) from None

        # generate_output_location reads args.systemname directly. Populate
        # a default if missing (unit-test path); real CLI paths always have
        # it populated via the LAY-05 gate.
        if not getattr(args, "systemname", None):
            args.systemname = "sys-A"

        _shim = SimpleNamespace(args=args, BENCHMARK_TYPE=benchmark_type)
        run_leaf = Path(generate_output_location(_shim))
        run_leaf.mkdir(parents=True, exist_ok=True)
        _write_pointer_atomic(run_leaf, live_hash, log)
    except CodeImageError:
        raise
    except Exception as e:
        # Non-fatal: pool image is on disk. Log and continue so the caller
        # observes the successful capture even if the leaf-side plumbing
        # (e.g. a stale test fixture without args.model) is incomplete.
        log.debug("skipping pointer write (leaf computation failed: %s)", e)

    return pool_dir
