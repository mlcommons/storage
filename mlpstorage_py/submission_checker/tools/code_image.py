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
from .code_checksum import compute_code_tree_md5
from ..constants import MD5_EXCLUDE_FILENAMES, MD5_EXCLUDE_PREFIXES


class CodeImageError(Exception):
    """Base for all code-image capture/verify failures (D-03)."""


class MissingHashFile(CodeImageError):
    """.code-hash.json not found in an image directory (D-14)."""


class MalformedHashFile(CodeImageError):
    """.code-hash.json present but unparseable or invalid (D-15)."""


class SourceRootNotFound(CodeImageError):
    """find_source_root walked to filesystem root without finding pyproject.toml (D-05)."""


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
        CodeImageError: If target_dir/code/ already exists (D-16).
        SourceRootNotFound: If source_root is missing or hashing fails.
    """
    code_dir = target_dir / _CODE_DIRNAME
    code_tmp = target_dir / _TMP_SUFFIX

    if code_dir.exists():
        raise CodeImageError(f"Code image already exists at {code_dir} (D-16)")

    if code_tmp.exists():
        log.warning("stale code.tmp/ at %s removed before capture (D-18)", code_tmp)
        shutil.rmtree(code_tmp)

    # Behavior 5: Exclusion delegated to identical logic as hash
    _atomic_capture(source_root, code_tmp, log)

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
        SourceRootNotFound: If source_root cannot be hashed.
        CodeImageError: If image_dir is malformed.
    """
    img = load_code_image(image_dir, log)
    current_hash = compute_code_tree_md5(str(source_root), log)
    if current_hash is None:
        raise SourceRootNotFound(f"Source root not found or unreadable: {source_root}")
    
    return current_hash == img.hash


def verify_image_self_consistent(image_dir: Path, log) -> bool:
    """Verify that a captured 'code/' tree matches its own recorded hash (D-12, VALS-02/04).

    Used by the submission validator to detect post-capture tampering.

    Args:
        image_dir: Path to the captured 'code/' directory.
        log: Logger object.

    Returns:
        True if the tree hash matches .code-hash.json, False otherwise.
    """
    img = load_code_image(image_dir, log)
    actual_hash = compute_code_tree_md5(str(image_dir), log)
    if actual_hash is None:
        raise MissingHashFile(f"Captured code directory is missing or unreadable: {image_dir}")
    
    return actual_hash == img.hash


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _atomic_capture(source_root: Path, target_dir: Path, log) -> None:
    """Copy source_root to target_dir using identical exclusion logic as hashing (Behavior 5)."""
    source_str = str(source_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    
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
