"""Deterministic MD5 predicate for a submission code/ tree.

Implements the code-tree checksum algorithm specified in Rules.md 2.1.6
(codeDirectoryContents) and 3.6.1 (trainingClosedSubmissionChecksum).

Design decisions (D-13):
- Walk with ``os.walk(root_path, followlinks=False)`` to prevent path-traversal
  via symlinks (threat T-02-01).
- Exclusions come from ``MD5_EXCLUDE_PREFIXES`` and ``MD5_EXCLUDE_FILENAMES``
  constants in ``..constants`` — never redefined here.
- Files are read in binary mode only; no line-ending normalization.
- Surviving entries are sorted by POSIX-byte order (``relative_path.encode('utf-8')``)
  so the digest is stable across filesystems that return entries in different orders.
- Relative path bytes are hashed BEFORE file content so renames of identically-
  content files produce different digests (D-14 rename detection).
- Symlinks encountered during the walk emit a ``[2.1.6 codeDirectoryContents]``
  WARNING via ``log.warning`` and are skipped (do NOT raise).

Public API:
    compute_code_tree_md5(root_path, log) -> str | None
"""

import fnmatch
import hashlib
import os

from ..constants import MD5_EXCLUDE_FILENAMES, MD5_EXCLUDE_PREFIXES

# Chunk size for streaming binary reads — 64 KiB balances memory and syscall overhead.
_READ_CHUNK = 64 * 1024


def compute_code_tree_md5(root_path: str, log) -> str | None:
    """Compute a deterministic MD5 hex digest of a code directory tree.

    Walks ``root_path`` with ``os.walk(..., followlinks=False)``, skipping:

    - Any entry whose POSIX-relative path starts with a prefix in
      ``MD5_EXCLUDE_PREFIXES`` (e.g. ``.git/``, ``__pycache__/``).
    - Any directory whose name ends with ``.egg-info`` (handled here per D-13,
      not in the constant, so the suffix test is against the directory name).
    - Any file whose basename matches a pattern in ``MD5_EXCLUDE_FILENAMES``
      (e.g. ``*.pyc``, ``.DS_Store``).
    - Any symlink — emits a ``warn``-level ``[2.1.6 codeDirectoryContents]``
      message and continues (does NOT raise).

    Surviving files are sorted by their POSIX-relative path encoded as UTF-8
    bytes (POSIX byte order), then each file contributes:
    1. Its relative POSIX path bytes.
    2. Its full binary content (chunked reads, 64 KiB per chunk).

    Args:
        root_path: Absolute or relative path to the root of the code tree.
        log: Logger object with a ``warning(msg, *args)`` method. Used to emit
            the symlink-rejection warning.

    Returns:
        Hex MD5 digest string (32 lowercase hex characters), or ``None`` if
        ``root_path`` does not exist.
    """
    if not os.path.exists(root_path):
        return None

    # Collect surviving (relative_posix_path, absolute_path) tuples.
    entries = []

    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
        # Prune excluded directory names in-place so os.walk does not recurse
        # into them.  We iterate over a copy because we are modifying dirnames.
        dirnames[:] = [
            d for d in dirnames
            if not _is_excluded_dir(dirpath, d, root_path)
        ]

        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, root_path).replace(os.sep, "/")

            # Skip files in excluded directories (belt-and-suspenders check for
            # deeply nested paths that escaped the dirnames pruning above).
            if _is_excluded_by_prefix(rel_path):
                continue

            # Skip excluded filenames.
            if _is_excluded_filename(filename):
                continue

            # Reject symlinks — warn and skip (D-13, T-02-01).
            if os.path.islink(full_path):
                log.warning(
                    "[2.1.6 codeDirectoryContents] %s: symlink rejected (skipped)",
                    full_path,
                )
                continue

            entries.append((rel_path, full_path))

    # Sort by POSIX-byte order (explicit UTF-8 encoding of the relative path).
    entries.sort(key=lambda t: t[0].encode("utf-8"))

    # Build rolling MD5: relative-path bytes then file-content bytes per entry.
    hasher = hashlib.md5()
    for rel_path, full_path in entries:
        hasher.update(rel_path.encode("utf-8"))
        with open(full_path, "rb") as fh:
            while True:
                chunk = fh.read(_READ_CHUNK)
                if not chunk:
                    break
                hasher.update(chunk)

    return hasher.hexdigest()


def _is_excluded_dir(dirpath: str, dirname: str, root_path: str) -> bool:
    """Return True if ``dirname`` inside ``dirpath`` should be pruned.

    Checks both the MD5_EXCLUDE_PREFIXES list (path-prefix match) and the
    ``.egg-info`` suffix (D-13: handled in predicate, not in constant).

    Args:
        dirpath: Absolute path of the parent directory (from os.walk).
        dirname: Name of the subdirectory to evaluate.
        root_path: Root of the tree being hashed (used to compute relative path).

    Returns:
        True if the directory should be excluded from the walk.
    """
    full_dir = os.path.join(dirpath, dirname)
    rel_dir = os.path.relpath(full_dir, root_path).replace(os.sep, "/") + "/"

    if _is_excluded_by_prefix(rel_dir):
        return True

    if dirname.endswith(".egg-info"):
        return True

    return False


def _is_excluded_by_prefix(posix_rel_path: str) -> bool:
    """Return True if ``posix_rel_path`` starts with any MD5_EXCLUDE_PREFIXES entry.

    Each prefix already has a trailing slash (e.g. ``.git/``) so ``.gitignore``
    (file) does not match ``.git/`` (directory prefix).

    Args:
        posix_rel_path: POSIX-style relative path (forward slashes, may end in /).

    Returns:
        True if the path should be excluded.
    """
    for prefix in MD5_EXCLUDE_PREFIXES:
        if posix_rel_path.startswith(prefix):
            return True
    return False


def _is_excluded_filename(basename: str) -> bool:
    """Return True if ``basename`` matches any pattern in MD5_EXCLUDE_FILENAMES.

    Uses ``fnmatch.fnmatch`` for glob patterns (``*.pyc``) and exact string
    comparison for literal names (``.DS_Store``, ``Thumbs.db``).

    Args:
        basename: The filename without directory path.

    Returns:
        True if the file should be excluded from the hash.
    """
    for pattern in MD5_EXCLUDE_FILENAMES:
        if fnmatch.fnmatch(basename, pattern):
            return True
    return False
