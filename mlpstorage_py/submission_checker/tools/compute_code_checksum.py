"""Standalone CLI tool that prints the deterministic MD5 digest of a code tree.

Usage:
    python -m mlpstorage_py.submission_checker.tools.compute_code_checksum <path>

Prints the 32-character hex MD5 digest to stdout (one line, no decoration) so a
maintainer can pipe the output directly into a constants update:

    python -m mlpstorage_py.submission_checker.tools.compute_code_checksum <path> \\
        | tr -d '\\n'   # strip newline if needed

See ``code_checksum.compute_code_tree_md5`` for the exclusion rules and algorithm
details (D-11, D-13 per phase context).
"""

import argparse
import logging
import sys

from .code_checksum import compute_code_tree_md5

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s",
)
log = logging.getLogger("compute_code_checksum")


def get_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with a ``path`` attribute.
    """
    parser = argparse.ArgumentParser(
        description="Compute a deterministic MD5 digest of a code/ tree.",
    )
    parser.add_argument(
        "path",
        help="Path to the root of the code tree to hash.",
    )
    return parser.parse_args()


def main():
    """Run the code-tree checksum tool.

    Resolves the MD5 digest of the provided path using ``compute_code_tree_md5``
    and prints it to stdout (32 lowercase hex characters, followed by a newline).

    Returns:
        int: 0 on success, 1 if the path does not exist.
    """
    args = get_args()
    digest = compute_code_tree_md5(args.path, log)
    if digest is None:
        log.error(
            "could not compute checksum for %s (path does not exist)",
            args.path,
        )
        return 1
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
