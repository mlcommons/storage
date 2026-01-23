"""
CLI argument builder for lockfile management commands.

Provides arguments for:
- mlpstorage lockfile generate: Create lockfile from pyproject.toml
- mlpstorage lockfile verify: Validate installed packages against lockfile
"""

from mlpstorage.cli.common_args import add_universal_arguments


def add_lockfile_arguments(parser):
    """Add lockfile subcommands to the parser.

    Args:
        parser: The lockfile subparser from argparse.
    """
    subparsers = parser.add_subparsers(dest="lockfile_command", required=True)

    # Generate subcommand
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate lockfile from pyproject.toml",
        description="Generate a requirements.txt lockfile with pinned versions using uv pip compile.",
    )
    generate_parser.add_argument(
        "-o", "--output",
        default="requirements.txt",
        help="Output path for lockfile (default: requirements.txt)",
    )
    generate_parser.add_argument(
        "--extra",
        action="append",
        dest="extras",
        help="Include optional dependency group (can be repeated, e.g., --extra test --extra full)",
    )
    generate_parser.add_argument(
        "--hashes",
        action="store_true",
        help="Include SHA256 hashes in lockfile (slower but more secure)",
    )
    generate_parser.add_argument(
        "--python-version",
        help="Target Python version (e.g., 3.10)",
    )
    generate_parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to pyproject.toml (default: pyproject.toml)",
    )
    generate_parser.add_argument(
        "--all",
        action="store_true",
        dest="generate_all",
        help="Generate both requirements.txt and requirements-full.txt",
    )
    add_universal_arguments(generate_parser)

    # Verify subcommand
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify installed packages match lockfile",
        description="Check that installed package versions match the lockfile.",
    )
    verify_parser.add_argument(
        "-l", "--lockfile",
        default="requirements.txt",
        help="Path to lockfile (default: requirements.txt)",
    )
    verify_parser.add_argument(
        "--skip",
        action="append",
        dest="skip_packages",
        help="Package names to skip validation (can be repeated)",
    )
    verify_parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Don't fail if packages are missing (only check installed versions)",
    )
    verify_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any difference (default: fail only on version mismatch)",
    )
    add_universal_arguments(verify_parser)

    return parser
