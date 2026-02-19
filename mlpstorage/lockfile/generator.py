"""
Lockfile generation for reproducible environments.

Uses uv pip compile to generate requirements.txt lockfiles from pyproject.toml.
Supports CPU-only builds and optional dependency groups.
"""

import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class LockfileGenerationError(Exception):
    """Raised when lockfile generation fails."""

    def __init__(self, message: str, stderr: str = "", return_code: int = 1):
        self.stderr = stderr
        self.return_code = return_code
        super().__init__(message)


@dataclass
class GenerationOptions:
    """Options for lockfile generation."""
    output_path: str = "requirements.txt"
    extras: list[str] | None = None  # Optional dependency groups to include
    generate_hashes: bool = False    # Include SHA256 hashes (slower but more secure)
    universal: bool = True           # Generate cross-platform lockfile
    python_version: str = ""         # Target Python version (e.g., "3.10")
    exclude_newer: str = ""          # Exclude packages newer than date (YYYY-MM-DD)


def check_uv_available() -> tuple[bool, str]:
    """Check if uv is available in PATH.

    Returns:
        Tuple of (available, message). If available is False, message contains
        installation instructions.
    """
    if shutil.which("uv") is None:
        return False, (
            "uv is not installed. Install with: pip install uv\n"
            "Or: curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
    return True, "uv is available"


def generate_lockfile(
    pyproject_path: str = "pyproject.toml",
    options: Optional[GenerationOptions] = None,
) -> tuple[int, str]:
    """Generate a lockfile using uv pip compile.

    Args:
        pyproject_path: Path to pyproject.toml (default: current directory)
        options: Generation options (default: requirements.txt output)

    Returns:
        Tuple of (return_code, output_path). Return code 0 indicates success.

    Raises:
        LockfileGenerationError: If uv is not available or compilation fails.
        FileNotFoundError: If pyproject.toml doesn't exist.
    """
    if options is None:
        options = GenerationOptions()

    # Check pyproject.toml exists
    pyproject = Path(pyproject_path)
    if not pyproject.exists():
        raise FileNotFoundError(f"pyproject.toml not found: {pyproject_path}")

    # Check uv is available
    available, msg = check_uv_available()
    if not available:
        raise LockfileGenerationError(msg)

    # Build command
    cmd = ["uv", "pip", "compile", str(pyproject), "-o", options.output_path]

    if options.generate_hashes:
        cmd.append("--generate-hashes")

    if options.universal:
        cmd.append("--universal")

    if options.python_version:
        cmd.extend(["--python-version", options.python_version])

    if options.exclude_newer:
        cmd.extend(["--exclude-newer", options.exclude_newer])

    for extra in options.extras or []:
        cmd.extend(["--extra", extra])

    # Run uv pip compile
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise LockfileGenerationError(
            f"Lockfile generation failed: {result.stderr}",
            stderr=result.stderr,
            return_code=result.returncode,
        )

    return 0, options.output_path


def generate_lockfiles_for_project(
    pyproject_path: str = "pyproject.toml",
    base_output: str = "requirements.txt",
    full_output: str = "requirements-full.txt",
) -> dict[str, str]:
    """Generate both base and full lockfiles for the project.

    Args:
        pyproject_path: Path to pyproject.toml
        base_output: Output path for base lockfile (no extras)
        full_output: Output path for full lockfile (with [full] extra)

    Returns:
        Dict mapping lockfile type to output path.

    Raises:
        LockfileGenerationError: If generation fails.
    """
    results = {}

    # Generate base lockfile (no DLIO)
    _, path = generate_lockfile(
        pyproject_path,
        GenerationOptions(output_path=base_output),
    )
    results["base"] = path

    # Generate full lockfile (with DLIO)
    _, path = generate_lockfile(
        pyproject_path,
        GenerationOptions(output_path=full_output, extras=["full"]),
    )
    results["full"] = path

    return results
