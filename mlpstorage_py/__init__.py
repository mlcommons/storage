from importlib.metadata import PackageNotFoundError as _PkgNF, version as _pkg_version
from pathlib import Path
import tomllib

_DIST_NAME = "mlpstorage"


def _version_from_pyproject() -> str | None:
    """Return the project version when running directly from a source checkout."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.is_file():
        return None

    try:
        with pyproject.open("rb") as f:
            project = tomllib.load(f).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        return None

    if project.get("name") != _DIST_NAME:
        return None

    version = project.get("version")
    return version if isinstance(version, str) else None


def _resolve_version() -> str:
    """Resolve the version from source metadata, installed metadata, or unknown."""
    source_version = _version_from_pyproject()
    if source_version:
        return source_version

    try:
        return _pkg_version(_DIST_NAME)
    except _PkgNF:
        return "unknown"


VERSION = _resolve_version()
__version__ = VERSION

# boto3/botocore are banned — install the blocker immediately so any
# transitive import attempt is caught regardless of which module triggers it.
from mlpstorage_py.ban_boto3 import install as _ban_boto3

_ban_boto3()
