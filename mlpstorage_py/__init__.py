from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNF
import pathlib
import tomllib  # stdlib since Python 3.11; project requires >=3.12

# Distribution name. Must match `project.name` in pyproject.toml — tested in
# tests/unit/test_version.py::test_dist_name_matches_pyproject.
_DIST_NAME = "mlpstorage"


def _resolve_version() -> str:
    # Primary: installed distribution metadata
    try:
        return _pkg_version(_DIST_NAME)
    except _PkgNF:
        pass
    # Fallback: parse pyproject.toml for source-checkout usage
    _pyproject = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    try:
        with open(_pyproject, "rb") as _f:
            return tomllib.load(_f)["project"]["version"]
    except Exception:
        return "unknown"


VERSION = _resolve_version()
__version__ = VERSION

# boto3/botocore are banned — install the blocker immediately so any
# transitive import attempt is caught regardless of which module triggers it.
from mlpstorage_py.ban_boto3 import install as _ban_boto3
_ban_boto3()
