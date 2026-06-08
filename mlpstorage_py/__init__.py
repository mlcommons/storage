from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNF
import pathlib
import tomllib  # stdlib since Python 3.11; project requires >=3.12


def _resolve_version() -> str:
    # Primary: installed distribution metadata (correct dist name is "mlpstorage")
    try:
        return _pkg_version("mlpstorage")
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
