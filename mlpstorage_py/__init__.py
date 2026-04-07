from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNF
try:
    VERSION = _pkg_version("mlpstorage_py")
except _PkgNF:
    VERSION = "unknown"
__version__ = VERSION

# boto3/botocore are banned — install the blocker immediately so any
# transitive import attempt is caught regardless of which module triggers it.
from mlpstorage_py.ban_boto3 import install as _ban_boto3
_ban_boto3()
