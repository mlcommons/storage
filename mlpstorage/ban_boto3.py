"""boto3 / botocore import prohibition.

boto3 and botocore are BANNED from this project.  They are horrendously
slow for high-throughput S3 workloads and must never be used.

Approved S3 libraries:
  - s3dlio       (primary — multi-protocol, highest throughput)
  - s3torchconnector  (PyTorch-native S3)
  - minio        (S3-compatible SDK, acceptable for MinIO targets)

This module installs a sys.meta_path finder that raises ImportError
the instant any code (including transitive deps) attempts to import
boto3 or botocore.
"""

import sys


_BANNED = frozenset({'boto3', 'botocore'})


class _Boto3Banned:
    """Meta path finder that blocks boto3 and botocore unconditionally."""

    def find_module(self, fullname, path=None):  # Python <3.4 compat shim
        top = fullname.split('.')[0]
        if top in _BANNED:
            return self
        return None

    def find_spec(self, fullname, path, target=None):
        top = fullname.split('.')[0]
        if top in _BANNED:
            raise ImportError(
                f"\n\n"
                f"  ╔══════════════════════════════════════════════════════════════╗\n"
                f"  ║  BANNED LIBRARY: {fullname!r:<44}║\n"
                f"  ║                                                              ║\n"
                f"  ║  boto3 and botocore are PROHIBITED in this project.          ║\n"
                f"  ║  They are horrendously slow for high-throughput S3 I/O.      ║\n"
                f"  ║                                                              ║\n"
                f"  ║  Use instead:                                                ║\n"
                f"  ║    s3dlio            — multi-protocol, highest throughput    ║\n"
                f"  ║    s3torchconnector  — PyTorch-native S3                     ║\n"
                f"  ║    minio             — MinIO / S3-compatible SDK             ║\n"
                f"  ╚══════════════════════════════════════════════════════════════╝\n"
            )
        return None

    def load_module(self, fullname):  # never reached, but required by protocol
        raise ImportError(f"boto3/botocore are banned: {fullname!r}")


def install():
    """Install the boto3 ban.  Safe to call multiple times."""
    for finder in sys.meta_path:
        if isinstance(finder, _Boto3Banned):
            return  # already installed
    sys.meta_path.insert(0, _Boto3Banned())
