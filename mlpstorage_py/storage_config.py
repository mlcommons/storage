"""Centralized S3 environment-variable resolver for MLPerf Storage.

Provides a single source of truth for all S3 configuration env var reads
used in run summaries and storage backend initialization.

NOTE 1: resolve_object_storage_config() reads env vars at call time, not at
import time, so the .env file may not yet be loaded when called from
run_summary.py.

NOTE 2: resolve_object_storage_config() is for DISPLAY/SUMMARY purposes only.
The raw os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY')
reads used for Minio SDK authentication remain as direct calls in minio_reader.py and
minio_writer.py and are NOT moved into this resolver.
"""

import os
from typing import Optional, Tuple


def _redact(val: Optional[str]) -> str:
    """Return a redacted representation of a credential value.

    Args:
        val: The raw credential string (or None/empty).

    Returns:
        "[SET — N chars]" if val is truthy, else "[not set]".
    """
    if val:
        return f"[SET — {len(val)} chars]"
    return "[not set]"


def _resolve_endpoint() -> Tuple[Optional[str], str]:
    """Resolve the S3 endpoint using a 5-link priority chain.

    Priority order:
        1. S3_ENDPOINT_URIS   — comma-separated list; first element used
        2. S3_ENDPOINT_TEMPLATE
        3. S3_ENDPOINT_FILE   — plain text file; first non-comment line used
        4. AWS_ENDPOINT_URL
        5. S3_ENDPOINT

    Returns:
        (value, var_name) where value is the first non-empty string found,
        or (None, '') if all five vars are unset / empty.
    """
    chain = [
        'S3_ENDPOINT_URIS',
        'S3_ENDPOINT_TEMPLATE',
        'S3_ENDPOINT_FILE',
        'AWS_ENDPOINT_URL',
        'S3_ENDPOINT',
    ]
    for var in chain:
        raw = os.environ.get(var, '').strip()
        if not raw:
            continue
        if var == 'S3_ENDPOINT_FILE':
            file_path = raw
            try:
                with open(file_path) as fh:
                    for line in fh:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            return line, var
            except OSError:
                pass
            continue
        return raw, var
    return None, ''


def resolve_object_storage_config() -> dict:
    """Resolve all S3 / object-storage configuration from the environment.

    Reads env vars at call time (not at import time).  Safe to call before
    or after dotenv loading — the caller is responsible for sequencing.

    Returns:
        dict with keys:
            bucket (str)                       — BUCKET env var, default ''
            storage_library (str)              — STORAGE_LIBRARY, default 's3dlio'
            uri_scheme (str)                   — STORAGE_URI_SCHEME with ':/'.rstrip(),
                                                 default 's3'
            endpoint (tuple)                   — (value_or_None, source_label_str)
            load_balance_strategy (str)        — S3_LOAD_BALANCE_STRATEGY,
                                                 default 'round_robin'
            aws_region (str)                   — AWS_REGION, default 'us-east-1'
            aws_ca_bundle (Optional[str])      — AWS_CA_BUNDLE, default None
            aws_access_key_id_redacted (str)   — redacted form, never raw value
            aws_secret_access_key_redacted (str) — redacted form, never raw value
    """
    endpoint_val, endpoint_src = _resolve_endpoint()

    return {
        'bucket': os.environ.get('BUCKET', ''),
        'storage_library': os.environ.get('STORAGE_LIBRARY', 's3dlio'),
        'uri_scheme': os.environ.get('STORAGE_URI_SCHEME', 's3').rstrip(':/'),
        'endpoint': (endpoint_val, endpoint_src),
        'load_balance_strategy': os.environ.get('S3_LOAD_BALANCE_STRATEGY', 'round_robin'),
        'aws_region': os.environ.get('AWS_REGION', 'us-east-1'),
        'aws_ca_bundle': os.environ.get('AWS_CA_BUNDLE') or None,
        'aws_access_key_id_redacted': _redact(os.environ.get('AWS_ACCESS_KEY_ID')),
        'aws_secret_access_key_redacted': _redact(os.environ.get('AWS_SECRET_ACCESS_KEY')),
    }
