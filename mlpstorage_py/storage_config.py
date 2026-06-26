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


# Phase 4 / Plan 04-02 (D-25, COLL-06) — unified credential redaction.
# Two helpers replace the legacy `_redact(val)` and are shared with
# `cluster_collector.collect_environment` (the new YAML emit path).
# Option B chosen at refactor time (the grep `_redact\b mlpstorage_py/`
# returned zero non-test, non-self consumers) — `_redact` is gone, all
# call sites use the new names. See 04-02-SUMMARY.md for the rationale.


def _redact_secret(val: Optional[str]) -> str:
    """Length-only credential redactor (D-24).

    Branches:
      - ``None`` → ``"[not set]"``
      - ``""``   → ``"[SET — empty]"`` (Phase 4 deliberate UX improvement:
        the legacy ``_redact()`` returned ``"[not set]"`` for both branches,
        which hid the misleading set-but-empty case from operators)
      - non-empty → ``"[SET — N chars]"`` where ``N == len(val)``

    No sha256 fingerprint — ROADMAP.md SC #2 was reconciled to match this
    decision in the same commit that landed the helper.
    """
    if val is None:
        return "[not set]"
    if val == "":
        return "[SET — empty]"
    return f"[SET — {len(val)} chars]"


def _mask_credential_id(val: Optional[str]) -> str:
    """First-4 / last-4 mask for credential identifiers (D-23).

    Preserves enough identifying prefix for an operator to recognize *which*
    credential is configured without leaking the full value.

    Branches:
      - ``None`` → ``"[not set]"``
      - ``""``   → ``"[SET — empty]"``
      - ``1..7`` chars → ``"****"`` (too short to mask meaningfully without
        leaking >50% of the value)
      - ``>= 8`` chars → ``f"{val[:4]}****{val[-4:]}"``

    Canonical example: ``AKIAIOSFODNN7EXAMPLE`` → ``"AKIA****MPLE"``.
    """
    if val is None:
        return "[not set]"
    if val == "":
        return "[SET — empty]"
    if len(val) < 8:
        return "****"
    return f"{val[:4]}****{val[-4:]}"


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
        # D-23 / D-25: KEY_ID uses the masked-form helper (deliberate UX
        # change vs. the legacy `_redact` length-only shape — run_summary.py
        # output now reads "AKIA****MPLE" instead of "[SET — 20 chars]").
        # SECRET stays length-only per D-24.
        'aws_access_key_id_redacted': _mask_credential_id(os.environ.get('AWS_ACCESS_KEY_ID')),
        'aws_secret_access_key_redacted': _redact_secret(os.environ.get('AWS_SECRET_ACCESS_KEY')),
    }
