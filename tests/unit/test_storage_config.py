"""Unit tests for mlpstorage_py.storage_config.

Tests cover:
  - TestCredentialRedaction: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are
    never exposed as raw strings in the resolved dict.
  - TestEndpointResolution: The 5-link fallback chain
    (S3_ENDPOINT_URIS → S3_ENDPOINT_TEMPLATE → S3_ENDPOINT_FILE →
     AWS_ENDPOINT_URL → S3_ENDPOINT) and the (value, source_label) tuple shape.
  - TestCentralizedResolver: Default values for every key when env is unset.
  - TestRedactSecret / TestMaskCredentialId (Phase 4 / Plan 04-02, D-23/24/25):
    the two unified redactor helpers shared with cluster_collector.collect_environment.
  - TestRedactBackwardCompat: the existing run_summary.py output path still
    works after the D-25 refactor (Option A keeps `_redact` alias; Option B
    deletes it and updates resolve_object_storage_config).
"""

import tempfile
import os
import pytest

from mlpstorage_py.storage_config import resolve_object_storage_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_ENDPOINT_VARS = [
    'S3_ENDPOINT_URIS',
    'S3_ENDPOINT_TEMPLATE',
    'S3_ENDPOINT_FILE',
    'AWS_ENDPOINT_URL',
    'S3_ENDPOINT',
]


def _clear_all_endpoint_vars(monkeypatch):
    for var in _ALL_ENDPOINT_VARS:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# TestCredentialRedaction
# ---------------------------------------------------------------------------

class TestCredentialRedaction:
    def test_access_key_redacted_when_set(self, monkeypatch):
        # Phase 4 / Plan 04-02 (D-23 / D-25) — KEY_ID now flows through
        # `_mask_credential_id` rather than the legacy length-only `_redact`.
        # The deliberate UX change is called out in 04-CONTEXT.md D-25.
        # Contract preserved: the raw value MUST NOT appear; the redacted
        # form is the first-4 / **** / last-4 mask.
        raw = 'AKIAIOSFODNN7EXAMPLE'
        monkeypatch.setenv('AWS_ACCESS_KEY_ID', raw)
        config = resolve_object_storage_config()
        assert raw not in str(config), "Raw access key must not appear in resolved config"
        assert config['aws_access_key_id_redacted'] == 'AKIA****MPLE'

    def test_access_key_not_set(self, monkeypatch):
        monkeypatch.delenv('AWS_ACCESS_KEY_ID', raising=False)
        config = resolve_object_storage_config()
        assert config['aws_access_key_id_redacted'] == '[not set]'

    def test_secret_key_redacted(self, monkeypatch):
        raw = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
        monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', raw)
        config = resolve_object_storage_config()
        assert raw not in str(config), "Raw secret key must not appear in resolved config"
        assert '[SET —' in config['aws_secret_access_key_redacted']

    def test_secret_key_not_set(self, monkeypatch):
        monkeypatch.delenv('AWS_SECRET_ACCESS_KEY', raising=False)
        config = resolve_object_storage_config()
        assert config['aws_secret_access_key_redacted'] == '[not set]'


# ---------------------------------------------------------------------------
# TestEndpointResolution
# ---------------------------------------------------------------------------

class TestEndpointResolution:
    def test_s3_endpoint_uris_takes_priority(self, monkeypatch):
        """S3_ENDPOINT_URIS wins over all lower-priority env vars."""
        _clear_all_endpoint_vars(monkeypatch)
        monkeypatch.setenv('S3_ENDPOINT_URIS', 'http://high-priority:9000')
        monkeypatch.setenv('AWS_ENDPOINT_URL', 'http://low-priority:9001')
        config = resolve_object_storage_config()
        val, src = config['endpoint']
        assert val == 'http://high-priority:9000'
        assert src == 'S3_ENDPOINT_URIS'

    def test_aws_endpoint_url_fallback(self, monkeypatch):
        """AWS_ENDPOINT_URL is used when the three higher-priority vars are absent."""
        _clear_all_endpoint_vars(monkeypatch)
        monkeypatch.setenv('AWS_ENDPOINT_URL', 'http://fallback:9000')
        config = resolve_object_storage_config()
        val, src = config['endpoint']
        assert val == 'http://fallback:9000'
        assert src == 'AWS_ENDPOINT_URL'

    def test_no_endpoint_set(self, monkeypatch):
        """Returns (None, '') when no endpoint var is set."""
        _clear_all_endpoint_vars(monkeypatch)
        config = resolve_object_storage_config()
        assert config['endpoint'] == (None, '')

    def test_endpoint_returns_tuple(self, monkeypatch):
        """endpoint value is always a 2-tuple."""
        _clear_all_endpoint_vars(monkeypatch)
        config = resolve_object_storage_config()
        assert isinstance(config['endpoint'], tuple)
        assert len(config['endpoint']) == 2

    def test_s3_endpoint_template_wins_over_aws_endpoint_url(self, monkeypatch):
        """S3_ENDPOINT_TEMPLATE (priority 2) beats AWS_ENDPOINT_URL (priority 4)."""
        _clear_all_endpoint_vars(monkeypatch)
        monkeypatch.setenv('S3_ENDPOINT_TEMPLATE', 'http://template-host:9000')
        monkeypatch.setenv('AWS_ENDPOINT_URL', 'http://fallback:9001')
        config = resolve_object_storage_config()
        val, src = config['endpoint']
        assert val == 'http://template-host:9000'
        assert src == 'S3_ENDPOINT_TEMPLATE'

    def test_s3_endpoint_file_wins_over_aws_endpoint_url(self, monkeypatch):
        """S3_ENDPOINT_FILE (priority 3) beats AWS_ENDPOINT_URL (priority 4).

        The resolver must return the URI *inside* the file, not the file path.
        """
        _clear_all_endpoint_vars(monkeypatch)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
            tf.write("# comment line — should be skipped\n")
            tf.write("http://file-endpoint:9000\n")
            tf.write("http://second-endpoint:9001\n")
            tmp_path = tf.name
        try:
            monkeypatch.setenv('S3_ENDPOINT_FILE', tmp_path)
            monkeypatch.setenv('AWS_ENDPOINT_URL', 'http://fallback:9002')
            config = resolve_object_storage_config()
            val, src = config['endpoint']
            # Must return the URI from the file, not the file path itself
            assert val == 'http://file-endpoint:9000', (
                f"Expected URI from file, got {val!r} — "
                "resolver may be returning the file path instead of its contents"
            )
            assert src == 'S3_ENDPOINT_FILE'
        finally:
            os.unlink(tmp_path)

    def test_s3_endpoint_fallback_last_resort(self, monkeypatch):
        """S3_ENDPOINT (priority 5) is used when all higher-priority vars are absent."""
        _clear_all_endpoint_vars(monkeypatch)
        monkeypatch.setenv('S3_ENDPOINT', 'http://last-resort:9000')
        config = resolve_object_storage_config()
        val, src = config['endpoint']
        assert val == 'http://last-resort:9000'
        assert src == 'S3_ENDPOINT'


# ---------------------------------------------------------------------------
# TestCentralizedResolver
# ---------------------------------------------------------------------------

class TestCentralizedResolver:
    def test_defaults_returned_when_env_unset(self, monkeypatch):
        """Default values are returned for every key when no env vars are set."""
        for var in ['BUCKET', 'STORAGE_LIBRARY', 'STORAGE_URI_SCHEME',
                    'S3_LOAD_BALANCE_STRATEGY', 'AWS_REGION', 'AWS_CA_BUNDLE']:
            monkeypatch.delenv(var, raising=False)
        _clear_all_endpoint_vars(monkeypatch)
        monkeypatch.delenv('AWS_ACCESS_KEY_ID', raising=False)
        monkeypatch.delenv('AWS_SECRET_ACCESS_KEY', raising=False)

        config = resolve_object_storage_config()

        assert config['bucket'] == ''
        assert config['storage_library'] == 's3dlio'
        assert config['uri_scheme'] == 's3'
        assert config['load_balance_strategy'] == 'round_robin'
        assert config['aws_region'] == 'us-east-1'
        assert config['aws_ca_bundle'] is None


# ---------------------------------------------------------------------------
# Phase 4 / Plan 04-02 — D-25 redactor unification
# ---------------------------------------------------------------------------
#
# `_redact_secret` and `_mask_credential_id` are the two shared helpers
# consumed by:
#   1. `resolve_object_storage_config()` → `run_summary.py` display.
#   2. `cluster_collector.collect_environment()` → the new YAML emit block.
#
# Branch shapes are locked in CONTEXT.md D-23 / D-24 and PLAN.md must_haves.
# ---------------------------------------------------------------------------


class TestRedactSecret:
    """`_redact_secret(val)` produces a length-only sentinel (D-24).

    Branches:
      - None → '[not set]'
      - ''   → '[SET — empty]' (Phase 4 deliberate UX improvement; the legacy
        `_redact()` returned '[not set]' for both branches, hiding the
        misleading set-but-empty case from the operator)
      - non-empty → '[SET — N chars]' where N == len(val)
    """

    @pytest.mark.parametrize(
        "val,expected",
        [
            (None, "[not set]"),
            ("", "[SET — empty]"),
            ("x", "[SET — 1 chars]"),
            (
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "[SET — 40 chars]",
            ),
        ],
    )
    def test_redact_secret_branches(self, val, expected):
        from mlpstorage_py.storage_config import _redact_secret

        assert _redact_secret(val) == expected


class TestMaskCredentialId:
    """`_mask_credential_id(val)` produces a first-4 / last-4 mask (D-23).

    Branches:
      - None → '[not set]'
      - ''   → '[SET — empty]'
      - 1..7 char → '****' (collapse — too short to mask meaningfully)
      - >= 8 char → f'{val[:4]}****{val[-4:]}'
    """

    @pytest.mark.parametrize(
        "val,expected",
        [
            (None, "[not set]"),
            ("", "[SET — empty]"),
            ("x", "****"),
            ("1234567", "****"),  # exactly 7 → collapse
            ("12345678", "1234****5678"),  # boundary at 8
            ("AKIAIOSFODNN7EXAMPLE", "AKIA****MPLE"),  # canonical AWS access key
        ],
    )
    def test_mask_credential_id_branches(self, val, expected):
        from mlpstorage_py.storage_config import _mask_credential_id

        assert _mask_credential_id(val) == expected


class TestRedactBackwardCompat:
    """The D-25 refactor preserves the `resolve_object_storage_config()`
    consumer contract for `run_summary.py`. Executor picks Option A
    (`_redact` alias retained) or Option B (alias removed, callsite updated
    to new names). Both branches are exercised; the unselected branch
    pytest-skips itself."""

    def test_redact_alias_kept_or_removed_cleanly(self, monkeypatch):
        """Option A: `_redact = _redact_secret` alias still importable AND
        for non-empty inputs matches `_redact_secret`. Option B: `_redact`
        no longer importable; the dict from `resolve_object_storage_config()`
        carries the new shape and the SECRET key matches `_redact_secret`."""
        import importlib

        sc = importlib.import_module("mlpstorage_py.storage_config")
        _redact_secret = getattr(sc, "_redact_secret", None)
        if _redact_secret is None:
            pytest.fail(
                "_redact_secret must exist in storage_config after the D-25 refactor"
            )

        legacy_redact = getattr(sc, "_redact", None)
        if legacy_redact is not None:
            # Option A — alias retained.
            assert legacy_redact("some-value") == _redact_secret("some-value"), (
                "If `_redact` is kept as an alias it must produce the same "
                "output as `_redact_secret` for non-empty inputs."
            )
            assert legacy_redact(None) == "[not set]"
        else:
            # Option B — alias removed; verify the consumer still works
            # by going through resolve_object_storage_config.
            monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "mysecret")
            config = resolve_object_storage_config()
            assert (
                config["aws_secret_access_key_redacted"]
                == _redact_secret("mysecret")
            ), (
                "When `_redact` is removed, resolve_object_storage_config "
                "must call `_redact_secret` directly so the run_summary.py "
                "consumer dict key still carries a redacted value."
            )

    def test_resolve_object_storage_config_uses_mask_for_access_key(self, monkeypatch):
        """Per D-25 side effect: the KEY_ID dict key now carries the
        masked-form `_mask_credential_id` output, NOT the legacy
        length-only sentinel. This is the deliberate UX change called out
        in CONTEXT.md."""
        from mlpstorage_py.storage_config import _mask_credential_id

        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        config = resolve_object_storage_config()
        assert config["aws_access_key_id_redacted"] == _mask_credential_id(
            "AKIAIOSFODNN7EXAMPLE"
        )

    def test_resolve_object_storage_config_uses_secret_for_secret_access_key(
        self, monkeypatch
    ):
        """Per D-24: SECRET keeps the length-only sentinel shape. The dict
        key value must match `_redact_secret` exactly."""
        from mlpstorage_py.storage_config import _redact_secret

        raw = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", raw)
        config = resolve_object_storage_config()
        assert config["aws_secret_access_key_redacted"] == _redact_secret(raw)
