"""
Regression tests for issue #572:

  * RESOURCE_EXHAUSTED when a FLAT ground-truth copy response exceeds the
    pymilvus 256 MiB gRPC client default.
  * "loud failure" validity verdict so users never have to grep logs to know
    whether a run is trustworthy.

These tests are dependency-free: they install a fake ``pymilvus`` before
importing ``vdbbench`` so they run in CI without a Milvus client.
"""

import sys
import json
import tempfile
import os
from unittest.mock import MagicMock

import pytest


def _install_fake_pymilvus():
    if "pymilvus" in sys.modules:
        return
    fake = MagicMock(name="pymilvus")
    # connections.connect must record kwargs so we can assert on the gRPC limits.
    fake.connections = MagicMock(name="connections")
    sys.modules["pymilvus"] = fake


_install_fake_pymilvus()

from vdbbench.connection import (  # noqa: E402
    MAX_GRPC_MESSAGE_LENGTH,
    open_connection,
)
from vdbbench import simple_bench  # noqa: E402
from vdbbench.simple_bench import (  # noqa: E402
    FlatSetupResult,
    emit_result_verdict,
)


class TestGrpcConnectionHelper:
    """open_connection must always raise the gRPC message-size limits (#572)."""

    def test_open_connection_sets_message_limits(self):
        from vdbbench import connection as conn_mod

        captured = {}

        def fake_connect(**kwargs):
            captured.update(kwargs)

        conn_mod.connections.connect = fake_connect
        open_connection(alias="flat_setup", host="127.0.0.1", port="19530")

        assert captured["max_receive_message_length"] == MAX_GRPC_MESSAGE_LENGTH
        assert captured["max_send_message_length"] == MAX_GRPC_MESSAGE_LENGTH
        # Must exceed the observed 393631413-byte response in the issue and the
        # 256 MiB (268435456) client default that caused the failure.
        assert MAX_GRPC_MESSAGE_LENGTH > 393_631_413
        assert MAX_GRPC_MESSAGE_LENGTH > 268_435_456

    def test_limit_matches_config_declared_value(self):
        # configs/*.yaml declare 514_983_574; the helper is the single source
        # of truth those literals must match.
        assert MAX_GRPC_MESSAGE_LENGTH == 514_983_574


class TestResultVerdict:
    """emit_result_verdict must classify runs unambiguously (#572)."""

    def _verdict(self, flat_result, nqe):
        d = tempfile.mkdtemp()
        v = emit_result_verdict(
            d, flat_result=flat_result, num_queries_evaluated=nqe
        )
        payload = json.load(open(os.path.join(d, "result_verdict.json")))
        return v, payload

    def test_full_coverage_is_valid(self):
        fr = FlatSetupResult(
            ok=True, coverage=1.0, total_vectors=1_000_000, copied_vectors=1_000_000
        )
        v, payload = self._verdict(fr, 1000)
        assert v == "valid"
        assert payload["valid"] is True

    def test_recovered_full_coverage_is_valid_but_noted(self):
        fr = FlatSetupResult(
            ok=True,
            coverage=1.0,
            total_vectors=1_000_000,
            copied_vectors=1_000_000,
            had_recoverable_error=True,
        )
        v, payload = self._verdict(fr, 1000)
        assert v.startswith("valid")
        assert "recovered" in v
        assert payload["valid"] is True

    def test_partial_coverage_is_degraded(self):
        # Above the 99% abort threshold but below 100%: recall was computed on
        # an incomplete ground truth -> not trustworthy.
        fr = FlatSetupResult(
            ok=True, coverage=0.995, total_vectors=1_000_000, copied_vectors=995_000
        )
        v, payload = self._verdict(fr, 1000)
        assert v.startswith("degraded")
        assert payload["valid"] is False

    def test_failed_setup_is_invalid(self):
        fr = FlatSetupResult(ok=False, reason="coverage 0.00% below 99% minimum")
        v, payload = self._verdict(fr, 0)
        assert v.startswith("invalid")
        assert payload["valid"] is False

    def test_zero_queries_is_invalid_even_with_ok_flat(self):
        fr = FlatSetupResult(
            ok=True, coverage=1.0, total_vectors=10, copied_vectors=10
        )
        v, payload = self._verdict(fr, 0)
        assert v == "invalid: 0 queries had valid ground truth"
        assert payload["valid"] is False

    def test_exactly_one_result_line(self, capsys):
        fr = FlatSetupResult(
            ok=True, coverage=1.0, total_vectors=10, copied_vectors=10
        )
        d = tempfile.mkdtemp()
        emit_result_verdict(d, flat_result=fr, num_queries_evaluated=5)
        out = capsys.readouterr().out
        result_lines = [ln for ln in out.splitlines() if ln.startswith("RESULT:")]
        assert len(result_lines) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
