"""
Pydantic v2 schema for the ``mlperf-results.yaml`` sentinel file.

The sentinel is the single source of truth for the organisation name pinned
to a results-dir (LAY-03). It is created by ``mlpstorage init`` and read by
every command that takes ``--results-dir`` (Slice 4 — orgname resolution
gate).

Schema (locked in CONTEXT.md "Schema for mlperf-results.yaml"):

    mlperf_results_version: 1
    orgname: <as supplied>
    initialized_at: <ISO-8601 timestamp>
    initialized_by: mlpstorage <version>

Hardening (T-1-03 — RESEARCH.md Security Domain V5):

``orgname`` is constrained by ``pattern=r"^[A-Za-z0-9._-]+$"`` so a crafted
value containing path separators (``/``, ``\\``), parent-dir traversal
(``..`` on its own still matches the pattern but cannot escape after
``os.path.join`` because the orgname is a single path segment), control
chars, NUL bytes, whitespace, etc. is rejected at the schema layer before
it ever reaches the filesystem. ``min_length=1`` rejects the empty string.

``model_config = ConfigDict(extra='forbid')`` rejects unknown keys —
catches typos in hand-edited sentinels at read time, mirroring the
``StrictModel`` precedent in ``system_description/schema_validator.py``.

YAML I/O uses ``yaml.safe_load`` only (V12 ASVS — RESEARCH.md Security
Domain "YAML deserialization RCE"); the unsafe loader is never imported.

Refs: 01-canonical-layout-and-init / 01-01-PLAN.md Task 1; CONTEXT.md
"Locked Decisions → Schema for mlperf-results.yaml"; PATTERNS.md row
``results_dir/schema.py``.
"""

from __future__ import annotations

import yaml
from pydantic import BaseModel, ConfigDict, Field


class MlperfResultsSentinel(BaseModel):
    """Pydantic v2 model for ``<results-dir>/mlperf-results.yaml``.

    Fields are ordered to match the on-disk YAML layout produced by
    ``write_sentinel`` so a ``model_dump()`` round-trips byte-equivalently
    under ``sort_keys=False``.
    """

    model_config = ConfigDict(extra="forbid")

    mlperf_results_version: int = Field(ge=1)
    # T-1-03 — pattern rejects path separators, NUL bytes, control chars,
    # whitespace; min_length=1 rejects empty. Allowed alphabet matches the
    # "submitter directory name" convention in Rules.md §2.1.5.
    orgname: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    initialized_at: str = Field(min_length=1)
    initialized_by: str = Field(min_length=1)


def validate_dict(payload: dict) -> MlperfResultsSentinel:
    """Validate a pre-loaded dict and return the model.

    Raises ``pydantic.ValidationError`` (unwrapped) on failure — the caller
    decides whether to re-raise as a domain-specific error (e.g.
    ``ResultsDirNotInitializedError`` wraps it in ``read_sentinel``).

    Mirrors the helper shape of
    ``mlpstorage_py.system_description.schema_validator.validate_dict``,
    but returns the validated model rather than an error-string list — the
    sentinel is small enough that returning the parsed object is more
    useful than a YAML-line-annotated error list.
    """
    return MlperfResultsSentinel.model_validate(payload)


def validate_file(path: str) -> MlperfResultsSentinel:
    """Load YAML at ``path`` via ``yaml.safe_load`` and validate.

    Raises:
        OSError / FileNotFoundError — when the path cannot be read.
        yaml.YAMLError — when the file is not parseable as YAML.
        pydantic.ValidationError — when the parsed dict fails schema
            validation (missing fields, unknown fields, pattern mismatch,
            version < 1, etc.).

    The function deliberately re-raises these exceptions unwrapped so the
    sentinel-read layer (``read_sentinel``) can re-raise them as
    ``ResultsDirNotInitializedError`` with the correct ``__cause__``.

    Security: only ``yaml.safe_load`` is used — never the unsafe loader
    (V12 ASVS).
    """
    # WR-04: pin encoding="utf-8" so the sentinel round-trips deterministically
    # regardless of LANG / LC_ALL on the host. Today the constraints on
    # ``orgname`` (``[A-Za-z0-9._-]+``) and the ISO-8601 timestamp keep both
    # files ASCII, but ``initialized_by`` embeds ``VERSION`` and a future
    # release tag with a non-ASCII character (or a new schema field) would
    # corrupt on a ``LANG=C`` host.
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        # Pydantic would also reject a non-dict, but the resulting error
        # is opaque ("Input should be a valid dictionary"); raise a tighter
        # exception so the caller's wrap-as-NotInitialized message is clear.
        raise ValueError(
            f"sentinel file {path!r} must contain a YAML mapping at the top "
            f"level (got {type(data).__name__})"
        )
    return validate_dict(data)
