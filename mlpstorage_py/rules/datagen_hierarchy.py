"""DLIO leaf hierarchy validation and self-describing datagen manifest.

This module packages the helpers used by:

- the training benchmark's datagen path in ``mlpstorage_py/benchmarks/``
  (pre-run guard + post-run manifest write + leaf WARN)
- the report_generator's datagen group handling in
  ``mlpstorage_py/report_generator.py``
  (supported-model INVALID gate + leaf WARN)
- the training and checkpointing benchmarks' post-run loud-fail hooks
  (storage#761 sibling of storage#744 for datagen and storage#759 for
  kvcache) — ``validate_run_leaf`` / ``validate_checkpoint_leaf``.

Design notes
------------

Each DLIO leaf under
``<results-dir>/.../{training/<model>/{datagen,run},checkpointing/<model>}/<ts>/``
carries a required-file set listed in the rules editions table
(``mlpstorage_py/rules/editions.yaml``, the current edition's ``checker:``
block). We read the same table the submission checker's ``Config`` reads so
the file-set contract is enforced from every call site (submission checker,
run-checker, reportgen) without drift.

The manifest at ``<data-dir>/<model>/.mlps-datagen-manifest.json`` is the
self-describing record ``training run`` / ``configview`` read at start
(``TrainingBenchmark._apply_datagen_manifest``) to compare the requested
workload against the dataset the operator generated. Schema v1: the three
DLIO knobs that decide whether a run fits the dataset (``num_files_train``,
``num_samples_per_file``, ``record_length_bytes``), provenance fields for
audit, and — additive, same schema version — the layout/invariant fields
the run side needs to reproduce the generated names under ``skip_listing``:
``num_subfolders_train``, ``dataset_format``, ``rules_edition``. Readers
tolerate their absence (manifests written before the extension).

Whatif exemption
----------------

Per the D-29 policy already in place in the report_generator, ``whatif`` is
a simulation modality that skips submission-strict validation. The
``validate_supported_model`` helper preserves that policy so unsupported
research models pass through when the operator is exploring configurations
before committing to a closed/open submission.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse

import s3dlio

from mlpstorage_py import __version__ as _MLPSTORAGE_VERSION
from mlpstorage_py.config import MODELS, RULES_EDITION
from mlpstorage_py.errors import ConfigurationError, ErrorCode
from mlpstorage_py.editions import checker_parameters, current_edition


DATAGEN_MANIFEST_FILENAME = ".mlps-datagen-manifest.json"
DATAGEN_MANIFEST_SCHEMA_VERSION = 1

# Files required inside dlio_config/ per submission_checker rule §2.1.15
# (see checks/directory_checks.py:129-130). Kept local because §2.1.15
# defines them inline in the check rather than exporting a constant.
_DLIO_CONFIG_REQUIRED_FILES = ("config.yaml", "hydra.yaml", "overrides.yaml")

_SUBMISSION_MODES = ("closed", "open", "whatif")


def _model_allowlist(mode: str) -> Optional[List[str]]:
    """The training models ``mode`` accepts: closed / open from the current
    rules edition (editions.yaml ``workloads:``); whatif carries no
    submission-strict model check (the reportgen D-29 policy already skips
    whatif for INVALID gates), so every model the tool can run is allowed."""
    if mode == "whatif":
        return list(MODELS)
    if mode in _SUBMISSION_MODES:
        return current_edition().models("training", mode)
    return None


# --------------------------------------------------------------------------- #
# validate_supported_model                                                    #
# --------------------------------------------------------------------------- #


def validate_supported_model(model: str, mode: str) -> None:
    """Raise ``ConfigurationError`` if ``model`` is not in the allowlist for ``mode``.

    Args:
        model: The training model name (e.g. ``"unet3d"``).
        mode: One of ``"closed"``, ``"open"``, ``"whatif"``.

    The CLI already restricts model choices per mode at parse time
    (see ``mlpstorage_py/cli/training_args.py``); this helper is the
    runtime defense that catches call sites which construct
    benchmark runs from persisted state (e.g. reportgen scanning a
    submission tree from a prior version).
    """
    if mode == "whatif":
        return

    allowed = _model_allowlist(mode)
    if allowed is None:
        raise ConfigurationError(
            f"Unsupported submission mode {mode!r} — expected one of "
            f"{sorted(_SUBMISSION_MODES)}.",
            parameter="mode",
            expected=sorted(_SUBMISSION_MODES),
            actual=mode,
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )

    if model not in allowed:
        raise ConfigurationError(
            f"Model {model!r} is not permitted for {mode!r} submissions "
            f"per Rules.md v3.0 — allowed models: {sorted(allowed)}.",
            parameter="model",
            expected=sorted(allowed),
            actual=model,
            suggestion=(
                f"Use one of {sorted(allowed)} for {mode!r}, or run in "
                "whatif mode for exploratory work with other models."
            ),
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )


# --------------------------------------------------------------------------- #
# URI helpers — local vs object storage dispatch                              #
# --------------------------------------------------------------------------- #

# Object-storage URI schemes that s3dlio abstracts uniformly. The
# ``direct://`` scheme is s3dlio's zero-copy file-backed backend; from
# our perspective it's just another URI-based backend so it lives in
# the same branch as s3/gs/az.
_OBJECT_STORAGE_SCHEMES = frozenset({"s3", "gs", "az", "direct"})


def _is_object_uri(path: str) -> bool:
    """Return True if ``path`` is an object-storage URI s3dlio understands.

    An empty scheme (bare ``/local/path``) and ``file://`` are both
    treated as local filesystem — the local branch uses ``os.path``
    and ``open()`` directly.
    """
    scheme = urlparse(path).scheme
    return scheme in _OBJECT_STORAGE_SCHEMES


def _join_model_location(data_dir: str, model: str) -> str:
    """Join ``data_dir`` and ``model`` into a per-model location string.

    For local paths uses ``os.path.join``. For object-storage URIs
    normalizes any trailing slash and appends ``/<model>``. The
    caller may further append ``/<filename>`` for a manifest URI.
    """
    if _is_object_uri(data_dir):
        return data_dir.rstrip("/") + "/" + model
    return os.path.join(data_dir, model)


# --------------------------------------------------------------------------- #
# assert_data_dir_hierarchy_absent                                            #
# --------------------------------------------------------------------------- #


def assert_data_dir_hierarchy_absent(data_dir: str, model: str) -> None:
    """Refuse to overwrite an existing ``<data-dir>/<model>`` tree.

    Rationale: a re-datagen with different ``num_samples_per_file``
    (or any other per-file byte-size knob) would leave the old and
    new files interleaved on filename collision, silently corrupting
    the dataset for a subsequent run. Rather than teach DLIO to
    detect that, we refuse and force the operator to ``rm -rf`` or
    supply a different ``--data-dir``.

    Dispatches on ``data_dir``'s URI scheme:

    - Local filesystem (empty scheme or ``file://``): stat-based
      check with ``os.scandir``.
    - Object storage (``s3://``, ``gs://``, ``az://``, ``direct://``):
      bounded LIST via ``s3dlio.list(uri, recursive=False)``. The
      ``recursive=False`` flag yields a delimiter-based listing that
      returns only direct children — bounded regardless of how many
      objects sit under a prior datagen's prefix. Any s3dlio
      exception during the LIST fails safe (aborts) per the design
      decision: a transient object-store error must NOT be silently
      treated as "assume empty, proceed".

    Accepted states (no raise):
        - ``<data-dir>/<model>`` does not exist.
        - ``<data-dir>/<model>`` exists as an empty directory (local
          case only — the operator pre-created the mount point).

    Rejected states (raise ``ConfigurationError``):
        - ``<data-dir>/<model>`` contains any file/object.
        - ``<data-dir>/<model>`` exists as a non-directory node
          (local case only).
        - Object-storage LIST raises for any reason.
    """
    model_location = _join_model_location(data_dir, model)

    if _is_object_uri(data_dir):
        _assert_object_hierarchy_absent(model_location, model)
        return

    if not os.path.lexists(model_location):
        return

    if not os.path.isdir(model_location):
        raise ConfigurationError(
            f"{model_location!r} exists and is not a directory — refusing to "
            f"proceed with datagen.",
            parameter="data_dir",
            actual=model_location,
            suggestion=(
                f"Remove {model_location!r} manually or pass a different "
                "--data-dir."
            ),
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )

    try:
        has_entries = any(True for _ in os.scandir(model_location))
    except OSError as e:
        raise ConfigurationError(
            f"Cannot inspect {model_location!r}: {e}.",
            parameter="data_dir",
            actual=model_location,
            code=ErrorCode.CONFIG_INVALID_VALUE,
        ) from e

    if has_entries:
        raise ConfigurationError(
            f"{model_location!r} already exists and is not empty — refusing to "
            f"overwrite an existing {model!r} dataset. A re-datagen with "
            f"different parameters (e.g. num_samples_per_file) may only "
            f"partially overwrite files on filename collision, silently "
            f"corrupting the dataset.",
            parameter="data_dir",
            actual=model_location,
            suggestion=(
                f"Remove {model_location!r} (e.g. rm -rf) or pass a different "
                "--data-dir."
            ),
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )


def _assert_object_hierarchy_absent(model_uri: str, model: str) -> None:
    """Object-storage half of :func:`assert_data_dir_hierarchy_absent`.

    See parent docstring for semantics. Any s3dlio exception during
    the LIST is wrapped as a ``ConfigurationError`` and the original
    is chained via ``__cause__`` so the operator can diagnose the
    underlying object-store failure.
    """
    try:
        entries = s3dlio.list(model_uri, recursive=False)
    except Exception as e:
        raise ConfigurationError(
            f"Cannot inspect {model_uri!r} for prior {model!r} datagen "
            f"output: {e}. Refusing to proceed rather than silently "
            f"assume the prefix is empty.",
            parameter="data_dir",
            actual=model_uri,
            code=ErrorCode.CONFIG_INVALID_VALUE,
        ) from e

    if entries:
        raise ConfigurationError(
            f"{model_uri!r} already contains objects — refusing to "
            f"overwrite an existing {model!r} dataset. A re-datagen with "
            f"different parameters (e.g. num_samples_per_file) may only "
            f"partially overwrite objects on key collision, silently "
            f"corrupting the dataset.",
            parameter="data_dir",
            actual=model_uri,
            suggestion=(
                f"Remove the existing objects under {model_uri!r} or pass a "
                "different --data-dir."
            ),
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )


# --------------------------------------------------------------------------- #
# validate_datagen_leaf                                                       #
# --------------------------------------------------------------------------- #


def _params():
    """The current rules edition's ``checker:`` block (required leaf contents)."""
    return checker_parameters()


def _validate_leaf(
    leaf_path: str,
    files_map: List[str],
    folders_map: List[str],
    leaf_kind: str,
) -> List[str]:
    """Stat-only presence check against a required-files / folders contract.

    Shared helper for the three DLIO leaf validators: datagen, run,
    and checkpointing. Each leaf carries a different required-file
    regex set (see ``rules/editions.yaml`` ``checker:``) but the check
    shape — presence of the regex-matched files plus presence of the
    ``dlio_config/`` folder with its three inner YAMLs — is identical.

    Returns:
        Human-readable missing-item strings; empty list means complete.
    """
    if not os.path.isdir(leaf_path):
        return [f"{leaf_kind} leaf directory not found: {leaf_path}"]

    missing: List[str] = []
    entries = os.listdir(leaf_path)

    for pattern in files_map:
        if not any(re.search(pattern, name) for name in entries):
            missing.append(_pretty_file_pattern(pattern))

    for folder in folders_map:
        folder_path = os.path.join(leaf_path, folder)
        if not os.path.isdir(folder_path):
            missing.append(f"required folder missing: {folder}/")
            # No point drilling into the folder's expected inner files
            # if the folder itself is absent — one message is clearer.
            continue
        inner_entries = set(os.listdir(folder_path))
        for inner_name in _DLIO_CONFIG_REQUIRED_FILES:
            if inner_name not in inner_entries:
                missing.append(f"{folder}/{inner_name}")

    return missing


def validate_datagen_leaf(leaf_path: str) -> List[str]:
    """Return a list of missing-item descriptions for a datagen leaf.

    The check is stat-only: presence of the four datagen-required
    files (regex-matched against the edition's ``datagen_required_files``) and
    the dlio_config/ folder with its three inner YAMLs. Contents are
    not inspected — the caller is expected to fail fast on
    presence, then let downstream tooling (submission_checker, DLIO)
    speak to content correctness.

    Returns:
        A list of human-readable missing-item strings. Empty list
        means the leaf is complete. Never raises.
    """
    return _validate_leaf(
        leaf_path, _params().datagen_required_files, _params().datagen_required_folders, "datagen"
    )


def validate_run_leaf(leaf_path: str) -> List[str]:
    """Return a list of missing-item descriptions for a training-run leaf.

    Mirrors ``validate_datagen_leaf`` but against the run-side contract
    (the edition's ``run_required_files`` / ``run_required_folders``). Used by the
    training benchmark's post-run loud-fail hook so that a DLIO
    subprocess which exits non-zero — or exits zero but writes no
    outputs — surfaces as an ERROR at run time rather than a silent
    "success" the operator only discovers when
    ``mlpstorage validate`` rejects the submission (storage#761).
    """
    return _validate_leaf(
        leaf_path, _params().run_required_files, _params().run_required_folders, "run"
    )


def validate_checkpoint_leaf(leaf_path: str) -> List[str]:
    """Return a list of missing-item descriptions for a checkpointing leaf.

    Same shape as ``validate_run_leaf`` but against
    the edition's ``checkpoint_required_files`` / ``checkpoint_required_folders``.
    Called from ``CheckpointingBenchmark._run`` to close the
    silent-success gap on the checkpointing subprocess (sibling of
    storage#761).
    """
    return _validate_leaf(
        leaf_path, _params().checkpoint_required_files, _params().checkpoint_required_folders,
        "checkpoint",
    )


def _pretty_file_pattern(pattern: str) -> str:
    """Convert a required-file regex to a human-readable name.

    The submission_checker patterns are anchored regexes like
    ``r"training_datagen\\.stdout\\.log$"``. Strip the trailing ``$``
    and un-escape dots so the missing-item message reads like a
    filename rather than a regex.
    """
    name = pattern.rstrip("$")
    name = name.replace(r"\.", ".")
    # Metadata pattern uses ``.*`` for the timestamp segment — leave
    # it in place; the operator will recognize the shape.
    return name


# --------------------------------------------------------------------------- #
# write_datagen_manifest                                                      #
# --------------------------------------------------------------------------- #


def write_datagen_manifest(
    data_dir: str,
    model: str,
    dataset_params: Mapping[str, Any],
    source_datagen_result_dir: str,
    now: Optional[datetime.datetime] = None,
) -> str:
    """Write the self-describing manifest for a completed datagen run.

    Emits ``<data-dir>/<model>/.mlps-datagen-manifest.json`` with the
    schema documented in this module's docstring. ``training run`` /
    ``configview`` read it back (:func:`read_datagen_manifest`) to
    compare the requested workload against what the dataset actually
    supports and to tell DLIO the generated file count.

    Args:
        data_dir: The ``--data-dir`` path the operator passed to
            datagen. The manifest lands at ``data_dir/model/FILENAME``.
        model: The training model name (drives the per-model subdir).
        dataset_params: A dict with (at minimum) the three keys
            ``num_files_train``, ``num_samples_per_file``, and
            ``record_length_bytes``. ``num_subfolders_train`` (default
            0, DLIO's own default) and ``format`` are recorded when
            present; every other key is ignored.
        source_datagen_result_dir: Absolute path to the datagen leaf
            under ``--results-dir``. Written as-is for provenance.
        now: Injectable clock for tests. Defaults to
            ``datetime.datetime.now(timezone.utc)``.

    Returns:
        Absolute path to the written manifest file.

    Raises:
        ConfigurationError: If any of the three required
        ``dataset_params`` keys is absent. Loud rather than defaulted
        to zero — a missing field means the caller passed a broken
        merged config, and silently writing ``0`` would corrupt the
        future run-vs-datagen comparison.
    """
    required = ("num_files_train", "num_samples_per_file", "record_length_bytes")
    missing = [k for k in required if k not in dataset_params]
    if missing:
        raise ConfigurationError(
            f"Datagen manifest cannot be written — dataset_params is "
            f"missing required key(s): {missing}. All three of "
            f"{list(required)} must be present in the merged DLIO "
            f"config.",
            parameter="dataset_params",
            expected=list(required),
            actual=sorted(dataset_params.keys()),
            code=ErrorCode.CONFIG_MISSING_REQUIRED,
        )

    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    # ISO 8601 UTC with Z suffix — the tests pin the Z suffix and this
    # format is unambiguous across parsers.
    created_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        num_subfolders_train = int(dataset_params.get("num_subfolders_train") or 0)
    except (TypeError, ValueError) as e:
        raise ConfigurationError(
            f"Datagen manifest cannot be written — dataset.num_subfolders_train "
            f"is not an integer: {dataset_params.get('num_subfolders_train')!r}.",
            parameter="dataset.num_subfolders_train",
            actual=dataset_params.get("num_subfolders_train"),
            code=ErrorCode.CONFIG_INVALID_VALUE,
        ) from e
    dataset_format = dataset_params.get("format")

    manifest = {
        "schema_version": DATAGEN_MANIFEST_SCHEMA_VERSION,
        "model": model,
        "num_files_train": dataset_params["num_files_train"],
        "num_samples_per_file": dataset_params["num_samples_per_file"],
        "record_length_bytes": dataset_params["record_length_bytes"],
        "created_at": created_at,
        "mlpstorage_version": _MLPSTORAGE_VERSION,
        "source_datagen_result_dir": source_datagen_result_dir,
        # Additive v1 fields (storage#571 Q4 reader): the layout the run
        # must reproduce under skip_listing, the format invariant, and the
        # rules edition the dataset was generated under.
        "num_subfolders_train": num_subfolders_train,
        "dataset_format": None if dataset_format is None else str(dataset_format),
        "rules_edition": RULES_EDITION,
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")

    if _is_object_uri(data_dir):
        manifest_uri = (
            _join_model_location(data_dir, model)
            + "/"
            + DATAGEN_MANIFEST_FILENAME
        )
        try:
            s3dlio.put_bytes(manifest_uri, payload)
        except Exception as e:
            # Loud failure over silent skip — a manifest that fails
            # to land leaves the dataset without provenance and
            # breaks the future run-vs-datagen size check.
            raise ConfigurationError(
                f"Failed to write datagen manifest to {manifest_uri!r}: {e}.",
                parameter="data_dir",
                actual=manifest_uri,
                code=ErrorCode.CONFIG_INVALID_VALUE,
            ) from e
        return manifest_uri

    model_dir = os.path.join(data_dir, model)
    os.makedirs(model_dir, exist_ok=True)
    manifest_path = os.path.abspath(
        os.path.join(model_dir, DATAGEN_MANIFEST_FILENAME)
    )
    with open(manifest_path, "wb") as f:
        f.write(payload)
    return manifest_path


# --------------------------------------------------------------------------- #
# read_datagen_manifest                                                       #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class DatagenManifest:
    """A parsed ``.mlps-datagen-manifest.json`` (schema v1).

    ``num_subfolders_train`` / ``dataset_format`` / ``rules_edition`` are
    ``None`` for manifests written before the additive extension; the
    run-side check skips the invariants it cannot see.
    """

    location: str
    schema_version: int
    model: str
    num_files_train: int
    num_samples_per_file: int
    record_length_bytes: int
    created_at: Optional[str] = None
    mlpstorage_version: Optional[str] = None
    source_datagen_result_dir: Optional[str] = None
    num_subfolders_train: Optional[int] = None
    dataset_format: Optional[str] = None
    rules_edition: Optional[str] = None


_MANIFEST_REQUIRED_KEYS = (
    "schema_version", "model", "num_files_train", "num_samples_per_file",
    "record_length_bytes",
)


def datagen_manifest_location(data_dir: str, model: str) -> str:
    """Path (local) or URI (object storage) of the manifest for ``model``."""
    if _is_object_uri(data_dir):
        return _join_model_location(data_dir, model) + "/" + DATAGEN_MANIFEST_FILENAME
    return os.path.join(data_dir, model, DATAGEN_MANIFEST_FILENAME)


def _parse_datagen_manifest(raw: bytes, location: str) -> DatagenManifest:
    """Decode + validate a manifest body. Loud on anything malformed."""
    try:
        data = json.loads(bytes(raw).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise ConfigurationError(
            f"Datagen manifest at {location!r} is not valid JSON: {e}.",
            parameter="data_dir",
            actual=location,
            suggestion=(
                "The dataset's provenance record is corrupt. Regenerate the "
                "dataset, or pass --skip-validation to run without the "
                "manifest check."
            ),
            code=ErrorCode.CONFIG_PARSE_ERROR,
        ) from e

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Datagen manifest at {location!r} must be a JSON object, got "
            f"{type(data).__name__}.",
            parameter="data_dir",
            actual=location,
            code=ErrorCode.CONFIG_PARSE_ERROR,
        )

    missing = [k for k in _MANIFEST_REQUIRED_KEYS if k not in data]
    if missing:
        raise ConfigurationError(
            f"Datagen manifest at {location!r} is missing required key(s): "
            f"{missing}.",
            parameter="data_dir",
            expected=list(_MANIFEST_REQUIRED_KEYS),
            actual=sorted(data.keys()),
            code=ErrorCode.CONFIG_MISSING_REQUIRED,
        )

    def _int(key: str, required: bool) -> Optional[int]:
        value = data.get(key)
        if value is None and not required:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise ConfigurationError(
                f"Datagen manifest at {location!r}: {key} must be an "
                f"integer, got {value!r}.",
                parameter=key,
                actual=value,
                code=ErrorCode.CONFIG_PARSE_ERROR,
            ) from e

    schema_version = _int("schema_version", required=True)
    if schema_version != DATAGEN_MANIFEST_SCHEMA_VERSION:
        raise ConfigurationError(
            f"Datagen manifest at {location!r} has schema_version "
            f"{schema_version}; this mlpstorage reads schema_version "
            f"{DATAGEN_MANIFEST_SCHEMA_VERSION}.",
            parameter="schema_version",
            expected=DATAGEN_MANIFEST_SCHEMA_VERSION,
            actual=schema_version,
            suggestion="Upgrade mlpstorage, or regenerate the dataset with this version.",
            code=ErrorCode.CONFIG_INCOMPATIBLE,
        )

    def _opt_str(key: str) -> Optional[str]:
        value = data.get(key)
        return None if value is None else str(value)

    return DatagenManifest(
        location=location,
        schema_version=schema_version,
        model=str(data["model"]),
        num_files_train=_int("num_files_train", required=True),
        num_samples_per_file=_int("num_samples_per_file", required=True),
        record_length_bytes=_int("record_length_bytes", required=True),
        created_at=_opt_str("created_at"),
        mlpstorage_version=_opt_str("mlpstorage_version"),
        source_datagen_result_dir=_opt_str("source_datagen_result_dir"),
        num_subfolders_train=_int("num_subfolders_train", required=False),
        dataset_format=_opt_str("dataset_format"),
        rules_edition=_opt_str("rules_edition"),
    )


def read_datagen_manifest(data_dir: str, model: str) -> Optional[DatagenManifest]:
    """Read ``<data-dir>/<model>/.mlps-datagen-manifest.json`` if present.

    Returns ``None`` when no manifest exists (the caller decides how loud
    to be about that — a missing manifest is a warning, not an error).
    A manifest that exists but cannot be read or parsed raises
    ``ConfigurationError``: silence there would hide a real problem.

    Object storage (``s3://``, ``gs://``, ``az://``, ``direct://``): one
    ``s3dlio.exists`` (HEAD) tells "absent" from "read failed", then a
    single ``s3dlio.get``. No retry — the eventual-consistency window is
    dwarfed by the human latency between ``datagen`` and ``run``. Any
    s3dlio exception fails safe as an error (same policy as
    :func:`assert_data_dir_hierarchy_absent`).
    """
    location = datagen_manifest_location(data_dir, model)

    if _is_object_uri(data_dir):
        try:
            present = s3dlio.exists(location)
        except Exception as e:
            raise ConfigurationError(
                f"Cannot check for the datagen manifest at {location!r}: {e}. "
                f"Refusing to guess whether the dataset has a manifest.",
                parameter="data_dir",
                actual=location,
                code=ErrorCode.CONFIG_INVALID_VALUE,
            ) from e
        if not present:
            return None
        try:
            raw = s3dlio.get(location)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to read the datagen manifest at {location!r}: {e}.",
                parameter="data_dir",
                actual=location,
                code=ErrorCode.CONFIG_INVALID_VALUE,
            ) from e
        return _parse_datagen_manifest(raw, location)

    if not os.path.exists(location):
        return None
    try:
        with open(location, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise ConfigurationError(
            f"Cannot read the datagen manifest at {location!r}: {e}.",
            parameter="data_dir",
            actual=location,
            code=ErrorCode.CONFIG_INVALID_VALUE,
        ) from e
    return _parse_datagen_manifest(raw, location)
