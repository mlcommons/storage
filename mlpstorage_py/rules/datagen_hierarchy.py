"""Training datagen hierarchy validation and self-describing manifest.

This module packages the four helpers used by:

- the training benchmark's datagen path in ``mlpstorage_py/benchmarks/``
  (pre-run guard + post-run manifest write + leaf WARN)
- the report_generator's datagen group handling in
  ``mlpstorage_py/report_generator.py``
  (supported-model INVALID gate + leaf WARN)

Design notes
------------

The DLIO datagen leaf under ``<results-dir>/.../training/<model>/datagen/<ts>/``
carries the files listed in
``mlpstorage_py/submission_checker/constants.py``
(DATAGEN_REQUIRED_FILES / DATAGEN_REQUIRED_FOLDERS). We reuse those constants
so the same file-set contract is enforced from three call sites (submission
checker, run-checker, reportgen) without drift.

The manifest at ``<data-dir>/<model>/.mlps-datagen-manifest.json`` is the
self-describing record a future ``training run`` command will read to compare
its requested workload size against the dataset the operator generated. The
schema is intentionally minimal — only the three DLIO knobs that determine
whether a run's request fits the dataset, plus provenance fields for audit.

Whatif exemption
----------------

Per the D-29 policy already in place in the report_generator, ``whatif`` is
a simulation modality that skips submission-strict validation. The
``validate_supported_model`` helper preserves that policy so unsupported
research models pass through when the operator is exploring configurations
before committing to a closed/open submission.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional

from mlpstorage_py import __version__ as _MLPSTORAGE_VERSION
from mlpstorage_py.config import (
    MODELS,
    MODELS_CLOSED,
    MODELS_OPEN,
)
from mlpstorage_py.errors import ConfigurationError, ErrorCode
from mlpstorage_py.submission_checker.constants import (
    DATAGEN_REQUIRED_FILES,
    DATAGEN_REQUIRED_FOLDERS,
)


DATAGEN_MANIFEST_FILENAME = ".mlps-datagen-manifest.json"
DATAGEN_MANIFEST_SCHEMA_VERSION = 1

# Files required inside dlio_config/ per submission_checker rule §2.1.15
# (see checks/directory_checks.py:129-130). Kept local because §2.1.15
# defines them inline in the check rather than exporting a constant.
_DLIO_CONFIG_REQUIRED_FILES = ("config.yaml", "hydra.yaml", "overrides.yaml")

_MODEL_ALLOWLIST: Dict[str, List[str]] = {
    "closed": MODELS_CLOSED,
    "open": MODELS_OPEN,
    # Whatif carries no submission-strict model check; the reportgen
    # D-29 policy already skips whatif for INVALID gates, and this
    # helper mirrors that so a whatif operator can iterate on any
    # known model without fighting the validator.
    "whatif": MODELS,
}


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

    allowed = _MODEL_ALLOWLIST.get(mode)
    if allowed is None:
        raise ConfigurationError(
            f"Unsupported submission mode {mode!r} — expected one of "
            f"{sorted(_MODEL_ALLOWLIST)}.",
            parameter="mode",
            expected=sorted(_MODEL_ALLOWLIST),
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

    Accepted states (no raise):
        - ``<data-dir>/<model>`` does not exist.
        - ``<data-dir>/<model>`` exists as an empty directory (the
          operator pre-created the mount point).

    Rejected states (raise ``ConfigurationError``):
        - ``<data-dir>/<model>`` exists as a non-empty directory.
        - ``<data-dir>/<model>`` exists as a file, symlink to a file,
          or other non-directory node.
    """
    model_dir = os.path.join(data_dir, model)

    if not os.path.lexists(model_dir):
        return

    if not os.path.isdir(model_dir):
        raise ConfigurationError(
            f"{model_dir!r} exists and is not a directory — refusing to "
            f"proceed with datagen.",
            parameter="data_dir",
            actual=model_dir,
            suggestion=(
                f"Remove {model_dir!r} manually or pass a different "
                "--data-dir."
            ),
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )

    try:
        has_entries = any(True for _ in os.scandir(model_dir))
    except OSError as e:
        raise ConfigurationError(
            f"Cannot inspect {model_dir!r}: {e}.",
            parameter="data_dir",
            actual=model_dir,
            code=ErrorCode.CONFIG_INVALID_VALUE,
        ) from e

    if has_entries:
        raise ConfigurationError(
            f"{model_dir!r} already exists and is not empty — refusing to "
            f"overwrite an existing {model!r} dataset. A re-datagen with "
            f"different parameters (e.g. num_samples_per_file) may only "
            f"partially overwrite files on filename collision, silently "
            f"corrupting the dataset.",
            parameter="data_dir",
            actual=model_dir,
            suggestion=(
                f"Remove {model_dir!r} (e.g. rm -rf) or pass a different "
                "--data-dir."
            ),
            code=ErrorCode.CONFIG_INVALID_VALUE,
        )


# --------------------------------------------------------------------------- #
# validate_datagen_leaf                                                       #
# --------------------------------------------------------------------------- #


def _select_regex_set(mapping: Dict[str, List[str]]) -> List[str]:
    """Prefer v3.0 patterns; fall back to ``default`` if absent."""
    return mapping.get("v3.0") or mapping.get("default") or []


def validate_datagen_leaf(leaf_path: str) -> List[str]:
    """Return a list of missing-item descriptions for a datagen leaf.

    The check is stat-only: presence of the four datagen-required
    files (regex-matched against the DATAGEN_REQUIRED_FILES set) and
    the dlio_config/ folder with its three inner YAMLs. Contents are
    not inspected — the caller is expected to fail fast on
    presence, then let downstream tooling (submission_checker, DLIO)
    speak to content correctness.

    Returns:
        A list of human-readable missing-item strings. Empty list
        means the leaf is complete. Never raises.
    """
    if not os.path.isdir(leaf_path):
        return [f"datagen leaf directory not found: {leaf_path}"]

    missing: List[str] = []
    entries = os.listdir(leaf_path)

    for pattern in _select_regex_set(DATAGEN_REQUIRED_FILES):
        if not any(re.search(pattern, name) for name in entries):
            missing.append(_pretty_file_pattern(pattern))

    for folder in _select_regex_set(DATAGEN_REQUIRED_FOLDERS):
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


def _pretty_file_pattern(pattern: str) -> str:
    """Convert a DATAGEN_REQUIRED_FILES regex to a human-readable name.

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
    schema documented in this module's docstring. A future
    ``training run`` command will read this file to compare its
    requested workload size against what the dataset actually
    supports.

    Args:
        data_dir: The ``--data-dir`` path the operator passed to
            datagen. The manifest lands at ``data_dir/model/FILENAME``.
        model: The training model name (drives the per-model subdir).
        dataset_params: A dict with (at minimum) the three keys
            ``num_files_train``, ``num_samples_per_file``, and
            ``record_length_bytes``. Extra keys are ignored — the
            manifest schema is intentionally tight.
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

    manifest = {
        "schema_version": DATAGEN_MANIFEST_SCHEMA_VERSION,
        "model": model,
        "num_files_train": dataset_params["num_files_train"],
        "num_samples_per_file": dataset_params["num_samples_per_file"],
        "record_length_bytes": dataset_params["record_length_bytes"],
        "created_at": created_at,
        "mlpstorage_version": _MLPSTORAGE_VERSION,
        "source_datagen_result_dir": source_datagen_result_dir,
    }

    model_dir = os.path.join(data_dir, model)
    os.makedirs(model_dir, exist_ok=True)
    manifest_path = os.path.abspath(
        os.path.join(model_dir, DATAGEN_MANIFEST_FILENAME)
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest_path
