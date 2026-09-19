"""Run-leaf provenance stamps and the per-submission manifest.

Every run leaf written by ``mlpstorage`` carries a ``provenance.json`` sidecar
that says what produced it:

* ``rules_edition`` — the Rules.md edition the tool implements
  (``config.RULES_EDITION``); the comparability key across rounds;
* ``layout_version`` — the results-dir / leaf layout
  (``results_dir.MLPERF_RESULTS_VERSION``); only readers key on it;
* ``tool`` — mlpstorage version, git SHA and the code image the leaf points at;
* ``dlio`` — the DLIO package version, source URL and git commit, read from
  the installed distribution's PEP 610 ``direct_url.json``;
* ``storage_library`` — the client library for ``object`` runs (s3dlio);
* ``core_config`` — the ``core-config-v1`` hash of the workload-defining DLIO
  parameters (allowlist in ``rules/core_config_keys.yaml``), plus the exact
  keys hashed so the value is reproducible from the leaf alone.

Every stamp has a ``provenance`` tag from a closed vocabulary saying how it
was obtained, and an undeterminable stamp is the string ``"unknown"`` — never
``null``, never absent.

Leaves written before this sidecar existed (the whole frozen v3.0 tree) are
never rewritten. ``read_leaf_provenance`` derives an in-memory stamp for them
from the pointed-to image's ``.code-hash.json`` and ``uv.lock`` and from the
leaf's own metadata, tagged accordingly. The archive importer will later write
the same file shape as a sidecar; the reader has one seam either way.

``reportgen`` writes ``<results-dir>/<mode>/<org>/submission.yaml`` — an
inventory (systems, run leaves with their stamps, code images) plus the
declared rules edition — for every organization in a live results-dir. It
sits under ``<mode>/<org>/`` so it travels with the submission. Submission
checker rules PROV-01 / PROV-02 (``checks/provenance_checks.py``) verify both
files and stay silent on trees that predate them.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from dataclasses import dataclass
from importlib import metadata as _importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml

from mlpstorage_py import VERSION
from mlpstorage_py.config import RULES_EDITION
from mlpstorage_py.results_dir import MLPERF_RESULTS_FILENAME, MLPERF_RESULTS_VERSION

# --- Constants --------------------------------------------------------------

PROVENANCE_FILENAME = "provenance.json"
MANIFEST_FILENAME = "submission.yaml"
PROVENANCE_SCHEMA = "mlps-run-provenance/1"
MANIFEST_SCHEMA = "mlps-submission-manifest/1"
CORE_CONFIG_ALGORITHM = "core-config-v1"
UNKNOWN = "unknown"
# Key that *_metadata.json gains when the run wrote a sidecar; PROV-01 treats
# a leaf that declares one but has none as an error, and stays silent for
# leaves that never declared one (pre-stamp trees).
METADATA_PROVENANCE_KEY = "provenance_file"

STAMP_FIELDS = ("rules_edition", "layout_version", "tool", "dlio",
                "storage_library", "core_config")
PROVENANCE_TAGS = frozenset({
    "runtime",           # measured in-process at run time
    "tool-constant",     # baked into the release
    "package-metadata",  # importlib.metadata (version only)
    "direct-url",        # PEP 610 direct_url.json (VCS commit)
    "code-hash-json",    # recovered from the code image's .code-hash.json
    "uv-lock",           # parsed from the code image's uv.lock
    "declared",          # a human wrote it (importer)
    "inferred",          # importer / reader heuristic
    "n/a",               # not applicable to this run
    "unknown",
})

_POINTER_ALGORITHM = "md5-tree-v2"
_POINTER_FILENAME = ".mlps-code-image"
_HASH_FILENAME = ".code-hash.json"
_POOL_SENTINEL_FILENAME = ".mlps-image-pool"
_GLOBAL_POOL_DIRNAME = "code-images"
_ALLOWLIST_FILE = Path(__file__).resolve().parent / "rules" / "core_config_keys.yaml"
_ALLOWLIST_SCHEMA = "mlps-core-config-allowlist/1"
_MODES = ("closed", "open", "whatif")
_FAMILIES = ("training", "checkpointing", "vector_database", "kv_cache")
_DLIO_DIST = "dlio_benchmark"
_S3DLIO_DIST = "s3dlio"


class ProvenanceError(ValueError):
    """A provenance.json / submission.yaml is missing, unparseable or invalid."""


def _now() -> str:
    return datetime.datetime.now(tz=datetime.UTC).isoformat(
        timespec="seconds").replace("+00:00", "Z")


# --- The stamp --------------------------------------------------------------

@dataclass
class RunProvenance:
    rules_edition: str
    layout_version: int
    tool: Dict[str, str]
    dlio: Dict[str, str]
    storage_library: Dict[str, str]
    core_config: Dict[str, Any]
    stamped_at: str
    stamped_by: str
    provenance: Dict[str, str]
    notes: Optional[str] = None
    schema: str = PROVENANCE_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        """Fixed key order — the on-disk layout."""
        d: Dict[str, Any] = {
            "schema": self.schema,
            "rules_edition": self.rules_edition,
            "layout_version": self.layout_version,
            "tool": dict(self.tool),
            "dlio": dict(self.dlio),
            "storage_library": dict(self.storage_library),
            "core_config": {
                "algorithm": self.core_config["algorithm"],
                "allowlist": self.core_config["allowlist"],
                "hash": self.core_config["hash"],
                "keys": list(self.core_config["keys"]),
            },
            "stamped_at": self.stamped_at,
            "stamped_by": self.stamped_by,
            "provenance": {k: self.provenance[k] for k in STAMP_FIELDS},
        }
        if self.notes is not None:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, data: Any) -> "RunProvenance":
        _validate_stamp(data)
        return cls(
            schema=data["schema"],
            rules_edition=data["rules_edition"],
            layout_version=data["layout_version"],
            tool=dict(data["tool"]),
            dlio=dict(data["dlio"]),
            storage_library=dict(data["storage_library"]),
            core_config={"algorithm": data["core_config"]["algorithm"],
                         "allowlist": data["core_config"]["allowlist"],
                         "hash": data["core_config"]["hash"],
                         "keys": list(data["core_config"]["keys"])},
            stamped_at=data["stamped_at"],
            stamped_by=data["stamped_by"],
            provenance=dict(data["provenance"]),
            notes=data.get("notes"),
        )


def _require_str(d: dict, key: str, where: str) -> None:
    v = d.get(key)
    if not isinstance(v, str) or not v:
        raise ProvenanceError(f"{where}.{key} must be a non-empty string, got {v!r}")


def _validate_stamp(data: Any) -> None:
    if not isinstance(data, dict):
        raise ProvenanceError("provenance stamp must be a JSON object")
    if data.get("schema") != PROVENANCE_SCHEMA:
        raise ProvenanceError(
            f"unsupported provenance schema {data.get('schema')!r} "
            f"(expected {PROVENANCE_SCHEMA!r})")
    required = ("schema", "rules_edition", "layout_version", "tool", "dlio",
                "storage_library", "core_config", "stamped_at", "stamped_by", "provenance")
    for key in required:
        if key not in data:
            raise ProvenanceError(f"provenance stamp is missing {key!r}")
    extra = set(data) - set(required) - {"notes"}
    if extra:
        raise ProvenanceError(f"provenance stamp has unexpected keys {sorted(extra)}")
    _reject_null(data, "stamp")
    _require_str(data, "rules_edition", "stamp")
    lv = data["layout_version"]
    if isinstance(lv, bool) or not isinstance(lv, int) or lv < 1:
        raise ProvenanceError(f"stamp.layout_version must be an integer >= 1, got {lv!r}")
    for section, keys in (("tool", ("name", "version", "git_sha", "code_image")),
                          ("dlio", ("version", "source", "commit"))):
        block = data[section]
        if not isinstance(block, dict) or set(block) != set(keys):
            raise ProvenanceError(f"stamp.{section} must have exactly the keys {list(keys)}")
        for k in keys:
            _require_str(block, k, f"stamp.{section}")
    sl = data["storage_library"]
    if not isinstance(sl, dict) or not set(sl) <= {"name", "version"}:
        raise ProvenanceError("stamp.storage_library must be {name[, version]}")
    _require_str(sl, "name", "stamp.storage_library")
    if sl["name"] != "none":
        _require_str(sl, "version", "stamp.storage_library")
    cc = data["core_config"]
    if not isinstance(cc, dict) or set(cc) != {"algorithm", "allowlist", "hash", "keys"}:
        raise ProvenanceError(
            "stamp.core_config must have exactly the keys [algorithm, allowlist, hash, keys]")
    if cc["algorithm"] != CORE_CONFIG_ALGORITHM:
        raise ProvenanceError(f"unsupported core_config algorithm {cc['algorithm']!r}")
    _require_str(cc, "allowlist", "stamp.core_config")
    _require_str(cc, "hash", "stamp.core_config")
    if not isinstance(cc["keys"], list) or not all(isinstance(k, str) for k in cc["keys"]):
        raise ProvenanceError("stamp.core_config.keys must be a list of strings")
    _require_str(data, "stamped_at", "stamp")
    _require_str(data, "stamped_by", "stamp")
    prov = data["provenance"]
    if not isinstance(prov, dict) or set(prov) != set(STAMP_FIELDS):
        raise ProvenanceError(
            f"stamp.provenance must have exactly the keys {list(STAMP_FIELDS)}")
    for k, tag in prov.items():
        if tag not in PROVENANCE_TAGS:
            raise ProvenanceError(
                f"stamp.provenance.{k}: {tag!r} is not one of {sorted(PROVENANCE_TAGS)}")
    if "notes" in data and not isinstance(data["notes"], str):
        raise ProvenanceError("stamp.notes must be a string")


def _reject_null(value: Any, where: str) -> None:
    if value is None:
        raise ProvenanceError(f"{where} is null; undeterminable stamps must be {UNKNOWN!r}")
    if isinstance(value, dict):
        for k, v in value.items():
            _reject_null(v, f"{where}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _reject_null(v, f"{where}[{i}]")


# --- core-config-v1 ---------------------------------------------------------

def load_allowlists() -> Dict[str, Any]:
    """The allowlist table from ``rules/core_config_keys.yaml``."""
    with open(_ALLOWLIST_FILE, "r", encoding="utf-8") as fh:
        table = yaml.safe_load(fh)
    if not isinstance(table, dict) or table.get("schema") != _ALLOWLIST_SCHEMA:
        raise ProvenanceError(f"{_ALLOWLIST_FILE}: unsupported allowlist schema")
    for family, list_id in table.get("families", {}).items():
        if list_id not in table.get("allowlists", {}):
            raise ProvenanceError(f"{_ALLOWLIST_FILE}: family {family} names unknown "
                                  f"allowlist {list_id}")
    return table


def flatten_parameters(params: Any, prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """``{"a": {"b": 1}}`` → ``("a.b", 1)``; non-dict leaves are yielded as-is."""
    if not isinstance(params, dict):
        return
    for key, value in params.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from flatten_parameters(value, dotted + ".")
        else:
            yield dotted, value


def _canon(value: Any) -> Any:
    """Canonical scalar form so ``7``, ``7.0`` and ``"7"`` hash alike."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        s = value.strip()
        low = s.lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return int(s)
        except ValueError:
            pass
        try:
            f = float(s)
        except ValueError:
            return s
        return int(f) if f.is_integer() else f
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    return value


def core_config_hash(mapping: Dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of ``{key: value}``, first 16 hex chars."""
    canonical = json.dumps({k: _canon(v) for k, v in mapping.items()},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _key_allowed(key: str, include: List[str], exclude: List[str]) -> bool:
    if key in exclude:
        return False
    for entry in include:
        if entry.endswith(".*"):
            if key.startswith(entry[:-1]):
                return True
        elif key == entry:
            return True
    return False


def compute_core_config(parameters: Any, family: str) -> Dict[str, Any]:
    """The ``core_config`` block for a run of ``family`` with these parameters.

    ``family`` is a ``BENCHMARK_TYPES`` value. Families without an allowlist
    (kv_cache, vector_database) get ``allowlist``/``hash`` = ``"unknown"``.
    """
    table = load_allowlists()
    list_id = table["families"].get(family)
    if list_id is None:
        return {"algorithm": CORE_CONFIG_ALGORITHM, "allowlist": UNKNOWN,
                "hash": UNKNOWN, "keys": []}
    spec = table["allowlists"][list_id]
    include = list(spec.get("include") or [])
    exclude = list(spec.get("exclude") or [])
    try:
        normalised = json.loads(json.dumps(parameters or {}, default=str))
    except (TypeError, ValueError):
        normalised = {}
    flat = dict(flatten_parameters(normalised))
    keys = sorted(k for k in flat if _key_allowed(k, include, exclude))
    if not keys:
        return {"algorithm": CORE_CONFIG_ALGORITHM, "allowlist": list_id,
                "hash": UNKNOWN, "keys": []}
    return {"algorithm": CORE_CONFIG_ALGORITHM, "allowlist": list_id,
            "hash": core_config_hash({k: flat[k] for k in keys}), "keys": keys}


# --- Runtime collection -----------------------------------------------------

def _read_direct_url(dist_name: str) -> Optional[dict]:
    """PEP 610 ``direct_url.json`` of an installed distribution, or None."""
    try:
        dist = _importlib_metadata.distribution(dist_name)
    except _importlib_metadata.PackageNotFoundError:
        return None
    try:
        text = dist.read_text("direct_url.json")
    except OSError:
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _package_version(dist_name: str) -> Optional[str]:
    try:
        return _importlib_metadata.version(dist_name)
    except _importlib_metadata.PackageNotFoundError:
        return None


def _dlio_runtime() -> Tuple[Dict[str, str], str]:
    version = _package_version(_DLIO_DIST)
    info = _read_direct_url(_DLIO_DIST)
    vcs = (info or {}).get("vcs_info") or {}
    commit = vcs.get("commit_id") if isinstance(vcs, dict) else None
    if info and isinstance(commit, str) and commit:
        return ({"version": version or UNKNOWN, "source": str(info.get("url") or UNKNOWN),
                 "commit": commit}, "direct-url")
    if version:
        return {"version": version, "source": UNKNOWN, "commit": UNKNOWN}, "package-metadata"
    return {"version": UNKNOWN, "source": UNKNOWN, "commit": UNKNOWN}, "unknown"


def _storage_library_runtime(data_access_protocol: Optional[str]) -> Tuple[Dict[str, str], str]:
    if data_access_protocol != "object":
        return {"name": "none"}, "n/a"
    version = _package_version(_S3DLIO_DIST)
    if version:
        return {"name": _S3DLIO_DIST, "version": version}, "package-metadata"
    return {"name": _S3DLIO_DIST, "version": UNKNOWN}, "unknown"


def _git_sha_runtime(log=None) -> str:
    try:
        from mlpstorage_py.submission_checker.tools.code_image import (
            _resolve_git_sha, find_source_root,
        )
        sha = _resolve_git_sha(find_source_root(), log or _NullLog())
    except Exception:  # noqa: BLE001 — provenance must never fail a run
        sha = None
    return sha or UNKNOWN


class _NullLog:
    def __getattr__(self, name):
        return lambda *a, **k: None


def collect_runtime_provenance(*, family: str, parameters: Any,
                               code_image_hash: Optional[str],
                               data_access_protocol: Optional[str],
                               log=None) -> RunProvenance:
    """The stamp for a run happening right now in this process."""
    tool = {
        "name": "mlpstorage",
        "version": VERSION,
        "git_sha": _git_sha_runtime(log),
        "code_image": (f"{_POINTER_ALGORITHM}:{code_image_hash}"
                       if code_image_hash else UNKNOWN),
    }
    dlio, dlio_tag = _dlio_runtime()
    storage_library, sl_tag = _storage_library_runtime(data_access_protocol)
    core_config = compute_core_config(parameters, family)
    return RunProvenance(
        rules_edition=RULES_EDITION,
        layout_version=MLPERF_RESULTS_VERSION,
        tool=tool,
        dlio=dlio,
        storage_library=storage_library,
        core_config=core_config,
        stamped_at=_now(),
        stamped_by=f"mlpstorage {VERSION}",
        provenance={
            "rules_edition": "tool-constant",
            "layout_version": "tool-constant",
            "tool": "runtime",
            "dlio": dlio_tag,
            "storage_library": sl_tag,
            "core_config": "runtime" if core_config["hash"] != UNKNOWN else "unknown",
        },
    )


# --- Leaf file I/O ----------------------------------------------------------

def write_leaf_provenance(leaf_dir, stamp: RunProvenance, log=None) -> Path:
    """Write ``<leaf>/provenance.json`` atomically (tmp sibling + rename)."""
    leaf = Path(leaf_dir)
    dst = leaf / PROVENANCE_FILENAME
    tmp = leaf / f".{PROVENANCE_FILENAME}.tmp.{os.getpid()}"
    payload = json.dumps(stamp.to_dict(), indent=2) + "\n"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if log is not None:
        log.debug("wrote %s", dst)
    return dst


def has_leaf_provenance(leaf_dir) -> bool:
    return (Path(leaf_dir) / PROVENANCE_FILENAME).is_file()


def _load_stamp_file(path: Path) -> RunProvenance:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        raise ProvenanceError(f"{path}: cannot read provenance stamp: {e}") from e
    try:
        return RunProvenance.from_dict(data)
    except ProvenanceError as e:
        raise ProvenanceError(f"{path}: {e}") from e


def read_leaf_provenance(leaf_dir, results_root=None, log=None) -> RunProvenance:
    """The leaf's stamp: its ``provenance.json`` when present, else a stamp
    derived from the code image and metadata (nothing is written).

    Raises ``ProvenanceError`` when a sidecar exists but is invalid.
    """
    leaf = Path(leaf_dir)
    path = leaf / PROVENANCE_FILENAME
    if path.is_file():
        return _load_stamp_file(path)
    return derive_leaf_provenance(leaf, results_root, log)


def pointer_hash(leaf_dir) -> Optional[str]:
    """The 32-hex hash from ``<leaf>/.mlps-code-image``, or None."""
    try:
        line = (Path(leaf_dir) / _POINTER_FILENAME).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    alg, _sep, hex_part = line.partition(":")
    if alg != _POINTER_ALGORITHM or not re.fullmatch(r"[0-9a-f]{32}", hex_part):
        return None
    return hex_part


def _locate_pool_image(leaf: Path, full_hash: str, results_root=None) -> Optional[Path]:
    """``code-<hash8>/`` in the tree-wide pool or a sentinelled per-org pool,
    searched from ``results_root`` (when given) and every ancestor of the leaf."""
    name = f"code-{full_hash[:8]}"
    roots: List[Path] = []
    if results_root is not None:
        # Bounded walk: the root first, then the leaf's ancestors below it.
        root = Path(results_root).resolve()
        roots.append(root)
        for parent in leaf.resolve().parents:
            if parent == root:
                break
            if root in parent.parents:
                roots.append(parent)
    else:
        roots.extend(leaf.resolve().parents)
    for root in roots:
        try:
            candidate = root / _GLOBAL_POOL_DIRNAME / name
            if candidate.is_dir():
                return candidate
            children = sorted(p for p in root.iterdir() if p.is_dir())
            for child in children:
                if (child / _POOL_SENTINEL_FILENAME).is_file() and (child / name).is_dir():
                    return child / name
        except OSError:
            continue  # unreadable ancestor (e.g. another user's /tmp entry)
    return None


def _read_hash_json(image: Path) -> Optional[dict]:
    try:
        with open(image / _HASH_FILENAME, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _uv_lock_package(text: str, name: str) -> Optional[Dict[str, str]]:
    """``{version, source, commit}`` for one ``[[package]]`` block of a uv.lock."""
    for block in text.split("[[package]]"):
        m = re.search(r'^name = "([^"]+)"', block, re.M)
        if not m or m.group(1) != name:
            continue
        v = re.search(r'^version = "([^"]+)"', block, re.M)
        version = v.group(1) if v else UNKNOWN
        git = re.search(r'^source = \{ git = "([^"]+)" \}', block, re.M)
        if git:
            url = git.group(1)
            base = url.split("?", 1)[0].split("#", 1)[0]
            commit = url.rsplit("#", 1)[1] if "#" in url else UNKNOWN
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                rev = re.search(r"[?&]rev=([0-9a-f]{40})", url)
                commit = rev.group(1) if rev else UNKNOWN
            return {"version": version, "source": base, "commit": commit}
        reg = re.search(r'^source = \{ registry = "([^"]+)" \}', block, re.M)
        return {"version": version, "source": reg.group(1) if reg else UNKNOWN,
                "commit": UNKNOWN}
    return None


def _read_leaf_metadata(leaf: Path) -> Optional[dict]:
    try:
        names = sorted(n for n in os.listdir(leaf) if n.endswith("_metadata.json"))
    except OSError:
        return None
    for n in names:
        try:
            with open(leaf / n, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _family_of(leaf: Path, metadata: Optional[dict]) -> Optional[str]:
    bt = (metadata or {}).get("benchmark_type")
    if isinstance(bt, str) and bt in _FAMILIES:
        return bt
    for part in reversed(leaf.parts):
        if part in _FAMILIES:
            return part
    return None


def derive_leaf_provenance(leaf_dir, results_root=None, log=None) -> RunProvenance:
    """An in-memory stamp for a leaf that has no ``provenance.json``."""
    leaf = Path(leaf_dir)
    hex_hash = pointer_hash(leaf)
    tool = {"name": "mlpstorage", "version": UNKNOWN, "git_sha": UNKNOWN,
            "code_image": f"{_POINTER_ALGORITHM}:{hex_hash}" if hex_hash else UNKNOWN}
    tool_tag = "unknown"
    dlio = {"version": UNKNOWN, "source": UNKNOWN, "commit": UNKNOWN}
    dlio_tag = "unknown"
    lock_text: Optional[str] = None
    if hex_hash:
        image = _locate_pool_image(leaf, hex_hash, results_root)
        if image is not None:
            hash_json = _read_hash_json(image)
            if hash_json:
                tool["version"] = str(hash_json.get("mlpstorage_version") or UNKNOWN)
                tool["git_sha"] = str(hash_json.get("git_sha") or UNKNOWN)
                tool_tag = "code-hash-json"
            lock = image / "uv.lock"
            if lock.is_file():
                try:
                    lock_text = lock.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    lock_text = None
            if lock_text:
                pkg = _uv_lock_package(lock_text, "dlio-benchmark")
                if pkg:
                    dlio, dlio_tag = pkg, "uv-lock"

    metadata = _read_leaf_metadata(leaf)
    args = (metadata or {}).get("args")
    protocol = args.get("data_access_protocol") if isinstance(args, dict) else None
    if metadata is None:
        storage_library, sl_tag = {"name": UNKNOWN, "version": UNKNOWN}, "unknown"
    elif protocol != "object":
        storage_library, sl_tag = {"name": "none"}, "n/a"
    else:
        pkg = _uv_lock_package(lock_text, _S3DLIO_DIST) if lock_text else None
        if pkg:
            storage_library, sl_tag = {"name": _S3DLIO_DIST, "version": pkg["version"]}, "uv-lock"
        else:
            storage_library, sl_tag = {"name": _S3DLIO_DIST, "version": UNKNOWN}, "unknown"

    family = _family_of(leaf, metadata)
    params = (metadata or {}).get("parameters")
    if family and isinstance(params, dict):
        core_config = compute_core_config(params, family)
    else:
        core_config = {"algorithm": CORE_CONFIG_ALGORITHM, "allowlist": UNKNOWN,
                       "hash": UNKNOWN, "keys": []}
    return RunProvenance(
        rules_edition=UNKNOWN,
        layout_version=2,
        tool=tool,
        dlio=dlio,
        storage_library=storage_library,
        core_config=core_config,
        stamped_at=_now(),
        stamped_by=f"mlpstorage {VERSION} (derived at read time)",
        provenance={
            "rules_edition": "unknown",
            "layout_version": "inferred",
            "tool": tool_tag,
            "dlio": dlio_tag,
            "storage_library": sl_tag,
            "core_config": "runtime" if core_config["hash"] != UNKNOWN else "unknown",
        },
        notes=(f"derived: the leaf has no {PROVENANCE_FILENAME} (written before "
               "stamping existed); nothing was written to the leaf"),
    )


# --- submission.yaml --------------------------------------------------------

def write_submission_manifest(results_dir, mode: str, orgname: str, log=None) -> Path:
    """Write ``<results-dir>/<mode>/<orgname>/submission.yaml`` from the tree."""
    from mlpstorage_py.runs.ledger import iter_leaves, parse_leaf, read_metadata

    rd = str(results_dir)
    org_dir = Path(rd) / mode / orgname
    org_prefix = f"{mode}/{orgname}/"
    leaf_prefix = org_prefix + "results/"
    runs: List[Dict[str, Any]] = []
    tool_versions, dlio_commits, code_images = set(), set(), set()
    for rel in iter_leaves(rd):
        if not rel.startswith(leaf_prefix):
            continue
        info = parse_leaf(rel) or {}
        leaf = Path(rd) / rel
        try:
            stamp = read_leaf_provenance(leaf, rd, log)
        except ProvenanceError as e:
            if log is not None:
                log.warning("reportgen: %s; describing the leaf from its metadata instead", e)
            stamp = derive_leaf_provenance(leaf, rd, log)
        metadata = read_metadata(str(leaf)) or {}
        accelerator = metadata.get("accelerator")
        rel_org = rel[len(org_prefix):]
        runs.append({
            "leaf": rel_org,
            "benchmark": info.get("benchmark", UNKNOWN),
            "model": info.get("model", UNKNOWN),
            "command": info.get("command", UNKNOWN),
            "accelerator": accelerator if isinstance(accelerator, str) and accelerator else UNKNOWN,
            "system": info.get("systemname", UNKNOWN),
            "rules_edition": stamp.rules_edition,
            "core_config": stamp.core_config["hash"],
            "code_image": stamp.tool["code_image"],
            "provenance": (f"{rel_org}/{PROVENANCE_FILENAME}"
                           if has_leaf_provenance(leaf) else "derived"),
        })
        if stamp.tool["version"] != UNKNOWN:
            tool_versions.add(stamp.tool["version"])
        if stamp.dlio["commit"] != UNKNOWN:
            dlio_commits.add(stamp.dlio["commit"])
        if stamp.tool["code_image"] != UNKNOWN:
            code_images.add(stamp.tool["code_image"])

    systems: List[Dict[str, str]] = []
    systems_dir = org_dir / "systems"
    if systems_dir.is_dir():
        for y in sorted(systems_dir.glob("*.yaml")):
            pdf = y.with_suffix(".pdf")
            systems.append({"name": y.stem, "description": f"systems/{y.name}",
                            "pdf": f"systems/{pdf.name}" if pdf.is_file() else "missing"})

    manifest: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "orgname": orgname,
        "mode": mode,
        "rules_edition": RULES_EDITION,
        "layout_version": MLPERF_RESULTS_VERSION,
        "generated_at": _now(),
        "generated_by": f"mlpstorage {VERSION}",
        "tool_versions": sorted(tool_versions),
        "dlio_commits": sorted(dlio_commits),
        "systems": systems,
        "runs": runs,
        "code_images": sorted(code_images),
        "provenance": {"rules_edition": "tool-constant", "runs": "walked"},
    }
    org_dir.mkdir(parents=True, exist_ok=True)
    dst = org_dir / MANIFEST_FILENAME
    tmp = org_dir / f".{MANIFEST_FILENAME}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            yaml.safe_dump(manifest, fh, sort_keys=False, default_flow_style=False,
                           allow_unicode=True)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if log is not None:
        log.info("wrote %s (%d run leaves, %d systems)", dst, len(runs), len(systems))
    return dst


def read_submission_manifest(path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as e:
        raise ProvenanceError(f"{path}: cannot read submission manifest: {e}") from e
    if not isinstance(data, dict) or data.get("schema") != MANIFEST_SCHEMA:
        raise ProvenanceError(
            f"{path}: unsupported submission manifest schema "
            f"{(data or {}).get('schema') if isinstance(data, dict) else None!r} "
            f"(expected {MANIFEST_SCHEMA!r})")
    return data


def write_manifests_if_results_dir(results_dir, log=None) -> List[Path]:
    """Write a manifest for every ``<mode>/<org>/`` in a live results-dir.

    A tree without the ``mlperf-results.yaml`` sentinel (a submissions or
    archive tree) is left alone — the importer is the only writer there.
    """
    rd = str(results_dir)
    if not os.path.isfile(os.path.join(rd, MLPERF_RESULTS_FILENAME)):
        return []
    written: List[Path] = []
    for mode in _MODES:
        mode_dir = os.path.join(rd, mode)
        if not os.path.isdir(mode_dir):
            continue
        for org in sorted(os.listdir(mode_dir)):
            if os.path.isdir(os.path.join(mode_dir, org, "results")):
                written.append(write_submission_manifest(rd, mode, org, log))
    return written
