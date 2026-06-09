# Phase 5: Run Configuration Summary — Pattern Map

**Mapped:** 2026-06-09
**Files analyzed:** 12 (4 new, 8 modified)
**Analogs found:** 12 / 12

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `mlpstorage_py/storage_config.py` | utility | transform | `mlpstorage_py/benchmarks/dlio.py` (`_apply_object_storage_params`) | role-match |
| `mlpstorage_py/run_summary.py` | utility | request-response | `mlpstorage_py/reporting/formatters.py` + `ban_boto3.py` | role-match |
| `tests/unit/test_storage_config.py` | test | — | `tests/unit/test_dlio_object_storage.py` | exact |
| `tests/unit/test_run_summary.py` | test | — | `tests/unit/test_main_warnings.py` | exact |
| `mlpstorage_py/cli/common_args.py` | config | — | self (existing `--debug`/`--verbose` pattern, lines 236–251) | exact |
| `mlpstorage_py/main.py` | utility | request-response | self (existing `update_args` insertion point, lines 316–335) | exact |
| `mlpstorage_py/benchmarks/dlio.py` | service | CRUD | self (existing `_apply_object_storage_params`, lines 180–187) | exact |
| `mlpstorage_py/checkpointing/storage_readers/minio_reader.py` | service | file-I/O | self (existing `_detect_endpoint`, lines 30–55) | exact |
| `mlpstorage_py/checkpointing/storage_readers/s3torch_reader.py` | service | file-I/O | self (existing `_detect_endpoint`, lines 52–73) | exact |
| `mlpstorage_py/checkpointing/storage_writers/minio_writer.py` | service | file-I/O | self (existing `_detect_and_select_endpoint`, lines 88–135) | exact |
| `mlpstorage_py/checkpointing/storage_writers/s3dlio_writer.py` | service | file-I/O | self (existing `_detect_multi_endpoint_config`, lines 127–175) | exact |
| `mlpstorage_py/checkpointing/storage_writers/s3torch_writer.py` | service | file-I/O | self (existing `_detect_and_select_endpoint`, lines 83–129) | exact |

---

## Pattern Assignments

### `mlpstorage_py/storage_config.py` (utility, transform)

**Analog:** `mlpstorage_py/benchmarks/dlio.py` (env-var reads at lines 180–187) and the four per-file `_detect_endpoint()` / `_detect_and_select_endpoint()` static methods

**Imports pattern** — copy from `dlio.py` lines 1–4 and existing storage reader/writer imports:
```python
import os
from typing import Optional, Tuple
```

**Core pattern** — the resolver owns ALL S3 env reads. The structure is taken from the scattered `_detect_endpoint()` implementations in `minio_reader.py` lines 30–55, `s3torch_reader.py` lines 52–73, `minio_writer.py` lines 88–135, and `s3dlio_writer.py` lines 127–175. All four implement the same `S3_ENDPOINT_URIS → S3_ENDPOINT_TEMPLATE → S3_ENDPOINT_FILE → AWS_ENDPOINT_URL → S3_ENDPOINT` fallback chain; the new file centralizes it:
```python
def resolve_object_storage_config() -> dict:
    def _redact(val: Optional[str]) -> str:
        if val:
            return f"[SET — {len(val)} chars]"
        return "[not set]"

    def _resolve_endpoint() -> Tuple[Optional[str], str]:
        for var in ('S3_ENDPOINT_URIS', 'S3_ENDPOINT_TEMPLATE', 'S3_ENDPOINT_FILE',
                    'AWS_ENDPOINT_URL', 'S3_ENDPOINT'):
            val = os.environ.get(var)
            if val:
                return val, var
        return None, ''

    endpoint_val, endpoint_src = _resolve_endpoint()

    return {
        'bucket':                         os.environ.get('BUCKET', ''),
        'storage_library':                os.environ.get('STORAGE_LIBRARY', 's3dlio'),
        'uri_scheme':                     os.environ.get('STORAGE_URI_SCHEME', 's3').rstrip(':/'),
        'endpoint':                       (endpoint_val, endpoint_src),
        'load_balance_strategy':          os.environ.get('S3_LOAD_BALANCE_STRATEGY', 'round_robin'),
        'aws_region':                     os.environ.get('AWS_REGION', 'us-east-1'),
        'aws_ca_bundle':                  os.environ.get('AWS_CA_BUNDLE'),
        'aws_access_key_id_redacted':     _redact(os.environ.get('AWS_ACCESS_KEY_ID')),
        'aws_secret_access_key_redacted': _redact(os.environ.get('AWS_SECRET_ACCESS_KEY')),
    }
```

**Key constraint:** `resolve_object_storage_config()` is called at runtime (inside `__init__` or in `run_summary.py`), never at module import time. The `.env` file is not yet loaded when `run_summary.py` calls it — this is by design (summary shows pre-load env state).

---

### `mlpstorage_py/run_summary.py` (utility, request-response)

**Analog:** `mlpstorage_py/reporting/formatters.py` (fixed-width labeled-output pattern, lines 1–100) and `mlpstorage_py/ban_boto3.py` (box-drawing f-string table, lines 36–47)

**Imports pattern** — no new packages; follow `main.py` logger import pattern (line 45):
```python
from mlpstorage_py.mlps_logging import setup_logging
from mlpstorage_py.storage_config import resolve_object_storage_config

logger = setup_logging("MLPerfStorage")
```

**Core pattern — table with fixed label column** (based on `ban_boto3.py` lines 36–47 f-string approach):
```python
def print_run_summary(args) -> None:
    # NOTE: .env file loading happens in _apply_object_storage_params(), which runs
    # after run_benchmark(). This summary shows pre-load env state — by design.
    _WIDTH = 28  # label column width

    def _row(label: str, value) -> str:
        return f"  {label:<{_WIDTH}}{value}"

    lines = ["", "--- Run Configuration ---"]
    lines.append(_row("benchmark:", getattr(args, 'benchmark', '—')))
    lines.append(_row("command:",   getattr(args, 'command', '—')))
    # ... one _row() per Tier 1 CLI arg ...

    if getattr(args, 'data_access_protocol', None) == 'object':
        config = resolve_object_storage_config()
        endpoint_val, endpoint_src = config['endpoint']
        lines.append("")
        lines.append("--- Object Storage (S3) ---")
        lines.append(_row("bucket:",        config['bucket'] or '[not set]'))
        lines.append(_row("storage_library:", config['storage_library']))
        lines.append(_row("uri_scheme:",    config['uri_scheme']))
        ep_display = f"{endpoint_val}  [from {endpoint_src}]" if endpoint_val else "[not set]"
        lines.append(_row("endpoint:",      ep_display))
        lines.append(_row("aws_region:",    config['aws_region']))
        lines.append(_row("aws_ca_bundle:", config['aws_ca_bundle'] or '[not set]'))
        lines.append(_row("AWS_ACCESS_KEY_ID:",     config['aws_access_key_id_redacted']))
        lines.append(_row("AWS_SECRET_ACCESS_KEY:", config['aws_secret_access_key_redacted']))

    lines.append("")
    for line in lines:
        logger.status(line)
```

**Guard pattern** — copy from `main.py` line 222 `getattr` guard style:
```python
# In main.py _main_impl(), after update_args(args) at line 319:
if not getattr(args, 'quiet', False):
    from mlpstorage_py.run_summary import print_run_summary
    print_run_summary(args)
```

**Log level:** Use `logger.status()` (level 25, blue) — matches `main.py` operational messages at lines 87, 99, 127, 178, 248.

---

### `tests/unit/test_storage_config.py` (test)

**Analog:** `tests/unit/test_dlio_object_storage.py` — exact match for env-var mocking with `monkeypatch`

**Imports pattern** (lines 1–21 of test_dlio_object_storage.py):
```python
import os
from argparse import Namespace
from unittest.mock import MagicMock, patch, call

import pytest

from mlpstorage_py.storage_config import resolve_object_storage_config
```

**Core pattern — env var mocking** (lines 163–185 of test_dlio_object_storage.py):
```python
class TestCredentialRedaction:
    def test_access_key_redacted_when_set(self, monkeypatch):
        monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'AKIAIOSFODNN7EXAMPLE')
        config = resolve_object_storage_config()
        assert 'AKIAIOSFODNN7EXAMPLE' not in str(config)
        assert '[SET —' in config['aws_access_key_id_redacted']

    def test_access_key_not_set(self, monkeypatch):
        monkeypatch.delenv('AWS_ACCESS_KEY_ID', raising=False)
        config = resolve_object_storage_config()
        assert config['aws_access_key_id_redacted'] == '[not set]'
```

**Endpoint resolution pattern** (mirrors `TestApplyObjectStorageParamsInjection._call_with_env` pattern at lines 163–185):
```python
class TestEndpointResolution:
    def test_s3_endpoint_uris_takes_priority(self, monkeypatch):
        monkeypatch.setenv('S3_ENDPOINT_URIS', 'http://minio1:9000,http://minio2:9000')
        monkeypatch.setenv('AWS_ENDPOINT_URL', 'http://fallback:9000')
        config = resolve_object_storage_config()
        val, src = config['endpoint']
        assert val == 'http://minio1:9000,http://minio2:9000'
        assert src == 'S3_ENDPOINT_URIS'

    def test_endpoint_returns_tuple(self, monkeypatch):
        monkeypatch.setenv('AWS_ENDPOINT_URL', 'http://s3.example.com')
        config = resolve_object_storage_config()
        assert isinstance(config['endpoint'], tuple)
        assert len(config['endpoint']) == 2
```

**Class structure** — use the same class-per-behavior grouping as `test_dlio_object_storage.py`:
- `TestCredentialRedaction` — covers RUNSUM-04
- `TestEndpointResolution` — covers RUNSUM-05
- `TestCentralizedResolver` — covers RUNSUM-06 (assert no `os.environ.get` for S3 vars in the 6 modified files)

---

### `tests/unit/test_run_summary.py` (test)

**Analog:** `tests/unit/test_main_warnings.py` — exact match for patching logger and asserting on output

**Imports pattern** (lines 1–20 of test_main_warnings.py):
```python
import os
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from mlpstorage_py.config import EXIT_CODE
```

**Helper pattern** (lines 26–43 of test_main_warnings.py — `_make_args` and `_mock_benchmark`):
```python
def _make_args(**kwargs):
    """Return a minimal Namespace for print_run_summary() testing."""
    return Namespace(
        benchmark='training',
        command='run',
        data_access_protocol='file',
        quiet=False,
        **kwargs,
    )
```

**Core pattern — patch logger and assert calls** (lines 52–68 of test_main_warnings.py):
```python
class TestPrintRunSummary:
    @patch('mlpstorage_py.run_summary.logger')
    def test_summary_prints_benchmark(self, mock_logger):
        from mlpstorage_py.run_summary import print_run_summary
        print_run_summary(_make_args())
        assert mock_logger.status.called

class TestQuietFlag:
    def test_quiet_suppresses_summary(self, monkeypatch):
        """--quiet flag prevents print_run_summary from being called."""
        # Test via _main_impl() integration or directly via args.quiet guard
        # in main.py — mirror test_main_warnings.py @patch('mlpstorage_py.main.logger') pattern
        pass

class TestProtocolFiltering:
    @patch('mlpstorage_py.run_summary.logger')
    def test_s3_section_absent_for_file_protocol(self, mock_logger, monkeypatch):
        from mlpstorage_py.run_summary import print_run_summary
        print_run_summary(_make_args(data_access_protocol='file'))
        all_output = ' '.join(str(c) for c in mock_logger.status.call_args_list)
        assert 'Object Storage' not in all_output

    @patch('mlpstorage_py.run_summary.logger')
    def test_s3_section_present_for_object_protocol(self, mock_logger, monkeypatch):
        monkeypatch.setenv('BUCKET', 'test-bucket')
        from mlpstorage_py.run_summary import print_run_summary
        print_run_summary(_make_args(data_access_protocol='object'))
        all_output = ' '.join(str(c) for c in mock_logger.status.call_args_list)
        assert 'Object Storage' in all_output or 'S3' in all_output
```

---

### `mlpstorage_py/cli/common_args.py` — add `--quiet` (config)

**Analog:** self — existing `--debug` and `--verbose` flags at lines 236–251

**Exact insertion point:** After `--stream-log-level` at line 251, inside the `output_control` argument group created at line 236:
```python
# common_args.py lines 236–251 (existing Output Control group):
output_control = parser.add_argument_group("Output Control")
output_control.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode"
)
output_control.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose mode"
)
output_control.add_argument(
    "--stream-log-level",
    type=str,
    default="INFO"
)
# ADD after --stream-log-level:
output_control.add_argument(
    "--quiet",
    action="store_true",
    help="Suppress run configuration summary table"
)
```

**Do NOT add** `quiet` to any `set_defaults()` call. Use `getattr(args, 'quiet', False)` in `main.py` for safety (handles modes like `reports` that bypass `add_universal_arguments()`).

---

### `mlpstorage_py/main.py` — wire `print_run_summary()` (utility)

**Analog:** self — existing `update_args(args)` dispatch at lines 316–335

**Exact insertion point:** Lines 316–322 of `main.py` (current `_main_impl()` tail):
```python
# Current (lines 316–322):
run_datetime = datetime_str
update_args(args)

# For other commands, run the benchmark
for i in range(args.loops):

# New (insert between update_args and the for loop):
run_datetime = datetime_str
update_args(args)

if not getattr(args, 'quiet', False):
    from mlpstorage_py.run_summary import print_run_summary
    print_run_summary(args)

for i in range(args.loops):
```

**Why lazy import:** Follows `main.py` line 155 pattern (`from mlpstorage_py.benchmarks import KVCacheBenchmark`) — deferred import keeps startup fast and avoids circular import risk.

---

### `mlpstorage_py/benchmarks/dlio.py` — replace 4 inline env reads (service)

**Analog:** self — lines 180–187 in `_apply_object_storage_params()`

**Current code** (lines 180–187):
```python
bucket = os.environ.get('BUCKET', '')
storage_library = os.environ.get('STORAGE_LIBRARY', 's3dlio')
endpoint_url = os.environ.get('AWS_ENDPOINT_URL', '')
uri_scheme = os.environ.get('STORAGE_URI_SCHEME', 's3').rstrip(':/')
```

**Replacement pattern:**
```python
from mlpstorage_py.storage_config import resolve_object_storage_config
config = resolve_object_storage_config()
bucket = config['bucket']
storage_library = config['storage_library']
endpoint_url, _src = config['endpoint']   # unpack tuple; only value needed here
endpoint_url = endpoint_url or ''         # keep existing empty-string semantics
uri_scheme = config['uri_scheme']
```

**Scope boundary:** Only replace the 4 reads listed above. The `OMPI_COMM_WORLD_RANK`, `PMI_RANK`, and `MLPERF_RESULTS_DIR` reads elsewhere in `dlio.py` are NOT moved.

---

### `mlpstorage_py/checkpointing/storage_readers/minio_reader.py` — replace `_detect_endpoint()` (service)

**Analog:** self — `_detect_endpoint()` static method at lines 30–55; `__init__` env reads at lines 75–114

**Current reads to replace:**
- Line 32: `os.environ.get('S3_ENDPOINT_URIS')`
- Line 38: `os.environ.get('S3_ENDPOINT_TEMPLATE')`
- Line 44: `os.environ.get('S3_ENDPOINT_FILE')`
- Line 75: `os.environ.get('AWS_ACCESS_KEY_ID')`
- Line 76: `os.environ.get('AWS_SECRET_ACCESS_KEY')`
- Line 80: `os.environ.get('AWS_ENDPOINT_URL')` / `os.environ.get('S3_ENDPOINT')`
- Line 96: `os.environ.get('AWS_CA_BUNDLE')`
- Line 114: `os.environ.get('AWS_REGION', 'us-east-1')`

**Replacement pattern in `__init__`:**
```python
from mlpstorage_py.storage_config import resolve_object_storage_config
config = resolve_object_storage_config()
# The resolver returns pre-redacted credentials — raw values are NOT available.
# For actual auth, still read raw from os.environ directly:
access_key = os.environ.get('AWS_ACCESS_KEY_ID')
secret_key  = os.environ.get('AWS_SECRET_ACCESS_KEY')
# Endpoint comes from resolver (tuple):
endpoint, _src = config['endpoint']
endpoint = endpoint or os.environ.get('AWS_ENDPOINT_URL') or os.environ.get('S3_ENDPOINT')
ca_bundle = config['aws_ca_bundle']
region    = config['aws_region']
```

**Note on credentials:** The resolver REDACTS credentials for display purposes. The actual raw credential strings for SDK initialization must still come from `os.environ.get('AWS_ACCESS_KEY_ID')` directly. The resolver's `aws_access_key_id_redacted` field is for display only. The `_detect_endpoint()` static method can be removed or replaced with a call to the centralized resolver.

---

### `mlpstorage_py/checkpointing/storage_readers/s3torch_reader.py` — replace `_detect_endpoint()` (service)

**Analog:** self — `_detect_endpoint()` at lines 52–73; `__init__` at lines 75–119

**Current reads to replace:**
- Lines 53, 58, 63: `S3_ENDPOINT_URIS`, `S3_ENDPOINT_TEMPLATE`, `S3_ENDPOINT_FILE`
- Line 96: `AWS_REGION`
- Lines 98–99: `AWS_ENDPOINT_URL` / `S3_ENDPOINT`

**Replacement pattern in `__init__`** (same tuple-unpack idiom):
```python
from mlpstorage_py.storage_config import resolve_object_storage_config
config = resolve_object_storage_config()
region   = config['aws_region']
endpoint, _src = config['endpoint']
endpoint = endpoint or None  # keep existing None-means-AWS-S3 semantics
```

The `_detect_endpoint()` static method (lines 52–73) becomes dead code and can be removed.

---

### `mlpstorage_py/checkpointing/storage_writers/minio_writer.py` — replace `_detect_and_select_endpoint()` (service)

**Analog:** self — `_detect_and_select_endpoint()` at lines 88–135; `__init__` env reads at lines 186–239

**Current reads to replace:**
- Lines 103, 109, 115: `S3_ENDPOINT_URIS`, `S3_ENDPOINT_TEMPLATE`, `S3_ENDPOINT_FILE`
- Line 186: `AWS_ACCESS_KEY_ID`
- Line 187: `AWS_SECRET_ACCESS_KEY`
- Line 193: `AWS_ENDPOINT_URL` / `S3_ENDPOINT`
- Line 221: `AWS_CA_BUNDLE`
- Line 239: `AWS_REGION`

**Replacement pattern in `__init__`:**
```python
from mlpstorage_py.storage_config import resolve_object_storage_config
config = resolve_object_storage_config()
# Raw credentials still needed for SDK — read directly:
access_key = os.environ.get('AWS_ACCESS_KEY_ID')
secret_key  = os.environ.get('AWS_SECRET_ACCESS_KEY')
endpoint, _src = config['endpoint']
if not endpoint:
    endpoint = None  # fall through to default AWS S3
ca_bundle = config['aws_ca_bundle']
region    = config['aws_region']
```

**MPI rank reads stay:** Lines 53 (`OMPI_COMM_WORLD_RANK`) and 61 (`PMI_RANK`) are MPI internal vars and must NOT be moved to the centralized resolver.

---

### `mlpstorage_py/checkpointing/storage_writers/s3dlio_writer.py` — replace `_detect_multi_endpoint_config()` (service)

**Analog:** self — `_detect_multi_endpoint_config()` at lines 127–175

**Current reads to replace:**
- Lines 140, 148, 156: `S3_ENDPOINT_URIS`, `S3_ENDPOINT_TEMPLATE`, `S3_ENDPOINT_FILE`
- Lines 234, 259: `S3_LOAD_BALANCE_STRATEGY`

**Special case — line 173 write stays:**
```python
os.environ['AWS_ENDPOINT_URL'] = selected  # line 173 — this WRITE must stay
```
This line sets `AWS_ENDPOINT_URL` for MPI worker subprocesses. The centralized resolver provides the resolved endpoint value; this writer then applies it to the environment. The write is NOT moved.

**Replacement pattern:**
```python
from mlpstorage_py.storage_config import resolve_object_storage_config
config = resolve_object_storage_config()
endpoint_val, _src = config['endpoint']
strategy = config['load_balance_strategy']
# Then use endpoint_val where the old _detect_multi_endpoint_config() result was used.
# The os.environ['AWS_ENDPOINT_URL'] = selected write at line 173 stays unchanged.
```

---

### `mlpstorage_py/checkpointing/storage_writers/s3torch_writer.py` — replace `_detect_and_select_endpoint()` (service)

**Analog:** self — `_detect_and_select_endpoint()` at lines 83–129; `__init__` at lines 131–198

**Current reads to replace:**
- Lines 97, 103, 109: `S3_ENDPOINT_URIS`, `S3_ENDPOINT_TEMPLATE`, `S3_ENDPOINT_FILE`
- Line 170: `AWS_REGION`
- Lines 176: `AWS_ENDPOINT_URL` / `S3_ENDPOINT`

**Replacement pattern** (same as `s3torch_reader.py`):
```python
from mlpstorage_py.storage_config import resolve_object_storage_config
config = resolve_object_storage_config()
region   = config['aws_region']
endpoint, _src = config['endpoint']
endpoint = endpoint or None
```

**MPI rank reads stay:** Lines 47 (`OMPI_COMM_WORLD_RANK`) and 55 (`PMI_RANK`) are NOT moved.

---

## Shared Patterns

### `os.environ` access style
**Source:** All 6 storage files + `dlio.py`
**Apply to:** `storage_config.py` resolver
All env reads use `os.environ.get(VAR, default)` — never `os.environ[VAR]` (which raises `KeyError`). The one exception is the `os.environ['AWS_ENDPOINT_URL'] = selected` write in `s3dlio_writer.py` line 173, which must stay.

### `logger.status()` for operational output
**Source:** `mlpstorage_py/main.py` lines 87, 99, 127, 178, 248; `mlpstorage_py/benchmarks/base.py` line 149
**Apply to:** `run_summary.py` summary lines
`logger.status()` is level 25 (blue), used for pre-execution operational messages. It is suppressed by `--stream-log-level WARNING` and above — consistent with the intended `--quiet` semantics.

### `getattr(args, 'attr', default)` guard
**Source:** `mlpstorage_py/main.py` line 222 (`getattr(args, 'results_dir', DEFAULT_RESULTS_DIR)`); `dlio.py` lines 56–58 (`getattr(args, 'exec_type', None)`)
**Apply to:** `main.py` `--quiet` check and `run_summary.py` attribute reads
Use `getattr(args, 'quiet', False)` instead of `args.quiet` — handles `reports`, `history`, and `lockfile` modes where `add_universal_arguments()` was not called.

### f-string fixed-width label column
**Source:** `mlpstorage_py/ban_boto3.py` lines 36–47 (`{fullname!r:<44}` format)
**Apply to:** `run_summary.py` `_row()` helper
```python
_WIDTH = 28
def _row(label: str, value) -> str:
    return f"  {label:<{_WIDTH}}{value}"
```

### Lazy import inside function body
**Source:** `mlpstorage_py/main.py` line 155 (`from mlpstorage_py.benchmarks import KVCacheBenchmark` inside `run_benchmark()`)
**Apply to:** `main.py` insertion of `print_run_summary` call
Import `run_summary` lazily inside the `if not getattr(args, 'quiet', False):` block, not at module top level.

### `monkeypatch.setenv` / `monkeypatch.delenv` for env tests
**Source:** `tests/unit/test_dlio_object_storage.py` lines 76–79, 147–148
**Apply to:** `test_storage_config.py` and `test_run_summary.py`
Always pair `setenv` with corresponding `delenv` in a helper to prevent cross-test contamination.

### `@patch('module.path.logger')` for logger assertion
**Source:** `tests/unit/test_main_warnings.py` lines 52–53
**Apply to:** `test_run_summary.py`
```python
@patch('mlpstorage_py.run_summary.logger')
def test_summary_prints_benchmark(self, mock_logger):
    ...
    assert mock_logger.status.called
```

---

## No Analog Found

All files have close analogs in the codebase. No entries.

---

## Metadata

**Analog search scope:** `mlpstorage_py/`, `tests/unit/`
**Files scanned:** 12 source files read directly; 6 test files referenced
**Key constraint confirmed:** Package lives at `mlpstorage_py/` (not `mlpstorage/`); import paths use `mlpstorage_py.*`
**Pattern extraction date:** 2026-06-09
