---
phase: 06-ssh-based-host-collection
plan: 02
subsystem: cluster-collection
tags:
  - ssh-collection
  - parallel-execution
  - cluster-collector
  - interface-implementation
requires:
  - 06-01 (proc parsers)
provides:
  - ssh-cluster-collector
  - localhost-detection
  - parallel-ssh-collection
affects:
  - Non-MPI benchmarks (KV Cache, VectorDB) can now collect cluster info
  - HOST-02 requirement fulfillment
tech-stack:
  added: []
  patterns:
    - ThreadPoolExecutor for parallel SSH collection
    - Interface implementation pattern (ClusterCollectorInterface)
    - Localhost optimization to avoid unnecessary SSH
    - BatchMode SSH for non-interactive automation
decisions:
  - id: localhost-skip-ssh
    choice: Use direct local collection for localhost/127.0.0.1/::1
    rationale: Avoids SSH overhead and configuration requirements
  - id: parallel-ssh
    choice: Use ThreadPoolExecutor for parallel host collection
    rationale: Faster collection from multiple hosts, configurable max_workers
  - id: batch-mode-ssh
    choice: Use SSH BatchMode=yes for non-interactive execution
    rationale: Prevents password prompts in automated collection
key-files:
  created: []
  modified:
    - mlpstorage/cluster_collector.py
    - tests/unit/test_cluster_collector.py
metrics:
  duration: ~3 minutes
  completed: 2026-01-24
---

# Phase 06 Plan 02: SSH-Based Host Collection Summary

**One-liner:** SSHClusterCollector implementing ClusterCollectorInterface with parallel SSH execution, localhost optimization, and comprehensive error handling

## What Was Built

1. **_is_localhost() Helper**: Function to detect localhost variants (localhost, 127.0.0.1, ::1, local hostname, local FQDN)

2. **SSH_COLLECTOR_SCRIPT**: Embedded Python script for remote execution that collects /proc files and returns JSON

3. **SSHClusterCollector Class**: Full implementation of ClusterCollectorInterface with:
   - Parallel SSH collection via ThreadPoolExecutor
   - Localhost optimization (direct local collection)
   - SSH BatchMode for non-interactive execution
   - Configurable timeout and max_workers
   - Raw data parsing using existing /proc parsers

4. **Comprehensive Unit Tests**: 33 new tests covering localhost detection and SSHClusterCollector functionality

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Implement SSHClusterCollector class | 4515dbe | mlpstorage/cluster_collector.py |
| 2 | Add unit tests for SSHClusterCollector | 03daa94 | tests/unit/test_cluster_collector.py |

## Technical Details

### Localhost Detection

```python
LOCALHOST_IDENTIFIERS = ('localhost', '127.0.0.1', '::1')

def _is_localhost(hostname: str) -> bool:
    """Check if hostname refers to local machine."""
    hostname_lower = hostname.lower()
    if hostname_lower in LOCALHOST_IDENTIFIERS:
        return True
    try:
        local_hostname = socket.gethostname()
        if hostname_lower == local_hostname.lower():
            return True
        local_fqdn = socket.getfqdn()
        if hostname_lower == local_fqdn.lower():
            return True
    except Exception:
        pass
    return False
```

### SSHClusterCollector Interface

```python
class SSHClusterCollector(ClusterCollectorInterface):
    def __init__(
        self,
        hosts: List[str],
        logger,
        ssh_username: Optional[str] = None,
        timeout_seconds: int = 60,
        max_workers: int = 10
    ): ...

    def collect(self, hosts: List[str], timeout: int = 60) -> CollectionResult: ...
    def collect_local(self) -> CollectionResult: ...
    def is_available(self) -> bool: ...
    def get_collection_method(self) -> str: ...
```

### SSH Command Options

The collector builds SSH commands with:
- `BatchMode=yes`: Prevents password prompts
- `ConnectTimeout={timeout}`: Configurable connection timeout
- `StrictHostKeyChecking=accept-new`: Accepts new host keys automatically
- Optional `-l {username}`: SSH username override

## Verification Results

All verification criteria met:

1. **SSHClusterCollector imports correctly**:
   ```python
   from mlpstorage.cluster_collector import SSHClusterCollector
   # Success
   ```

2. **SSHClusterCollector implements ClusterCollectorInterface**:
   ```python
   assert issubclass(SSHClusterCollector, ClusterCollectorInterface)
   # True
   ```

3. **All unit tests pass**: 62 tests passed (29 existing + 33 new)

4. **Localhost detection works**:
   ```python
   assert _is_localhost('localhost') is True
   assert _is_localhost('127.0.0.1') is True
   assert _is_localhost('::1') is True
   assert _is_localhost('remote-host') is False
   ```

### Must-Haves Verification

**Truths:**
- SSHClusterCollector implements ClusterCollectorInterface: VERIFIED
- SSHClusterCollector collects from remote hosts via SSH: VERIFIED (subprocess.run with ssh command)
- SSHClusterCollector uses direct local collection for localhost: VERIFIED (_is_localhost check)
- SSHClusterCollector collects hosts in parallel using ThreadPoolExecutor: VERIFIED

**Artifacts:**
- mlpstorage/cluster_collector.py provides "SSHClusterCollector class": VERIFIED
- tests/unit/test_cluster_collector.py provides "Unit tests for SSHClusterCollector": VERIFIED (TestSSHClusterCollector class)

**Key Links:**
- SSHClusterCollector inherits from ClusterCollectorInterface: VERIFIED
- _collect_from_single_host uses subprocess.run with ssh: VERIFIED

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

**Decision 1: Localhost skip for SSH**
- **Context:** Need to collect from localhost in hosts list
- **Choice:** Use direct local collection for localhost/127.0.0.1/::1
- **Rationale:** Avoids SSH overhead, works without SSH key setup, faster execution
- **Impact:** Localhost collection is always instant and reliable

**Decision 2: Parallel SSH collection**
- **Context:** Multiple remote hosts need collection
- **Choice:** ThreadPoolExecutor with configurable max_workers (default 10)
- **Rationale:** Parallel execution scales well, controllable resource usage
- **Impact:** Collection time scales with slowest host, not sum of all hosts

**Decision 3: BatchMode SSH**
- **Context:** Automated collection must not prompt for passwords
- **Choice:** Use SSH BatchMode=yes option
- **Rationale:** Ensures non-interactive execution, fails fast if auth needed
- **Impact:** Requires SSH key-based authentication to be configured

## Integration Points

**Upstream Dependencies:**
- 06-01 parsers: parse_proc_vmstat, parse_proc_mounts, parse_proc_cgroups
- interfaces/collector.py: ClusterCollectorInterface, CollectionResult

**Downstream Consumers:**
- KV Cache benchmark can use SSHClusterCollector for cluster info
- VectorDB benchmark can use SSHClusterCollector for cluster info
- Any non-MPI benchmark needing distributed system information

## Files Changed

### Modified
- `mlpstorage/cluster_collector.py`
  - Added imports: shutil, concurrent.futures.ThreadPoolExecutor, as_completed
  - Added LOCALHOST_IDENTIFIERS constant
  - Added _is_localhost() helper function
  - Added SSH_COLLECTOR_SCRIPT constant
  - Added SSHClusterCollector class (~200 lines)

- `tests/unit/test_cluster_collector.py`
  - Added TestIsLocalhost class (7 tests)
  - Added TestSSHClusterCollector class (26 tests)

## Testing Notes

Test execution results:
```
tests/unit/test_cluster_collector.py::TestIsLocalhost::test_localhost_string PASSED
tests/unit/test_cluster_collector.py::TestIsLocalhost::test_localhost_ipv4 PASSED
tests/unit/test_cluster_collector.py::TestIsLocalhost::test_localhost_ipv6 PASSED
tests/unit/test_cluster_collector.py::TestIsLocalhost::test_localhost_case_insensitive PASSED
tests/unit/test_cluster_collector.py::TestIsLocalhost::test_remote_host PASSED
tests/unit/test_cluster_collector.py::TestIsLocalhost::test_matches_local_hostname PASSED
tests/unit/test_cluster_collector.py::TestIsLocalhost::test_matches_local_fqdn PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_get_unique_hosts PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_get_unique_hosts_removes_duplicates PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_get_unique_hosts_handles_empty_strings PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_build_ssh_command_basic PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_build_ssh_command_with_username PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_build_ssh_command_has_connect_timeout PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_build_ssh_command_has_strict_host_key PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_is_available_with_ssh PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_is_available_without_ssh PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_get_collection_method PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collect_local PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collect_from_localhost_uses_direct_collection PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collect_from_127_uses_direct_collection PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collect_from_remote_host PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collect_parses_meminfo PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collect_handles_ssh_failure PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collect_handles_ssh_timeout PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collect_handles_json_parse_error PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collect_handles_generic_exception PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collect_parallel_execution PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collect_returns_success_when_all_succeed PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collect_returns_success_with_partial_failure PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collect_returns_error_list PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collect_local_returns_collection_result PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collector_init_defaults PASSED
tests/unit/test_cluster_collector.py::TestSSHClusterCollector::test_collector_init_custom_values PASSED
```

All 62 tests pass (29 existing + 33 new) with no failures or warnings.

## Lessons Learned

**What Went Well:**
- ClusterCollectorInterface provided clear contract for implementation
- Existing parsers from 06-01 were easily reused for SSH-collected data
- Mock-based testing allowed comprehensive coverage without actual SSH

**For Future Plans:**
- 06-03 will integrate SSHClusterCollector into benchmark wiring
- May need to add retry logic for transient SSH failures
- Consider adding SSH connection pooling for very large clusters

## Performance Notes

Execution time: ~3 minutes

Tasks: 2 completed in 2 commits

Commits:
- 4515dbe: feat(06-02): implement SSHClusterCollector class
- 03daa94: test(06-02): add unit tests for SSHClusterCollector

---

**Summary created:** 2026-01-24
**Executor:** Claude Opus 4.5
**Status:** Complete
