"""
Detect whether systemd-logind will reap POSIX IPC objects out from under the
benchmark, which causes two independent shm-reap failures: the SemLock
FileNotFoundError reported in #447 and the PyTorch tensor-storage
"could not unlink the shared memory file /torch_*" RuntimeError reported
in #528.

systemd-logind defaults to `RemoveIPC=yes`, which removes /dev/shm/sem.*,
/dev/shm/torch_*, SysV semaphores, and shared memory belonging to a user
once that user has no remaining login sessions. mpirun launches that
briefly detach from the controlling session can hit this window and crash
in one of two ways:
  - `FileNotFoundError: [Errno 2]` inside multiprocessing.SemLock._rebuild
    (#447 — mitigated by storage PR #460's multiprocessing_context=fork)
  - `RuntimeError: could not unlink the shared memory file /torch_*`
    inside a PyTorch DataLoader worker (#528 — INDEPENDENT of fork vs
    spawn; the multiprocessing_context fix does not cover it)

`loginctl enable-linger USER` keeps a user-level systemd manager alive
regardless of login sessions, which suppresses the reap and covers both
vectors. For users without privilege to enable linger, the #528 vector
can additionally be sidestepped by setting
`DLIO_TORCH_SHARING_STRATEGY=file_descriptor` before launching (switches
PyTorch IPC to FD-passing so no /dev/shm/torch_* files are created); the
#447 vector requires sysadmin escalation if linger is unavailable.

This module only inspects host state — it never modifies it. The intended
caller is `validate_benchmark_environment`, which logs the warning string
returned by `check_removeipc_risk`.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from typing import Optional, Sequence

_LOGIND_CONF = "/etc/systemd/logind.conf"
_LOGIND_CONF_D_GLOB = "/etc/systemd/logind.conf.d/*.conf"

# systemd ships with RemoveIPC=yes as the default. Only an explicit `no` flips it.
_REMOVE_IPC_RE = re.compile(r"^\s*RemoveIPC\s*=\s*(\S+)", re.IGNORECASE | re.MULTILINE)


def _is_systemd_active() -> bool:
    return os.path.isdir("/run/systemd/system")


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def read_logind_remove_ipc(
    main_conf: str = _LOGIND_CONF,
    drop_in_glob: str = _LOGIND_CONF_D_GLOB,
) -> bool:
    """Return True if systemd-logind will reap user IPC (RemoveIPC=yes).

    Drop-ins in /etc/systemd/logind.conf.d/*.conf override the main file in
    lexical order, matching systemd's own precedence. If no file mentions
    RemoveIPC, the systemd default (yes) is returned.
    """
    value: Optional[str] = None

    main = _read_text(main_conf)
    if main is not None:
        for match in _REMOVE_IPC_RE.finditer(main):
            value = match.group(1)

    for drop_in in sorted(glob.glob(drop_in_glob)):
        contents = _read_text(drop_in)
        if contents is None:
            continue
        for match in _REMOVE_IPC_RE.finditer(contents):
            value = match.group(1)

    if value is None:
        return True
    return value.strip().lower() in ("yes", "true", "on", "1")


def is_user_lingering(
    username: Optional[str] = None,
    loginctl_path: Optional[str] = None,
) -> Optional[bool]:
    """Return True if `loginctl show-user --property=Linger USER` reports yes.

    Returns None when loginctl is unavailable or the call fails — callers
    should treat that as "unknown" rather than "not lingering."
    """
    if loginctl_path is None:
        loginctl_path = shutil.which("loginctl")
    if not loginctl_path:
        return None

    if username is None:
        username = os.environ.get("USER") or os.environ.get("LOGNAME")
    if not username:
        return None

    try:
        result = subprocess.run(
            [loginctl_path, "show-user", "--property=Linger", username],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip().lower() == "linger":
            return val.strip().lower() == "yes"
    return None


def check_removeipc_risk(
    username: Optional[str] = None,
    main_conf: str = _LOGIND_CONF,
    drop_in_glob: str = _LOGIND_CONF_D_GLOB,
    loginctl_path: Optional[str] = None,
    systemd_active: Optional[bool] = None,
) -> Optional[str]:
    """Return a remediation message if the host is at risk for #447, else None.

    Risk = systemd active AND RemoveIPC=yes AND user is not lingering.

    Returns a single multi-line string the caller can pass straight to a
    logger.warning(). Returns None on any "not risky" or "unknown" branch
    so this stays a soft advisory.
    """
    active = _is_systemd_active() if systemd_active is None else systemd_active
    if not active:
        return None

    if not read_logind_remove_ipc(main_conf=main_conf, drop_in_glob=drop_in_glob):
        return None

    lingering = is_user_lingering(username=username, loginctl_path=loginctl_path)
    if lingering:
        return None

    user_for_msg = (
        username
        or os.environ.get("USER")
        or os.environ.get("LOGNAME")
        or "$USER"
    )
    return (
        "systemd-logind has RemoveIPC=yes (the distro default). When a "
        "benchmark spawns Python multiprocessing workers, the kernel can "
        "reap two independent classes of user-owned /dev/shm files out "
        "from under live ranks:\n"
        "  - /dev/shm/sem.mp-*  -> FileNotFoundError in "
        "multiprocessing/synchronize.py (issue #447)\n"
        "  - /dev/shm/torch_*   -> 'could not unlink the shared memory "
        "file' RuntimeError in PyTorch DataLoader workers (issue #528)\n"
        "\n"
        "PRIMARY FIX (covers BOTH vectors; persistent):\n"
        f"  sudo loginctl enable-linger {user_for_msg}\n"
        "  loginctl show-user "
        f"{user_for_msg} --property=Linger   # should report Linger=yes\n"
        "\n"
        "FALLBACK if you cannot enable linger (HPC cluster, container, "
        "hosted env), to be applied IN ADDITION TO any existing #447 fix:\n"
        "  - Set DLIO_TORCH_SHARING_STRATEGY=file_descriptor before "
        "launching (switches PyTorch IPC away from named shm so #528's "
        "vector has nothing to reap).\n"
        "  - Raise FD limits before launching mpirun "
        "(`ulimit -n 65536`); file_descriptor strategy opens one FD per "
        "shared tensor and the default ulimit -n=1024 is too low.\n"
        "  - Or sysadmin-side: set `RemoveIPC=no` in "
        "/etc/systemd/logind.conf."
    )
