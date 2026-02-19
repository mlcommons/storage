# Phase 02: Environment Validation and Fail-Fast - User Acceptance Testing

**Phase Goal:** Users receive clear, actionable guidance when environment is misconfigured.

**Session Started:** 2026-01-24
**Status:** In Progress

---

## Test Results

| # | Test | Status | Notes |
|---|------|--------|-------|
| 1 | OS detection returns correct system info | PASS | |
| 2 | Install hints are OS-specific | PASS | |
| 3 | MPI missing error shows actionable install command | - | |
| 4 | DLIO missing error shows pip install suggestion | - | |
| 5 | SSH missing error shows OS-specific install command | - | |
| 6 | SSH connectivity check skips localhost | - | |
| 7 | Multiple validation errors shown together | - | |
| 8 | Validation runs before benchmark execution | - | |
| 9 | --skip-validation flag works | - | |

---

## Test Details

### Test 1: OS detection returns correct system info

**Expected:** Running `python -c "from mlpstorage.environment import detect_os; info = detect_os(); print(f'System: {info.system}, Distro: {info.distro_id}')"` shows your current OS and distro (e.g., "System: Linux, Distro: ubuntu").

**Result:**

---

### Test 2: Install hints are OS-specific

**Expected:** Running install hint lookup returns commands appropriate for your OS (apt-get for Ubuntu, dnf for RHEL, brew for macOS).

**Result:**

---

### Test 3: MPI missing error shows actionable install command

**Expected:** When MPI is not available, error message includes OS-specific install command that can be copy-pasted.

**Result:**

---

### Test 4: DLIO missing error shows pip install suggestion

**Expected:** When DLIO is not available, error message includes `pip install` command.

**Result:**

---

### Test 5: SSH missing error shows OS-specific install command

**Expected:** When SSH client is not found, error message includes OS-specific install command.

**Result:**

---

### Test 6: SSH connectivity check skips localhost

**Expected:** SSH validation automatically succeeds for localhost without making actual SSH connection.

**Result:**

---

### Test 7: Multiple validation errors shown together

**Expected:** When multiple issues exist, ALL issues are reported at once (not one at a time).

**Result:**

---

### Test 8: Validation runs before benchmark execution

**Expected:** Running a benchmark with validation issues shows errors BEFORE any benchmark work starts.

**Result:**

---

### Test 9: --skip-validation flag works

**Expected:** The `--skip-validation` flag appears in CLI help and bypasses environment validation when used.

**Result:**

---

## Summary

**Passed:** 2/9
**Failed:** 0/9
**Pending:** 7/9 (UAT interrupted - user moved to Phase 3)

---

*Last updated: 2026-01-24*
