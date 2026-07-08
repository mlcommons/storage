"""Earliest wall-clock instant of the mlpstorage process.

Captured at first import of this module. ``main.py`` imports this before
any heavier ``mlpstorage_py`` submodule so the timestamp is recorded
before framework startup (Python imports, MPI spawn, CAP env-validation,
etc.). ``benchmarks/base.py`` reads the value when populating
``invocation_start_time`` in the run's metadata; the submission checker
uses that field as the read-side origin of the §4.7.1 gap check so
per-invocation framework overhead is not charged against the 30-second
failover-callout budget.
"""
import time

INVOCATION_START: float = time.time()
