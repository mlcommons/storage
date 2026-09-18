"""
``mlpstorage runs`` — management of the run leaves inside a results-dir.

- :mod:`mlpstorage_py.runs.ledger` owns ``<results-dir>/.mlps/runs.jsonl``,
  the append-only event log that gives every canonical run leaf a small,
  stable integer ID, plus leaf discovery and status derivation.
- :mod:`mlpstorage_py.runs.manage` implements the ``list``, ``show``,
  ``rm``, ``purge`` and ``gc`` subcommands on top of it.

Both modules are dependency-light (stdlib + ``mlpstorage_py.config``) so
the ``Benchmark`` base class can register a leaf at reservation time
without pulling anything heavy into the run path.
"""
