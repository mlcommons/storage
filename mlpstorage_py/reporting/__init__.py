"""
Reporting system for MLPerf Storage benchmark results.

This package provides:
- Directory structure validation
- Validation message formatting
- OPEN vs CLOSED submission messaging
- Report generation utilities

Modules:
    - directory_validator: Validate results directory structure
    - formatters: Format validation messages for display

Usage:
    from mlpstorage_py.reporting import (
        ResultsDirectoryValidator,
        ValidationMessageFormatter,
        ClosedRequirementsFormatter,
        ReportSummaryFormatter,
    )
"""

from mlpstorage_py.reporting.directory_validator import (
    ResultsDirectoryValidator,
    DirectoryValidationError,
    DirectoryValidationResult,
    discover_scan_roots,
)

from mlpstorage_py.reporting.formatters import (
    ValidationMessageFormatter,
    ClosedRequirementsFormatter,
    ReportSummaryFormatter,
)

__all__ = [
    # Directory validation
    'ResultsDirectoryValidator',
    'DirectoryValidationError',
    'DirectoryValidationResult',
    'discover_scan_roots',
    # Formatters
    'ValidationMessageFormatter',
    'ClosedRequirementsFormatter',
    'ReportSummaryFormatter',
]
