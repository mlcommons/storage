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
    from mlpstorage.reporting import (
        ResultsDirectoryValidator,
        ValidationMessageFormatter,
        ClosedRequirementsFormatter,
        ReportSummaryFormatter,
    )
"""

from mlpstorage.reporting.directory_validator import (
    ResultsDirectoryValidator,
    DirectoryValidationError,
    DirectoryValidationResult,
)

from mlpstorage.reporting.formatters import (
    ValidationMessageFormatter,
    ClosedRequirementsFormatter,
    ReportSummaryFormatter,
)

__all__ = [
    # Directory validation
    'ResultsDirectoryValidator',
    'DirectoryValidationError',
    'DirectoryValidationResult',
    # Formatters
    'ValidationMessageFormatter',
    'ClosedRequirementsFormatter',
    'ReportSummaryFormatter',
]
