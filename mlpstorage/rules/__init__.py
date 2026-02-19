"""
Rules engine for MLPerf Storage benchmark validation.

This package provides validation of benchmark parameters and results
against MLPerf Storage submission rules for OPEN and CLOSED divisions.

Package Structure:
    - base.py: RulesChecker ABC and RuleState enum
    - issues.py: Issue dataclass for validation findings
    - models.py: Data classes (HostInfo, ClusterInformation, BenchmarkRun, etc.)
    - run_checkers/: Single-run validation checkers
    - submission_checkers/: Multi-run submission validation checkers
    - verifier.py: BenchmarkVerifier orchestrator
    - utils.py: Utility functions

Usage:
    # Verify a single benchmark run
    from mlpstorage.rules import BenchmarkVerifier

    verifier = BenchmarkVerifier(benchmark_run, logger=logger)
    result = verifier.verify()  # Returns PARAM_VALIDATION.CLOSED, .OPEN, or .INVALID

    # Access validation issues
    for issue in verifier.issues:
        print(issue)
"""

# Base classes
from mlpstorage.rules.base import RulesChecker, RuleState

# Issue dataclass
from mlpstorage.rules.issues import Issue

# Data models
from mlpstorage.rules.models import (
    RunID,
    ProcessedRun,
    BenchmarkRunData,
    HostMemoryInfo,
    HostCPUInfo,
    HostInfo,
    ClusterInformation,
    BenchmarkResult,
    BenchmarkInstanceExtractor,
    DLIOResultParser,
    ResultFilesExtractor,
    BenchmarkRun,
)

# Run checkers
from mlpstorage.rules.run_checkers import (
    RunRulesChecker,
    TrainingRunRulesChecker,
    CheckpointingRunRulesChecker,
    KVCacheRunRulesChecker,
    VectorDBRunRulesChecker,
)

# Submission checkers
from mlpstorage.rules.submission_checkers import (
    MultiRunRulesChecker,
    TrainingSubmissionRulesChecker,
    CheckpointSubmissionRulesChecker,
)

# Verifier
from mlpstorage.rules.verifier import BenchmarkVerifier

# Utility functions
from mlpstorage.rules.utils import (
    calculate_training_data_size,
    generate_output_location,
    get_runs_files,
)

__all__ = [
    # Base
    'RulesChecker',
    'RuleState',
    # Issues
    'Issue',
    # Models
    'RunID',
    'ProcessedRun',
    'BenchmarkRunData',
    'HostMemoryInfo',
    'HostCPUInfo',
    'HostInfo',
    'ClusterInformation',
    'BenchmarkResult',
    'BenchmarkInstanceExtractor',
    'DLIOResultParser',
    'ResultFilesExtractor',
    'BenchmarkRun',
    # Run Checkers
    'RunRulesChecker',
    'TrainingRunRulesChecker',
    'CheckpointingRunRulesChecker',
    'KVCacheRunRulesChecker',
    'VectorDBRunRulesChecker',
    # Submission Checkers
    'MultiRunRulesChecker',
    'TrainingSubmissionRulesChecker',
    'CheckpointSubmissionRulesChecker',
    # Verifier
    'BenchmarkVerifier',
    # Utils
    'calculate_training_data_size',
    'generate_output_location',
    'get_runs_files',
]
