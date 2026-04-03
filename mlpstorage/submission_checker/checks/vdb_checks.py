from .base import BaseCheck
from ..constants import *
from ..configuration.configuration import Config
from ..loader import SubmissionLogs

import os


class VDBCheck(BaseCheck):
    """
    A check class for validating VDB (Vector Database Benchmark) parameters and related properties.
    Inherits from BaseCheck and receives a config and loader instance.
    """

    def __init__(self, log, config: Config, submissions_logs: SubmissionLogs):
        """
        Initialize VDBChecks with configuration and loader.

        Args:
            config: A Config instance containing submission configuration.
            loader: A SubmissionLogs instance for accessing submission logs.
        """
        # Call parent constructor with the loader's log and submission path
        super().__init__(log=log, path=submissions_logs.loader_metadata.folder)
        self.config = config
        self.submissions_logs = submissions_logs
        self.mode = submissions_logs.loader_metadata.mode
        self.model = submissions_logs.loader_metadata.benchmark
        self.name = "vdb checks"
        self.init_checks()

    def init_checks(self):
        """Initialize the list of checks to run."""
        self.checks = [
            self.vdb_sizing_options,
            self.vdb_generation_options,
            self.vdb_run_options,
            self.vdb_access_via_posix_api_options,
            self.vdb_access_via_object_api_options,
            self.vdb_open_vs_closed_options,
        ]

    def vdb_sizing_options(self):
        """
        Validate VDB sizing options.
        """
        valid = True
        if self.mode != "vectordb":
            return valid

        # TODO: Implement VDB sizing validation rules
        # Add specific checks based on section 5.1 rules

        return valid

    def vdb_generation_options(self):
        """
        Validate VDB generation options.
        """
        valid = True
        if self.mode != "vectordb":
            return valid

        # TODO: Implement VDB generation validation rules
        # Add specific checks based on section 5.2 rules

        return valid

    def vdb_run_options(self):
        """
        Validate VDB run options.
        """
        valid = True
        if self.mode != "vectordb":
            return valid

        # TODO: Implement VDB run validation rules
        # Add specific checks based on section 5.3 rules

        return valid

    def vdb_access_via_posix_api_options(self):
        """
        Validate VDB access via POSIX API options.
        """
        valid = True
        if self.mode != "vectordb":
            return valid

        # TODO: Implement VDB POSIX API validation rules
        # Add specific checks based on section 5.4 rules

        return valid

    def vdb_access_via_object_api_options(self):
        """
        Validate VDB access via Object API options.
        """
        valid = True
        if self.mode != "vectordb":
            return valid

        # TODO: Implement VDB Object API validation rules
        # Add specific checks based on section 5.5 rules

        return valid

    def vdb_open_vs_closed_options(self):
        """
        Validate VDB OPEN versus CLOSED options.
        """
        valid = True
        if self.mode != "vectordb":
            return valid

        # TODO: Implement VDB OPEN/CLOSED validation rules
        # Add specific checks based on section 5.6 rules

        return valid