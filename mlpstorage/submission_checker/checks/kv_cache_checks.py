from .base import BaseCheck
from ..constants import *
from ..configuration.configuration import Config
from ..loader import SubmissionLogs

import os


class KVCacheCheck(BaseCheck):
    """
    A check class for validating KVCache (KV Cache Benchmark) parameters and related properties.
    Inherits from BaseCheck and receives a config and loader instance.
    """

    def __init__(self, log, config: Config, submissions_logs: SubmissionLogs):
        """
        Initialize KVCacheChecks with configuration and loader.

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
        self.name = "kv_cache checks"
        self.init_checks()

    def init_checks(self):
        """Initialize the list of checks to run."""
        self.checks = [
            self.kv_cache_sizing_options,
            self.kv_cache_generation_options,
            self.kv_cache_run_options,
            self.kv_cache_access_via_posix_api_options,
            self.kv_cache_access_via_object_api_options,
            self.kv_cache_open_vs_closed_options,
        ]

    def kv_cache_sizing_options(self):
        """
        Validate KVCache sizing options.
        """
        valid = True
        if self.mode != "kvcache":
            return valid

        # TODO: Implement KVCache sizing validation rules
        # Add specific checks based on section 6.1 rules

        return valid

    def kv_cache_generation_options(self):
        """
        Validate KVCache generation options.
        """
        valid = True
        if self.mode != "kvcache":
            return valid

        # TODO: Implement KVCache generation validation rules
        # Add specific checks based on section 6.2 rules

        return valid

    def kv_cache_run_options(self):
        """
        Validate KVCache run options.
        """
        valid = True
        if self.mode != "kvcache":
            return valid

        # TODO: Implement KVCache run validation rules
        # Add specific checks based on section 6.3 rules

        return valid

    def kv_cache_access_via_posix_api_options(self):
        """
        Validate KVCache access via POSIX API options.
        """
        valid = True
        if self.mode != "kvcache":
            return valid

        # TODO: Implement KVCache POSIX API validation rules
        # Add specific checks based on section 6.4 rules

        return valid

    def kv_cache_access_via_object_api_options(self):
        """
        Validate KVCache access via Object API options.
        """
        valid = True
        if self.mode != "kvcache":
            return valid

        # TODO: Implement KVCache Object API validation rules
        # Add specific checks based on section 6.5 rules

        return valid

    def kv_cache_open_vs_closed_options(self):
        """
        Validate KVCache OPEN versus CLOSED options.
        """
        valid = True
        if self.mode != "kvcache":
            return valid

        # TODO: Implement KVCache OPEN/CLOSED validation rules
        # Add specific checks based on section 6.6 rules

        return valid