#!/usr/bin/env python3
"""
Comprehensive test runner for vdb-bench test suite
"""
import sys
import os
import argparse
import pytest
import coverage
from pathlib import Path
from typing import List, Optional
import json
import time
from datetime import datetime


class TestRunner:
    """Main test runner for vdb-bench test suite."""
    
    def __init__(self, test_dir: Path = None):
        """Initialize test runner."""
        self.test_dir = test_dir or Path(__file__).parent
        self.results = {
            "start_time": None,
            "end_time": None,
            "duration": 0,
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "coverage": None
        }
    
    def run_all_tests(self, verbose: bool = False, 
                     coverage_enabled: bool = True) -> int:
        """Run all tests with optional coverage."""
        print("=" * 60)
        print("VDB-Bench Test Suite Runner")
        print("=" * 60)
        
        self.results["start_time"] = datetime.now().isoformat()
        start = time.time()
        
        # Setup coverage if enabled
        cov = None
        if coverage_enabled:
            cov = coverage.Coverage()
            cov.start()
            print("Coverage tracking enabled")
        
        # Prepare pytest arguments
        pytest_args = [
            str(self.test_dir),
            "-v" if verbose else "-q",
            "--tb=short",
            "--color=yes",
            f"--junitxml={self.test_dir}/test_results.xml",
            f"--html={self.test_dir}/test_report.html",
            "--self-contained-html"
        ]
        
        # Run pytest
        print(f"\nRunning tests from: {self.test_dir}")
        print("-" * 60)
        
        exit_code = pytest.main(pytest_args)
        
        # Stop coverage and generate report
        if cov:
            cov.stop()
            cov.save()
            
            # Generate coverage report
            print("\n" + "=" * 60)
            print("Coverage Report")
            print("-" * 60)
            
            cov.report()
            
            # Save HTML coverage report
            html_dir = self.test_dir / "coverage_html"
            cov.html_report(directory=str(html_dir))
            print(f"\nHTML coverage report saved to: {html_dir}")
            
            # Get coverage percentage
            self.results["coverage"] = cov.report(show_missing=False)
        
        # Update results
        self.results["end_time"] = datetime.now().isoformat()
        self.results["duration"] = time.time() - start
        
        # Parse test results
        self._parse_test_results(exit_code)
        
        # Save results to JSON
        self._save_results()
        
        # Print summary
        self._print_summary()
        
        return exit_code
    
    def run_specific_tests(self, test_modules: List[str], 
                          verbose: bool = False) -> int:
        """Run specific test modules."""
        print("=" * 60)
        print(f"Running specific tests: {', '.join(test_modules)}")
        print("=" * 60)
        
        pytest_args = []
        for module in test_modules:
            test_path = self.test_dir / f"{module}.py"
            if test_path.exists():
                pytest_args.append(str(test_path))
            else:
                print(f"Warning: Test module not found: {test_path}")
        
        if not pytest_args:
            print("No valid test modules found!")
            return 1
        
        if verbose:
            pytest_args.append("-v")
        else:
            pytest_args.append("-q")
        
        pytest_args.extend(["--tb=short", "--color=yes"])
        
        return pytest.main(pytest_args)
    
    def run_by_category(self, category: str, verbose: bool = False) -> int:
        """Run tests by category."""
        category_map = {
            "config": ["test_config"],
            "connection": ["test_database_connection"],
            "loading": ["test_load_vdb", "test_vector_generation"],
            "benchmark": ["test_simple_bench"],
            "index": ["test_index_management"],
            "monitoring": ["test_compact_and_watch"],
            "all": None  # Run all tests
        }
        
        if category not in category_map:
            print(f"Unknown category: {category}")
            print(f"Available categories: {', '.join(category_map.keys())}")
            return 1
        
        if category == "all":
            return self.run_all_tests(verbose=verbose)
        
        test_modules = category_map[category]
        return self.run_specific_tests(test_modules, verbose=verbose)
    
    def run_performance_tests(self, verbose: bool = False) -> int:
        """Run performance-related tests."""
        print("=" * 60)
        print("Running Performance Tests")
        print("=" * 60)
        
        pytest_args = [
            str(self.test_dir),
            "-v" if verbose else "-q",
            "-k", "performance or benchmark or throughput",
            "--tb=short",
            "--color=yes"
        ]
        
        return pytest.main(pytest_args)
    
    def run_integration_tests(self, verbose: bool = False) -> int:
        """Run integration tests."""
        print("=" * 60)
        print("Running Integration Tests")
        print("=" * 60)
        
        pytest_args = [
            str(self.test_dir),
            "-v" if verbose else "-q",
            "-m", "integration",
            "--tb=short",
            "--color=yes"
        ]
        
        return pytest.main(pytest_args)
    
    def _parse_test_results(self, exit_code: int) -> None:
        """Parse test results from pytest exit code."""
        # Basic result parsing based on exit code
        if exit_code == 0:
            self.results["status"] = "SUCCESS"
        elif exit_code == 1:
            self.results["status"] = "TESTS_FAILED"
        elif exit_code == 2:
            self.results["status"] = "INTERRUPTED"
        elif exit_code == 3:
            self.results["status"] = "INTERNAL_ERROR"
        elif exit_code == 4:
            self.results["status"] = "USAGE_ERROR"
        elif exit_code == 5:
            self.results["status"] = "NO_TESTS"
        else:
            self.results["status"] = "UNKNOWN_ERROR"
        
        # Try to parse XML results if available
        xml_path = self.test_dir / "test_results.xml"
        if xml_path.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                testsuite = root.find("testsuite") or root
                self.results["total_tests"] = int(testsuite.get("tests", 0))
                self.results["failed"] = int(testsuite.get("failures", 0))
                self.results["errors"] = int(testsuite.get("errors", 0))
                self.results["skipped"] = int(testsuite.get("skipped", 0))
                self.results["passed"] = (
                    self.results["total_tests"] - 
                    self.results["failed"] - 
                    self.results["errors"] - 
                    self.results["skipped"]
                )
            except Exception as e:
                print(f"Warning: Could not parse XML results: {e}")
    
    def _save_results(self) -> None:
        """Save test results to JSON file."""
        results_path = self.test_dir / "test_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nTest results saved to: {results_path}")
    
    def _print_summary(self) -> None:
        """Print test execution summary."""
        print("\n" + "=" * 60)
        print("Test Execution Summary")
        print("=" * 60)
        
        print(f"Status: {self.results.get('status', 'UNKNOWN')}")
        print(f"Duration: {self.results['duration']:.2f} seconds")
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"Passed: {self.results['passed']}")
        print(f"Failed: {self.results['failed']}")
        print(f"Errors: {self.results['errors']}")
        print(f"Skipped: {self.results['skipped']}")
        
        if self.results.get("coverage"):
            print(f"Code Coverage: {self.results['coverage']:.1f}%")
        
        print("=" * 60)
        
        # Print pass rate
        if self.results['total_tests'] > 0:
            pass_rate = (self.results['passed'] / self.results['total_tests']) * 100
            print(f"Pass Rate: {pass_rate:.1f}%")
            
            if pass_rate == 100:
                print("✅ All tests passed!")
            elif pass_rate >= 90:
                print("⚠️  Most tests passed, but some failures detected.")
            else:
                print("❌ Significant test failures detected.")
        
        print("=" * 60)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="VDB-Bench Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--category", "-c",
        choices=["all", "config", "connection", "loading", 
                "benchmark", "index", "monitoring"],
        default="all",
        help="Test category to run"
    )
    
    parser.add_argument(
        "--modules", "-m",
        nargs="+",
        help="Specific test modules to run"
    )
    
    parser.add_argument(
        "--performance", "-p",
        action="store_true",
        help="Run performance tests only"
    )
    
    parser.add_argument(
        "--integration", "-i",
        action="store_true",
        help="Run integration tests only"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage tracking"
    )
    
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Test directory path"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(test_dir=args.test_dir)
    
    # Determine which tests to run
    if args.modules:
        exit_code = runner.run_specific_tests(args.modules, verbose=args.verbose)
    elif args.performance:
        exit_code = runner.run_performance_tests(verbose=args.verbose)
    elif args.integration:
        exit_code = runner.run_integration_tests(verbose=args.verbose)
    elif args.category != "all":
        exit_code = runner.run_by_category(args.category, verbose=args.verbose)
    else:
        exit_code = runner.run_all_tests(
            verbose=args.verbose,
            coverage_enabled=not args.no_coverage
        )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
