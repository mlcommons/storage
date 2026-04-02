#!/usr/bin/env python3
"""
Test Suite Verification Script
Verifies that all test fixes have been applied correctly
"""
import subprocess
import sys
import json
from pathlib import Path

def run_single_test(test_path):
    """Run a single test and return result."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    return result.returncode == 0, result.stdout, result.stderr

def main():
    """Run all previously failing tests to verify fixes."""
    
    # List of previously failing tests
    failing_tests = [
        "tests/test_compact_and_watch.py::TestMonitoring::test_collection_stats_monitoring",
        "tests/test_config.py::TestConfigurationLoader::test_config_environment_variable_override",
        "tests/test_database_connection.py::TestConnectionResilience::test_automatic_reconnection",
        "tests/test_index_management.py::TestIndexManagement::test_index_status_check",
        "tests/test_load_vdb.py::TestVectorLoading::test_insertion_with_error_handling",
        "tests/test_load_vdb.py::TestVectorLoading::test_insertion_rate_monitoring",
        "tests/test_simple_bench.py::TestBenchmarkConfiguration::test_workload_generation"
    ]
    
    print("=" * 60)
    print("VDB-Bench Test Suite - Verification of Fixes")
    print("=" * 60)
    print()
    
    results = []
    
    for test in failing_tests:
        print(f"Testing: {test}")
        passed, stdout, stderr = run_single_test(test)
        
        results.append({
            "test": test,
            "passed": passed,
            "output": stdout if not passed else ""
        })
        
        if passed:
            print("  ✅ PASSED")
        else:
            print("  ❌ FAILED")
            print(f"  Error: {stderr[:200]}")
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = len(results) - passed_count
    
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    
    if failed_count == 0:
        print("\n✅ All previously failing tests now pass!")
        return 0
    else:
        print("\n❌ Some tests are still failing. Please review the fixes.")
        for result in results:
            if not result["passed"]:
                print(f"  - {result['test']}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
