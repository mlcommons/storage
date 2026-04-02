# VDB-Bench Test Suite

Comprehensive unit test suite for the vdb-bench vector database benchmarking tool.

## Overview

This test suite provides extensive coverage for all components of vdb-bench, including:

- Configuration management
- Database connections
- Vector generation and loading
- Index management
- Benchmarking operations
- Compaction and monitoring
- Performance metrics

## Directory Structure

```
tests/
├── __init__.py                    # Test suite package initialization
├── conftest.py                    # Pytest configuration and shared fixtures
├── run_tests.py                   # Main test runner script
├── requirements-test.txt          # Testing dependencies
│
├── test_config.py                 # Configuration management tests
├── test_database_connection.py    # Database connection tests
├── test_load_vdb.py              # Vector loading tests
├── test_vector_generation.py      # Vector generation tests
├── test_index_management.py       # Index management tests
├── test_simple_bench.py          # Benchmarking functionality tests
├── test_compact_and_watch.py     # Compaction and monitoring tests
│
├── utils/                         # Test utilities
│   ├── __init__.py
│   ├── test_helpers.py           # Helper functions and utilities
│   └── mock_data.py              # Mock data generators
│
└── fixtures/                      # Test fixtures
    └── test_config.yaml          # Sample configuration file
```

## Installation

1. Install test dependencies:

```bash
pip install -r tests/requirements.txt
```

2. Install vdb-bench in development mode:

```bash
pip install -e .
```

## Running Tests

### Run All Tests

```bash
# Using pytest directly
pytest tests/

# Using the test runner
python tests/run_tests.py

# With coverage
python tests/run_tests.py --verbose
```

### Run Specific Test Categories

```bash
# Configuration tests
python tests/run_tests.py --category config

# Connection tests
python tests/run_tests.py --category connection

# Loading tests
python tests/run_tests.py --category loading

# Benchmark tests
python tests/run_tests.py --category benchmark

# Index management tests
python tests/run_tests.py --category index

# Monitoring tests
python tests/run_tests.py --category monitoring
```

### Run Specific Test Modules

```bash
# Run specific test files
python tests/run_tests.py --modules test_config test_load_vdb

# Or using pytest
pytest tests/test_config.py tests/test_load_vdb.py
```

### Run Performance Tests

```bash
# Run only performance-related tests
python tests/run_tests.py --performance

# Or using pytest markers
pytest tests/ -k "performance or benchmark"
```

### Run with Verbose Output

```bash
python tests/run_tests.py --verbose

# Or with pytest
pytest tests/ -v
```

## Test Coverage

### Generate Coverage Report

```bash
# Run tests with coverage
pytest tests/ --cov=vdbbench --cov-report=html

# Or using the test runner
python tests/run_tests.py  # Coverage is enabled by default
```

### View Coverage Report

After running tests with coverage, open the HTML report:

```bash
# Open coverage report in browser
open tests/coverage_html/index.html
```

## Test Configuration

### Environment Variables

Set these environment variables to configure test behavior:

```bash
# Database connection
export VDB_BENCH_TEST_HOST=localhost
export VDB_BENCH_TEST_PORT=19530

# Test data size
export VDB_BENCH_TEST_VECTORS=1000
export VDB_BENCH_TEST_DIMENSION=128

# Performance test settings
export VDB_BENCH_TEST_TIMEOUT=60
```

### Custom Test Configuration

Create a custom test configuration file:

```yaml
# tests/custom_config.yaml
test_settings:
  use_mock_database: true
  vector_count: 5000
  dimension: 256
  test_timeout: 30
```

## Writing New Tests

### Test Structure

Follow this template for new test files:

```python
"""
Unit tests for [component name]
"""
import pytest
from unittest.mock import Mock, patch
import numpy as np

class TestComponentName:
    """Test [component] functionality."""
    
    def test_basic_operation(self):
        """Test basic [operation]."""
        # Test implementation
        assert result == expected
    
    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_parametrized(self, input, expected):
        """Test with multiple inputs."""
        result = function_under_test(input)
        assert result == expected
    
    @pytest.mark.skipif(condition, reason="Reason for skipping")
    def test_conditional(self):
        """Test that runs conditionally."""
        pass
```

### Using Fixtures

Common fixtures are available in `conftest.py`:

```python
def test_with_fixtures(mock_collection, sample_vectors, temp_config_file):
    """Test using provided fixtures."""
    # mock_collection: Mock Milvus collection
    # sample_vectors: Pre-generated test vectors
    # temp_config_file: Temporary config file path
    
    result = process_vectors(mock_collection, sample_vectors)
    assert result is not None
```

### Adding Mock Data

Use mock data generators from `utils/mock_data.py`:

```python
from tests.utils.mock_data import MockDataGenerator

def test_with_mock_data():
    """Test using mock data generators."""
    generator = MockDataGenerator(seed=42)
    
    # Generate SIFT-like vectors
    vectors = generator.generate_sift_like_vectors(1000, 128)
    
    # Generate deep learning embeddings
    embeddings = generator.generate_deep_learning_embeddings(
        500, 768, model_type="bert"
    )
```

## Test Reports

### HTML Report

Tests automatically generate an HTML report:

```bash
# View test report
open tests/test_report.html
```

### JUnit XML Report

JUnit XML format for CI/CD integration:

```bash
# Located at
tests/test_results.xml
```

### JSON Results

Detailed test results in JSON format:

```bash
# Located at
tests/test_results.json
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r tests/requirements-test.txt
        pip install -e .
    
    - name: Run tests
      run: python tests/run_tests.py --verbose
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Debugging Tests

### Run Tests in Debug Mode

```bash
# Run with pytest debugging
pytest tests/ --pdb

# Run specific test with debugging
pytest tests/test_config.py::TestConfigurationLoader::test_load_valid_config --pdb
```

### Increase Verbosity

```bash
# Maximum verbosity
pytest tests/ -vvv

# Show print statements
pytest tests/ -s
```

### Run Failed Tests Only

```bash
# Re-run only failed tests from last run
pytest tests/ --lf

# Run failed tests first, then others
pytest tests/ --ff
```

## Performance Testing

### Run Benchmark Tests

```bash
# Run with benchmark plugin
pytest tests/ --benchmark-only

# Save benchmark results
pytest tests/ --benchmark-save=results

# Compare benchmark results
pytest tests/ --benchmark-compare=results
```

### Memory Profiling

```bash
# Profile memory usage
python -m memory_profiler tests/test_load_vdb.py
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Mocking**: Mock external dependencies (database, file I/O)
3. **Fixtures**: Use fixtures for common setup
4. **Parametrization**: Test multiple inputs with parametrize
5. **Assertions**: Use clear, specific assertions
6. **Documentation**: Document complex test logic
7. **Performance**: Keep tests fast (< 1 second each)
8. **Coverage**: Aim for >80% code coverage

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure vdb-bench is installed in development mode
2. **Mock Failures**: Check that pymilvus mocks are properly configured
3. **Timeout Issues**: Increase timeout for slow tests
4. **Resource Issues**: Some tests may require more memory/CPU

### Getting Help

For issues or questions:
1. Check test logs in `tests/test_results.json`
2. Review HTML report at `tests/test_report.html`
3. Enable verbose mode for detailed output
4. Check fixture definitions in `conftest.py`

## Contributing

When contributing new features, please:
1. Add corresponding unit tests
2. Ensure all tests pass
3. Maintain or improve code coverage
4. Follow the existing test structure
5. Update this README if needed

## License

Same as vdb-bench main project.
