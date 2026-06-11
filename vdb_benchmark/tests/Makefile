# Makefile for VDB-Bench Test Suite

.PHONY: help install test test-all test-config test-connection test-loading \
        test-benchmark test-index test-monitoring test-performance \
        test-integration coverage coverage-html clean lint format \
        test-verbose test-failed test-parallel

# Default target
help:
	@echo "VDB-Bench Test Suite Makefile"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  make install          - Install test dependencies"
	@echo "  make test            - Run all tests"
	@echo "  make test-verbose    - Run tests with verbose output"
	@echo "  make test-parallel   - Run tests in parallel"
	@echo "  make test-failed     - Re-run only failed tests"
	@echo ""
	@echo "Test categories:"
	@echo "  make test-config     - Run configuration tests"
	@echo "  make test-connection - Run connection tests"
	@echo "  make test-loading    - Run loading tests"
	@echo "  make test-benchmark  - Run benchmark tests"
	@echo "  make test-index      - Run index management tests"
	@echo "  make test-monitoring - Run monitoring tests"
	@echo ""
	@echo "Special test suites:"
	@echo "  make test-performance - Run performance tests"
	@echo "  make test-integration - Run integration tests"
	@echo ""
	@echo "Coverage and reports:"
	@echo "  make coverage        - Run tests with coverage"
	@echo "  make coverage-html   - Generate HTML coverage report"
	@echo ""
	@echo "Code quality:"
	@echo "  make lint           - Run code linting"
	@echo "  make format         - Format code with black"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Clean test artifacts"

# Installation
install:
	pip install -r tests/requirements-test.txt
	pip install -e .

# Basic test execution
test:
	python tests/run_tests.py

test-all: test

test-verbose:
	python tests/run_tests.py --verbose

test-parallel:
	pytest tests/ -n auto --dist loadscope

test-failed:
	pytest tests/ --lf

# Test categories
test-config:
	python tests/run_tests.py --category config

test-connection:
	python tests/run_tests.py --category connection

test-loading:
	python tests/run_tests.py --category loading

test-benchmark:
	python tests/run_tests.py --category benchmark

test-index:
	python tests/run_tests.py --category index

test-monitoring:
	python tests/run_tests.py --category monitoring

# Special test suites
test-performance:
	python tests/run_tests.py --performance

test-integration:
	python tests/run_tests.py --integration

# Coverage
coverage:
	pytest tests/ --cov=vdbbench --cov-report=term --cov-report=html

coverage-html: coverage
	@echo "Opening coverage report in browser..."
	@python -m webbrowser tests/htmlcov/index.html

# Code quality
lint:
	@echo "Running flake8..."
	flake8 tests/ --max-line-length=100 --ignore=E203,W503
	@echo "Running pylint..."
	pylint tests/ --max-line-length=100 --disable=C0111,R0903,R0913
	@echo "Running mypy..."
	mypy tests/ --ignore-missing-imports

format:
	black tests/ --line-length=100
	isort tests/ --profile black --line-length=100

# Clean up
clean:
	@echo "Cleaning test artifacts..."
	rm -rf tests/__pycache__
	rm -rf tests/utils/__pycache__
	rm -rf tests/.pytest_cache
	rm -rf tests/htmlcov
	rm -rf tests/coverage_html
	rm -f tests/.coverage
	rm -f tests/test_results.xml
	rm -f tests/test_results.json
	rm -f tests/test_report.html
	rm -f tests/*.pyc
	rm -rf tests/**/*.pyc
	find tests/ -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

# Watch mode (requires pytest-watch)
watch:
	ptw tests/ -- --verbose

# Run specific test file
test-file:
	@read -p "Enter test file name (without .py): " file; \
	pytest tests/$$file.py -v

# Run tests matching pattern
test-match:
	@read -p "Enter test pattern: " pattern; \
	pytest tests/ -k "$$pattern" -v

# Generate test report
report:
	pytest tests/ --html=tests/test_report.html --self-contained-html
	@echo "Test report generated at tests/test_report.html"

# Check test coverage for specific module
coverage-module:
	@read -p "Enter module name: " module; \
	pytest tests/ --cov=vdbbench.$$module --cov-report=term

# Quick test (fast subset of tests)
test-quick:
	pytest tests/ -m "not slow" --maxfail=1 -x

# Full test suite with all checks
test-full: clean lint test-parallel coverage report
	@echo "Full test suite complete!"

# Continuous Integration target
ci: install lint test-parallel coverage
	@echo "CI test suite complete!"

# Development target (format, lint, and test)
dev: format lint test-verbose
	@echo "Development test cycle complete!"
