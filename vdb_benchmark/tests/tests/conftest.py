"""
Pytest configuration and fixtures for vdb-bench tests
"""
import pytest
import yaml
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from typing import Dict, Any, Generator
import os

# Mock pymilvus if not installed
try:
    from pymilvus import connections, Collection, utility
except ImportError:
    connections = MagicMock()
    Collection = MagicMock()
    utility = MagicMock()


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create a temporary directory for test data that persists for the session."""
    temp_dir = Path(tempfile.mkdtemp(prefix="vdb_bench_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def temp_config_file(test_data_dir) -> Generator[Path, None, None]:
    """Create a temporary configuration file for testing."""
    config_path = test_data_dir / "test_config.yaml"
    config_data = {
        "database": {
            "host": "127.0.0.1",
            "port": 19530,
            "database": "milvus_test",
            "max_receive_message_length": 514983574,
            "max_send_message_length": 514983574
        },
        "dataset": {
            "collection_name": "test_collection",
            "num_vectors": 1000,
            "dimension": 128,
            "distribution": "uniform",
            "batch_size": 100,
            "num_shards": 2,
            "vector_dtype": "FLOAT_VECTOR"
        },
        "index": {
            "index_type": "DISKANN",
            "metric_type": "COSINE",
            "max_degree": 64,
            "search_list_size": 200
        },
        "workflow": {
            "compact": True
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    yield config_path
    
    if config_path.exists():
        config_path.unlink()


@pytest.fixture
def mock_milvus_connection():
    """Mock Milvus connection for testing."""
    with patch('pymilvus.connections.connect') as mock_connect:
        mock_connect.return_value = Mock()
        yield mock_connect


@pytest.fixture
def mock_collection():
    """Mock Milvus collection for testing."""
    mock_coll = Mock(spec=Collection)
    mock_coll.name = "test_collection"
    mock_coll.schema = Mock()
    mock_coll.num_entities = 1000
    mock_coll.insert = Mock(return_value=Mock(primary_keys=[1, 2, 3]))
    mock_coll.create_index = Mock()
    mock_coll.load = Mock()
    mock_coll.release = Mock()
    mock_coll.flush = Mock()
    mock_coll.compact = Mock()
    return mock_coll


@pytest.fixture
def sample_vectors() -> np.ndarray:
    """Generate sample vectors for testing."""
    np.random.seed(42)
    return np.random.randn(100, 128).astype(np.float32)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration dictionary."""
    return {
        "database": {
            "host": "localhost",
            "port": 19530,
            "database": "default"
        },
        "dataset": {
            "collection_name": "test_vectors",
            "num_vectors": 10000,
            "dimension": 1536,
            "distribution": "uniform",
            "batch_size": 1000
        },
        "index": {
            "index_type": "DISKANN",
            "metric_type": "COSINE"
        }
    }


@pytest.fixture
def mock_time():
    """Mock time module for testing time-based operations."""
    with patch('time.time') as mock_time_func:
        mock_time_func.side_effect = [0, 1, 2, 3, 4, 5]  # Incremental time
        yield mock_time_func


@pytest.fixture
def mock_multiprocessing():
    """Mock multiprocessing for testing parallel operations."""
    with patch('multiprocessing.Pool') as mock_pool:
        mock_pool_instance = Mock()
        mock_pool_instance.map = Mock(side_effect=lambda func, args: [func(arg) for arg in args])
        mock_pool_instance.close = Mock()
        mock_pool_instance.join = Mock()
        mock_pool.return_value.__enter__ = Mock(return_value=mock_pool_instance)
        mock_pool.return_value.__exit__ = Mock(return_value=None)
        yield mock_pool


@pytest.fixture
def benchmark_results():
    """Sample benchmark results for testing."""
    return {
        "qps": 1250.5,
        "latency_p50": 0.8,
        "latency_p95": 1.2,
        "latency_p99": 1.5,
        "total_queries": 10000,
        "runtime": 8.0,
        "errors": 0
    }


@pytest.fixture(autouse=True)
def reset_milvus_connections():
    """Reset Milvus connections before each test."""
    connections.disconnect("default")
    yield
    connections.disconnect("default")


@pytest.fixture
def env_vars():
    """Set up environment variables for testing."""
    original_env = os.environ.copy()
    
    os.environ['VDB_BENCH_HOST'] = 'test_host'
    os.environ['VDB_BENCH_PORT'] = '19530'
    
    yield os.environ
    
    os.environ.clear()
    os.environ.update(original_env)
