"""
Test helper utilities for vdb-bench tests
"""
import numpy as np
import time
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, MagicMock
import random
import string
from contextlib import contextmanager
import tempfile
import shutil


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def generate_vectors(num_vectors: int, dimension: int, 
                        distribution: str = "normal",
                        seed: Optional[int] = None) -> np.ndarray:
        """Generate test vectors with specified distribution."""
        if seed is not None:
            np.random.seed(seed)
        
        if distribution == "normal":
            return np.random.randn(num_vectors, dimension).astype(np.float32)
        elif distribution == "uniform":
            return np.random.uniform(-1, 1, (num_vectors, dimension)).astype(np.float32)
        elif distribution == "sparse":
            vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
            mask = np.random.random((num_vectors, dimension)) < 0.9
            vectors[mask] = 0
            return vectors
        elif distribution == "clustered":
            vectors = []
            clusters = 10
            vectors_per_cluster = num_vectors // clusters
            
            for _ in range(clusters):
                center = np.random.randn(dimension) * 10
                cluster_vectors = center + np.random.randn(vectors_per_cluster, dimension) * 0.5
                vectors.append(cluster_vectors)
            
            return np.vstack(vectors).astype(np.float32)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    @staticmethod
    def generate_ids(num_ids: int, start: int = 0) -> List[int]:
        """Generate sequential IDs."""
        return list(range(start, start + num_ids))
    
    @staticmethod
    def generate_metadata(num_items: int) -> List[Dict[str, Any]]:
        """Generate random metadata for vectors."""
        metadata = []
        
        for i in range(num_items):
            metadata.append({
                "id": i,
                "category": random.choice(["A", "B", "C", "D"]),
                "timestamp": time.time() + i,
                "score": random.random(),
                "tags": random.sample(["tag1", "tag2", "tag3", "tag4", "tag5"], 
                                     k=random.randint(1, 3))
            })
        
        return metadata
    
    @staticmethod
    def generate_ground_truth(num_queries: int, num_vectors: int, 
                             top_k: int = 100) -> Dict[int, List[int]]:
        """Generate ground truth for recall calculation."""
        ground_truth = {}
        
        for query_id in range(num_queries):
            # Generate random ground truth IDs
            true_ids = random.sample(range(num_vectors), 
                                   min(top_k, num_vectors))
            ground_truth[query_id] = true_ids
        
        return ground_truth
    
    @staticmethod
    def generate_config(collection_name: str = "test_collection") -> Dict[str, Any]:
        """Generate test configuration."""
        return {
            "database": {
                "host": "localhost",
                "port": 19530,
                "database": "default",
                "timeout": 30
            },
            "dataset": {
                "collection_name": collection_name,
                "num_vectors": 10000,
                "dimension": 128,
                "distribution": "uniform",
                "batch_size": 1000,
                "num_shards": 2
            },
            "index": {
                "index_type": "HNSW",
                "metric_type": "L2",
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            },
            "benchmark": {
                "num_queries": 1000,
                "top_k": 10,
                "num_processes": 4,
                "runtime": 60
            }
        }


class MockMilvusCollection:
    """Advanced mock Milvus collection for testing."""
    
    def __init__(self, name: str, dimension: int = 128):
        self.name = name
        self.dimension = dimension
        self.vectors = []
        self.ids = []
        self.num_entities = 0
        self.index = None
        self.is_loaded = False
        self.partitions = []
        self.schema = Mock()
        self.description = f"Mock collection {name}"
        
        # Index-related attributes
        self.index_progress = 0
        self.index_state = "NotExist"
        self.index_params = None
        
        # Compaction-related
        self.compaction_id = None
        self.compaction_state = "Idle"
        
        # Search behavior
        self.search_latency = 0.01  # Default 10ms
        self.search_results = None
    
    def insert(self, data: List) -> Mock:
        """Mock insert operation."""
        vectors = data[0] if isinstance(data[0], (list, np.ndarray)) else data
        num_new = len(vectors) if hasattr(vectors, '__len__') else 1
        
        self.vectors.extend(vectors)
        new_ids = list(range(self.num_entities, self.num_entities + num_new))
        self.ids.extend(new_ids)
        self.num_entities += num_new
        
        result = Mock()
        result.primary_keys = new_ids
        result.insert_count = num_new
        
        return result
    
    def search(self, data: List, anns_field: str, param: Dict, 
               limit: int = 10, **kwargs) -> List:
        """Mock search operation."""
        time.sleep(self.search_latency)  # Simulate latency
        
        if self.search_results:
            return self.search_results
        
        # Generate mock results
        results = []
        for query in data:
            query_results = []
            for i in range(min(limit, 10)):
                result = Mock()
                result.id = random.randint(0, max(self.num_entities - 1, 0))
                result.distance = random.random()
                query_results.append(result)
            results.append(query_results)
        
        return results
    
    def create_index(self, field_name: str, index_params: Dict) -> bool:
        """Mock index creation."""
        self.index_params = index_params
        self.index_state = "InProgress"
        self.index_progress = 0
        
        # Simulate index building
        self.index = Mock()
        self.index.params = index_params
        self.index.field_name = field_name
        
        return True
    
    def drop_index(self, field_name: str) -> None:
        """Mock index dropping."""
        self.index = None
        self.index_state = "NotExist"
        self.index_progress = 0
        self.index_params = None
    
    def load(self) -> None:
        """Mock collection loading."""
        self.is_loaded = True
    
    def release(self) -> None:
        """Mock collection release."""
        self.is_loaded = False
    
    def flush(self) -> None:
        """Mock flush operation."""
        pass  # Simulate successful flush
    
    def compact(self) -> int:
        """Mock compaction operation."""
        self.compaction_id = random.randint(1000, 9999)
        self.compaction_state = "Executing"
        return self.compaction_id
    
    def get_compaction_state(self, compaction_id: int) -> str:
        """Mock getting compaction state."""
        return self.compaction_state
    
    def drop(self) -> None:
        """Mock collection drop."""
        self.vectors = []
        self.ids = []
        self.num_entities = 0
        self.index = None
    
    def create_partition(self, partition_name: str) -> None:
        """Mock partition creation."""
        if partition_name not in self.partitions:
            self.partitions.append(partition_name)
    
    def has_partition(self, partition_name: str) -> bool:
        """Check if partition exists."""
        return partition_name in self.partitions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "row_count": self.num_entities,
            "partitions": len(self.partitions),
            "index_state": self.index_state,
            "loaded": self.is_loaded
        }


class PerformanceSimulator:
    """Simulate performance metrics for testing."""
    
    def __init__(self):
        self.base_latency = 10  # Base latency in ms
        self.base_qps = 1000
        self.variation = 0.2  # 20% variation
    
    def simulate_latency(self, num_samples: int = 100) -> List[float]:
        """Generate simulated latency values."""
        latencies = []
        
        for _ in range(num_samples):
            # Add random variation
            variation = random.uniform(1 - self.variation, 1 + self.variation)
            latency = self.base_latency * variation
            
            # Occasionally add outliers
            if random.random() < 0.05:  # 5% outliers
                latency *= random.uniform(2, 5)
            
            latencies.append(latency)
        
        return latencies
    
    def simulate_throughput(self, duration: int = 60) -> List[Tuple[float, float]]:
        """Generate simulated throughput over time."""
        throughput_data = []
        current_time = 0
        
        while current_time < duration:
            # Simulate varying QPS
            variation = random.uniform(1 - self.variation, 1 + self.variation)
            qps = self.base_qps * variation
            
            # Occasionally simulate load spikes or drops
            if random.random() < 0.1:  # 10% chance of anomaly
                if random.random() < 0.5:
                    qps *= 0.5  # Drop
                else:
                    qps *= 1.5  # Spike
            
            throughput_data.append((current_time, qps))
            current_time += 1
        
        return throughput_data
    
    def simulate_resource_usage(self, duration: int = 60) -> Dict[str, List[Tuple[float, float]]]:
        """Simulate CPU and memory usage over time."""
        cpu_usage = []
        memory_usage = []
        
        base_cpu = 50
        base_memory = 60
        
        for t in range(duration):
            # CPU usage
            cpu = base_cpu + random.uniform(-10, 20)
            cpu = max(0, min(100, cpu))  # Clamp to 0-100
            cpu_usage.append((t, cpu))
            
            # Memory usage (more stable)
            memory = base_memory + random.uniform(-5, 10)
            memory = max(0, min(100, memory))
            memory_usage.append((t, memory))
            
            # Gradually increase if simulating memory leak
            if random.random() < 0.1:
                base_memory += 0.5
        
        return {
            "cpu": cpu_usage,
            "memory": memory_usage
        }


@contextmanager
def temporary_directory():
    """Context manager for temporary directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@contextmanager
def mock_time_progression(increments: List[float]):
    """Mock time.time() with controlled progression."""
    time_values = []
    current = 0
    
    for increment in increments:
        current += increment
        time_values.append(current)
    
    with patch('time.time', side_effect=time_values):
        yield


def create_test_yaml_config(path: Path, config: Dict[str, Any]) -> None:
    """Create a YAML configuration file for testing."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_test_json_results(path: Path, results: Dict[str, Any]) -> None:
    """Create a JSON results file for testing."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def assert_performance_within_bounds(actual: float, expected: float, 
                                   tolerance: float = 0.1) -> None:
    """Assert that performance metric is within expected bounds."""
    lower_bound = expected * (1 - tolerance)
    upper_bound = expected * (1 + tolerance)
    
    assert lower_bound <= actual <= upper_bound, \
        f"Performance {actual} not within {tolerance*100}% of expected {expected}"


def calculate_recall(retrieved: List[int], relevant: List[int], k: int) -> float:
    """Calculate recall@k metric."""
    retrieved_k = set(retrieved[:k])
    relevant_k = set(relevant[:k])
    
    if not relevant_k:
        return 0.0
    
    intersection = retrieved_k.intersection(relevant_k)
    return len(intersection) / len(relevant_k)


def calculate_precision(retrieved: List[int], relevant: List[int], k: int) -> float:
    """Calculate precision@k metric."""
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    
    if not retrieved_k:
        return 0.0
    
    intersection = retrieved_k.intersection(relevant_set)
    return len(intersection) / len(retrieved_k)


def generate_random_string(length: int = 10) -> str:
    """Generate random string for testing."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


class BenchmarkResultValidator:
    """Validate benchmark results for consistency."""
    
    @staticmethod
    def validate_metrics(metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate that metrics are reasonable."""
        errors = []
        
        # Check required fields
        required_fields = ["qps", "latency_p50", "latency_p95", "latency_p99"]
        for field in required_fields:
            if field not in metrics:
                errors.append(f"Missing required field: {field}")
        
        # Check value ranges
        if "qps" in metrics:
            if metrics["qps"] <= 0:
                errors.append("QPS must be positive")
            if metrics["qps"] > 1000000:
                errors.append("QPS seems unrealistically high")
        
        if "latency_p50" in metrics and "latency_p95" in metrics:
            if metrics["latency_p50"] > metrics["latency_p95"]:
                errors.append("P50 latency cannot be greater than P95")
        
        if "latency_p95" in metrics and "latency_p99" in metrics:
            if metrics["latency_p95"] > metrics["latency_p99"]:
                errors.append("P95 latency cannot be greater than P99")
        
        if "error_rate" in metrics:
            if not (0 <= metrics["error_rate"] <= 1):
                errors.append("Error rate must be between 0 and 1")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_consistency(results: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Check consistency across multiple benchmark runs."""
        if len(results) < 2:
            return True, []
        
        errors = []
        
        # Check for extreme variations
        qps_values = [r["qps"] for r in results if "qps" in r]
        if qps_values:
            mean_qps = sum(qps_values) / len(qps_values)
            for i, qps in enumerate(qps_values):
                if abs(qps - mean_qps) / mean_qps > 0.5:  # 50% variation
                    errors.append(f"Run {i} has QPS {qps} which varies >50% from mean {mean_qps}")
        
        return len(errors) == 0, errors
