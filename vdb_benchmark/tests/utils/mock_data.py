"""
Mock data generators for vdb-bench testing
"""
import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import json


class MockDataGenerator:
    """Generate various types of mock data for testing."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    @staticmethod
    def generate_sift_like_vectors(num_vectors: int, dimension: int = 128) -> np.ndarray:
        """Generate SIFT-like vectors (similar to common benchmark datasets)."""
        # SIFT vectors are typically L2-normalized and have specific distribution
        vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        
        # Add some structure (make some dimensions more important)
        important_dims = random.sample(range(dimension), k=dimension // 4)
        vectors[:, important_dims] *= 3
        
        # L2 normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-10)
        
        # Scale to typical SIFT range
        vectors = vectors * 512
        
        return vectors.astype(np.float32)
    
    @staticmethod
    def generate_deep_learning_embeddings(num_vectors: int, 
                                         dimension: int = 768,
                                         model_type: str = "bert") -> np.ndarray:
        """Generate embeddings similar to deep learning models."""
        if model_type == "bert":
            # BERT-like embeddings (768-dimensional)
            vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
            # BERT embeddings typically have values in [-2, 2] range
            vectors = np.clip(vectors * 0.5, -2, 2)
            
        elif model_type == "resnet":
            # ResNet-like features (2048-dimensional typical)
            vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
            # Apply ReLU-like sparsity
            vectors[vectors < 0] = 0
            # L2 normalize
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / (norms + 1e-10)
            
        elif model_type == "clip":
            # CLIP-like embeddings (512-dimensional, normalized)
            vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
            # Normalize to unit sphere
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / (norms + 1e-10)
            
        else:
            # Generic embeddings
            vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        
        return vectors
    
    @staticmethod
    def generate_time_series_vectors(num_vectors: int,
                                    dimension: int = 100,
                                    num_series: int = 10) -> Tuple[np.ndarray, List[int]]:
        """Generate time series data as vectors with series labels."""
        vectors = []
        labels = []
        
        for series_id in range(num_series):
            # Generate base pattern for this series
            base_pattern = np.sin(np.linspace(0, 4 * np.pi, dimension))
            base_pattern += np.random.randn(dimension) * 0.1  # Add noise
            
            # Generate variations of the pattern
            series_vectors = num_vectors // num_series
            for _ in range(series_vectors):
                # Add temporal drift and noise
                variation = base_pattern + np.random.randn(dimension) * 0.3
                variation += np.random.randn() * 0.1  # Global shift
                
                vectors.append(variation)
                labels.append(series_id)
        
        # Handle remaining vectors
        remaining = num_vectors - len(vectors)
        for _ in range(remaining):
            vectors.append(vectors[-1] + np.random.randn(dimension) * 0.1)
            labels.append(labels[-1])
        
        return np.array(vectors).astype(np.float32), labels
    
    @staticmethod
    def generate_categorical_embeddings(num_vectors: int,
                                       num_categories: int = 100,
                                       dimension: int = 64) -> Tuple[np.ndarray, List[str]]:
        """Generate embeddings for categorical data."""
        # Create embedding for each category
        category_embeddings = np.random.randn(num_categories, dimension).astype(np.float32)
        
        # Normalize category embeddings
        norms = np.linalg.norm(category_embeddings, axis=1, keepdims=True)
        category_embeddings = category_embeddings / (norms + 1e-10)
        
        vectors = []
        categories = []
        
        # Generate vectors by sampling categories
        for _ in range(num_vectors):
            cat_idx = random.randint(0, num_categories - 1)
            
            # Add small noise to category embedding
            vector = category_embeddings[cat_idx] + np.random.randn(dimension) * 0.05
            
            vectors.append(vector)
            categories.append(f"category_{cat_idx}")
        
        return np.array(vectors).astype(np.float32), categories
    
    @staticmethod
    def generate_multimodal_vectors(num_vectors: int,
                                   text_dim: int = 768,
                                   image_dim: int = 2048) -> Dict[str, np.ndarray]:
        """Generate multimodal vectors (text + image embeddings)."""
        # Generate text embeddings (BERT-like)
        text_vectors = np.random.randn(num_vectors, text_dim).astype(np.float32)
        text_vectors = np.clip(text_vectors * 0.5, -2, 2)
        
        # Generate image embeddings (ResNet-like)
        image_vectors = np.random.randn(num_vectors, image_dim).astype(np.float32)
        image_vectors[image_vectors < 0] = 0  # ReLU
        norms = np.linalg.norm(image_vectors, axis=1, keepdims=True)
        image_vectors = image_vectors / (norms + 1e-10)
        
        # Combined embeddings (concatenated and projected)
        combined_dim = 512
        projection_matrix = np.random.randn(text_dim + image_dim, combined_dim).astype(np.float32)
        projection_matrix /= np.sqrt(text_dim + image_dim)  # Xavier initialization
        
        concatenated = np.hstack([text_vectors, image_vectors])
        combined_vectors = np.dot(concatenated, projection_matrix)
        
        # Normalize combined vectors
        norms = np.linalg.norm(combined_vectors, axis=1, keepdims=True)
        combined_vectors = combined_vectors / (norms + 1e-10)
        
        return {
            "text": text_vectors,
            "image": image_vectors,
            "combined": combined_vectors
        }


class BenchmarkDatasetGenerator:
    """Generate datasets similar to common benchmarks."""
    
    @staticmethod
    def generate_ann_benchmark_dataset(dataset_type: str = "random",
                                      num_train: int = 100000,
                                      num_test: int = 10000,
                                      dimension: int = 128,
                                      num_neighbors: int = 100) -> Dict[str, Any]:
        """Generate dataset similar to ANN-Benchmarks format."""
        
        if dataset_type == "random":
            train_vectors = np.random.randn(num_train, dimension).astype(np.float32)
            test_vectors = np.random.randn(num_test, dimension).astype(np.float32)
            
        elif dataset_type == "clustered":
            train_vectors = []
            num_clusters = 100
            vectors_per_cluster = num_train // num_clusters
            
            for _ in range(num_clusters):
                center = np.random.randn(dimension) * 10
                cluster = center + np.random.randn(vectors_per_cluster, dimension)
                train_vectors.append(cluster)
            
            train_vectors = np.vstack(train_vectors).astype(np.float32)
            
            # Test vectors from same distribution
            test_vectors = []
            test_per_cluster = num_test // num_clusters
            
            for _ in range(num_clusters):
                center = np.random.randn(dimension) * 10
                cluster = center + np.random.randn(test_per_cluster, dimension)
                test_vectors.append(cluster)
            
            test_vectors = np.vstack(test_vectors).astype(np.float32)
            
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Generate ground truth (simplified - random for now)
        ground_truth = np.random.randint(0, num_train, 
                                        (num_test, num_neighbors))
        
        # Calculate distances for ground truth (simplified)
        distances = np.random.random((num_test, num_neighbors)).astype(np.float32)
        distances.sort(axis=1)  # Ensure sorted by distance
        
        return {
            "train": train_vectors,
            "test": test_vectors,
            "neighbors": ground_truth,
            "distances": distances,
            "dimension": dimension,
            "metric": "euclidean"
        }
    
    @staticmethod
    def generate_streaming_dataset(initial_size: int = 10000,
                                  dimension: int = 128,
                                  stream_rate: int = 100,
                                  drift_rate: float = 0.01) -> Dict[str, Any]:
        """Generate dataset that simulates streaming/incremental scenarios."""
        # Initial dataset
        initial_vectors = np.random.randn(initial_size, dimension).astype(np.float32)
        
        # Streaming batches with concept drift
        stream_batches = []
        current_center = np.zeros(dimension)
        
        for batch_id in range(10):  # 10 batches
            # Drift the distribution center
            current_center += np.random.randn(dimension) * drift_rate
            
            # Generate batch around drifted center
            batch = current_center + np.random.randn(stream_rate, dimension)
            stream_batches.append(batch.astype(np.float32))
        
        return {
            "initial": initial_vectors,
            "stream_batches": stream_batches,
            "dimension": dimension,
            "stream_rate": stream_rate,
            "drift_rate": drift_rate
        }


class QueryWorkloadGenerator:
    """Generate different types of query workloads."""
    
    @staticmethod
    def generate_uniform_workload(num_queries: int, 
                                 dimension: int,
                                 seed: Optional[int] = None) -> np.ndarray:
        """Generate uniformly distributed queries."""
        if seed:
            np.random.seed(seed)
        
        return np.random.uniform(-1, 1, (num_queries, dimension)).astype(np.float32)
    
    @staticmethod
    def generate_hotspot_workload(num_queries: int,
                                 dimension: int,
                                 num_hotspots: int = 5,
                                 hotspot_ratio: float = 0.8) -> np.ndarray:
        """Generate workload with hotspots (skewed distribution)."""
        queries = []
        
        # Generate hotspot centers
        hotspots = np.random.randn(num_hotspots, dimension) * 10
        
        num_hot_queries = int(num_queries * hotspot_ratio)
        num_cold_queries = num_queries - num_hot_queries
        
        # Hot queries - concentrated around hotspots
        for _ in range(num_hot_queries):
            hotspot_idx = random.randint(0, num_hotspots - 1)
            query = hotspots[hotspot_idx] + np.random.randn(dimension) * 0.1
            queries.append(query)
        
        # Cold queries - random distribution
        cold_queries = np.random.randn(num_cold_queries, dimension) * 5
        queries.extend(cold_queries)
        
        # Shuffle to mix hot and cold queries
        queries = np.array(queries)
        np.random.shuffle(queries)
        
        return queries.astype(np.float32)
    
    @staticmethod
    def generate_temporal_workload(num_queries: int,
                                  dimension: int,
                                  time_windows: int = 10) -> List[np.ndarray]:
        """Generate workload that changes over time."""
        queries_per_window = num_queries // time_windows
        workload_windows = []
        
        # Start with initial distribution center
        current_center = np.zeros(dimension)
        
        for window in range(time_windows):
            # Drift the center over time
            drift = np.random.randn(dimension) * 0.5
            current_center += drift
            
            # Generate queries for this time window
            window_queries = current_center + np.random.randn(queries_per_window, dimension)
            workload_windows.append(window_queries.astype(np.float32))
        
        return workload_windows
    
    @staticmethod
    def generate_mixed_workload(num_queries: int,
                              dimension: int) -> Dict[str, np.ndarray]:
        """Generate mixed workload with different query types."""
        workload = {}
        
        # Point queries (exact vectors)
        num_point = num_queries // 4
        workload["point"] = np.random.randn(num_point, dimension).astype(np.float32)
        
        # Range queries (represented as center + radius)
        num_range = num_queries // 4
        range_centers = np.random.randn(num_range, dimension).astype(np.float32)
        range_radii = np.random.uniform(0.1, 2.0, num_range).astype(np.float32)
        workload["range"] = {"centers": range_centers, "radii": range_radii}
        
        # KNN queries (standard similarity search)
        num_knn = num_queries // 4
        workload["knn"] = np.random.randn(num_knn, dimension).astype(np.float32)
        
        # Filtered queries (queries with metadata filters)
        num_filtered = num_queries - num_point - num_range - num_knn
        filtered_queries = np.random.randn(num_filtered, dimension).astype(np.float32)
        filters = [{"category": random.choice(["A", "B", "C"])} for _ in range(num_filtered)]
        workload["filtered"] = {"queries": filtered_queries, "filters": filters}
        
        return workload


class MetricDataGenerator:
    """Generate realistic metric data for testing."""
    
    @staticmethod
    def generate_latency_distribution(num_samples: int = 1000,
                                     distribution: str = "lognormal",
                                     mean: float = 10,
                                     std: float = 5) -> np.ndarray:
        """Generate realistic latency distribution."""
        if distribution == "lognormal":
            # Log-normal distribution (common for latencies)
            log_mean = np.log(mean / np.sqrt(1 + (std / mean) ** 2))
            log_std = np.sqrt(np.log(1 + (std / mean) ** 2))
            latencies = np.random.lognormal(log_mean, log_std, num_samples)
            
        elif distribution == "exponential":
            # Exponential distribution
            latencies = np.random.exponential(mean, num_samples)
            
        elif distribution == "gamma":
            # Gamma distribution
            shape = (mean / std) ** 2
            scale = std ** 2 / mean
            latencies = np.random.gamma(shape, scale, num_samples)
            
        else:
            # Normal distribution (less realistic for latencies)
            latencies = np.random.normal(mean, std, num_samples)
            latencies = np.maximum(latencies, 0.1)  # Ensure positive
        
        return latencies.astype(np.float32)
    
    @staticmethod
    def generate_throughput_series(duration: int = 3600,  # 1 hour in seconds
                                  base_qps: float = 1000,
                                  pattern: str = "steady") -> List[Tuple[float, float]]:
        """Generate time series of throughput measurements."""
        series = []
        
        if pattern == "steady":
            for t in range(duration):
                qps = base_qps + np.random.normal(0, base_qps * 0.05)
                series.append((t, max(0, qps)))
                
        elif pattern == "diurnal":
            # Simulate daily pattern
            for t in range(duration):
                # Use sine wave for daily pattern
                hour = (t / 3600) % 24
                multiplier = 0.5 + 0.5 * np.sin(2 * np.pi * (hour - 6) / 24)
                qps = base_qps * multiplier + np.random.normal(0, base_qps * 0.05)
                series.append((t, max(0, qps)))
                
        elif pattern == "spike":
            # Occasional spikes
            for t in range(duration):
                if random.random() < 0.01:  # 1% chance of spike
                    qps = base_qps * random.uniform(2, 5)
                else:
                    qps = base_qps + np.random.normal(0, base_qps * 0.05)
                series.append((t, max(0, qps)))
                
        elif pattern == "degrading":
            # Performance degradation over time
            for t in range(duration):
                degradation = 1 - (t / duration) * 0.5  # 50% degradation
                qps = base_qps * degradation + np.random.normal(0, base_qps * 0.05)
                series.append((t, max(0, qps)))
        
        return series
