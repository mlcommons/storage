"""
Unit tests for benchmarking functionality in vdb-bench
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
import time
import multiprocessing as mp
from typing import List, Dict, Any
import statistics
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


class TestBenchmarkExecution:
    """Test benchmark execution and query operations."""
    
    def test_single_query_execution(self, mock_collection):
        """Test executing a single query."""
        # Mock search result
        mock_collection.search.return_value = [[
            Mock(id=1, distance=0.1),
            Mock(id=2, distance=0.2),
            Mock(id=3, distance=0.3)
        ]]
        
        def execute_single_query(collection, query_vector, top_k=10):
            """Execute a single vector search query."""
            start_time = time.time()
            
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            return {
                "latency": latency,
                "num_results": len(results[0]) if results else 0,
                "top_result": results[0][0].id if results and results[0] else None
            }
        
        query = np.random.randn(128).astype(np.float32)
        result = execute_single_query(mock_collection, query)
        
        assert result["latency"] >= 0
        assert result["num_results"] == 3
        assert result["top_result"] == 1
        mock_collection.search.assert_called_once()
    
    def test_batch_query_execution(self, mock_collection):
        """Test executing batch queries."""
        # Mock batch search results
        mock_results = [
            [Mock(id=i, distance=0.1*i) for i in range(1, 6)]
            for _ in range(10)
        ]
        mock_collection.search.return_value = mock_results
        
        def execute_batch_queries(collection, query_vectors, top_k=10):
            """Execute batch vector search queries."""
            start_time = time.time()
            
            results = collection.search(
                data=query_vectors,
                anns_field="embedding",
                param={"metric_type": "L2"},
                limit=top_k
            )
            
            end_time = time.time()
            total_latency = end_time - start_time
            
            return {
                "total_latency": total_latency,
                "queries_per_second": len(query_vectors) / total_latency if total_latency > 0 else 0,
                "num_queries": len(query_vectors),
                "results_per_query": [len(r) for r in results]
            }
        
        queries = np.random.randn(10, 128).astype(np.float32)
        result = execute_batch_queries(mock_collection, queries)
        
        assert result["num_queries"] == 10
        assert len(result["results_per_query"]) == 10
        assert all(r == 5 for r in result["results_per_query"])
    
    @patch('time.time')
    def test_throughput_measurement(self, mock_time, mock_collection):
        """Test measuring query throughput."""
        # Simulate time progression
        time_counter = [0]
        def time_side_effect():
            time_counter[0] += 0.001  # 1ms per call
            return time_counter[0]
        
        mock_time.side_effect = time_side_effect
        mock_collection.search.return_value = [[Mock(id=1, distance=0.1)]]
        
        class ThroughputBenchmark:
            def __init__(self):
                self.results = []
            
            def run(self, collection, queries, duration=10):
                """Run throughput benchmark for specified duration."""
                start_time = time.time()
                end_time = start_time + duration
                query_count = 0
                latencies = []
                
                query_idx = 0
                while time.time() < end_time:
                    query_start = time.time()
                    
                    # Execute query
                    collection.search(
                        data=[queries[query_idx % len(queries)]],
                        anns_field="embedding",
                        param={"metric_type": "L2"},
                        limit=10
                    )
                    
                    query_end = time.time()
                    latencies.append(query_end - query_start)
                    query_count += 1
                    query_idx += 1
                    
                    # Break if we've done enough queries for the test
                    if query_count >= 100:  # Limit for testing
                        break
                
                actual_duration = time.time() - start_time
                
                return {
                    "total_queries": query_count,
                    "duration": actual_duration,
                    "qps": query_count / actual_duration if actual_duration > 0 else 0,
                    "avg_latency": statistics.mean(latencies) if latencies else 0,
                    "p50_latency": statistics.median(latencies) if latencies else 0,
                    "p95_latency": self._percentile(latencies, 95) if latencies else 0,
                    "p99_latency": self._percentile(latencies, 99) if latencies else 0
                }
            
            def _percentile(self, data, percentile):
                """Calculate percentile of data."""
                size = len(data)
                if size == 0:
                    return 0
                sorted_data = sorted(data)
                index = int(size * percentile / 100)
                return sorted_data[min(index, size - 1)]
        
        benchmark = ThroughputBenchmark()
        queries = np.random.randn(10, 128).astype(np.float32)
        
        result = benchmark.run(mock_collection, queries, duration=1)
        
        assert result["total_queries"] > 0
        assert result["qps"] > 0
        assert result["avg_latency"] > 0
    
    def test_concurrent_query_execution(self, mock_collection):
        """Test concurrent query execution with multiple threads."""
        query_counter = {'count': 0}
        
        def mock_search(data, **kwargs):
            query_counter['count'] += 1
            time.sleep(0.01)  # Simulate query time
            return [[Mock(id=i, distance=0.1*i) for i in range(5)]]
        
        mock_collection.search = mock_search
        
        class ConcurrentBenchmark:
            def __init__(self, num_threads=4):
                self.num_threads = num_threads
            
            def worker(self, args):
                """Worker function for concurrent execution."""
                collection, queries, worker_id = args
                results = []
                
                for i, query in enumerate(queries):
                    start = time.time()
                    result = collection.search(
                        data=[query],
                        anns_field="embedding",
                        param={"metric_type": "L2"},
                        limit=10
                    )
                    latency = time.time() - start
                    results.append({
                        "worker_id": worker_id,
                        "query_id": i,
                        "latency": latency
                    })
                
                return results
            
            def run(self, collection, queries):
                """Run concurrent benchmark."""
                # Split queries among workers
                queries_per_worker = len(queries) // self.num_threads
                worker_args = []
                
                for i in range(self.num_threads):
                    start_idx = i * queries_per_worker
                    end_idx = start_idx + queries_per_worker if i < self.num_threads - 1 else len(queries)
                    worker_queries = queries[start_idx:end_idx]
                    worker_args.append((collection, worker_queries, i))
                
                start_time = time.time()
                
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    results = list(executor.map(self.worker, worker_args))
                
                end_time = time.time()
                
                # Flatten results
                all_results = []
                for worker_results in results:
                    all_results.extend(worker_results)
                
                total_duration = end_time - start_time
                latencies = [r["latency"] for r in all_results]
                
                return {
                    "num_threads": self.num_threads,
                    "total_queries": len(all_results),
                    "duration": total_duration,
                    "qps": len(all_results) / total_duration if total_duration > 0 else 0,
                    "avg_latency": statistics.mean(latencies) if latencies else 0,
                    "min_latency": min(latencies) if latencies else 0,
                    "max_latency": max(latencies) if latencies else 0
                }
        
        benchmark = ConcurrentBenchmark(num_threads=4)
        queries = np.random.randn(100, 128).astype(np.float32)
        
        result = benchmark.run(mock_collection, queries)
        
        assert result["total_queries"] == 100
        assert result["num_threads"] == 4
        assert result["qps"] > 0
        assert query_counter['count'] == 100


class TestBenchmarkMetrics:
    """Test benchmark metric collection and analysis."""
    
    def test_latency_distribution(self):
        """Test calculating latency distribution metrics."""
        class LatencyAnalyzer:
            def __init__(self):
                self.latencies = []
            
            def add_latency(self, latency):
                """Add a latency measurement."""
                self.latencies.append(latency)
            
            def get_distribution(self):
                """Calculate latency distribution statistics."""
                if not self.latencies:
                    return {}
                
                sorted_latencies = sorted(self.latencies)
                
                return {
                    "count": len(self.latencies),
                    "mean": statistics.mean(self.latencies),
                    "median": statistics.median(self.latencies),
                    "stdev": statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0,
                    "min": min(self.latencies),
                    "max": max(self.latencies),
                    "p50": self._percentile(sorted_latencies, 50),
                    "p90": self._percentile(sorted_latencies, 90),
                    "p95": self._percentile(sorted_latencies, 95),
                    "p99": self._percentile(sorted_latencies, 99),
                    "p999": self._percentile(sorted_latencies, 99.9)
                }
            
            def _percentile(self, sorted_data, percentile):
                """Calculate percentile from sorted data."""
                index = len(sorted_data) * percentile / 100
                lower = int(index)
                upper = lower + 1
                
                if upper >= len(sorted_data):
                    return sorted_data[-1]
                
                weight = index - lower
                return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
        
        analyzer = LatencyAnalyzer()
        
        # Add sample latencies (in milliseconds)
        np.random.seed(42)
        latencies = np.random.exponential(10, 1000)  # Exponential distribution
        for latency in latencies:
            analyzer.add_latency(latency)
        
        dist = analyzer.get_distribution()
        
        assert dist["count"] == 1000
        assert dist["p50"] < dist["p90"]
        assert dist["p90"] < dist["p95"]
        assert dist["p95"] < dist["p99"]
        assert dist["min"] < dist["mean"] < dist["max"]
    
    def test_recall_metric(self):
        """Test calculating recall metrics for search results."""
        class RecallCalculator:
            def __init__(self, ground_truth):
                self.ground_truth = ground_truth
            
            def calculate_recall(self, query_id, retrieved_ids, k):
                """Calculate recall@k for a query."""
                if query_id not in self.ground_truth:
                    return None
                
                true_ids = set(self.ground_truth[query_id][:k])
                retrieved_ids_set = set(retrieved_ids[:k])
                
                intersection = true_ids.intersection(retrieved_ids_set)
                recall = len(intersection) / len(true_ids) if true_ids else 0
                
                return recall
            
            def calculate_average_recall(self, results, k):
                """Calculate average recall@k across multiple queries."""
                recalls = []
                
                for query_id, retrieved_ids in results.items():
                    recall = self.calculate_recall(query_id, retrieved_ids, k)
                    if recall is not None:
                        recalls.append(recall)
                
                return statistics.mean(recalls) if recalls else 0
        
        # Mock ground truth data
        ground_truth = {
            0: [1, 2, 3, 4, 5],
            1: [6, 7, 8, 9, 10],
            2: [11, 12, 13, 14, 15]
        }
        
        calculator = RecallCalculator(ground_truth)
        
        # Test perfect recall
        perfect_results = {
            0: [1, 2, 3, 4, 5],
            1: [6, 7, 8, 9, 10],
            2: [11, 12, 13, 14, 15]
        }
        
        avg_recall = calculator.calculate_average_recall(perfect_results, k=5)
        assert avg_recall == 1.0
        
        # Test partial recall
        partial_results = {
            0: [1, 2, 3, 16, 17],  # 3/5 correct
            1: [6, 7, 18, 19, 20],  # 2/5 correct
            2: [11, 12, 13, 14, 21]  # 4/5 correct
        }
        
        avg_recall = calculator.calculate_average_recall(partial_results, k=5)
        assert 0.5 < avg_recall < 0.7  # Should be (3+2+4)/15 = 0.6
    
    def test_benchmark_summary_generation(self):
        """Test generating comprehensive benchmark summary."""
        class BenchmarkSummary:
            def __init__(self):
                self.metrics = {
                    "latencies": [],
                    "throughputs": [],
                    "errors": 0,
                    "total_queries": 0
                }
                self.start_time = None
                self.end_time = None
            
            def start(self):
                """Start benchmark timing."""
                self.start_time = time.time()
            
            def end(self):
                """End benchmark timing."""
                self.end_time = time.time()
            
            def add_query_result(self, latency, success=True):
                """Add a query result."""
                self.metrics["total_queries"] += 1
                
                if success:
                    self.metrics["latencies"].append(latency)
                else:
                    self.metrics["errors"] += 1
            
            def add_throughput_sample(self, qps):
                """Add a throughput sample."""
                self.metrics["throughputs"].append(qps)
            
            def generate_summary(self):
                """Generate comprehensive benchmark summary."""
                if not self.start_time or not self.end_time:
                    return None
                
                duration = self.end_time - self.start_time
                latencies = self.metrics["latencies"]
                
                summary = {
                    "duration": duration,
                    "total_queries": self.metrics["total_queries"],
                    "successful_queries": len(latencies),
                    "failed_queries": self.metrics["errors"],
                    "error_rate": self.metrics["errors"] / self.metrics["total_queries"] 
                                  if self.metrics["total_queries"] > 0 else 0
                }
                
                if latencies:
                    summary.update({
                        "latency_mean": statistics.mean(latencies),
                        "latency_median": statistics.median(latencies),
                        "latency_min": min(latencies),
                        "latency_max": max(latencies),
                        "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)],
                        "latency_p99": sorted(latencies)[int(len(latencies) * 0.99)]
                    })
                
                if self.metrics["throughputs"]:
                    summary.update({
                        "throughput_mean": statistics.mean(self.metrics["throughputs"]),
                        "throughput_max": max(self.metrics["throughputs"]),
                        "throughput_min": min(self.metrics["throughputs"])
                    })
                
                # Overall QPS
                summary["overall_qps"] = self.metrics["total_queries"] / duration if duration > 0 else 0
                
                return summary
        
        summary = BenchmarkSummary()
        summary.start()
        
        # Simulate query results
        np.random.seed(42)
        for i in range(1000):
            latency = np.random.exponential(10)  # 10ms average
            success = np.random.random() > 0.01  # 99% success rate
            summary.add_query_result(latency, success)
        
        # Add throughput samples
        for i in range(10):
            summary.add_throughput_sample(100 + np.random.normal(0, 10))
        
        time.sleep(0.1)  # Simulate benchmark duration
        summary.end()
        
        result = summary.generate_summary()
        
        assert result["total_queries"] == 1000
        assert result["error_rate"] < 0.02  # Should be around 1%
        assert result["latency_p99"] > result["latency_p95"]
        assert result["latency_p95"] > result["latency_median"]


class TestBenchmarkConfiguration:
    """Test benchmark configuration and parameter tuning."""
    
    def test_search_parameter_tuning(self):
        """Test tuning search parameters for optimal performance."""
        class SearchParameterTuner:
            def __init__(self, collection):
                self.collection = collection
                self.results = []
            
            def test_parameters(self, params, test_queries):
                """Test a set of search parameters."""
                latencies = []
                
                for query in test_queries:
                    start = time.time()
                    self.collection.search(
                        data=[query],
                        anns_field="embedding",
                        param=params,
                        limit=10
                    )
                    latencies.append(time.time() - start)
                
                return {
                    "params": params,
                    "avg_latency": statistics.mean(latencies),
                    "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)]
                }
            
            def tune(self, parameter_sets, test_queries):
                """Find optimal parameters."""
                for params in parameter_sets:
                    result = self.test_parameters(params, test_queries)
                    self.results.append(result)
                
                # Find best parameters based on latency
                best = min(self.results, key=lambda x: x["avg_latency"])
                return best
        
        mock_collection = Mock()
        mock_collection.search.return_value = [[Mock(id=1, distance=0.1)]]
        
        tuner = SearchParameterTuner(mock_collection)
        
        # Define parameter sets to test
        parameter_sets = [
            {"metric_type": "L2", "params": {"nprobe": 10}},
            {"metric_type": "L2", "params": {"nprobe": 20}},
            {"metric_type": "L2", "params": {"nprobe": 50}},
        ]
        
        test_queries = np.random.randn(10, 128).astype(np.float32)
        
        best_params = tuner.tune(parameter_sets, test_queries)
        
        assert best_params is not None
        assert "params" in best_params
        assert "avg_latency" in best_params
    
    def test_workload_generation(self):
        """Test generating different query workloads."""
        class WorkloadGenerator:
            def __init__(self, dimension, seed=None):
                self.dimension = dimension
                if seed:
                    np.random.seed(seed)
            
            def generate_uniform(self, num_queries):
                """Generate uniformly distributed queries."""
                return np.random.uniform(-1, 1, (num_queries, self.dimension)).astype(np.float32)
            
            def generate_gaussian(self, num_queries, centers=1):
                """Generate queries from Gaussian distributions."""
                if centers == 1:
                    return np.random.randn(num_queries, self.dimension).astype(np.float32)
                
                # Multiple centers
                queries_per_center = num_queries // centers
                remainder = num_queries % centers
                queries = []
                
                for i in range(centers):
                    center = np.random.randn(self.dimension) * 10
                    # Add extra query to first clusters if there's a remainder
                    extra = 1 if i < remainder else 0
                    cluster = np.random.randn(queries_per_center + extra, self.dimension) + center
                    queries.append(cluster)
                
                return np.vstack(queries).astype(np.float32)
            
            def generate_skewed(self, num_queries, hot_ratio=0.2):
                """Generate skewed workload with hot and cold queries."""
                num_hot = int(num_queries * hot_ratio)
                num_cold = num_queries - num_hot
                
                # Hot queries - concentrated around a few points
                hot_queries = np.random.randn(num_hot, self.dimension) * 0.1
                
                # Cold queries - widely distributed
                cold_queries = np.random.randn(num_cold, self.dimension) * 10
                
                # Mix them
                all_queries = np.vstack([hot_queries, cold_queries])
                np.random.shuffle(all_queries)
                
                return all_queries.astype(np.float32)
            
            def generate_temporal(self, num_queries, drift_rate=0.01):
                """Generate queries with temporal drift."""
                queries = []
                current_center = np.zeros(self.dimension)
                
                for i in range(num_queries):
                    # Drift the center
                    current_center += np.random.randn(self.dimension) * drift_rate
                    
                    # Generate query around current center
                    query = current_center + np.random.randn(self.dimension)
                    queries.append(query)
                
                return np.array(queries).astype(np.float32)
        
        generator = WorkloadGenerator(dimension=128, seed=42)
        
        # Test uniform workload
        uniform = generator.generate_uniform(100)
        assert uniform.shape == (100, 128)
        assert uniform.min() >= -1.1  # Small tolerance
        assert uniform.max() <= 1.1
        
        # Test Gaussian workload
        gaussian = generator.generate_gaussian(100, centers=3)
        assert gaussian.shape == (100, 128)
        
        # Test skewed workload
        skewed = generator.generate_skewed(100, hot_ratio=0.2)
        assert skewed.shape == (100, 128)
        
        # Test temporal workload
        temporal = generator.generate_temporal(100, drift_rate=0.01)
        assert temporal.shape == (100, 128)


class TestBenchmarkOutput:
    """Test benchmark result output and reporting."""
    
    def test_json_output_format(self, test_data_dir):
        """Test outputting benchmark results in JSON format."""
        results = {
            "timestamp": "2024-01-01T12:00:00",
            "configuration": {
                "collection": "test_collection",
                "dimension": 1536,
                "index_type": "DISKANN",
                "num_processes": 4,
                "batch_size": 100
            },
            "metrics": {
                "total_queries": 10000,
                "duration": 60.5,
                "qps": 165.29,
                "latency_p50": 5.2,
                "latency_p95": 12.8,
                "latency_p99": 18.3,
                "error_rate": 0.001
            },
            "system_info": {
                "cpu_count": 8,
                "memory_gb": 32,
                "platform": "Linux"
            }
        }
        
        output_file = test_data_dir / "benchmark_results.json"
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Verify saved file
        with open(output_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded["metrics"]["qps"] == 165.29
        assert loaded["configuration"]["index_type"] == "DISKANN"
    
    def test_csv_output_format(self, test_data_dir):
        """Test outputting benchmark results in CSV format."""
        import csv
        
        results = [
            {"timestamp": "2024-01-01T12:00:00", "qps": 150.5, "latency_p95": 12.3},
            {"timestamp": "2024-01-01T12:01:00", "qps": 155.2, "latency_p95": 11.8},
            {"timestamp": "2024-01-01T12:02:00", "qps": 148.9, "latency_p95": 12.7}
        ]
        
        output_file = test_data_dir / "benchmark_results.csv"
        
        # Save results
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "qps", "latency_p95"])
            writer.writeheader()
            writer.writerows(results)
        
        # Verify saved file
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            loaded = list(reader)
        
        assert len(loaded) == 3
        assert float(loaded[0]["qps"]) == 150.5
    
    def test_comparison_report_generation(self):
        """Test generating comparison reports between benchmarks."""
        class ComparisonReport:
            def __init__(self):
                self.benchmarks = {}
            
            def add_benchmark(self, name, results):
                """Add benchmark results."""
                self.benchmarks[name] = results
            
            def generate_comparison(self):
                """Generate comparison report."""
                if len(self.benchmarks) < 2:
                    return None
                
                comparison = {
                    "benchmarks": [],
                    "best_qps": None,
                    "best_latency": None
                }
                
                best_qps = 0
                best_latency = float('inf')
                
                for name, results in self.benchmarks.items():
                    benchmark_summary = {
                        "name": name,
                        "qps": results.get("qps", 0),
                        "latency_p95": results.get("latency_p95", 0),
                        "latency_p99": results.get("latency_p99", 0),
                        "error_rate": results.get("error_rate", 0)
                    }
                    
                    comparison["benchmarks"].append(benchmark_summary)
                    
                    if benchmark_summary["qps"] > best_qps:
                        best_qps = benchmark_summary["qps"]
                        comparison["best_qps"] = name
                    
                    if benchmark_summary["latency_p95"] < best_latency:
                        best_latency = benchmark_summary["latency_p95"]
                        comparison["best_latency"] = name
                
                # Calculate improvements
                if len(self.benchmarks) == 2:
                    names = list(self.benchmarks.keys())
                    baseline = self.benchmarks[names[0]]
                    comparison_bench = self.benchmarks[names[1]]
                    
                    comparison["qps_improvement"] = (
                        (comparison_bench["qps"] - baseline["qps"]) / baseline["qps"] * 100
                        if baseline.get("qps", 0) > 0 else 0
                    )
                    
                    comparison["latency_improvement"] = (
                        (baseline["latency_p95"] - comparison_bench["latency_p95"]) / baseline["latency_p95"] * 100
                        if baseline.get("latency_p95", 0) > 0 else 0
                    )
                
                return comparison
        
        report = ComparisonReport()
        
        # Add benchmark results
        report.add_benchmark("DISKANN", {
            "qps": 1500,
            "latency_p95": 10.5,
            "latency_p99": 15.2,
            "error_rate": 0.001
        })
        
        report.add_benchmark("HNSW", {
            "qps": 1200,
            "latency_p95": 8.3,
            "latency_p99": 12.1,
            "error_rate": 0.002
        })
        
        comparison = report.generate_comparison()
        
        assert comparison["best_qps"] == "DISKANN"
        assert comparison["best_latency"] == "HNSW"
        assert len(comparison["benchmarks"]) == 2
        assert comparison["qps_improvement"] == -20.0  # HNSW is 20% slower
