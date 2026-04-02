"""
Unit tests for index management functionality in vdb-bench
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
import time
import json
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor


class TestIndexCreation:
    """Test index creation operations."""
    
    def test_create_diskann_index(self, mock_collection):
        """Test creating DiskANN index."""
        mock_collection.create_index.return_value = True
        
        def create_diskann_index(collection, field_name="embedding", params=None):
            """Create DiskANN index on collection."""
            if params is None:
                params = {
                    "metric_type": "L2",
                    "index_type": "DISKANN",
                    "params": {
                        "max_degree": 64,
                        "search_list_size": 200,
                        "pq_code_budget_gb": 0.1,
                        "build_algo": "IVF_PQ"
                    }
                }
            
            try:
                result = collection.create_index(
                    field_name=field_name,
                    index_params=params
                )
                return {
                    "success": True,
                    "index_type": params["index_type"],
                    "field": field_name,
                    "params": params
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        
        result = create_diskann_index(mock_collection)
        
        assert result["success"] is True
        assert result["index_type"] == "DISKANN"
        mock_collection.create_index.assert_called_once()
    
    def test_create_hnsw_index(self, mock_collection):
        """Test creating HNSW index."""
        mock_collection.create_index.return_value = True
        
        def create_hnsw_index(collection, field_name="embedding", params=None):
            """Create HNSW index on collection."""
            if params is None:
                params = {
                    "metric_type": "L2",
                    "index_type": "HNSW",
                    "params": {
                        "M": 16,
                        "efConstruction": 200
                    }
                }
            
            try:
                result = collection.create_index(
                    field_name=field_name,
                    index_params=params
                )
                return {
                    "success": True,
                    "index_type": params["index_type"],
                    "field": field_name,
                    "params": params
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        
        result = create_hnsw_index(mock_collection)
        
        assert result["success"] is True
        assert result["index_type"] == "HNSW"
        assert result["params"]["params"]["M"] == 16
    
    def test_create_ivf_index(self, mock_collection):
        """Test creating IVF index variants."""
        class IVFIndexBuilder:
            def __init__(self, collection):
                self.collection = collection
            
            def create_ivf_flat(self, field_name, nlist=128):
                """Create IVF_FLAT index."""
                params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": nlist}
                }
                return self._create_index(field_name, params)
            
            def create_ivf_sq8(self, field_name, nlist=128):
                """Create IVF_SQ8 index."""
                params = {
                    "metric_type": "L2",
                    "index_type": "IVF_SQ8",
                    "params": {"nlist": nlist}
                }
                return self._create_index(field_name, params)
            
            def create_ivf_pq(self, field_name, nlist=128, m=8, nbits=8):
                """Create IVF_PQ index."""
                params = {
                    "metric_type": "L2",
                    "index_type": "IVF_PQ",
                    "params": {
                        "nlist": nlist,
                        "m": m,
                        "nbits": nbits
                    }
                }
                return self._create_index(field_name, params)
            
            def _create_index(self, field_name, params):
                """Internal method to create index."""
                try:
                    self.collection.create_index(
                        field_name=field_name,
                        index_params=params
                    )
                    return {"success": True, "params": params}
                except Exception as e:
                    return {"success": False, "error": str(e)}
        
        mock_collection.create_index.return_value = True
        builder = IVFIndexBuilder(mock_collection)
        
        # Test IVF_FLAT
        result = builder.create_ivf_flat("embedding", nlist=256)
        assert result["success"] is True
        assert result["params"]["index_type"] == "IVF_FLAT"
        
        # Test IVF_SQ8
        result = builder.create_ivf_sq8("embedding", nlist=512)
        assert result["success"] is True
        assert result["params"]["index_type"] == "IVF_SQ8"
        
        # Test IVF_PQ
        result = builder.create_ivf_pq("embedding", nlist=256, m=16)
        assert result["success"] is True
        assert result["params"]["index_type"] == "IVF_PQ"
        assert result["params"]["params"]["m"] == 16
    
    def test_index_creation_with_retry(self, mock_collection):
        """Test index creation with retry logic."""
        # Simulate failures then success
        mock_collection.create_index.side_effect = [
            Exception("Index creation failed"),
            Exception("Still failing"),
            True
        ]
        
        def create_index_with_retry(collection, params, max_retries=3, backoff=2):
            """Create index with exponential backoff retry."""
            for attempt in range(max_retries):
                try:
                    collection.create_index(
                        field_name="embedding",
                        index_params=params
                    )
                    return {
                        "success": True,
                        "attempts": attempt + 1
                    }
                except Exception as e:
                    if attempt == max_retries - 1:
                        return {
                            "success": False,
                            "attempts": attempt + 1,
                            "error": str(e)
                        }
                    time.sleep(backoff ** attempt)
            
            return {"success": False, "attempts": max_retries}
        
        params = {
            "metric_type": "L2",
            "index_type": "DISKANN",
            "params": {"max_degree": 64}
        }
        
        with patch('time.sleep'):  # Speed up test
            result = create_index_with_retry(mock_collection, params)
        
        assert result["success"] is True
        assert result["attempts"] == 3
        assert mock_collection.create_index.call_count == 3


class TestIndexManagement:
    """Test index management operations."""
    
    def test_index_status_check(self, mock_collection):
        """Test checking index status."""
        # Create a proper mock index object
        mock_index = Mock()
        mock_index.params = {"index_type": "DISKANN"}
        mock_index.progress = 100
        mock_index.state = "Finished"
        
        # Set the index attribute on collection
        mock_collection.index = mock_index
        
        class IndexManager:
            def __init__(self, collection):
                self.collection = collection
            
            def get_index_status(self):
                """Get current index status."""
                try:
                    index = self.collection.index
                    return {
                        "exists": True,
                        "type": index.params.get("index_type"),
                        "progress": index.progress,
                        "state": index.state,
                        "params": index.params
                    }
                except:
                    return {
                        "exists": False,
                        "type": None,
                        "progress": 0,
                        "state": "Not Created"
                    }
            
            def is_index_ready(self):
                """Check if index is ready for use."""
                status = self.get_index_status()
                return (
                    status["exists"] and 
                    status["state"] == "Finished" and
                    status["progress"] == 100
                )
            
            def wait_for_index(self, timeout=300, check_interval=5):
                """Wait for index to be ready."""
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    if self.is_index_ready():
                        return True
                    time.sleep(check_interval)
                
                return False
        
        manager = IndexManager(mock_collection)
        
        status = manager.get_index_status()
        assert status["exists"] is True
        assert status["type"] == "DISKANN"
        assert status["progress"] == 100
        
        assert manager.is_index_ready() is True
    
    def test_drop_index(self, mock_collection):
        """Test dropping an index."""
        mock_collection.drop_index.return_value = None
        
        def drop_index(collection, field_name="embedding"):
            """Drop index from collection."""
            try:
                collection.drop_index(field_name=field_name)
                return {
                    "success": True,
                    "field": field_name,
                    "message": f"Index dropped for field {field_name}"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        
        result = drop_index(mock_collection)
        
        assert result["success"] is True
        assert result["field"] == "embedding"
        mock_collection.drop_index.assert_called_once_with(field_name="embedding")
    
    def test_rebuild_index(self, mock_collection):
        """Test rebuilding an index."""
        mock_collection.drop_index.return_value = None
        mock_collection.create_index.return_value = True
        
        class IndexRebuilder:
            def __init__(self, collection):
                self.collection = collection
            
            def rebuild_index(self, field_name, new_params):
                """Rebuild index with new parameters."""
                steps = []
                
                try:
                    # Step 1: Drop existing index
                    self.collection.drop_index(field_name=field_name)
                    steps.append("Index dropped")
                    
                    # Step 2: Wait for drop to complete
                    time.sleep(1)
                    steps.append("Waited for drop completion")
                    
                    # Step 3: Create new index
                    self.collection.create_index(
                        field_name=field_name,
                        index_params=new_params
                    )
                    steps.append("New index created")
                    
                    return {
                        "success": True,
                        "steps": steps,
                        "new_params": new_params
                    }
                
                except Exception as e:
                    return {
                        "success": False,
                        "steps": steps,
                        "error": str(e)
                    }
        
        rebuilder = IndexRebuilder(mock_collection)
        
        new_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 32, "efConstruction": 400}
        }
        
        with patch('time.sleep'):  # Speed up test
            result = rebuilder.rebuild_index("embedding", new_params)
        
        assert result["success"] is True
        assert len(result["steps"]) == 3
        assert mock_collection.drop_index.called
        assert mock_collection.create_index.called
    
    def test_index_comparison(self):
        """Test comparing different index configurations."""
        class IndexComparator:
            def __init__(self):
                self.results = {}
            
            def add_result(self, index_type, metrics):
                """Add benchmark result for an index type."""
                self.results[index_type] = metrics
            
            def compare(self):
                """Compare all index results."""
                if len(self.results) < 2:
                    return None
                
                comparison = {
                    "indexes": [],
                    "best_qps": None,
                    "best_recall": None,
                    "best_build_time": None
                }
                
                best_qps = 0
                best_recall = 0
                best_build_time = float('inf')
                
                for index_type, metrics in self.results.items():
                    comparison["indexes"].append({
                        "type": index_type,
                        "qps": metrics.get("qps", 0),
                        "recall": metrics.get("recall", 0),
                        "build_time": metrics.get("build_time", 0),
                        "memory_usage": metrics.get("memory_usage", 0)
                    })
                    
                    if metrics.get("qps", 0) > best_qps:
                        best_qps = metrics["qps"]
                        comparison["best_qps"] = index_type
                    
                    if metrics.get("recall", 0) > best_recall:
                        best_recall = metrics["recall"]
                        comparison["best_recall"] = index_type
                    
                    if metrics.get("build_time", float('inf')) < best_build_time:
                        best_build_time = metrics["build_time"]
                        comparison["best_build_time"] = index_type
                
                return comparison
            
            def get_recommendation(self, requirements):
                """Get index recommendation based on requirements."""
                if not self.results:
                    return None
                
                scores = {}
                
                for index_type, metrics in self.results.items():
                    score = 0
                    
                    # Weight different factors based on requirements
                    if requirements.get("prioritize_speed"):
                        score += metrics.get("qps", 0) * 2
                    
                    if requirements.get("prioritize_accuracy"):
                        score += metrics.get("recall", 0) * 1000
                    
                    if requirements.get("memory_constrained"):
                        # Penalize high memory usage
                        score -= metrics.get("memory_usage", 0) * 0.1
                    
                    if requirements.get("fast_build"):
                        # Penalize slow build time
                        score -= metrics.get("build_time", 0) * 10
                    
                    scores[index_type] = score
                
                best_index = max(scores, key=scores.get)
                
                return {
                    "recommended": best_index,
                    "score": scores[best_index],
                    "all_scores": scores
                }
        
        comparator = IndexComparator()
        
        # Add sample results
        comparator.add_result("DISKANN", {
            "qps": 1500,
            "recall": 0.95,
            "build_time": 300,
            "memory_usage": 2048
        })
        
        comparator.add_result("HNSW", {
            "qps": 1200,
            "recall": 0.98,
            "build_time": 150,
            "memory_usage": 4096
        })
        
        comparator.add_result("IVF_PQ", {
            "qps": 2000,
            "recall": 0.90,
            "build_time": 100,
            "memory_usage": 1024
        })
        
        comparison = comparator.compare()
        
        assert comparison["best_qps"] == "IVF_PQ"
        assert comparison["best_recall"] == "HNSW"
        assert comparison["best_build_time"] == "IVF_PQ"
        
        # Test recommendation
        requirements = {
            "prioritize_accuracy": True,
            "memory_constrained": False
        }
        
        recommendation = comparator.get_recommendation(requirements)
        assert recommendation["recommended"] == "HNSW"


class TestIndexOptimization:
    """Test index optimization strategies."""
    
    def test_parameter_tuning(self, mock_collection):
        """Test automatic parameter tuning for indexes."""
        class ParameterTuner:
            def __init__(self, collection):
                self.collection = collection
                self.test_results = []
            
            def tune_diskann(self, test_vectors, ground_truth):
                """Tune DiskANN parameters."""
                param_grid = [
                    {"max_degree": 32, "search_list_size": 100},
                    {"max_degree": 64, "search_list_size": 200},
                    {"max_degree": 96, "search_list_size": 300}
                ]
                
                best_params = None
                best_score = 0
                
                for params in param_grid:
                    score = self._test_params(
                        "DISKANN",
                        params,
                        test_vectors,
                        ground_truth
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                    
                    self.test_results.append({
                        "params": params,
                        "score": score
                    })
                
                return best_params, best_score
            
            def tune_hnsw(self, test_vectors, ground_truth):
                """Tune HNSW parameters."""
                param_grid = [
                    {"M": 8, "efConstruction": 100},
                    {"M": 16, "efConstruction": 200},
                    {"M": 32, "efConstruction": 400}
                ]
                
                best_params = None
                best_score = 0
                
                for params in param_grid:
                    score = self._test_params(
                        "HNSW",
                        params,
                        test_vectors,
                        ground_truth
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                    
                    self.test_results.append({
                        "params": params,
                        "score": score
                    })
                
                return best_params, best_score
            
            def _test_params(self, index_type, params, test_vectors, ground_truth):
                """Test specific parameters and return score."""
                # Simulated testing (in reality would rebuild index and test)
                # Score based on parameter values (simplified)
                
                if index_type == "DISKANN":
                    score = params["max_degree"] * 0.5 + params["search_list_size"] * 0.2
                elif index_type == "HNSW":
                    score = params["M"] * 2 + params["efConstruction"] * 0.1
                else:
                    score = 0
                
                # Add some randomness
                score += np.random.random() * 10
                
                return score
        
        tuner = ParameterTuner(mock_collection)
        
        # Create test data
        test_vectors = np.random.randn(100, 128).astype(np.float32)
        ground_truth = np.random.randint(0, 1000, (100, 10))
        
        # Tune DiskANN
        best_diskann, diskann_score = tuner.tune_diskann(test_vectors, ground_truth)
        assert best_diskann is not None
        assert diskann_score > 0
        
        # Tune HNSW  
        best_hnsw, hnsw_score = tuner.tune_hnsw(test_vectors, ground_truth)
        assert best_hnsw is not None
        assert hnsw_score > 0
        
        # Check that results were recorded
        assert len(tuner.test_results) == 6  # 3 for each index type
    
    def test_adaptive_index_selection(self):
        """Test adaptive index selection based on workload."""
        class AdaptiveIndexSelector:
            def __init__(self):
                self.workload_history = []
                self.current_index = None
            
            def analyze_workload(self, queries):
                """Analyze query workload characteristics."""
                characteristics = {
                    "query_count": len(queries),
                    "dimension": queries.shape[1] if len(queries) > 0 else 0,
                    "distribution": self._analyze_distribution(queries),
                    "sparsity": self._calculate_sparsity(queries),
                    "clustering": self._analyze_clustering(queries)
                }
                
                self.workload_history.append({
                    "timestamp": time.time(),
                    "characteristics": characteristics
                })
                
                return characteristics
            
            def select_index(self, characteristics, dataset_size):
                """Select best index for workload characteristics."""
                # Simple rule-based selection
                
                if dataset_size < 100000:
                    # Small dataset - use simple index
                    return "IVF_FLAT"
                
                elif dataset_size < 1000000:
                    # Medium dataset
                    if characteristics["clustering"] > 0.7:
                        # Highly clustered - IVF works well
                        return "IVF_PQ"
                    else:
                        # More uniform - HNSW
                        return "HNSW"
                
                else:
                    # Large dataset
                    if characteristics["sparsity"] > 0.5:
                        # Sparse vectors - specialized index
                        return "SPARSE_IVF"
                    elif characteristics["dimension"] > 1000:
                        # High dimension - DiskANN with PQ
                        return "DISKANN"
                    else:
                        # Default to HNSW for good all-around performance
                        return "HNSW"
            
            def _analyze_distribution(self, queries):
                """Analyze query distribution."""
                if len(queries) == 0:
                    return "unknown"
                
                # Simple variance check
                variance = np.var(queries)
                if variance < 0.5:
                    return "concentrated"
                elif variance < 2.0:
                    return "normal"
                else:
                    return "scattered"
            
            def _calculate_sparsity(self, queries):
                """Calculate sparsity of queries."""
                if len(queries) == 0:
                    return 0
                
                zero_count = np.sum(queries == 0)
                total_elements = queries.size
                
                return zero_count / total_elements if total_elements > 0 else 0
            
            def _analyze_clustering(self, queries):
                """Analyze clustering tendency."""
                # Simplified clustering score
                if len(queries) < 10:
                    return 0
                
                # Calculate pairwise distances for small sample
                sample = queries[:min(100, len(queries))]
                distances = []
                
                for i in range(len(sample)):
                    for j in range(i + 1, len(sample)):
                        dist = np.linalg.norm(sample[i] - sample[j])
                        distances.append(dist)
                
                if not distances:
                    return 0
                
                # High variance in distances indicates clustering
                distance_var = np.var(distances)
                return min(distance_var / 10, 1.0)  # Normalize to [0, 1]
        
        selector = AdaptiveIndexSelector()
        
        # Test with different workloads
        
        # Sparse workload
        sparse_queries = np.random.randn(100, 2000).astype(np.float32)
        sparse_queries[sparse_queries < 1] = 0  # Make sparse
        
        characteristics = selector.analyze_workload(sparse_queries)
        selected_index = selector.select_index(characteristics, 5000000)
        
        assert characteristics["sparsity"] > 0.3
        
        # Dense clustered workload
        clustered_queries = []
        for _ in range(5):
            center = np.random.randn(128) * 10
            cluster = center + np.random.randn(20, 128) * 0.1
            clustered_queries.append(cluster)
        clustered_queries = np.vstack(clustered_queries).astype(np.float32)
        
        characteristics = selector.analyze_workload(clustered_queries)
        selected_index = selector.select_index(characteristics, 500000)
        
        assert selected_index in ["IVF_PQ", "HNSW"]
    
    def test_index_warm_up(self, mock_collection):
        """Test index warm-up procedures."""
        class IndexWarmUp:
            def __init__(self, collection):
                self.collection = collection
                self.warm_up_stats = []
            
            def warm_up(self, num_queries=100, batch_size=10):
                """Warm up index with sample queries."""
                total_time = 0
                queries_executed = 0
                
                for batch in range(0, num_queries, batch_size):
                    # Generate random queries
                    batch_queries = np.random.randn(
                        min(batch_size, num_queries - batch),
                        128
                    ).astype(np.float32)
                    
                    start = time.time()
                    
                    # Execute warm-up queries
                    self.collection.search(
                        data=batch_queries.tolist(),
                        anns_field="embedding",
                        param={"metric_type": "L2"},
                        limit=10
                    )
                    
                    elapsed = time.time() - start
                    total_time += elapsed
                    queries_executed += len(batch_queries)
                    
                    self.warm_up_stats.append({
                        "batch": batch // batch_size,
                        "queries": len(batch_queries),
                        "time": elapsed,
                        "qps": len(batch_queries) / elapsed if elapsed > 0 else 0
                    })
                
                return {
                    "total_queries": queries_executed,
                    "total_time": total_time,
                    "avg_qps": queries_executed / total_time if total_time > 0 else 0,
                    "stats": self.warm_up_stats
                }
            
            def adaptive_warm_up(self, target_qps=100, max_queries=1000):
                """Adaptive warm-up that stops when performance stabilizes."""
                stable_threshold = 0.1  # 10% variation
                window_size = 5
                recent_qps = []
                
                batch_size = 10
                total_queries = 0
                
                while total_queries < max_queries:
                    queries = np.random.randn(batch_size, 128).astype(np.float32)
                    
                    start = time.time()
                    self.collection.search(
                        data=queries.tolist(),
                        anns_field="embedding",
                        param={"metric_type": "L2"},
                        limit=10
                    )
                    elapsed = time.time() - start
                    
                    qps = batch_size / elapsed if elapsed > 0 else 0
                    recent_qps.append(qps)
                    total_queries += batch_size
                    
                    # Check if performance is stable
                    if len(recent_qps) >= window_size:
                        recent = recent_qps[-window_size:]
                        avg = sum(recent) / len(recent)
                        variance = sum((q - avg) ** 2 for q in recent) / len(recent)
                        cv = (variance ** 0.5) / avg if avg > 0 else 1
                        
                        if cv < stable_threshold and avg >= target_qps:
                            return {
                                "warmed_up": True,
                                "queries_used": total_queries,
                                "final_qps": avg,
                                "stabilized": True
                            }
                
                return {
                    "warmed_up": True,
                    "queries_used": total_queries,
                    "final_qps": recent_qps[-1] if recent_qps else 0,
                    "stabilized": False
                }
        
        mock_collection.search.return_value = [[Mock(id=i, distance=0.1*i) for i in range(10)]]
        
        warmer = IndexWarmUp(mock_collection)
        
        # Test basic warm-up
        with patch('time.time', side_effect=[0, 0.1, 0.2, 0.3, 0.4, 0.5] * 20):
            result = warmer.warm_up(num_queries=50, batch_size=10)
        
        assert result["total_queries"] == 50
        assert len(warmer.warm_up_stats) == 5
        
        # Test adaptive warm-up
        warmer2 = IndexWarmUp(mock_collection)
        
        with patch('time.time', side_effect=[i * 0.01 for i in range(200)]):
            result = warmer2.adaptive_warm_up(target_qps=100, max_queries=100)
        
        assert result["warmed_up"] is True
        assert result["queries_used"] <= 100
