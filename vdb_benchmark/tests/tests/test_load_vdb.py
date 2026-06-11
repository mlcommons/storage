"""
Unit tests for vector loading functionality in vdb-bench
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
import time
from typing import List, Generator
import json


class TestVectorGeneration:
    """Test vector generation utilities."""
    
    def test_uniform_vector_generation(self):
        """Test generating vectors with uniform distribution."""
        def generate_uniform_vectors(num_vectors, dimension, seed=None):
            if seed is not None:
                np.random.seed(seed)
            return np.random.uniform(-1, 1, size=(num_vectors, dimension)).astype(np.float32)
        
        vectors = generate_uniform_vectors(100, 128, seed=42)
        
        assert vectors.shape == (100, 128)
        assert vectors.dtype == np.float32
        assert vectors.min() >= -1
        assert vectors.max() <= 1
        
        # Test reproducibility with seed
        vectors2 = generate_uniform_vectors(100, 128, seed=42)
        np.testing.assert_array_equal(vectors, vectors2)
    
    def test_normal_vector_generation(self):
        """Test generating vectors with normal distribution."""
        def generate_normal_vectors(num_vectors, dimension, mean=0, std=1, seed=None):
            if seed is not None:
                np.random.seed(seed)
            return np.random.normal(mean, std, size=(num_vectors, dimension)).astype(np.float32)
        
        vectors = generate_normal_vectors(1000, 256, seed=42)
        
        assert vectors.shape == (1000, 256)
        assert vectors.dtype == np.float32
        
        # Check distribution properties (should be close to normal)
        assert -0.1 < vectors.mean() < 0.1  # Mean should be close to 0
        assert 0.9 < vectors.std() < 1.1  # Std should be close to 1
    
    def test_normalized_vector_generation(self):
        """Test generating L2-normalized vectors."""
        def generate_normalized_vectors(num_vectors, dimension, seed=None):
            if seed is not None:
                np.random.seed(seed)
            
            vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
            # L2 normalize each vector
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / norms
        
        vectors = generate_normalized_vectors(50, 64, seed=42)
        
        assert vectors.shape == (50, 64)
        
        # Check that all vectors are normalized
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(50), decimal=5)
    
    def test_chunked_vector_generation(self):
        """Test generating vectors in chunks for memory efficiency."""
        def generate_vectors_chunked(total_vectors, dimension, chunk_size=1000):
            """Generate vectors in chunks to manage memory."""
            num_chunks = (total_vectors + chunk_size - 1) // chunk_size
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, total_vectors)
                chunk_vectors = end_idx - start_idx
                
                yield np.random.randn(chunk_vectors, dimension).astype(np.float32)
        
        # Generate 10000 vectors in chunks of 1000
        all_vectors = []
        for chunk in generate_vectors_chunked(10000, 128, chunk_size=1000):
            all_vectors.append(chunk)
        
        assert len(all_vectors) == 10
        assert all_vectors[0].shape == (1000, 128)
        
        # Concatenate and verify total
        concatenated = np.vstack(all_vectors)
        assert concatenated.shape == (10000, 128)
    
    def test_vector_generation_with_ids(self):
        """Test generating vectors with associated IDs."""
        def generate_vectors_with_ids(num_vectors, dimension, start_id=0):
            vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
            ids = np.arange(start_id, start_id + num_vectors, dtype=np.int64)
            return ids, vectors
        
        ids, vectors = generate_vectors_with_ids(100, 256, start_id=1000)
        
        assert len(ids) == 100
        assert ids[0] == 1000
        assert ids[-1] == 1099
        assert vectors.shape == (100, 256)
    
    def test_vector_generation_progress_tracking(self):
        """Test tracking progress during vector generation."""
        def generate_with_progress(num_vectors, dimension, chunk_size=100):
            total_generated = 0
            progress_updates = []
            
            for chunk_num in range(0, num_vectors, chunk_size):
                chunk_end = min(chunk_num + chunk_size, num_vectors)
                chunk_size_actual = chunk_end - chunk_num
                
                vectors = np.random.randn(chunk_size_actual, dimension).astype(np.float32)
                
                total_generated += chunk_size_actual
                progress = (total_generated / num_vectors) * 100
                progress_updates.append(progress)
                
                yield vectors, progress
        
        progress_list = []
        vector_list = []
        
        for vectors, progress in generate_with_progress(1000, 128, chunk_size=200):
            vector_list.append(vectors)
            progress_list.append(progress)
        
        assert len(progress_list) == 5
        assert progress_list[-1] == 100.0
        assert all(p > 0 for p in progress_list)


class TestVectorLoading:
    """Test vector loading into database."""
    
    def test_batch_insertion(self, mock_collection):
        """Test inserting vectors in batches."""
        inserted_data = []
        mock_collection.insert.side_effect = lambda data: inserted_data.append(data)
        
        def insert_vectors_batch(collection, vectors, batch_size=1000):
            """Insert vectors in batches."""
            num_vectors = len(vectors)
            total_inserted = 0
            
            for i in range(0, num_vectors, batch_size):
                batch = vectors[i:i + batch_size]
                collection.insert([batch])
                total_inserted += len(batch)
            
            return total_inserted
        
        vectors = np.random.randn(5000, 128).astype(np.float32)
        total = insert_vectors_batch(mock_collection, vectors, batch_size=1000)
        
        assert total == 5000
        assert mock_collection.insert.call_count == 5
    
    def test_insertion_with_error_handling(self, mock_collection):
        """Test vector insertion with error handling."""
        # Simulate occasional insertion failures
        call_count = 0
        def insert_side_effect(data):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Insert failed")
            return Mock(primary_keys=list(range(len(data[0]))))
        
        mock_collection.insert.side_effect = insert_side_effect
        
        def insert_with_retry(collection, vectors, max_retries=3):
            """Insert vectors with retry on failure."""
            for attempt in range(max_retries):
                try:
                    result = collection.insert([vectors])
                    return result
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)
            return None
        
        vectors = np.random.randn(100, 128).astype(np.float32)
        
        with patch('time.sleep'):
            result = insert_with_retry(mock_collection, vectors)
        
        assert result is not None
        assert mock_collection.insert.call_count == 2  # Failed once, succeeded on retry
    
    def test_parallel_insertion(self, mock_collection):
        """Test parallel vector insertion using multiple threads/processes."""
        from concurrent.futures import ThreadPoolExecutor
        
        def insert_chunk(args):
            collection, chunk_id, vectors = args
            collection.insert([vectors])
            return chunk_id, len(vectors)
        
        def parallel_insert(collection, vectors, num_workers=4, chunk_size=1000):
            """Insert vectors in parallel."""
            chunks = []
            for i in range(0, len(vectors), chunk_size):
                chunk = vectors[i:i + chunk_size]
                chunks.append((collection, i // chunk_size, chunk))
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(insert_chunk, chunks))
            
            total_inserted = sum(count for _, count in results)
            return total_inserted
        
        vectors = np.random.randn(4000, 128).astype(np.float32)
        
        # Mock the insert to track calls
        inserted_chunks = []
        mock_collection.insert.side_effect = lambda data: inserted_chunks.append(len(data[0]))
        
        total = parallel_insert(mock_collection, vectors, num_workers=2, chunk_size=1000)
        
        assert total == 4000
        assert len(inserted_chunks) == 4
    
    def test_insertion_with_metadata(self, mock_collection):
        """Test inserting vectors with additional metadata."""
        def insert_vectors_with_metadata(collection, vectors, metadata):
            """Insert vectors along with metadata."""
            data = [
                vectors,
                metadata.get("ids", list(range(len(vectors)))),
                metadata.get("tags", ["default"] * len(vectors))
            ]
            
            result = collection.insert(data)
            return result
        
        vectors = np.random.randn(100, 128).astype(np.float32)
        metadata = {
            "ids": list(range(1000, 1100)),
            "tags": [f"tag_{i % 10}" for i in range(100)]
        }
        
        mock_collection.insert.return_value = Mock(primary_keys=metadata["ids"])
        
        result = insert_vectors_with_metadata(mock_collection, vectors, metadata)
        
        assert result.primary_keys == metadata["ids"]
        mock_collection.insert.assert_called_once()
    
    @patch('time.time')
    def test_insertion_rate_monitoring(self, mock_time, mock_collection):
        """Test monitoring insertion rate and throughput."""
        # Start at 1 instead of 0 to avoid issues with 0 being falsy
        time_sequence = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        mock_time.side_effect = time_sequence
        
        class InsertionMonitor:
            def __init__(self):
                self.total_vectors = 0
                self.start_time = None
                self.batch_times = []
                self.last_time = None
            
            def start(self):
                self.start_time = time.time()
                self.last_time = self.start_time
            
            def record_batch(self, batch_size):
                current_time = time.time()
                if self.start_time is not None:
                    # Calculate elapsed since last batch
                    elapsed = current_time - self.last_time
                    self.last_time = current_time
                    self.batch_times.append(current_time)
                    self.total_vectors += batch_size
                    
                    # Calculate throughput
                    total_elapsed = current_time - self.start_time
                    throughput = self.total_vectors / total_elapsed if total_elapsed > 0 else 0
                    
                    return {
                        "batch_size": batch_size,
                        "batch_time": elapsed,
                        "total_vectors": self.total_vectors,
                        "throughput": throughput
                    }
                return None
            
            def get_summary(self):
                # Check if we have data to summarize
                if self.start_time is None or len(self.batch_times) == 0:
                    return None
                
                # Calculate total time from start to last batch
                total_time = self.batch_times[-1] - self.start_time
                
                # Return summary if we have valid data
                if self.total_vectors > 0:
                    return {
                        "total_vectors": self.total_vectors,
                        "total_time": total_time,
                        "average_throughput": self.total_vectors / total_time if total_time > 0 else 0
                    }
                    
                return None
        
        monitor = InsertionMonitor()
        monitor.start()  # Uses time value 1.0
        
        # Simulate inserting batches (uses time values 2.0-6.0)
        stats = []
        for i in range(5):
            stat = monitor.record_batch(1000)
            if stat:
                stats.append(stat)
        
        summary = monitor.get_summary()
        
        assert summary is not None
        assert summary["total_vectors"] == 5000
        assert summary["total_time"] == 5.0  # From time 1.0 to time 6.0
        assert summary["average_throughput"] == 1000.0  # 5000 vectors / 5 seconds
    
    def test_load_checkpoint_resume(self, test_data_dir):
        """Test checkpoint and resume functionality for large loads."""
        checkpoint_file = test_data_dir / "checkpoint.json"
        
        class LoadCheckpoint:
            def __init__(self, checkpoint_path):
                self.checkpoint_path = checkpoint_path
                self.state = self.load_checkpoint()
            
            def load_checkpoint(self):
                """Load checkpoint from file if exists."""
                if self.checkpoint_path.exists():
                    with open(self.checkpoint_path, 'r') as f:
                        return json.load(f)
                return {"last_batch": 0, "total_inserted": 0}
            
            def save_checkpoint(self, batch_num, total_inserted):
                """Save current progress to checkpoint."""
                self.state = {
                    "last_batch": batch_num,
                    "total_inserted": total_inserted,
                    "timestamp": time.time()
                }
                with open(self.checkpoint_path, 'w') as f:
                    json.dump(self.state, f)
            
            def get_resume_point(self):
                """Get the batch number to resume from."""
                return self.state["last_batch"]
            
            def clear(self):
                """Clear checkpoint after successful completion."""
                if self.checkpoint_path.exists():
                    self.checkpoint_path.unlink()
                self.state = {"last_batch": 0, "total_inserted": 0}
        
        checkpoint = LoadCheckpoint(checkpoint_file)
        
        # Simulate partial load
        checkpoint.save_checkpoint(5, 5000)
        assert checkpoint.get_resume_point() == 5
        
        # Simulate resume
        checkpoint2 = LoadCheckpoint(checkpoint_file)
        assert checkpoint2.get_resume_point() == 5
        assert checkpoint2.state["total_inserted"] == 5000
        
        # Clear checkpoint
        checkpoint2.clear()
        assert not checkpoint_file.exists()


class TestLoadOptimization:
    """Test load optimization strategies."""
    
    def test_dynamic_batch_sizing(self):
        """Test dynamic batch size adjustment based on performance."""
        class DynamicBatchSizer:
            def __init__(self, initial_size=1000, min_size=100, max_size=10000):
                self.current_size = initial_size
                self.min_size = min_size
                self.max_size = max_size
                self.history = []
            
            def adjust(self, insertion_time, batch_size):
                """Adjust batch size based on insertion performance."""
                throughput = batch_size / insertion_time if insertion_time > 0 else 0
                self.history.append((batch_size, throughput))
                
                if len(self.history) >= 3:
                    # Calculate trend
                    recent_throughputs = [tp for _, tp in self.history[-3:]]
                    avg_throughput = sum(recent_throughputs) / len(recent_throughputs)
                    
                    if throughput > avg_throughput * 1.1:
                        # Performance improving, increase batch size
                        self.current_size = min(
                            int(self.current_size * 1.2),
                            self.max_size
                        )
                    elif throughput < avg_throughput * 0.9:
                        # Performance degrading, decrease batch size
                        self.current_size = max(
                            int(self.current_size * 0.8),
                            self.min_size
                        )
                
                return self.current_size
        
        sizer = DynamicBatchSizer(initial_size=1000)
        
        # Simulate good performance - should increase batch size
        new_size = sizer.adjust(1.0, 1000)  # 1000 vectors/sec
        new_size = sizer.adjust(0.9, 1000)  # 1111 vectors/sec
        new_size = sizer.adjust(0.8, 1000)  # 1250 vectors/sec
        new_size = sizer.adjust(0.7, new_size)  # Improving performance
        
        assert new_size > 1000  # Should have increased
        
        # Simulate degrading performance - should decrease batch size
        sizer2 = DynamicBatchSizer(initial_size=5000)
        new_size = sizer2.adjust(1.0, 5000)  # 5000 vectors/sec
        new_size = sizer2.adjust(1.2, 5000)  # 4166 vectors/sec
        new_size = sizer2.adjust(1.5, 5000)  # 3333 vectors/sec
        new_size = sizer2.adjust(2.0, new_size)  # Degrading performance
        
        assert new_size < 5000  # Should have decreased
    
    def test_memory_aware_loading(self):
        """Test memory-aware vector loading."""
        import psutil
        
        class MemoryAwareLoader:
            def __init__(self, memory_threshold=0.8):
                self.memory_threshold = memory_threshold
                self.base_batch_size = 1000
            
            def get_memory_usage(self):
                """Get current memory usage percentage."""
                return psutil.virtual_memory().percent / 100
            
            def calculate_safe_batch_size(self, vector_dimension):
                """Calculate safe batch size based on available memory."""
                memory_usage = self.get_memory_usage()
                
                if memory_usage > self.memory_threshold:
                    # Reduce batch size when memory is high
                    reduction_factor = 1.0 - (memory_usage - self.memory_threshold)
                    return max(100, int(self.base_batch_size * reduction_factor))
                
                # Calculate based on vector size
                bytes_per_vector = vector_dimension * 4  # float32
                available_memory = (1.0 - memory_usage) * psutil.virtual_memory().total
                max_vectors = int(available_memory * 0.5 / bytes_per_vector)  # Use 50% of available
                
                return min(max_vectors, self.base_batch_size)
            
            def should_gc(self):
                """Determine if garbage collection should be triggered."""
                return self.get_memory_usage() > 0.7
        
        with patch('psutil.virtual_memory') as mock_memory:
            # Simulate different memory conditions
            mock_memory.return_value = Mock(percent=60, total=16 * 1024**3)  # 60% used, 16GB total
            
            loader = MemoryAwareLoader()
            batch_size = loader.calculate_safe_batch_size(1536)
            
            assert batch_size > 0
            assert not loader.should_gc()
            
            # Simulate high memory usage
            mock_memory.return_value = Mock(percent=85, total=16 * 1024**3)  # 85% used
            
            batch_size = loader.calculate_safe_batch_size(1536)
            assert batch_size < loader.base_batch_size  # Should be reduced
            assert loader.should_gc()
    
    def test_flush_optimization(self, mock_collection):
        """Test optimizing flush operations during loading."""
        flush_count = 0
        
        def mock_flush():
            nonlocal flush_count
            flush_count += 1
            time.sleep(0.1)  # Simulate flush time
        
        mock_collection.flush = mock_flush
        
        class FlushOptimizer:
            def __init__(self, flush_interval=10000, time_interval=60):
                self.flush_interval = flush_interval
                self.time_interval = time_interval
                self.vectors_since_flush = 0
                self.last_flush_time = time.time()
            
            def should_flush(self, vectors_inserted):
                """Determine if flush should be triggered."""
                self.vectors_since_flush += vectors_inserted
                current_time = time.time()
                
                # Flush based on vector count or time
                if (self.vectors_since_flush >= self.flush_interval or 
                    current_time - self.last_flush_time >= self.time_interval):
                    return True
                return False
            
            def flush(self, collection):
                """Perform flush and reset counters."""
                collection.flush()
                self.vectors_since_flush = 0
                self.last_flush_time = time.time()
        
        optimizer = FlushOptimizer(flush_interval=5000)
        
        with patch('time.sleep'):  # Speed up test
            # Simulate loading vectors
            for i in range(10):
                if optimizer.should_flush(1000):
                    optimizer.flush(mock_collection)
        
        assert flush_count == 2  # Should have flushed twice (at 5000 and 10000)
