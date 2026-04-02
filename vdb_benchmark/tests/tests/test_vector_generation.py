"""
Unit tests for vector generation utilities
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
import h5py
import tempfile
from pathlib import Path


class TestVectorGenerationUtilities:
    """Test vector generation utility functions."""
    
    def test_vector_normalization(self):
        """Test different vector normalization methods."""
        class VectorNormalizer:
            @staticmethod
            def l2_normalize(vectors):
                """L2 normalization."""
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                return vectors / (norms + 1e-10)  # Add epsilon to avoid division by zero
            
            @staticmethod
            def l1_normalize(vectors):
                """L1 normalization."""
                norms = np.sum(np.abs(vectors), axis=1, keepdims=True)
                return vectors / (norms + 1e-10)
            
            @staticmethod
            def max_normalize(vectors):
                """Max normalization (scale by maximum absolute value)."""
                max_vals = np.max(np.abs(vectors), axis=1, keepdims=True)
                return vectors / (max_vals + 1e-10)
            
            @staticmethod
            def standardize(vectors):
                """Standardization (zero mean, unit variance)."""
                mean = np.mean(vectors, axis=0, keepdims=True)
                std = np.std(vectors, axis=0, keepdims=True)
                return (vectors - mean) / (std + 1e-10)
        
        # Test data
        vectors = np.random.randn(100, 128).astype(np.float32)
        
        # Test L2 normalization
        l2_norm = VectorNormalizer.l2_normalize(vectors)
        norms = np.linalg.norm(l2_norm, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(100), decimal=5)
        
        # Test L1 normalization
        l1_norm = VectorNormalizer.l1_normalize(vectors)
        l1_sums = np.sum(np.abs(l1_norm), axis=1)
        np.testing.assert_array_almost_equal(l1_sums, np.ones(100), decimal=5)
        
        # Test max normalization
        max_norm = VectorNormalizer.max_normalize(vectors)
        max_vals = np.max(np.abs(max_norm), axis=1)
        np.testing.assert_array_almost_equal(max_vals, np.ones(100), decimal=5)
        
        # Test standardization
        standardized = VectorNormalizer.standardize(vectors)
        assert abs(np.mean(standardized)) < 0.01  # Mean should be close to 0
        assert abs(np.std(standardized) - 1.0) < 0.1  # Std should be close to 1
    
    def test_vector_quantization(self):
        """Test vector quantization methods."""
        class VectorQuantizer:
            @staticmethod
            def scalar_quantize(vectors, bits=8):
                """Scalar quantization to specified bit depth."""
                min_val = np.min(vectors)
                max_val = np.max(vectors)
                
                # Scale to [0, 2^bits - 1]
                scale = (2 ** bits - 1) / (max_val - min_val)
                quantized = np.round((vectors - min_val) * scale).astype(np.uint8 if bits == 8 else np.uint16)
                
                return quantized, (min_val, max_val, scale)
            
            @staticmethod
            def dequantize(quantized, params):
                """Dequantize vectors."""
                min_val, max_val, scale = params
                return quantized.astype(np.float32) / scale + min_val
            
            @staticmethod
            def product_quantize(vectors, num_subvectors=8, codebook_size=256):
                """Simple product quantization simulation."""
                dimension = vectors.shape[1]
                subvector_dim = dimension // num_subvectors
                
                codes = []
                codebooks = []
                
                for i in range(num_subvectors):
                    start = i * subvector_dim
                    end = start + subvector_dim
                    subvectors = vectors[:, start:end]
                    
                    # Simulate codebook (in reality would use k-means)
                    codebook = np.random.randn(codebook_size, subvector_dim).astype(np.float32)
                    codebooks.append(codebook)
                    
                    # Assign codes (find nearest codebook entry)
                    # Simplified - just random assignment for testing
                    subvector_codes = np.random.randint(0, codebook_size, len(vectors))
                    codes.append(subvector_codes)
                
                return np.array(codes).T, codebooks
        
        vectors = np.random.randn(100, 128).astype(np.float32)
        
        # Test scalar quantization
        quantizer = VectorQuantizer()
        quantized, params = quantizer.scalar_quantize(vectors, bits=8)
        
        assert quantized.dtype == np.uint8
        assert quantized.shape == vectors.shape
        
        # Test reconstruction
        reconstructed = quantizer.dequantize(quantized, params)
        assert reconstructed.shape == vectors.shape
        
        # Test product quantization
        pq_codes, codebooks = quantizer.product_quantize(vectors, num_subvectors=8)
        
        assert pq_codes.shape == (100, 8)  # 100 vectors, 8 subvectors
        assert len(codebooks) == 8
    
    def test_synthetic_dataset_generation(self):
        """Test generating synthetic datasets with specific properties."""
        class SyntheticDataGenerator:
            @staticmethod
            def generate_clustered(num_vectors, dimension, num_clusters=10, cluster_std=0.1):
                """Generate clustered vectors."""
                vectors_per_cluster = num_vectors // num_clusters
                vectors = []
                labels = []
                
                # Generate cluster centers
                centers = np.random.randn(num_clusters, dimension) * 10
                
                for i in range(num_clusters):
                    # Generate vectors around center
                    cluster_vectors = centers[i] + np.random.randn(vectors_per_cluster, dimension) * cluster_std
                    vectors.append(cluster_vectors)
                    labels.extend([i] * vectors_per_cluster)
                
                # Handle remaining vectors
                remaining = num_vectors - (vectors_per_cluster * num_clusters)
                if remaining > 0:
                    cluster_idx = np.random.randint(0, num_clusters)
                    extra_vectors = centers[cluster_idx] + np.random.randn(remaining, dimension) * cluster_std
                    vectors.append(extra_vectors)
                    labels.extend([cluster_idx] * remaining)
                
                return np.vstack(vectors).astype(np.float32), np.array(labels)
            
            @staticmethod
            def generate_sparse(num_vectors, dimension, sparsity=0.9):
                """Generate sparse vectors."""
                vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
                
                # Create mask for sparsity
                mask = np.random.random((num_vectors, dimension)) < sparsity
                vectors[mask] = 0
                
                return vectors
            
            @staticmethod
            def generate_correlated(num_vectors, dimension, correlation=0.8):
                """Generate vectors with correlated dimensions."""
                # Create correlation matrix
                base = np.random.randn(num_vectors, 1)
                
                vectors = []
                for i in range(dimension):
                    if i == 0:
                        vectors.append(base.flatten())
                    else:
                        # Mix with random noise based on correlation
                        noise = np.random.randn(num_vectors)
                        correlated = correlation * base.flatten() + (1 - correlation) * noise
                        vectors.append(correlated)
                
                return np.array(vectors).T.astype(np.float32)
        
        generator = SyntheticDataGenerator()
        
        # Test clustered generation
        vectors, labels = generator.generate_clustered(1000, 128, num_clusters=10)
        assert vectors.shape == (1000, 128)
        assert len(labels) == 1000
        assert len(np.unique(labels)) == 10
        
        # Test sparse generation
        sparse_vectors = generator.generate_sparse(100, 256, sparsity=0.9)
        assert sparse_vectors.shape == (100, 256)
        sparsity_ratio = np.sum(sparse_vectors == 0) / sparse_vectors.size
        assert 0.85 < sparsity_ratio < 0.95  # Should be approximately 0.9
        
        # Test correlated generation
        correlated = generator.generate_correlated(100, 64, correlation=0.8)
        assert correlated.shape == (100, 64)
    
    def test_vector_io_operations(self, test_data_dir):
        """Test saving and loading vectors in different formats."""
        class VectorIO:
            @staticmethod
            def save_npy(vectors, filepath):
                """Save vectors as NPY file."""
                np.save(filepath, vectors)
            
            @staticmethod
            def load_npy(filepath):
                """Load vectors from NPY file."""
                return np.load(filepath)
            
            @staticmethod
            def save_hdf5(vectors, filepath, dataset_name="vectors"):
                """Save vectors as HDF5 file."""
                with h5py.File(filepath, 'w') as f:
                    f.create_dataset(dataset_name, data=vectors, compression="gzip")
            
            @staticmethod
            def load_hdf5(filepath, dataset_name="vectors"):
                """Load vectors from HDF5 file."""
                with h5py.File(filepath, 'r') as f:
                    return f[dataset_name][:]
            
            @staticmethod
            def save_binary(vectors, filepath):
                """Save vectors as binary file."""
                vectors.tofile(filepath)
            
            @staticmethod
            def load_binary(filepath, dtype=np.float32, shape=None):
                """Load vectors from binary file."""
                vectors = np.fromfile(filepath, dtype=dtype)
                if shape:
                    vectors = vectors.reshape(shape)
                return vectors
            
            @staticmethod
            def save_text(vectors, filepath):
                """Save vectors as text file."""
                np.savetxt(filepath, vectors, fmt='%.6f')
            
            @staticmethod
            def load_text(filepath):
                """Load vectors from text file."""
                return np.loadtxt(filepath, dtype=np.float32)
        
        io_handler = VectorIO()
        vectors = np.random.randn(100, 128).astype(np.float32)
        
        # Test NPY format
        npy_path = test_data_dir / "vectors.npy"
        io_handler.save_npy(vectors, npy_path)
        loaded_npy = io_handler.load_npy(npy_path)
        np.testing.assert_array_almost_equal(vectors, loaded_npy)
        
        # Test HDF5 format
        hdf5_path = test_data_dir / "vectors.h5"
        io_handler.save_hdf5(vectors, hdf5_path)
        loaded_hdf5 = io_handler.load_hdf5(hdf5_path)
        np.testing.assert_array_almost_equal(vectors, loaded_hdf5)
        
        # Test binary format
        bin_path = test_data_dir / "vectors.bin"
        io_handler.save_binary(vectors, bin_path)
        loaded_bin = io_handler.load_binary(bin_path, shape=(100, 128))
        np.testing.assert_array_almost_equal(vectors, loaded_bin)
        
        # Test text format (smaller dataset for text)
        small_vectors = vectors[:10]
        txt_path = test_data_dir / "vectors.txt"
        io_handler.save_text(small_vectors, txt_path)
        loaded_txt = io_handler.load_text(txt_path)
        np.testing.assert_array_almost_equal(small_vectors, loaded_txt, decimal=5)


class TestIndexConfiguration:
    """Test index-specific configurations and parameters."""
    
    def test_diskann_parameter_validation(self):
        """Test DiskANN index parameter validation."""
        class DiskANNConfig:
            VALID_METRICS = ["L2", "IP", "COSINE"]
            
            @staticmethod
            def validate_params(params):
                """Validate DiskANN parameters."""
                errors = []
                
                # Check metric type
                if params.get("metric_type") not in DiskANNConfig.VALID_METRICS:
                    errors.append(f"Invalid metric_type: {params.get('metric_type')}")
                
                # Check max_degree
                max_degree = params.get("max_degree", 64)
                if not (1 <= max_degree <= 128):
                    errors.append(f"max_degree must be between 1 and 128, got {max_degree}")
                
                # Check search_list_size
                search_list = params.get("search_list_size", 200)
                if not (100 <= search_list <= 1000):
                    errors.append(f"search_list_size must be between 100 and 1000, got {search_list}")
                
                # Check PQ parameters if present
                if "pq_code_budget_gb" in params:
                    budget = params["pq_code_budget_gb"]
                    if budget <= 0:
                        errors.append(f"pq_code_budget_gb must be positive, got {budget}")
                
                return len(errors) == 0, errors
            
            @staticmethod
            def get_default_params(num_vectors, dimension):
                """Get default parameters based on dataset size."""
                if num_vectors < 1000000:
                    return {
                        "metric_type": "L2",
                        "max_degree": 32,
                        "search_list_size": 100
                    }
                elif num_vectors < 10000000:
                    return {
                        "metric_type": "L2",
                        "max_degree": 64,
                        "search_list_size": 200
                    }
                else:
                    return {
                        "metric_type": "L2",
                        "max_degree": 64,
                        "search_list_size": 300,
                        "pq_code_budget_gb": 0.2
                    }
        
        # Test valid parameters
        valid_params = {
            "metric_type": "L2",
            "max_degree": 64,
            "search_list_size": 200
        }
        
        is_valid, errors = DiskANNConfig.validate_params(valid_params)
        assert is_valid is True
        assert len(errors) == 0
        
        # Test invalid parameters
        invalid_params = {
            "metric_type": "INVALID",
            "max_degree": 200,
            "search_list_size": 50
        }
        
        is_valid, errors = DiskANNConfig.validate_params(invalid_params)
        assert is_valid is False
        assert len(errors) == 3
        
        # Test default parameter generation
        small_defaults = DiskANNConfig.get_default_params(100000, 128)
        assert small_defaults["max_degree"] == 32
        
        large_defaults = DiskANNConfig.get_default_params(20000000, 1536)
        assert "pq_code_budget_gb" in large_defaults
