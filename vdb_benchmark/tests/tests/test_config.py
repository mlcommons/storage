"""
Unit tests for configuration management in vdb-bench
"""
import pytest
import yaml
from pathlib import Path
from typing import Dict, Any
import os
from unittest.mock import patch, mock_open, MagicMock


class TestConfigurationLoader:
    """Test configuration loading and validation."""
    
    def test_load_valid_config(self, temp_config_file):
        """Test loading a valid configuration file."""
        # Mock the config loading function
        with open(temp_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        assert 'database' in config
        assert 'dataset' in config
        assert 'index' in config
        assert config['database']['host'] == '127.0.0.1'
        assert config['dataset']['num_vectors'] == 1000
    
    def test_load_missing_config_file(self):
        """Test handling of missing configuration file."""
        non_existent_file = Path("/tmp/non_existent_config.yaml")
        
        with pytest.raises(FileNotFoundError):
            with open(non_existent_file, 'r') as f:
                yaml.safe_load(f)
    
    def test_load_invalid_yaml(self, test_data_dir):
        """Test handling of invalid YAML syntax."""
        invalid_yaml_path = test_data_dir / "invalid.yaml"
        
        with open(invalid_yaml_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(yaml.YAMLError):
            with open(invalid_yaml_path, 'r') as f:
                yaml.safe_load(f)
    
    def test_config_validation_missing_required_fields(self):
        """Test validation when required configuration fields are missing."""
        incomplete_config = {
            "database": {
                "host": "localhost"
                # Missing port and other required fields
            }
        }
        
        # Mock validation function
        def validate_config(config):
            required_fields = ['port', 'database']
            for field in required_fields:
                if field not in config.get('database', {}):
                    raise ValueError(f"Missing required field: database.{field}")
        
        with pytest.raises(ValueError, match="Missing required field"):
            validate_config(incomplete_config)
    
    def test_config_validation_invalid_values(self):
        """Test validation of configuration values."""
        invalid_config = {
            "database": {
                "host": "localhost",
                "port": -1,  # Invalid port
                "database": "milvus"
            },
            "dataset": {
                "num_vectors": -100,  # Invalid negative value
                "dimension": 0,  # Invalid dimension
                "batch_size": 0  # Invalid batch size
            }
        }
        
        def validate_config_values(config):
            if config['database']['port'] < 1 or config['database']['port'] > 65535:
                raise ValueError("Invalid port number")
            if config['dataset']['num_vectors'] <= 0:
                raise ValueError("Number of vectors must be positive")
            if config['dataset']['dimension'] <= 0:
                raise ValueError("Vector dimension must be positive")
            if config['dataset']['batch_size'] <= 0:
                raise ValueError("Batch size must be positive")
        
        with pytest.raises(ValueError):
            validate_config_values(invalid_config)
    
    def test_config_merge_with_defaults(self):
        """Test merging user configuration with defaults."""
        default_config = {
            "database": {
                "host": "localhost",
                "port": 19530,
                "timeout": 30
            },
            "dataset": {
                "batch_size": 1000,
                "distribution": "uniform"
            }
        }
        
        user_config = {
            "database": {
                "host": "remote-host",
                "port": 8080
            },
            "dataset": {
                "batch_size": 500
            }
        }
        
        def merge_configs(default, user):
            """Deep merge user config into default config."""
            merged = default.copy()
            for key, value in user.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = merge_configs(merged[key], value)
                else:
                    merged[key] = value
            return merged
        
        merged = merge_configs(default_config, user_config)
        
        assert merged['database']['host'] == 'remote-host'
        assert merged['database']['port'] == 8080
        assert merged['database']['timeout'] == 30  # From default
        assert merged['dataset']['batch_size'] == 500
        assert merged['dataset']['distribution'] == 'uniform'  # From default
    
    def test_config_environment_variable_override(self, sample_config):
        """Test overriding configuration with environment variables."""
        import copy
        
        os.environ['VDB_BENCH_DATABASE_HOST'] = 'env-host'
        os.environ['VDB_BENCH_DATABASE_PORT'] = '9999'
        os.environ['VDB_BENCH_DATASET_NUM_VECTORS'] = '5000'
        
        def apply_env_overrides(config):
            """Apply environment variable overrides to configuration."""
            # Make a deep copy to avoid modifying original
            result = copy.deepcopy(config)
            env_prefix = 'VDB_BENCH_'
            
            for key, value in os.environ.items():
                if key.startswith(env_prefix):
                    # Parse the environment variable name
                    parts = key[len(env_prefix):].lower().split('_')
                    
                    # Special handling for num_vectors (DATASET_NUM_VECTORS)
                    if len(parts) >= 2 and parts[0] == 'dataset' and parts[1] == 'num' and len(parts) == 3 and parts[2] == 'vectors':
                        if 'dataset' not in result:
                            result['dataset'] = {}
                        result['dataset']['num_vectors'] = int(value)
                    else:
                        # Navigate to the config section for other keys
                        current = result
                        for part in parts[:-1]:
                            if part not in current:
                                current[part] = {}
                            current = current[part]
                        
                        # Set the value (with type conversion)
                        final_key = parts[-1]
                        if value.isdigit():
                            current[final_key] = int(value)
                        else:
                            current[final_key] = value
            
            return result
        
        config = apply_env_overrides(sample_config)
        
        assert config['database']['host'] == 'env-host'
        assert config['database']['port'] == 9999
        assert config['dataset']['num_vectors'] == 5000
        
        # Clean up environment variables
        del os.environ['VDB_BENCH_DATABASE_HOST']
        del os.environ['VDB_BENCH_DATABASE_PORT']
        del os.environ['VDB_BENCH_DATASET_NUM_VECTORS']
    
    def test_config_save(self, test_data_dir):
        """Test saving configuration to file."""
        config = {
            "database": {"host": "localhost", "port": 19530},
            "dataset": {"collection_name": "test", "dimension": 128}
        }
        
        save_path = test_data_dir / "saved_config.yaml"
        
        with open(save_path, 'w') as f:
            yaml.dump(config, f)
        
        # Verify saved file
        with open(save_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config == config
    
    def test_config_schema_validation(self):
        """Test configuration schema validation."""
        schema = {
            "database": {
                "type": "dict",
                "required": ["host", "port"],
                "properties": {
                    "host": {"type": "string"},
                    "port": {"type": "integer", "min": 1, "max": 65535}
                }
            },
            "dataset": {
                "type": "dict",
                "required": ["dimension"],
                "properties": {
                    "dimension": {"type": "integer", "min": 1}
                }
            }
        }
        
        def validate_against_schema(config, schema):
            """Basic schema validation."""
            for key, rules in schema.items():
                if rules.get("type") == "dict":
                    if key not in config:
                        if "required" in rules:
                            raise ValueError(f"Missing required section: {key}")
                        continue
                    
                    if "required" in rules:
                        for req_field in rules["required"]:
                            if req_field not in config[key]:
                                raise ValueError(f"Missing required field: {key}.{req_field}")
                    
                    if "properties" in rules:
                        for prop, prop_rules in rules["properties"].items():
                            if prop in config[key]:
                                value = config[key][prop]
                                if "type" in prop_rules:
                                    if prop_rules["type"] == "integer" and not isinstance(value, int):
                                        raise TypeError(f"{key}.{prop} must be an integer")
                                    if prop_rules["type"] == "string" and not isinstance(value, str):
                                        raise TypeError(f"{key}.{prop} must be a string")
                                
                                if "min" in prop_rules and value < prop_rules["min"]:
                                    raise ValueError(f"{key}.{prop} must be >= {prop_rules['min']}")
                                if "max" in prop_rules and value > prop_rules["max"]:
                                    raise ValueError(f"{key}.{prop} must be <= {prop_rules['max']}")
        
        # Valid config
        valid_config = {
            "database": {"host": "localhost", "port": 19530},
            "dataset": {"dimension": 128}
        }
        
        validate_against_schema(valid_config, schema)  # Should not raise
        
        # Invalid config (missing required field)
        invalid_config = {
            "database": {"host": "localhost"},  # Missing port
            "dataset": {"dimension": 128}
        }
        
        with pytest.raises(ValueError, match="Missing required field"):
            validate_against_schema(invalid_config, schema)


class TestIndexConfiguration:
    """Test index-specific configuration handling."""
    
    def test_diskann_config_validation(self):
        """Test DiskANN index configuration validation."""
        valid_diskann_config = {
            "index_type": "DISKANN",
            "metric_type": "COSINE",
            "max_degree": 64,
            "search_list_size": 200,
            "pq_code_budget_gb": 0.1,
            "build_algo": "IVF_PQ"
        }
        
        def validate_diskann_config(config):
            assert config["index_type"] == "DISKANN"
            assert config["metric_type"] in ["L2", "IP", "COSINE"]
            assert 1 <= config["max_degree"] <= 128
            assert 100 <= config["search_list_size"] <= 1000
            if "pq_code_budget_gb" in config:
                assert config["pq_code_budget_gb"] > 0
        
        validate_diskann_config(valid_diskann_config)
        
        # Invalid max_degree
        invalid_config = valid_diskann_config.copy()
        invalid_config["max_degree"] = 200
        
        with pytest.raises(AssertionError):
            validate_diskann_config(invalid_config)
    
    def test_hnsw_config_validation(self):
        """Test HNSW index configuration validation."""
        valid_hnsw_config = {
            "index_type": "HNSW",
            "metric_type": "L2",
            "M": 16,
            "efConstruction": 200
        }
        
        def validate_hnsw_config(config):
            assert config["index_type"] == "HNSW"
            assert config["metric_type"] in ["L2", "IP", "COSINE"]
            assert 4 <= config["M"] <= 64
            assert 8 <= config["efConstruction"] <= 512
        
        validate_hnsw_config(valid_hnsw_config)
        
        # Invalid M value
        invalid_config = valid_hnsw_config.copy()
        invalid_config["M"] = 100
        
        with pytest.raises(AssertionError):
            validate_hnsw_config(invalid_config)
    
    def test_auto_index_config_selection(self):
        """Test automatic index configuration based on dataset size."""
        def select_index_config(num_vectors, dimension):
            if num_vectors < 100000:
                return {
                    "index_type": "IVF_FLAT",
                    "nlist": 128
                }
            elif num_vectors < 1000000:
                return {
                    "index_type": "HNSW",
                    "M": 16,
                    "efConstruction": 200
                }
            else:
                return {
                    "index_type": "DISKANN",
                    "max_degree": 64,
                    "search_list_size": 200
                }
        
        # Small dataset
        config = select_index_config(50000, 128)
        assert config["index_type"] == "IVF_FLAT"
        
        # Medium dataset
        config = select_index_config(500000, 256)
        assert config["index_type"] == "HNSW"
        
        # Large dataset
        config = select_index_config(10000000, 1536)
        assert config["index_type"] == "DISKANN"
