"""
Unit tests for Milvus database connection management
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
import time
from typing import Dict, Any


class TestDatabaseConnection:
    """Test database connection management."""
    
    @patch('pymilvus.connections.connect')
    def test_successful_connection(self, mock_connect):
        """Test successful connection to Milvus."""
        mock_connect.return_value = True
        
        def connect_to_milvus(host="localhost", port=19530, **kwargs):
            from pymilvus import connections
            return connections.connect(
                alias="default",
                host=host,
                port=port,
                **kwargs
            )
        
        result = connect_to_milvus("localhost", 19530)
        assert result is True
        mock_connect.assert_called_once_with(
            alias="default",
            host="localhost",
            port=19530
        )
    
    @patch('pymilvus.connections.connect')
    def test_connection_with_timeout(self, mock_connect):
        """Test connection with custom timeout."""
        mock_connect.return_value = True
        
        def connect_with_timeout(host, port, timeout=30):
            from pymilvus import connections
            return connections.connect(
                alias="default",
                host=host,
                port=port,
                timeout=timeout
            )
        
        connect_with_timeout("localhost", 19530, timeout=60)
        mock_connect.assert_called_with(
            alias="default",
            host="localhost",
            port=19530,
            timeout=60
        )
    
    @patch('pymilvus.connections.connect')
    def test_connection_failure(self, mock_connect):
        """Test handling of connection failures."""
        mock_connect.side_effect = Exception("Connection refused")
        
        def connect_to_milvus(host, port):
            from pymilvus import connections
            try:
                return connections.connect(alias="default", host=host, port=port)
            except Exception as e:
                return f"Failed to connect: {e}"
        
        result = connect_to_milvus("localhost", 19530)
        assert "Failed to connect" in result
        assert "Connection refused" in result
    
    @patch('pymilvus.connections.connect')
    def test_connection_retry_logic(self, mock_connect):
        """Test connection retry mechanism."""
        # Fail twice, then succeed
        mock_connect.side_effect = [
            Exception("Connection failed"),
            Exception("Connection failed"),
            True
        ]
        
        def connect_with_retry(host, port, max_retries=3, retry_delay=1):
            from pymilvus import connections
            
            for attempt in range(max_retries):
                try:
                    return connections.connect(
                        alias="default",
                        host=host,
                        port=port
                    )
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(retry_delay)
            
            return False
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = connect_with_retry("localhost", 19530)
            assert result is True
            assert mock_connect.call_count == 3
    
    @patch('pymilvus.connections.list_connections')
    def test_list_connections(self, mock_list):
        """Test listing active connections."""
        mock_list.return_value = [
            ("default", {"host": "localhost", "port": 19530}),
            ("secondary", {"host": "remote", "port": 8080})
        ]
        
        def get_active_connections():
            from pymilvus import connections
            return connections.list_connections()
        
        connections_list = get_active_connections()
        assert len(connections_list) == 2
        assert connections_list[0][0] == "default"
        assert connections_list[1][1]["host"] == "remote"
    
    @patch('pymilvus.connections.disconnect')
    def test_disconnect(self, mock_disconnect):
        """Test disconnecting from Milvus."""
        mock_disconnect.return_value = None
        
        def disconnect_from_milvus(alias="default"):
            from pymilvus import connections
            connections.disconnect(alias)
            return True
        
        result = disconnect_from_milvus()
        assert result is True
        mock_disconnect.assert_called_once_with("default")
    
    @patch('pymilvus.connections.connect')
    def test_connection_pool(self, mock_connect):
        """Test connection pooling behavior."""
        mock_connect.return_value = True
        
        class ConnectionPool:
            def __init__(self, max_connections=5):
                self.max_connections = max_connections
                self.connections = []
                self.available = []
            
            def get_connection(self):
                if self.available:
                    return self.available.pop()
                elif len(self.connections) < self.max_connections:
                    from pymilvus import connections
                    conn = connections.connect(
                        alias=f"conn_{len(self.connections)}",
                        host="localhost",
                        port=19530
                    )
                    self.connections.append(conn)
                    return conn
                else:
                    raise Exception("Connection pool exhausted")
            
            def return_connection(self, conn):
                self.available.append(conn)
            
            def close_all(self):
                for conn in self.connections:
                    # In real code, would disconnect each connection
                    pass
                self.connections.clear()
                self.available.clear()
        
        pool = ConnectionPool(max_connections=3)
        
        # Get connections
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        conn3 = pool.get_connection()
        
        # Pool should be exhausted
        with pytest.raises(Exception, match="Connection pool exhausted"):
            pool.get_connection()
        
        # Return a connection
        pool.return_connection(conn1)
        
        # Should be able to get a connection now
        conn4 = pool.get_connection()
        assert conn4 == conn1  # Should reuse the returned connection
    
    @patch('pymilvus.connections.connect')
    def test_connection_with_authentication(self, mock_connect):
        """Test connection with authentication credentials."""
        mock_connect.return_value = True
        
        def connect_with_auth(host, port, user, password):
            from pymilvus import connections
            return connections.connect(
                alias="default",
                host=host,
                port=port,
                user=user,
                password=password
            )
        
        connect_with_auth("localhost", 19530, "admin", "password123")
        
        mock_connect.assert_called_with(
            alias="default",
            host="localhost",
            port=19530,
            user="admin",
            password="password123"
        )
    
    @patch('pymilvus.connections.connect')
    def test_connection_health_check(self, mock_connect):
        """Test connection health check mechanism."""
        mock_connect.return_value = True
        
        class MilvusConnection:
            def __init__(self, host, port):
                self.host = host
                self.port = port
                self.connected = False
                self.last_health_check = 0
            
            def connect(self):
                from pymilvus import connections
                try:
                    connections.connect(
                        alias="health_check",
                        host=self.host,
                        port=self.port
                    )
                    self.connected = True
                    return True
                except:
                    self.connected = False
                    return False
            
            def health_check(self):
                """Perform a health check on the connection."""
                current_time = time.time()
                
                # Only check every 30 seconds
                if current_time - self.last_health_check < 30:
                    return self.connected
                
                self.last_health_check = current_time
                
                # Try a simple operation to verify connection
                try:
                    # In real code, would perform a lightweight operation
                    # like checking server status
                    return self.connected
                except:
                    self.connected = False
                    return False
            
            def ensure_connected(self):
                """Ensure connection is active, reconnect if needed."""
                if not self.health_check():
                    return self.connect()
                return True
        
        conn = MilvusConnection("localhost", 19530)
        assert conn.connect() is True
        assert conn.health_check() is True
        assert conn.ensure_connected() is True


class TestCollectionManagement:
    """Test Milvus collection management operations."""
    
    @patch('pymilvus.Collection')
    def test_create_collection(self, mock_collection_class):
        """Test creating a new collection."""
        mock_collection = Mock()
        mock_collection_class.return_value = mock_collection
        
        def create_collection(name, dimension, metric_type="L2"):
            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]
            schema = CollectionSchema(fields, description=f"Collection {name}")
            
            # Create collection
            collection = Collection(name=name, schema=schema)
            return collection
        
        coll = create_collection("test_collection", 128)
        assert coll is not None
        mock_collection_class.assert_called_once()
    
    @patch('pymilvus.utility.has_collection')
    def test_check_collection_exists(self, mock_has_collection):
        """Test checking if a collection exists."""
        mock_has_collection.return_value = True
        
        def collection_exists(collection_name):
            from pymilvus import utility
            return utility.has_collection(collection_name)
        
        exists = collection_exists("test_collection")
        assert exists is True
        mock_has_collection.assert_called_once_with("test_collection")
    
    @patch('pymilvus.Collection')
    def test_drop_collection(self, mock_collection_class):
        """Test dropping a collection."""
        mock_collection = Mock()
        mock_collection.drop = Mock()
        mock_collection_class.return_value = mock_collection
        
        def drop_collection(collection_name):
            from pymilvus import Collection
            collection = Collection(collection_name)
            collection.drop()
            return True
        
        result = drop_collection("test_collection")
        assert result is True
        mock_collection.drop.assert_called_once()
    
    @patch('pymilvus.utility.list_collections')
    def test_list_collections(self, mock_list_collections):
        """Test listing all collections."""
        mock_list_collections.return_value = [
            "collection1",
            "collection2",
            "collection3"
        ]
        
        def get_all_collections():
            from pymilvus import utility
            return utility.list_collections()
        
        collections = get_all_collections()
        assert len(collections) == 3
        assert "collection1" in collections
    
    def test_collection_with_partitions(self, mock_collection):
        """Test creating and managing collection partitions."""
        mock_collection.create_partition = Mock()
        mock_collection.has_partition = Mock(return_value=False)
        mock_collection.partitions = []
        
        def create_partitions(collection, partition_names):
            for name in partition_names:
                if not collection.has_partition(name):
                    collection.create_partition(name)
                    collection.partitions.append(name)
            return collection.partitions
        
        partitions = create_partitions(mock_collection, ["partition1", "partition2"])
        assert len(partitions) == 2
        assert mock_collection.create_partition.call_count == 2
    
    def test_collection_properties(self, mock_collection):
        """Test getting collection properties."""
        mock_collection.num_entities = 10000
        mock_collection.description = "Test collection"
        mock_collection.name = "test_coll"
        mock_collection.schema = Mock()
        
        def get_collection_info(collection):
            return {
                "name": collection.name,
                "description": collection.description,
                "num_entities": collection.num_entities,
                "schema": collection.schema
            }
        
        info = get_collection_info(mock_collection)
        assert info["name"] == "test_coll"
        assert info["num_entities"] == 10000
        assert info["description"] == "Test collection"


class TestConnectionResilience:
    """Test connection resilience and error recovery."""
    
    @patch('pymilvus.connections.connect')
    def test_automatic_reconnection(self, mock_connect):
        """Test automatic reconnection after connection loss."""
        # Simulate connection loss and recovery
        mock_connect.side_effect = [
            True,  # Initial connection
            Exception("Connection lost"),  # Connection drops
            Exception("Still disconnected"),  # First retry fails
            True  # Reconnection succeeds
        ]
        
        class ResilientConnection:
            def __init__(self):
                self.connected = False
                self.retry_count = 0
                self.max_retries = 3
                self.connection_attempts = 0
            
            def execute_with_retry(self, operation):
                """Execute operation with automatic retry on connection failure."""
                for attempt in range(self.max_retries):
                    try:
                        if not self.connected or attempt > 0:
                            self._connect()
                        
                        result = operation()
                        self.retry_count = 0  # Reset retry count on success
                        return result
                    
                    except Exception as e:
                        self.retry_count += 1
                        self.connected = False
                        
                        if self.retry_count >= self.max_retries:
                            raise Exception(f"Max retries exceeded: {e}")
                        
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            def _connect(self):
                from pymilvus import connections
                self.connection_attempts += 1
                if self.connection_attempts <= 2:
                    # First two connection attempts fail
                    self.connected = False
                    if self.connection_attempts == 1:
                        raise Exception("Connection lost")
                    else:
                        raise Exception("Still disconnected")
                else:
                    # Third attempt succeeds
                    connections.connect(alias="resilient", host="localhost", port=19530)
                    self.connected = True
        
        conn = ResilientConnection()
        
        # Mock operation that will fail initially
        operation_calls = 0
        def test_operation():
            nonlocal operation_calls
            operation_calls += 1
            if operation_calls < 3 and not conn.connected:
                raise Exception("Operation failed")
            return "Success"
        
        with patch('time.sleep'):  # Mock sleep for faster testing
            result = conn.execute_with_retry(test_operation)
            
        # Operation should eventually succeed
        assert result == "Success"
    
    @patch('pymilvus.connections.connect')
    def test_connection_timeout_handling(self, mock_connect):
        """Test handling of connection timeouts."""
        import socket
        mock_connect.side_effect = socket.timeout("Connection timed out")
        
        def connect_with_timeout_handling(host, port, timeout=10):
            from pymilvus import connections
            
            try:
                return connections.connect(
                    alias="timeout_test",
                    host=host,
                    port=port,
                    timeout=timeout
                )
            except socket.timeout as e:
                return f"Connection timeout: {e}"
            except Exception as e:
                return f"Connection error: {e}"
        
        result = connect_with_timeout_handling("localhost", 19530, timeout=5)
        assert "Connection timeout" in result
    
    def test_connection_state_management(self):
        """Test managing connection state across operations."""
        class ConnectionManager:
            def __init__(self):
                self.connections = {}
                self.active_alias = None
            
            def add_connection(self, alias, host, port):
                """Add a connection configuration."""
                self.connections[alias] = {
                    "host": host,
                    "port": port,
                    "connected": False
                }
            
            def switch_connection(self, alias):
                """Switch to a different connection."""
                if alias not in self.connections:
                    raise ValueError(f"Unknown connection alias: {alias}")
                
                # Disconnect from current if connected
                if self.active_alias and self.connections[self.active_alias]["connected"]:
                    self.connections[self.active_alias]["connected"] = False
                
                self.active_alias = alias
                self.connections[alias]["connected"] = True
                return True
            
            def get_active_connection(self):
                """Get the currently active connection."""
                if not self.active_alias:
                    return None
                return self.connections.get(self.active_alias)
            
            def close_all(self):
                """Close all connections."""
                for alias in self.connections:
                    self.connections[alias]["connected"] = False
                self.active_alias = None
        
        manager = ConnectionManager()
        manager.add_connection("primary", "localhost", 19530)
        manager.add_connection("secondary", "remote", 8080)
        
        # Switch to primary
        assert manager.switch_connection("primary") is True
        active = manager.get_active_connection()
        assert active["host"] == "localhost"
        assert active["connected"] is True
        
        # Switch to secondary
        manager.switch_connection("secondary")
        assert manager.connections["primary"]["connected"] is False
        assert manager.connections["secondary"]["connected"] is True
        
        # Close all
        manager.close_all()
        assert all(not conn["connected"] for conn in manager.connections.values())
