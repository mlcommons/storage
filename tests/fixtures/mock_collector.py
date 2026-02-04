"""
Mock cluster collector for testing.

Provides a cluster collector that returns predefined data without
actual MPI or SSH calls, enabling testing of benchmark and validation
logic that depends on cluster information.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from mlpstorage.interfaces.collector import (
    ClusterCollectorInterface,
    CollectionResult,
)


class MockClusterCollector(ClusterCollectorInterface):
    """
    Mock cluster collector for testing without MPI or SSH.

    Returns predefined cluster data, useful for testing:
    - Benchmark initialization that requires cluster info
    - Validation logic that checks system requirements
    - Reporting code that processes cluster data

    Attributes:
        mock_data: The cluster data to return from collect().
        should_fail: If True, collect() will raise an exception.
        collect_calls: List of collect() call arguments for verification.

    Example:
        collector = MockClusterCollector({
            'hosts': [
                {'hostname': 'node1', 'memory': {'total': 256 * 1024**3}},
                {'hostname': 'node2', 'memory': {'total': 256 * 1024**3}},
            ]
        })

        result = collector.collect(['node1', 'node2'])
        assert result.success
        assert len(result.data['hosts']) == 2
    """

    def __init__(
        self,
        mock_data: Optional[Dict[str, Any]] = None,
        should_fail: bool = False,
        fail_message: str = "Mock collector failure"
    ):
        """
        Initialize the mock collector.

        Args:
            mock_data: Data to return from collect(). If None, uses default.
            should_fail: If True, collect() raises RuntimeError.
            fail_message: Message for the RuntimeError if should_fail is True.
        """
        self.mock_data = mock_data or self._default_data()
        self.should_fail = should_fail
        self.fail_message = fail_message
        self.collect_calls: List[Dict[str, Any]] = []
        self.collect_local_calls: int = 0

    def collect(self, hosts: List[str], timeout: int = 60) -> CollectionResult:
        """
        Return mock collection result.

        Args:
            hosts: List of hostnames (recorded but not used).
            timeout: Timeout in seconds (recorded but not used).

        Returns:
            CollectionResult with mock data.

        Raises:
            RuntimeError: If should_fail is True.
        """
        self.collect_calls.append({'hosts': hosts, 'timeout': timeout})

        if self.should_fail:
            raise RuntimeError(self.fail_message)

        return CollectionResult(
            success=True,
            data=self.mock_data,
            collection_method="mock",
            timestamp=datetime.now().isoformat()
        )

    def collect_local(self) -> CollectionResult:
        """
        Return mock local collection result.

        Returns:
            CollectionResult with first host's data.
        """
        self.collect_local_calls += 1

        if self.should_fail:
            raise RuntimeError(self.fail_message)

        # Return first host's data as local data
        hosts = self.mock_data.get('hosts', [{}])
        local_data = hosts[0] if hosts else {}

        return CollectionResult(
            success=True,
            data={'local': local_data},
            collection_method="mock_local",
            timestamp=datetime.now().isoformat()
        )

    def is_available(self) -> bool:
        """Return True unless should_fail is set."""
        return not self.should_fail

    def get_collection_method(self) -> str:
        """Return 'mock' as the collection method."""
        return "mock"

    def _default_data(self) -> Dict[str, Any]:
        """Generate default mock cluster data."""
        return {
            'hosts': [
                {
                    'hostname': 'test-host-1',
                    'memory': {
                        'total': 256 * 1024**3,  # 256 GB
                        'available': 240 * 1024**3,
                        'free': 200 * 1024**3,
                    },
                    'cpu': {
                        'num_cores': 32,
                        'num_logical_cores': 64,
                        'model': 'AMD EPYC 7763',
                        'architecture': 'x86_64',
                    },
                    'disks': [
                        {'name': 'nvme0n1', 'size': 2 * 1024**4},
                        {'name': 'nvme1n1', 'size': 2 * 1024**4},
                    ],
                    'system': {
                        'os': 'Linux',
                        'kernel': '5.15.0-generic',
                        'distribution': 'Ubuntu 22.04',
                    }
                }
            ],
            '_metadata': {
                'collection_method': 'mock',
                'collection_time': datetime.now().isoformat(),
            }
        }

    def set_hosts(self, num_hosts: int, memory_gb: int = 256, cpu_cores: int = 64):
        """
        Configure mock data for a specified number of identical hosts.

        Args:
            num_hosts: Number of hosts to generate.
            memory_gb: Memory per host in GB.
            cpu_cores: CPU cores per host.
        """
        hosts = []
        for i in range(num_hosts):
            hosts.append({
                'hostname': f'test-host-{i + 1}',
                'memory': {
                    'total': memory_gb * 1024**3,
                    'available': int(memory_gb * 0.9 * 1024**3),
                    'free': int(memory_gb * 0.8 * 1024**3),
                },
                'cpu': {
                    'num_cores': cpu_cores,
                    'num_logical_cores': cpu_cores * 2,
                    'model': 'AMD EPYC 7763',
                    'architecture': 'x86_64',
                },
                'disks': [
                    {'name': 'nvme0n1', 'size': 2 * 1024**4},
                ],
                'system': {
                    'os': 'Linux',
                    'kernel': '5.15.0-generic',
                    'distribution': 'Ubuntu 22.04',
                }
            })

        self.mock_data = {
            'hosts': hosts,
            '_metadata': {
                'collection_method': 'mock',
                'collection_time': datetime.now().isoformat(),
            }
        }

    def assert_collected(self, expected_hosts: Optional[List[str]] = None):
        """
        Assert that collect() was called.

        Args:
            expected_hosts: If provided, verify these hosts were passed.

        Raises:
            AssertionError: If collect() was not called or hosts don't match.
        """
        if not self.collect_calls:
            raise AssertionError("collect() was never called")

        if expected_hosts:
            last_call = self.collect_calls[-1]
            actual_hosts = last_call['hosts']
            if set(actual_hosts) != set(expected_hosts):
                raise AssertionError(
                    f"Expected hosts {expected_hosts}, got {actual_hosts}"
                )

    def clear(self):
        """Clear call history."""
        self.collect_calls.clear()
        self.collect_local_calls = 0
