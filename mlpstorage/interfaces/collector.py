"""
Cluster collector interface definitions for mlpstorage.

This module defines the abstract interface for collecting system and cluster
information from benchmark hosts. Implementations may use different methods
(MPI, SSH, local-only) to gather this information.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class CollectionResult:
    """Result of cluster information collection.

    Attributes:
        success: Whether collection completed successfully.
        data: Collected system information by hostname.
        errors: List of errors encountered during collection.
        collection_method: Method used for collection (e.g., 'mpi', 'local').
        timestamp: ISO timestamp when collection occurred.
    """
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    collection_method: str = "unknown"
    timestamp: Optional[str] = None


class ClusterCollectorInterface(ABC):
    """Interface for cluster information collectors.

    Collectors gather system information from benchmark hosts including:
    - Memory configuration (total, available, etc.)
    - CPU information (cores, model, architecture)
    - Disk statistics
    - Network interface information
    - OS and kernel versions

    This information is used for:
    - Validating system requirements for benchmarks
    - Recording system configuration with results
    - Calculating required dataset sizes

    Example:
        class MPICollector(ClusterCollectorInterface):
            def collect(self, hosts, timeout=60):
                # Use MPI to gather info from all hosts
                result = run_mpi_collection(hosts, timeout)
                return CollectionResult(
                    success=True,
                    data=result,
                    collection_method="mpi"
                )

            def is_available(self):
                return check_mpi_installed()
    """

    @abstractmethod
    def collect(self, hosts: List[str], timeout: int = 60) -> CollectionResult:
        """Collect information from all specified hosts.

        Args:
            hosts: List of hostnames or IP addresses to collect from.
            timeout: Maximum time in seconds to wait for collection.

        Returns:
            CollectionResult with data from all hosts.
        """
        pass

    @abstractmethod
    def collect_local(self) -> CollectionResult:
        """Collect information from local host only.

        This is used when MPI or distributed collection is not available
        or not needed.

        Returns:
            CollectionResult with local host data.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this collector is available for use.

        For example, MPI collector checks if mpi4py is installed
        and MPI runtime is available.

        Returns:
            True if collector can be used, False otherwise.
        """
        pass

    @abstractmethod
    def get_collection_method(self) -> str:
        """Return the name of the collection method.

        Returns:
            String identifier like 'mpi', 'ssh', 'local'.
        """
        pass


class LocalCollectorInterface(ABC):
    """Interface for local-only system information collection.

    Used when distributed collection is not needed or available.
    Collects information only from the local machine.
    """

    @abstractmethod
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information from /proc/meminfo or equivalent.

        Returns:
            Dictionary with memory statistics (total, free, available, etc.)
        """
        pass

    @abstractmethod
    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information from /proc/cpuinfo or equivalent.

        Returns:
            Dictionary with CPU details (model, cores, architecture, etc.)
        """
        pass

    @abstractmethod
    def get_disk_info(self) -> List[Dict[str, Any]]:
        """Get disk statistics from /proc/diskstats or equivalent.

        Returns:
            List of dictionaries, one per disk device.
        """
        pass

    @abstractmethod
    def get_network_info(self) -> List[Dict[str, Any]]:
        """Get network interface statistics.

        Returns:
            List of dictionaries, one per network interface.
        """
        pass

    @abstractmethod
    def get_system_info(self) -> Dict[str, Any]:
        """Get general system information (OS, kernel, hostname).

        Returns:
            Dictionary with system details.
        """
        pass
