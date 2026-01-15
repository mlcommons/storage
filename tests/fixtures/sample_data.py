"""
Sample data fixtures for mlpstorage tests.

Provides realistic sample data for testing parsers, validators,
and benchmark logic without requiring actual system files.
"""

from argparse import Namespace
from typing import Dict, Any, Optional, List
from unittest.mock import MagicMock


# =============================================================================
# Sample /proc/meminfo Content
# =============================================================================

SAMPLE_MEMINFO = """MemTotal:       263866580 kB
MemFree:        197523648 kB
MemAvailable:   245869900 kB
Buffers:          987456 kB
Cached:         42567892 kB
SwapCached:            0 kB
Active:         18234567 kB
Inactive:       36789012 kB
Active(anon):    8234567 kB
Inactive(anon):   123456 kB
Active(file):    9999999 kB
Inactive(file): 36666666 kB
Unevictable:           0 kB
Mlocked:               0 kB
SwapTotal:      8388604 kB
SwapFree:       8388604 kB
Dirty:               128 kB
Writeback:             0 kB
AnonPages:       8123456 kB
Mapped:          1234567 kB
Shmem:            234567 kB
KReclaimable:    3456789 kB
Slab:            4567890 kB
SReclaimable:    3456789 kB
SUnreclaim:      1111101 kB
KernelStack:       12345 kB
PageTables:        56789 kB
NFS_Unstable:          0 kB
Bounce:                0 kB
WritebackTmp:          0 kB
CommitLimit:   140321892 kB
Committed_AS:   22345678 kB
VmallocTotal:   34359738367 kB
VmallocUsed:       89012 kB
VmallocChunk:          0 kB
Percpu:            34567 kB
HardwareCorrupted:     0 kB
AnonHugePages:   2097152 kB
ShmemHugePages:        0 kB
ShmemPmdMapped:        0 kB
FileHugePages:         0 kB
FilePmdMapped:         0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB
DirectMap4k:     1234567 kB
DirectMap2M:    98765432 kB
DirectMap1G:   167772160 kB
"""


# =============================================================================
# Sample /proc/cpuinfo Content
# =============================================================================

SAMPLE_CPUINFO = """processor	: 0
vendor_id	: AuthenticAMD
cpu family	: 25
model		: 1
model name	: AMD EPYC 7763 64-Core Processor
stepping	: 1
microcode	: 0xa0011d1
cpu MHz		: 2450.000
cache size	: 512 KB
physical id	: 0
siblings	: 128
core id		: 0
cpu cores	: 64
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 16
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 invpcid_single hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 invpcid cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload vgif v_spec_ctrl umip pku ospke vaes vpclmulqdq rdpid overflow_recov succor smca fsrm
bugs		: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass
bogomips	: 4900.17
TLB size	: 2560 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 48 bits physical, 48 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]

processor	: 1
vendor_id	: AuthenticAMD
cpu family	: 25
model		: 1
model name	: AMD EPYC 7763 64-Core Processor
stepping	: 1
microcode	: 0xa0011d1
cpu MHz		: 2450.000
cache size	: 512 KB
physical id	: 0
siblings	: 128
core id		: 1
cpu cores	: 64
"""


# =============================================================================
# Sample /proc/diskstats Content
# =============================================================================

SAMPLE_DISKSTATS = """   8       0 sda 12345678 123456 987654321 1234567 87654321 654321 876543210 8765432 0 5678901 10000000 0 0 0 0 12345 67890
   8       1 sda1 12345 1234 98765 1234 8765 6543 87654 8765 0 5678 10000 0 0 0 0 0 0
 259       0 nvme0n1 98765432 1234567 8765432109 12345678 76543210 7654321 765432109 76543210 0 45678901 89012345 0 0 0 0 123456 789012
 259       1 nvme0n1p1 98765 12345 8765432 123456 765432 76543 7654321 765432 0 456789 890123 0 0 0 0 0 0
 259       2 nvme1n1 87654321 1123456 7654321098 11234567 65432109 6543210 654321098 65432109 0 34567890 78901234 0 0 0 0 112345 678901
"""


# =============================================================================
# Sample Host Configuration
# =============================================================================

SAMPLE_HOSTS = ['192.168.1.1', '192.168.1.2', '192.168.1.3', '192.168.1.4']


# =============================================================================
# Factory Functions
# =============================================================================

def create_sample_cluster_info(
    num_hosts: int = 2,
    memory_gb_per_host: int = 256,
    cpu_cores_per_host: int = 64,
    logger: Optional[Any] = None
) -> Any:
    """
    Create a sample ClusterInformation object for testing.

    Args:
        num_hosts: Number of hosts in the cluster.
        memory_gb_per_host: Memory per host in GB.
        cpu_cores_per_host: CPU cores per host.
        logger: Logger to use (creates mock if None).

    Returns:
        ClusterInformation instance with mock data.
    """
    from mlpstorage.rules import ClusterInformation, HostInfo, HostMemoryInfo

    if logger is None:
        logger = MagicMock()

    host_info_list = []
    for i in range(num_hosts):
        host_info = HostInfo(
            hostname=f'host-{i + 1}',
            cpu=None,
            memory=HostMemoryInfo.from_total_mem_int(memory_gb_per_host * 1024**3)
        )
        host_info_list.append(host_info)

    return ClusterInformation(host_info_list=host_info_list, logger=logger)


def create_sample_benchmark_args(
    benchmark_type: str = 'training',
    command: str = 'run',
    model: str = 'unet3d',
    **kwargs
) -> Namespace:
    """
    Create sample benchmark arguments for testing.

    Args:
        benchmark_type: Type of benchmark ('training', 'checkpointing', 'vectordb').
        command: Command to run ('run', 'datagen', 'datasize').
        model: Model name for the benchmark.
        **kwargs: Additional arguments to set.

    Returns:
        Namespace with benchmark arguments.
    """
    # Base args that all commands share
    args = Namespace(
        debug=False,
        verbose=False,
        what_if=False,
        allow_invalid_params=False,
        results_dir='/tmp/test_results',
        loops=1,
        closed=False,
        program=benchmark_type,
        command=command,
        model=model,
    )

    if benchmark_type == 'training':
        args.accelerator_type = kwargs.get('accelerator_type', 'h100')
        args.num_client_hosts = kwargs.get('num_client_hosts', 2)
        args.client_host_memory_in_gb = kwargs.get('client_host_memory_in_gb', 256)
        args.num_accelerators = kwargs.get('num_accelerators', 8)
        args.num_processes = kwargs.get('num_processes', 8)
        args.max_accelerators = kwargs.get('max_accelerators', 16)
        args.hosts = kwargs.get('hosts', ['127.0.0.1'])
        args.data_dir = kwargs.get('data_dir', '/data/benchmark')
        args.exec_type = kwargs.get('exec_type', 'mpi')
        args.mpi_bin = kwargs.get('mpi_bin', 'mpirun')
        args.oversubscribe = kwargs.get('oversubscribe', False)
        args.allow_run_as_root = kwargs.get('allow_run_as_root', False)
        args.params = kwargs.get('params', None)
        args.dlio_bin_path = kwargs.get('dlio_bin_path', None)
        args.mpi_params = kwargs.get('mpi_params', None)
        args.checkpoint_folder = kwargs.get('checkpoint_folder', '/data/checkpoints')

    elif benchmark_type == 'checkpointing':
        args.num_processes = kwargs.get('num_processes', 8)
        args.hosts = kwargs.get('hosts', ['127.0.0.1'])
        args.client_host_memory_in_gb = kwargs.get('client_host_memory_in_gb', 512)
        args.checkpoint_folder = kwargs.get('checkpoint_folder', '/data/checkpoints')
        args.num_checkpoints_read = kwargs.get('num_checkpoints_read', 10)
        args.num_checkpoints_write = kwargs.get('num_checkpoints_write', 10)
        args.exec_type = kwargs.get('exec_type', 'mpi')
        args.mpi_bin = kwargs.get('mpi_bin', 'mpirun')
        args.oversubscribe = kwargs.get('oversubscribe', False)
        args.allow_run_as_root = kwargs.get('allow_run_as_root', True)
        args.params = kwargs.get('params', None)
        args.dlio_bin_path = kwargs.get('dlio_bin_path', None)
        args.mpi_params = kwargs.get('mpi_params', None)

    elif benchmark_type == 'vectordb':
        args.host = kwargs.get('host', '127.0.0.1')
        args.port = kwargs.get('port', 19530)
        args.collection = kwargs.get('collection', 'test_collection')
        args.dimension = kwargs.get('dimension', 1536)
        args.num_vectors = kwargs.get('num_vectors', 1000000)
        args.batch_size = kwargs.get('batch_size', 1000)

    # Override with any additional kwargs
    for key, value in kwargs.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    return args


def create_sample_benchmark_run_data(
    benchmark_type: str = 'training',
    model: str = 'unet3d',
    command: str = 'run',
    metrics: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    Create a sample BenchmarkRunData for testing.

    Args:
        benchmark_type: Type of benchmark.
        model: Model name.
        command: Command that was run.
        metrics: Optional metrics dictionary.
        **kwargs: Additional parameters.

    Returns:
        BenchmarkRunData instance.
    """
    from mlpstorage.rules import BenchmarkRunData
    from mlpstorage.config import BENCHMARK_TYPES

    # Map string to enum
    type_map = {
        'training': BENCHMARK_TYPES.training,
        'checkpointing': BENCHMARK_TYPES.checkpointing,
        'vector_database': BENCHMARK_TYPES.vector_database,
    }

    parameters = kwargs.get('parameters', {
        'model': {'name': model},
        'dataset': {
            'num_files_train': 42000,
            'data_folder': '/data/benchmark',
        },
        'workflow': {
            'train': True,
            'checkpoint': False,
        },
    })

    if metrics is None:
        metrics = {'train_au_percentage': [95.0, 94.5, 95.2]}

    cluster_info = kwargs.get('system_info')
    if cluster_info is None:
        cluster_info = create_sample_cluster_info()

    return BenchmarkRunData(
        benchmark_type=type_map.get(benchmark_type, BENCHMARK_TYPES.training),
        model=model,
        command=command,
        run_datetime=kwargs.get('run_datetime', '20250115_120000'),
        num_processes=kwargs.get('num_processes', 8),
        parameters=parameters,
        override_parameters=kwargs.get('override_parameters', {}),
        system_info=cluster_info,
        metrics=metrics,
        result_dir=kwargs.get('result_dir', '/tmp/results'),
        accelerator=kwargs.get('accelerator', 'h100'),
    )


# =============================================================================
# Sample Parameters
# =============================================================================

SAMPLE_TRAINING_PARAMETERS = {
    'model': {'name': 'unet3d'},
    'dataset': {
        'num_files_train': 42000,
        'num_subfolders_train': 0,
        'data_folder': '/data/unet3d',
        'format': 'npz',
        'num_samples_per_file': 1,
    },
    'reader': {
        'read_threads': 8,
        'computation_threads': 1,
        'prefetch_size': 2,
        'transfer_size': 262144,
    },
    'workflow': {
        'generate_data': False,
        'train': True,
        'checkpoint': True,
    },
    'checkpoint': {
        'checkpoint_folder': '/data/checkpoints',
    },
}

SAMPLE_CHECKPOINTING_PARAMETERS = {
    'model': {'name': 'llama3_8b'},
    'checkpoint': {
        'checkpoint_folder': '/data/checkpoints',
        'num_checkpoints_read': 10,
        'num_checkpoints_write': 10,
        'mode': 'default',
    },
    'workflow': {
        'generate_data': False,
        'train': False,
        'checkpoint': True,
    },
}
