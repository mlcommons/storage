"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from enum import Enum


class CheckpointMechanismType(Enum):
    """
    Different Checkpoint mechanisms.
    """
    NONE = 'none'
    CUSTOM = 'custom'
    TF_SAVE = 'tf_save'
    PT_SAVE = 'pt_save'
    PT_S3_SAVE = 'pt_s3_save'

    def __str__(self):
        return self.value

class CheckpointLocationType(Enum):
    """
    Different types of Checkpointing Locations
    """
    RANK_ZERO = 'rank_zero'
    ALL_RANKS = 'all_ranks'

    def __str__(self):
        return self.value

class CheckpointModeType(Enum):
    """
    Different types of Checkpointing Modes
    """
    SUBSET = 'subset'
    DEFAULT = 'default'

    def __str__(self):
        return self.value

class StorageType(Enum):
    """
    Different types of underlying storage
    """
    LOCAL_FS = 'local_fs'
    PARALLEL_FS = 'parallel_fs'
    S3 = 's3'

    def __str__(self):
        return self.value

class MetadataType(Enum):
    """
    Different types of storage metadata
    """
    FILE = 'file'
    DIRECTORY = 'directory'
    S3_OBJECT = 's3_object'

    def __str__(self):
        return self.value

class NamespaceType(Enum):
    """
    Different types of Storage Namespace
    """
    FLAT = 'flat'
    HIERARCHICAL = 'Hierarchical'

    def __str__(self):
        return self.value

class DatasetType(Enum):
    """
    Training and Validation
    """
    TRAIN = 'train'
    VALID = 'valid'

    def __str__(self):
        return self.value

    @staticmethod
    def get_enum(value):
        if DatasetType.TRAIN.value == value:
            return DatasetType.TRAIN
        elif DatasetType.VALID.value == value:
            return DatasetType.VALID

class FrameworkType(Enum):
    """
    Different Computation Type for training loop.
    """
    TENSORFLOW = 'tensorflow'
    PYTORCH = 'pytorch'

    def __str__(self):
        return self.value

class ComputationType(Enum):
    """
    Different Computation Type for training loop.
    """
    NONE = 'none'
    SYNC = 'sync'
    ASYNC = 'async'

class FormatType(Enum):
    """
    Format Type supported by the benchmark.
    """
    TFRECORD = 'tfrecord'
    HDF5 = 'hdf5'
    CSV = 'csv'
    NPZ = 'npz'
    NPY = 'npy'
    HDF5_OPT = 'hdf5_opt'
    JPEG = 'jpeg'
    PNG = 'png'
    INDEXED_BINARY = 'indexed_binary'
    MMAP_INDEXED_BINARY = 'mmap_indexed_binary'
    SYNTHETIC = 'synthetic'
    
    def __str__(self):
        return self.value

    @staticmethod
    def get_enum(value):
        if FormatType.TFRECORD.value == value:
            return FormatType.TFRECORD
        elif FormatType.HDF5.value == value:
            return FormatType.HDF5
        elif FormatType.CSV.value == value:
            return FormatType.CSV
        elif FormatType.NPZ.value == value:
            return FormatType.NPZ
        elif FormatType.NPY.value == value:
            return FormatType.NPY            
        elif FormatType.HDF5_OPT.value == value:
            return FormatType.HDF5_OPT
        elif FormatType.JPEG.value == value:
            return FormatType.JPEG
        elif FormatType.PNG.value == value:
            return FormatType.PNG
        elif FormatType.INDEXED_BINARY.value == value:
            return FormatType.INDEXED_BINARY
        elif FormatType.MMAP_INDEXED_BINARY.value == value:
            return FormatType.MMAP_INDEXED_BINARY
        elif FormatType.SYNTHETIC.value == value:
            return FormatType.SYNTHETIC

class DataLoaderType(Enum):
    """
    Framework DataLoader Type
    """
    TENSORFLOW='tensorflow'
    PYTORCH='pytorch'
    DALI='dali'
    NATIVE_DALI='native_dali'
    CUSTOM='custom'
    NONE='none'
    SYNTHETIC='synthetic'
    
    def __str__(self):
        return self.value


class DataLoaderSampler(Enum):
    """
    Framework DataLoader Sampler Type
    """
    ITERATIVE = 'iterative'
    INDEX = 'index'
    NONE = 'none'

    def __str__(self):
        return self.value

class LoggerType(Enum):
    """
    Logger types supported by the benchmark.
    """
    DEFAULT = 'default'
    DFTRACER = 'dftracer'

    def __str__(self):
        return self.value

class Profiler(Enum):
    """
    Profiler types supported by the benchmark.
    """
    NONE = 'none'
    IOSTAT = 'iostat'
    DARSHAN = 'darshan'
    TENSORBOARD = 'tensorboard'

    def __str__(self):
        return self.value

class Shuffle(Enum):
    """
    Shuffle mode for files and memory.
    """
    OFF = 'off'
    SEED = 'seed'
    RANDOM = 'random'

    def __str__(self):
        return self.value

class ReadType(Enum):
    """
    Type of read to be performed in the benchmark. 
    - On Demand: loading data in a batch-by-batch fashion
    - In Memory: loading data all at once in the beginning. 
    """
    IN_MEMORY = 'memory'
    ON_DEMAND = 'on_demand'

    def __str__(self):
        return self.value

class FileAccess(Enum):
    """
    File access mode.
    - Multi = save dataset into multiple files
    - Shared = save everything in a single file
    - Collective = specific for the shared case, when we want to do collective I/O. Typically used for a huge file with small objects. 
      One thread T reads from disk and the other threads read from T's memory, which is used as a cache.
    """
    MULTI = 'multi'
    SHARED = 'shared'
    # TO(HZ): I see currently, this collective mode is not used. It might be good to separate it out
    COLLECTIVE = 'collective'
    MPIO = 'mpio'
    POSIX = 'posix'

    def __str__(self):
        return self.value

    @staticmethod
    def get_enum(value):
        if FileAccess.MPIO.value == value:
            return FileAccess.MPIO
        elif FileAccess.POSIX.value == value:
            return FileAccess.POSIX
        elif FileAccess.MULTI.value == value:
            return FileAccess.MULTI
        elif FileAccess.SHARED.value == value:
            return FileAccess.SHARED
        elif FileAccess.COLLECTIVE.value == value:
            return FileAccess.COLLECTIVE
                   
class Compression(Enum):
    """
    Different Compression Libraries.
    """
    NONE = 'none'
    GZIP = 'gzip'
    LZF = 'lzf'
    BZIP2 = 'bz2'
    ZIP = 'zip'
    XZ = 'xz'

    def __str__(self):
        return self.value

class MPIState(Enum):
    """
    MPI State for forked and spawned processes.
    """
    UNINITIALIZED = 0
    MPI_INITIALIZED = 1
    CHILD_INITIALIZED = 2
   
    @staticmethod
    def get_enum(value):
        if MPIState.UNINITIALIZED.value == value:
            return MPIState.UNINITIALIZED
        elif MPIState.MPI_INITIALIZE.value == value:
            return MPIState.MPI_INITIALIZE
        elif MPIState.CHILD_INITIALIZED.value == value:
            return MPIState.CHILD_INITIALIZED
