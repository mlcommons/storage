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
import numpy as np

import os 
import ctypes
import time
import struct
import zlib

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_DATA_READER)


class NPYReaderODirect(FormatReader):
    """
    O_DIRECT Reader for NPY files
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch, alignment=4096):
        super().__init__(dataset_type, thread_index)
        self.alignment = alignment

    @dlp.log
    def open(self, filename):
        super().open(filename)
        data = self.odirect_read(filename)
        data = self.parse_npy(data)
        return data

    def odirect_read(self, filepath):
        try:
            # Open the file with O_DIRECT
            fd = os.open(filepath, os.O_RDONLY | os.O_DIRECT)

            # Get the file size
            file_size = os.path.getsize(filepath)

            # Calculate the buffer size, aligned to the given alignment
            buffer_size = ((file_size + self.alignment - 1) // self.alignment) * self.alignment

            # Allocate the aligned buffer
            buf = self.allocate_aligned_buffer(buffer_size)
            mem_view = memoryview(buf)

            # Read the file into the buffer
            bytes_read = os.readv(fd, [mem_view[0:buffer_size]])
            if bytes_read != file_size:
                raise IOError(f"Could not read the entire file. Expected {file_size} bytes, got {bytes_read} bytes")
            return mem_view
        finally:
            os.close(fd)
            
    def allocate_aligned_buffer(self, size):
        buf_size = size + (self.alignment - 1)
        raw_memory = bytearray(buf_size)
        ctypes_raw_type = (ctypes.c_char * buf_size)
        ctypes_raw_memory = ctypes_raw_type.from_buffer(raw_memory)
        raw_address = ctypes.addressof(ctypes_raw_memory)
        offset = raw_address % self.alignment
        offset_to_aligned = (self.alignment - offset) % self.alignment
        ctypes_aligned_type = (ctypes.c_char * (buf_size - offset_to_aligned))
        ctypes_aligned_memory = ctypes_aligned_type.from_buffer(raw_memory, offset_to_aligned)
        return ctypes_aligned_memory
    
    @dlp.log
    def close(self, filename):
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        image = self.open_file_map[filename][..., sample_index]
        dlp.update(image_size=image.nbytes)

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
    
    # optimized to use in-ram buffer with 0 copy
    def parse_npy(self, mem_view):
        # Verify the magic string
        if mem_view[:6].tobytes() != b'\x93NUMPY':
            raise ValueError("This is not a valid .npy file.")

        # Read version information
        major, minor = struct.unpack('<BB', mem_view[6:8].tobytes())
        if major == 1:
            header_len = struct.unpack('<H', mem_view[8:10].tobytes())[0]
            header = mem_view[10:10 + header_len].tobytes()
        elif major == 2:
            header_len = struct.unpack('<I', mem_view[8:12].tobytes())[0]
            header = mem_view[12:12 + header_len].tobytes()
        else:
            raise ValueError(f"Unsupported .npy file version: {major}.{minor}")

        # Parse the header
        header_dict = eval(header.decode('latin1'))
        dtype = np.dtype(header_dict['descr'])
        shape = header_dict['shape']
        fortran_order = header_dict['fortran_order']

        # Calculate the data offset
        data_offset = (10 + header_len) if major == 1 else (12 + header_len)
        data_size = np.prod(shape) * dtype.itemsize

        # Load the array data
        data = np.ndarray(shape, dtype=dtype, buffer=mem_view[data_offset:data_offset + data_size])

        # If the array is in Fortran order, convert it
        if fortran_order:
            data = np.asfortranarray(data)
        return data