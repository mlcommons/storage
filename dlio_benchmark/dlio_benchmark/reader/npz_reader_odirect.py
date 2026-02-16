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
from dlio_benchmark.reader.npy_reader_odirect import NPYReaderODirect
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_DATA_READER)


class NPZReaderODIRECT(NPYReaderODirect):
    """
    O_DIRECT Reader for NPZ files
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch, alignment=4096):
        super().__init__(dataset_type, thread_index, epoch)
        self.alignment = alignment

    @dlp.log
    def open(self, filename):
        FormatReader.open(self, filename)
        data = self.odirect_read(filename)
        data = self.parse_npz(data)["x"]
        return data
    
    def parse_npz(self, mem_view):
        files = {}
        pos = 0

        while pos < len(mem_view):
            # Verify magic
            local_header_signature = mem_view[pos:pos+4].tobytes()
            if local_header_signature != b'\x50\x4b\x03\x04':
                break

            compressed_size = struct.unpack('<I', mem_view[pos+18:pos+22].tobytes())[0]
            uncompressed_size = struct.unpack('<I', mem_view[pos+22:pos+26].tobytes())[0]
            filename_len = struct.unpack('<H', mem_view[pos+26:pos+28].tobytes())[0]            
            extra_len = struct.unpack('<H', mem_view[pos+28:pos+30].tobytes())[0]
            filename = mem_view[pos+30:pos+30+filename_len].tobytes().decode('utf-8') 

            # skip to data offset
            pos += 30 + filename_len + extra_len
            if not filename.endswith('.npy'):
                raise ValueError(f"Unexpected file in npz: {filename}")
            filename = filename[:-4]  
                        
            compressed_data = mem_view[pos:pos+compressed_size]
            pos += compressed_size
            
            if compressed_size == uncompressed_size:
                uncompressed_data = compressed_data
            else:
                uncompressed_data = zlib.decompress(compressed_data)

            files[filename] = self.parse_npy(uncompressed_data)
        return files