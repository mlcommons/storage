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

from dlio_benchmark.profiler.io_profiler import IOProfiler
import os

class DarshanProfiler(IOProfiler):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if DarshanProfiler.__instance is None:
            DarshanProfiler()
        return DarshanProfiler.__instance

    def __init__(self):
        super().__init__()

        """ Virtually private constructor. """
        if DarshanProfiler.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            DarshanProfiler.__instance = self
            
        os.environ["DARSHAN_MOD_ENABLE"]="DXT_POSIX,DXT_MPIIO"            
        os.environ["DARSHAN_LOG_DIR"] = self._args.output_folder
        os.environ["DARSHAN_LOGFILE"] = self._args.output_folder + "/dlio_benchmark.darshan"

        
    def start(self):
        os.environ["DARSHAN_DISABLE"] = "0"

    def stop(self):
        os.environ['DARSHAN_DISABLE'] = '1'
