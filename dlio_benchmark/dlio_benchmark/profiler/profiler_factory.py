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

from dlio_benchmark.profiler.iostat_profiler import IostatProfiler
from dlio_benchmark.common.error_code import ErrorCodes
from dlio_benchmark.profiler.darshan_profiler import DarshanProfiler
from dlio_benchmark.profiler.no_profiler import NoProfiler
from dlio_benchmark.common.enumerations import Profiler
from dlio_benchmark.profiler.tf_profiler import TFProfiler

class ProfilerFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_profiler(type):
        if type == Profiler.NONE:
            return NoProfiler()
        if type == Profiler.IOSTAT:
            return IostatProfiler.get_instance()
        elif type == Profiler.DARSHAN:
            return DarshanProfiler.get_instance()
        elif type == Profiler.TENSORBOARD:
            return TFProfiler.get_instance()
        else:
            raise Exception(str(ErrorCodes.EC1001))
