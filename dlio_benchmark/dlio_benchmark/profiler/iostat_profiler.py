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
import signal
import subprocess as sp 

def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

class IostatProfiler(IOProfiler):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if IostatProfiler.__instance is None:
            IostatProfiler()
        return IostatProfiler.__instance

    def __init__(self):
        super().__init__()
        self.my_rank = self._args.my_rank
        self.devices = self._args.iostat_devices
        self.logfile = os.path.join(self._args.output_folder, 'iostat.json')
        """ Virtually private constructor. """
        if IostatProfiler.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            IostatProfiler.__instance = self

    def start(self):
        if self.my_rank == 0:
            # Open the logfile for writing
            self.logfile = open(self.logfile, 'w')
            
            # The following parameters are needed for the post-processing to parse correctly:
            #   -m: Display stats in MB
            #   -d: Display device utilisation report
            #   -x: Display extended statistics
            #   -t: Print the time for each report displayed
            #   -c: Display CPU utilization
            #   -y: Omit first report of stats since boot
            #   -o: Output in JSON format
            # If devs is empty, all devices are traced.
            cmd = f"iostat -mdxtcy -o JSON {' '.join(self.devices)} 1"
            cmd = cmd.split()
            self.process = sp.Popen(cmd, stdout=self.logfile, stderr=self.logfile)

    def stop(self):
        if self.my_rank == 0:
            self.logfile.flush()
            self.logfile.close()
            # If we send a stronger signal, the logfile json won't be ended correctly
            self.process.send_signal(signal.SIGINT) 
            # Might need a timeout here in case it hangs forever
            self.process.wait()

