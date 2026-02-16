"""
   Copyright (c) 2022, UChicago Argonne, LLC
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
#!/usr/bin/env python
from collections import namedtuple
import unittest

from dlio_benchmark.postprocessor import DLIOPostProcessor
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

class TestDLIOPostProcessor(unittest.TestCase):

    def create_DLIO_PostProcessor(self, args):
        return DLIOPostProcessor(args)

    def test_process_loading_and_processing_times(self):
        args = {
            'output_folder': 'tests/test_data',
            'name': '',
            'num_proc': 2,
            'epochs': 2,
            'do_eval': False,
            'do_checkpoint': False,
            'batch_size': 4,
            'batch_size_eval': 1,
            'record_size':234560851
        }
        args = namedtuple('args', args.keys())(*args.values())
        postproc = self.create_DLIO_PostProcessor(args)

        postproc.process_loading_and_processing_times()

        # Expected values: {
        #   'samples/s': {'mean': '3.27', 'std': '2.39', 'min': '1.33', 'median': '2.33', 'p90': '7.60', 'p99': '8.00', 'max': '8.00'}, 
        #   'sample_latency': {'mean': '3.27', 'std': '2.39', 'min': '1.33', 'median': '2.33', 'p90': '7.60', 'p99': '8.00', 'max': '8.00'}, 
        #   'avg_process_loading_time': '21.00', 
        #   'avg_process_processing_time': '21.00'
        # }
        self.assertEqual(postproc.overall_stats['samples/s']['mean'], '5.10')
        self.assertEqual(postproc.overall_stats['avg_process_loading_time'], '7.78')
        self.assertEqual(postproc.overall_stats['avg_process_processing_time'], '65.87')



if __name__ == '__main__':
    unittest.main()
