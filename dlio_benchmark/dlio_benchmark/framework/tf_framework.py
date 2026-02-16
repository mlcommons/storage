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

from dlio_benchmark.common.constants import MODULE_AI_FRAMEWORK
from dlio_benchmark.utils.utility import Profile, dft_ai
from dlio_benchmark.framework.framework import Framework
from dlio_benchmark.profiler.profiler_factory import ProfilerFactory
from dlio_benchmark.common.enumerations import FrameworkType, Profiler, DatasetType, MetadataType, \
    DataLoaderType

import tensorflow as tf
from tensorflow.python.framework import errors

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

dlp = Profile(MODULE_AI_FRAMEWORK)


class TFFramework(Framework):
    __instance = None

    @dlp.log_init
    def __init__(self, profiling):
        super().__init__()
        self.profiling = profiling
        # TODO: Temporary fix, need to separate the iostat profiler (needed for report gen) and the others
        if profiling:
            if self.args.profiler != Profiler.IOSTAT:
                self.tensorboard = ProfilerFactory.get_profiler(Profiler.NONE)
            else:
                self.tensorboard = ProfilerFactory.get_profiler(Profiler.TENSORBOARD)
        self.reader_handler = None

    @dlp.log
    def init_loader(self, format_type, epoch=0, data_loader=None):
        if data_loader is None:
            data_loader = DataLoaderType.TENSORFLOW
        super().init_loader(format_type, epoch, data_loader)
    @dlp.log
    def get_type(self):
        return FrameworkType.TENSORFLOW

    @staticmethod
    def get_instance(profiling):
        """ Static access method. """
        if TFFramework.__instance is None:
            TFFramework.__instance = TFFramework(profiling)
        return TFFramework.__instance

    @dlp.log
    def start_framework_profiler(self):
        if self.profiling:
            self.tensorboard.start()

    @dlp.log
    def stop_framework_profiler(self):
        # if self.profiling:
        #    self.tensorboard.stop()
        pass

    @dlp.log
    def trace_object(self, string, step, r):
        pass  # tf.profiler.experimental.Trace(string, step_num=step, _r=r)

    @dft_ai.compute
    def compute(self, batch, epoch_number, step, computation_time):
        return self.model(batch, computation_time)
        # tf.function(self.model)(epoch_number, step, computation_time)

    @dlp.log
    def get_loader(self, dataset_type=DatasetType.TRAIN):
        if dataset_type == DatasetType.TRAIN:
            return self.reader_train
        else:
            return self.reader_valid

    @dlp.log
    def is_nativeio_available(self):
        return True

    @dlp.log
    def create_node(self, id, exist_ok=False):
        tf.io.gfile.makedirs(id)
        return True

    @dlp.log
    def get_node(self, id):
        if tf.io.gfile.exists(id):
            if tf.io.gfile.isdir(id):
                return MetadataType.DIRECTORY
            else:
                return MetadataType.FILE
        else:
            return None

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        try:
            if not use_pattern:
                return tf.io.gfile.listdir(id)
            else:
                return tf.io.gfile.glob(id)
        except errors.NotFoundError:
            return []

    @dlp.log
    def delete_node(self, id):
        tf.io.gfile.rmtree(id)
        return True

    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        with tf.io.gfile.GFile(id, "w") as fd:
            fd.write(data)

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        with tf.io.gfile.GFile(id, "r") as fd:
            data = fd.read()
        return data

    @dlp.log
    def isfile(self, id):
        return tf.io.gfile.exists(id) and not tf.io.gfile.isdir(id)
