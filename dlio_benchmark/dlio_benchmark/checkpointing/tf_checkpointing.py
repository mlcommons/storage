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
import tensorflow as tf

from dlio_benchmark.common.constants import MODULE_CHECKPOINT
from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
from dlio_benchmark.utils.utility import Profile, dft_ai

def get_tf_datatype(datatype):
    if datatype == "fp32":
        return tf.float32
    elif datatype == "fp16":
        return tf.float16
    elif datatype == "fp64":
        return tf.float64
    elif datatype == "bf16": # bfloat16
        return tf.bfloat16
    elif datatype == "int8":
        return tf.int8
    elif datatype == "uint8":
        return tf.uint8
    else:
        raise Exception(f"Invalid datatype {datatype}")

dlp = Profile(MODULE_CHECKPOINT)


class TFCheckpointing(BaseCheckpointing):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if TFCheckpointing.__instance is None:
            TFCheckpointing.__instance = TFCheckpointing()
        return TFCheckpointing.__instance
    
    @dft_ai.checkpoint.init
    def __init__(self):
        super().__init__("pb")

    @dlp.log
    def get_tensor_core(self, length, datatype="int8", randomize=True):
        tf_dtype = get_tf_datatype(datatype)
        if randomize:
            if tf_dtype in [tf.float16, tf.float32, tf.float64, tf.bfloat16]:
                tensor = tf.random.uniform(shape=(length,), minval=0, maxval=1, dtype=tf_dtype)
            elif tf_dtype == tf.int8:
                random_tensor = tf.random.uniform(shape=(length,), minval=-128, maxval=128, dtype=tf.int32)
                tensor = tf.cast(random_tensor, dtype=tf.int8)
            elif tf_dtype == tf.uint8:
                random_tensor = tf.random.uniform(shape=(length,), minval=0, maxval=256, dtype=tf.int32)
                tensor = tf.cast(random_tensor, dtype=tf.uint8)
            else:
                raise Exception(f"Datatype {tf_dtype} cannot be randomized for random tensor generation.")
        else:
            tensor = tf.ones((length), dtype=tf_dtype)
    
        # Convert tensor to variable to make it trackable for checkpointing
        return tf.Variable(tensor, trainable=False)

    @dlp.log
    def set_madvise_mergeable(self, tensor):
        return False

    @dft_ai.checkpoint.capture
    def save_state(self, suffix, state, fsync = False):
        name = self.get_name(suffix)
        checkpoint = tf.train.Checkpoint(**state)
        checkpoint.save(name)

    @dft_ai.checkpoint.restart
    def load_state(self, suffix, state):
        name = self.get_name(suffix)
        name = f"{name}-1"
        state = {k: tf.Variable(tf.zeros(shape=v.shape, dtype=v.dtype), trainable=False) for k, v in state.items()}
        checkpoint = tf.train.Checkpoint(**state)
        checkpoint.restore(name)
        assert len(state.keys()) != 0
        
    @dlp.log
    def save_checkpoint(self, epoch, step_number):
        super().save_checkpoint(epoch, step_number)

    @dlp.log
    def load_checkpoint(self, epoch, step_number):
        super().load_checkpoint(epoch, step_number)

    @dlp.log
    def finalize(self):
        super().finalize()
