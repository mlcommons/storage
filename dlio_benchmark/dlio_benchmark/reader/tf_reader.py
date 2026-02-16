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
import math

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.utils.utility import utcnow, Profile
from dlio_benchmark.common.enumerations import Shuffle
from dlio_benchmark.reader.reader_handler import FormatReader
import tensorflow as tf

dlp = Profile(MODULE_DATA_READER)


class TFReader(FormatReader):
    """
    Reader for TFRecord files.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self._resized_image = tf.convert_to_tensor(self._args.resized_image, dtype=tf.uint8)        
        self._dataset = None

    @dlp.log
    def open(self, filename):
        pass

    @dlp.log
    def close(self, filename):
        pass

    @dlp.log
    def get_sample(self, filename, sample_index):
        pass

    @dlp.log
    def resize_sample(self, filename, sample_index):
        pass

    @dlp.log
    def _parse_image(self, serialized):
        """
        performs deserialization of the tfrecord.
        :param serialized: is the serialized version using protobuf
        :return: deserialized image and label.
        """
        features = \
            {
                'image': tf.io.FixedLenFeature([], tf.string),
                'size': tf.io.FixedLenFeature([], tf.int64)
            }
        parsed_example = tf.io.parse_example(serialized=serialized, features=features)
        # Get the image as raw bytes.
        #image_raw = parsed_example['image']
        #dimension = tf.cast(parsed_example['size'], tf.int32).numpy()
        # Decode the raw bytes so it becomes a tensor with type.
        #image_tensor = tf.io.decode_raw(image_raw, tf.uint8)
        #size = dimension * dimension
        #dlp.update(image_size=size)
        #image_tensor = tf.io.decode_image(image_raw)
        #resized_image = tf.convert_to_tensor(self._args.resized_image, dtype=tf.uint8)
        return self._resized_image

    @dlp.log
    def next(self):
        self.logger.debug(f"{utcnow()} Reading {len(self._file_list)} files thread {self.thread_index} rank {self._args.my_rank}")

        # @ray: solution to prevent error when tf.data.Dataset cannot find files provided within self._file_list
        # the use case is usually as follow: user is providing workload.dataset.num_files_eval=0 since they do not
        # want to do any evaluation
        # since this method (`next`) requires to return a iterator, we will just return an empty array where array
        # itself is an iterator
        if len(self._file_list) == 0:
            return []

        filenames = tf.data.Dataset.list_files(self._file_list, shuffle=False)
        # sharding in the file list if we have enought files. 
        if (len(self._file_list) >= self._args.comm_size):
            filenames = filenames.shard(num_shards=self._args.comm_size, index=self._args.my_rank)
            self.logger.debug(f"{utcnow()} shard {filenames} files index {self._args.my_rank} number {self._args.comm_size}")
        
        self._dataset = tf.data.TFRecordDataset(filenames=filenames, buffer_size=self._args.transfer_size,
                                                num_parallel_reads=self._args.read_threads)
				  
        if self._args.sample_shuffle != Shuffle.OFF:
            if self._args.sample_shuffle == Shuffle.SEED:
                self._dataset = self._dataset.shuffle(buffer_size=self._args.shuffle_size,
                                          seed=self._args.seed)
            else:
                self._dataset = self._dataset.shuffle(buffer_size=self._args.shuffle_size)
		
        # shard the dataset if it is not done already.
        if (len(self._file_list) < self._args.comm_size):
            self._dataset =  self._dataset.shard(num_shards=self._args.comm_size, index=self._args.my_rank)
	
        self._dataset = self._dataset.batch(self.batch_size, drop_remainder=True)
        self._dataset = self._dataset.map(
                lambda x: tf.py_function(func=self._parse_image, inp=[x], Tout=[tf.uint8]),
                num_parallel_calls=self._args.computation_threads)

        self._dataset = self._dataset.repeat()
        total = math.floor(len(self._file_list)/self._args.comm_size / self.batch_size * self._args.num_samples_per_file)
        
        return self._dataset.take(total*self._args.epochs).prefetch(buffer_size=self._args.prefetch_size)
    
    @dlp.log
    def read_index(self, image_idx, step):
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()
    
    def is_index_based(self):
        return False

    def is_iterator_based(self):
        return True
