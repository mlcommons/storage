.. _custom_data_reader: 

Creating a Custom Data Reader
==============================

Within DLIO Benchmark we can define custom data reader implementations. 
This feature allows us to extend DLIO Benchmark with new data reader implementation easily without changing existing code.
To achieve this developers have to take the following main steps.

1. Write their custom data reader.
2. Define workflow configuration.
3. Run the workload with custom data reader.

Defining custom data reader
--------------------------------

In this section, we will describe how to write a custom data reader.
To write a data reader, one needs to implement `FormatReader` Class.
This data reader needs to be added `<ROOT>/dlio_benchmark/plugins/experimental/src/reader`.
A complete examples can be seen at `<ROOT>/dlio_benchmark/reader/`

- For NPZ: npz_reader.py
- For TFRecord: tf_reader.py
- For HDF5: hdf5_reader.py
  
Say we store the custom data reader for pytorch into `<ROOT>/dlio_benchmark/plugins/experimental/src/reader/custom_npz_reader.py`

.. code-block:: python

    from dlio_benchmark.reader.reader_handler import FormatReader
    
    # MAKE SURE the name of class is unique
    class CustomNPZReader(FormatReader):
        
        def __init__(self, dataset_type, thread_index, epoch):
            super().__init__(dataset_type, thread_index)

        # define how to open the NPZ file
        def open(self, filename):
            super().open(filename)
            return np.load(filename, allow_pickle=True)["x"]
        
        # define how to close the NPZ file
        def close(self, filename):
            super().close(filename)

        # define how to read the sample
        def get_sample(self, filename, sample_index):
            super().get_sample(filename, sample_index)
            image = self.open_file_map[filename][..., sample_index]
            dlp.update(image_size=image.nbytes)

        # Used in Iterative data loader
        # THIS NEED NOT CHANGE AS WE HAVE A COMMON LOGIC UNLESS VERY SPECIFIC LOGIC OF ITERATION NEEDED
        def next(self):
            for batch in super().next():
                yield batch

        # Used in index based data loader
        # THIS NEED NOT CHANGE AS WE HAVE A COMMON LOGIC UNLESS VERY SPECIFIC LOGIC OF ITERATION NEEDED
        def read_index(self, image_idx, step):
            return super().read_index(image_idx, step)

        # Perform Cleanup as required.
        def finalize(self):
            return super().finalize()


Define workflow configuration.
------------------------------

In this section, we will detail how to create a custom workflow configuration for the new data reader in DLIO Benchmark.
The workload configuration for plugins exists in `<ROOT>/dlio_benchmark/plugins/experimental`.
You can copy an existing configuration from `<ROOT>/dlio_benchmark/configs/workload` and modify it for your custom data reader.
Main changes to the workflow configuration are:

.. code-block:: yaml

    # Rest remains as it is
    reader:
        reader_classname: dlio_benchmark.plugins.experimental.src.reader.custom_npz_reader.CustomNPZReader


In the above configuration, `reader_classname` should point to FQN of the class (as in the PYTHONPATH).


Run the workload with custom data reader.
------------------------------------------

To run the custom data reader, we have to define the plugin folder as the custom config folder. 
This is described in the :ref:`run` page. 
We need to pass path `plugins/experimental/configs` as the path.