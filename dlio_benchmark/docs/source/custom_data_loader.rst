.. _custom_data_loader: 

Creating a Data Loader Plugin
==============================

Within DLIO Benchmark we can define custom data loader implementations. 
This feature allows us to extend DLIO Benchmark with new data loader implementation easily without changing existing code.
To achieve this developers have to take the following main steps.

1. Write their custom data loader.
2. Define workflow configuration.
3. Run the workload with custom data loader.

Write their custom data loader.
--------------------------------

In this section, we will describe how to write the custom data loader.
To write a data loader you need to implement `BaseDataLoader` Class.
This data loader needs to added `<ROOT>/dlio_benchmark/plugins/experimental/src/data_loader`.
A complete examples can be seen at `<ROOT>/dlio_benchmark/data_loader/`

- For PyTorch: torch_data_loader.py
- For TensorFlow: tf_data_loader.py
- For Nvidia Dali: dali_data_loader.py
  
Say we store the custom data loader for pytorch into `<ROOT>/dlio_benchmark/plugins/experimental/src/data_loader/pytorch_custom_data_loader.py`

.. code-block:: python

    import torch
    from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader

    # MAKE SURE the name of class is unique
    class CustomTorchDataLoader(BaseDataLoader):
    
        def __init__(self, format_type, dataset_type, epoch_number):
            super().__init__(format_type, dataset_type, epoch_number, DataLoaderType.PYTORCH)

        
        def read(self):
            batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
            # Define your dataset definition here.
            self._dataset = DataLoader(PYTORCH_DATASET,
                                    batch_size=batch_size,
                                    sampler=PYTORCH_SAMPLER,
                                    num_workers=self._args.read_threads,
                                    pin_memory=True,
                                    drop_last=True,
                                    worker_init_fn=WORKER_INIT_FN)

        def next(self):
            # THIS PART OF CODE NEED NOT CHANGE
            # This iterates and gets the batch of images.
            super().next()
            total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
            for batch in self._dataset:
                yield batch

        def finalize(self):
            # Perform any cleanup as required.

Additionally, you may need to define your own PyTorch Dataset.

.. code-block:: python

    # MAKE SURE the name of class is unique
    class CustomTorchDataset(Dataset):
       
        def __init__(self, format_type, dataset_type, epoch, num_samples, num_workers, batch_size):
            self.format_type = format_type
            self.dataset_type = dataset_type
            self.epoch_number = epoch
            self.num_samples = num_samples
            self.reader = None
            self.num_images_read = 0
            self.batch_size = batch_size
            if num_workers == 0:
                self.worker_init(-1)
        
        def worker_init(self, worker_id):
            # If you wanna use Existing Data Reader.
            self.reader = ReaderFactory.get_reader(type=self.format_type,
                                                dataset_type=self.dataset_type,
                                                thread_index=worker_id,
                                                epoch_number=self.epoch_number)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, image_idx):
            # Example existing reader call.
            self.num_images_read += 1
            step = int(math.ceil(self.num_images_read / self.batch_size))
            return self.reader.read_index(image_idx, step)



Define workflow configuration.
------------------------------

In this section, we will detail how to create a custom workflow configuration for DLIO Benchmark.
The workload configuration for plugins exists in `<ROOT>/dlio_benchmark/plugins/experimental`.
You can copy an existing configuration from `<ROOT>/dlio_benchmark/configs/workload` and modify it for your custom data loader.
Main changes to the workflow configuration are:

.. code-block:: yaml

    # Rest remains as it is
    reader:
        data_loader_classname: dlio_benchmark.plugins.experimental.src.data_loader.pytorch_custom_data_loader.CustomTorchDataLoader
        data_loader_sampler: iterative/index # CHOOSE the correct sampler.


In the above configuration, `data_loader_classname` should point to FQN of the class (as in the PYTHONPATH).
Also, `data_loader_sampler` should be set to `iterative` if the data loader implements a iterative reading and `index` should be used if data loader is using an index based reading.
The `torch_data_loader.py` is an example of index based data loader and `tf_data_loader.py` is an example of iterative data loader.


Run the workload with custom data loader.
------------------------------------------

To run the custom data loader, we have to define the plugin folder as the custom config folder. 
This is described in the :ref:`run` page. 
We need to pass path `plugins/experimental/configs` as the path.