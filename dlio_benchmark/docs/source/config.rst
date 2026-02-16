.. _yaml: 

DLIO Configuration
==============================================
The characteristics of a workload is specified through a YAML file. This file will then be read by `DLIO` to setup the benchmark. Below is an example of such a YAML file. 

.. code-block:: yaml
  
  model: unet3d
    model_size_bytes: 99153191


  framework: pytorch

  workflow:
    generate_data: False
    train: True
    checkpoint: True

  dataset: 
    data_folder: data/unet3d/
    format: npz
    num_files_train: 168
    num_samples_per_file: 1
    record_length_bytes: 146600628
    record_length_bytes_stdev: 68341808
    record_length_bytes_resize: 2097152
    
  reader: 
    data_loader: pytorch
    batch_size: 4
    read_threads: 4
    file_shuffle: seed
    sample_shuffle: seed

  train:
    epochs: 5
    computation_time: 1.3604

  checkpoint:
    checkpoint_folder: checkpoints/unet3d
    checkpoint_after_epoch: 5
    epochs_between_checkpoints: 2


A `DLIO` YAML configuration file contains following sections: 

* **model** - specifying the name of the model. This is simply an indentifyer of the configuration file. It does not have impact on the actual simulation. 
* **framework** - specifying the framework to use for the benchmark, available options: tensorflow, pytorch
* **workflow** - specifying what workflow operations to execute in the pipeline. Workflow operations include: dataset generation (``generate_data``), training (``train``), evaluation (``evaluation``), checkpointing (``checkpoint``), debugging (``debug``), etc. 
* **dataset** - specifying all the information related to the dataset. 
* **reader** - specifying the configuration for data loading, such as data_loader, number of workers, etc. 
* **train** - specifying the setup for training
* **evaluation** - specifying the setup for evaluation. 
* **checkpoint** - specifying the setup for checkpointing. 
* **profiling** - specifying the setup for profiling

More built-in examples can be found in the `workload`_ folder. One can also create custom configuration file. How to load custom configuration file can be found in :ref:`run`. 

model
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - name 
     - default
     - The name of the model
   * - type
     - default
     - A string that specifies the type of the model, such as transformer, CNN, etc.
   * - model_size_bytes
     - 10240
     - The size of the model parameters per GPU in bytes
   * - model_datatype
     - fp16
     - the datatype of the model parameters. Available options are fp16, fp32, int8, uint8, bf16. 
   * - optimizer_datatype
     - fp32
     - the datatype of the optimizer parameters. Available options are fp16, fp32, int8, uint8, bf16. 
   * - optimization_groups
     - []
     - List of optimization group tensors. Use Array notation for yaml.
   * - num_layers
     - -1
     - Number of layers to checkpoint. Each layer would be checkpointed separately.
   * - layer_parameters
     - []
     - List of parameters per layer. This is used to perform I/O per layer. 
   * - parallelism
     - {tensor: 1, pipeline: 1, data: -1, zero_stage: 0}
     - Parallelism configuration for the model. 
   * - transformer
     - {hidden_size: 2048, ffn_hidden_size: 8196, vocab_size: 32000, num_attention_heads: 32, num_kv_heads: 8}
     - Transformer layer configuration for the model.

The model information is used to determine the checkpoint files. 
The user can specify the model architecture using either optimizaton_groups & layer_parameters, or by specifying the transformer configuration. 

The ``optimization_groups`` is a list of tensors that are grouped together for optimization. Suppose optimization_groups is specified as [1024, 528], 
each rank will write the following tensors to the checkpoint file: {"0": {"a": array of 1024, "b": array of 1024}, "1": {"a": array of 528, "b": array of 528}}. The total size of the tensor will be 1024*2 + 528*2. The ``layer_parameters`` is a list of parameters per layer. The ``num_layers`` is used to specify the number of layers to checkpoint. Each layer would be checkpointed separately. 
Suppose layer_parameters is [1024, 2048], each rank in the tensor parallelism group will write the following tensors to the checkpoint file: 
{'0': array of 1024/TP, "1": array of (2048/TP)}. Please notice the difference in how the optimization groups and layer parameters are treated internally.

We do not suggest the users to specify the model architeure in this way. Instead, we suggest the users to specify the transformer configuration directly which is more intuitive. 
The ``transformer`` configuration is used to specify the hidden size, FFN hidden size, vocab size, number of attention heads and number of kv heads for the transformer layer, which together determined the 
optimization_groups and layer_parameters. 

.. note::

  By default, if ``parallelism.data`` is not set explicitly, it would be -1. The actual data parallelism size will 
  be determined internally: 

  ```math
  data\_parallelism = \frac{world\_size}{pipeline\_parallelism*tensor\_parallelism}
  ```
  If ``parallelism.data`` is set explicitly, the value provided by the user will be used. In this case, if ``world_size`` < ``data_parallelism``*``pipeline_parallelism``*``tensor_parallelism``, only 
  part of the data will be written (``world_size`` of ``data_parallelism*pipeline_parallelism*tensor_parallelism``). 
  This is useful if one would like to do testing at smaller scale as a subset of a larger scale simulation. In this case, one has to set
  ``checkpoint.mode`` to be ``subset``.

.. attention::

  Please note that if optimization_groups and layer_parameters are specified, the transformer configuration will be ignored. But we 
  always suggest to specify the transformer configuration for better readability.

  Please also note that ZeRO stage 3 is not compatiable with ``parallelism.pipeline > 1``.  

.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - hidden_size
     - 2048
     - Hidden dimension of the transformer layer.
   * - ffn_hidden_size
     - 8196
     - FFN hidden dimension 
   * - vocab_size
     - 32000
     - vocab size for the embedding layer
   * - num_attention_heads:
     - 32
     - number of attention heads
   * - num_kv_heads
     - 8 
     - Number of key-value heads 
  
In future, we would support more non-transformer type of layers. 

framework
-------------------
Specify the frameork (tensorflow or pytorch) as 

.. code-block:: yaml

  framework: tensorflow

No parameters under this group. 


workflow
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - generate_data
     - False
     - whether to generate dataset
   * - train
     - True
     - whether to perform training
   * - evaluation
     - False
     - whether to perform evaluation
   * - checkpoint
     - False
     - whether to perform checkpointing
   * - profiling
     - False
     - whether to perform profiling

.. note:: 

 ``evaluation``, ``checkpoint``, and ``profiling`` have depency on ``train``. If ``train`` is set to be ```False```, ``evaluation``, ``checkpoint``, ``profiling`` will be reset to ```False``` automatically. 

  Even though ``generate_data`` and ``train`` can be performed together in one job, we suggest to perform them seperately to eliminate potential caching effect. One can generate the data first by running DLIO with ```generate_data=True``` and ```train=False```, and then run training benchmark with ```generate_data=False``` and ```train=True```. 

dataset
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - record_length
     - 65536
     - size of each sample
   * - record_length_stdev
     - 0.
     - standard deviation of the sample size
   * - record_length_resize
     - 0. 
     - resized sample size 
   * - format
     - tfrecord
     - data format [tfrecord|csv|npz|jpeg|png|hdf5]
   * - num_files_train
     - 1
     - number of files for the training set
   * - num_files_eval
     - 0
     - number of files for evaluation/validation set
   * - num_samples_per_file
     - 1
     - number of samples per file
   * - data_folder
     - ./data
     - the path to store the dataset. 
   * - num_subfolders_train
     - 0
     - number of subfolders that the training set is stored
   * - num_subfolders_eval
     - 0
     - number of subfolders that the evaluation/validation set is stored
   * - file_prefix
     - img
     - the prefix of the dataset file(s)
   * - compression
     - none
     - what compressor to use to compress the dataset. (limited support)
   * - compression_level
     - 4
     - level of compression for gzip
   * - enable_chunking
     - False
     - whether to use chunking to store hdf5. 
   * - chunk_size
     - 0
     - the chunk size for hdf5. 
   * - keep_files
     - True
     - whether to keep the dataset files afer the simulation.
   * - record_dims
     - []
     - The dimensions of each record in the dataset. This will be prioritized over record_length and record_length_resize if provided
   * - record_element_type
     - uint8
     - The data type of each element in the record. Default is `uint8` (1 byte), supports all `NumPy data types <https://numpy.org/devdocs/user/basics.types.html>`_
   * - num_dset_per_record
     - 1
     - (HDF5 only) The number of datasets to generate per record. The value of this parameter need to be divisible by first element of record_dims
   * - chunk_dims
     - []
     - (HDF5 only) The dimensions of chunking mechanism in HDF5
   * - max_shape
     - []
     - (HDF5 only) The maximum shape of resizeable dataset. if not provided, the dataset will not be resizeable and HDF5 will internally set it to the value of `record_dims`


.. note:: 

  The training and validation datasets will be put in ```${data_folder}/train``` and ```${data_folder}/valid``` respectively. If ``num_subfolders_train`` and ``num_subfolders_eval`` are larger than one, the datasets will be split into multiple subfolders within ```${data_folder}/train``` and ```${data_folder}/valid``` in a round robin manner. 

.. note:: 

  If ``format`` is set to be ``synthetic``, samples will be generated in memory and fed through the data loader specified. 

.. attention::
  
  For `format: jpeg`, it is not recommended to generate data due to its lossy compression nature. Instead, provide the path to original dataset in the `data_folder` parameter. 

  More information on JPEG image generator analysis is provided at :ref:`jpeg_generator_issue` section. 
  Follow the original dataset directory structure as described in :ref:`directory structure <directory-structure-label>`
  
reader 
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - data_loader
     - tensorflow
     - select the data loader to use [tensorflow|pytorch|synthetic]. 
   * - batch_size
     - 1 
     - batch size for training
   * - batch_size_eval
     - 1 
     - batch size for evaluation
   * - read_threads* 
     - 1
     - number of threads to load the data (for tensorflow and pytorch data loader)
   * - pin_memory
     - True
     - whether to pin the memory for pytorch data loader
   * - computation_threads
     - 1
     - number of threads to preprocess the data
   * - prefetch_size
     - 0
     - number of batches to prefetch (0 - no prefetch at all)
   * - sample_shuffle
     - off
     - [seed|random|off] whether and how to shuffle the dataset samples
   * - file_shuffle
     - off
     - [seed|random|off] whether and how to shuffle the dataset file list
   * - transfer_size
     - 262144
     - transfer size in byte for tensorflow data loader. 
   * - preprocess_time
     - 0.0
     - | The amount of emulated preprocess time (sleep) in second. 
       | Can be specified as a distribution, see :ref:`Time Configuration` for more details.
   * - preprocess_time_stdev
     - 0.0
     - The standard deviation of the amount of emulated preprocess time (sleep) in second.
   * - odirect
     - False
     - enable O_DIRECT for the npy and npz formats only to bypass OS cache. 
   * - transformed_record_dims
     - []
     - The shape of the transformed sample. This will be prioritized over `record_length_resize` if provided.
   * - transformed_record_element_type
     - uint8
     - The data type of the transformed sample. Default is `uint8` (1 byte), supports all `NumPy data types <https://numpy.org/devdocs/user/basics.types.html>`_

.. note:: 

  TensorFlow and PyTorch behave differently for some parameters. For ``read_threads``, tensorflow does 
  not support ``read_threads=0``, but pytorch does, in which case, the main thread will be doing data loader and no overlap between I/O and compute. 

  For pytorch, if ``prefetch_size`` is set to be 0, it will be changed to 2. In other words, the default value for ``prefetch_size`` in pytorch is 2. 

  In order to be consistent, we set ``prefetch_size`` to be 2 all the time for both pytorch and tensorflow. 

.. note:: 
  For``synthetic`` data loader, dataset will be generated in memory directly rather than loading from the storage. 

.. note:: 

  We also support custom data reader and data loader. The detailed instruction on how to create custom data loader and data reader are provided here: :ref:`custom_data_loader` and :ref:`custom_data_reader`. 

.. note:: 

  For odirect, it is only available for npy and npz formats.  Not yet implimented for all other formats so an error will be raised.

train
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - epochs
     - 1
     - number of epochs to simulate
   * - computation_time
     - 0.0
     - | emulated computation time per step in second
       | Can be specified as a distribution, see :ref:`Time Configuration` for more details.
   * - computation_time_stdev
     - 0.0
     - standard deviation of the emulated computation time per step in second
   * - total_training_steps
     - -1
     - number of training steps to simulate, assuming running the benchmark less than one epoch. 
   * - seed_change_epoch
     - True
     - whether to change random seed after each epoch
   * - seed
     - 123
     - the random seed     

.. note:: 

  To get the simulated computation time, one has to run the actual workload and get out the timing information. 

  In actual distributed training, the communication overhead will increase the time per time step. In DLIO however, we do not simulate communication. Therefore, one can in principle include the communication time as part of `computation_time`. 


evaluation
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - eval_time
     - 0
     - | emulated computation time (sleep) for each evaluation step. 
       | Can be specified as a distribution, see :ref:`Time Configuration` for more details.
   * - eval_time_stdev
     - 0
     - standard deviation of the emulated computation time (sleep) for each evaluation step. 
   * - epochs_between_evals
     - 1
     - evaluate after x number of epochs
checkpoint
------------------
.. list-table::
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - checkpoint_folder
     - ./checkpoints/
     - the folder to save the checkpoints
   * - checkpoint_after_epoch
     - 1
     - start checkpointing after certain number of epochs specified
   * - epochs_between_checkpoints
     - 1
     - performing one checkpointing per certain number of epochs specified
   * - steps_between_checkpoints
     - -1
     - performing one checkpointing per certain number of steps specified
   * - fsync
     - False
     - whether to perform fsync after writing the checkpoint
   * - time_between_checkpoints
     - -1
     - | performing one checkpointing per {time_between_checkpoint} seconds;
       | this parameter is used only when workflow.train=False
   * - num_checkpoints_write
     - -1
     - | How many checkpoints to write;
       | this parameter is used only when workflow.train=False
   * - num_checkpoints_read
     - -1
     - | How many checkpoints to read;
       | this parameter is used only when workflow.train=False
   * - recovery_rank_shift
     - False
     - | Shift the rank ID by ppn (number of processes per node);
       | this can be used to avoid potential caching effect for checkpoint recovery.
   * - rank_sync
     - False
     - | Whether to synchronize all the ranks after checkpoint write / read or not.
       | If this is True, the synchronization time will be included in the overall checkpoint write / read time.
   * - mode
     - default
     - | The mode of the checkpointing.
       | Available options are: default, subset.
   * - randomize_tensor
     - True
     - | randomize the tensors data. If it is False, all the checkpoint data will be tensor of ones. 
   * - ksm
     - (omitted)
     - | Optional subsection to configure and enable Kernel Samepage Merging (KSM) optimization.
       | **Simply adding this ``ksm:`` section (even if empty, e.g., ``ksm: {}``) enables KSM features.**
       | See the KSM Configuration table below for optional nested keys to fine-tune KSM behavior. 
       | To use ksm, one has to set randomize_tensor = False. 

**KSM Configuration (Optional keys under `checkpoint.ksm`)**

.. list-table::
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter (within `ksm`)
     - Default
     - Description
   * - madv_mergeable_id
     - 12
     - ID for the madvise MADV_MERGEABLE system call.
   * - high_ram_trigger
     - 30.0
     - RAM usage percentage (%) threshold to start the KSM await logic (waiting for potential page merging).
   * - low_ram_exit
     - 15.0
     - RAM usage percentage (%) threshold to exit the KSM await logic early if memory usage drops below this level.
   * - await_time
     - 200
     - Maximum seconds to wait for KSM to potentially merge pages after marking them mergeable.

**Example YAML for KSM**

.. code-block:: yaml

   # Example 1: Enable KSM with default settings
   checkpoint:
     checkpoint_folder: checkpoints/my_model
     # ... other checkpoint settings ...
     ksm: {} # Presence enables KSM

   # Example 2: Enable KSM with custom settings
   checkpoint:
     checkpoint_folder: checkpoints/another_model
     # ... other checkpoint settings ...
     randomize_tensor: False
     ksm:
       high_ram_trigger: 25.0
       await_time: 150
       # Other KSM parameters will use defaults

**Example KSM System Configuration (Linux)**

The following bash script provides an example of configuring the Linux Kernel Samepage Merging (KSM) feature for potentially faster background merging (e.g., aiming for ~4GB/s). These settings adjust the KSM advisor and scanning parameters. Note that optimal settings can vary significantly depending on the system, workload, and kernel version. Use with caution and test thoroughly. Requires root privileges.

.. code-block:: bash

   #!/bin/bash
   # Example KSM configuration for potentially faster merging
   # Adjust values based on system testing and requirements
   echo 1 > /sys/kernel/mm/ksm/run
   echo scan-time > /sys/kernel/mm/ksm/advisor_mode
   echo 1 > /sys/kernel/mm/ksm/advisor_target_scan_time
   echo 900 > /sys/kernel/mm/ksm/advisor_max_cpu
   echo 9999999 > /sys/kernel/mm/ksm/advisor_min_pages_to_scan
   echo 99999999999999 > /sys/kernel/mm/ksm/advisor_max_pages_to_scan
   echo 999999999 > /sys/kernel/mm/ksm/max_page_sharing
   echo 2 > /sys/kernel/mm/ksm/run # Stop KSM temporarily
   sleep 1
   echo 1 > /sys/kernel/mm/ksm/run # Restart KSM with new settings
   echo 1 > /sys/kernel/mm/ksm/merge_across_nodes
   echo 1 > /sys/kernel/mm/ksm/run
   echo 1 > /sys/kernel/mm/ksm/use_zero_pages
   echo 1 > /sys/kernel/mm/ksm/smart_scan
   echo 1 > /sys/kernel/mm/ksm/sleep_millisecs # Example: 1 millisecond sleep


.. note::

   By default, if checkpoint is enabled, it will perform checkpointing from every epoch. One can perform multiple checkpoints within a single epoch,
   by setting ``steps_between_checkpoints``. If ``steps_between_checkpoints`` is set to be a positive number, ``epochs_between_checkpoints`` will be ignored.

   One can also perform checkpoint only benchmark, without doing training, i.e., without loading dataset. To do this, one can set ``workflow.train = False``, and then set ``num_checkpoints``, ``time_between_checkpoints``, and ``recovery_rank_shift``. These
   are effective only in checkpoint only mode.

   One can set ``checkpoint.mode`` to be ``subset`` to simulate checkpointing a set of GPUs which are a subset of a targed larger scale run. This is particularly useful
   if one would like to test the performance of a single NVMe drive, in the context of a larger scale run. In this case, only a subset of the entire checkpoint will be written.

output
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - folder
     - None
     - The output folder name.
   * - log_file
     - dlio.log
     - log file name  
   * - metric
     - {exclude_start_steps: 1, exclude_end_steps: 0}
     - To specify the steps to be excluded in the metric calculation. By default, we exclude the first step in 
   the beginning. 

.. note::
   
   If ``folder`` is not set (None), the output folder will be ```hydra_log/unet3d/$DATE-$TIME```. 

profiling
------------------
.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - iostat_devices**
     - [sda, sdb]
     - specifying the devices to perform iostat tracing.  

.. note::
   
   We support multi-level profiling using:
    * ``dftracer``: https://github.com/hariharan-devarajan/dftracer. DFTRACER_ENABLE=1 has to be set to enable profiler.
    Please refer to :ref:`profiling` on how to enable these profiling tools. 

Time Configuration
============================================

The time configuration is crucial for the emulation. Here, we are able to specify distribution of the time configuration.

For example, to specify distribution of the computation time, one can specify the configuration as ``dictionary`` with the following format:


* Normal Distribution

.. code-block:: yaml
   computation_time:
      mean: 1.0
      stdev: 0.1
      type: normal

   # or

   computation_time:
      mean: 1.0

   # or

   computation_time:
      mean: 1.0
      stdev: 0.1

* Uniform Distribution

.. code-block:: yaml
   computation_time:
      min: 0.5
      max: 1.5
      type: uniform

* Gamma Distribution

.. code-block:: yaml
   computation_time:
      shape: 1.0
      scale: 1.0
      type: gamma

* Exponential Distribution

.. code-block:: yaml
   computation_time:
      scale: 1.0
      type: exponential

* Poisson Distribution

.. code-block:: yaml
   computation_time:
      lam: 1.0
      type: poisson

How to create a DLIO configuration YAML file
=============================================
Creating a YAML file for a workload is very straight forward. Most of the options are essentially the same with the actual workload, such as ``framework``, ``reader``, and many options in ``train``, ``evaluation``, such as ``epochs``. The main work involved is to find out the dataset information and the computation time. For the former, one can to check the original dataset to find out the number of files for training, how many samples per file, and the sample size, data format, etc. For the latter, one has to run the actual workload to find out the comptuation time per training step. One might have to add timing stamp before and after the training step. 

The YAML files are stored in the `workload`_ folder.
It then can be loaded by ```dlio_benchmark``` through hydra (https://hydra.cc/). This will override the default settings. One can override the configurations through command line (https://hydra.cc/docs/advanced/override_grammar/basic/).

.. _workload: https://github.com/argonne-lcf/dlio_benchmark/tree/main/dlio_benchmark/configs/workload


Environment variables
============================================
There are a few environment variables that controls and logging and profiling information. 

.. list-table:: 
   :widths: 15 10 30
   :header-rows: 1
   
   * - Variable name
     - Default
     - Description
   * - DLIO_LOG_LEVEL
     - warning
     - Specifying the loging level [error|warning|info|debug]. If info is set, it will output the progress for each step. 
   * - DFTRACER_ENABLE
     - 0
     - Enabling the dftracer profiling or not [0|1]
   * - DFTRACER_INC_METADATA
     - 0
     - Whether to include the meta data in the trace output or not [0|1] 
