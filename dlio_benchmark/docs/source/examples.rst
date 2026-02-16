Examples
=============

We here list a set of example workloads. In the first example, we show the benchmarking process, including generating the dataset, running the benchmark with profiling, and processing the logs and profiling data. For the rest of the workloads, we list the YAML configure files.

UNET3D: 3D Medical Image Segmentation
---------------------------------------
* Reference Implementation: https://github.com/mlcommons/training/tree/master/image_segmentation/pytorch
* Framework: PyTorch
* Dataset: .npz format image files containing a single sample.
* Trains over multiple epochs, performs evaluation on a held-out test set periodically.

.. code-block:: yaml

    # contents of unßet3d.yaml

    model: unet3d

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
        record_length: 146600628
        record_length_stdev: 68341808
        record_length_resize: 2097152
    
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
        model_size: 499153191

First, we generate the dataset with ```++workload.workflow.generate=False```

.. code-block:: bash
    
    mpirun -np 8 dlio_benchmark workload=unet3d ++workload.workflow.generate_data=True ++workload.workflow.train=False

Then, we run the appliation with iostat profiling

.. code-block:: bash
    
    dlio_benchmark workload=unet3d ++workload.workflow.profiling=iostat

To run in data parallel mode, one can do

.. code-block:: bash

    mpirun -np 8 dlio_benchmark workload=unet3d ++workload.workflow.profiling=iostat

This will run the benchmark and produce the following logging output: 

.. code-block:: text

    [INFO] 2023-06-27T21:27:12.956820 Running DLIO with 8 process(es) [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/main.py:108]
    [INFO] 2023-06-27T21:27:12.956967 Reading workload YAML config file 'dlio_benchmark.configs/workload/unet3d.yaml' [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/main.py:110]
    [INFO] 2023-06-27T21:27:13.010843 Starting data generation [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/main.py:165]
    [INFO] 2023-06-27T21:27:13.011399 Generating dataset in data/unet3d/train and data/unet3d/valid [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/data_generator/data_generator.py:73]
    [INFO] 2023-06-27T21:27:13.011457 Number of files for training dataset: 168 [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/data_generator/data_generator.py:74]
    [INFO] 2023-06-27T21:27:13.011500 Number of files for validation dataset: 0 [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/data_generator/data_generator.py:75]
    [INFO] 2023-06-27T21:27:14.149995 Generating NPZ Data: [>------------------------------------------------------------] 0.6% 1 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:15.919235 Generating NPZ Data: [===>---------------------------------------------------------] 5.4% 9 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:17.240473 Generating NPZ Data: [======>------------------------------------------------------] 10.1% 17 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:18.181652 Generating NPZ Data: [=========>---------------------------------------------------] 14.9% 25 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:19.070685 Generating NPZ Data: [============>------------------------------------------------] 19.6% 33 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:19.761225 Generating NPZ Data: [===============>---------------------------------------------] 24.4% 41 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:21.772731 Generating NPZ Data: [==================>------------------------------------------] 29.2% 49 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:22.621811 Generating NPZ Data: [====================>----------------------------------------] 33.9% 57 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:23.523462 Generating NPZ Data: [=======================>-------------------------------------] 38.7% 65 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:24.455943 Generating NPZ Data: [==========================>----------------------------------] 43.5% 73 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:25.243788 Generating NPZ Data: [=============================>-------------------------------] 48.2% 81 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:25.811104 Generating NPZ Data: [================================>----------------------------] 53.0% 89 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:26.787472 Generating NPZ Data: [===================================>-------------------------] 57.7% 97 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:28.969593 Generating NPZ Data: [======================================>----------------------] 62.5% 105 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:29.958574 Generating NPZ Data: [========================================>--------------------] 67.3% 113 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:31.206116 Generating NPZ Data: [===========================================>-----------------] 72.0% 121 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:32.909674 Generating NPZ Data: [==============================================>--------------] 76.8% 129 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:34.357919 Generating NPZ Data: [=================================================>-----------] 81.5% 137 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:35.710920 Generating NPZ Data: [====================================================>--------] 86.3% 145 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:38.266190 Generating NPZ Data: [=======================================================>-----] 91.1% 153 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:39.301475 Generating NPZ Data: [==========================================================>--] 95.8% 161 of 168  [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/utility.py:108]
    [INFO] 2023-06-27T21:27:39.846579 Generation done [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/main.py:170]
    [INFO] 2023-06-27T21:27:39.850430 Profiling Started with iostat [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/main.py:177]
    [INFO] 2023-06-27T21:27:39.888114 Max steps per epoch: 5 = 1 * 168 / 4 / 8 (samples per file * num files / batch size / comm size) [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/main.py:324]
    [INFO] 2023-06-27T21:27:39.888787 Starting epoch 1: 5 steps expected [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:128]
    [INFO] 2023-06-27T21:27:39.979028 Starting block 1 [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:198]
    [INFO] 2023-06-27T21:27:59.680070 Rank 0 step 1 processed 4 samples in 19.699954509735107 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:27:59.680076 Rank 1 step 1 processed 4 samples in 19.703863859176636 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:27:59.694070 Rank 3 step 1 processed 4 samples in 19.726907968521118 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:27:59.693802 Rank 4 step 1 processed 4 samples in 19.708129405975342 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:27:59.691022 Rank 2 step 1 processed 4 samples in 19.712920427322388 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:27:59.695373 Rank 6 step 1 processed 4 samples in 19.72462296485901 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:27:59.706875 Rank 5 step 1 processed 4 samples in 19.735779762268066 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:27:59.712785 Rank 7 step 1 processed 4 samples in 19.74686098098755 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:01.326995 Rank 0 step 2 processed 4 samples in 1.6458377838134766 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:01.327250 Rank 2 step 2 processed 4 samples in 1.6303155422210693 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:01.335634 Rank 1 step 2 processed 4 samples in 1.644171953201294 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:01.343710 Rank 4 step 2 processed 4 samples in 1.6453940868377686 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:01.355700 Rank 3 step 2 processed 4 samples in 1.6606194972991943 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:01.361624 Rank 5 step 2 processed 4 samples in 1.6541204452514648 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:01.364827 Rank 6 step 2 processed 4 samples in 1.6675446033477783 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:01.372457 Rank 7 step 2 processed 4 samples in 1.659090280532837 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:02.774831 Rank 0 step 3 processed 4 samples in 1.4467418193817139 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:02.775530 Rank 1 step 3 processed 4 samples in 1.4396388530731201 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:02.777924 Rank 6 step 3 processed 4 samples in 1.4070987701416016 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:02.778453 Rank 7 step 3 processed 4 samples in 1.4057674407958984 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:02.782499 Rank 2 step 3 processed 4 samples in 1.4540395736694336 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:02.783395 Rank 3 step 3 processed 4 samples in 1.4274392127990723 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:02.783894 Rank 4 step 3 processed 4 samples in 1.439401388168335 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:02.799731 Rank 5 step 3 processed 4 samples in 1.4285638332366943 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:04.229823 Rank 0 step 4 processed 4 samples in 1.454030990600586 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:04.229826 Rank 1 step 4 processed 4 samples in 1.453265905380249 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:04.240324 Rank 2 step 4 processed 4 samples in 1.4558677673339844 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:04.240330 Rank 3 step 4 processed 4 samples in 1.4567136764526367 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:04.245584 Rank 6 step 4 processed 4 samples in 1.4674956798553467 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:04.247221 Rank 4 step 4 processed 4 samples in 1.4627764225006104 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:04.250820 Rank 7 step 4 processed 4 samples in 1.4712388515472412 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:04.252102 Rank 5 step 4 processed 4 samples in 1.4519073963165283 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:13.523484 Rank 0 step 5 processed 4 samples in 9.293325901031494 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:13.527061 Maximum number of steps reached [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/main.py:297]
    [INFO] 2023-06-27T21:28:13.527543 Rank 6 step 5 processed 4 samples in 9.281713724136353 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:13.523490 Rank 1 step 5 processed 4 samples in 9.28818964958191 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:13.527551 Rank 7 step 5 processed 4 samples in 9.267073631286621 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:13.539249 Rank 4 step 5 processed 4 samples in 9.291641473770142 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:13.546242 Rank 2 step 5 processed 4 samples in 9.305717945098877 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:13.545463 Rank 5 step 5 processed 4 samples in 9.277906894683838 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:13.548088 Rank 3 step 5 processed 4 samples in 9.307523012161255 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:13.541554 Ending block 1 - 5 steps completed in 33.56 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:216]
    [INFO] 2023-06-27T21:28:13.712092 Epoch 1 - Block 1 [Training] Accelerator Utilization [AU] (%): 39.2945 [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:219]
    [INFO] 2023-06-27T21:28:13.713038 Epoch 1 - Block 1 [Training] Throughput (samples/second): 4.7693 [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:220]
    [INFO] 2023-06-27T21:28:20.379070 Ending epoch 1 - 5 steps completed in 40.49 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:159]
    [INFO] 2023-06-27T21:28:20.387992 Starting epoch 2: 5 steps expected [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:128]
    [INFO] 2023-06-27T21:28:20.458422 Starting block 1 [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:198]
    [INFO] 2023-06-27T21:28:38.420511 Rank 0 step 1 processed 4 samples in 17.950562000274658 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:38.423065 Rank 2 step 1 processed 4 samples in 17.90280842781067 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:38.423041 Rank 4 step 1 processed 4 samples in 17.953059911727905 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:38.425153 Rank 6 step 1 processed 4 samples in 17.904606580734253 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:38.427028 Rank 1 step 1 processed 4 samples in 17.957058906555176 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:38.430326 Rank 3 step 1 processed 4 samples in 17.909387826919556 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:38.444290 Rank 5 step 1 processed 4 samples in 17.92300271987915 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:38.450703 Rank 7 step 1 processed 4 samples in 17.980567455291748 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:39.852909 Rank 0 step 2 processed 4 samples in 1.4301834106445312 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:39.860430 Rank 4 step 2 processed 4 samples in 1.437042474746704 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:39.864937 Rank 1 step 2 processed 4 samples in 1.4373478889465332 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:39.865620 Rank 5 step 2 processed 4 samples in 1.4209046363830566 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:39.871567 Rank 2 step 2 processed 4 samples in 1.4482154846191406 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:39.879498 Rank 6 step 2 processed 4 samples in 1.4534542560577393 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:39.888964 Rank 7 step 2 processed 4 samples in 1.437666416168213 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:39.890346 Rank 3 step 2 processed 4 samples in 1.4595756530761719 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:41.311217 Rank 0 step 3 processed 4 samples in 1.4581162929534912 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:41.312092 Rank 2 step 3 processed 4 samples in 1.4399495124816895 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:41.313566 Rank 5 step 3 processed 4 samples in 1.4474966526031494 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:41.314422 Rank 6 step 3 processed 4 samples in 1.434694528579712 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:41.311211 Rank 4 step 3 processed 4 samples in 1.4503426551818848 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:41.318728 Rank 1 step 3 processed 4 samples in 1.4535951614379883 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:41.323162 Rank 7 step 3 processed 4 samples in 1.4327857494354248 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:41.339936 Rank 3 step 3 processed 4 samples in 1.4491026401519775 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:42.749878 Rank 0 step 4 processed 4 samples in 1.4382779598236084 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:42.749646 Rank 1 step 4 processed 4 samples in 1.4295282363891602 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:42.759622 Rank 4 step 4 processed 4 samples in 1.4434914588928223 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:42.759677 Rank 5 step 4 processed 4 samples in 1.445906162261963 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:42.760392 Rank 6 step 4 processed 4 samples in 1.4456770420074463 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:42.762643 Rank 2 step 4 processed 4 samples in 1.450068712234497 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:42.767003 Rank 7 step 4 processed 4 samples in 1.4435951709747314 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:42.766916 Rank 3 step 4 processed 4 samples in 1.4258863925933838 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:50.486273 Rank 0 step 5 processed 4 samples in 7.736128330230713 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:50.489983 Maximum number of steps reached [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/main.py:297]
    [INFO] 2023-06-27T21:28:50.496764 Rank 2 step 5 processed 4 samples in 7.733910799026489 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:50.507343 Rank 4 step 5 processed 4 samples in 7.74742317199707 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:50.507864 Rank 3 step 5 processed 4 samples in 7.7405922412872314 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:50.516752 Rank 1 step 5 processed 4 samples in 7.766550779342651 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:50.519272 Rank 5 step 5 processed 4 samples in 7.759366273880005 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:50.522207 Rank 6 step 5 processed 4 samples in 7.76110053062439 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]
    [INFO] 2023-06-27T21:28:50.522231 Rank 7 step 5 processed 4 samples in 7.754213571548462 s [/usr/local/lib/python3.10/dist-packages/dlio_benchmark/utils/statscounter.py:259]

    ... 

This will generate the logs and profiling data inside hydra_log/${model}/${data}-${time} folder.

.. code-block:: bash

    $ hydra_log/unet3d/2023-06-27-21-27-12
    0_output.json  2_output.json  4_output.json  6_output.json  dlio.log     per_epoch_stats.json
    1_output.json  3_output.json  5_output.json  7_output.json  iostat.json  summary.json

BERT: Natural Language Processing Model
---------------------------------------

* Reference Implementation: https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert
* Framework: Tensorflow
* Dataset: Multiple tfrecord files containing many samples each.
* Trains in a single epoch, performs periodic checkpointing of its parameters.

.. code-block:: yaml

    model: bert

    framework: tensorflow

    workflow:
        generate_data: False
        train: True
        checkpoint: True
    
    dataset: 
        data_folder: data/bert
        format: tfrecord
        num_files_train: 500
        num_samples_per_file: 313532
        record_length: 2500
        file_prefix: part

    train:
        computation_time: 0.968
        total_training_steps: 5000
    
    reader:
        data_loader: tensorflow
        read_threads: 1
        computation_threads: 1
        transfer_size: 262144
        batch_size: 48
        file_shuffle: seed
        sample_shuffle: seed

    checkpoint:
        checkpoint_folder: checkpoints/bert
        steps_between_checkpoints: 1250
        model_size: 4034713312

CosmoFlow: 3D CNN to Learn the Universe at Scale
----------------------------------------------------
* Reference Implementation: https://github.com/mlcommons/hpc/tree/main/cosmoflow
* Framework: Tensorflow Keras
* Dataset: Multiple tfrecord files containing many samples each.
* Trains in multiple epochs

.. code-block:: yaml

    # contents of cosmoflow.yaml
    model: cosmoflow

    framework: tensorflow

    workflow:
        generate_data: False
        train: True

    dataset:
        data_folder: ./data/cosmoflow
        num_files_train: 1024
        num_samples_per_file: 512
        record_length: 131072

    reader:
        data_loader: tensorflow
        computation_threads: 8
        read_threads: 8
        batch_size: 1
    
    train: 
        epochs: 4

ResNet50: 3D Image classification
-------------------------------------
* Reference Implementation: https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks
* Framework: Tensorflow
* Dataset: ImageNet datasets saved in tfrecords files
* Trains in multiple epochs. 

.. code-block:: yaml

    # contents of resnet50.yaml
    model: resnet50

    framework: tensorflow

    workflow:
        generate_data: False
        train: True

    dataset:
        num_files_train: 1024
        num_samples_per_file: 1024
        record_length: 150528
        data_folder: data/resnet50
        format: tfrecord
        
    data_loader:
        data_loader: tensorflow
        read_threads: 8
        computation_threads: 8

LLM (Large Language Model) checkpointing
-----------------------------------------
* Reference Implementation: git@github.com:argonne-lcf/Megatron-DeepSpeed.git
* Framework: PyTorch + DeepSpeed
* Dataset: Binary Index files

In this example, one can specify the model size, number of layers, parallelism (tensor, pipepline and zero_stage), and other parameters. 
The checkpoint data contains three different kinds of files: model, optimizer and training state. One can specify 
different ZeRO stages for the model and optimizer.
* For Stage 3, both the model and optimizer are sharded across all the data parallel instances. 
* For Stage 1 and 2 the optimizer is sharded across all the data parallel instances, but the model is sharded only across the first data parallel instance. 
* Pipeline parallelism and ZeRO 3 are not compatiable to each other. 
  
One can also specify the datatype for the model and optimizer to be saved. By default, the model is saved in fp16 and the optimizer in fp32.

The output log will contain the checkpoint duration and throughput. In the final summary.json, `checkpoint_duration` and `checkpoint_io` will be reported.

.. code-block:: yaml
    
    model: 
        name: llama_70b
        type: transformer
        model_size: 30102
        num_layers: 80
        parallelism: 
            tensor: 8
            pipeline: 4
            zero_stage: 1
        transformer: 
            vocab_size: 128000
            hidden_size: 8192
            ffn_hidden_size: 28672

    framework: pytorch

    workflow:
        generate_data: True
        train: True
        checkpoint: True

    dataset: 
        data_folder: data/llama_70b/
        format: mmap_indexed_binary
        num_files_train: 1
        num_samples_per_file: 1048576
        record_length: 2048
        
    reader: 
        data_loader: pytorch
        batch_size: 16
        read_threads: 1
        file_shuffle: seed
        sample_shuffle: seed

    train:
        epochs: 1
        computation_time: 5 # 2.44 sec per step
        total_training_steps: 5

    checkpoint:
        checkpoint_folder: checkpoints/llama_70b
        steps_between_checkpoints: 1
        model_datatype: fp16
        optimizer_datatype: fp32