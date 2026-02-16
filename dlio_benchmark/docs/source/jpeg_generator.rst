.. _jpeg_generator_issue:

Analysis on JPEG data generator
===================================

JPEG images are generally compressed using lossy compression algorithms.  Lossy compression strips bits of data from the image and this process is irreversible and varies everytime. Due to this lossy nature of JPEG images, generating JPEG files using DLIO will produce JPEG files not according to the provided record_length (file size per sample) in the workload configuration file. We tried to circumvent this issue with below approaches but it resulted in either generating file sizes not according to the record_length or impacting the IO performance. Hence, it is adviced to use the original JPEG files (pass the input data directory path to the data_folder parameter) instead of generating your own.  This is applicable only for the JPEG formats.

In below example, the provided record_length is 150528 but the generated data file sizes is roughly 85334. 

.. code-block:: yaml
    
        dataset:
        num_files_train: 1024
        num_samples_per_file: 1
        record_length: 150528
        data_folder: data/resnet50
        format: jpeg

        ....
        datascience 85334 Aug 16 00:59 img_1266999_0f_1300000.jpeg
        datascience 85267 Aug 16 00:59 img_1267999_0f_1300000.jpeg
        datascience 85272 Aug 16 00:59 img_1268999_0f_1300000.jpeg
        datascience 85233 Aug 16 00:59 img_1269999_0f_1300000.jpeg
        datascience 85273 Aug 16 00:59 img_1270999_0f_1300000.jpeg
        datascience 85198 Aug 16 00:59 img_1271999_0f_1300000.jpeg
        datascience 85355 Aug 16 00:59 img_1272999_0f_1300000.jpeg
        datascience 85296 Aug 16 00:59 img_1273999_0f_1300000.jpeg
        datascience 85279 Aug 16 01:00 img_1274999_0f_1300000.jpeg
        datascience 85488 Aug 16 01:00 img_1275999_0f_1300000.jpeg
        datascience 85241 Aug 16 01:00 img_1276999_0f_1300000.jpeg
        datascience 85324 Aug 16 01:00 img_1277999_0f_1300000-jpeg
        datascience 85344 Aug 16 01:00 img_1278999_0f_1300000-jpeg
        datascience 85303 Aug 16 01:00 img_1279999_0f_1300000-jpeg
        ....

- In order to circumvent this problem, we tried different `pillow.image.save` attributes in dlio_benchmark/data_generator/jpeg_generator.py. In a protype using 10,000 sample JPEG files, we read each JPEG file saved them as lossless PNG types. Even though the generated PNG file sizes were very close to the original JPEG file size, the time to just open  `PIL.Image.open(filepath)` JPEG file vs PNG file is different as shown below. This performance could be affected due to the different meta data associated with the file formats as well as the different number of I/O calls for JPEG and PNG files. 

.. code-block:: python

    for input in temp_input_filenames:
        jpeg_file_size_in = os.path.getsize(input)
        dim = int(math.sqrt(jpeg_file_size_in))
        in_records_jpeg_file_size = np.arange(dim * dim, dtype=np.uint8).reshape((dim, dim))
        with open(input, "rb") as f:
            image   = PIL.Image.open(f)
            img     = PIL.Image.fromarray(in_records_jpeg_file_size)
            img.save(output_file_png, format='PNG', bits=8, compress_level=0)


.. code-block:: bash
 
    Mean of jpeg_file_size_input_list     = 111259.80
    Mean of png_file_size_output_list     = 111354.83
    Mean of file size png:jpeg ratio      = 1.001907
    pstdev of jpeg_file_size_input_list   = 151862.96
    pstdev of png_file_size_output_list   = 151921.45
    pstdev of file size png:jpeg ratio    = 0.00465

    Total number of JPEG Files 10250
    Total number of PNG Files 10250


.. code-block:: python

    start = time.time()
    for input in temp_input_filenames:
        with open(input, "rb") as f:
            image = PIL.Image.open(f)      
    end = time.time()


.. code-block:: bash

    output from mac laptop:
    
    Run 1: Time to open png_samples 0.4237
    Run 2: Time to open png_samples 0.4237
    Run 3: Time to open png_samples 0.4209

    Run 1: Time to open jpeg_samples 0.5534
    Run 2: Time to open jpeg_samples 0.5579
    Run 3: Time to open jpeg_samples 0.5592


.. code-block:: bash

    Output from polaris using lustre grand file system:

    Run 1: Time to open png_samples 132.7067
    Run 2: Time to open png_samples 131.0787
    Run 3: Time to open png_samples 128.8040

    Run 1: Time to open jpeg_samples 172.5443
    Run 2: Time to open jpeg_samples 165.7361
    Run 3: Time to open jpeg_samples 165.8489


Using the different attributes of `PIL.Image.save()` with quality, subsampling, optimize, compress_level resulted in saving images of JPEG file sizes different from the provided record_length

.. code-block:: python

        img.save("test.jpg", format='JPEG', bits=8, quality=100, subsampling=0)
        img.save("test.jpg", format='JPEG', bits=8, quality=99,  subsampling=0)
        img.save("test.jpg", format='JPEG', bits=8, quality=100, subsampling=0)
        img.save("test.png", format='PNG',  bits=8, compress_level=0)
        img.save("test.png", format='JPEG', bits=8, quality="keep", subsampling="keep", optimize=False)


.. _directory-structure-label: 

The original dataset folder is expected to be in the below structure when using JPEG.

.. code-block:: bash

    data_dir
    ├── train
    │   ├── XXX.JPEG
    │   ├── XXX.JPEG
    ├── valid
    │   ├── XXX.JPEG
    │   ├── XXX.JPEG
    ├── test
    │   ├── XXX.JPEG
    │   ├── XXX.JPEG


If there are subfolders in the original dataset, it should be mentioned in the num_subfolders configuration parameter.

.. code-block:: bash

    dataset:
    data_folder: /lus/grand/projects/datasets/original-resnet/CLS-LOC
    format: jpeg
    num_subfolders_train: 1000
    num_subfolders_eval: 1000
    num_files_train: 1300
    num_samples_per_file: 1
    file_prefix: jpeg_gen_img_

    output:
    folder: ~/my_work_dir/dlio_resnet_1
    log_file: dlio_resnet_jpeg_
