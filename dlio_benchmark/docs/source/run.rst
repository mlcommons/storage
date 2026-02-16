.. _run: 

Running DLIO
======================
A DLIO run is split in 3 phases:

1. Generate synthetic data DLIO will use
2. Run the benchmark using the previously generated data
3. Post-process the results to generate a report

One can specify the workload through ```workload=WORKLOAD``` option in the command line. This will read in corresponding configuration file that provided in the `workload`_ folder. All the configuration will be installed in ``INSTALL_PREFIX_DIR/dlio_benchmark/configs/workload/`` The configuration can be overridden through command line following the hyra syntax (e.g.++workload.framework=tensorflow). 

.. note::

   **Custom configuration file**: If one would like to use custom configuration file, one can save the file in ```CUSTOM_CONFIG_FOLDER/workload/custom_workload.yaml``, and then pass the command line ```--config-dir CUSTOM_CONFIG_FOLDER workload=custom_workload```. It will then load the configuration from custom_workload.yaml. 

   **Output folder**: By default the logs and results will be saved in the```hydra_log/unet3d/$DATE-$TIME``` folder. One can change the output folder to a different one by setting ```--hydra.run.dir=OUTPUT_FOLDER```



1 and 2 can be done either together or in separate. This is controlled by ```workflow.generate_data``` and ```workflow.train``` in the configure file. If ```workflow.generate_data```, ```workflow.train```are all set to be ``True``, it will generate the data and then run the benchark. However, we always suggest to run it seperately, to avoid caching effect, and to avoid I/O profiling in the data generation part. 

'''''''''''''''''''''''
Generate data
'''''''''''''''''''''''

.. code-block:: bash

    mpirun -np 8 dlio_benchmark workload=unet3d ++workload.workflow.generate_data=True ++workload.workflow.train=False

In this case, we override ```workflow.generate_data``` and ```workflow.train``` in the configuration to perform the data generation.  

''''''''''''''''''''''
Running benchmark
''''''''''''''''''''''

.. code-block:: bash 

    mpirun -np 8 dlio_benchmark workload=unet3d ++workload.workflow.generate_data=False ++workload.workflow.train=True ++workload.workflow.evaluation=True

In this case, we set ```workflow.generate_data=False```, so it will perform training and evaluation with the data generated previously. 

.. note::
    DLIO Benchmark will show a warning when you have core affinity set to less than number of workers spawned by each GPU process. 
    Core affinity is set using MPI execution wrappers such as `mpirun`, `jsrun`, `lrun`, or `srun`.

'''''''''''''''''
Post processing
'''''''''''''''''
After running the benchmark, the outputs will be stored in the ```hydra_log/unet3d/$DATE-$TIME``` folder created by hydra by default. The folder will contains: (1) logging output from the run; (2) profiling outputs; (3) YAML config files: `config.yaml`, `overrides.yaml`, and `hydra.yaml`. The workload configuration file is included in `config.yaml`. Any overrides in the command line are included in `overrides.yaml`. 

To post process the data, one only need to specify the output folder. All the other setups will be automatically read from `config.yaml` inside the folder. 

.. code-block:: bash 

    dlio_postprocessor --output_folder=hydra_log/unet3d/$DATE-$TIME

This will generate DLIO_$model_report.txt inside the output folder.

.. _workload: https://github.com/argonne-lcf/dlio_benchmark/blob/main/dlio_benchmark/configs/workload
.. _unet3d.yaml: https://github.com/argonne-lcf/dlio_benchmark/blob/main/dlio_benchmark/configs/workload/unet3d.yaml


'''''''''
Profiling
'''''''''

Application Profiling
'''''''''''''''''''''

DLIO_Benchmark has an application level profiler by default. The profiler outputs all application level python function calls in <OUTPUT_FOLDER>/trace*.pfw files.
These files are in chrome tracing's json line format. This can be visualized using `perfetto UI https://ui.perfetto.dev/`_


Full Stack Profiling
'''''''''''''''''''''

DLIO_Benchmark has a optional full stack profiler called `dftracer https://github.com/hariharan-devarajan/dftracer`_. 

Installing Profiler
*******************

Installing just dftracer

.. code-block:: bash

    pip install git+https://github.com/hariharan-devarajan/dftracer.git@dev


DFTracer is always installed along with dlio_benchmark

.. code-block:: bash

    cd <DLIO_BENCHMARK_SRC>
    pip install .


The profiler outputs all profiling output in <OUTPUT_FOLDER>/trace*.pfw files.
It contains application level profiling as well as low-level I/O calls from POSIX and STDIO layers.
The low-level I/O events are only way to understand I/O pattern from internal framework functions such as TFRecordDataset or DaliDataLoader.
These files are in chrome tracing's json line format. This can be visualized using `perfetto UI https://ui.perfetto.dev/`_