.. _instructions_lassen:

Instructions for running DLIO Benchmark on Lassen@LLNL
================================================

''''''''''''
Installation
''''''''''''
On the login node: 

* **Clone the github repository**:

.. code-block:: bash

	git clone https://github.com/argonne-lcf/dlio_benchmark
	cd dlio_benchmark/

* **Use conda**:

.. code-block:: bash

	# Setup the required channels:
	conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/

	# Create and activate environment
	conda env create --prefix ./dlio_env_ppc --file environment-ppc.yaml --force
	conda activate ./dlio_env_ppc

	#Install other dependencies and make sure it finishes successfully with no errors:
	python -m pip install .


.. note::

	If there is any problem with mpi4py, make sure that mpi is pointing to the right version of gcc.
	Do not install packages using the $conda install command but rather install all required versions of packages using pip only.
	To check versions of mpicc and gcc:

.. code-block:: bash

	gcc --version
	mpicc --version

To specify a new link for gcc:

.. code-block:: bash

	which mpicc
	export CC='which mpicc'
	export CXX=mpic++

''''''''''''''''''''''''''''''''''''''''''
Generate synthetic data that DLIO will use
''''''''''''''''''''''''''''''''''''''''''

**On Lassen generate data with the use of JSRUN scheduler**:


Arguments to use:

1. --bind packed:4 (to bind tasks with 4 GPUs)
2. --smpiargs="-gpu" (enables gpu support)
3. --nrs x (allocation of x node, it can be set to to 1, 2, 4 etc On Lassen we have 756 compute nodes)
4. --rs_per_host 1 (resources per node)
5. --tasks_per_rs y (y processes per resourse set/per node, it can be set to to 1, 2, 4 as on Lassen we have 4 GPUs per node)
6. --launch_distribution packed (specify how tasks are started on the available resource sets within the allocation. Packed assigns task to the first resource set until each CPU in the resource set is assigned to a task, and then starts assigning tasks to the second resource set, third resource set, fourth resource set (and so on))
7. --cpu_per_rs ALL_CPUS (each resource set contains the number of CPUs that are available on each compute node)
8. --gpu_per_rs ALL_GPUS (each resource set contains the number of GPUs that are available on each compute node)

For more information on these arguments, please turn to: https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=SSWRJV_10.1.0/jsm/jsrun.htm

.. note::

	Lassen machine has a custom wrapper over jsrun which is also called `jsrun` used by default on the system.

You can use the already existing workloads (.yaml files) located at `workload`_ or you can create your own custom workload (.yaml file) based on the following instructions: `config`_

.. note::

	Do not forget to set a "data_folder" in the dataset section and a "folder" in the output section with abs existent paths if you create a custom .yaml workload file.
	Before generating the data, make sure you are in the your conda env and in the folder where your dlio_benchmark was installed having allocated a compute node

* To allocate a compute node for 1 hr in the queue pdebug run:

.. code-block:: bash

	lalloc 1 -W 60 -q pdebug

**Example**: in order to generate data having 1 compute node and 4 processes per node and using the configurations of the `resnet50` workload you would run the following command:

.. code-block:: bash

	jsrun --bind packed:4 --smpiargs="-gpu" --nrs 1 --rs_per_host 1 --tasks_per_rs 4 --launch_distribution packed --cpu_per_rs ALL_CPUS --gpu_per_rs ALL_GPUS dlio_benchmark workload=resnet50 ++workload.workflow.generate_data=True ++workload.workflow.train=False

.. note::

	Instead of running the jsrun command directly from the compute node(s) (you have to allocate as many nodes as your jsrun command requests otherwise there aren't going to be enough nodes for your scheduler to use) you can also write a script and run the script from the node you have allocated. To find detailed instructions on how to write BSUB scripts and placing jobs on queues please turn to: https://hpc.llnl.gov/banks-jobs/running-jobs/lsf-quick-start-guide 

Your data will be generated in the following folder if you are using the existing workloads, where WORKLOAD could be `cosmoflow`, `resnet50` etc: ```/path/to/your/dlio_benchmark/data/WORKLOAD/train/``` or in the absolute path folder that you specified in your custom .yaml file.

If you run a custom workload file provide the path to that by adding the following argument in your jsrun command: ```--config-dir /path/to/your/custom/workload/```.

'''''''''''''''''''''
Running the Benchmark
'''''''''''''''''''''

* To avoid cached results you can allocate a different compute node and run the benchmark from there.

**Example**: in order to run the benchmark with 1 compute node and 4 processes per node and using the configurations of the `resnet50` workload you would run the following command:

.. code-block:: bash

	jsrun --bind packed:4 --smpiargs="-gpu" --nrs 1 --rs_per_host 1 --tasks_per_rs 4 --launch_distribution packed --cpu_per_rs ALL_CPUS --gpu_per_rs ALL_GPUS dlio_benchmark workload=resnet50 ++workload.workflow.generate_data=False ++workload.workflow.train=True

If you want to use a profiler: Same example with using DFTracer, isting the io devices you would like to trace:

.. code-block:: bash

    export DFTRACER_ENABLE=1
	jsrun --bind packed:4 --smpiargs="-gpu" --nrs 1 --rs_per_host 1 --tasks_per_rs 4 --launch_distribution packed --cpu_per_rs ALL_CPUS --gpu_per_rs ALL_GPUS dlio_benchmark workload=resnet50 ++workload.workflow.generate_data=False ++workload.workflow.profiling=True

All the outputs will be stored in ```hydra_log/WORKLOAD/$DATE-$TIME``` folder, where WORKLOAD could be `cosmoflow` etc or in our examples resnet50 if you are using the existing workloads. If you are using a custom workload this will be in the absolute path that you specified in your .yaml file.

