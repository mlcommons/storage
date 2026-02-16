Installation
=============
The installation of DLIO follows the standard python package installation as follows: 

.. code-block:: bash

    git clone https://github.com/argonne-lcf/dlio_benchmark
    cd dlio_benchmark/
    pip install .

One can also build and install the package as follows 

.. code-block:: bash

    git clone https://github.com/argonne-lcf/dlio_benchmark
    cd dlio_benchmark/
    python setup.py build
    python setup.py install

One can also install the package directly from github

.. code-block:: bash

    pip install git+https://github.com/argonne-lcf/dlio_benchmark.git@main

    
One can build a docker image run DLIO inside a container.  

.. code-block:: bash

    git clone https://github.com/argonne-lcf/dlio_benchmark
    cd dlio_benchmark/
    docker build -t dlio .
    docker run -t dlio dlio_benchmark

A prebuilt docker image is available in docker hub (might not be up-to-date)

.. code-block:: bash 

    docker pull docker.io/zhenghh04/dlio:latest
    docker run -t docker.io/zhenghh04/dlio:latest dlio_benchmark

To run interactively in the docker container. 

.. code-block:: bash

    docker run -t docker.io/zhenghh04/dlio:latest bash
    root@30358dd47935:/workspace/dlio# dlio_benchmark
