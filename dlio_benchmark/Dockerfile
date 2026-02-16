FROM ubuntu:22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y git sysstat mpich libc6 libhwloc-dev python3.10 python3-pip python3-venv cmake

RUN python3 -m pip install --upgrade pip
RUN python3 -m venv /workspace/venv
ENV PATH="/workspace/venv/bin:$PATH"
RUN pip install pybind11 

# Add contents of the current directory to /workspace/dlio in the container
ADD . /workspace/dlio
RUN pip install --no-cache-dir /workspace/dlio
RUN rm -rf /workspace/dlio /root/.cache/pip
