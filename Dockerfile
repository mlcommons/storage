ARG FROM_IMAGE_NAME=ubuntu:20.04
FROM ${FROM_IMAGE_NAME}

ENV DEBIAN_FRONTEND noninteractive

RUN apt update \
    && apt install -y mpich python3 python3-pip bc

WORKDIR /workspace/storage/

COPY dlio_benchmark/requirements.txt .
RUN pip3 install -r requirements.txt && rm -f requirements.txt

COPY . .
ENTRYPOINT ["/workspace/storage/benchmark.sh"]
