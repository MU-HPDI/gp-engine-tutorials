FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

RUN apt update -y && apt install -y python3 python3-pip

RUN pip3 install torch numpy

RUN mkdir -p /workspace
COPY /matmul.py /workspace/matmul.py

WORKDIR /workspace
CMD /bin/bash
