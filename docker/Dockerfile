FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

WORKDIR /pixelda
COPY requirements.txt /pixelda


RUN apt update && apt install python3-pip -y
RUN cat /pixelda/requirements.txt
RUN pip install -r requirements.txt
