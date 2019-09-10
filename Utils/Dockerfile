# Download base image ubuntu 18.04
FROM ubuntu:18.04

USER root

RUN apt-get update

RUN apt-get install -y python python-dev python-pip  && pip install opencv-python-headless opencv-contrib-python-headless

COPY . /main

RUN pip install -r main/requirements.txt

RUN pip --version && \
    python -c "import cv2 ; print(cv2.__version__)"

WORKDIR main/Code 

